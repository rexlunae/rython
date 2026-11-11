//! TestCase's field layout is static, but an attribute exists only after
//! Python assigns it. Prove that reads cannot see construction defaults.

use std::collections::{HashMap, HashSet};

use super::{
    binding_flow, rename,
    visit::{self, Descend, Flow},
};
use crate::{
    ClassDef, ExprType, FunctionDef, PythonOptions, Statement, StatementType, SymbolTableScopes,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Member {
    Constructor,
    Setup,
    Teardown,
    Test,
    AttributeHook,
    Helper,
}

impl Member {
    pub(crate) fn from_name(name: &str) -> Self {
        match Fixture::from_name(name) {
            Some(Fixture::SetUp) => return Self::Setup,
            Some(Fixture::TearDown) => return Self::Teardown,
            // The class fixtures take `cls`: no instance lifecycle role.
            Some(Fixture::SetUpClass | Fixture::TearDownClass) | None => {}
        }
        match name {
            "__init__" | "__new__" => Self::Constructor,
            "__getattribute__" | "__getattr__" | "__setattr__" | "__delattr__" => {
                Self::AttributeHook
            }
            n if n.starts_with("test") => Self::Test,
            _ => Self::Helper,
        }
    }
}

/// `unittest.TestCase`'s fixture hooks. CPython defines each one as a
/// no-op (`def setUp(self): pass`; `setUpClass` and `tearDownClass` are
/// classmethods), so a `super()` call that reaches TestCase's own hook
/// does nothing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fixture {
    SetUp,
    TearDown,
    SetUpClass,
    TearDownClass,
}

impl Fixture {
    pub(crate) const CLASS: [Self; 2] = [Self::SetUpClass, Self::TearDownClass];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::SetUp => "setUp",
            Self::TearDown => "tearDown",
            Self::SetUpClass => "setUpClass",
            Self::TearDownClass => "tearDownClass",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "setUp" => Some(Self::SetUp),
            "tearDown" => Some(Self::TearDown),
            "setUpClass" => Some(Self::SetUpClass),
            "tearDownClass" => Some(Self::TearDownClass),
            _ => None,
        }
    }
}

/// One entry of [`ClassDef::cross_module_chain`]: an ancestor, the symbols
/// and options of its defining module, and that module's path when it is
/// another module.
pub(crate) type ChainEntry = (ClassDef, SymbolTableScopes, PythonOptions, Option<Vec<String>>);

/// Where a zero-argument `super().<name>(...)` lands (issue #524).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SuperTarget {
    /// `chain[index]` defines `name` as an instance method.
    Defined(usize),
    /// `unittest.TestCase`'s own fixture hook, which does nothing.
    NoOp,
}

/// The method name of a zero-argument `super().<name>(...)` call.
pub(crate) fn super_method_name(call: &crate::Call) -> Option<&str> {
    let ExprType::Attribute(attr) = call.func.as_ref() else {
        return None;
    };
    let ExprType::Call(receiver) = attr.value.as_ref() else {
        return None;
    };
    (matches!(receiver.func.as_ref(), ExprType::Name(n) if n.id == "super")
        && receiver.args.is_empty()
        && receiver.keywords.is_empty())
    .then_some(attr.attr.as_str())
}

/// Whether the chain, followed by `unittest.TestCase`, IS the method
/// resolution order of `chain[0]`. Each class's first base must be the
/// next chain class (the base `cross_module_chain` followed), and every
/// other base must be TestCase itself; then C3 linearizes the family to
/// `chain + [TestCase, object]`. Any other shape (a second user base, an
/// unresolvable external base, TestCase listed before a mixin) puts a
/// class rython does not see between two chain entries.
fn mro_is_chain_then_testcase(chain: &[ChainEntry]) -> bool {
    let not_object = |b: &&ExprType| !matches!(b, ExprType::Name(n) if n.id == "object");
    let shape = chain.iter().enumerate().all(|(k, (class, ..))| {
        let mut bases = class.bases.iter().filter(not_object);
        if k + 1 < chain.len()
            && !matches!(bases.next(), Some(ExprType::Name(n))
                if !super::class_def::is_metadata_base_name(&n.id))
        {
            return false;
        }
        bases.all(super::module::is_testcase_base)
    });
    shape
        && chain
            .iter()
            .any(|(class, ..)| class.bases.iter().any(super::module::is_testcase_base))
}

/// Resolve `super().<name>(...)` (`call`) written in a method of
/// `chain[owner]`, where `chain` is the `cross_module_chain` of a TestCase
/// class: the nearest later chain class defining `name`, else TestCase's
/// own method. Every shape whose target rython cannot state, or whose
/// TestCase method it does not model, is an error naming the rewrite.
pub(crate) fn testcase_super_target(
    chain: &[ChainEntry],
    owner: usize,
    call: &crate::Call,
) -> Result<SuperTarget, String> {
    let name = super_method_name(call).unwrap_or_default();
    let line = call.lineno.unwrap_or(0);
    let owner_name = chain.get(owner).map_or("?", |(c, ..)| c.name.as_str());
    if !mro_is_chain_then_testcase(chain) {
        return Err(format!(
            "line {line}: super().{name}(...) in TestCase class `{owner_name}` is not supported yet: \
             its method resolution order passes a base rython does not follow (only a \
             single-inheritance chain ending in unittest.TestCase, or `Child(Mixin, \
             unittest.TestCase)`, is resolved); restructure the bases that way; rython \
             refuses to guess which base implementation runs"
        ));
    }
    for (index, (class, ..)) in chain.iter().enumerate().skip(owner + 1) {
        if let Some(method) = class.methods().find(|m| m.name == name) {
            if has_no_receiver(method) {
                return Err(format!(
                    "line {line}: super().{name}(...) in TestCase class `{owner_name}` reaches the \
                     classmethod or staticmethod `{}.{name}`, which is not supported yet; \
                     call it through the class (`{}.{name}(...)`); rython refuses to pass \
                     the instance where CPython passes the class",
                    class.name, class.name
                ));
            }
            return Ok(SuperTarget::Defined(index));
        }
    }
    match Fixture::from_name(name) {
        Some(Fixture::SetUp | Fixture::TearDown) => {}
        // TestCase's class fixtures are no-ops too, but the emitted runner
        // does not run a class's own setUpClass/tearDownClass yet (issue
        // #526), so a program calling them through super() stays loud
        // instead of building into that silent skip. Return NoOp here once
        // the runner calls class fixtures.
        Some(Fixture::SetUpClass | Fixture::TearDownClass) => {
            return Err(format!(
                "line {line}: super().{name}() in TestCase class `{owner_name}` is not supported yet: \
                 the emitted test runner does not run class fixtures \
                 (setUpClass/tearDownClass, issue #526); move the class fixture's work \
                 into setUp/tearDown; rython refuses to silently skip it"
            ));
        }
        None => {
            return Err(format!(
                "line {line}: super().{name}(...) in TestCase class `{owner_name}` reaches \
                 `unittest.TestCase.{name}`, which rython does not model as a base \
                 implementation yet; call `self.{name}(...)` from a class that does not \
                 override `{name}`; rython refuses to call the subclass's own `{name}` \
                 in its place"
            ));
        }
    }
    if !call.args.is_empty() || !call.keywords.is_empty() {
        return Err(format!(
            "line {line}: super().{name}(...) in TestCase class `{owner_name}` passes arguments to \
             `unittest.TestCase.{name}`, which takes none (CPython raises TypeError); \
             this is not supported; remove the arguments; rython refuses to silently \
             drop them"
        ));
    }
    Ok(SuperTarget::NoOp)
}

/// [`testcase_super_target`] for codegen: the chain of `enclosing` (the
/// class whose method body is being emitted) and the index of `owner`
/// (the class whose `super()` this is — `enclosing` itself, or the
/// ancestor whose body is re-emitted as a trait default). `None` when
/// `enclosing` is not a TestCase class.
pub(crate) fn resolve_testcase_super(
    enclosing: &str,
    owner: &str,
    call: &crate::Call,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    anchor: Option<(&str, &[String])>,
) -> Option<Result<(SuperTarget, Vec<ChainEntry>), String>> {
    let line = call.lineno.unwrap_or(0);
    let (class, class_symbols, class_options) = if let Some((anchor_name, path)) = anchor {
        let Some((class, class_symbols)) = crate::module_class_def(options, path, anchor_name)
        else {
            return Some(Err(format!(
                "line {line}: cannot resolve TestCase ancestry for `{anchor_name}` while \
                 lowering super().{}(...); this is not supported yet; keep the override in \
                 its defining module; rython refuses to guess its base implementation",
                super_method_name(call).unwrap_or_default()
            )));
        };
        let mut class_options = options.clone();
        let is_package = options.module_defs.keys().any(|k| {
            k.len() > path.len() && k[..path.len()] == path[..]
        });
        class_options.module_path = if is_package {
            path.to_vec()
        } else {
            path[..path.len().saturating_sub(1)].to_vec()
        };
        class_options.this_module_path = path.to_vec();
        (class, class_symbols, class_options)
    } else {
        let Some(crate::SymbolTableNode::ClassDef(class)) = symbols.get(enclosing) else {
            return None;
        };
        (class.clone(), symbols.clone(), options.clone())
    };
    if !class.is_testcase(&class_symbols, &class_options) {
        return None;
    }
    let chain = class.cross_module_chain(&class_symbols, &class_options);
    let Some(index) = chain.iter().position(|(c, ..)| c.name == owner) else {
        return Some(Err(format!(
            "line {line}: super() in `{owner}` is not on TestCase class `{}`'s base chain; \
             this is not supported yet; call the base method explicitly",
            class.name
        )));
    };
    Some(testcase_super_target(&chain, index, call).map(|target| (target, chain)))
}

/// Whether `expr`, in the method body `ctx` renders, is a `super().<hook>()`
/// call that reaches `unittest.TestCase`'s own no-op fixture: as a
/// statement it emits nothing, and `return super().setUp()` is a bare
/// return.
pub(crate) fn is_noop_testcase_super(
    expr: &ExprType,
    ctx: &crate::CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let ExprType::Call(call) = expr else {
        return false;
    };
    let Some(enclosing) = ctx.enclosing_class_name() else {
        return false;
    };
    if super_method_name(call).is_none() {
        return false;
    }
    let owner = ctx.super_target().unwrap_or(enclosing);
    matches!(
        resolve_testcase_super(
            enclosing,
            owner,
            call,
            symbols,
            options,
            ctx.testcase_anchor(),
        ),
        Some(Ok((SuperTarget::NoOp, _)))
    )
}

#[derive(Clone, Copy)]
enum RunnerAttribute {
    Skip,
    SkipReason,
    ExpectedFailure,
}

impl RunnerAttribute {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "__unittest_skip__" => Some(Self::Skip),
            "__unittest_skip_why__" => Some(Self::SkipReason),
            "__unittest_expecting_failure__" => Some(Self::ExpectedFailure),
            _ => None,
        }
    }
}

/// A bare receiver can escape to code that observes or changes attribute
/// presence. Direct attribute receivers are inspected by binding_flow;
/// the one visitor enumerates every other expression child here.
fn opaque_receiver(expr: &ExprType, lifecycle: &HashSet<&str>) -> bool {
    if let ExprType::Attribute(attr) = expr
        && visit::is_self(&attr.value)
    {
        // Fixtures/tests assume the runner's lifecycle. Calling or
        // capturing one from setUp could bypass that ordering proof.
        return lifecycle.contains(attr.attr.as_str());
    }
    if visit::is_self(expr) {
        return true;
    }
    if matches!(expr, ExprType::Lambda(_)) && visit::any_expr(expr, uses_receiver) {
        return true;
    }
    if let ExprType::Call(call) = expr
        && super_method_name(call).is_some()
    {
        // `super().m(...)` resolves statically (testcase_super_target) and
        // its body is checked where it runs; only an argument can still
        // hand the receiver on.
        return visit::subexprs(expr)
            .into_iter()
            .filter(|child| !std::ptr::eq(*child, call.func.as_ref()))
            .any(|child| opaque_receiver(child, lifecycle));
    }
    if calls_super(expr) {
        // A `super()` object kept as a value, or the two-argument form: an
        // implicit receiver escape through a base method.
        return true;
    }
    visit::subexprs(expr)
        .into_iter()
        .any(|child| opaque_receiver(child, lifecycle))
}

/// Whether `expr` is a call of `super`, in any form.
fn calls_super(expr: &ExprType) -> bool {
    matches!(expr, ExprType::Call(call)
        if matches!(call.func.as_ref(), ExprType::Name(name) if name.id == "super"))
}

/// Whether `expr` reaches the instance: `self`, or `super()`, which
/// carries it implicitly. A nested scope using either captures it.
fn uses_receiver(expr: &ExprType) -> bool {
    visit::is_self(expr) || calls_super(expr)
}

/// Every zero-argument `super().<name>(...)` call in `body`, nested scopes
/// included (so a target is checked even where the field proof refuses
/// the capture).
fn super_calls(body: &[Statement]) -> Vec<&crate::Call> {
    let mut out = Vec::new();
    visit::walk_stmts(body, Descend::All, &mut |stmt| {
        for expr in visit::stmt_all_exprs(stmt) {
            visit::walk_expr(expr, &mut |e| {
                if let ExprType::Call(call) = e
                    && super_method_name(call).is_some()
                {
                    out.push(call);
                }
            });
        }
        Flow::Continue
    });
    out
}

/// Whether a method has no instance receiver (`@staticmethod`,
/// `@classmethod`), so it cannot read or write instance attributes.
fn has_no_receiver(method: &FunctionDef) -> bool {
    matches!(
        super::decorator::parse_decorator(&method.decorator_list),
        Ok(Some(super::decorator::Decorator::StaticMethod | super::decorator::Decorator::ClassMethod))
    )
}

/// The methods `body` reaches through its receiver (`self.helper(...)`,
/// or `self.helper` passed on as a value).
fn receiver_members(body: &[crate::Statement]) -> HashSet<String> {
    let mut out = HashSet::new();
    visit::walk_stmts(body, Descend::All, &mut |stmt| {
        for expr in visit::stmt_all_exprs(stmt) {
            visit::walk_expr(expr, &mut |e| {
                if let ExprType::Attribute(a) = e
                    && visit::is_self(&a.value)
                {
                    out.insert(a.attr.clone());
                }
            });
        }
        Flow::Continue
    });
    out
}

impl ClassDef {
    pub(crate) fn is_testcase(&self, symbols: &SymbolTableScopes, options: &PythonOptions) -> bool {
        // Each ancestor resolves its own base in its defining module, so an
        // imported base's imported base is found (Devin review on #519).
        self.cross_module_chain(symbols, options)
            .iter()
            .any(|(c, ..)| c.bases.iter().any(super::module::is_testcase_base))
    }

    pub(crate) fn refuse_manual_testcase_construction(
        &self,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_testcase(symbols, options) {
            for (ancestor, ancestor_symbols, ancestor_options, _) in
                self.cross_module_chain(symbols, options)
            {
                if !ancestor.infer_fields(&ancestor_symbols, &ancestor_options)?.is_empty() {
                    return Err(format!(
                        "manual construction of TestCase `{}` with instance fields is not supported yet; \
                         use unittest.main() so setUp runs before each test; \
                         rython refuses to silently expose unassigned attribute defaults",
                        self.name,
                    ).into());
                }
            }
        }
        Ok(())
    }

    /// The receiver restrictions here protect the proof that no read of an
    /// instance attribute sees a struct default. A TestCase with no
    /// instance fields has no default to expose, so only the runner-flag
    /// and method-shadowing refusals (which are about what runs, not about
    /// fields) apply to it.
    pub(crate) fn validate_testcase_fields(
        &self,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_testcase(symbols, options) {
            return Ok(());
        }
        let chain = self.cross_module_chain(symbols, options);
        let mut fields = HashSet::new();
        let mut methods: Vec<&FunctionDef> = Vec::new();
        for (class, class_symbols, class_options, _) in &chain {
            fields.extend(
                class
                    .infer_fields(class_symbols, class_options)?
                    .into_iter()
                    .map(|(name, _)| name),
            );
            for method in class.methods() {
                if !methods.iter().any(|m| m.name == method.name) {
                    methods.push(method);
                }
            }
        }
        let lifecycle = methods
            .iter()
            .filter(|m| Member::from_name(&m.name) != Member::Helper)
            .map(|m| m.name.as_str())
            .collect();
        let refuse = |construct: &str, line: usize, rewrite: &str| -> Box<dyn std::error::Error> {
            format!(
                "TestCase `{}` {construct} at line {line} is not supported yet; \
                     {rewrite}; rython refuses to silently expose unassigned attribute defaults",
                self.name
            )
            .into()
        };
        // CPython's runner reads its skip/expected-failure flags off the
        // class and the test function, so binding one by hand (as a class
        // attribute or `test_b.__unittest_skip__ = True`) changes what runs.
        // The emitted runner honors only the decorators.
        for (class, ..) in &chain {
            let mut flag = None;
            visit::walk_stmts(&class.body, Descend::OwnScope, &mut |stmt| {
                let bound = visit::stmt_bound_names(stmt, visit::Bindings::Scope);
                let attribute_targets = visit::stmt_targets(stmt).into_iter().filter_map(|t| match t {
                    ExprType::Attribute(a) => Some(a.attr.clone()),
                    _ => None,
                });
                if let Some(name) = bound
                    .into_iter()
                    .chain(attribute_targets)
                    .find(|name| RunnerAttribute::from_name(name).is_some())
                {
                    flag = Some((name, stmt.lineno.unwrap_or(0)));
                    return Flow::Stop;
                }
                Flow::Continue
            });
            if let Some((name, line)) = flag {
                return Err(format!(
                    "TestCase `{}` explicit runner attribute `{name}` at line {line} is not supported yet; \
                     use unittest skip/skipIf/skipUnless/expectedFailure decorators without setting \
                     their runner attributes; rython refuses to silently run a test CPython would skip",
                    self.name
                ).into());
            }
        }
        // A class value hiding an inherited method needs Python's
        // instance-then-class lookup, which a plain struct cannot supply.
        // A class DATA attribute that an instance field shadows is fine:
        // the field proof below refuses any read that could come before
        // the instance store, so the struct default is never observed.
        for (class, ..) in &chain {
            let mut conflict = None;
            visit::walk_stmts(&class.body, Descend::OwnScope, &mut |stmt| {
                if matches!(stmt.statement, StatementType::AnnotatedName { .. }) {
                    return Flow::Continue; // A bare annotation binds no Python value.
                }
                let method_definition = matches!(
                    stmt.statement,
                    StatementType::FunctionDef(_) | StatementType::AsyncFunctionDef(_)
                );
                if let Some(name) = visit::stmt_bound_names(stmt, visit::Bindings::Scope)
                    .into_iter()
                    .find(|name| {
                        (method_definition && fields.contains(name))
                            || (!method_definition && methods.iter().any(|m| m.name == *name))
                    })
                {
                    conflict = Some((name, stmt.lineno.unwrap_or(0)));
                    return Flow::Stop;
                }
                Flow::Continue
            });
            if let Some((name, line)) = conflict {
                return Err(refuse(
                    &format!(
                        "class member `{name}` shadows a method or is shadowed by an instance field"
                    ),
                    line,
                    "use distinct method and attribute names",
                ));
            }
        }
        if fields.is_empty() {
            return Ok(());
        }
        let owner_of = |method: &FunctionDef| {
            chain
                .iter()
                .position(|(c, ..)| c.methods().any(|m| std::ptr::eq(m, method)))
                .unwrap_or(0)
        };
        let line_of = |method: &FunctionDef| {
            chain
                .iter()
                .flat_map(|(c, ..)| &c.body)
                .find_map(|stmt| {
                    matches!(&stmt.statement, StatementType::FunctionDef(m) if m == method)
                        .then_some(stmt.lineno.unwrap_or(0))
                })
                .unwrap_or(0)
        };
        let receiver_methods = |member: Member| {
            methods
                .iter()
                .copied()
                .filter(move |m| Member::from_name(&m.name) == member && !has_no_receiver(m))
        };
        for member in [Member::Constructor, Member::AttributeHook] {
            if let Some(method) = receiver_methods(member).next() {
                return Err(refuse(
                    &format!("method `{}` with instance fields", method.name),
                    line_of(method),
                    "use ordinary assignments in setUp and test methods",
                ));
            }
        }
        // Every receiver body the runner can reach, renamed to `self` and
        // checked once: each method on the MRO, and each base method a
        // `super().m(...)` call reaches (issue #524), keyed by the defining
        // class's chain index and the method name.
        let mut bodies: HashMap<(usize, &str), Vec<Statement>> = HashMap::new();
        let ordered = [Member::Setup, Member::Test, Member::Teardown, Member::Helper];
        for method in ordered.into_iter().flat_map(receiver_methods) {
            let mut pending = vec![(owner_of(method), method)];
            while let Some((owner, method)) = pending.pop() {
                if bodies.contains_key(&(owner, method.name.as_str())) {
                    continue;
                }
                let line = line_of(method);
                let receiver = method.args.posonlyargs.first().or(method.args.args.first());
                let body = rename::rename_receiver_in_body(
                    &method.body,
                    receiver.map_or("self", |p| p.arg.as_str()),
                    "self",
                )?;
                let mut opaque = None;
                visit::walk_stmts(&body, Descend::OwnScope, &mut |stmt| {
                    let captures = visit::opens_scope(stmt)
                        && visit::stmt_bodies(stmt)
                            .into_iter()
                            .any(|b| visit::any_expr_in(b, Descend::All, uses_receiver));
                    if captures
                        || visit::stmt_all_exprs(stmt)
                            .into_iter()
                            .any(|expr| opaque_receiver(expr, &lifecycle))
                        || matches!(&stmt.statement, StatementType::Delete(targets)
                            if targets.iter().any(|t| visit::any_expr(t, visit::is_self)))
                    {
                        opaque = Some(stmt.lineno.unwrap_or(line));
                        return Flow::Stop;
                    }
                    Flow::Continue
                });
                if let Some(line) = opaque {
                    return Err(refuse(
                        "receiver escape, capture, lifecycle access, or attribute deletion",
                        line,
                        "keep self attribute access in direct assignments and reads",
                    ));
                }
                for call in super_calls(&body) {
                    if let SuperTarget::Defined(index) = testcase_super_target(&chain, owner, call)? {
                        let name = super_method_name(call).unwrap_or_default();
                        if let Some(target) = chain[index].0.methods().find(|m| m.name == name) {
                            pending.push((index, target));
                        }
                    }
                }
                bodies.insert((owner, method.name.as_str()), body);
            }
        }
        // The body a resolved `super().m(...)` runs, for the field proof.
        let see_through: binding_flow::SeeThrough<'_> = &|call, owner| {
            let name = super_method_name(call)?;
            match testcase_super_target(&chain, owner, call).ok()? {
                SuperTarget::Defined(index) => {
                    let target = chain[index].0.methods().find(|m| m.name == name)?;
                    let body = bodies.get(&(index, target.name.as_str()))?;
                    Some((body.as_slice(), index))
                }
                SuperTarget::NoOp => None,
            }
        };
        // A helper may run from setUp (directly, through other helpers, or
        // through a base fixture `super()` reaches) before setUp's stores;
        // any other helper runs only from a test or tearDown, after setUp
        // returned.
        let mut early_helpers: HashSet<String> = HashSet::new();
        let mut frontier: Vec<(usize, &str)> = receiver_methods(Member::Setup)
            .map(|m| (owner_of(m), m.name.as_str()))
            .collect();
        while let Some(key) = frontier.pop() {
            let Some(body) = bodies.get(&key) else {
                continue;
            };
            for name in receiver_members(body) {
                if let Some(helper) = receiver_methods(Member::Helper).find(|m| m.name == name)
                    && early_helpers.insert(name)
                {
                    frontier.push((owner_of(helper), helper.name.as_str()));
                }
            }
            for call in super_calls(body) {
                if let Ok(SuperTarget::Defined(index)) = testcase_super_target(&chain, key.0, call) {
                    frontier.push((index, super_method_name(call).unwrap_or_default()));
                }
            }
        }
        let empty = HashSet::new();
        let mut setup_fields = Some(empty.clone());
        // setUp runs first; test methods and tearDown may assume ONLY its
        // successful assignments. A different test's writes never carry
        // over. tearDown still runs when the test raises before a store.
        for member in ordered {
            for method in receiver_methods(member) {
                let owner = owner_of(method);
                let Some(body) = bodies.get(&(owner, method.name.as_str())) else {
                    continue;
                };
                let after_setup = matches!(member, Member::Test | Member::Teardown)
                    || (member == Member::Helper && !early_helpers.contains(&method.name));
                let initialized = if after_setup {
                    let Some(initialized) = &setup_fields else {
                        // setUp has no successful exit: the runner invokes
                        // neither the test nor tearDown. Explicit TestCase
                        // construction and lifecycle calls are refused.
                        continue;
                    };
                    initialized
                } else {
                    &empty
                };
                let bound = binding_flow::receiver_field_bindings(
                    body,
                    initialized,
                    &fields,
                    owner,
                    see_through,
                )
                .map_err(|(field, line)| refuse(
                    &format!("attribute `self.{field}` may be read before assignment in `{}`", method.name),
                    line, "assign the attribute on every path before reading it (or initialize it unconditionally in setUp); a helper that setUp calls must initialize its own reads"))?;
                if member == Member::Setup {
                    setup_fields = bound;
                }
            }
        }
        Ok(())
    }
}
