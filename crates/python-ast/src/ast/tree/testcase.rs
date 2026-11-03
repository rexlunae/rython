//! TestCase's field layout is static, but an attribute exists only after
//! Python assigns it. Prove that reads cannot see construction defaults.

use std::collections::HashSet;

use super::{
    binding_flow, rename,
    visit::{self, Descend, Flow},
};
use crate::{ClassDef, ExprType, FunctionDef, PythonOptions, StatementType, SymbolTableScopes};

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
        match name {
            "__init__" | "__new__" => Self::Constructor,
            "setUp" => Self::Setup,
            "tearDown" => Self::Teardown,
            "__getattribute__" | "__getattr__" | "__setattr__" | "__delattr__" => {
                Self::AttributeHook
            }
            n if n.starts_with("test") => Self::Test,
            _ => Self::Helper,
        }
    }
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
    if matches!(expr, ExprType::Lambda(_)) && visit::any_expr(expr, visit::is_self) {
        return true;
    }
    if let ExprType::Call(call) = expr
        && matches!(call.func.as_ref(), ExprType::Name(name) if name.id == "super")
    {
        return true; // An implicit receiver escape through a base method.
    }
    visit::subexprs(expr)
        .into_iter()
        .any(|child| opaque_receiver(child, lifecycle))
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
                // Every attribute leaf, an unpacking target's included
                // (`test_b.__unittest_skip__, n = True, 0` — #527).
                let attribute_targets = visit::stmt_targets(stmt)
                    .into_iter()
                    .flat_map(visit::target_leaves)
                    .filter_map(|t| match t {
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
        // A helper may run from setUp (directly or through other helpers)
        // before setUp's stores; any other helper runs only from a test or
        // tearDown, after setUp returned.
        let mut early_helpers: HashSet<String> = HashSet::new();
        let mut frontier: Vec<&FunctionDef> = methods
            .iter()
            .copied()
            .filter(|m| Member::from_name(&m.name) == Member::Setup)
            .collect();
        while let Some(method) = frontier.pop() {
            for name in receiver_members(&method.body) {
                if let Some(helper) = methods.iter().find(|m| {
                    m.name == name && Member::from_name(&m.name) == Member::Helper
                }) && early_helpers.insert(name)
                {
                    frontier.push(helper);
                }
            }
        }
        let empty = HashSet::new();
        let mut setup_fields = Some(empty.clone());
        // setUp runs first; test methods and tearDown may assume ONLY its
        // successful assignments. A different test's writes never carry
        // over. tearDown still runs when the test raises before a store.
        for member in [
            Member::Constructor,
            Member::AttributeHook,
            Member::Setup,
            Member::Test,
            Member::Teardown,
            Member::Helper,
        ] {
            for method in methods
                .iter()
                .filter(|m| Member::from_name(&m.name) == member)
            {
                if has_no_receiver(method) {
                    continue;
                }
                let line = chain
                    .iter()
                    .flat_map(|(c, ..)| &c.body)
                    .find_map(|stmt| {
                        matches!(&stmt.statement, StatementType::FunctionDef(m) if m == *method)
                            .then_some(stmt.lineno.unwrap_or(0))
                    })
                    .unwrap_or(0);
                if matches!(member, Member::Constructor | Member::AttributeHook) {
                    return Err(refuse(
                        &format!("method `{}` with instance fields", method.name),
                        line,
                        "use ordinary assignments in setUp and test methods",
                    ));
                }
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
                            .any(|b| visit::any_expr_in(b, Descend::All, visit::is_self));
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
                let bound = binding_flow::receiver_field_bindings(&body, initialized, &fields)
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
