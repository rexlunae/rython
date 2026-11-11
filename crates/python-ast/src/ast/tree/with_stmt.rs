use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, Statement, SymbolTableNode,
    SymbolTableScopes, extract_list, WithItem,
};

/// The `unittest.TestCase` context managers a `with` item can call
/// (`with self.assertRaises(E):`) — the method name parsed once, here.
/// Each one is either lowered by [`With::to_rust`] or refused there: the
/// generic `with` path would drop the call with a warning and run the body
/// unchecked, which is silently different (issue #523).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TestCaseContext {
    AssertRaises,
    AssertRaisesRegex,
    AssertWarns,
    AssertWarnsRegex,
    AssertLogs,
    AssertNoLogs,
    SubTest,
}

impl TestCaseContext {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "assertRaises" => Self::AssertRaises,
            "assertRaisesRegex" => Self::AssertRaisesRegex,
            "assertWarns" => Self::AssertWarns,
            "assertWarnsRegex" => Self::AssertWarnsRegex,
            "assertLogs" => Self::AssertLogs,
            "assertNoLogs" => Self::AssertNoLogs,
            "subTest" => Self::SubTest,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::AssertRaises => "assertRaises",
            Self::AssertRaisesRegex => "assertRaisesRegex",
            Self::AssertWarns => "assertWarns",
            Self::AssertWarnsRegex => "assertWarnsRegex",
            Self::AssertLogs => "assertLogs",
            Self::AssertNoLogs => "assertNoLogs",
            Self::SubTest => "subTest",
        }
    }
}

/// The TestCase context manager a `with` item's expression calls
/// (`<receiver>.assertRaises(...)`), with the call.
pub(crate) fn testcase_context(expr: &ExprType) -> Option<(TestCaseContext, &crate::Call)> {
    let ExprType::Call(c) = expr else {
        return None;
    };
    let ExprType::Attribute(a) = c.func.as_ref() else {
        return None;
    };
    if !matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self") {
        return None;
    }
    TestCaseContext::from_name(&a.attr).map(|kind| (kind, c))
}

fn is_testcase_receiver(
    expr: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let Some((_, call)) = testcase_context(expr) else {
        return false;
    };
    let ExprType::Attribute(attr) = call.func.as_ref() else {
        return false;
    };
    if !matches!(attr.value.as_ref(), ExprType::Name(name) if name.id == "self") {
        return false;
    }
    let Some(class_name) = ctx.enclosing_class_name() else {
        return false;
    };
    matches!(symbols.get(class_name), Some(SymbolTableNode::ClassDef(class)) if class.is_testcase(symbols, options))
}

fn assert_raises_exception_name(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    line: usize,
) -> Result<String, String> {
    let Some((_, call)) = testcase_context(expr) else {
        return Err(format!(
            "line {line}: `assertRaises` context is not a statically recognized TestCase receiver; \
             use a named exception class directly; rython refuses to guess which class is caught"
        ));
    };
    let Some(exception) = call.args.first() else {
        return Err(format!(
            "line {line}: `assertRaises` needs one statically known exception class; \
             pass the class directly; rython refuses to guess which class is caught"
        ));
    };
    match exception {
        ExprType::Name(_) => crate::ast::tree::try_stmt::caught_class_name(exception, symbols, options)
            .and_then(Result::ok)
            .ok_or_else(|| format!(
                "line {line}: `assertRaises` exception class is not statically known; \
                 name a supported exception class directly; rython refuses to guess which class is caught"
            ))
            .map(|name| {
                if symbols.get(&name).is_none() {
                    crate::ast::tree::raise_stmt::builtin_class_dunder_name(&name).to_string()
                } else {
                    name
                }
            }),
        ExprType::Attribute(attr) => {
            let ExprType::Name(module_name) = attr.value.as_ref() else {
                return Err(format!(
                    "line {line}: dynamic exception attribute `.{}` in `assertRaises` is not supported yet; \
                     use a statically imported exception class such as `ValueError`; \
                     rython refuses to guess which class is caught",
                    attr.attr
                ));
            };
            let mut imported_name = module_name.id.clone();
            let mut aliased_module = None;
            let mut imported_module = None;
            for _ in 0..8 {
                match symbols.get(&imported_name) {
                    Some(SymbolTableNode::Alias(target)) => {
                        aliased_module = Some(target.clone());
                        imported_name = target.clone();
                    }
                    Some(SymbolTableNode::Import(import)) => {
                        imported_module = import
                            .names
                            .iter()
                            .find(|alias| {
                                alias.asname.as_deref().unwrap_or(&alias.name) == imported_name
                            })
                            .map(|alias| alias.name.clone());
                        break;
                    }
                    _ => break,
                }
            }
            let module = imported_module
                .as_deref()
                .or(aliased_module.as_deref())
                .unwrap_or(&module_name.id);
            crate::ast::tree::raise_stmt::stdlib_exception_canonical(module, &attr.attr)
                .map(str::to_string)
                .or_else(|| {
                    (module == "builtins"
                        && crate::ast::tree::raise_stmt::is_builtin_exception_name(&attr.attr))
                        .then(|| attr.attr.clone())
                })
                .ok_or_else(|| format!(
                    "line {line}: exception attribute `{}.{}` in `assertRaises` is not statically known; \
                     use a supported imported exception class directly; \
                     rython refuses to guess which class is caught",
                    module_name.id, attr.attr
                ))
        }
        _ => Err(format!(
            "line {line}: dynamic exception expression in `assertRaises` is not supported yet; \
             name a supported exception class directly; rython refuses to guess which class is caught"
        )),
    }
}

/// The exception class EXPRESSION of a `self.assertRaises(Exc)` context
/// manager in the shape the closure lowering supports: one positional
/// argument, no keywords, the class a name or a dotted name. None for any
/// other shape — `With::to_rust` refuses those loudly.
fn assert_raises_class(expr: &ExprType) -> Option<&ExprType> {
    let (TestCaseContext::AssertRaises, c) = testcase_context(expr)? else {
        return None;
    };
    if c.args.len() != 1 || !c.keywords.is_empty() {
        return None;
    }
    matches!(&c.args[0], ExprType::Name(_) | ExprType::Attribute(_)).then_some(&c.args[0])
}

/// Whether a `with` statement's items are the one shape that lowers to the
/// assertRaises CLOSURE (`with self.assertRaises(Exc):` or `... as cm:`):
/// a single item in the supported call shape whose target, if any, is a
/// plain name. The body renders inside a generated closure, which the scope
/// and binding-flow analyses must account for (issue #457, #523).
pub(crate) fn is_assert_raises_with(items: &[WithItem]) -> bool {
    items.len() == 1
        && assert_raises_class(&items[0].context_expr).is_some()
        && items[0]
            .optional_vars
            .as_ref()
            .is_none_or(|v| matches!(v, ExprType::Name(_)))
}

pub(crate) fn is_testcase_assert_raises_with(
    items: &[WithItem],
    class_name: Option<&str>,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    class_name
        .and_then(|name| symbols.get(name))
        .is_some_and(|node| matches!(node, SymbolTableNode::ClassDef(class) if class.is_testcase(symbols, options)))
        && is_assert_raises_with(items)
}

/// Whether a `with` body contains a `return`, or a `break`/`continue`
/// aimed at a loop outside it, that would have to escape the closure the
/// assertRaises lowering wraps it in. A loop INSIDE the body owns its own
/// `break`.
fn body_escapes_a_closure(body: &[Statement]) -> bool {
    crate::body_contains_function_return(body) || crate::body_breaks_outward(body)
}

/// What the function lowering's assertRaises pass
/// ([`bind_assert_raises_targets`]) proved about a
/// `with self.assertRaises(E) as cm:` statement's target.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssertRaisesTarget {
    /// Not checked: the statement was not part of a function body the pass
    /// ran over (module level, a class body). The `as` form is refused.
    #[default]
    Unchecked,
    /// Checked, and no `cm.exception` read refers to this binding.
    Unread,
    /// Checked, and a later `cm.exception` read (rewritten to `cm`) refers
    /// to this binding.
    Read,
}

/// The function-body pass behind `with self.assertRaises(E) as cm:`
/// (issue #523).
///
/// CPython binds `cm` to an `_AssertRaisesContext` whose one observable
/// attribute is `cm.exception`, set when the block catches `E`. rython binds
/// the Rust local `cm` to the CAUGHT EXCEPTION itself and rewrites every
/// `cm.exception` read to `cm`, so the read is an ordinary exception-typed
/// local — `str()`, `repr()`, `isinstance`, `type(...).__name__`, printing,
/// `SystemExit.code` all lower exactly as they do for one. That is only
/// equivalent while the name is used as nothing BUT the context's
/// `.exception`, so the pass refuses, naming the line and the rewrite:
///
/// - any other use of the name (bare `cm`, `cm.expected`, `cm.msg`, a store
///   through it, `del`), and any other binding of it in the function (an
///   assignment, a parameter, a loop or other `with` target, an import, a
///   nested definition, an `except ... as`, a comprehension target);
/// - a `cm.exception` read inside the block's own body (CPython raises
///   AttributeError there: nothing has been caught yet), before the block,
///   or outside the statement list holding the block (the Rust binding is
///   scoped to that list, while Python may have skipped the block on that
///   path);
/// - a read from a nested function, class body or lambda.
///
/// Each checked statement records whether a read refers to it
/// ([`AssertRaisesTarget`]), so an unread target emits no binding.
pub(crate) fn bind_assert_raises_targets(
    body: &mut [Statement],
    params: &[String],
    testcase: bool,
) -> Result<(), String> {
    use crate::ast::tree::visit::{self, Bindings, Descend, Flow};
    if !testcase {
        return Ok(());
    }
    // The names this scope binds through the supported `as` form, with the
    // line of their first such binding.
    let mut targets: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        if let Some(name) = assert_raises_target_name(s) {
            targets.entry(name.to_string()).or_insert(s.lineno.unwrap_or(0));
        }
        Flow::Continue
    });
    if targets.is_empty() {
        return Ok(());
    }
    let refuse_rebinding = |name: &str, first: usize, other: &str| -> String {
        format!(
            "`{name}` is bound by `with self.assertRaises(...) as {name}:` (line {first}) and \
             {other}; rython models the context only through `{name}.exception` and needs the \
             name to be bound by assertRaises blocks alone; use a distinct name for the other \
             binding; rython refuses to silently read a different value"
        )
    };
    for param in params {
        if let Some(first) = targets.get(param) {
            return Err(refuse_rebinding(param, *first, "is also a parameter of the function"));
        }
    }
    let mut rebound: Option<String> = None;
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        let own = assert_raises_target_name(s);
        for name in visit::stmt_bound_names(s, Bindings::Scope) {
            if let Some(first) = targets.get(&name)
                && own != Some(name.as_str())
            {
                rebound = Some(refuse_rebinding(
                    &name,
                    *first,
                    &format!("bound again at line {}", s.lineno.unwrap_or(0)),
                ));
                return Flow::Stop;
            }
        }
        Flow::Continue
    });
    if let Some(err) = rebound {
        return Err(err);
    }
    let mut pass = TargetPass {
        targets: &targets,
        next_id: 0,
        read: std::collections::HashSet::new(),
    };
    pass.list(body, &std::collections::HashMap::new())
}

/// The name a statement binds through the supported
/// `with self.assertRaises(E) as name:` form, if it is one.
fn assert_raises_target_name(s: &Statement) -> Option<&str> {
    let crate::StatementType::With(w) = &s.statement else {
        return None;
    };
    if !is_assert_raises_with(&w.items) {
        return None;
    }
    match w.items[0].optional_vars.as_ref()? {
        ExprType::Name(n) => Some(n.id.as_str()),
        _ => None,
    }
}

/// The walk of [`bind_assert_raises_targets`]: statement lists in source
/// order, each with the bindings live in it (name → the binding
/// statement's id, numbered in walk order).
struct TargetPass<'t> {
    targets: &'t std::collections::HashMap<String, usize>,
    next_id: usize,
    read: std::collections::HashSet<usize>,
}

impl TargetPass<'_> {
    fn list(
        &mut self,
        stmts: &mut [Statement],
        outer: &std::collections::HashMap<String, usize>,
    ) -> Result<(), String> {
        use crate::ast::tree::visit;
        let mut live = outer.clone();
        // (statement index, binding id) of this list's assertRaises blocks:
        // their read flags are known once the whole list has been walked
        // (a read can only follow its binding in this list or nest in it).
        let mut binders: Vec<(usize, usize)> = Vec::new();
        for (index, s) in stmts.iter_mut().enumerate() {
            let line = s.lineno.unwrap_or(0);
            let binds = assert_raises_target_name(s).map(str::to_string);
            // A store or `del` THROUGH the name (`cm.exception = x`,
            // `del cm.exception`, `del cm`): the rewrite would turn it into
            // a rebinding. (A plain-name store is a binding, which the
            // caller already refused.)
            let mut written: Vec<&str> = Vec::new();
            match &s.statement {
                crate::StatementType::Delete(targets) => {
                    for target in targets {
                        if let ExprType::Name(n) = target {
                            written.push(&n.id);
                        }
                        written_through(target, &mut written);
                    }
                }
                _ => {
                    for target in visit::stmt_targets(s) {
                        written_through(target, &mut written);
                    }
                }
            }
            if let Some(name) = written.into_iter().find(|n| self.targets.contains_key(*n)) {
                return Err(format!(
                    "line {line}: `{name}` is bound by `with self.assertRaises(...) as {name}:`, \
                     and writing through it or deleting it is not supported; rython models the \
                     context only through reads of `{name}.exception`; rython refuses to \
                     silently drop the write"
                ));
            }
            // A nested scope's BODY is its own: a def that binds the name
            // itself is unrelated (its own pass checks it); any other
            // mention would read this scope's binding from a closure.
            match &s.statement {
                crate::StatementType::FunctionDef(f) | crate::StatementType::AsyncFunctionDef(f) => {
                    if let Some(name) = self.targets.keys().find(|n| {
                        !visit::def_owns_name(f, n)
                            && visit::any_expr_in(&f.body, visit::Descend::All, |e| {
                                matches!(e, ExprType::Name(x) if &x.id == *n)
                            })
                    }) {
                        return Err(nested_scope_read(name, line, "a nested function"));
                    }
                }
                crate::StatementType::ClassDef(c) => {
                    if let Some(name) = self.targets.keys().find(|n| {
                        visit::any_expr_in(&c.body, visit::Descend::All, |e| {
                            matches!(e, ExprType::Name(x) if &x.id == *n)
                        })
                    }) {
                        return Err(nested_scope_read(name, line, "a class body"));
                    }
                }
                _ => {}
            }
            // The statement's own expressions run with the bindings live
            // BEFORE it (a with item's context runs before its target is
            // bound); its own `as` target is the binding, not a use.
            match &mut s.statement {
                crate::StatementType::With(w) if binds.is_some() => {
                    self.expr(&mut w.items[0].context_expr, &live, line)?;
                }
                _ => {
                    for e in visit::stmt_all_exprs_mut(s) {
                        self.expr(e, &live, line)?;
                    }
                }
            }
            // Bodies: an assertRaises block's own body cannot read its new
            // binding (nothing is caught yet: CPython's AttributeError, and
            // the closure runs before the Rust binding exists); any other
            // body sees the live bindings, and what it binds stays inside
            // it, as the Rust block it renders to does.
            let mut inner = live.clone();
            if let Some(name) = &binds {
                inner.remove(name);
            }
            if !visit::opens_scope(s) {
                for sub in visit::stmt_bodies_mut(s) {
                    self.list(sub, &inner)?;
                }
            }
            if let Some(name) = binds {
                let id = self.next_id;
                self.next_id += 1;
                live.insert(name, id);
                binders.push((index, id));
            }
        }
        for (index, id) in binders {
            if let crate::StatementType::With(w) = &mut stmts[index].statement {
                w.assert_raises_target = if self.read.contains(&id) {
                    AssertRaisesTarget::Read
                } else {
                    AssertRaisesTarget::Unread
                };
            }
        }
        Ok(())
    }

    /// The target name an expression mentions anywhere, if any.
    fn mentioned(&self, e: &ExprType) -> Option<String> {
        let mut found = None;
        crate::ast::tree::visit::any_expr(e, |x| match x {
            ExprType::Name(n) if self.targets.contains_key(&n.id) => {
                found = Some(n.id.clone());
                true
            }
            _ => false,
        });
        found
    }

    /// Check and rewrite one expression: `cm.exception` becomes `cm` where
    /// a binding is live; every other mention of a target name is refused.
    fn expr(
        &mut self,
        e: &mut ExprType,
        live: &std::collections::HashMap<String, usize>,
        line: usize,
    ) -> Result<(), String> {
        use crate::ast::tree::visit;
        if let ExprType::Attribute(a) = e
            && let ExprType::Name(n) = a.value.as_ref()
            && self.targets.contains_key(&n.id)
        {
            let name = n.id.clone();
            if a.attr != "exception" {
                return Err(format!(
                    "line {line}: `{name}.{attr}` reads the `assertRaises` context's `{attr}`, \
                     which is not supported yet; rython models only `{name}.exception` (the \
                     caught exception); rython refuses to silently read a different value",
                    attr = a.attr
                ));
            }
            let Some(id) = live.get(&name) else {
                return Err(format!(
                    "line {line}: `{name}.exception` is read where the `with \
                     self.assertRaises(...) as {name}:` block has not run on every path (inside \
                     the block, before it, or outside the statement list that holds it); \
                     CPython raises AttributeError or NameError there, or reads a block the \
                     path skipped; read `{name}.exception` after the block, at the same \
                     nesting level; rython refuses to silently read a different value"
                ));
            };
            self.read.insert(*id);
            *e = ExprType::Name(crate::Name { id: name });
            return Ok(());
        }
        match e {
            ExprType::Name(n) if self.targets.contains_key(&n.id) => {
                return Err(format!(
                    "line {line}: `{name}` is bound by `with self.assertRaises(...) as {name}:` \
                     and used as a value; rython models the context only through \
                     `{name}.exception` (the caught exception) and has no context object to \
                     hand out; rython refuses to silently substitute a different value",
                    name = n.id
                ));
            }
            ExprType::Lambda(_) => {
                if let Some(name) = self.mentioned(e) {
                    return Err(nested_scope_read(&name, line, "a lambda"));
                }
                return Ok(());
            }
            _ => {}
        }
        let mut result = Ok(());
        visit::each_subexpr_mut(e, visit::Descend::OwnScope, &mut |sub| {
            match self.expr(sub, live, line) {
                Ok(()) => true,
                Err(err) => {
                    result = Err(err);
                    false
                }
            }
        });
        result
    }
}

/// The names a store target writes THROUGH: the base of each attribute or
/// subscript chain in it (`cm` in `cm.exception.x = 1` or `cm[0] = 1`).
/// A bare name is a binding, not a write through it.
fn written_through<'e>(target: &'e ExprType, out: &mut Vec<&'e str>) {
    fn base<'e>(e: &'e ExprType, out: &mut Vec<&'e str>) {
        match e {
            ExprType::Name(n) => out.push(&n.id),
            ExprType::Attribute(a) => base(&a.value, out),
            ExprType::Subscript(s) => base(&s.value, out),
            _ => {}
        }
    }
    match target {
        ExprType::Attribute(a) => base(&a.value, out),
        ExprType::Subscript(s) => base(&s.value, out),
        ExprType::Tuple(t) => t.elts.iter().for_each(|e| written_through(e, out)),
        ExprType::List(items) => items.iter().for_each(|e| written_through(e, out)),
        ExprType::Starred(s) => written_through(&s.value, out),
        _ => {}
    }
}

fn nested_scope_read(name: &str, line: usize, scope: &str) -> String {
    format!(
        "line {line}: `{name}` is bound by `with self.assertRaises(...) as {name}:` and read \
         from {scope}; that read is not supported yet; read `{name}.exception` into a local \
         after the block and use the local; rython refuses to silently capture a different \
         value"
    )
}

fn has_subtest_repr(ty: &crate::TypeInfo) -> bool {
    match ty {
        crate::TypeInfo::Int
        | crate::TypeInfo::BigInt
        | crate::TypeInfo::Float
        | crate::TypeInfo::Bool
        | crate::TypeInfo::StrRef
        | crate::TypeInfo::String
        | crate::TypeInfo::Complex
        | crate::TypeInfo::PyValue => true,
        crate::TypeInfo::Vec(inner)
        | crate::TypeInfo::PyTuple(inner)
        | crate::TypeInfo::Option(inner)
        | crate::TypeInfo::PyValueMember(inner)
        | crate::TypeInfo::Borrowed(inner) => has_subtest_repr(inner),
        crate::TypeInfo::Dict(key, value) => {
            has_subtest_repr(key) && has_subtest_repr(value)
        }
        crate::TypeInfo::Tuple(items) => {
            !items.is_empty() && items.len() <= 8 && items.iter().all(has_subtest_repr)
        }
        _ => false,
    }
}

/// Whether a with-item's context expression is a threading synchronization
/// object (Lock/RLock/Semaphore) — constructed inline
/// (`with threading.Lock():`), a name assigned from such a construction,
/// or a PARAMETER annotated as one (`def crit(lock: threading.Lock):` —
/// the exact pass-a-lock-to-a-worker pattern; local_types records the
/// annotation). These have REAL `__enter__`/`__exit__` semantics
/// (acquire/release), so the with-statement must lower to the RAII guard
/// instead of the plain binding.
fn is_threading_sync_call(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let is_sync_name =
        |name: &str| crate::ThreadingType::from_name(name).is_some_and(|t| t.is_sync_guard());
    match expr {
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Attribute(attr) => {
                is_sync_name(&attr.attr)
                    && matches!(attr.value.as_ref(), ExprType::Name(n)
                        if crate::StdModule::from_name(&n.id)
                            == Some(crate::StdModule::Threading))
                    && !crate::module_name_shadowed(crate::StdModule::Threading.name(), symbols)
            }
            ExprType::Name(n) => {
                is_sync_name(&n.id)
                    && matches!(
                        symbols.get(&n.id),
                        Some(SymbolTableNode::ImportFrom(i))
                            if crate::StdModule::from_name(&i.module)
                                == Some(crate::StdModule::Threading)
                    )
            }
            _ => false,
        },
        ExprType::Name(n) => {
            // An annotated parameter (or annotated local) recorded in
            // local_types as "threading.Lock"/"threading.RLock"/
            // "threading.Semaphore".
            let annotated_sync = options.local_types.get(&n.id).is_some_and(|py| {
                py.strip_prefix("threading.").is_some_and(is_sync_name)
            });
            annotated_sync
                || matches!(
                    symbols.get(&n.id),
                    Some(SymbolTableNode::Assign { value: ExprType::Call(c), .. })
                        if is_threading_sync_call(&ExprType::Call(c.clone()), symbols, options)
                )
        }
        _ => false,
    }
}

/// Regular with statement (with context as var: ...)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct With {
    /// The with items (context managers)
    pub items: Vec<WithItem>,
    /// The body of the with statement
    pub body: Vec<Statement>,
    /// For `with self.assertRaises(E) as cm:`, what the function lowering's
    /// pass proved about `cm` (issue #523); `Unchecked` everywhere else.
    #[serde(default)]
    pub assert_raises_target: AssertRaisesTarget,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for With {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract items (list of withitem objects)
        let items: Vec<WithItem> = extract_list(&ob, "items", "with items")?;

        // Extract body
        let body: Vec<Statement> = extract_list(&ob, "body", "with body")?;

        Ok(With {
            items,
            body,
            assert_raises_target: AssertRaisesTarget::Unchecked,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for With {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for With {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process items and body
        let symbols = self.items.into_iter().fold(symbols, |acc, item| {
            let acc = item.context_expr.find_symbols(acc);
            if let Some(vars) = item.optional_vars {
                vars.find_symbols(acc)
            } else {
                acc
            }
        });
        self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Preserve the typed TestCase context dispatch from main. An
        // assert-family context beside subTest must still be refused.
        let kinds: Vec<TestCaseContext> = self.items.iter()
            .filter(|item| is_testcase_receiver(&item.context_expr, &ctx, &symbols, &options))
            .filter_map(|item| testcase_context(&item.context_expr).map(|(kind, _)| kind))
            .collect();
        if let Some(kind) = kinds.iter().copied().find(|k| *k != TestCaseContext::SubTest) {
            return self.lower_assert_raises(kind, ctx, options, symbols);
        }
        // A subtest is a failure boundary. Escaping control flow cannot
        // cross the generated closure and is refused below.
        if kinds.contains(&TestCaseContext::SubTest)
        {
            if self.items.len() != 1 || self.items[0].optional_vars.is_some() {
                return Err(format!(
                    "`with self.subTest(...)` at line {} must be the only context manager \
                     and cannot bind an `as` target; rython refuses to silently skip \
                     subtest failure handling",
                    self.lineno.unwrap_or(0)
                )
                .into());
            }
            if body_escapes_a_closure(&self.body) {
                return Err(format!(
                    "`with self.subTest(...)` at line {} containing `return`, `break`, or \
                     `continue` is not supported yet; move that control flow outside the \
                     subTest body; rython refuses to silently change its target",
                    self.lineno.unwrap_or(0)
                )
                .into());
            }
            let ExprType::Call(subtest_call) = &self.items[0].context_expr else {
                unreachable!("is_subtest_cm only matches calls");
            };
            if !subtest_call.args.is_empty()
                || subtest_call
                    .keywords
                    .iter()
                    .any(|keyword| keyword.arg.is_none() || keyword.arg.as_deref() == Some("msg"))
            {
                return Err(format!(
                     "`self.subTest` at line {} supports named context parameters only; positional \
                      messages, `msg=`, and `**kwargs` are not supported yet; use \
                      `name=value`; \
                     rython refuses to silently change subtest reporting",
                    self.lineno.unwrap_or(0)
                )
                .into());
            }
            let mut context_values = Vec::new();
            for keyword in &subtest_call.keywords {
                let name = keyword
                    .arg
                    .as_ref()
                    .expect("unpacked kwargs rejected above");
                let ty = crate::infer_type(Some(&ctx), &keyword.value, &options, &symbols);
                if !has_subtest_repr(&ty) {
                    return Err(format!(
                        "`self.subTest({name}=...)` at line {} has a value without a \
                         supported Python repr; use a scalar or supported container; \
                         rython refuses to silently change subtest reporting",
                        keyword.lineno.or(self.lineno).unwrap_or(0)
                    )
                    .into());
                }
                let value = keyword.value.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                context_values.push(quote! {
                    (#name.to_string(), stdpython::PyRepr::py_repr(&(#value)))
                });
            }
            let context = quote!(vec![#(#context_values),*]);
            let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
                .body
                .iter()
                .map(|stmt| {
                    stmt.clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())
                })
                .collect();
            let body_tokens = body_tokens?;
            let tail = if crate::ast::tree::function_def::guarantees_return(&self.body) {
                quote!()
            } else {
                quote!(Ok(()))
            };
            return Ok(quote! {
                stdpython::unittest::run_subtest(#context, || -> Result<(), PyException> {
                    #(#body_tokens;)*
                    #tail
                })?;
            });
        }
        // Every other TestCase context manager is lowered here or refused:
        // the generic path below would drop the call (a warning) and run the
        // body unchecked — silently different (issue #523).
        if let Some(kind) = kinds.into_iter().find(|k| *k != TestCaseContext::SubTest) {
            return self.lower_assert_raises(kind, ctx, options, symbols);
        }
        self.lower_plain(ctx, options, symbols)
    }
}

impl With {
    /// `with self.assertRaises(Exc):` and `with self.assertRaises(Exc) as
    /// cm:` — the unittest context-manager forms (issues #334, #523). The
    /// body runs inside a closure whose `Result` is asserted to have raised
    /// `Exc`, as `_AssertRaisesContext.__exit__` does: a raised `Exc` (or a
    /// subclass) is caught, no raise is CPython's `AssertionError: Exc not
    /// raised`, a different exception propagates unchanged. With a target,
    /// the caught exception is bound to `cm` (CPython's `cm.exception`;
    /// see [`bind_assert_raises_targets`]). Every shape CPython accepts that
    /// this cannot reproduce is refused with the rewrite.
    fn lower_assert_raises(
        self,
        kind: TestCaseContext,
        ctx: CodeGenContext,
        options: PythonOptions,
        symbols: SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let method = kind.name();
        let unsupported = |what: &str, rewrite: &str| -> Box<dyn std::error::Error> {
            format!(
                "`with self.{method}(...)` {what} is not supported yet; {rewrite}; rython \
                 refuses to silently run the block unchecked"
            )
            .into()
        };
        match kind {
            TestCaseContext::AssertRaises => {}
            TestCaseContext::AssertRaisesRegex => {
                return Err(unsupported(
                    "(the regex check, and CPython's chained report when it fails)",
                    "use `with self.assertRaises(E) as cm:` and check `str(cm.exception)` \
                     after the block",
                ));
            }
            TestCaseContext::AssertWarns | TestCaseContext::AssertWarnsRegex => {
                return Err(unsupported(
                    "(capturing warnings)",
                    "check the warning's condition directly",
                ));
            }
            TestCaseContext::AssertLogs | TestCaseContext::AssertNoLogs => {
                return Err(unsupported(
                    "(capturing log records)",
                    "check the logged condition directly",
                ));
            }
            TestCaseContext::SubTest => unreachable!("subTest lowers in to_rust"),
        }
        if self.items.len() != 1 {
            return Err(unsupported(
                "beside another context manager in one `with` statement",
                "nest the other context manager in its own `with` statement",
            ));
        }
        let Some((_, call)) = testcase_context(&self.items[0].context_expr) else {
            unreachable!("the item is a TestCase context manager");
        };
        if !call.keywords.is_empty() {
            return Err(unsupported(
                "with a keyword argument (`msg=`: CPython appends it to the failure message)",
                "drop the keyword, or assert the condition with an explicit message after \
                 the block",
            ));
        }
        if call.args.len() != 1 {
            return Err(unsupported(
                "with other than one exception class",
                "pass the exception class alone (the callable form is \
                 `self.assertRaises(E, f, *args)`, outside a `with`)",
            ));
        }
        if matches!(call.args[0], ExprType::Tuple(_)) {
            return Err(unsupported(
                "with a tuple of exception classes",
                "use one class (a common base class), or one block per class",
            ));
        }
        // The class name as the except clause canonicalizes it — CPython's
        // `expected.__name__` for the failure message, and the name the
        // raise side records for the match.
        let exc = assert_raises_exception_name(
            &self.items[0].context_expr,
            &symbols,
            &options,
            self.lineno.unwrap_or(0),
        )?;
        if body_escapes_a_closure(&self.body) {
            return Err(unsupported(
                "around a `return`, `break` or `continue` that leaves the block (CPython's \
                 `__exit__` reports it as `not raised`)",
                "move the statement after the block",
            ));
        }
        let target = match &self.items[0].optional_vars {
            None => None,
            Some(ExprType::Name(n)) => match self.assert_raises_target {
                AssertRaisesTarget::Unchecked => {
                    return Err(unsupported(
                        "with an `as` target outside a function body",
                        "move the block into a test method",
                    ));
                }
                AssertRaisesTarget::Unread => None,
                AssertRaisesTarget::Read => Some(crate::safe_ident(&n.id)),
            },
            Some(_) => {
                return Err(unsupported(
                    "with an `as` target that is not a plain name",
                    "bind a plain name (`as cm`) and read `cm.exception` after the block",
                ));
            }
        };
        let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
            .body
            .iter()
            .map(|stmt| {
                stmt.clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
            })
            .collect();
        let body_tokens = body_tokens?;
        // A body that RAISES unconditionally (`raise E(...)` as its
        // last statement) already ends in `return Err(..)`, which is
        // the closure's exit: appending `Ok(())` after it is dead code
        // and makes rustc see the closure return `()`, not
        // `Result<(), PyException>` (issue #457 - the reported E0308
        // lands on the ENCLOSING method, because `guarantees_return`
        // unwraps a `with` into its body and so withholds the
        // method's own tail). Only a body that can fall through needs
        // the tail.
        let tail = if crate::ast::tree::function_def::guarantees_return(&self.body) {
            quote!()
        } else {
            quote!(Ok(()))
        };
        let closure = quote! {
            || -> Result<(), PyException> {
                #(#body_tokens;)*
                #tail
            }
        };
        Ok(match target {
            // `cm` is the caught exception: every `cm.exception` read was
            // rewritten to it.
            Some(cm) => quote! {
                let #cm = unittest::assert_raises_exception(#exc, #closure)?;
            },
            None => quote! {
                unittest::assert_raises(#exc, #closure)?;
            },
        })
    }

    /// The generic `with`: no context-manager protocol, one real lowering
    /// (threading locks).
    fn lower_plain(
        self,
        ctx: CodeGenContext,
        options: PythonOptions,
        symbols: SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Evaluate each context manager and bind its `as` target (or a
        // throwaway binding when there is none, so side effects still run).
        // The general __enter__/__exit__ protocol is not modeled yet
        // (Rust's Drop at end of block approximates __exit__ cleanup), with
        // one REAL implementation: threading Lock/RLock/Semaphore context
        // expressions lower to the runtime's RAII guard — acquire at entry,
        // release when the guard drops, exception-safe through unwinding
        // `?`, exactly Python's with-lock discipline.
        let mut item_tokens = Vec::new();
        for (index, item) in self.items.into_iter().enumerate() {
            let is_sync = is_threading_sync_call(&item.context_expr, &symbols, &options);
            // A text file's methods all take `&self` (the handle is a
            // shared reference, like Python's), so its binding is never
            // `mut`.
            let is_file = crate::ast::tree::type_ctx::is_file_typeinfo(&crate::infer_type(
                Some(&ctx),
                &item.context_expr,
                &options,
                &symbols,
            ));
            let context_expr =
                item.context_expr
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            match item.optional_vars {
                Some(vars) if is_sync => {
                    // CPython binds the target to __enter__'s return (True
                    // for locks) — a shape with no honest lowering here.
                    let target = vars.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    return Err(format!(
                        "`with <lock> as {}:` is not supported yet (a lock's __enter__ \
                         returns True, not the lock); use `with <lock>:` and name the \
                         lock outside the statement",
                        target
                    )
                    .into());
                }
                Some(vars) => {
                    // The item's target is bound right after its context
                    // expression, before the next item's runs: its marks go
                    // there (Devin review on #338, round 12).
                    let bind = crate::ast::tree::import::binds_for(
                        options.stmt_binds.as_deref(),
                        crate::ast::tree::visit::target_names(&vars).into_iter(),
                    );
                    let target = vars.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    if is_file {
                        item_tokens.push(quote! { let #target = #context_expr; #bind });
                    } else {
                        item_tokens.push(quote! { let mut #target = #context_expr; #bind });
                    }
                }
                None if is_sync => {
                    let guard = crate::safe_ident(&format!("__rython_with_guard_{}", index));
                    item_tokens.push(quote! { let #guard = (#context_expr).py_guard()?; });
                }
                None => {
                    item_tokens.push(quote! { let _ = #context_expr; });
                }
            }
        }

        let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self.body.into_iter()
            .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let body_tokens = body_tokens?;

        Ok(quote! {
            {
                #(#item_tokens)*
                #(#body_tokens;)*
            }
        })
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_with, "with context:\n    pass", "test.py");
}