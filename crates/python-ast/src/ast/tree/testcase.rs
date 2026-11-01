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

impl ClassDef {
    pub(crate) fn is_testcase(&self, symbols: &SymbolTableScopes, options: &PythonOptions) -> bool {
        self.base_chain_with_options(symbols, options)
            .iter()
            .any(|c| c.bases.iter().any(super::module::is_testcase_base))
    }

    pub(crate) fn validate_testcase_fields(
        &self,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_testcase(symbols, options) {
            return Ok(());
        }
        let chain = self.base_chain_with_options(symbols, options);
        let mut fields = HashSet::new();
        let mut methods: Vec<&FunctionDef> = Vec::new();
        for class in &chain {
            fields.extend(
                class
                    .infer_fields(symbols, options)?
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
        // A class attribute/method shadowed by a field, or a class value
        // hiding an inherited method, needs Python's instance-then-class
        // lookup. A plain struct cannot supply it.
        for class in &chain {
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
                        fields.contains(name)
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
                    "use distinct class and instance attribute names",
                ));
            }
        }
        if fields.is_empty() {
            return Ok(());
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
                let line = chain
                    .iter()
                    .flat_map(|c| &c.body)
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
                let initialized = if matches!(member, Member::Test | Member::Teardown) {
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
                        line, "assign the attribute on every path before reading it (or initialize it unconditionally in setUp); helpers must initialize their own reads"))?;
                if member == Member::Setup {
                    setup_fields = bound;
                }
            }
        }
        Ok(())
    }
}
