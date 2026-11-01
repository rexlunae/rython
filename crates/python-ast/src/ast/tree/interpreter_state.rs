//! Interpreter-owned namespaces have no equivalent in the static runtime.
//! Reject access instead of inventing an empty registry or dropping a store.

use crate::{
    Attribute, Call, ExprType, PythonOptions, StdModule, SymbolTableNode, SymbolTableScopes,
};
use std::collections::HashSet;

#[derive(Clone, Copy)]
pub(crate) enum InterpreterNamespace {
    Modules,
    Locals,
}

#[derive(Clone, Copy)]
enum NamespaceModule {
    Sys,
    Builtins,
}

impl NamespaceModule {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "builtins" => Some(Self::Builtins),
            _ if StdModule::from_name(name) == Some(StdModule::Sys) => Some(Self::Sys),
            _ => None,
        }
    }
}

impl InterpreterNamespace {
    fn from_sys_member(name: &str) -> Option<Self> {
        match name {
            "modules" => Some(Self::Modules),
            _ => None,
        }
    }

    pub(crate) fn from_import(module: &str, member: &str) -> Option<Self> {
        match NamespaceModule::from_name(module)? {
            NamespaceModule::Sys => Self::from_sys_member(member),
            NamespaceModule::Builtins => Self::from_builtin(member),
        }
    }

    fn from_builtin(name: &str) -> Option<Self> {
        match name {
            "locals" => Some(Self::Locals),
            _ => None,
        }
    }

    pub(crate) fn from_call(
        call: &Call,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Option<Self> {
        let ExprType::Name(name) = call.func.as_ref() else {
            return None;
        };
        if symbols.get(&name.id).is_some()
            || options.name_types.contains_key(&name.id)
            || options.local_types.contains_key(&name.id)
        {
            return None;
        }
        Self::from_builtin(&name.id)
    }

    pub(crate) fn from_expr(
        expr: &ExprType,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Option<Self> {
        match expr {
            ExprType::Attribute(attr) => Self::from_attribute(attr, options, symbols),
            ExprType::Call(call) => Self::from_call(call, options, symbols),
            _ => None,
        }
    }

    pub(crate) fn from_attribute(
        attr: &Attribute,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Option<Self> {
        Self::from_member(&attr.value, &attr.attr, options, symbols)
    }

    fn from_member(
        value: &ExprType,
        member: &str,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Option<Self> {
        match module_of_expr(value, options, symbols)? {
            NamespaceModule::Sys => Self::from_sys_member(member),
            NamespaceModule::Builtins => Self::from_builtin(member),
        }
    }

    /// Literal getattr/hasattr/setattr must not bypass attribute lowering.
    pub(crate) fn check_attribute_call(
        call: &Call,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let (Some(value), Some(ExprType::Constant(name))) = (call.args.first(), call.args.get(1))
            && let Some(litrs::Literal::String(name)) = &name.0
            && let Some(namespace) = Self::from_member(value, name.value(), options, symbols)
        {
            return Err(namespace.refusal().into());
        }
        Ok(())
    }

    pub(crate) fn refusal(self) -> &'static str {
        match self {
            Self::Modules => {
                "sys.modules access is not supported yet: the static runtime has no CPython import registry; replace registry lookups or mutation with explicit imports and an application-owned dictionary; rython refuses to silently invent an empty registry or drop a store"
            }
            Self::Locals => {
                "locals() namespace access is not supported yet: the static runtime has no Python frame dictionary; pass an explicit dictionary or use named variables instead; rython refuses to silently drop namespace access or mutation"
            }
        }
    }
}

fn module_of_expr(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<NamespaceModule> {
    let ExprType::Name(name) = expr else {
        return None;
    };
    if options.name_types.contains_key(&name.id) || options.local_types.contains_key(&name.id) {
        return None;
    }
    let mut current = name.id.as_str();
    let mut seen = HashSet::new();
    while let Some(SymbolTableNode::Alias(canonical)) = symbols.get(current) {
        if !seen.insert(current) {
            return None;
        }
        current = canonical;
    }
    match symbols.get(current) {
        Some(SymbolTableNode::Import(import)) => import
            .names
            .iter()
            .find(|alias| alias.name == current || alias.asname.as_deref() == Some(current))
            .and_then(|alias| NamespaceModule::from_name(&alias.name)),
        _ => None,
    }
}

/// Assignment paths do not necessarily use attribute load lowering.
pub(crate) fn check_target(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut found = None;
    crate::ast::tree::visit::any_expr(expr, |expr| {
        found = InterpreterNamespace::from_expr(expr, options, symbols);
        found.is_some()
    });
    match found {
        Some(namespace) => Err(namespace.refusal().into()),
        None => Ok(()),
    }
}
