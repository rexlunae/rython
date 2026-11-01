//! The static boundary for sys.exit and the builtin SystemExit constructor.

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, StdModule, SymbolTableNode,
    SymbolTableScopes, TypeInfo,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExitCall {
    Function,
    Constructor,
}

#[derive(Clone, Copy)]
enum Module {
    Sys,
    Builtins,
}

impl Module {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "builtins" => Some(Self::Builtins),
            _ if StdModule::from_name(name) == Some(StdModule::Sys) => Some(Self::Sys),
            _ => None,
        }
    }

    fn member(self, name: &str) -> Option<ExitCall> {
        match self {
            Self::Sys if name == "exit" => Some(ExitCall::Function),
            Self::Builtins => ExitCall::builtin(name),
            _ => None,
        }
    }
}

impl ExitCall {
    pub(crate) fn runtime_function(module: &str, member: &str) -> bool {
        Module::from_name(module).and_then(|module| module.member(member)) == Some(Self::Function)
    }

    pub(crate) fn builtin(name: &str) -> Option<Self> {
        match name {
            "SystemExit" => Some(Self::Constructor),
            _ => None,
        }
    }

    pub(crate) fn of(
        expr: &ExprType,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Option<Self> {
        fn resolve(
            expr: &ExprType,
            options: &PythonOptions,
            symbols: &SymbolTableScopes,
            seen: &mut HashSet<String>,
        ) -> Option<ExitCall> {
            match expr {
                ExprType::Name(name) => {
                    if options.name_types.contains_key(&name.id)
                        || options.local_types.contains_key(&name.id)
                        || !seen.insert(name.id.clone())
                    {
                        return None;
                    }
                    match symbols.get(&name.id) {
                        Some(SymbolTableNode::Alias(canonical)) => resolve(
                            &ExprType::Name(crate::Name {
                                id: canonical.clone(),
                            }),
                            options,
                            symbols,
                            seen,
                        ),
                        Some(SymbolTableNode::Assign { value, .. }) => {
                            resolve(value, options, symbols, seen)
                        }
                        Some(SymbolTableNode::ImportFrom(import)) if import.level == 0 => {
                            Module::from_name(&import.module)?
                                .member(&import.defining_name(&name.id))
                        }
                        None => ExitCall::builtin(&name.id),
                        _ => None,
                    }
                }
                ExprType::Attribute(attr) => {
                    module_of(&attr.value, options, symbols)?.member(&attr.attr)
                }
                _ => None,
            }
        }
        resolve(expr, options, symbols, &mut HashSet::new())
    }
}

fn module_of(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<Module> {
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
    let Some(SymbolTableNode::Import(import)) = symbols.get(current) else {
        return None;
    };
    import
        .names
        .iter()
        .find(|alias| alias.name == current || alias.asname.as_deref() == Some(current))
        .and_then(|alias| Module::from_name(&alias.name))
}

/// A bare runtime function value would bypass call-site arity and propagation.
pub(crate) fn reject_function_value(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    if ExitCall::of(expr, options, symbols) == Some(ExitCall::Function) {
        return Err("sys.exit used as a function value is not supported yet; call sys.exit(status) directly or wrap it in an annotated Python function; rython refuses to lose SystemExit propagation".into());
    }
    Ok(())
}

pub(crate) fn lower(
    call: &crate::Call,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<Option<TokenStream>, Box<dyn std::error::Error>> {
    let Some(kind) = ExitCall::of(&call.func, options, symbols) else {
        return Ok(None);
    };
    if !call.keywords.is_empty()
        || call
            .args
            .iter()
            .any(|arg| matches!(arg, ExprType::Starred(_)))
    {
        return Err("sys.exit/SystemExit keywords or starred arguments are not supported yet; pass explicit positional arguments; rython refuses to drop or misbind an exit status".into());
    }
    if kind == ExitCall::Function && call.args.len() > 1 {
        return Err("sys.exit takes at most one positional argument; pass one status (use a tuple to carry several values); rython refuses to ignore an extra argument".into());
    }
    let mut args = Vec::new();
    for arg in &call.args {
        args.push(status_value(arg, ctx, options, symbols)?);
    }
    Ok(Some(match kind {
        ExitCall::Constructor => quote!(stdpython::PyException::system_exit(vec![#(#args),*])),
        ExitCall::Function => {
            // Preserve the imported spelling so its import remains used.
            let function = match call.func.as_ref() {
                ExprType::Name(name) => {
                    let name = crate::safe_ident(&name.id);
                    quote!(#name)
                }
                ExprType::Attribute(attr) => {
                    let receiver = attr.value.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?;
                    let member = crate::safe_ident(&attr.attr);
                    quote!(#receiver::#member)
                }
                _ => unreachable!("classified exit callee"),
            };
            match args.first() {
                None => quote!(#function(stdpython::sys::ExitCode::Missing)?),
                Some(code) => quote!(#function(#code)?),
            }
        }
    }))
}

fn scalar_type(ty: &TypeInfo) -> bool {
    match ty {
        TypeInfo::Int
        | TypeInfo::Bool
        | TypeInfo::Float
        | TypeInfo::String
        | TypeInfo::StrRef
        | TypeInfo::Bytes => true,
        TypeInfo::Option(inner) | TypeInfo::Borrowed(inner) | TypeInfo::PyValueMember(inner) => {
            scalar_type(inner)
        }
        TypeInfo::Tuple(parts) => parts.iter().all(scalar_type),
        TypeInfo::Custom(tokens) => tokens.to_string() == "()",
        _ => false,
    }
}

fn status_value(
    arg: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if crate::is_none_expr(arg) {
        return Ok(quote!(stdpython::PyValue::None_));
    }
    if let ExprType::Call(call) = arg
        && let ExprType::Name(name) = call.func.as_ref()
        && let Some(SymbolTableNode::FunctionDef(function)) = symbols.get(&name.id)
        && function.returns.as_deref().is_some_and(crate::is_none_expr)
    {
        let call = arg
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        return Ok(quote!({ #call; stdpython::PyValue::None_ }));
    }
    if let ExprType::Tuple(tuple) = arg {
        let parts = tuple
            .elts
            .iter()
            .map(|part| status_value(part, ctx, options, symbols))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(quote!(stdpython::PyValue::Tuple(vec![#(#parts),*].into())));
    }
    let ty = crate::infer_type(Some(ctx), arg, options, symbols);
    if !scalar_type(&ty) {
        return Err("sys.exit/SystemExit status type is not supported yet; pass an int, bool, float, str, bytes, None, or a tuple of these (annotate helper returns); rython refuses to narrow integers, clone mutable status objects, or change when they are rendered".into());
    }
    crate::render_typed_reused(
        arg,
        ctx.clone(),
        options.clone(),
        symbols.clone(),
        Some(TypeInfo::PyValue),
    )
}

#[derive(Clone, Copy)]
enum ExitAttribute {
    Code,
}

impl ExitAttribute {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "code" => Some(Self::Code),
            _ => None,
        }
    }
}

pub(crate) fn code_attribute(
    attr: &crate::Attribute,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    ExitAttribute::from_name(&attr.attr).is_some()
        && crate::ast::tree::arguments::is_exception_typeinfo(&crate::infer_type(
            None,
            &attr.value,
            options,
            symbols,
        ))
}

/// Stored exception fields are immutable in the generated value model.
pub(crate) fn check_mutation(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    if crate::ast::tree::visit::any_expr(
        expr,
        |expr| matches!(expr, ExprType::Attribute(attr) if code_attribute(attr, options, symbols)),
    ) {
        return Err("SystemExit.code mutation or method calls are not supported yet; keep a typed local status and raise SystemExit(status) again; rython refuses to drop a code change or mutate a cloned value".into());
    }
    Ok(())
}

pub(crate) fn check_reflection(
    call: &crate::Call,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    if let (Some(receiver), Some(ExprType::Constant(name))) = (call.args.first(), call.args.get(1))
        && let Some(litrs::Literal::String(name)) = &name.0
        && ExitAttribute::from_name(name.value()).is_some()
        && crate::ast::tree::arguments::is_exception_typeinfo(&crate::infer_type(
            None, receiver, options, symbols,
        ))
    {
        return Err("reflection on SystemExit.code is not supported yet; read error.code directly or keep a typed local status; rython refuses to invent an attribute result or drop a code change".into());
    }
    Ok(())
}
