//! The supported scalar builtin boundary for arbitrary-precision integers.
//! Names are classified once and shared by typing and rendering.

use crate::{Call, CodeGenContext, PythonOptions, SymbolTableScopes, TypeInfo};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub(crate) fn class_storage_line(
    body: &[crate::Statement],
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<usize> {
    use crate::ast::tree::visit::{self, Descend, Flow};
    let mut line = None;
    visit::walk_stmts(body, Descend::OwnScope, &mut |statement| {
        if visit::stmt_exprs(statement).into_iter().any(|expr| {
            visit::any_expr_for(expr, Descend::OwnScope, |expr| {
                crate::ast::tree::type_ctx::type_mentions_bigint(&crate::infer_type(
                    None, expr, options, symbols,
                ))
            })
        }) {
            line = Some(statement.lineno.unwrap_or(0));
            Flow::Stop
        } else if visit::opens_scope(statement) {
            // Methods and nested classes have their own storage/locals.
            Flow::Skip
        } else {
            Flow::Continue
        }
    });
    line
}

/// These runtime knobs are not implemented by the scalar slice. Classify
/// names once, including imported aliases, rather than dropping sys calls.
enum IntegerLimitApi {
    Get,
    Set,
}
impl IntegerLimitApi {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "get_int_max_str_digits" => Some(Self::Get),
            "set_int_max_str_digits" => Some(Self::Set),
            _ => None,
        }
    }
}

pub(crate) fn is_integer_limit_call(
    call: &Call,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    use crate::{ExprType, StdModule, SymbolTableNode};
    match call.func.as_ref() {
        ExprType::Name(name) if !options.name_types.contains_key(&name.id) => {
            match symbols.get(&name.id) {
                Some(SymbolTableNode::ImportFrom(import))
                    if StdModule::from_name(&import.module) == Some(StdModule::Sys) =>
                {
                    IntegerLimitApi::from_name(&import.defining_name(&name.id)).is_some()
                }
                _ => false,
            }
        }
        ExprType::Attribute(attr) if IntegerLimitApi::from_name(&attr.attr).is_some() => {
            let ExprType::Name(root) = attr.value.as_ref() else {
                return false;
            };
            if options.name_types.contains_key(&root.id)
                || options.local_types.contains_key(&root.id)
            {
                return false;
            }
            let mut bound = root.id.as_str();
            let mut seen = std::collections::HashSet::new();
            while let Some(SymbolTableNode::Alias(canonical)) = symbols.get(bound) {
                if !seen.insert(bound) {
                    return false;
                }
                bound = canonical;
            }
            match symbols.get(bound) {
                Some(SymbolTableNode::Import(import)) => import.names.iter().any(|alias| {
                    alias.asname.as_deref().unwrap_or(&alias.name) == root.id
                        && StdModule::from_name(&alias.name) == Some(StdModule::Sys)
                }),
                _ => false,
            }
        }
        _ => false,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum IntegerBuiltin {
    Print,
    Bool,
    Int,
    Str,
    Repr,
    Hash,
    Abs,
    Float,
}

/// Python evaluates every print argument first, then converts and writes
/// each one in order. Big integers make this observable without user
/// __str__: decimal conversion may raise after an earlier partial write.
pub(crate) fn render_print(
    call: &Call,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if options.no_std {
        return Err("print of an arbitrary-precision integer requires the std tier; return str(value) from an alloc-only function instead".into());
    }
    if !call.keywords.is_empty() {
        return Err("print keywords with an arbitrary-precision integer are not supported yet; use positional scalar arguments; rython refuses to silently ignore them".into());
    }
    let mut capture = TokenStream::new();
    let mut output = TokenStream::new();
    for (index, arg) in call.args.iter().enumerate() {
        let ty = crate::infer_type(Some(&ctx), arg, &options, &symbols);
        if !matches!(
            ty,
            TypeInfo::Int
                | TypeInfo::BigInt
                | TypeInfo::Float
                | TypeInfo::Bool
                | TypeInfo::String
                | TypeInfo::StrRef
        ) {
            return Err("print of an arbitrary-precision integer alongside a non-scalar value is not supported yet; print scalar values; rython refuses to change display evaluation order".into());
        }
        let value = crate::render_reused(arg, ctx.clone(), options.clone(), symbols.clone())?;
        let temporary = format_ident!("__rython_integer_print_{index}");
        capture.extend(quote!(let #temporary = #value;));
        if index != 0 {
            output.extend(quote!(print_parts(&[] as &[&str], "", " ")?;));
        }
        let display = if ty == TypeInfo::BigInt {
            quote!(#temporary.checked_decimal()?)
        } else {
            quote!(py_display(&#temporary))
        };
        output.extend(quote!(print_parts(&[#display], "", "")?;));
    }
    Ok(quote!({ #capture #output print_parts(&[] as &[&str], "", "\n")?; }))
}

impl IntegerBuiltin {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "print" => Self::Print,
            "bool" => Self::Bool,
            "int" => Self::Int,
            "str" => Self::Str,
            "repr" => Self::Repr,
            "hash" => Self::Hash,
            "abs" => Self::Abs,
            "float" => Self::Float,
            _ => return None,
        })
    }

    pub(crate) fn result_type(self) -> TypeInfo {
        match self {
            Self::Print => TypeInfo::PyObject,
            Self::Bool => TypeInfo::Bool,
            Self::Int | Self::Abs => TypeInfo::BigInt,
            Self::Str | Self::Repr => TypeInfo::String,
            Self::Hash => TypeInfo::Int,
            Self::Float => TypeInfo::Float,
        }
    }
}
