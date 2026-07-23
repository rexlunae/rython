use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    extract_required_attr, CodeGen, CodeGenContext, ExprType, Keyword, PythonOptions,
    SymbolTableNode, SymbolTableScopes,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Call {
    pub func: Box<ExprType>,
    pub args: Vec<ExprType>,
    pub keywords: Vec<Keyword>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Call {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let func: ExprType = extract_required_attr(&ob, "func", "function call expression")?;
        let args: Vec<ExprType> = extract_required_attr(&ob, "args", "function call arguments")?;
        let keywords: Vec<Keyword> = extract_required_attr(&ob, "keywords", "function call keywords")?;
        
        Ok(Call {
            func: Box::new(func),
            args,
            keywords,
        })
    }
}

impl<'a> CodeGen for Call {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Calls to functions that return Result<T, PyException> get `?` so
        // exceptions propagate to the caller (or an enclosing try block),
        // as in Python: user-defined functions (known from the symbol
        // table), names imported from user modules, and the Result-returning
        // stdpython builtins.
        let propagates_exceptions = match self.func.as_ref() {
            ExprType::Name(name) => {
                matches!(name.id.as_str(), "int" | "float")
                    || match symbols.get(&name.id) {
                        Some(SymbolTableNode::FunctionDef(_)) => true,
                        Some(SymbolTableNode::ImportFrom(import)) => {
                            let root = import.module.split('.').next().unwrap_or("");
                            !crate::is_stdpython_module(root)
                        }
                        _ => false,
                    }
            }
            _ => false,
        };

        // Constructing a class instance: `Point(args)` lowers to
        // `Point::new(args)?`, with arguments resolved against __init__'s
        // signature (minus self) so keywords and defaults follow Python
        // call semantics.
        if let ExprType::Name(n) = self.func.as_ref() {
            if let Some(SymbolTableNode::ClassDef(c)) = symbols.get(&n.id) {
                let cname = crate::safe_ident(&n.id);
                match c.init_method() {
                    Some(init) => {
                        if init.args.vararg.is_some() || init.args.kwarg.is_some() {
                            return Err(format!(
                                "`{}.__init__` takes *args/**kwargs, which is not \
                                 supported yet",
                                n.id
                            )
                            .into());
                        }
                        let mut sig = init.clone();
                        crate::strip_self(&mut sig.args);
                        let mapped = map_call_arguments(
                            &sig,
                            &self.args,
                            &self.keywords,
                            &ctx,
                            &options,
                            &symbols,
                        )?;
                        return Ok(quote!(#cname::new(#(#mapped),*)?));
                    }
                    None => {
                        if !self.args.is_empty() || !self.keywords.is_empty() {
                            return Err(format!(
                                "{}() takes no arguments: the class defines no __init__",
                                n.id
                            )
                            .into());
                        }
                        return Ok(quote!(#cname::new()?));
                    }
                }
            }
        }

        // Python methods whose Rust inherent namesakes have DIFFERENT
        // semantics (or the wrong shape) are rewritten here; methods with no
        // Rust conflict resolve through the stdpython PyListOps/PyStrOps
        // traits without any rewriting.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            // A method call on a receiver whose class is known — `self`
            // inside a method, or a name assigned a construction — resolves
            // against the class's own methods FIRST, so a user-defined
            // method named like a builtin (`get`, `pop`, ...) is not
            // rewritten out from under the class. Calls propagate
            // exceptions (`?`) and map keywords/defaults like any user
            // function call.
            if let Some(class) = receiver_class(&attr.value, &ctx, &symbols) {
                if let Some(method) = class.methods().find(|m| m.name == attr.attr).cloned() {
                    if method.args.vararg.is_some() || method.args.kwarg.is_some() {
                        return Err(format!(
                            "`{}.{}` takes *args/**kwargs, which is not supported yet",
                            class.name, method.name
                        )
                        .into());
                    }
                    let mut sig = method;
                    crate::strip_self(&mut sig.args);
                    let mapped = map_call_arguments(
                        &sig,
                        &self.args,
                        &self.keywords,
                        &ctx,
                        &options,
                        &symbols,
                    )?;
                    let receiver = attr.value.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?;
                    let method_name = crate::safe_ident(&attr.attr);
                    return Ok(quote!((#receiver).#method_name(#(#mapped),*)?));
                }
            }
            let receiver = attr
                .value
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let mut rendered_args = Vec::new();
            for arg in &self.args {
                rendered_args.push(arg.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?);
            }
            match (attr.attr.as_str(), rendered_args.as_slice()) {
                // list.append(x) pushes one element; Vec::append (inherent)
                // concatenates another Vec — silently different.
                ("append", [value]) => {
                    return Ok(quote!((#receiver).push(#value)));
                }
                // list.count(x): the PyListOps method takes a reference.
                ("count", [value]) => {
                    return Ok(quote!((#receiver).count(&(#value))));
                }
                // list.pop() returns the last element or raises IndexError
                // (Vec::pop returns an Option).
                ("pop", []) => {
                    return Ok(quote! {
                        (#receiver).pop().ok_or_else(|| {
                            PyException::new("IndexError", "pop from empty list")
                        })?
                    });
                }
                // pop with an argument dispatches by receiver through the
                // PyPop trait: list.pop(i) by index (IndexError), dict.pop(k)
                // by key (KeyError).
                ("pop", [arg]) => {
                    return Ok(quote!((#receiver).py_pop(#arg)?));
                }
                ("pop", [key, default]) => {
                    return Ok(quote!((#receiver).py_pop_default(#key, #default)));
                }
                // dict.get never raises: value-or-None (an Option), or the
                // provided default. IndexMap's inherent get returns a
                // borrowed Option, so both forms map to py_ versions.
                ("get", [key]) => {
                    return Ok(quote!((#receiver).py_get(&(#key))));
                }
                ("get", [key, default]) => {
                    return Ok(quote!((#receiver).py_get_default(&(#key), #default)));
                }
                // Views materialize as Vecs in insertion order.
                ("keys", []) => {
                    return Ok(quote!((#receiver).py_keys()));
                }
                ("values", []) => {
                    return Ok(quote!((#receiver).py_values()));
                }
                ("items", []) => {
                    return Ok(quote!((#receiver).py_items()));
                }
                ("setdefault", [key, default]) => {
                    return Ok(quote!((#receiver).py_setdefault(#key, #default)));
                }
                // list.remove(x) removes by VALUE and raises ValueError;
                // Vec::remove removes by index — silently different.
                ("remove", [value]) => {
                    return Ok(quote! {
                        {
                            let __rython_pos = (#receiver)
                                .iter()
                                .position(|__rython_e| __rython_e == &(#value))
                                .ok_or_else(|| {
                                    PyException::new(
                                        "ValueError",
                                        "list.remove(x): x not in list",
                                    )
                                })?;
                            (#receiver).remove(__rython_pos);
                        }
                    });
                }
                // list.insert follows Python index rules (negative counts
                // from the end, out-of-range clamps); Vec::insert takes a
                // usize and panics past len.
                ("insert", [idx, value]) => {
                    return Ok(quote!((#receiver).py_insert(#idx, #value)));
                }
                // str.split() with no argument splits on whitespace runs;
                // str::split (inherent) returns an iterator, so all forms
                // map to the PyStrOps versions returning Vec<String>. An
                // empty separator raises ValueError (hence `?`), and
                // maxsplit selects a distinct method (Rust lacks
                // overloading).
                ("split", []) => {
                    return Ok(quote!((#receiver).py_split_whitespace()));
                }
                ("split", [sep]) => {
                    return Ok(quote!((#receiver).py_split(&(#sep))?));
                }
                ("split", [sep, maxsplit]) => {
                    return Ok(quote!((#receiver).py_split_maxsplit(&(#sep), #maxsplit)?));
                }
                // str.rsplit: str::rsplit is an inherent iterator method.
                // With no separator (or full splits) it equals split.
                ("rsplit", []) => {
                    return Ok(quote!((#receiver).py_split_whitespace()));
                }
                ("rsplit", [sep]) => {
                    return Ok(quote!((#receiver).py_rsplit(&(#sep))?));
                }
                ("rsplit", [sep, maxsplit]) => {
                    return Ok(quote!((#receiver).py_rsplit_maxsplit(&(#sep), #maxsplit)?));
                }
                // partition/rpartition raise ValueError on an empty
                // separator, so the calls take `?`.
                ("partition", [sep]) => {
                    return Ok(quote!((#receiver).partition(&(#sep))?));
                }
                ("rpartition", [sep]) => {
                    return Ok(quote!((#receiver).rpartition(&(#sep))?));
                }
                // strip family with a chars argument (the no-arg forms
                // resolve through PyStrOps directly).
                ("strip", [chars]) => {
                    return Ok(quote!((#receiver).py_strip_chars(&(#chars))));
                }
                ("lstrip", [chars]) => {
                    return Ok(quote!((#receiver).py_lstrip_chars(&(#chars))));
                }
                ("rstrip", [chars]) => {
                    return Ok(quote!((#receiver).py_rstrip_chars(&(#chars))));
                }
                // ljust/rjust: the optional fillchar selects the py_ form
                // (space by default).
                ("ljust", [width]) => {
                    return Ok(quote!((#receiver).py_ljust(#width, " ")));
                }
                ("ljust", [width, fill]) => {
                    return Ok(quote!((#receiver).py_ljust(#width, &(#fill))));
                }
                ("rjust", [width]) => {
                    return Ok(quote!((#receiver).py_rjust(#width, " ")));
                }
                ("rjust", [width, fill]) => {
                    return Ok(quote!((#receiver).py_rjust(#width, &(#fill))));
                }
                // str.find returns -1 when absent; str::find an Option.
                ("find", [needle]) => {
                    return Ok(quote!((#receiver).py_find(&(#needle))));
                }
                _ => {}
            }
        }

        // Keyword arguments and omitted defaulted parameters resolve
        // against the callee's signature: keywords map to their parameter
        // positions and missing parameters fill from their default values,
        // matching Python call semantics. Without a known signature,
        // keywords would silently become misordered positional arguments —
        // that is a loud conversion error instead.
        let callee = match self.func.as_ref() {
            ExprType::Name(n) => match symbols.get(&n.id) {
                Some(SymbolTableNode::FunctionDef(f)) => Some(f.clone()),
                _ => None,
            },
            _ => None,
        };
        if let Some(callee_def) = &callee {
            let simple_signature =
                callee_def.args.vararg.is_none() && callee_def.args.kwarg.is_none();
            let pos_param_count =
                callee_def.args.posonlyargs.len() + callee_def.args.args.len();
            let has_optional_params = callee_def
                .args
                .posonlyargs
                .iter()
                .chain(callee_def.args.args.iter())
                .chain(callee_def.args.kwonlyargs.iter())
                .any(|p| {
                    p.annotation
                        .as_deref()
                        .is_some_and(crate::is_optional_annotation)
                });
            let needs_mapping = !self.keywords.is_empty()
                || !callee_def.args.kwonlyargs.is_empty()
                || self.args.len() < pos_param_count
                || has_optional_params;
            if simple_signature && needs_mapping {
                let mapped = map_call_arguments(
                    callee_def,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                )?;
                let name = self.func.to_rust(ctx, options, symbols)?;
                let call = quote!(#name(#(#mapped),*));
                return Ok(if propagates_exceptions {
                    quote!(#call?)
                } else {
                    call
                });
            }
            if !simple_signature && !self.keywords.is_empty() {
                return Err(format!(
                    "keyword arguments in a call to `{}` are not supported yet: \
                     its signature takes *args/**kwargs",
                    callee_def.name
                )
                .into());
            }
        } else if !self.keywords.is_empty() {
            return Err(format!(
                "keyword arguments require the callee's signature, and `{}` is not \
                 a function defined in this module; pass the arguments positionally",
                self.func
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
                    .map(|t| t.to_string())
                    .unwrap_or_else(|_| "<callee>".to_string())
            )
            .into());
        }

        let name = self.func.to_rust(ctx.clone(), options.clone(), symbols.clone())?;

        let mut all_args = Vec::new();

        // Add positional arguments
        for arg in self.args {
            let rust_arg = arg.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            all_args.push(rust_arg);
        }
        
        // Add keyword arguments
        for keyword in self.keywords {
            let rust_kw = keyword.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            all_args.push(rust_kw);
        }
        
        // Check if we're in an async context and if the function being called is async
        let call_expr = quote!(#name(#(#all_args),*));
        
        // Check if this function returns a Result that should be unwrapped
        let name_str = format!("{}", name);
        let needs_unwrap = matches!(name_str.as_str(), 
            "subprocess :: run" | "subprocess :: run_with_env" | "subprocess :: check_call" | 
            "subprocess :: check_output" | "os :: getcwd" | "os :: chdir" | "os :: execv" |
            "os :: path :: abspath"
        );
        
        // Special handling for subprocess.run and os.execv with fallback for compatibility
        let final_call = if propagates_exceptions {
            quote!(#call_expr?)
        } else if name_str == "subprocess :: run" {
            // Try mixed_args version first, fallback to regular version
            if all_args.len() >= 2 {
                let args_param = &all_args[0];
                let cwd_param = &all_args[1];
                // Convert args to Vec<String> to avoid lifetime issues, then pass owned strings
                quote!({
                    let args_owned: Vec<String> = #args_param;
                    let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                    let cwd_str = #cwd_param;
                    subprocess::run(args_vec, Some(&cwd_str)).unwrap()
                })
            } else {
                let args_param = &all_args[0];
                quote!({
                    let args_owned: Vec<String> = #args_param;
                    let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                    subprocess::run(args_vec, None).unwrap()
                })
            }
        } else if name_str == "os :: execv" {
            // Convert to Vec<&str> for compatibility with standard execv function
            let program_param = &all_args[0];
            let args_param = &all_args[1];
            quote!({
                let program_str: String = (#program_param).clone();
                let args_owned: Vec<String> = #args_param;
                let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                os::execv(&program_str, args_vec).unwrap()
            })
        } else if needs_unwrap {
            quote!(#call_expr.unwrap())
        } else {
            call_expr
        };
        
        // `.await` is added only by an explicit `await` expression (the Await
        // node), mirroring Python: calling an async function without await
        // does not implicitly run it. The old behavior appended `.await` to
        // any call whose name started with "a" in async contexts, which broke
        // calls like abs(x).
        Ok(final_call)
    }
}

/// The class of a method-call receiver, when it is statically known:
/// `self` inside a class's method body, or a local/module name whose
/// (symbol-table-recorded) assignment constructs a known class. Unknown
/// receivers return None and fall through to the generic lowering — where
/// a genuine user-method call fails to compile (loud), never silently
/// drops exception propagation.
pub(crate) fn receiver_class(
    recv: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
) -> Option<crate::ClassDef> {
    let class_name = match recv {
        ExprType::Name(n) if n.id == "self" => ctx.enclosing_class_name()?.to_string(),
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::Assign { value: ExprType::Call(call), .. }) => {
                match call.func.as_ref() {
                    ExprType::Name(cn) => cn.id.clone(),
                    _ => return None,
                }
            }
            _ => return None,
        },
        // Composition: `self.field.method()` resolves through the owner
        // class's field types.
        ExprType::Attribute(attr) => {
            let owner = receiver_class(&attr.value, ctx, symbols)?;
            owner.field_class(&attr.attr, symbols)?
        }
        _ => return None,
    };
    match symbols.get(&class_name) {
        Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
        _ => None,
    }
}

/// Resolve a call's arguments against the callee's signature, in Python's
/// order: positionals fill left to right, keywords map by name, missing
/// parameters take their default values, and every mismatch Python would
/// raise a TypeError for is a conversion-time error.
fn map_call_arguments(
    func: &crate::FunctionDef,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<Vec<TokenStream>, Box<dyn std::error::Error>> {
    let fname = &func.name;
    // Optional-annotated parameters take Option values: the Option-slot
    // lowering wraps plain arguments in Some, passes None and
    // already-Option values (dict.get, another optional name, an
    // Optional-returning call) through unwrapped, and handles conditional
    // arms independently.
    let fill = |param: &crate::Parameter,
                expr: &ExprType|
     -> Result<TokenStream, Box<dyn std::error::Error>> {
        let optional = param
            .annotation
            .as_deref()
            .is_some_and(crate::is_optional_annotation);
        if optional {
            crate::lower_optional_value(expr, ctx.clone(), options.clone(), symbols.clone())
        } else {
            expr.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())
        }
    };

    let pos_params: Vec<&crate::Parameter> = func
        .args
        .posonlyargs
        .iter()
        .chain(func.args.args.iter())
        .collect();
    let n = pos_params.len();
    if args.len() > n {
        return Err(format!(
            "{}() takes {} positional argument(s) but {} were given",
            fname,
            n,
            args.len()
        )
        .into());
    }

    let mut slots: Vec<Option<TokenStream>> = vec![None; n];
    for (i, arg) in args.iter().enumerate() {
        slots[i] = Some(fill(pos_params[i], arg)?);
    }

    let mut kwonly_slots: Vec<Option<TokenStream>> = vec![None; func.args.kwonlyargs.len()];
    for kw in keywords {
        let Some(kw_name) = &kw.arg else {
            return Err(format!(
                "**kwargs unpacking in a call to {}() is not supported",
                fname
            )
            .into());
        };
        if let Some(idx) = pos_params.iter().position(|p| &p.arg == kw_name) {
            let value = fill(pos_params[idx], &kw.value)?;
            if idx < func.args.posonlyargs.len() {
                return Err(format!(
                    "{}(): parameter `{}` is positional-only and cannot be passed by keyword",
                    fname, kw_name
                )
                .into());
            }
            if slots[idx].is_some() {
                return Err(format!(
                    "{}() got multiple values for argument `{}`",
                    fname, kw_name
                )
                .into());
            }
            slots[idx] = Some(value);
        } else if let Some(idx) = func
            .args
            .kwonlyargs
            .iter()
            .position(|p| &p.arg == kw_name)
        {
            let value = fill(&func.args.kwonlyargs[idx], &kw.value)?;
            if kwonly_slots[idx].is_some() {
                return Err(format!(
                    "{}() got multiple values for argument `{}`",
                    fname, kw_name
                )
                .into());
            }
            kwonly_slots[idx] = Some(value);
        } else {
            return Err(format!(
                "{}() got an unexpected keyword argument `{}`",
                fname, kw_name
            )
            .into());
        }
    }

    // Defaults align with the tail of the positional parameter list.
    let default_offset = n - func.args.defaults.len();
    for i in 0..n {
        if slots[i].is_none() {
            if i >= default_offset {
                slots[i] = Some(fill(pos_params[i], &func.args.defaults[i - default_offset])?);
            } else {
                return Err(format!(
                    "{}() missing required argument `{}`",
                    fname, pos_params[i].arg
                )
                .into());
            }
        }
    }
    for (i, param) in func.args.kwonlyargs.iter().enumerate() {
        if kwonly_slots[i].is_none() {
            match func.args.kw_defaults.get(i).and_then(|d| d.as_ref()) {
                Some(default) => kwonly_slots[i] = Some(fill(param, default)?),
                None => {
                    return Err(format!(
                        "{}() missing required keyword-only argument `{}`",
                        fname, param.arg
                    )
                    .into())
                }
            }
        }
    }

    Ok(slots
        .into_iter()
        .chain(kwonly_slots)
        .map(|s| s.expect("all argument slots filled"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_of_function() {
        let options = PythonOptions::default();
        let result = crate::parse(
            "def foo(a = 7):
    pass

foo(a=9)",
            "test.py",
        )
        .unwrap();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let code = result
            .to_rust(
                CodeGenContext::Module("test".to_string()),
                options,
                symbols,
            )
            .unwrap()
            .to_string();
        assert!(code.contains("foo (9)"), "generated: {}", code);
    }

    #[test]
    fn unknown_keyword_argument_is_a_conversion_error() {
        // Python raises TypeError for foo(b=9) when foo has no parameter b;
        // silently passing it positionally would be wrong.
        let options = PythonOptions::default();
        let result = crate::parse(
            "def foo(a = 7):
    pass

foo(b=9)",
            "test.py",
        )
        .unwrap();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let err = result
            .to_rust(
                CodeGenContext::Module("test".to_string()),
                options,
                symbols,
            )
            .expect_err("unexpected keyword must not convert");
        assert!(
            format!("{}", err).contains("unexpected keyword"),
            "error: {}",
            err
        );
    }
}
