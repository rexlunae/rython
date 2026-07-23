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

        // Python methods whose Rust inherent namesakes have DIFFERENT
        // semantics (or the wrong shape) are rewritten here; methods with no
        // Rust conflict resolve through the stdpython PyListOps/PyStrOps
        // traits without any rewriting.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
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
                // str::split (inherent) returns an iterator, so both forms
                // map to the PyStrOps versions returning Vec<String>.
                ("split", []) => {
                    return Ok(quote!((#receiver).py_split_whitespace()));
                }
                ("split", [sep]) => {
                    return Ok(quote!((#receiver).py_split(&(#sep))));
                }
                // str.find returns -1 when absent; str::find an Option.
                ("find", [needle]) => {
                    return Ok(quote!((#receiver).py_find(&(#needle))));
                }
                _ => {}
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_of_function() {
        let options = PythonOptions::default();
        let result = crate::parse(
            "def foo(a = 7):
    pass

foo(b=9)",
            "test.py",
        )
        .unwrap();
        let _code = result
            .to_rust(
                CodeGenContext::Module("test".to_string()),
                options,
                SymbolTableScopes::new(),
            )
            .unwrap();
    }
}
