use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, PyAttributeExtractor, extract_list
};

use super::Statement;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct For {
    pub target: ExprType,
    pub iter: ExprType,
    pub body: Vec<Statement>,
    pub orelse: Vec<Statement>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for For {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let target = ob.extract_attr_with_context("target", "for loop target")?;
        let iter = ob.extract_attr_with_context("iter", "for loop iterator")?;
        
        let target = target
            .extract()
            .map_err(|e| crate::extraction_failure("for loop target", &ob, e))?;
        let iter = iter
            .extract()
            .map_err(|e| crate::extraction_failure("for loop iterator", &ob, e))?;
        
        let body: Vec<Statement> = extract_list(&ob, "body", "for body statements")?;
        let orelse: Vec<Statement> = extract_list(&ob, "orelse", "for else statements")?;
        
        Ok(For {
            target,
            iter,
            body,
            orelse,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(For { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for For {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let symbols = self.target.find_symbols(symbols);
        let symbols = self.iter.find_symbols(symbols);
        let symbols = self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc));
        self.orelse.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Python's loop variable is function-scoped: when the target name
        // LEAKS (a later statement reads it, unshadowed), the scope analysis
        // hoists it and this loop must STORE into that binding — a fresh
        // `for` binding would shadow the hoisted variable and the loop
        // writes would silently vanish (issue #80). A target that merely
        // shares a name with a hoisted variable but whose value is never
        // observed keeps its fresh per-loop binding (otherwise a hoisted
        // binding of a different type, or a possibly-uninitialized one,
        // would break the generated code). Loop-local names also keep their
        // fresh binding, and an index name that is never read in the body
        // (or else clause) lowers to `_` so rustc does not warn about the
        // unused binding (issue #101). The reference checks run before the
        // body is consumed below.
        let leaked = options.leaked_loop_targets.as_ref().clone();
        let target_names = {
            let mut names = Vec::new();
            collect_target_names(&self.target, &mut names);
            names
        };
        let unused_index = matches!(
            &self.target,
            ExprType::Name(n)
                if !leaked.contains(&n.id)
                    && !crate::name_referenced_in(&self.body, &n.id)
                    && !crate::name_referenced_in(&self.orelse, &n.id)
        );
        let any_hoisted = target_names
            .iter()
            .any(|n| leaked.contains(*n));
        let has_else = !self.orelse.is_empty();
        // Break-tracking is only needed when the else clause could be skipped
        // by a break belonging to this loop; otherwise the flag would be
        // declared mutable but never written.
        let tracks_break = has_else && crate::loop_body_has_direct_break(&self.body);
        // Body statements compile inside a Loop context so `break` can honor
        // the else clause; the else clause itself is outside the loop.
        let body_ctx = crate::CodeGenContext::Loop {
            has_else: tracks_break,
            parent: Box::new(ctx.clone()),
        };
        let body_stmts: Result<Vec<_>, _> = self.body
            .into_iter()
            .map(|stmt| stmt.to_rust(body_ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let body_stmts = body_stmts?;

        // When any target name leaks, the loop element binds to a temp
        // (`__rython_elt`) and the target lowering stores it into the
        // real bindings; otherwise the target pattern binds directly.
        let (loop_target, loop_inner): (TokenStream, TokenStream) = if any_hoisted {
            let mut stmts = Vec::new();
            let mut counter = 0usize;
            lower_loop_target(
                &self.target,
                quote!(__rython_elt),
                &leaked,
                &mut stmts,
                &mut counter,
            );
            (quote!(__rython_elt), quote!({ #(#stmts)* #(#body_stmts;)* }))
        } else {
            let target = if unused_index {
                quote!(_)
            } else {
                self.target.to_rust(ctx.clone(), options.clone(), symbols.clone())?
            };
            (target, quote!(#(#body_stmts;)*))
        };
        // Issue #155: `for x in iter(callable, sentinel):` — the
        // two-argument iter() form calls `callable()` until it produces
        // `sentinel` (botocore's chunked payload reads). Desugared into a
        // `loop` that binds each result, breaks on equality with the
        // sentinel (bound once, before the loop), and otherwise runs the
        // body — a sentinel break is normal exhaustion, so it never sets
        // the for/else break flag. The callable lowers as an ordinary
        // zero-argument call, so everything the call machinery supports
        // (module functions, partial-bound names) works here.
        if let ExprType::Call(c) = &self.iter
            && matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "iter")
            && symbols.get("iter").is_none()
            && c.args.len() == 2
            && c.keywords.is_empty()
        {
            let call_expr = ExprType::Call(crate::Call {
                func: Box::new(c.args[0].clone()),
                args: Vec::new(),
                keywords: Vec::new(),
            });
            let call = call_expr.to_rust(body_ctx.clone(), options.clone(), symbols.clone())?;
            let sentinel = crate::render_reused(
                &c.args[1],
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )?;
            let core = quote! {
                let __rython_sentinel = #sentinel;
                loop {
                    let __rython_item = #call;
                    if __rython_item == __rython_sentinel {
                        break;
                    }
                    let #loop_target = __rython_item;
                    #loop_inner
                }
            };
            if !has_else {
                return Ok(quote!({ #core }));
            }
            let else_stmts: Result<Vec<_>, _> = self
                .orelse
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let else_stmts = else_stmts?;
            return Ok(if tracks_break {
                quote! {
                    {
                        let mut __rython_broke = false;
                        #core
                        if !__rython_broke {
                            #(#else_stmts;)*
                        }
                    }
                }
            } else {
                quote! {
                    {
                        #core
                        #(#else_stmts;)*
                    }
                }
            });
        }

        // The iterable is CONSUMED by the loop (IntoIterator::into_iter),
        // so a name reused later is cloned here — Python's for loop does
        // not consume its iterable (issue #109, M2: iteration over
        // inferred parameters). The reuse-clone rule's `T: Clone` bound
        // makes this compile for generic parameters.
        let mut iter = crate::render_reused(
            &self.iter,
            ctx.clone(),
            options.clone(),
            symbols.clone(),
        )?;
        // A DICT-typed iterable (`for w in counts` — text_stats's starts-
        // with-q comprehension, round 99): Python iterates the dict's
        // KEYS, but the typed PyDict's IntoIterator yields the (K, V)
        // pairs. Route through the keys view.
        if matches!(
            crate::infer_type(Some(&ctx), &self.iter, &options, &symbols),
            crate::TypeInfo::Dict(_, _)
        ) {
            iter = quote!(#iter . py_keys ());
        }
        // A TUPLE-LITERAL iterable (`for key in ("headers",
        // "_proxy_headers", "_socks_options")` — urllib3's poolmanager):
        // Python iterates the tuple; rython's tuple value is a Rust tuple,
        // which is not IntoIterator (E0277 x4). An all-constant tuple
        // iterates as an array of the same element tokens — STRING
        // literals own themselves (the loop target feeds String-keyed
        // dict calls: `request_context.pop(key, None)`,
        // `key in request_context`).
        if let ExprType::Tuple(t) = &self.iter
            && t.elts.iter().all(|e| {
                matches!(
                    e,
                    ExprType::Constant(_) | ExprType::UnaryOp(_)
                )
            })
        {
            let mut elts = Vec::with_capacity(t.elts.len());
            for elt in &t.elts {
                let tok = elt.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                if matches!(
                    elt,
                    ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_)))
                ) {
                    elts.push(quote!((#tok).to_string()));
                } else {
                    elts.push(tok);
                }
            }
            iter = quote!([#(#elts),*]);
        }

        if !has_else {
            Ok(quote! {
                for #loop_target in #iter {
                    #loop_inner
                }
            })
        } else {
            // Python's for/else: the else clause runs iff the loop finished
            // without hitting `break` (which sets __rython_broke).
            let else_stmts: Result<Vec<_>, _> = self.orelse
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let else_stmts = else_stmts?;

            if tracks_break {
                Ok(quote! {
                    {
                        let mut __rython_broke = false;
                        for #loop_target in #iter {
                            #loop_inner
                        }
                        if !__rython_broke {
                            #(#else_stmts;)*
                        }
                    }
                })
            } else {
                // No break can skip the else clause; run it unconditionally.
                Ok(quote! {
                    {
                        for #loop_target in #iter {
                            #loop_inner
                        }
                        #(#else_stmts;)*
                    }
                })
            }
        }
    }
}

/// Collect the names in a `for` target (`i` and `v` in `for i, v in ...`).
pub(crate) fn collect_target_names<'a>(target: &'a ExprType, out: &mut Vec<&'a str>) {
    match target {
        ExprType::Name(n) => out.push(&n.id),
        ExprType::Tuple(t) => {
            for elt in &t.elts {
                collect_target_names(elt, out);
            }
        }
        _ => {}
    }
}

/// Lower a `for` target into statements that bind/store each name: leaked
/// names get a plain store into the prologue-managed binding, loop-local
/// names a fresh `let`, and tuple targets destructure through unique temps
/// so nested patterns never shadow a leaked outer name.
pub(crate) fn lower_loop_target(
    target: &ExprType,
    value: TokenStream,
    hoisted: &std::collections::HashSet<String>,
    out: &mut Vec<TokenStream>,
    counter: &mut usize,
) {
    match target {
        ExprType::Name(n) => {
            let id = crate::safe_ident(&n.id);
            if hoisted.contains(&n.id) {
                out.push(quote!(#id = #value;));
            } else {
                out.push(quote!(let #id = #value;));
            }
        }
        ExprType::Tuple(t) => {
            let mut pats = Vec::new();
            let mut inner = Vec::new();
            for elt in &t.elts {
                let tname = format_ident!("__rython_elt_{}", *counter);
                *counter += 1;
                pats.push(quote!(#tname));
                let v = quote!(#tname);
                lower_loop_target(elt, v, hoisted, &mut inner, counter);
            }
            // The temps are unique (the counter), so the destructure
            // splices WITHOUT a block: a closing block would end the
            // fresh bindings' scope before the loop body runs (`for i,
            // row in enumerate(m): for j, x in enumerate(row): if i ==
            // j` — the inner body reads `i` outside the block, round 99,
            // matrix). The temps are part of the same destructure
            // statement.
            out.push(quote!(let (#(#pats),*) = #value;));
            out.extend(inner);
        }
        // Subscript/attribute targets cannot reach here (they collect no
        // names, so `any_hoisted` is false and the direct pattern path
        // runs instead); panic loudly rather than emit a silent drop.
        _ => unreachable!("for target must be a name or a tuple of names"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_simple_for, "for x in range(10):\n    print(x)", "for_test.py");
    create_parse_test!(test_for_else, "for x in range(10):\n    print(x)\nelse:\n    print('done')", "for_test.py");
    create_parse_test!(test_for_list, "for item in [1, 2, 3]:\n    print(item)", "for_test.py");
}