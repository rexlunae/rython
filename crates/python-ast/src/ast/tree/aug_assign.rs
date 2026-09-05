use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes,
    BinOps, FromPythonString, PyAttributeExtractor,
};

/// Augmented assignment statement (e.g., x += 1, y -= 2, etc.)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AugAssign {
    /// The target being assigned to (left side)
    pub target: ExprType,
    /// The operator (+=, -=, *=, etc.)
    pub op: BinOps,
    /// The value being assigned (right side)
    pub value: ExprType,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for AugAssign {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract target
        let target = ob.extract_attr_with_context("target", "augmented assignment target")?;
        let target: ExprType = target.extract()?;
        
        // Extract operator
        let op = ob.extract_attr_with_context("op", "augmented assignment operator")?;
        let op_type_str = op.extract_type_name("augmented assignment operator")?;
        let op = BinOps::parse_or_unknown(&op_type_str);
        
        // Extract value
        let value = ob.extract_attr_with_context("value", "augmented assignment value")?;
        let value: ExprType = value.extract()?;
        
        Ok(AugAssign {
            target,
            op,
            value,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for AugAssign {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for AugAssign {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process the value for symbols, but don't add new symbols for augmented assignment
        self.value.find_symbols(symbols)
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Issue #115: a compound assignment to a `global`-declared name
        // whose module binding is a MUTABLE static is a read-modify-write
        // through the static's helpers: load the global, evaluate the
        // operand, combine, store — CPython's LOAD/op/STORE order, and
        // just as non-atomic. Each helper call takes the lock briefly, so
        // an operand reading the same global cannot deadlock. (A
        // class-instance global — issue #189 — never reaches here: any
        // augmented assignment disqualifies the name from the typed
        // pattern in module.rs, leaving the boxed static and its loud
        // conversion error.)
        if let ExprType::Name(n) = &self.target
            && options.scope_global_writables.contains(&n.id)
            && let Some(kind) = options.mutable_statics.get(&n.id)
        {
            let ident = crate::safe_ident(&n.id);
            let global_ref = kind.static_ref(&ident);
            let value = self
                .value
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let elem = quote!(__rython_g);
            let combined = combine_op(&self.op, &elem, &value)?;
            return Ok(quote! {
                {
                    let __rython_g = stdpython::py_global_read(#global_ref);
                    stdpython::py_global_write(#global_ref, #combined);
                }
            });
        }

        // Compound assignment to a subscript (`counts[k] += 1`) is a
        // read-modify-write: the Load lowering of the target is a cloned
        // temporary (py_index), not a place, so read via py_index, combine,
        // and store back via py_set_index. The index is evaluated once.
        if let ExprType::Subscript(sub) = &self.target {
            // The receiver must be a place (see subscript_receiver_place);
            // a cloned receiver would silently drop the write-back.
            let receiver = crate::subscript_receiver_place(
                &sub.value,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )?;
            // String-keyed dicts (literal or `dict[str, V]`) store through
            // py_set_index with an owned String key; literal `"a"` keys
            // are coerced at the call site like plain stores (assign.rs).
            let string_keyed_dict = matches!(
                sub.value.as_ref(),
                ExprType::Name(n)
                    if matches!(
                        options.name_types.get(&n.id),
                        Some(crate::TypeInfo::Dict(k, _))
                            if matches!(**k, crate::TypeInfo::String)
                    )
            );
            let index = match &sub.kind {
                crate::SubscriptKind::Index(index) => {
                    if string_keyed_dict {
                        crate::render_typed(
                            index,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?
                    } else {
                        index
                            .clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())?
                    }
                }
                crate::SubscriptKind::Slice { .. } => {
                    return Err(
                        "augmented assignment to a slice (`x[a:b] += ...`) is not supported"
                            .to_string()
                            .into(),
                    )
                }
            };
            let value = self.value.to_rust(ctx, options, symbols)?;
            let elem = quote!(__rython_elem);
            let combined = combine_op(&self.op, &elem, &value)?;
            // The receiver place is bound once so a nested chain
            // (`grid[i][j] += 1`) evaluates its intermediate lookups — and
            // any side effects in their indices — exactly once.
            return Ok(quote! {
                {
                    let __rython_recv = &mut (#receiver);
                    let __rython_idx = #index;
                    let __rython_elem = (__rython_recv).py_index(__rython_idx.clone())?;
                    (__rython_recv).py_set_index(__rython_idx, #combined)?;
                }
            });
        }

        // A PROPERTY target (`d.x += v` where `x` is `@property` with a
        // setter): the read goes through the getter and the store through
        // the setter call, the current value bound first, then the
        // operand, then the setter's (mutable) borrow — the same order as
        // a field (Devin review on #331).
        if let ExprType::Attribute(attr) = &self.target
            && let Some((class, _)) = crate::receiver_class(&attr.value, &ctx, &symbols, &options)
            && class.is_property_setter(&attr.attr)
        {
            let recv = attr.value.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let load = self.target.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let value = self.value.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let setter = crate::safe_ident(&format!("{}_set", attr.attr));
            let combined = combine_op(&self.op, &quote!(__rython_load), &quote!(__rython_val))?;
            let receiver_is_self = matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "self");
            if crate::ast::tree::shared::is_shared(&class.name) && !receiver_is_self {
                return Ok(quote! {{
                    let __rython_load = #load;
                    let __rython_val = #value;
                    (#recv).borrow_mut().#setter(#combined)?;
                }});
            }
            return Ok(quote! {{
                let __rython_load = #load;
                let __rython_val = #value;
                #recv.#setter(#combined)?;
            }});
        }
        // The `self.<field>` target's Rust type, captured before the moves
        // below: an Option or boxed field changes the aug-op (the inner
        // arithmetic).
        let target_field_rust_ty =
            target_field_ty(&self.target, &ctx, &options, &symbols);

        // An attribute target (`self.age += 1`) is a read-modify-write on a
        // place. In a generic trait default the LOAD accessor clones
        // (`self.age()`) while the STORE must go through the mutable accessor
        // (`*self.age_mut()`), so the two sides render differently.
        // A read-modify-write through a SHARED receiver's mutable borrow
        // (`c.n += 1` on a PyRef): the load's `Ref` temporary would live
        // to the end of the store statement, so the load and the operand
        // are bound first and the `borrow_mut` taken after (the same rule
        // the plain store applies; the evaluation on issue #137).
        let shared_store = matches!(&self.target, ExprType::Attribute(a)
            if crate::ast::tree::attribute::shared_receiver(&a.value, &ctx, &symbols, &options));
        let mut prelude = TokenStream::new();
        let mut shared_load: Option<TokenStream> = None;
        let (target, target_load) = match &self.target {
            ExprType::Attribute(attr) => {
                let store = crate::ast::tree::attribute::to_rust_place(
                    &attr.value,
                    &attr.attr,
                    &ctx,
                    &options,
                    &symbols,
                    true,
                )?;
                let load = self
                    .target
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                if shared_store {
                    shared_load = Some(load);
                    (store, quote!(__rython_load))
                } else {
                    (store, load)
                }
            }
            _ => {
                let t = self.target.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                (t.clone(), t)
            }
        };
        // Whether the RHS is itself Option-typed (needed by the Option
        // target arm below) — captured before `to_rust` moves `self.value`.
        let value_is_option = matches!(
            crate::infer_type(Some(&ctx), &self.value, &options, &symbols),
            crate::TypeInfo::Option(_)
        );

        // A shared store binds the target's CURRENT value first — Python
        // evaluates the target once, then the operand, then stores — so
        // an operand that mutates the same object (`c.n -= take(c)`)
        // never changes what the operation reads, whatever the operator
        // (Devin review on #331: the typed fast paths `-=`/`*=` read the
        // field after the operand ran). Then the operand: a name or a
        // constant takes no borrow and renders in place; anything else
        // may read the same object and is bound too.
        if shared_store {
            let load = shared_load.take().expect("the shared store rendered its load");
            prelude.extend(quote!(let __rython_load = #load;));
        }
        let operand_reads = !matches!(self.value, ExprType::Name(_) | ExprType::Constant(_));
        let value = self.value.to_rust(ctx, options, symbols)?;
        let value = if shared_store && operand_reads {
            prelude.extend(quote!(let __rython_val = #value;));
            quote!(__rython_val)
        } else {
            value
        };
        // A compound Rust operator (`-=`, `*=`, `&=`, ...) reads the place
        // at store time: through a shared receiver the store reads the
        // bound load instead, one rule for every operator.
        let compound = |op: &BinOps, in_place: TokenStream| -> Result<TokenStream, Box<dyn std::error::Error>> {
            if shared_store {
                let combined = combine_op(op, &target_load, &value)?;
                Ok(quote!(#target = #combined))
            } else {
                Ok(in_place)
            }
        };

        // Generate the appropriate augmented assignment operator
        let stmt: Result<TokenStream, Box<dyn std::error::Error>> = match self.op {
            // `+=` mirrors Python's `+` (string concat, list concat,
            // numeric promotion) via PyAdd. An OPTION-typed target
            // (`self._data += data` where the field is `bytes | None` —
            // urllib3's DeflateDecoder, whose `_data` widens when a None
            // store joins) operates on the INNER value and stores back
            // wrapped; a None here is CPython's TypeError (`None +
            // bytes`), a loud §12.2 panic with the message (the guard in
            // real code prevents it). A BOXED (PyValue) target does the
            // read-modify-write through the runtime py_add (the boxed
            // arithmetic).
            BinOps::Add => match &target_field_rust_ty {
                Some(crate::TypeInfo::Option(inner)) => {
                    let type_name = option_inner_py_name(inner);
                    // CPython's exact message text, spliced as one literal
                    // (`None + b""` → "unsupported operand type(s) for +:
                    // 'NoneType' and 'bytes'").
                    let msg = format!(
                        "unsupported operand type(s) for +=: 'NoneType' and '{}'",
                        type_name
                    );
                    // The RHS may itself be Option-typed: unwrap both.
                    let value = if value_is_option {
                        quote! {
                            match (#value).clone() {
                                Some(__rython_w) => __rython_w,
                                None => panic!(#msg),
                            }
                        }
                    } else {
                        quote!(#value)
                    };
                    Ok(quote! {
                        #target = {
                            let __rython_w = #value;
                            match (#target_load).clone() {
                                Some(__rython_v) => {
                                    Some((__rython_v).py_add(&__rython_w))
                                }
                                None => panic!(#msg),
                            }
                        }
                    })
                }
                _ => Ok(quote!(#target = (#target_load).py_add(&(#value)))),
            },
            BinOps::Sub => {
                // An OPTION-typed target (`self.length_remaining -= n`
                // where the field is `i64 | None` — urllib3's
                // HTTPResponse): the Python guards with `is not None`
                // before subtracting; the aug-assign operates on the
                // INNER value. A None here is CPython's TypeError — a
                // loud §12.2 panic with the message (the guard in real
                // code prevents it). A BOXED (PyValue) target does the
                // read-modify-write through the runtime py_sub (the
                // boxed int arithmetic).
                match &target_field_rust_ty {
                    Some(crate::TypeInfo::Option(_)) => {
                        // The RHS may itself be Option-typed
                        // (`self.chunk_left -= amt` where amt is
                        // `int | None` — urllib3's _fp_read): unwrap both.
                        let value = if value_is_option {
                            quote! {
                                match (#value).clone() {
                                    Some(__rython_w) => __rython_w,
                                    None => panic!(
                                        "unsupported operand type(s) for -=: 'NoneType' and 'i64'"
                                    ),
                                }
                            }
                        } else {
                            quote!(#value)
                        };
                        Ok(quote! {
                            #target = {
                                let __rython_w = #value;
                                match (#target_load).clone() {
                                    Some(__rython_v) => {
                                        Some((__rython_v).py_sub(&__rython_w))
                                    }
                                    None => panic!(
                                        "unsupported operand type(s) for -=: 'NoneType' and 'i64'"
                                    ),
                                }
                            }
                        })
                    }
                    Some(crate::TypeInfo::PyValue) => {
                        Ok(quote!(#target = (#target_load).py_sub(&(#value))))
                    }
                    _ => compound(&BinOps::Sub, quote!(#target -= #value)),
                }
            }
            BinOps::Mult => compound(&BinOps::Mult, quote!(#target *= #value)),
            // Python's `/` is TRUE division: `x /= 2` on an int yields a
            // float. Route through py_div (numeric → f64, numpy arrays →
            // elementwise) instead of Rust's truncating `/=` or an `as f64`
            // cast (an int target then fails to compile, which is loud
            // rather than quietly wrong). The `?` propagates a catchable
            // ZeroDivisionError (issue #107).
            BinOps::Div => Ok(quote!(#target = py_div(#target_load, #value)?)),
            // Python // and % floor toward negative infinity / take the
            // divisor's sign; use the stdpython helpers instead of Rust's
            // truncating operators. The `?` propagates a catchable
            // ZeroDivisionError (issue #75).
            BinOps::FloorDiv => Ok(quote!(#target = py_floordiv(#target_load, #value)?)),
            BinOps::Mod => Ok(quote!(#target = py_mod(#target_load, #value)?)),
            BinOps::BitAnd => compound(&BinOps::BitAnd, quote!(#target &= #value)),
            BinOps::BitOr => {
                // An Option-typed target (`ssl_options |= X` where the
                // local is `int | None` — urllib3's ssl_): OR the inner
                // value; a None is CPython's TypeError (loud §12.2 panic
                // — the guard in real code prevents it).
                match &target_field_rust_ty {
                    Some(crate::TypeInfo::Option(_)) => Ok(quote! {
                        #target = match (#target_load).clone() {
                            Some(__rython_v) => Some(__rython_v | (#value)),
                            None => panic!(
                                "unsupported operand type(s) for |=: 'NoneType' and 'i64'"
                            ),
                        }
                    }),
                    _ => compound(&BinOps::BitOr, quote!(#target |= #value)),
                }
            }
            BinOps::BitXor => compound(&BinOps::BitXor, quote!(#target ^= #value)),
            BinOps::LShift => compound(&BinOps::LShift, quote!(#target <<= #value)),
            BinOps::RShift => compound(&BinOps::RShift, quote!(#target >>= #value)),
            BinOps::Pow => {
                // Rust doesn't have **= operator, so we need to expand it
                Ok(quote!(#target = py_pow(#target_load, #value)))
            },
            BinOps::MatMult => {
                // Matrix multiplication assignment - not directly supported in Rust
                // Would need specific matrix library support
                Err(format!("Matrix multiplication assignment not supported in Rust").into())
            },
            BinOps::Unknown => {
                Err(format!("Unknown augmented assignment operator").into())
            },
        };
        let stmt = stmt?;
        Ok(if prelude.is_empty() {
            stmt
        } else {
            quote!({ #prelude #stmt })
        })
    }
}

/// The read-modify-write combination for a compound assignment: how the
/// current element and the operand produce the stored value.
fn combine_op(
    op: &BinOps,
    elem: &TokenStream,
    value: &TokenStream,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    Ok(match op {
        BinOps::Add => quote!((#elem).py_add(&(#value))),
        BinOps::Sub => quote!(#elem - #value),
        BinOps::Mult => quote!(#elem * #value),
        BinOps::Div => quote!(py_div(#elem, #value)?),
        BinOps::FloorDiv => quote!(py_floordiv(#elem, #value)?),
        BinOps::Mod => quote!(py_mod(#elem, #value)?),
        BinOps::Pow => quote!(py_pow(#elem, #value)),
        BinOps::BitAnd => quote!(#elem & #value),
        BinOps::BitOr => quote!(#elem | #value),
        BinOps::BitXor => quote!(#elem ^ #value),
        BinOps::LShift => quote!(#elem << #value),
        BinOps::RShift => quote!(#elem >> #value),
        other => {
            return Err(format!(
                "augmented assignment operator {:?} not supported on subscripts",
                other
            )
            .into())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_add_assign, "x += 1", "test.py");
    create_parse_test!(test_sub_assign, "x -= 1", "test.py");
    create_parse_test!(test_mul_assign, "x *= 2", "test.py");
    create_parse_test!(test_div_assign, "x /= 3", "test.py");
    create_parse_test!(test_mod_assign, "x %= 4", "test.py");
    create_parse_test!(test_pow_assign, "x **= 2", "test.py");
    create_parse_test!(test_bitand_assign, "x &= 5", "test.py");
    create_parse_test!(test_bitor_assign, "x |= 6", "test.py");
    create_parse_test!(test_bitxor_assign, "x ^= 7", "test.py");
    create_parse_test!(test_lshift_assign, "x <<= 2", "test.py");
    create_parse_test!(test_rshift_assign, "x >>= 3", "test.py");
}

/// The CPython type name of an Option's inner type, for the §12.2 panic
/// messages mirroring CPython's `unsupported operand type(s)` text
/// (`None + b""` names `'bytes'`, `x -= 1` on None names `'int'`). A
/// non-primitive inner falls back to the empty string — the panic still
/// names the operator and the None side.
fn option_inner_py_name(inner: &crate::TypeInfo) -> String {
    match inner {
        crate::TypeInfo::Int => "int".to_string(),
        crate::TypeInfo::Float => "float".to_string(),
        crate::TypeInfo::Bool => "bool".to_string(),
        crate::TypeInfo::String | crate::TypeInfo::StrRef | crate::TypeInfo::StrOrBytes => {
            "str".to_string()
        }
        crate::TypeInfo::Bytes => "bytes".to_string(),
        crate::TypeInfo::Vec(_) => "list".to_string(),
        crate::TypeInfo::Dict(_, _) => "dict".to_string(),
        _ => String::new(),
    }
}

/// The RUST type of an aug-assign target, when it is known: a
/// `self.<field>` through the class table, an OPTION-typed name
/// (`connect -= 1` where the parameter is `int | None` — urllib3's
/// Retry), or a local assigned from a self-field (`total = self.total`).
/// An Option or boxed target needs the inner arithmetic — the Option
/// unwrap or the runtime py_sub.
fn target_field_ty(
    target: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<crate::TypeInfo> {
    match target {
        // infer_type's self-field arm resolves `self.<field>` through the
        // class table (round 99 — replaces the self_field_rust_ty
        // fallback, Directive 2).
        ExprType::Attribute(attr)
            if crate::ast::tree::visit::is_self(attr.value.as_ref()) =>
        {
            Some(crate::infer_type(Some(ctx), target, options, symbols))
        }
        ExprType::Name(n) => {
            if let Some(t) = options.name_types.get(&n.id) {
                return Some(t.clone());
            }
            // A local assigned from a self-field (`total = self.total` —
            // urllib3's Retry): the field's type, through the same arm.
            if let Some(crate::SymbolTableNode::Assign {
                value: field_expr @ ExprType::Attribute(attr),
                ..
            }) = symbols.get(&n.id)
                && crate::ast::tree::visit::is_self(attr.value.as_ref())
            {
                return Some(crate::infer_type(
                    Some(ctx),
                    field_expr,
                    options,
                    symbols,
                ));
            }
            None
        }
        _ => None,
    }
}
