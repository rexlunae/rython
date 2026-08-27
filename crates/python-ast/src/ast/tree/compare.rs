use proc_macro2::TokenStream;
use pyo3::{
    Borrowed, Bound, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods,
};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, CompareNotYetImplemented, ExprType, PythonOptions, SymbolTableScopes,
    dump, err_from, extraction_failure,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Compares {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    Is,
    IsNot,
    In,
    NotIn,

    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Compare {
    pub ops: Vec<Compares>,
    pub left: Box<ExprType>,
    pub comparators: Vec<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Compare {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        tracing::debug!("ob: {}", dump(&ob, None)?);

        // Python allows for multiple comparators, rust we only supports one, so we have to rewrite the comparison a little.
        let ops_bound: Vec<Bound<PyAny>> = ob
            .getattr("ops")
            .map_err(|e| extraction_failure("comparison operators", &ob, e))?
            .extract()
            .map_err(|e| extraction_failure("comparison operators", &ob, e))?;

        let mut op_list = Vec::new();

        for op in ops_bound.iter() {
            let op_type = op
                .get_type()
                .name()
                .map_err(|e| extraction_failure("comparison operator type", &ob, e))?;

            let op_type_str: String = op_type.extract()?;
            let op = match op_type_str.as_str() {
                "Eq" => Compares::Eq,
                "NotEq" => Compares::NotEq,
                "Lt" => Compares::Lt,
                "LtE" => Compares::LtE,
                "Gt" => Compares::Gt,
                "GtE" => Compares::GtE,
                "Is" => Compares::Is,
                "IsNot" => Compares::IsNot,
                "In" => Compares::In,
                "NotIn" => Compares::NotIn,

                _ => {
                    tracing::debug!("Found unknown Compare with type: {}", op_type_str);
                    Compares::Unknown
                }
            };
            op_list.push(op);
        }

        let left = ob
            .getattr("left")
            .map_err(|e| extraction_failure("left", &ob, e))?;

        let comparators = ob
            .getattr("comparators")
            .map_err(|e| extraction_failure("comparators", &ob, e))?;
        tracing::debug!(
            "left: {}, comparators: {}",
            dump(&left, None)?,
            dump(&comparators, None)?
        );

        let left = left
            .extract()
            .map_err(|e| extraction_failure("getting binary operator operand", &ob, e))?;
        let comparators: Vec<ExprType> = comparators
            .extract()
            .map_err(|e| extraction_failure("comparators", &ob, e))?;

        tracing::debug!(
            "left: {:?}, comparators: {:?}, op: {:?}",
            left,
            comparators,
            op_list
        );

        return Ok(Compare {
            ops: op_list,
            left: Box::new(left),
            comparators: comparators,
        });
    }
}

impl CodeGen for Compare {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // A CHAINED comparison (`a < b < c`) must evaluate every operand
        // exactly once — the naive `a < b && b < c` expansion evaluates
        // `b` twice, running its side effects twice and, for a
        // non-deterministic operand, even yielding a different answer
        // than Python. Bind each operand to a temporary at the point
        // Python evaluates it, and nest the remaining tests inside the
        // `&&` so a false prefix leaves later operands unevaluated, as
        // Python's short circuit does. The temporaries bind by
        // REFERENCE so an operand that is a live variable is not moved
        // out of the enclosing scope.
        if self.ops.len() > 1 {
            return self.to_rust_chained(ctx, options, symbols);
        }
        let mut outer_ts = TokenStream::new();
        // Python chains comparisons pairwise: `a < b < c` means
        // `a < b && b < c`, so each comparator becomes the left operand of
        // the next comparison.
        let mut left = self
            .left
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        let ops = self.ops.clone();
        let comparators = self.comparators.clone();

        let mut index = 0;
        for op in ops.iter() {
            let comparator_ast = comparators
                .get(index)
                .ok_or("comparison has more operators than comparators")?;
            // The operand AST feeding this comparison's left side: the
            // original left for the first op, the previous comparator after.
            let left_ast = if index == 0 {
                self.left.as_ref()
            } else {
                &comparators[index - 1]
            };
            // `x is None` / `x is not None` test None-ness, not equality:
            // Option values report is_none(), plain values are never None.
            if matches!(op, Compares::Is | Compares::IsNot) {
                let none_check = if crate::is_none_expr(comparator_ast) {
                    Some(left_ast)
                } else if crate::is_none_expr(left_ast) {
                    Some(comparator_ast)
                } else {
                    None
                };
                if let Some(operand) = none_check {
                    let operand_tokens =
                        operand
                            .clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    let tokens = match op {
                        Compares::Is => quote!((#operand_tokens).py_is_none()),
                        _ => quote!(!(#operand_tokens).py_is_none()),
                    };
                    index += 1;
                    left = quote!(#operand_tokens);
                    outer_ts.extend(tokens);
                    if index < ops.len() {
                        outer_ts.extend(quote!( && ));
                    }
                    continue;
                }
                // `x is False` / `x is True` on a boxed PyValue (issue
                // #121): test the Bool member, not Rust reference equality
                // (`&x == &false` would not type-check).
                if matches!(op, Compares::Is | Compares::IsNot) {
                    let bool_lit = |e: &ExprType| -> Option<bool> {
                        match e {
                            ExprType::Constant(c) => match &c.0 {
                                Some(litrs::Literal::Bool(b)) => Some(b.value()),
                                _ => None,
                            },
                            _ => None,
                        }
                    };
                    let pyvalue_operand = |e: &ExprType| -> Option<TokenStream> {
                        if let ExprType::Name(n) = e
                            && options
                                .name_types
                                .get(&n.id)
                                .is_some_and(|t| matches!(t, crate::TypeInfo::PyValue))
                        {
                            e.clone()
                                .to_rust(ctx.clone(), options.clone(), symbols.clone())
                                .ok()
                        } else {
                            None
                        }
                    };
                    let (val, operand) = if let Some(b) = bool_lit(comparator_ast) {
                        (b, pyvalue_operand(left_ast))
                    } else if let Some(b) = bool_lit(left_ast) {
                        (b, pyvalue_operand(comparator_ast))
                    } else {
                        (false, None)
                    };
                    if let Some(operand) = operand {
                        let tokens = match op {
                            Compares::Is => quote!(
                                (#operand).is_bool() && (#operand).as_bool() == Some(#val)
                            ),
                            _ => quote!(
                                !((#operand).is_bool() && (#operand).as_bool() == Some(#val))
                            ),
                        };
                        index += 1;
                        left = quote!(#operand);
                        outer_ts.extend(tokens);
                        if index < ops.len() {
                            outer_ts.extend(quote!( && ));
                        }
                        continue;
                    }
                    // `x is SomeClass` / `x is not SomeClass`
                    // (`self.ConnectionCls is DummyConnection` — urllib3's
                    // connectionpool): classes cannot be runtime values (the
                    // classes-as-values divergence) — the identity check is
                    // statically false/true.
                    let class_operand = if crate::is_class_value_expr(comparator_ast, &symbols) {
                        Some(left_ast)
                    } else if crate::is_class_value_expr(left_ast, &symbols) {
                        Some(comparator_ast)
                    } else {
                        None
                    };
                    if class_operand.is_some() {
                        let tokens = match op {
                            Compares::Is => quote!(false),
                            _ => quote!(true),
                        };
                        index += 1;
                        outer_ts.extend(tokens);
                        if index < ops.len() {
                            outer_ts.extend(quote!( && ));
                        }
                        continue;
                    }
                }
            }
            let comparator =
                comparator_ast
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            // A GENERIC (inferred) parameter compares with an integer
            // literal converted to the parameter's own type via
            // stdpython's PyFromInt (`B::py_from_int(0)`): Rust std has no
            // int/float cross-PartialOrd, so the bounds
            // `B: PyLe<B> + PyFromInt` are satisfied by both i64 and f64
            // (Python promotes `2.5 <= 0` to a float comparison).
            let comparator = if let ExprType::Name(n) = left_ast {
                if let Some(tv) = options.param_type_vars.get(&n.id) {
                    if matches!(
                        comparator_ast,
                        ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::Integer(_)))
                    ) {
                        quote!(#tv :: py_from_int(#comparator))
                    } else {
                        comparator
                    }
                } else {
                    comparator
                }
            } else {
                comparator
            };
            // Comparisons route through the stdpython PyEq/PyNe/PyLt/PyLe/
            // PyGt/PyGe traits (in scope via `use stdpython::*`): scalars
            // and containers get their existing PartialEq/PartialOrd
            // behaviour (bool result) through blanket impls, while NdArray
            // overrides them to broadcast elementwise and return an array —
            // the same pattern `+` uses with PyAdd.
            let tokens = match op {
                Compares::Eq => quote!((#left).py_eq(&(#comparator))),
                Compares::NotEq => quote!((#left).py_ne(&(#comparator))),
                Compares::Lt => quote!((#left).py_lt(&(#comparator))),
                Compares::LtE => quote!((#left).py_le(&(#comparator))),
                Compares::Gt => quote!((#left).py_gt(&(#comparator))),
                Compares::GtE => quote!((#left).py_ge(&(#comparator))),
                Compares::Is => quote!(&#left == &#comparator),
                Compares::IsNot => quote!(&#left != &#comparator),
                // Python `in` dispatches on the container: substring for
                // strings, key lookup for dicts, element lookup for
                // sequences. The stdpython PyContains trait models that.
                // String-keyed dicts take &String; literal `"a"` keys are
                // owned so the generic impl applies.
                Compares::In => {
                    if matches!(
                        comparator_ast,
                        ExprType::Name(n)
                            if matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Dict(k, _))
                                    if matches!(**k, crate::TypeInfo::String)
                            )
                    ) {
                        let left = crate::render_typed(
                            left_ast,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        quote!((#comparator).py_contains(&(#left)))
                    } else {
                        quote!((#comparator).py_contains(&(#left)))
                    }
                }
                Compares::NotIn => {
                    if matches!(
                        comparator_ast,
                        ExprType::Name(n)
                            if matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Dict(k, _))
                                    if matches!(**k, crate::TypeInfo::String)
                            )
                    ) {
                        let left = crate::render_typed(
                            left_ast,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        quote!(!(#comparator).py_contains(&(#left)))
                    } else {
                        quote!(!(#comparator).py_contains(&(#left)))
                    }
                }

                _ => return Err(err_from(CompareNotYetImplemented(self)).into()),
            };

            index += 1;
            left = comparator;

            outer_ts.extend(tokens);
            if index < ops.len() {
                outer_ts.extend(quote!( && ));
            }
        }
        Ok(outer_ts)
    }
}

impl Compare {
    /// Lower `a OP b OP c ...` with each operand evaluated exactly once
    /// and Python's short-circuit order preserved:
    ///
    /// ```text
    /// { let t0 = &a; let t1 = &b; t0 OP t1 && { let t2 = &c; t1 OP t2 } }
    /// ```
    fn to_rust_chained(
        self,
        ctx: CodeGenContext,
        options: PythonOptions,
        symbols: SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut operands: Vec<&ExprType> = Vec::with_capacity(self.comparators.len() + 1);
        operands.push(self.left.as_ref());
        operands.extend(self.comparators.iter());

        let mut rendered = Vec::with_capacity(operands.len());
        for operand in &operands {
            rendered.push((*operand).clone().to_rust(
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )?);
        }
        let names: Vec<proc_macro2::Ident> = (0..operands.len())
            .map(|i| quote::format_ident!("__rython_cmp{}", i))
            .collect();

        // A None literal is side-effect free and has no nameable type of
        // its own, so it is never bound to a temporary; `is None` tests
        // consume only the other side.
        let is_none: Vec<bool> = operands.iter().map(|e| crate::is_none_expr(e)).collect();
        let bind = |i: usize| -> TokenStream {
            if is_none[i] {
                return quote!();
            }
            let name = &names[i];
            let value = &rendered[i];
            quote!(let #name = &(#value);)
        };

        // The comparison for one link of the chain, over the temporaries.
        let compare_pair = |i: usize| -> Result<TokenStream, Box<dyn std::error::Error>> {
            let op = &self.ops[i];
            let (l, r) = (&names[i], &names[i + 1]);
            if matches!(op, Compares::Is | Compares::IsNot) {
                let operand = if is_none[i + 1] {
                    Some(l)
                } else if is_none[i] {
                    Some(r)
                } else {
                    None
                };
                if let Some(operand) = operand {
                    return Ok(match op {
                        Compares::Is => quote!((#operand).py_is_none()),
                        _ => quote!(!(#operand).py_is_none()),
                    });
                }
            }
            Ok(match op {
                Compares::Eq => quote!((#l).py_eq(#r)),
                Compares::NotEq => quote!((#l).py_ne(#r)),
                Compares::Lt => quote!((#l).py_lt(#r)),
                Compares::LtE => quote!((#l).py_le(#r)),
                Compares::Gt => quote!((#l).py_gt(#r)),
                Compares::GtE => quote!((#l).py_ge(#r)),
                Compares::Is => quote!((#l) == (#r)),
                Compares::IsNot => quote!((#l) != (#r)),
                Compares::In => quote!((#r).py_contains(#l)),
                Compares::NotIn => quote!(!(#r).py_contains(#l)),
                _ => return Err(err_from(CompareNotYetImplemented(self.clone())).into()),
            })
        };

        // Build inside out so each operand is bound immediately before
        // the test that first needs it.
        let mut acc: Option<TokenStream> = None;
        for i in (0..self.ops.len()).rev() {
            let rhs_bind = bind(i + 1);
            let test = compare_pair(i)?;
            acc = Some(match acc {
                None => quote!({ #rhs_bind #test }),
                Some(rest) => quote!({ #rhs_bind #test && #rest }),
            });
        }
        let first_bind = bind(0);
        let body = acc.expect("a chained comparison has at least one operator");
        Ok(quote!({ #first_bind #body }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_eq() {
        let options = PythonOptions::default();
        let result = crate::parse("1 == 2", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //info!("{}", result);

        let code = result.to_rust(
            CodeGenContext::Module("test_case".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        tracing::info!("module: {:?}", code);
    }

    #[test]
    fn test_complex_compare() {
        let options = PythonOptions::default();
        let result = crate::parse("1 < a > 6", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //info!("{}", result);

        let code = result.to_rust(
            CodeGenContext::Module("test_case".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        tracing::info!("module: {:?}", code);
    }
}
