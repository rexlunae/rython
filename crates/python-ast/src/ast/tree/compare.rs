use proc_macro2::TokenStream;
use pyo3::{Borrowed, Bound, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    dump, extraction_failure, err_from, CodeGen, CodeGenContext, CompareNotYetImplemented, ExprType,
    PythonOptions, SymbolTableScopes,
};

/// The `is None` read for an operand (issue #189): a class-instance module
/// global's VALUE reads render the unwrapped instance (name.rs), so a
/// None-ness test must read the Option the static actually holds. Returns
/// the Option-read tokens when the operand is such a global, else None.
fn class_global_none_check(operand: &ExprType, options: &PythonOptions) -> Option<TokenStream> {
    if let ExprType::Name(n) = operand
        && matches!(
            options.mutable_statics.get(&n.id),
            Some(crate::MutableGlobalKind::Class { .. })
        )
    {
        let ident = crate::safe_ident(&n.id);
        return Some(quote!(stdpython::py_global_read(&#ident)));
    }
    None
}

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

        let left = ob.getattr("left").map_err(|e| extraction_failure("left", &ob, e))?;

        let comparators = ob.getattr("comparators").map_err(|e| extraction_failure("comparators", &ob, e))?;
        tracing::debug!(
            "left: {}, comparators: {}",
            dump(&left, None)?,
            dump(&comparators, None)?
        );

        let left = left.extract().map_err(|e| extraction_failure("getting binary operator operand", &ob, e))?;
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
                    let operand_tokens = match class_global_none_check(operand, &options) {
                        Some(ts) => ts,
                        None => operand
                            .clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())?,
                    };
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
            let comparator = comparator_ast
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            // A GENERIC (inferred) parameter compares with an integer
            // literal converted to the parameter's own type via
            // stdpython's PyFromInt (`B::py_from_int(0)`): Rust std has no
            // int/float cross-PartialOrd, so the bounds
            // `B: PyLe<B> + PyFromInt` are satisfied by both i64 and f64
            // (Python promotes `2.5 <= 0` to a float comparison).
            let mut comparator = if let ExprType::Name(n) = left_ast {
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
            // Python promotes an INT operand to FLOAT in a numeric
            // comparison (`read_timeout == 0` where read_timeout is
            // `float | None` — urllib3's _make_request, whose Option-match
            // and plain py_eq paths compare the f64 with an i64 literal):
            // Rust std has no int/float cross-PartialEq, so the literal
            // renders as the float — the same `as f64` the coercion
            // machinery accepts for numeric contexts (lossy above 2^53,
            // where Python's comparison is exact; accepted because the
            // alternative is a rustc error, round 87). Only when the LEFT
            // side is a Float (or an Option whose inner is Float) — an
            // int-typed side keeps the int comparison.
            {
                let left_ty = crate::infer_type(Some(&ctx), left_ast, &options, &symbols);
                let float_side = match &left_ty {
                    crate::TypeInfo::Float => true,
                    crate::TypeInfo::Option(inner) => {
                        matches!(**inner, crate::TypeInfo::Float)
                    }
                    _ => false,
                };
                if float_side
                    && matches!(
                        crate::infer_type(Some(&ctx), comparator_ast, &options, &symbols),
                        crate::TypeInfo::Int
                    )
                {
                    comparator = quote!((#comparator) as f64);
                }
            }
            // An OPTION-typed comparator unwraps to the inner value
            // (`len(...) < amt` where amt is `int | None` — urllib3's
            // _read; `amt < self.chunk_left` where BOTH sides are
            // `int | None` — _handle_chunk): Python compares the inner
            // when non-None, and an ordered compare on None is CPython's
            // TypeError — the loud §12.2 panic. Applies to the py_* six
            // ops whether or not the LHS is itself an Option (round 92 —
            // the unwrap previously lived ONLY inside the LHS-Option
            // branch, so a plain LHS with an Option comparator compared
            // the raw Option — E0277 on the PyLt<Option<i64>> bound).
            {
                let is_py_cmp = matches!(
                    op,
                    Compares::Eq
                        | Compares::NotEq
                        | Compares::Lt
                        | Compares::LtE
                        | Compares::Gt
                        | Compares::GtE
                );
                let comparator_is_option = !matches!(
                    comparator_ast,
                    crate::ExprType::Name(n)
                        if options.narrowed_names.contains_key(&n.id)
                ) && (matches!(
                    crate::infer_type(Some(&ctx), comparator_ast, &options, &symbols),
                    crate::TypeInfo::Option(_)
                ) || matches!(
                    comparator_ast,
                    crate::ExprType::Attribute(attr)
                        if matches!(attr.value.as_ref(), crate::ExprType::Name(n) if n.id == "self")
                            && crate::ast::tree::aug_assign::self_field_rust_ty(
                                &attr.attr, &ctx, &options, &symbols,
                            )
                            .is_some_and(|t| matches!(t, crate::TypeInfo::Option(_)))
                ));
                if is_py_cmp && comparator_is_option {
                    let op_name = match op {
                        Compares::Eq => "==",
                        Compares::NotEq => "!=",
                        Compares::Lt => "<",
                        Compares::LtE => "<=",
                        Compares::Gt => ">",
                        Compares::GtE => ">=",
                        _ => "?",
                    };
                    // CPython names the LHS's type (`5 < None` → "'<' not
                    // supported between instances of 'int' and
                    // 'NoneType'").
                    let lhs_ty = match crate::infer_type(Some(&ctx), left_ast, &options, &symbols) {
                        crate::TypeInfo::Int => "int",
                        crate::TypeInfo::Float => "float",
                        crate::TypeInfo::String | crate::TypeInfo::StrRef => "str",
                        crate::TypeInfo::Bytes => "bytes",
                        _ => "NoneType",
                    };
                    let msg = format!(
                        "'{}' not supported between instances of '{}' and 'NoneType'",
                        op_name, lhs_ty
                    );
                    comparator = quote! {
                        match (#comparator).clone() {
                            Some(__rython_r) => __rython_r,
                            None => panic!(#msg),
                        }
                    };
                }
            }
            // Comparisons route through the stdpython PyEq/PyNe/PyLt/PyLe/
            // PyGt/PyGe traits (in scope via `use stdpython::*`): scalars
            // and containers get their existing PartialEq/PartialOrd
            // behaviour (bool result) through blanket impls, while NdArray
            // overrides them to broadcast elementwise and return an array —
            // the same pattern `+` uses with PyAdd.
            // An OPTION-typed comparator (`key in context` where context is
            // `dict[str, Any] | None` — urllib3's poolmanager): the
            // membership READ unwraps the Option with a loud §12.2 panic
            // (CPython's TypeError on a None comparator), mirroring the
            // call path's receiver_option_inner.
            let membership_receiver = || {
                // A NARROWED comparator (`if character_range is None:
                // continue` then `keyword in character_range` — charset's
                // utils) already reads the inner value; unwrapping it
                // again breaks on the String (round 77).
                let narrowed = matches!(comparator_ast, ExprType::Name(n)
                    if options.narrowed_names.contains_key(&n.id));
                if !narrowed
                    && crate::ast::tree::attribute::receiver_option_inner(
                        comparator_ast, &ctx, &symbols, &options,
                    )
                    .is_some()
                {
                    quote!((#comparator).clone().unwrap_or_else(|| {
                        panic!("TypeError: argument of type 'NoneType' is not iterable")
                    }))
                } else {
                    comparator.clone()
                }
            };
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
                    // A user-class comparator that defines `__contains__`
                    // routes the membership test to ITS method — Python's
                    // behavior (the class's own key semantics; §7's
                    // mapping-protocol slice). The method must exist;
                    // anything else keeps py_contains, loud in rustc for
                    // classes (§12.1).
                    if let Some((class, class_symbols)) =
                        crate::receiver_class(&comparator_ast, &ctx, &symbols, &options)
                        && let Some(method) =
                            class
                                .method_on_mro("__contains__", &class_symbols)
                                .filter(|m| crate::ast::tree::call::dunder_method_well_typed(m))
                    {
                        crate::ast::tree::call::dunder_method_call(
                            &method,
                            &comparator,
                            std::slice::from_ref(left_ast),
                            true,
                            &ctx,
                            &options,
                            &symbols,
                        )?
                    } else if matches!(
                        comparator_ast,
                        ExprType::Name(n)
                            if (matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Dict(k, _))
                                    if matches!(**k, crate::TypeInfo::String)
                            ) || matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Option(inner))
                                    if matches!(
                                        &**inner,
                                        crate::TypeInfo::Dict(k, _)
                                            if matches!(**k, crate::TypeInfo::String)
                                    )
                            ))
                    ) {
                        let left = crate::render_typed(
                            left_ast,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        let recv = membership_receiver();
                        quote!((#recv).py_contains(&(#left)))
                    } else {
                        let recv = membership_receiver();
                        quote!((#recv).py_contains(&(#left)))
                    }
                }
                Compares::NotIn => {
                    // The __contains__ twin of the In arm above.
                    if let Some((class, class_symbols)) =
                        crate::receiver_class(&comparator_ast, &ctx, &symbols, &options)
                        && let Some(method) =
                            class
                                .method_on_mro("__contains__", &class_symbols)
                                .filter(|m| crate::ast::tree::call::dunder_method_well_typed(m))
                    {
                        let inner = crate::ast::tree::call::dunder_method_call(
                            &method,
                            &comparator,
                            std::slice::from_ref(left_ast),
                            true,
                            &ctx,
                            &options,
                            &symbols,
                        )?;
                        quote!(!#inner)
                    } else if matches!(
                        comparator_ast,
                        ExprType::Name(n)
                            if (matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Dict(k, _))
                                    if matches!(**k, crate::TypeInfo::String)
                            ) || matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Option(inner))
                                    if matches!(
                                        &**inner,
                                        crate::TypeInfo::Dict(k, _)
                                            if matches!(**k, crate::TypeInfo::String)
                                    )
                            ))
                    ) {
                        let left = crate::render_typed(
                            left_ast,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        let recv = membership_receiver();
                        quote!(!(#recv).py_contains(&(#left)))
                    } else {
                        let recv = membership_receiver();
                        quote!(!(#recv).py_contains(&(#left)))
                    }
                }

                _ => return Err(err_from(CompareNotYetImplemented(self)).into()),
            };
            // An OPTION-typed LHS (`amt != 0` where amt is `int | None` —
            // urllib3's _read_next_chunk, guarded by `amt is not None`):
            // the runtime PyEq/PyNe/PyLt blanket impls need
            // `Option<i64>: PartialEq<i64>`, which does not exist (Option
            // only compares with Option). Python compares the INNER value
            // when non-None; a None LHS answers Python's equality
            // semantics (`None == x` is False, `None != x` is True — the
            // values are merely unequal) while an ORDERED compare on None
            // is CPython's TypeError — a loud §12.2 panic with the
            // message. The `is None` guard in real code makes the panic
            // unreachable; the equality answers are always reachable. The
            // py_* six ops are wrapped; `is`/`is not`/`in` keep their own
            // None-aware lowerings.
            // The LHS Option-ness: infer_type for a plain name, or the
            // FIELD TABLE for a `self.<field>` accessor (`self.length_remaining != 0`,
            // `self.chunk_left == 0` — urllib3's _handle_chunk, where the
            // fields are `int | None` and infer_type cannot see through
            // self-fields — round 89).
            let opt_inner = match crate::infer_type(Some(&ctx), left_ast, &options, &symbols) {
                crate::TypeInfo::Option(inner) => Some(*inner),
                _ => None,
            };
            let opt_inner = opt_inner.or_else(|| {
                if let ExprType::Attribute(attr) = left_ast {
                    // The field's OWNER: the enclosing class's chain for a
                    // `self.<field>` read (round 89), or the BASE structs
                    // for a `self.__rython_base.<field>` chain (a base-class
                    // field read through the embedded struct — urllib3's
                    // `self.__rython_base._tunnel_scheme == "https"`,
                    // round 91). Walk the receiver chain to confirm it
                    // roots at `self`, then look the field up in every
                    // class of the chain.
                    let mut cur = attr.value.as_ref();
                    loop {
                        match cur {
                            crate::ExprType::Name(n) if n.id == "self" => break,
                            crate::ExprType::Attribute(a) => cur = a.value.as_ref(),
                            _ => return None,
                        }
                    }
                    let class_name = ctx.enclosing_class_name()?;
                    let Some(crate::SymbolTableNode::ClassDef(class)) = symbols.get(class_name)
                    else {
                        return None;
                    };
                    class
                        .base_chain_with_options(&symbols, &options)
                        .iter()
                        .find_map(|c| {
                            c.infer_fields(&symbols, &options).ok().and_then(|fields| {
                                fields
                                    .iter()
                                    .find(|(n, _)| *n == attr.attr)
                                    .map(|(_, t)| t.clone())
                            })
                        })
                        .and_then(|t| match t {
                            crate::TypeInfo::Option(inner) => Some(*inner),
                            _ => None,
                        })
                } else {
                    None
                }
            });
            let tokens = if let Some(inner) = opt_inner
            {
                // A NARROWED LHS (round 81's `and`-chain narrowing:
                // `if conn and is_connection_dropped(conn)` proves conn
                // non-None; `amt and amt > c_int_max` — urllib3's
                // _read_next_chunk) already reads the INNER value
                // (`(amt).clone().unwrap()` via narrowed_names); wrapping
                // it in the Option-match again double-unwraps (E0308 on
                // the i64 scrutinee). The read-side unwrap and the
                // compare-side match must not both fire — the narrowed
                // read is authoritative.
                let lhs_narrowed = matches!(left_ast, ExprType::Name(n)
                    if options.narrowed_names.contains_key(&n.id));
                if lhs_narrowed {
                    tokens
                } else {
                    let inner_ty = inner.clone();
                    let is_py_cmp = matches!(
                        op,
                        Compares::Eq
                            | Compares::NotEq
                            | Compares::Lt
                            | Compares::LtE
                            | Compares::Gt
                            | Compares::GtE
                    );
                    if is_py_cmp {
                    // An OPTION-typed comparator (`amt < self.chunk_left`
                    // where BOTH are `int | None` — urllib3's
                    // _handle_chunk): unwrap it the same way, with the
                    // loud ordered-compare panic (both sides are guarded
                    // `is not None`).
                    fn type_name(t: &crate::TypeInfo) -> &'static str {
                        match t {
                            crate::TypeInfo::Int => "int",
                            crate::TypeInfo::Float => "float",
                            crate::TypeInfo::String | crate::TypeInfo::StrRef => "str",
                            crate::TypeInfo::Bytes => "bytes",
                            _ => "NoneType",
                        }
                    }
                    let inner_cmp = match op {
                        Compares::Eq => quote!((__rython_v).py_eq(&(#comparator))),
                        Compares::NotEq => quote!((__rython_v).py_ne(&(#comparator))),
                        Compares::Lt => quote!((__rython_v).py_lt(&(#comparator))),
                        Compares::LtE => quote!((__rython_v).py_le(&(#comparator))),
                        Compares::Gt => quote!((__rython_v).py_gt(&(#comparator))),
                        Compares::GtE => quote!((__rython_v).py_ge(&(#comparator))),
                        _ => unreachable!(),
                    };
                    // Equality with None is not an error (Python answers
                    // False/True); ordered comparison of None is a
                    // TypeError.
                    let none_arm = match op {
                        Compares::Eq => quote!(false),
                        Compares::NotEq => quote!(true),
                        _ => {
                            let op_name = match op {
                                Compares::Lt => "<",
                                Compares::LtE => "<=",
                                Compares::Gt => ">",
                                Compares::GtE => ">=",
                                _ => "?",
                            };
                            let ty_name = type_name(&inner_ty);
                            let msg = format!(
                                "'{}' not supported between instances of 'NoneType' and '{}'",
                                op_name, ty_name
                            );
                            quote!(panic!(#msg))
                        }
                    };
                    quote! {
                        match (#left).clone() {
                            Some(__rython_v) => #inner_cmp,
                            None => #none_arm,
                        }
                    }
                    } else {
                        tokens
                    }
                }
            } else {
                tokens
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
                    // Issue #189: the None-tested operand may be a
                    // class-instance global — its temporary was rendered as
                    // the unwrapped instance, so test the Option directly.
                    let operand_ast = if is_none[i + 1] {
                        operands[i]
                    } else {
                        operands[i + 1]
                    };
                    if let Some(ts) = class_global_none_check(operand_ast, &options) {
                        return Ok(match op {
                            Compares::Is => quote!((#ts).py_is_none()),
                            _ => quote!(!(#ts).py_is_none()),
                        });
                    }
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
