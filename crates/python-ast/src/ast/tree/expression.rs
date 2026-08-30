use proc_macro2::TokenStream;
use pyo3::{Borrowed, Bound, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    dump, err_from, extraction_failure, Attribute, Await, BinOp, BoolOp, Call, CodeGen, CodeGenContext, Compare,
    Constant, Dict, DictComp, ExprTypeNotYetImplemented, FormattedValue, GeneratorExp, IfExp,
    JoinedStr, Lambda, ListComp, Name, NamedExpr, Node, PythonOptions, Set, SetComp, Starred,
    Subscript, SymbolTableScopes, Tuple, UnaryOp, Yield, YieldFrom,
};

/// Mostly this shouldn't be used, but it exists so that we don't have to manually implement FromPyObject on all of ExprType
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[repr(transparent)]
pub struct Container<T>(pub T);

impl<'a, 'py> FromPyObject<'a, 'py> for Container<crate::pytypes::List<ExprType>> {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let list = crate::pytypes::List::<ExprType>::new();

        tracing::debug!("pylist: {}", dump(&ob, Some(4))?);
        let _converted_list: Vec<Bound<PyAny>> = ob.extract()?;
        for item in _converted_list.iter() {
            tracing::debug!("item: {:?}", item);
        }

        Ok(Self(list))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum ExprType {
    BoolOp(BoolOp),
    NamedExpr(NamedExpr),
    BinOp(BinOp),
    UnaryOp(UnaryOp),
    Lambda(Lambda),
    IfExp(IfExp),
    Dict(Dict),
    Set(Set),
    ListComp(ListComp),
    DictComp(DictComp),
    SetComp(SetComp),
    GeneratorExp(GeneratorExp),
    Await(Await),
    Yield(Yield),
    YieldFrom(YieldFrom),
    Compare(Compare),
    Call(Call),
    FormattedValue(FormattedValue),
    JoinedStr(JoinedStr),
    Constant(Constant),

    /// These can appear in a few places, such as the left side of an assignment.
    Attribute(Attribute),
    Subscript(Subscript),
    Starred(Starred),
    Name(Name),
    List(Vec<ExprType>),
    Tuple(Tuple),
    /*Slice(),*/
    NoneType(Constant),

    Unimplemented(String),
    #[default]
    Unknown,
}

impl<'a, 'py> FromPyObject<'a, 'py> for ExprType {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        tracing::debug!("exprtype ob: {}", dump(&ob, Some(4))?);

        let expr_type = ob
            .get_type()
            .name()
            .map_err(|e| extraction_failure("expression type name", &ob, e))?;
        tracing::debug!("expression type: {}, value: {}", expr_type, dump(&ob, None)?);

        let r = match expr_type.extract::<String>()?.as_str() {
            "Attribute" => {
                let a = ob.extract().map_err(|e| extraction_failure("extracting Attribute in expression", &ob, e))?;
                Ok(Self::Attribute(a))
            }
            "Await" => {
                //println!("await: {}", dump(&ob, None)?);
                let a = ob.extract().map_err(|e| extraction_failure("extracting await value in expression", &ob, e))?;
                Ok(Self::Await(a))
            }
            "BoolOp" => {
                let b = ob.extract().map_err(|e| extraction_failure("extracting BoolOp in expression", &ob, e))?;
                Ok(Self::BoolOp(b))
            }
            "Call" => {
                let et = ob.extract().map_err(|e| extraction_failure("parsing Call expression", &ob, e))?;
                Ok(Self::Call(et))
            }
            "Compare" => {
                let c = ob.extract().map_err(|e| extraction_failure("extracting Compare in expression", &ob, e))?;
                Ok(Self::Compare(c))
            }
            "Constant" => {
                tracing::debug!("constant: {}", dump(&ob, None)?);
                let c = ob.extract().map_err(|e| extraction_failure("extracting Constant in expression", &ob, e))?;
                Ok(Self::Constant(c))
            }
            "List" => {
                // Extract the list elements using the 'elts' attribute
                let elts_attr = ob
                    .getattr("elts")
                    .map_err(|e| extraction_failure("list elements", &ob, e))?;
                let elts_vec: Vec<Bound<PyAny>> = elts_attr
                    .extract()
                    .map_err(|e| extraction_failure("list elements", &ob, e))?;

                // Convert each element to ExprType
                let mut expr_list = Vec::new();
                for elt in elts_vec {
                    let expr: ExprType = elt
                        .extract()
                        .map_err(|e| extraction_failure("list element", &elt, e))?;
                    expr_list.push(expr);
                }
                
                Ok(Self::List(expr_list))
            }
            "ListComp" => {
                let lc = ob.extract().map_err(|e| extraction_failure("extracting ListComp in expression", &ob, e))?;
                Ok(Self::ListComp(lc))
            }
            "DictComp" => {
                let dc = ob.extract().map_err(|e| extraction_failure("extracting DictComp in expression", &ob, e))?;
                Ok(Self::DictComp(dc))
            }
            "SetComp" => {
                let sc = ob.extract().map_err(|e| extraction_failure("extracting SetComp in expression", &ob, e))?;
                Ok(Self::SetComp(sc))
            }
            "GeneratorExp" => {
                let ge = ob.extract().map_err(|e| extraction_failure("extracting GeneratorExp in expression", &ob, e))?;
                Ok(Self::GeneratorExp(ge))
            }
            "Name" => {
                let name = ob.extract().map_err(|e| extraction_failure("parsing Name expression", &ob, e))?;
                Ok(Self::Name(name))
            }
            // The walrus operator (`if (x := f()) is not None:`): a
            // NamedExpr assigns its target and evaluates to it.
            "NamedExpr" => {
                let ne = ob.extract().map_err(|e| extraction_failure("extracting NamedExpr in expression", &ob, e))?;
                Ok(Self::NamedExpr(ne))
            }
            "UnaryOp" => {
                let c = ob.extract().map_err(|e| extraction_failure("extracting UnaryOp in expression", &ob, e))?;
                Ok(Self::UnaryOp(c))
            }
            "BinOp" => {
                let c = ob.extract().map_err(|e| extraction_failure("extracting BinOp in expression", &ob, e))?;
                Ok(Self::BinOp(c))
            }
            "Lambda" => {
                let l = ob.extract().map_err(|e| extraction_failure("extracting Lambda in expression", &ob, e))?;
                Ok(Self::Lambda(l))
            }
            "IfExp" => {
                let i = ob.extract().map_err(|e| extraction_failure("extracting IfExp in expression", &ob, e))?;
                Ok(Self::IfExp(i))
            }
            "Dict" => {
                let d = ob.extract().map_err(|e| extraction_failure("extracting Dict in expression", &ob, e))?;
                Ok(Self::Dict(d))
            }
            "Set" => {
                let s = ob.extract().map_err(|e| extraction_failure("extracting Set in expression", &ob, e))?;
                Ok(Self::Set(s))
            }
            "Tuple" => {
                let t = ob.extract().map_err(|e| extraction_failure("extracting Tuple in expression", &ob, e))?;
                Ok(Self::Tuple(t))
            }
            "Subscript" => {
                let s = ob.extract().map_err(|e| extraction_failure("extracting Subscript in expression", &ob, e))?;
                Ok(Self::Subscript(s))
            }
            "Starred" => {
                let s = ob.extract().map_err(|e| extraction_failure("extracting Starred in expression", &ob, e))?;
                Ok(Self::Starred(s))
            }
            "Yield" => {
                let y = ob.extract().map_err(|e| extraction_failure("extracting Yield in expression", &ob, e))?;
                Ok(Self::Yield(y))
            }
            "YieldFrom" => {
                let yf = ob.extract().map_err(|e| extraction_failure("extracting YieldFrom in expression", &ob, e))?;
                Ok(Self::YieldFrom(yf))
            }
            "JoinedStr" => {
                let js = ob.extract().map_err(|e| extraction_failure("extracting JoinedStr in expression", &ob, e))?;
                Ok(Self::JoinedStr(js))
            }
            "FormattedValue" => {
                let fv = ob.extract().map_err(|e| extraction_failure("extracting FormattedValue in expression", &ob, e))?;
                Ok(Self::FormattedValue(fv))
            }
            _ => {
                let err_msg = format!(
                    "Unimplemented expression type {}, {}",
                    expr_type,
                    dump(&ob, None)?
                );
                Err(pyo3::exceptions::PyValueError::new_err(
                    ob.error_message("<unknown>", err_msg.as_str()),
                ))
            }
        };
        r
    }
}

impl<'a> CodeGen for ExprType {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        thread_local! {
            static E_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }
        let d = E_DEPTH.with(|c| c.get());
        if d > 100 && d % 20 == 0 {
        }
        E_DEPTH.with(|c| c.set(d + 1));
        let result = self.to_rust_inner(ctx, options, symbols);
        E_DEPTH.with(|c| c.set(d));
        return result;
    }
}

impl ExprType {
    fn to_rust_inner(
        self,
        ctx: CodeGenContext,
        options: PythonOptions,
        symbols: SymbolTableScopes,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        match self {
            ExprType::Attribute(attribute) => attribute.to_rust(ctx, options, symbols),
            ExprType::Await(func) => func.to_rust(ctx, options, symbols),
            ExprType::BinOp(binop) => binop.to_rust(ctx, options, symbols),
            ExprType::BoolOp(boolop) => boolop.to_rust(ctx, options, symbols),
            ExprType::Call(call) => call.to_rust(ctx, options, symbols),
            ExprType::Compare(c) => c.to_rust(ctx, options, symbols),
            ExprType::Constant(c) => c.to_rust(ctx, options, symbols),
            ExprType::Lambda(l) => l.to_rust(ctx, options, symbols),
            ExprType::IfExp(i) => i.to_rust(ctx, options, symbols),
            ExprType::Dict(d) => d.to_rust(ctx, options, symbols),
            ExprType::Set(s) => s.to_rust(ctx, options, symbols),
            ExprType::ListComp(lc) => lc.to_rust(ctx, options, symbols),
            ExprType::DictComp(dc) => dc.to_rust(ctx, options, symbols),
            ExprType::SetComp(sc) => sc.to_rust(ctx, options, symbols),
            ExprType::GeneratorExp(ge) => ge.to_rust(ctx, options, symbols),
            ExprType::Tuple(t) => t.to_rust(ctx, options, symbols),
            ExprType::Subscript(s) => s.to_rust(ctx, options, symbols),
            ExprType::Starred(s) => s.to_rust(ctx, options, symbols),
            ExprType::Yield(y) => y.to_rust(ctx, options, symbols),
            ExprType::YieldFrom(yf) => yf.to_rust(ctx, options, symbols),
            ExprType::JoinedStr(js) => js.to_rust(ctx, options, symbols),
            ExprType::FormattedValue(fv) => fv.to_rust(ctx, options, symbols),
            ExprType::NamedExpr(ne) => ne.to_rust(ctx, options, symbols),
            ExprType::List(l) => {
                // Type-aware list lowering: infer the element type across
                // the literal, then coerce each element to it —
                // `[1, 2.0]` → `Vec<f64>` with `1 as f64`, `["a", s]` →
                // `Vec<String>` with `"a".to_string()`. Incompatible kinds
                // ([1, "a"]) are a loud conversion-time error rather than
                // a cryptic rustc mismatch inside generated code.
                // A FORCED element type (a `-> List[Union[...]]` return
                // whose element boxes — idna's `_seg_N` tables, round 57)
                // overrides the inference: every fixed element boxes,
                // and a Starred element SPREADS its collection exactly
                // like the normal path (Devin review on #263: the first
                // version emitted the spread as one list element).
                if let Some(forced) = &*options.forced_list_elt {
                    let expected = Some(forced.clone());
                    let mut elements = Vec::new();
                    let mut spreads: Vec<TokenStream> = Vec::new();
                    for li in &l {
                        if let ExprType::Starred(starred) = li {
                            let inner = crate::render_reused(
                                &starred.value,
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            spreads.push(quote!(#inner));
                            continue;
                        }
                        elements.push(crate::render_typed(
                            li,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            expected.clone(),
                        )?);
                    }
                    if spreads.is_empty() {
                        return Ok(quote!(vec![#(#elements),*]));
                    }
                    // Source-order interleave of fixed and spread
                    // segments (`[*a, x, *b]` extends a, pushes x, then
                    // extends b — the same shape the non-forced starred
                    // path emits below).
                    let mut segments: Vec<TokenStream> = Vec::new();
                    let mut si = 0usize;
                    let mut ei = 0usize;
                    for li in &l {
                        if let ExprType::Starred(_) = li {
                            let s = spreads[si].clone();
                            segments.push(quote!(__rython_list.extend(#s);));
                            si += 1;
                        } else {
                            let e = elements[ei].clone();
                            segments.push(quote!(__rython_list.push(#e);));
                            ei += 1;
                        }
                    }
                    return Ok(quote!({
                        let mut __rython_list = Vec::new();
                        #(#segments)*
                        __rython_list
                    }));
                }
                let mut has_starred = false;
                let mut elt_types: Vec<crate::TypeInfo> = Vec::new();
                for li in &l {
                    if matches!(li, ExprType::Starred(_)) {
                        // Starred elements spread their collection's ELEMENT
                        // type, not the collection's type: counting them here
                        // makes `[*xs, 1]` look like a (list, int) mix and
                        // reject it before the (accurate) starred-unpacking
                        // error can surface (Devin review on #103).
                        has_starred = true;
                        continue;
                    }
                    let t = crate::infer_type(&li, &options, &symbols);
                    if !matches!(t, crate::TypeInfo::PyObject) {
                        elt_types.push(t);
                    }
                }
                let mut expected = crate::TypeInfo::PyObject;
                let mut distinct: Vec<crate::TypeInfo> = Vec::new();
                for t in &elt_types {
                    if !distinct.contains(t) {
                        distinct.push(t.clone());
                    }
                }
                // Unify the DISTINCT types (a repeat of an early element at
                // the END of the literal must not re-absorb the result:
                // `[(0, "3"), (65, "M", "a"), (76, "V")]` — idna's
                // _seg tables mix 2- and 3-tuples — folding the raw
                // element list ends on a 2-tuple and `unify(PyObject,
                // Tuple2)` snaps expected back to Tuple2, hiding the
                // heterogeneity from the boxable-union check below).
                for t in &distinct {
                    expected = crate::unify(expected, t.clone());
                }
                if distinct.len() > 1 && matches!(expected, crate::TypeInfo::PyObject) {
                    // A list of DIFFERENT class instances (`[d_sp, d_ta,
                    // ...]` — charset_normalizer's debug plugin list) has
                    // no single Rust element type. That is a documented
                    // divergence (heterogeneous class lists cannot build in
                    // rython; rustc reports the Vec element mismatch), but
                    // the conversion itself proceeds — primitive mixes
                    // ([1, "a"]) stay a loud error.
                    if !distinct.iter().all(|t| matches!(t, crate::TypeInfo::Class(_))) {
                        // A HETEROGENEOUS list involving a TUPLE
                        // (`['s3_use_arn_region', ('s3',
                        // 'use_arn_region')]` — botocore's
                        // configprovider): a structured config list — box
                        // the elements as PyValue (documented divergence).
                        // A list mixing an OPTIONAL element (`["--username",
                        // username]` where username is `str | None` — pip's
                        // subversion) boxes the same way. Primitive mixes
                        // without tuples/optionals ([1, 'a']) stay a loud
                        // error.
                        // Issue #130: ANY mix whose element types are all
                        // boxable (`[1, "a"]`, `[None, "x", 2, b"y"]`, ...)
                        // boxes to Vec<PyValue> - not just the >=3-kind and
                        // tuple/optional shapes this branch used to cover.
                        if distinct.iter().all(crate::is_boxable_value_type) {
                            expected = crate::TypeInfo::PyValue;
                        } else {
                            let kinds = distinct
                                .iter()
                                .map(|d| d.display())
                                .collect::<Vec<_>>()
                                .join(", ");
                            return Err(format!(
                                "list literal mixes incompatible element types ({kinds}); \
                                 elements must share a common type (or annotate the \
                                 variable, e.g. `xs: list[float] = [...]`)"
                            )
                            .into());
                        }
                    }
                }
                let expected_elt = if matches!(expected, crate::TypeInfo::PyObject) {
                    None
                } else {
                    Some(expected)
                };

                let mut elements = Vec::new();
                let mut starred_vals: Vec<TokenStream> = Vec::new();
                for li in &l {
                    if let ExprType::Starred(starred) = li {
                        // `[*xs]` — the spread's collection extends the
                        // list (`[key, *val]` — urllib3). The spread reads
                        // the collection WITHOUT consuming it (Python's
                        // `[*xs, a]` leaves `xs` usable), so the reuse-clone
                        // rule applies like any other name read.
                        let inner = crate::render_reused(
                            &starred.value,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?;
                        starred_vals.push(inner);
                        continue;
                    }
                    let code = crate::render_typed(
                        &li,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                        expected_elt.clone(),
                    )?;
                    elements.push(code);
                }
                
                // If we have starred expressions, handle them specially
                if has_starred {
                    // Emit elements in SOURCE ORDER: a spread before or
                    // between fixed elements must interleave (`[*xs, a]` is
                    // `xs` then `a`, not `a` then `xs` — the old lowering
                    // pushed all fixed elements first and extended with the
                    // spreads after, silently reordering).
                    enum Seg {
                        Fixed(proc_macro2::TokenStream),
                        Spread(proc_macro2::TokenStream),
                    }
                    // `elements` holds the fixed renders in source order;
                    // merge them back with the spreads by walking the
                    // literal once.
                    let mut segments: Vec<Seg> = Vec::new();
                    let mut si = 0usize;
                    let mut ei = 0usize;
                    for li in &l {
                        if let ExprType::Starred(_) = li {
                            segments.push(Seg::Spread(starred_vals[si].clone()));
                            si += 1;
                        } else {
                            segments.push(Seg::Fixed(elements[ei].clone()));
                            ei += 1;
                        }
                    }
                    let elt_ty = expected_elt
                        .as_ref()
                        .map(|t| t.to_rust_type())
                        .unwrap_or_else(|| quote!(_));
                    let stmts = segments.iter().map(|seg| match seg {
                        Seg::Fixed(t) => quote!(__rython_list.push(#t);),
                        Seg::Spread(t) => quote!(__rython_list.extend(#t);),
                    });
                    Ok(quote! {
                        {
                            let mut __rython_list: Vec<#elt_ty> = Vec::new();
                            #(#stmts)*
                            __rython_list
                        }
                    })
                } else {
                    // Elements keep their own types: [1, 2, 3] must become a
                    // Vec<i64>, not a Vec<String>.
                    Ok(quote! {
                        vec![#(#elements),*]
                    })
                }
            }
            ExprType::Name(name) => name.to_rust(ctx, options, symbols),
            // Python's None is Rust's Option::None: `x = None` initializes
            // an Option, `f(None)` passes one, `d.get(k)` results compare
            // against it.
            ExprType::NoneType(_) => Ok(quote!(None)),
            ExprType::UnaryOp(operand) => operand.to_rust(ctx, options, symbols),

            _ => {
                let error = err_from(ExprTypeNotYetImplemented(self));
                Err(error.into())
            }
        }
    }
}

/// An Expr only contains a single value key, which leads to the actual expression,
/// which is one of several types.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Expr {
    pub value: ExprType,
    pub ctx: Option<String>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Expr {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let err_msg = format!("extracting object value {} in expression", dump(&ob, None)?);

        let ob_value = ob
            .getattr("value")
            .map_err(|e| extraction_failure("expression value", &ob, format!("{}: {}", err_msg, e)))?;
        tracing::debug!("ob_value: {}", dump(&ob_value, None)?);

        // The context is Load, Store, etc. For some types of expressions such as Constants, it does not exist.
        let ctx: Option<String> = if let Ok(pyany) = ob_value.getattr("ctx") {
            pyany.get_type().extract().unwrap_or_default()
        } else {
            None
        };

        let mut r = Self {
            value: ExprType::Unknown,
            ctx: ctx,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        };

        let expr_type = ob_value
            .get_type()
            .name()
            .map_err(|e| extraction_failure("expression type name", &ob, e))?;
        tracing::debug!(
            "expression type: {}, value: {}",
            expr_type,
            dump(&ob_value, None)?
        );
        match expr_type.extract::<String>()?.as_str() {
            "Attribute" => {
                let a = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Attribute expression", &ob_value, e))?;
                r.value = ExprType::Attribute(a);
                Ok(r)
            }
            "Await" => {
                let a = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Await expression", &ob_value, e))?;
                r.value = ExprType::Await(a);
                Ok(r)
            }
            "BinOp" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("BinOp expression", &ob_value, e))?;
                r.value = ExprType::BinOp(c);
                Ok(r)
            }
            "BoolOp" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("BoolOp expression", &ob_value, e))?;
                r.value = ExprType::BoolOp(c);
                Ok(r)
            }
            "Call" => {
                let et = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Call expression", &ob_value, e))?;
                r.value = ExprType::Call(et);
                Ok(r)
            }
            "Constant" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Constant expression", &ob_value, e))?;
                r.value = ExprType::Constant(c);
                Ok(r)
            }
            "Compare" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Compare expression", &ob_value, e))?;
                r.value = ExprType::Compare(c);
                Ok(r)
            }
            "List" => {
                // Extract the list elements using the 'elts' attribute
                let elts_attr = ob_value
                    .getattr("elts")
                    .map_err(|e| extraction_failure("list elements", &ob_value, e))?;
                let elts_vec: Vec<Bound<PyAny>> = elts_attr
                    .extract()
                    .map_err(|e| extraction_failure("list elements", &ob_value, e))?;

                // Convert each element to ExprType
                let mut expr_list = Vec::new();
                for elt in elts_vec {
                    let expr: ExprType = elt
                        .extract()
                        .map_err(|e| extraction_failure("list element", &elt, e))?;
                    expr_list.push(expr);
                }
                
                r.value = ExprType::List(expr_list);
                Ok(r)
            }
            "Name" => {
                let name = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Name expression", &ob_value, e))?;
                r.value = ExprType::Name(name);
                Ok(r)
            }
            "UnaryOp" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("UnaryOp expression", &ob_value, e))?;
                r.value = ExprType::UnaryOp(c);
                Ok(r)
            }
            "Lambda" => {
                let l = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Lambda expression", &ob_value, e))?;
                r.value = ExprType::Lambda(l);
                Ok(r)
            }
            "IfExp" => {
                let i = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("IfExp expression", &ob_value, e))?;
                r.value = ExprType::IfExp(i);
                Ok(r)
            }
            "Dict" => {
                let d = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Dict expression", &ob_value, e))?;
                r.value = ExprType::Dict(d);
                Ok(r)
            }
            "Set" => {
                let s = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Set expression", &ob_value, e))?;
                r.value = ExprType::Set(s);
                Ok(r)
            }
            "Tuple" => {
                let t = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Tuple expression", &ob_value, e))?;
                r.value = ExprType::Tuple(t);
                Ok(r)
            }
            "Subscript" => {
                let s = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Subscript expression", &ob_value, e))?;
                r.value = ExprType::Subscript(s);
                Ok(r)
            }
            "Yield" => {
                let y = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Yield expression", &ob_value, e))?;
                r.value = ExprType::Yield(y);
                Ok(r)
            }
            "YieldFrom" => {
                let yf = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("YieldFrom expression", &ob_value, e))?;
                r.value = ExprType::YieldFrom(yf);
                Ok(r)
            }
            "JoinedStr" => {
                let js = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("JoinedStr expression", &ob_value, e))?;
                r.value = ExprType::JoinedStr(js);
                Ok(r)
            }
            "FormattedValue" => {
                let fv = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("FormattedValue expression", &ob_value, e))?;
                r.value = ExprType::FormattedValue(fv);
                Ok(r)
            }
            "GeneratorExp" => {
                let ge = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("GeneratorExp expression", &ob_value, e))?;
                r.value = ExprType::GeneratorExp(ge);
                Ok(r)
            }
            // In sitations where an expression is optional, we may see a NoneType expressions.
            "NoneType" => {
                r.value = ExprType::NoneType(Constant(None));
                Ok(r)
            }
            _ => {
                let err_msg = format!(
                    "Unimplemented expression type {}, {}",
                    expr_type,
                    dump(&ob, None)?
                );
                Err(pyo3::exceptions::PyValueError::new_err(
                    ob.error_message("<unknown>", err_msg.as_str()),
                ))
            }
        }
    }
}

impl CodeGen for Expr {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        // Delegate to the (complete) ExprType dispatch rather than keeping a
        // second, drifting copy of the match here. NoneType statements
        // generate no code.
        if matches!(self.value, ExprType::NoneType(_)) {
            return Ok(quote!());
        }
        // A bare `...` statement is a no-op (Python's Ellipsis as a
        // statement — the Protocol-stub idiom `def f(...) -> None: ...`).
        // As a VALUE (assignment, argument, return) Constant::to_rust
        // rejects it loudly.
        if matches!(
            &self.value,
            ExprType::Constant(c)
                if c.0
                    .as_ref()
                    .is_some_and(crate::ast::tree::constant::is_ellipsis_literal)
        ) {
            return Ok(quote!());
        }
        self.value.to_rust(ctx, options, symbols)
    }
}

impl Node for Expr {
    fn lineno(&self) -> Option<usize> {
        self.lineno
    }

    fn col_offset(&self) -> Option<usize> {
        self.col_offset
    }

    fn end_lineno(&self) -> Option<usize> {
        self.end_lineno
    }

    fn end_col_offset(&self) -> Option<usize> {
        self.end_col_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_call_expression() {
        let expression = crate::parse("test()", "test.py").unwrap();
        let mut options = PythonOptions::default();
        options.with_std_python = false;
        let symbols = SymbolTableScopes::new();
        let tokens = expression
            .clone()
            .to_rust(CodeGenContext::Module("test".to_string()), options, symbols)
            .unwrap();
        assert_eq!(
            tokens.to_string(),
            "fn __module_init__ () -> Result < () , PyException > { test () ; Ok (()) } \
             fn main () { if let Err (e) = __module_init__ () { eprintln ! (\"{}\" , e) ; \
             std :: process :: exit (1) ; } }"
        );
    }
}

/// Lower an expression in condition position (if/while/ternary/assert
/// tests): Python implicitly calls bool() on it. Boolean operators recurse
/// into their operands and `not` negates a condition; comparisons already
/// yield bool; anything else is wrapped in stdpython's Truthy::is_truthy,
/// giving Python's truth table (empty string/collection and zero are
/// false).
pub fn condition_to_rust(    expr: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    match expr {
        ExprType::BoolOp(op)
            if matches!(op.op, crate::BoolOps::And | crate::BoolOps::Or) =>
        {
            let mut parts = Vec::new();
            for value in &op.values {
                parts.push(condition_to_rust(
                    value,
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?);
            }
            Ok(match op.op {
                crate::BoolOps::And => quote!(#((#parts))&&*),
                _ => quote!(#((#parts))||*),
            })
        }
        ExprType::UnaryOp(u) if matches!(u.op, crate::Ops::Not) => {
            let inner = condition_to_rust(&u.operand, ctx, options, symbols)?;
            Ok(quote!(!(#inner)))
        }
        // Comparisons (including `in` and `is None`) already produce bool.
        ExprType::Compare(_) => expr.clone().to_rust(ctx, options, symbols),
        // Bool literals are already bool.
        ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::Bool(_))) => {
            expr.clone().to_rust(ctx, options, symbols)
        }
        other => {
            let tokens = other.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            // An Option/boxed-optional value in condition position (`if
            // conn:` where conn is `BaseHTTPConnection | None` — urllib3):
            // Python's truthiness is "not None" — a user object without
            // __bool__/__len__ is always truthy, and None is false. The
            // generic `Truthy for Option<T>` impl needs `T: Truthy`,
            // which a user class lacks (E0599 ×12 in the corpus);
            // `!(x).py_is_none()` works for Option AND boxed bindings
            // (both have unconditional PyIsNone) and reproduces CPython
            // exactly for these shapes.
            if crate::ast::tree::attribute::receiver_option_inner(
                other,
                &ctx,
                &symbols,
                &options,
            )
            .is_some()
            {
                return Ok(quote!(!(#tokens).py_is_none()));
            }
            Ok(quote!((#tokens).is_truthy()))
        }
    }
}

/// Issue #125: if the test is `x is not None` (or `None is not x`) and x
/// holds an Option, return (x, inner_type) so the if body can narrow x:
/// reads unwrap, and the comprehension/iteration over x sees the inner
/// element type. Returns None for any other test shape.
pub fn narrowing_from_test(
    test: &ExprType,
    options: &PythonOptions,
) -> Option<(String, Option<crate::TypeInfo>)> {
    let ExprType::Compare(cmp) = test else {
        return None;
    };
    if cmp.ops.len() != 1 || !matches!(cmp.ops[0], crate::Compares::IsNot) {
        return None;
    }
    // One side must be None, the other a plain name.
    let left_is_none = crate::is_none_expr(&cmp.left);
    let right_is_none = cmp
        .comparators
        .first()
        .is_some_and(crate::is_none_expr);
    if left_is_none == right_is_none {
        return None; // both None (degenerate) or neither — no narrowing
    }
    let name = if left_is_none {
        cmp.comparators.first()?
    } else {
        &cmp.left
    };
    let ExprType::Name(n) = name else {
        return None;
    };
    // Only names that are statically known to hold an Option narrow.
    if !options.optional_names.contains(&n.id) {
        return None;
    }
    let inner = options
        .name_types
        .get(&n.id)
        .and_then(|t| match t {
            crate::TypeInfo::Option(inner) => Some((**inner).clone()),
            _ => None,
        });
    Some((n.id.clone(), inner))
}

/// Issue #121: `if isinstance(x, (bytes, bytearray)):` (or
/// `if not isinstance(...)`) narrows a `str | bytes` union or a boxed
/// PyValue union: the tested branch becomes the concrete member type, the
/// other branch becomes the complement (String for StrOrBytes, the boxed
/// PyValue for PyValue). A compound test — `if isinstance(x, tuple) and
/// len(x) == 2:` — narrows only the body (the `and` could fail on the
/// other conjunct, so the else stays the original type). Returns
/// (name, body_type, else_type). None for any other test.
/// The member TypeInfo an isinstance NARROWING target maps to — the ONE
/// authority for the narrowing name set: the alias-resolution guard and
/// the known-target gate in isinstance_narrowing both derive from it, so
/// the three lists (previously kept in sync by hand, 55 lines apart)
/// cannot drift.
fn narrowing_member_of(id: &str) -> Option<crate::TypeInfo> {
    match id {
        "str" => Some(crate::TypeInfo::String),
        "bytes" | "bytearray" => Some(crate::TypeInfo::Bytes),
        "int" => Some(crate::TypeInfo::Int),
        "float" => Some(crate::TypeInfo::Float),
        "bool" => Some(crate::TypeInfo::Bool),
        // A tuple member is read as its element vector
        // (PyValue::as_tuple().unwrap().clone()).
        "tuple" => Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::PyValue))),
        _ => None,
    }
}

pub fn isinstance_narrowing(
    test: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<(String, crate::TypeInfo, crate::TypeInfo)> {
    use crate::ExprType;
    // The test may be wrapped in a unary not: `if not isinstance(...)`.
    let (negated, inner) = match test {
        ExprType::UnaryOp(u) if matches!(u.op, crate::ast::tree::unary_op::Ops::Not) => {
            (true, u.operand.as_ref())
        }
        // A compound `isinstance(x, T) and <rest>` narrows the body only.
        ExprType::BoolOp(op) if matches!(op.op, crate::BoolOps::And) => {
            for value in &op.values {
                if let Some((name, body_ty, _)) =
                    isinstance_narrowing(value, options, symbols)
                {
                    // The else branch keeps the ORIGINAL type: the `and`
                    // may have failed on the other conjunct, so the
                    // complement narrowing does not apply.
                    let original = options
                        .name_types
                        .get(&name)
                        .cloned()
                        .unwrap_or(crate::TypeInfo::StrOrBytes);
                    return Some((name, body_ty, original));
                }
            }
            return None;
        }
        other => (false, other),
    };
    let ExprType::Call(call) = inner else {
        return None;
    };
    let ExprType::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id != "isinstance" || call.args.len() != 2 {
        return None;
    }
    let ExprType::Name(n) = &call.args[0] else {
        return None;
    };
    // The name must be a str|bytes union or a boxed PyValue union.
    let is_union = options
        .name_types
        .get(&n.id)
        .is_some_and(|t| matches!(t, crate::TypeInfo::StrOrBytes | crate::TypeInfo::PyValue));
    if !is_union {
        return None;
    }
    // The second argument: a tuple of type names or a single type name
    // (str narrows to String, bytes/bytearray to Bytes, int/float/bool/
    // tuple to the PyValue members). Aliased type names (`builtin_str =
    // str`) and imported aliases resolve through symbols.
    fn resolve_type_name(
        id: &str,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Option<String> {
        resolve_type_name_depth(id, options, symbols, 0)
    }
    fn resolve_type_name_depth(
        id: &str,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
        depth: usize,
    ) -> Option<String> {
        if depth > 16 {
            return None;
        }
        match symbols.get(id) {
            Some(crate::SymbolTableNode::Assign { value, .. }) => match value {
                ExprType::Name(n) if narrowing_member_of(&n.id).is_some() => {
                    Some(n.id.clone())
                }
                _ => None,
            },
            Some(crate::SymbolTableNode::Alias(canonical)) => {
                // A self-aliasing re-export (`from .connection import
                // ProxyConfig as ProxyConfig` — urllib3) would recurse
                // forever; the alias is a no-op.
                if canonical == id {
                    None
                } else {
                    resolve_type_name_depth(canonical, options, symbols, depth + 1)
                }
            }
            Some(crate::SymbolTableNode::ImportFrom(i)) => {
                let path = i.resolved_module_path(options);
                if options.module_defs.contains_key(&path) {
                    let module = options.module_defs.get(&path)?;
                    let module: &crate::Module = module;
                    let syms = module
                        .clone()
                        .find_symbols(SymbolTableScopes::new());
                    resolve_type_name_depth(id, options, &syms, depth + 1)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
    // Collect the resolved target type names (deduplicated, in order).
    let mut targets: Vec<String> = Vec::new();
    match &call.args[1] {
        ExprType::Name(t) => {
            let id = resolve_type_name(&t.id, options, symbols).unwrap_or_else(|| t.id.clone());
            targets.push(id);
        }
        ExprType::Tuple(tup) => {
            for elt in &tup.elts {
                let ExprType::Name(t) = elt else {
                    return None;
                };
                let id = resolve_type_name(&t.id, options, symbols)
                    .unwrap_or_else(|| t.id.clone());
                targets.push(id);
            }
        }
        _ => return None,
    }
    if targets.is_empty() || targets.iter().any(|t| narrowing_member_of(t).is_none()) {
        return None;
    }
    let members: Vec<crate::TypeInfo> = targets
        .iter()
        .filter_map(|t| narrowing_member_of(t))
        .collect();
    // Deduplicate: (bytes, bytearray) both narrow to Bytes.
    let mut distinct: Vec<crate::TypeInfo> = Vec::new();
    for m in members {
        if !distinct.contains(&m) {
            distinct.push(m);
        }
    }
    if distinct.is_empty() {
        return None;
    }
    let original_is_pyvalue = options
        .name_types
        .get(&n.id)
        .is_some_and(|t| matches!(t, crate::TypeInfo::PyValue));
    let (tested, other) = if distinct.len() == 1 {
        let member = distinct.pop().unwrap();
        if original_is_pyvalue {
            (
                crate::TypeInfo::PyValueMember(Box::new(member.clone())),
                crate::TypeInfo::PyValue,
            )
        } else {
            // StrOrBytes: bytes → Bytes, str → String; the complement is
            // the other branch.
            match member {
                crate::TypeInfo::Bytes => {
                    (crate::TypeInfo::Bytes, crate::TypeInfo::String)
                }
                crate::TypeInfo::String => {
                    (crate::TypeInfo::String, crate::TypeInfo::Bytes)
                }
                _ => return None,
            }
        }
    } else {
        // Several DISTINCT member targets on a PyValue narrow nothing
        // (the test itself still dispatches at runtime via call.rs).
        return None;
    };
    let (body_ty, else_ty) = if negated {
        (other, tested)
    } else {
        (tested, other)
    };
    Some((n.id.clone(), body_ty, else_ty))
}

/// Issue #125: update the function-level narrowed set after a statement.
/// An `if x is not None: <body> else: <else>` where BOTH branches leave x
/// non-None narrows x for the rest of the function. Any statement that can
/// assign None to a narrowed name drops it from the set (a store of a
/// possibly-None value — conservative: an `x = ...` whose value is not
/// statically non-None removes x).
pub fn update_narrowed_after_statement(
    stmt: &crate::Statement,
    narrowed: &mut std::collections::HashMap<String, crate::TypeInfo>,
    options: &PythonOptions,
) {
    match &stmt.statement {
        crate::StatementType::If(i) => {
            // The test narrows x in the body; both branches leaving x
            // non-None narrows x AFTER the if/else.
            if let Some((name, inner)) = narrowing_from_test(&i.test, options) {
                let body_ok = branch_ends_non_none(&i.body);
                let else_ok = branch_ends_non_none(&i.orelse);
                if body_ok && else_ok {
                    narrowed.insert(
                        name,
                        inner.unwrap_or(crate::TypeInfo::StrOrBytes),
                    );
                }
            }
            // A name narrowed by an INNER statement only narrows within that
            // branch, not after the if (a branch may not run). Nothing to
            // propagate.
        }
        // `x = None` (or any store that may produce None) invalidates the
        // narrowing; an assignment of a statically non-None value keeps it.
        crate::StatementType::Assign(a) => {
            if let [crate::ExprType::Name(n)] = a.targets.as_slice() {
                if narrowed.contains_key(&n.id) && !statically_non_none(&a.value) {
                    narrowed.remove(&n.id);
                }
            }
        }
        crate::StatementType::AugAssign(a) => {
            if let crate::ExprType::Name(n) = &a.target {
                narrowed.remove(&n.id);
            }
        }
        _ => {}
    }
}

/// Whether a statement list's last statement is an assignment of a
/// statically non-None value to some name (the "branch leaves the name
/// non-None" check for post-if narrowing). Only the LAST store matters:
/// earlier stores are overwritten.
fn branch_ends_non_none(body: &[crate::Statement]) -> bool {
    for stmt in body.iter().rev() {
        match &stmt.statement {
            crate::StatementType::Assign(a) => {
                return statically_non_none(&a.value);
            }
            crate::StatementType::Pass => continue,
            // A nested if/else is treated conservatively: only when its own
            // branches both assign non-None does it count as ending non-None.
            crate::StatementType::If(i) => {
                if i.orelse.is_empty() {
                    return false;
                }
                return branch_ends_non_none(&i.body) && branch_ends_non_none(&i.orelse);
            }
            _ => return false,
        }
    }
    false
}

/// Whether an expression is statically NOT None: a literal other than None,
/// a non-None constant, a call (functions never return the None literal
/// here — conservative), a non-Option name, or a container literal.
fn statically_non_none(expr: &crate::ExprType) -> bool {
    match expr {
        crate::ExprType::Constant(c) => c.0.is_some(),
        crate::ExprType::Name(n) => !matches!(n.id.as_str(), "None" | "True" | "False"),
        crate::ExprType::List(_)
        | crate::ExprType::Dict(_)
        | crate::ExprType::Set(_)
        | crate::ExprType::Tuple(_)
        | crate::ExprType::ListComp(_)
        | crate::ExprType::DictComp(_)
        | crate::ExprType::SetComp(_)
        | crate::ExprType::Call(_)
        | crate::ExprType::JoinedStr(_)
        | crate::ExprType::FormattedValue(_) => true,
        _ => false,
    }
}
