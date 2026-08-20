//! M1 of issue #109: parameter type inference — trait-bound generic
//! signatures for unannotated parameters.
//!
//! An unannotated parameter used to lower to `impl Into<PyObject>`, which
//! no ordinary rython value satisfies — such functions converted but were
//! uncallable, and the failure surfaced in rustc, not at conversion time.
//! This pass instead looks at how the parameter is USED, derives the
//! weakest trait bounds those uses require, and emits a generic signature
//! (`fn add<A, B>(a: A, b: B) -> Result<<A as PyAdd<B>>::Output, ...>
//! where A: PyAdd<B>`), monomorphized by rustc per call site. A use with
//! no existing or generatable trait is a loud conversion error — never
//! silently accepted, never deferred to rustc.
//!
//! Scope (M1): free functions; operator/comparison/conversion-builtin/
//! Truthy/Len/PyDisplay/PyRepr/PyHash/PyIndex/PyContains rows. Method
//! parameters, callable parameters, iteration, and interprocedural flow are
//! loud errors naming the gap (later milestones).

use std::collections::{HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::quote;

use crate::ast::tree::{BinOps, Compares, StatementType};
use crate::{ExprType, Statement, SymbolTableScopes, TypeInfo};

/// The requirement one use of a parameter places on it: the weakest bound
/// admitting the use.
#[derive(Clone, Debug)]
pub enum ParamReq {
    /// `p OP rhs` — the Py* operator trait and the other operand's type.
    Op(&'static str, RhsType),
    /// `p CMP rhs` — the Py* comparison trait and the other operand's type.
    Cmp(&'static str, RhsType),
    /// `int(p)` / `float(p)` / `bool(p)` / `str(p)` / `abs(p)` — the
    /// runtime conversion trait (`PyInt`, `PyFloat`, ...).
    Conversion(&'static str),
    /// `if p:` / `while p:` / `not p` / `p and x` / `p or x`.
    Truthy,
    /// `len(p)`.
    Len,
    /// `print(p)`, `f"{p}"`.
    Display,
    /// `repr(p)`, `f"{p!r}"`.
    Repr,
    /// `hash(p)`.
    Hash,
    /// `p is None` / `p is not None` — `PyIsNone`.
    IsNone,
    /// `p[i]` read — `PyIndex<Idx>`.
    Index(RhsType),
    /// `p[i] = v` / `p[i] += v` — `PySetIndex<Idx, Value>`.
    SetIndex(RhsType, RhsType),
    /// `x in p` — `PyContains<Item>`.
    Contains(RhsType),
    /// A use no existing or generatable trait admits.
    Untranslatable(String),
}

/// The type of the OTHER operand of an operator/comparison/index use.
#[derive(Clone, Debug)]
pub enum RhsType {
    /// A concrete Rust type (anchored literals, typed locals).
    Concrete(TokenStream),
    /// Another unannotated parameter — its type variable.
    Param(String),
    /// The same parameter on both sides (`p + p`).
    Same,
    /// Not statically knowable.
    Unknown,
}

/// The inferred signature for a function with unannotated parameters.
#[derive(Debug, Default)]
pub struct InferredSignature {
    /// The generic parameter list: `<A, B>` (single param: `<T>`).
    pub type_params: Vec<TokenStream>,
    /// The where-clause bounds: `A: PyAdd<B>, A: Clone`.
    pub where_bounds: Vec<TokenStream>,
    /// Parameter name → its Rust type (the type variable).
    pub param_types: HashMap<String, TokenStream>,
    /// The Ok type of the Result return, when inferable from the returns.
    pub return_type: Option<TokenStream>,
}

impl InferredSignature {
    /// Whether any unannotated parameter produced requirements (i.e. this
    /// function has an inferred generic signature at all).
    pub fn is_generic(&self) -> bool {
        !self.type_params.is_empty()
    }

    /// The `where A: ..., B: ...` clause, or empty when there are no bounds.
    pub fn where_clause(&self) -> TokenStream {
        if self.where_bounds.is_empty() {
            return TokenStream::new();
        }
        let bounds = &self.where_bounds;
        quote!(where #(#bounds),*)
    }

    /// The full generic header: `<A, B>` or empty.
    pub fn generic_header(&self) -> TokenStream {
        if self.type_params.is_empty() {
            return TokenStream::new();
        }
        let params = &self.type_params;
        quote!(<#(#params),*>)
    }
}

/// The per-function inference pass. `params` are the unannotated parameter
/// names in declaration order (excluding `self`); `name_types` and
/// `use_counts` come from the same `analyze_function_types` pass that
/// drives clone-on-reuse and empty-container pinning.
pub fn infer_unannotated_signature(
    body: &[Statement],
    params: &[String],
    name_types: &HashMap<String, TypeInfo>,
    use_counts: &HashMap<String, usize>,
    symbols: &SymbolTableScopes,
) -> Result<InferredSignature, String> {
    if params.is_empty() {
        return Ok(InferredSignature::default());
    }
    let unannotated: HashSet<String> = params.iter().cloned().collect();

    let mut collector = Collector {
        unannotated: &unannotated,
        name_types,
        symbols,
        reqs: HashMap::new(),
        alias: HashMap::new(),
        returns: Vec::new(),
        reassigned: HashSet::new(),
    };
    collector.walk(body);

    // A reassigned parameter cannot keep a single inferred type.
    for name in params {
        if collector.reassigned.contains(name) {
            return Err(format!(
                "parameter `{name}` is assigned to inside the function; an inferred \
                 generic type cannot change. Annotate `{name}` with its type"
            ));
        }
    }

    // One type variable per parameter, in declaration order (single param:
    // `T`, otherwise `A`, `B`, ... per the issue's naming guidance).
    let mut tv_names: HashMap<String, String> = HashMap::new();
    let mut type_params = Vec::new();
    let mut param_types = HashMap::new();
    if params.len() == 1 {
        tv_names.insert(params[0].clone(), "T".to_string());
        type_params.push(quote!(T));
        param_types.insert(params[0].clone(), quote!(T));
    } else {
        for (i, name) in params.iter().enumerate() {
            let tv = format!("{}", (b'A' + i as u8) as char);
            tv_names.insert(name.clone(), tv.clone());
            let ident = quote::format_ident!("{}", tv);
            type_params.push(quote!(#ident));
            param_types.insert(name.clone(), quote!(#ident));
        }
    }

    // Resolve each parameter's requirement set into bounds.
    let mut where_bounds = Vec::new();
    for name in params {
        let tv = quote::format_ident!("{}", tv_names.get(name).unwrap());
        let reqs = collector.reqs.get(name).cloned().unwrap_or_default();
        let mut seen = HashSet::new();
        for req in &reqs {
            if let ParamReq::Untranslatable(what) = req {
                return Err(format!(
                    "parameter `{name}`: {what}. Annotate `{name}` with a concrete \
                     type, or use it only through operations the transpiler can \
                     bound (issue #109: parameter type inference, M1)"
                ));
            }
            let bound = match req {
                ParamReq::Op(trait_name, rhs) => {
                    let t = quote::format_ident!("{}", trait_name);
                    let rhs = render_rhs(rhs, &tv_names)?;
                    quote!(#tv: #t<#rhs>)
                }
                ParamReq::Cmp(trait_name, rhs) => {
                    let t = quote::format_ident!("{}", trait_name);
                    let rhs = render_rhs(rhs, &tv_names)?;
                    quote!(#tv: #t<#rhs>)
                }
                ParamReq::Conversion(trait_name) => {
                    let t = quote::format_ident!("{}", trait_name);
                    quote!(#tv: #t)
                }
                ParamReq::Truthy => quote!(#tv: Truthy),
                ParamReq::Len => quote!(#tv: Len),
                ParamReq::Display => quote!(#tv: PyDisplay),
                ParamReq::Repr => quote!(#tv: PyRepr),
                ParamReq::Hash => quote!(#tv: PyHash),
                ParamReq::IsNone => quote!(#tv: PyIsNone),
                ParamReq::Index(idx) => {
                    let idx = render_rhs(idx, &tv_names)?;
                    quote!(#tv: PyIndex<#idx>)
                }
                ParamReq::SetIndex(idx, val) => {
                    let idx = render_rhs(idx, &tv_names)?;
                    let val = render_rhs(val, &tv_names)?;
                    quote!(#tv: PySetIndex<#idx, #val>)
                }
                ParamReq::Contains(item) => {
                    let item = render_rhs(item, &tv_names)?;
                    quote!(#tv: PyContains<#item>)
                }
                ParamReq::Untranslatable(_) => unreachable!("handled above"),
            };
            if seen.insert(bound.to_string()) {
                where_bounds.push(bound);
            }
        }
        // The reuse-clone rule: a generic parameter is not known Copy, so a
        // parameter read more than once needs `T: Clone` — the rule itself
        // is a use, so the bound stays minimal.
        if use_counts.get(name).copied().unwrap_or(0) > 1 {
            where_bounds.push(quote!(#tv: Clone));
        }
    }

    // Return type: every return value must unify to one type expression.
    let return_type = if collector.returns.is_empty() {
        None
    } else {
        let mut inferred: Option<TokenStream> = None;
        for ret in &collector.returns {
            let ty = return_type_of(ret, &collector, &tv_names)?;
            match &inferred {
                None => inferred = Some(ty),
                Some(prev) if prev.to_string() == ty.to_string() => {}
                _ => {
                    return Err(format!(
                        "return statements have different types; annotate the \
                         function's return type (issue #109, M1)"
                    ));
                }
            }
        }
        inferred
    };

    Ok(InferredSignature {
        type_params,
        where_bounds,
        param_types,
        return_type,
    })
}

/// Render an operand's type for a where-bound: the parameter's variable,
/// the concrete type, or Self for same-param operands.
fn render_rhs(rhs: &RhsType, tv_names: &HashMap<String, String>) -> Result<TokenStream, String> {
    Ok(match rhs {
        RhsType::Concrete(t) => t.clone(),
        RhsType::Param(name) => match tv_names.get(name) {
            Some(tv) => {
                let ident = quote::format_ident!("{}", tv);
                quote!(#ident)
            }
            None => {
                return Err(format!(
                    "internal: parameter `{name}` used as an operand but has no \
                     type variable"
                ))
            }
        },
        RhsType::Same => quote!(Self),
        RhsType::Unknown => {
            return Err(
                "the other operand's type cannot be inferred; annotate the \
                 parameter or the other operand"
                    .to_string(),
            )
        }
    })
}

/// The return type expression for one return value, in terms of the type
/// variables.
fn return_type_of(
    expr: &ExprType,
    collector: &Collector,
    tv_names: &HashMap<String, String>,
) -> Result<TokenStream, String> {
    let param_tv = |name: &str| -> Option<TokenStream> {
        let p = if tv_names.contains_key(name) {
            Some(name.to_string())
        } else {
            collector.alias.get(name).cloned()
        };
        p.and_then(|p| tv_names.get(&p)).map(|tv| {
            let ident = quote::format_ident!("{}", tv);
            quote!(#ident)
        })
    };
    let err = || {
        "cannot infer the type of this return value; annotate the function's \
         return type (issue #109, M1)"
            .to_string()
    };

    match expr {
        ExprType::Name(n) => {
            if let Some(tv) = param_tv(&n.id) {
                Ok(tv)
            } else if let Some(t) = collector.name_types.get(&n.id) {
                Ok(t.to_rust_type())
            } else {
                Err(err())
            }
        }
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => Ok(quote!(i64)),
            Some(litrs::Literal::Float(_)) => Ok(quote!(f64)),
            Some(litrs::Literal::String(_)) => Ok(quote!(String)),
            Some(litrs::Literal::Bool(_)) => Ok(quote!(bool)),
            _ => Err(err()),
        },
        ExprType::Call(c) => {
            if let ExprType::Name(f) = c.func.as_ref() {
                match f.id.as_str() {
                    "int" => return Ok(quote!(i64)),
                    "float" => return Ok(quote!(f64)),
                    "bool" => return Ok(quote!(bool)),
                    "str" => return Ok(quote!(String)),
                    "len" => return Ok(quote!(i64)),
                    "abs" => {
                        if let Some(arg) = c.args.first()
                            && let ExprType::Name(n) = arg
                            && let Some(tv) = param_tv(&n.id)
                        {
                            return Ok(quote!(<#tv as PyAbs>::Output));
                        }
                    }
                    _ => {}
                }
            }
            Err(err())
        }
        ExprType::BinOp(b) => {
            let trait_name = bin_op_trait(&b.op).ok_or_else(err)?;
            let t = quote::format_ident!("{}", trait_name);
            let left = operand_type(&b.left, collector, tv_names)?;
            let right = operand_type(&b.right, collector, tv_names)?;
            Ok(quote!(<#left as #t<#right>>::Output))
        }
        ExprType::Compare(c) => {
            let trait_name = match c.ops.first() {
                Some(Compares::Eq) => "PyEq",
                Some(Compares::NotEq) => "PyNe",
                Some(Compares::Lt) => "PyLt",
                Some(Compares::LtE) => "PyLe",
                Some(Compares::Gt) => "PyGt",
                Some(Compares::GtE) => "PyGe",
                _ => return Err(err()),
            };
            let t = quote::format_ident!("{}", trait_name);
            let left = operand_type(&c.left, collector, tv_names)?;
            let right = match c.comparators.first() {
                Some(r) => operand_type(r, collector, tv_names)?,
                None => quote!(Self),
            };
            Ok(quote!(<#left as #t<#right>>::Output))
        }
        _ => Err(err()),
    }
}

/// The type of an operand inside a return expression: a parameter's
/// variable or a concrete type.
fn operand_type(
    expr: &ExprType,
    collector: &Collector,
    tv_names: &HashMap<String, String>,
) -> Result<TokenStream, String> {
    let err = || {
        "the operand's type cannot be inferred; annotate the function's return \
         type (issue #109, M1)"
            .to_string()
    };
    Ok(match expr {
        ExprType::Name(n) => {
            // A parameter (directly or via an alias).
            let p = if tv_names.contains_key(&n.id) {
                Some(n.id.clone())
            } else {
                collector.alias.get(&n.id).cloned()
            };
            if let Some(p) = p {
                if let Some(tv) = tv_names.get(&p) {
                    let ident = quote::format_ident!("{}", tv);
                    quote!(#ident)
                } else {
                    return Err(err());
                }
            } else if let Some(t) = collector.name_types.get(&n.id) {
                t.to_rust_type()
            } else {
                return Err(err());
            }
        }
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => quote!(i64),
            Some(litrs::Literal::Float(_)) => quote!(f64),
            // 'static: a literal is &'static str, and a return-position
            // associated type needs a named lifetime.
            Some(litrs::Literal::String(_)) => quote!(&'static str),
            Some(litrs::Literal::Bool(_)) => quote!(bool),
            _ => return Err(err()),
        },
        // Conversion builtins give concrete operand types (`"v=" + str(x)`
        // needs String on the right).
        ExprType::Call(c) => {
            if let ExprType::Name(f) = c.func.as_ref() {
                match f.id.as_str() {
                    "int" => quote!(i64),
                    "float" => quote!(f64),
                    "bool" => quote!(bool),
                    "str" => quote!(String),
                    "len" => quote!(i64),
                    _ => return Err(err()),
                }
            } else {
                return Err(err());
            }
        }
        _ => return Err(err()),
    })
}

fn bin_op_trait(op: &BinOps) -> Option<&'static str> {
    match op {
        BinOps::Add => Some("PyAdd"),
        BinOps::Sub => Some("PySub"),
        BinOps::Mult => Some("PyMul"),
        BinOps::Div => Some("PyDiv"),
        BinOps::FloorDiv => Some("PyFloorDiv"),
        BinOps::Mod => Some("PyMod"),
        BinOps::Pow => Some("PyPow"),
        BinOps::MatMult => Some("PyMatMul"),
        _ => None,
    }
}

struct Collector<'a> {
    unannotated: &'a HashSet<String>,
    name_types: &'a HashMap<String, TypeInfo>,
    symbols: &'a SymbolTableScopes,
    reqs: HashMap<String, Vec<ParamReq>>,
    /// Local names that alias a parameter (`x = p` → x ↦ p).
    alias: HashMap<String, String>,
    returns: Vec<ExprType>,
    reassigned: HashSet<String>,
}

impl<'a> Collector<'a> {
    fn add(&mut self, param: &str, req: ParamReq) {
        if self.unannotated.contains(param) {
            self.reqs.entry(param.to_string()).or_default().push(req);
        }
    }

    fn walk(&mut self, body: &[Statement]) {
        for stmt in body {
            match &stmt.statement {
                StatementType::Assign(a) => {
                    // `x = p` makes x an alias of p; `p = ...` reassigns p.
                    if let [ExprType::Name(target)] = a.targets.as_slice() {
                        if let ExprType::Name(src) = &a.value {
                            if self.unannotated.contains(&src.id) {
                                self.alias.insert(target.id.clone(), src.id.clone());
                            } else if self.unannotated.contains(&target.id) {
                                self.reassigned.insert(target.id.clone());
                            }
                        }
                    }
                    // Subscript stores mutate the container: `p[i] = v`.
                    for target in &a.targets {
                        if let ExprType::Subscript(s) = target {
                            if let ExprType::Name(n) = s.value.as_ref() {
                                let idx = match s.kind_expr() {
                                    Some(e) => self.rhs_of(e),
                                    None => RhsType::Unknown,
                                };
                                let val = self.rhs_of(&a.value);
                                self.add(&n.id, ParamReq::SetIndex(idx, val));
                            }
                        }
                    }
                    self.walk_expr(&a.value, false);
                    for target in &a.targets {
                        self.walk_expr(target, false);
                    }
                }
                StatementType::AugAssign(a) => {
                    // `p += x`: read + write through the operator; `p[i] += x`
                    // also stores through the index.
                    let op_trait = bin_op_trait(&a.op);
                    if let ExprType::Name(n) = &a.target {
                        if let Some(t) = op_trait {
                            self.add(&n.id, ParamReq::Op(t, self.rhs_of(&a.value)));
                        }
                    } else if let ExprType::Subscript(s) = &a.target {
                        if let ExprType::Name(n) = s.value.as_ref() {
                            if let Some(t) = op_trait {
                                self.add(&n.id, ParamReq::Op(t, self.rhs_of(&a.value)));
                            }
                            let idx = match s.kind_expr() {
                                Some(e) => self.rhs_of(e),
                                None => RhsType::Unknown,
                            };
                            self.add(
                                &n.id,
                                ParamReq::SetIndex(idx, self.rhs_of(&a.value)),
                            );
                        }
                    }
                    self.walk_expr(&a.target, false);
                    self.walk_expr(&a.value, false);
                }
                StatementType::Expr(e) => self.walk_expr(&e.value, false),
                StatementType::Call(c) => {
                    self.walk_expr(&ExprType::Call(c.clone()), false);
                }
                StatementType::Return(Some(e)) => {
                    self.returns.push(e.value.clone());
                    self.walk_expr(&e.value, false);
                }
                StatementType::Return(None) => {}
                StatementType::If(s) => {
                    self.walk_expr(&s.test, true);
                    self.walk(&s.body);
                    self.walk(&s.orelse);
                }
                StatementType::While(s) => {
                    self.walk_expr(&s.test, true);
                    self.walk(&s.body);
                    self.walk(&s.orelse);
                }
                StatementType::For(s) => {
                    // Iterating a parameter is M2 (async-stream protocol).
                    if let ExprType::Name(n) = &s.iter {
                        self.add(
                            &n.id,
                            ParamReq::Untranslatable(
                                "iterating over a parameter (`for x in p`) is not \
                                 inferred yet (issue #109, M2)"
                                    .to_string(),
                            ),
                        );
                    }
                    self.walk_expr(&s.iter, false);
                    self.walk(&s.body);
                    self.walk(&s.orelse);
                }
                StatementType::AsyncFor(s) => {
                    self.walk_expr(&s.iter, false);
                    self.walk(&s.body);
                    self.walk(&s.orelse);
                }
                StatementType::With(s) => {
                    for item in &s.items {
                        self.walk_expr(&item.context_expr, false);
                    }
                    self.walk(&s.body);
                }
                StatementType::AsyncWith(s) => {
                    for item in &s.items {
                        self.walk_expr(&item.context_expr, false);
                    }
                    self.walk(&s.body);
                }
                StatementType::Try(t) => {
                    self.walk(&t.body);
                    for handler in &t.handlers {
                        self.walk(&handler.body);
                    }
                    self.walk(&t.orelse);
                    self.walk(&t.finalbody);
                }
                StatementType::Assert { test, msg } => {
                    self.walk_expr(test, true);
                    if let Some(m) = msg {
                        self.walk_expr(m, false);
                    }
                }
                StatementType::Raise(r) => {
                    if let Some(e) = &r.exc {
                        self.walk_expr(e, false);
                    }
                    if let Some(c) = &r.cause {
                        self.walk_expr(c, false);
                    }
                }
                // Nested functions/classes have their own parameters.
                StatementType::FunctionDef(_)
                | StatementType::AsyncFunctionDef(_)
                | StatementType::ClassDef(_)
                | StatementType::Import(_)
                | StatementType::ImportFrom(_)
                | StatementType::Pass
                | StatementType::Break
                | StatementType::Continue
                | StatementType::Unimplemented(_) => {}
            }
        }
    }

    /// Walk an expression, recording requirements. `truthy` marks a
    /// truthiness context: a parameter that IS the operand gets `Truthy`,
    /// while sub-expressions (e.g. the inside of a comparison) are walked
    /// in normal context.
    fn walk_expr(&mut self, expr: &ExprType, truthy: bool) {
        match expr {
            ExprType::Name(n) => {
                if self.unannotated.contains(&n.id) && truthy {
                    self.add(&n.id, ParamReq::Truthy);
                }
            }
            ExprType::BinOp(b) => {
                if let Some(t) = bin_op_trait(&b.op) {
                    // Only the RECEIVER (left) operand is constrained: the
                    // body emits `left.py_op(&right)`, which needs
                    // `Left: PyOp<Right>` — the right operand is merely the
                    // operand type (minimal constraint; `2 + b` needs no
                    // bound on `b`).
                    if let ExprType::Name(l) = b.left.as_ref() {
                        if self.unannotated.contains(&l.id) {
                            let same = matches!(b.right.as_ref(), ExprType::Name(r) if r.id == l.id);
                            let rhs = if same { RhsType::Same } else { self.rhs_of(&b.right) };
                            self.add(&l.id, ParamReq::Op(t, rhs));
                        }
                    }
                }
                self.walk_expr(&b.left, false);
                self.walk_expr(&b.right, false);
            }
            ExprType::Compare(c) => {
                let mut operands: Vec<&ExprType> =
                    std::iter::once(c.left.as_ref()).chain(c.comparators.iter()).collect();
                for (i, op) in c.ops.iter().enumerate() {
                    let left = operands[i];
                    let right = operands[i + 1];
                    match op {
                        Compares::In | Compares::NotIn => {
                            // `x in p` bounds the CONTAINER (comparator).
                            if let ExprType::Name(n) = right {
                                if self.unannotated.contains(&n.id) {
                                    self.add(&n.id, ParamReq::Contains(self.rhs_of(left)));
                                }
                            }
                        }
                        Compares::Is | Compares::IsNot => {
                            // `x is None` → py_is_none (PyIsNone bound);
                            // a parameter on either side is tested.
                            for operand in [left, right] {
                                if let ExprType::Name(n) = operand {
                                    if self.unannotated.contains(&n.id) {
                                        self.add(&n.id, ParamReq::IsNone);
                                    }
                                }
                            }
                        }
                        _ => {
                            let trait_name = match op {
                                Compares::Eq => "PyEq",
                                Compares::NotEq => "PyNe",
                                Compares::Lt => "PyLt",
                                Compares::LtE => "PyLe",
                                Compares::Gt => "PyGt",
                                Compares::GtE => "PyGe",
                                _ => "PyEq",
                            };
                            // Only the receiver (left) operand is
                            // constrained: `left.py_cmp(&right)`.
                            if let ExprType::Name(l) = left {
                                if self.unannotated.contains(&l.id) {
                                    self.add(&l.id, ParamReq::Cmp(trait_name, self.rhs_of(right)));
                                }
                            }
                        }
                    }
                }
                for operand in operands {
                    self.walk_expr(operand, false);
                }
            }
            ExprType::UnaryOp(u) => {
                // `not p` truthiness-tests p.
                if matches!(u.op, crate::Ops::Not) {
                    self.walk_expr(&u.operand, true);
                } else {
                    self.walk_expr(&u.operand, false);
                }
            }
            ExprType::BoolOp(b) => {
                for v in &b.values {
                    self.walk_expr(v, true);
                }
            }
            ExprType::Call(c) => {
                // Builtins with a parameter argument: conversions, len,
                // repr, hash, print. A user function shadowing the name
                // means it is not the builtin.
                if let ExprType::Name(f) = c.func.as_ref() {
                    let shadowed = self
                        .symbols
                        .get(&f.id)
                        .is_some_and(|s| matches!(s, crate::SymbolTableNode::FunctionDef(_)));
                    if !shadowed {
                        let builtin_req = match f.id.as_str() {
                            "int" => Some(ParamReq::Conversion("PyInt")),
                            "float" => Some(ParamReq::Conversion("PyFloat")),
                            "bool" => Some(ParamReq::Conversion("PyBool")),
                            "str" => Some(ParamReq::Conversion("PyToString")),
                            "abs" => Some(ParamReq::Conversion("PyAbs")),
                            "len" => Some(ParamReq::Len),
                            "repr" => Some(ParamReq::Repr),
                            "hash" => Some(ParamReq::Hash),
                            "print" => Some(ParamReq::Display),
                            _ => None,
                        };
                        if let Some(req) = builtin_req {
                            if let Some(arg) = c.args.first()
                                && let ExprType::Name(n) = arg
                                && self.unannotated.contains(&n.id)
                            {
                                self.add(&n.id, req);
                            }
                        }
                        // isinstance(p, T): the type must be known
                        // statically; an unannotated parameter cannot be
                        // checked, so the call is loud here (mirroring
                        // call.rs's own refusal for unknown types).
                        if f.id == "isinstance" {
                            if let Some(arg) = c.args.first()
                                && let ExprType::Name(n) = arg
                                && self.unannotated.contains(&n.id)
                            {
                                self.add(
                                    &n.id,
                                    ParamReq::Untranslatable(format!(
                                        "isinstance cannot determine `{}`'s type                                          statically; annotate `{}`",
                                        n.id, n.id
                                    )),
                                );
                            }
                        }
                    }
                    // A parameter CALLED as a function.
                    if self.unannotated.contains(&f.id) {
                        self.add(
                            &f.id,
                            ParamReq::Untranslatable(
                                "calling a parameter (`p(...)`) is not supported yet \
                                 (issue #109: callable parameters are out of scope)"
                                    .to_string(),
                            ),
                        );
                    }
                }
                // A parameter passed to a user function (interprocedural
                // flow is M4).
                if let ExprType::Name(f) = c.func.as_ref() {
                    if self
                        .symbols
                        .get(&f.id)
                        .is_some_and(|s| matches!(s, crate::SymbolTableNode::FunctionDef(_)))
                    {
                        for arg in &c.args {
                            if let ExprType::Name(n) = arg {
                                if self.unannotated.contains(&n.id) {
                                    self.add(
                                        &n.id,
                                        ParamReq::Untranslatable(format!(
                                            "passing `{}` to user function `{}` is not \
                                             inferred yet (issue #109, M4); annotate `{}`",
                                            n.id, f.id, n.id
                                        )),
                                    );
                                }
                            }
                        }
                    }
                }
                self.walk_expr(&c.func, false);
                for arg in &c.args {
                    self.walk_expr(arg, false);
                }
                for kw in &c.keywords {
                    self.walk_expr(&kw.value, false);
                }
            }
            ExprType::Attribute(a) => {
                // `p.attr` / `p.method(...)` — the method table is M2.
                if let ExprType::Name(n) = a.value.as_ref() {
                    if self.unannotated.contains(&n.id) {
                        self.add(
                            &n.id,
                            ParamReq::Untranslatable(format!(
                                "accessing `.{}(...)` on a parameter is not inferred yet \
                                 (issue #109, M2); annotate `{}`",
                                a.attr, n.id
                            )),
                        );
                    }
                }
                self.walk_expr(&a.value, false);
            }
            ExprType::Subscript(s) => {
                // `p[i]` read: PyIndex<Idx>.
                let idx_expr = s.kind_expr();
                if let ExprType::Name(n) = s.value.as_ref() {
                    if self.unannotated.contains(&n.id) {
                        let idx = match idx_expr {
                            Some(e) => self.rhs_of(e),
                            None => RhsType::Unknown,
                        };
                        self.add(&n.id, ParamReq::Index(idx));
                    }
                }
                // A parameter used as the INDEX is not inferable (M1).
                if let Some(idx) = idx_expr {
                    if let ExprType::Name(n) = idx {
                        if self.unannotated.contains(&n.id) {
                            self.add(
                                &n.id,
                                ParamReq::Untranslatable(format!(
                                    "using `{}` as an index is not inferred yet (issue #109, \
                                     M1); annotate `{}`",
                                    n.id, n.id
                                )),
                            );
                        }
                    }
                }
                self.walk_expr(&s.value, false);
                if let Some(idx) = idx_expr {
                    self.walk_expr(idx, false);
                }
            }
            ExprType::JoinedStr(f) => {
                for v in &f.values {
                    self.walk_expr(v, false);
                }
            }
            ExprType::FormattedValue(f) => {
                // `f"{p}"` → PyDisplay; `f"{p!r}"` → PyRepr.
                if let ExprType::Name(n) = f.value.as_ref() {
                    if self.unannotated.contains(&n.id) {
                        let req = if f.conversion == Some(114) {
                            ParamReq::Repr
                        } else {
                            ParamReq::Display
                        };
                        self.add(&n.id, req);
                    }
                }
                self.walk_expr(&f.value, false);
                if let Some(spec) = &f.format_spec {
                    self.walk_expr(spec, false);
                }
            }
            ExprType::Dict(d) => {
                for k in &d.keys {
                    if let Some(k) = k {
                        self.walk_expr(k, false);
                    }
                }
                for v in &d.values {
                    self.walk_expr(v, false);
                }
            }
            ExprType::Set(s) => {
                for elt in &s.elts {
                    self.walk_expr(elt, false);
                }
            }
            ExprType::List(items) => {
                for item in items {
                    self.walk_expr(item, false);
                }
            }
            ExprType::Tuple(t) => {
                for elt in &t.elts {
                    self.walk_expr(elt, false);
                }
            }
            ExprType::IfExp(i) => {
                self.walk_expr(&i.test, true);
                self.walk_expr(&i.body, false);
                self.walk_expr(&i.orelse, false);
            }
            ExprType::Starred(s) => self.walk_expr(&s.value, false),
            ExprType::NamedExpr(n) => {
                self.walk_expr(&n.left, false);
                self.walk_expr(&n.right, false);
            }
            ExprType::Await(a) => self.walk_expr(&a.value, false),
            ExprType::Yield(y) => {
                if let Some(v) = &y.value {
                    self.walk_expr(v, false);
                }
            }
            ExprType::YieldFrom(y) => self.walk_expr(&y.value, false),
            ExprType::Lambda(l) => self.walk_expr(&l.body, false),
            ExprType::ListComp(l) => self.walk_comprehension(&l.elt, &l.generators),
            ExprType::SetComp(s) => self.walk_comprehension(&s.elt, &s.generators),
            ExprType::DictComp(d) => self.walk_comprehension(&d.value, &d.generators),
            ExprType::GeneratorExp(g) => self.walk_comprehension(&g.elt, &g.generators),
            ExprType::Constant(_)
            | ExprType::NoneType(_)
            | ExprType::Unknown
            | ExprType::Unimplemented(_) => {}
        }
    }

    fn walk_comprehension(&mut self, elt: &ExprType, generators: &[crate::Comprehension]) {
        self.walk_expr(elt, false);
        for generator in generators {
            self.walk_expr(&generator.iter, false);
            for cond in &generator.ifs {
                self.walk_expr(cond, true);
            }
        }
    }

    /// The RhsType of an operand: a literal, a param, a param alias, a
    /// typed local, or Unknown.
    fn rhs_of(&self, expr: &ExprType) -> RhsType {
        match expr {
            ExprType::Name(n) => {
                if self.unannotated.contains(&n.id) {
                    RhsType::Param(n.id.clone())
                } else if let Some(p) = self.alias.get(&n.id) {
                    RhsType::Param(p.clone())
                } else if let Some(t) = self.name_types.get(&n.id) {
                    RhsType::Concrete(t.to_rust_type())
                } else {
                    RhsType::Unknown
                }
            }
            ExprType::Constant(c) => match &c.0 {
                Some(litrs::Literal::Integer(_)) => RhsType::Concrete(quote!(i64)),
                Some(litrs::Literal::Float(_)) => RhsType::Concrete(quote!(f64)),
                Some(litrs::Literal::String(_)) => RhsType::Concrete(quote!(&str)),
                Some(litrs::Literal::Bool(_)) => RhsType::Concrete(quote!(bool)),
                _ => RhsType::Unknown,
            },
            ExprType::UnaryOp(u) => {
                // -1 / -1.0 anchor like their operand.
                if matches!(u.op, crate::Ops::USub) {
                    self.rhs_of(&u.operand)
                } else {
                    RhsType::Unknown
                }
            }
            _ => RhsType::Unknown,
        }
    }
}

/// The expression inside a subscript's kind (index, or slice lower bound
/// when present).
pub(crate) trait SubscriptKindExpr {
    fn kind_expr(&self) -> Option<&ExprType>;
}
impl SubscriptKindExpr for crate::Subscript {
    fn kind_expr(&self) -> Option<&ExprType> {
        match &self.kind {
            crate::SubscriptKind::Index(e) => Some(e.as_ref()),
            crate::SubscriptKind::Slice { lower, .. } => lower.as_deref(),
        }
    }
}
