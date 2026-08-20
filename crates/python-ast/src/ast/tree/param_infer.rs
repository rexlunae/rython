//! Issue #109: parameter type inference — trait-bound generic signatures
//! for unannotated parameters (milestones M1–M4).
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
//! Scope:
//! - M1: free functions; operator/comparison/conversion-builtin/Truthy/Len/
//!   PyDisplay/PyRepr/PyHash/PyIndex/PyContains rows.
//! - M2: stdlib method table (PyStrOps/PyListOps/PyPop bounds) with
//!   forwarding impls, and per-method return types. Iteration:
//!   `for x in p` over an unannotated parameter (or an alias of one)
//!   bounds it `T: IntoIterator<Item = E>`, and the loop variable becomes
//!   a virtual parameter whose OWN bounds come from the body's uses of it
//!   (`for w in words: out.append(w.upper())` → `B: PyStrOps`). The bound
//!   and the element requirements flow through callers. A loop-target
//!   return that can fall through (Python returns None) is a loud error.
//! - M3: user-class duck typing — a generated `Has*` trait per method,
//!   unified across every defining class in the package.
//! - M4: interprocedural FlowsTo — a call to a user function adopts the
//!   callee's parameter requirements (a concretely-annotated callee
//!   parameter identity-forces the argument's type; self-recursion is a
//!   fixpoint: `repeat(x, n)` gets `A: PyAdd<A, Output = A>` and
//!   `B: PyLe<B> + PyFromInt + PySub<i64, Output = B>`, callable with
//!   `n = 2` or `n = 2.5`), and the callee's return type flows to the
//!   caller's return. Mutual recursion without annotations is a loud
//!   error.
//! - M5: call-site satisfiability — a call whose arguments are statically
//!   known (literals, typed locals) is verified against the callee's
//!   inferred bounds at conversion time, mirroring stdpython's actual
//!   impls, so `add("a", 1)` or `hear(5)` is a loud error naming the call
//!   instead of a rustc surprise. Runs for every body: inferred functions,
//!   annotated/paramless functions, module init, and the __main__ block.
//!   A definition whose bound set no known type satisfies (`p.upper()` +
//!   `p.pop()`) stays a well-formed definition — Python allows it — and
//!   surfaces as a -W warning (a #[deprecated] note at -W warn, a
//!   conversion failure at -W deny).
//!
//! Still loud errors (later milestones): callable parameters, tuple/
//! attribute loop targets, and method parameters.

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
    /// `p OP rhs` where the result flows BACK into the same parameter (the
    /// recursion fixpoint, M4): the op's Output must be the parameter's
    /// type (`T: PySub<i64, Output = T>` for `repeat(x, n - 1)`).
    OpOutput(&'static str, RhsType),
    /// `p CMP rhs` in a CONDITION (`if p <= 0:`): the comparison must be a
    /// bool (`T: PyLe<i64, Output = bool>`), mirroring Python's always-bool
    /// comparisons.
    CmpCond(&'static str, RhsType),
    /// `p CMP rhs` — the Py* comparison trait and the other operand's type.
    Cmp(&'static str, RhsType),
    /// `p CMP <int literal>`: the literal is converted to the parameter's
    /// own type via stdpython's `PyFromInt` (identity for i64, float
    /// promotion for f64), because Rust std has no int/float
    /// cross-PartialOrd — the bound `T: PyLe<T>` is satisfied by both i64
    /// and f64 (M4).
    PyFromInt,
    /// `for x in p` (M2): the parameter is an iterable — `T: IntoIterator<
    /// Item = E>`, where the element name is a virtual parameter whose own
    /// requirements come from the loop body's uses of `x`.
    Iterate(String),
    /// An element of a `"sep".join(...)` argument (issue #116): the
    /// element must be `AsRef<str>` so `str::join`/PyStrOps::join accept it.
    AsRefStr,
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
    /// `p.m(...)` for a method — the trait declaring it (stdlib or a
    /// generated Has* duck trait), whether it mutates, and the RhsType of
    /// parameterized traits (pop's index).
    Method(String, bool, Option<RhsType>),
    /// `p` flows into a concretely-annotated parameter of a known function
    /// (M4): the concrete type is identity-forced.
    Identity(TokenStream),
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

/// The return type of a stdlib method, for return-type inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MethodReturn {
    /// `-> String`
    Str,
    /// `-> Vec<String>`
    VecStr,
    /// `-> i64`
    I64,
    /// `-> bool`
    Bool,
    /// `-> (String, String, String)`
    TripleStr,
    /// `-> ()` (mutating, returns None in Python)
    Unit,
    /// pop(): the element type is unknown without deeper propagation.
    Unknown,
}

/// The M2 stdlib method table: Python method name → (trait bound, mutates
/// the receiver, return type). Every entry's trait method exists in
/// stdpython (PyStrOps/PyListOps/PyPop), and the codegen emits the
/// trait-method call for these names on any receiver.
pub const STDLIB_METHOD_TABLE: &[(&str, &str, bool, MethodReturn)] = &[
    // str methods (PyStrOps, non-mutating).
    ("upper", "PyStrOps", false, MethodReturn::Str),
    ("lower", "PyStrOps", false, MethodReturn::Str),
    ("strip", "PyStrOps", false, MethodReturn::Str),
    ("lstrip", "PyStrOps", false, MethodReturn::Str),
    ("rstrip", "PyStrOps", false, MethodReturn::Str),
    ("capitalize", "PyStrOps", false, MethodReturn::Str),
    ("title", "PyStrOps", false, MethodReturn::Str),
    ("splitlines", "PyStrOps", false, MethodReturn::VecStr),
    ("find", "PyStrOps", false, MethodReturn::I64),
    ("count", "PyStrOps", false, MethodReturn::I64),
    ("split", "PyStrOps", false, MethodReturn::VecStr),
    ("rsplit", "PyStrOps", false, MethodReturn::VecStr),
    ("partition", "PyStrOps", false, MethodReturn::TripleStr),
    ("rpartition", "PyStrOps", false, MethodReturn::TripleStr),
    ("zfill", "PyStrOps", false, MethodReturn::Str),
    ("ljust", "PyStrOps", false, MethodReturn::Str),
    ("rjust", "PyStrOps", false, MethodReturn::Str),
    // join is not listed: its argument needs a compound
    // IntoIterator<Item: AsRef<str>> bound, which M2 does not express. A
    // `p.join(...)` call on an unannotated parameter is a loud error. 

    // list methods (PyListOps / PyPop, mutating).
    ("insert", "PyListOps", true, MethodReturn::Unit),
    ("count", "PyListOps", false, MethodReturn::I64),
    ("pop", "PyPop", true, MethodReturn::Unknown),
];

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
    /// Parameters with stdlib-method requirements (M2): their method calls
    /// dispatch through the stdlib traits.
    pub method_params: HashSet<String>,
    /// Parameters with duck-typed user-method requirements (M3): param →
    /// the method names whose generated Has* trait returns Result.
    pub duck_methods_on_params: HashMap<String, HashSet<String>>,
    /// M5 definition-time warning: set when some parameter's bound set is
    /// satisfied by no known rython type (`p.upper()` + `p.pop()`). A
    /// well-formed definition in Python, so it never blocks conversion —
    /// it surfaces through the -W machinery and as a #[deprecated] note.
    pub definition_warning: Option<String>,
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

/// Pre-scan for M2 iteration: the names bound by `for <name> in <name>`
/// where the iterable resolves (through simple `a = p` aliases) to an
/// unannotated parameter. These become virtual parameters whose type
/// variable is the iteration's element type.
fn loop_element_names(body: &[Statement], unannotated: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut alias: HashMap<String, String> = HashMap::new();
    fn root<'a>(name: &'a str, alias: &'a HashMap<String, String>) -> Option<&'a str> {
        let mut cur = name;
        let mut seen = HashSet::new();
        while let Some(next) = alias.get(cur) {
            if !seen.insert(cur.to_string()) {
                return None;
            }
            cur = next;
        }
        Some(cur)
    }
    // Comprehension/generator-expression targets whose iterable resolves
    // to an unannotated parameter (`str(v) for v in version` — issue #116).
    fn scan_comprehensions<'a>(
        expr: &'a ExprType,
        unannotated: &HashSet<String>,
        alias: &'a HashMap<String, String>,
        out: &mut Vec<String>,
    ) {
        let (elt, generators): (&ExprType, &[crate::Comprehension]) = match expr {
            ExprType::ListComp(l) => (l.elt.as_ref(), &l.generators),
            ExprType::SetComp(s) => (s.elt.as_ref(), &s.generators),
            ExprType::DictComp(d) => (d.value.as_ref(), &d.generators),
            ExprType::GeneratorExp(g) => (g.elt.as_ref(), &g.generators),
            _ => return,
        };
        for g in generators {
            if let ExprType::Name(n) = &g.iter
                && let ExprType::Name(t) = &g.target
                && root(&n.id, alias).is_some_and(|r| unannotated.contains(r))
                && !out.contains(&t.id)
            {
                out.push(t.id.clone());
            }
        }
        scan_expr(elt, unannotated, alias, out);
    }
    fn scan_expr(
        expr: &ExprType,
        unannotated: &HashSet<String>,
        alias: &HashMap<String, String>,
        out: &mut Vec<String>,
    ) {
        match expr {
            ExprType::Call(c) => {
                for arg in &c.args {
                    scan_expr(arg, unannotated, alias, out);
                }
                for kw in &c.keywords {
                    scan_expr(&kw.value, unannotated, alias, out);
                }
            }
            ExprType::ListComp(_)
            | ExprType::SetComp(_)
            | ExprType::DictComp(_)
            | ExprType::GeneratorExp(_) => scan_comprehensions(expr, unannotated, alias, out),
            ExprType::IfExp(e) => {
                scan_expr(&e.body, unannotated, alias, out);
                scan_expr(&e.orelse, unannotated, alias, out);
            }
            _ => {}
        }
    }
    fn walk(
        stmts: &[Statement],
        unannotated: &HashSet<String>,
        alias: &mut HashMap<String, String>,
        out: &mut Vec<String>,
    ) {
        for stmt in stmts {
            match &stmt.statement {
                StatementType::Assign(a) => {
                    if let [ExprType::Name(target)] = a.targets.as_slice()
                        && let ExprType::Name(src) = &a.value
                        && (unannotated.contains(&src.id) || alias.contains_key(&src.id))
                    {
                        alias.insert(target.id.clone(), src.id.clone());
                    }
                    scan_expr(&a.value, unannotated, alias, out);
                }
                StatementType::For(s) => {
                    if let ExprType::Name(n) = &s.iter
                        && let ExprType::Name(t) = &s.target
                        && root(&n.id, alias).is_some_and(|r| unannotated.contains(r))
                        && !out.contains(&t.id)
                    {
                        out.push(t.id.clone());
                    }
                    walk(&s.body, unannotated, alias, out);
                    walk(&s.orelse, unannotated, alias, out);
                }
                StatementType::If(s) => {
                    walk(&s.body, unannotated, alias, out);
                    walk(&s.orelse, unannotated, alias, out);
                }
                StatementType::While(s) => {
                    walk(&s.body, unannotated, alias, out);
                    walk(&s.orelse, unannotated, alias, out);
                }
                StatementType::Try(t) => {
                    walk(&t.body, unannotated, alias, out);
                    for h in &t.handlers {
                        walk(&h.body, unannotated, alias, out);
                    }
                    walk(&t.orelse, unannotated, alias, out);
                    walk(&t.finalbody, unannotated, alias, out);
                }
                StatementType::With(w) => walk(&w.body, unannotated, alias, out),
                StatementType::AsyncWith(w) => walk(&w.body, unannotated, alias, out),
                StatementType::Return(Some(e)) => scan_expr(&e.value, unannotated, alias, out),
                StatementType::Expr(e) => scan_expr(&e.value, unannotated, alias, out),
                _ => {}
            }
        }
    }
    walk(body, unannotated, &mut alias, &mut out);
    out
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
    options: &crate::PythonOptions,
) -> Result<InferredSignature, String> {
    if params.is_empty() {
        return Ok(InferredSignature::default());
    }
    // M2 iteration: `for x in p` makes the loop variable a VIRTUAL
    // parameter whose type variable is the iterable's element type. The
    // pre-scan finds those targets (through `a = p` aliases) so they get
    // type variables and their body uses are collected.
    let fn_unannotated: HashSet<String> = params.iter().cloned().collect();
    let elements = loop_element_names(body, &fn_unannotated);
    let mut all_params: Vec<String> = params.to_vec();
    for e in &elements {
        if !all_params.contains(e) {
            all_params.push(e.clone());
        }
    }
    let unannotated: HashSet<String> = all_params.iter().cloned().collect();
    let current_fn = {
        // The function whose body we are collecting: the symbol-table
        // FunctionDef whose unannotated params match, if exactly one
        // module function has this param list (used for self-recursion).
        let mut fns = symbols
            .all_functions()
            .into_iter()
            .filter(|f| {
                let un: HashSet<String> = f
                    .args
                    .posonlyargs
                    .iter()
                    .chain(f.args.args.iter())
                    .chain(f.args.kwonlyargs.iter())
                    .filter(|p| p.arg != "self" && p.annotation.is_none())
                    .map(|p| p.arg.clone())
                    .collect();
                un == unannotated
            })
            .collect::<Vec<_>>();
        if fns.len() == 1 {
            fns.pop().map(|f| f.name)
        } else {
            None
        }
    };

    let mut collector = Collector {
        unannotated: &unannotated,
        name_types,
        symbols,
        options,
        reqs: HashMap::new(),
        alias: HashMap::new(),
        returns: Vec::new(),
        reassigned: HashSet::new(),
        duck_returns: HashMap::new(),
        duck_method_calls: HashMap::new(),
        error: None,
        current_fn: current_fn.clone(),
        callee_cache: HashMap::new(),
        visiting: HashSet::new(),
        return_visiting: HashSet::new(),
        loop_elements: HashMap::new(),
    };
    if let Some(f) = &current_fn {
        collector.visiting.insert(f.clone());
    }
    collector.walk(body);
    if let Some(err) = &collector.error {
        return Err(err.clone());
    }

    // Element names discovered during the walk (an alias-iterated
    // parameter, or an Iterate requirement adopted from a callee) also
    // need type variables.
    for (elt, _param) in &collector.loop_elements {
        if !all_params.contains(elt) {
            all_params.push(elt.clone());
        }
    }
    for reqs in collector.reqs.values() {
        for req in reqs {
            if let ParamReq::Iterate(elt) = req
                && !all_params.contains(elt)
            {
                all_params.push(elt.clone());
            }
        }
    }

    // A reassigned parameter cannot keep a single inferred type.
    for name in &all_params {
        if collector.reassigned.contains(name) {
            return Err(format!(
                "parameter `{name}` is assigned to inside the function; an inferred \
                 generic type cannot change. Annotate `{name}` with its type"
            ));
        }
    }

    // Identity-forced parameters first (M4): a parameter that flows into a
    // concretely-annotated callee parameter takes that concrete type; it
    // gets no type variable and no bounds (the concrete type's trait impls
    // are checked at the call site). Conflicting identity forces are loud.
    let mut identity_types: HashMap<String, TokenStream> = HashMap::new();
    for name in &all_params {
        let Some(reqs) = collector.reqs.get(name) else { continue };
        for req in reqs {
            if let ParamReq::Identity(ty) = req {
                match identity_types.get(name) {
                    Some(prev) if prev.to_string() != ty.to_string() => {
                        return Err(format!(
                            "parameter `{name}` is forced to two different concrete \
                             types ({} and {}); the call sites disagree — annotate \
                             `{name}` explicitly",
                            prev, ty
                        ));
                    }
                    Some(_) => {}
                    None => {
                        identity_types.insert(name.clone(), ty.clone());
                    }
                }
            }
        }
    }

    // One type variable per GENERIC parameter, in declaration order (single
    // param: `T`, otherwise `A`, `B`, ...).
    let mut generic_params: Vec<&String> = all_params
        .iter()
        .filter(|p| !identity_types.contains_key(*p))
        .collect();
    // A loop element of an identity-forced parameter is driven by the
    // concrete type — drop its type variable (a declared-but-unconstrained
    // type parameter would not compile).
    let mut forced_elements: HashSet<String> = HashSet::new();
    for (elt, param) in &collector.loop_elements {
        if identity_types.contains_key(param) {
            forced_elements.insert(elt.clone());
        }
    }
    for (param, reqs) in &collector.reqs {
        if identity_types.contains_key(param) {
            for req in reqs {
                if let ParamReq::Iterate(elt) = req {
                    forced_elements.insert(elt.clone());
                }
            }
        }
    }
    generic_params.retain(|p| !forced_elements.contains(*p));
    let mut tv_names: HashMap<String, String> = HashMap::new();
    let mut type_params = Vec::new();
    let mut param_types = HashMap::new();
    if generic_params.len() == 1 {
        tv_names.insert(generic_params[0].clone(), "T".to_string());
        type_params.push(quote!(T));
    } else {
        for (i, name) in generic_params.iter().enumerate() {
            let tv = format!("{}", (b'A' + i as u8) as char);
            tv_names.insert((*name).clone(), tv.clone());
            let ident = quote::format_ident!("{}", tv);
            type_params.push(quote!(#ident));
        }
    }
    for name in &all_params {
        if let Some(ty) = identity_types.get(name) {
            param_types.insert(name.clone(), ty.clone());
        } else if let Some(tv) = tv_names.get(name) {
            let ident = quote::format_ident!("{}", tv);
            param_types.insert(name.clone(), quote!(#ident));
        }
    }

    // Resolve each GENERIC parameter's requirement set into bounds.
    let mut where_bounds = Vec::new();
    for name in &generic_params {
        let tv = quote::format_ident!("{}", tv_names.get(*name).unwrap());
        let reqs = collector.reqs.get(*name).cloned().unwrap_or_default();
        let mut seen = HashSet::new();
        for req in &reqs {
            if let ParamReq::Untranslatable(what) = req {
                return Err(format!(
                    "parameter `{name}`: {what}. Annotate `{name}` with a concrete \
                     type, or use it only through operations the transpiler can \
                     bound (issue #109: parameter type inference, M1)"
                ));
            }
            // Identity is impossible here (identity params aren't generic).
            if matches!(req, ParamReq::Identity(_)) {
                continue;
            }
            let bound = match req {
                ParamReq::Op(trait_name, rhs) => {
                    let t = quote::format_ident!("{}", trait_name);
                    // A self-referential op (`x + x`, `x + repeat(x, ...)`)
                    // feeds its result back as the parameter itself — the
                    // recursion fixpoint — so its Output must BE the
                    // parameter's type (true for every stdpython scalar,
                    // string, and list).
                    let is_self = matches!(rhs, RhsType::Same)
                        || matches!(rhs, RhsType::Param(p) if p == *name);
                    let rhs = render_rhs(rhs, &param_types, &quote!(#tv))?;
                    if is_self {
                        quote!(#tv: #t<#rhs, Output = #tv>)
                    } else {
                        quote!(#tv: #t<#rhs>)
                    }
                }
                ParamReq::OpOutput(trait_name, rhs) => {
                    let t = quote::format_ident!("{}", trait_name);
                    let rhs = render_rhs(rhs, &param_types, &quote!(#tv))?;
                    quote!(#tv: #t<#rhs, Output = #tv>)
                }
                ParamReq::CmpCond(trait_name, rhs) => {
                    let t = quote::format_ident!("{}", trait_name);
                    let rhs = render_rhs(rhs, &param_types, &quote!(#tv))?;
                    quote!(#tv: #t<#rhs, Output = bool>)
                }
                ParamReq::Cmp(trait_name, rhs) => {
                    let t = quote::format_ident!("{}", trait_name);
                    let rhs = render_rhs(rhs, &param_types, &quote!(#tv))?;
                    quote!(#tv: #t<#rhs>)
                }
                ParamReq::PyFromInt => quote!(#tv: PyFromInt),
                ParamReq::Iterate(elt) => {
                    let elt_tv = tv_names.get(elt).ok_or_else(|| {
                        format!("internal: loop element `{elt}` has no type variable")
                    })?;
                    let elt_ident = quote::format_ident!("{}", elt_tv);
                    quote!(#tv: IntoIterator<Item = #elt_ident>)
                }
                ParamReq::AsRefStr => quote!(#tv: AsRef<str>),
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
                    let idx = render_rhs(idx, &param_types, &quote!(#tv))?;
                    quote!(#tv: PyIndex<#idx>)
                }
                ParamReq::SetIndex(idx, val) => {
                    let idx = render_rhs(idx, &param_types, &quote!(#tv))?;
                    let val = render_rhs(val, &param_types, &quote!(#tv))?;
                    quote!(#tv: PySetIndex<#idx, #val>)
                }
                ParamReq::Contains(item) => {
                    let item = render_rhs(item, &param_types, &quote!(#tv))?;
                    quote!(#tv: PyContains<#item>)
                }
                ParamReq::Method(trait_name, _, rhs) => {
                    let t = quote::format_ident!("{}", trait_name);
                    match rhs {
                        Some(rhs) => {
                            let rhs = render_rhs(rhs, &param_types, &quote!(#tv))?;
                            quote!(#tv: #t<#rhs>)
                        }
                        None => quote!(#tv: #t),
                    }
                }
                ParamReq::Identity(_) => unreachable!("skipped above"),
                ParamReq::Untranslatable(_) => unreachable!("handled above"),
            };
            if seen.insert(bound.to_string()) {
                where_bounds.push(bound);
            }
        }
        // The reuse-clone rule: a generic parameter is not known Copy, so a
        // parameter read more than once needs `T: Clone` — the rule itself
        // is a use, so the bound stays minimal.
        if use_counts.get(*name).copied().unwrap_or(0) > 1 {
            where_bounds.push(quote!(#tv: Clone));
        }
    }

    // Parameters with stdlib-method uses, and the duck-typed user-method
    // calls (M3) that must thread `?` at their call sites.
    let mut method_params = HashSet::new();
    let mut duck_methods_on_params: HashMap<String, HashSet<String>> = HashMap::new();
    for name in params {
        let Some(reqs) = collector.reqs.get(name) else { continue };
        if reqs.iter().any(|r| matches!(r, ParamReq::Method(..))) {
            method_params.insert(name.clone());
        }
        if let Some(methods) = collector.duck_method_calls.get(name) {
            duck_methods_on_params.insert(name.clone(), methods.clone());
        }
    }

    // Return type: every return value must unify to one type expression.
    let return_type = if collector.returns.is_empty() {
        None
    } else {
        let mut inferred: Option<TokenStream> = None;
        let returns: Vec<ExprType> = collector.returns.clone();
        for ret in &returns {
            let ty = return_type_of(ret, &mut collector, &param_types)?;
            match &inferred {
                None => inferred = Some(ty),
                Some(prev) if prev.to_string() == ty.to_string() => {}
                // Recursive fixpoint (M4): `<X as PyOp<X>>::Output` unifies
                // with X — the recursive call returns the parameter's type
                // (int/str/list all satisfy PyOp<Self>::Output == Self).
                Some(prev) if unifies_with_recursion(prev, &ty) => {}
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

    // M5 definition-time warning: a bound set no known type satisfies
    // (`p.upper()` + `p.pop()` → PyStrOps + PyPop) is a well-formed
    // Python definition — it never blocks conversion, but it is reported
    // through the -W machinery and as a #[deprecated] note.
    let definition_warning = {
        let mut warning = None;
        for name in &generic_params {
            let reqs = collector.reqs.get(*name).cloned().unwrap_or_default();
            if !reqs.is_empty() && !definitionally_satisfiable(&reqs) {
                let traits: Vec<&str> = reqs.iter().map(trait_name_of).collect();
                warning = Some(format!(
                    "parameter `{name}`'s inferred bounds ({}) are satisfied by no \
                     known rython type; every call site with a statically-known \
                     argument type will fail — annotate the parameter (issue #109, M5)",
                    traits.join(" + ")
                ));
                break;
            }
        }
        warning
    };

    Ok(InferredSignature {
        type_params,
        where_bounds,
        param_types,
        return_type,
        method_params,
        definition_warning,
        duck_methods_on_params,
    })
}

/// Render an operand's type for a where-bound: the parameter's variable
/// (or identity-forced concrete type), the concrete type, or Self for
/// same-param operands.
fn render_rhs(
    rhs: &RhsType,
    param_types: &HashMap<String, TokenStream>,
    same: &TokenStream,
) -> Result<TokenStream, String> {
    Ok(match rhs {
        RhsType::Concrete(t) => t.clone(),
        RhsType::Param(name) => match param_types.get(name) {
            Some(ty) => ty.clone(),
            None => {
                return Err(format!(
                    "internal: parameter `{name}` used as an operand but has no \
                     type"
                ))
            }
        },
        RhsType::Same => same.clone(),
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
    collector: &mut Collector,
    param_types: &HashMap<String, TokenStream>,
) -> Result<TokenStream, String> {
    let param_tv = |name: &str| -> Option<TokenStream> {
        let p = if param_types.contains_key(name) {
            Some(name.to_string())
        } else {
            collector.alias.get(name).cloned()
        };
        p.and_then(|p| param_types.get(&p)).cloned()
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
            // A call to a user function (M4): its return annotation, or —
            // via FlowsTo — the callee's return type in the caller's terms.
            // Self-recursion resolves to the returned parameter's argument
            // type (the fixpoint: repeat returns x).
            if let ExprType::Name(f) = c.func.as_ref() {
                if let Some(crate::SymbolTableNode::FunctionDef(callee)) =
                    collector.symbols.get(&f.id)
                {
                    if let Some(ann) = callee
                        .returns
                        .as_deref()
                        .and_then(crate::python_annotation_to_rust_type)
                    {
                        return Ok(ann);
                    }
                    if collector.current_fn.as_deref() == Some(callee.name.as_str()) {
                        if let Some(param_index) = callee_returned_param(callee) {
                            // The fixpoint: the recursive call returns the
                            // same thing the function returns on its base
                            // path — the RETURNED parameter's own type
                            // variable (repeat: x → A; fibonacci: n → N).
                            // The decremented ARGUMENT's type is irrelevant
                            // to the result (it is bound separately by
                            // `N: PySub<i64, Output = N>`).
                            if let Some(param_name) = callee
                                .args
                                .posonlyargs
                                .iter()
                                .chain(callee.args.args.iter())
                                .nth(param_index)
                            {
                                if let Some(tv) = param_types.get(&param_name.arg) {
                                    return Ok(tv.clone());
                                }
                            }
                        }
                        return Err(format!(
                            "recursive call to `{}` does not return one of its \
                             parameters; annotate `{}`'s return type (issue #109, M4)",
                            callee.name, callee.name
                        ));
                    }
                    return callee_return_type(callee, &c.args, collector, param_types);
                }
            }
            // A method call on a parameter: the table's return type.
            if let ExprType::Attribute(a) = c.func.as_ref()
                && let ExprType::Name(n) = a.value.as_ref()
                && (param_types.contains_key(&n.id) || collector.alias.contains_key(&n.id))
            {
                // M3: a duck-typed user method's return comes from the
                // unified class signature.
                if let Some(ret) = collector.duck_returns.get(&a.attr) {
                    return Ok(ret.clone());
                }
                if let Some((_, _, _, ret)) =
                    STDLIB_METHOD_TABLE.iter().find(|(m, ..)| *m == a.attr)
                {
                    // pop() returns the element: `<T as PyPop<Idx>>::Output`.
                    if *ret == MethodReturn::Unknown && a.attr == "pop" {
                        let tv = param_tv_of(&n.id, collector, param_types);
                        let idx = match c.args.first() {
                            Some(arg) => operand_type(arg, collector, param_types)?,
                            None => quote!(i64),
                        };
                        return Ok(quote!(<#tv as PyPop<#idx>>::Output));
                    }
                    return Ok(match ret {
                        MethodReturn::Str => quote!(String),
                        MethodReturn::VecStr => quote!(Vec<String>),
                        MethodReturn::I64 => quote!(i64),
                        MethodReturn::Bool => quote!(bool),
                        MethodReturn::TripleStr => quote!((String, String, String)),
                        MethodReturn::Unit => quote!(()),
                        MethodReturn::Unknown => {
                            return Err(
                                "cannot infer the type of this method call in a return; \
                                 annotate the function's return type (issue #109)"
                                    .to_string(),
                            )
                        }
                    });
                }
            }
            // `"sep".join(...)` on a string literal (or a String/&str
            // local) returns an owned String — the method table omits join
            // (its bound needs a compound IntoIterator), but the concrete
            // receiver's return is a plain String (issue #116).
            if let ExprType::Attribute(a) = c.func.as_ref()
                && a.attr == "join"
                && (matches!(
                    a.value.as_ref(),
                    ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_)))
                ) || matches!(
                    a.value.as_ref(),
                    ExprType::Name(n)
                        if matches!(
                            collector.name_types.get(&n.id),
                            Some(TypeInfo::String | TypeInfo::StrRef)
                        )
                ))
            {
                return Ok(quote!(String));
            }
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
            // Operands are typed with return_type_of (not operand_type):
            // an operand may itself be a user-function call whose return
            // type flows here (M4, e.g. `x + repeat(x, n - 1)`).
            let left = return_type_of(&b.left, collector, param_types)?;
            let right = return_type_of(&b.right, collector, param_types)?;
            Ok(quote!(<#left as #t<#right>>::Output))
        }
        ExprType::ListComp(l) => {
            // `[f(x) for x in p]` produces a Vec of the element expression's
            // type (issue #116).
            let elt_ty = return_type_of(&l.elt, collector, param_types)?;
            Ok(quote!(Vec<#elt_ty>))
        }
        ExprType::IfExp(e) => {
            // `x if cond else y`: both branches must unify (recursion
            // fixpoint included: repeat's `x` vs `x + repeat(...)`).
            let body_ty = return_type_of(&e.body, collector, param_types)?;
            let orelse_ty = return_type_of(&e.orelse, collector, param_types)?;
            if body_ty.to_string() == orelse_ty.to_string() {
                Ok(body_ty)
            } else if unifies_with_recursion(&body_ty, &orelse_ty) {
                Ok(body_ty)
            } else {
                Err(
                    "if-expression branches have different types; annotate the \
                     function's return type (issue #109)"
                        .to_string(),
                )
            }
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
            let left = return_type_of(&c.left, collector, param_types)?;
            // An integer-literal comparator against a parameter compares
            // with the parameter's OWN type (`n <= 0` → `<B as PyLe<B>>`),
            // matching the `B: PyLe<Self> + From<i64>` bounds.
            let right = match c.comparators.first() {
                Some(r) => {
                    if matches!(
                        c.left.as_ref(),
                        ExprType::Name(n)
                            if param_types.contains_key(&n.id)
                                || collector.alias.contains_key(&n.id)
                    ) && matches!(
                        r,
                        ExprType::Constant(cn)
                            if matches!(&cn.0, Some(litrs::Literal::Integer(_)))
                    ) {
                        left.clone()
                    } else {
                        return_type_of(r, collector, param_types)?
                    }
                }
                None => quote!(Self),
            };
            Ok(quote!(<#left as #t<#right>>::Output))
        }
        _ => Err(err()),
    }
}

/// The type tokens for a parameter name (or its alias): its type variable
/// or identity-forced concrete type.
fn param_tv_of(
    name: &str,
    collector: &Collector,
    param_types: &HashMap<String, TokenStream>,
) -> TokenStream {
    let p = if param_types.contains_key(name) {
        Some(name.to_string())
    } else {
        collector.alias.get(name).cloned()
    };
    match p.and_then(|p| param_types.get(&p)) {
        Some(ty) => ty.clone(),
        None => quote!(__rython_unknown__),
    }
}

/// The type of an operand inside a return expression: a parameter's
/// variable or a concrete type.
fn operand_type(
    expr: &ExprType,
    collector: &Collector,
    param_types: &HashMap<String, TokenStream>,
) -> Result<TokenStream, String> {
    let err = || {
        "the operand's type cannot be inferred; annotate the function's return \
         type (issue #109, M1)"
            .to_string()
    };
    Ok(match expr {
        ExprType::Name(n) => {
            // A parameter (directly or via an alias).
            let p = if param_types.contains_key(&n.id) {
                Some(n.id.clone())
            } else {
                collector.alias.get(&n.id).cloned()
            };
            if let Some(p) = p {
                if let Some(ty) = param_types.get(&p) {
                    ty.clone()
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
            } else if let ExprType::Attribute(a) = c.func.as_ref() {
                // A method call on a parameter: its table return type
                // (`s.upper() + str(n)` needs String on the left); a duck
                // method's comes from the unified class signature.
                if let ExprType::Name(n) = a.value.as_ref()
                    && (param_types.contains_key(&n.id) || collector.alias.contains_key(&n.id))
                {
                    if let Some(ret) = collector.duck_returns.get(&a.attr) {
                        return Ok(ret.clone());
                    }
                }
                if let ExprType::Name(n) = a.value.as_ref()
                    && (param_types.contains_key(&n.id) || collector.alias.contains_key(&n.id))
                    && let Some((_, _, _, ret)) = STDLIB_METHOD_TABLE
                        .iter()
                        .find(|(m, ..)| *m == a.attr)
                {
                    match ret {
                        MethodReturn::Str => quote!(String),
                        MethodReturn::VecStr => quote!(Vec<String>),
                        MethodReturn::I64 => quote!(i64),
                        MethodReturn::Bool => quote!(bool),
                        MethodReturn::TripleStr => quote!((String, String, String)),
                        MethodReturn::Unit => quote!(()),
                        MethodReturn::Unknown => return Err(err()),
                    }
                } else {
                    return Err(err());
                }
            } else {
                return Err(err());
            }
        }
        _ => return Err(err()),
    })
}

/// Suggest the closest known stdlib methods for an unknown one, for the
/// loud-error message ("did you mean ...?").
fn nearest_methods(method: &str) -> String {
    let known: Vec<&str> = STDLIB_METHOD_TABLE.iter().map(|(m, ..)| *m).collect();
    let mut scored: Vec<(usize, &str)> = known
        .iter()
        .map(|m| (levenshtein(method, m), *m))
        .collect();
    scored.sort_by_key(|(d, _)| *d);
    let close: Vec<&str> = scored
        .iter()
        .take(3)
        .filter(|(d, _)| *d <= 3)
        .map(|(_, m)| *m)
        .collect();
    if close.is_empty() {
        String::new()
    } else {
        format!(" (did you mean {})", close.join(", "))
    }
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// The unified duck-typing signature of one class's method: (param names,
/// param type strings, param name idents), all annotated.
fn class_method_signature(
    class: &crate::ClassDef,
    method: &str,
) -> Result<(Vec<String>, Vec<String>, Vec<String>), String> {
    let m = class
        .methods()
        .find(|m| m.name == method)
        .ok_or_else(|| format!("internal: `{}` has no method `{}`", class.name, method))?;
    let mut names = Vec::new();
    let mut types = Vec::new();
    let mut idents = Vec::new();
    for p in m.args.posonlyargs.iter().chain(m.args.args.iter()) {
        if p.arg == "self" {
            continue;
        }
        let ann = p.annotation.as_deref().ok_or_else(|| {
            format!(
                "duck typing `{method}`: parameter `{}` of `{}.{}` is unannotated; \
                 duck typing needs fully annotated signatures",
                p.arg, class.name, method
            )
        })?;
        let t = crate::python_annotation_to_rust_type(ann).ok_or_else(|| {
            format!(
                "duck typing `{method}`: unsupported annotation on `{}.{}`",
                class.name, method
            )
        })?;
        names.push(p.arg.clone());
        types.push(t.to_string());
        idents.push(crate::safe_ident(&p.arg).to_string());
    }
    let _ = class_method_return(class, method)?;
    Ok((names, types, idents))
}

/// The unified return type of one class's method.
fn class_method_return(class: &crate::ClassDef, method: &str) -> Result<TokenStream, String> {
    let m = class
        .methods()
        .find(|m| m.name == method)
        .ok_or_else(|| format!("internal: `{}` has no method `{}`", class.name, method))?;
    m.returns
        .as_deref()
        .and_then(crate::python_annotation_to_rust_type)
        .ok_or_else(|| {
            format!(
                "duck typing `{method}`: `{}.{method}` needs a return annotation so its \
                 trait can be generated",
                class.name
            )
        })
}

/// The return type of a call to a NON-recursive user function, expressed in
/// the CALLER's terms: the callee's return expressions are re-analyzed with
/// the callee's parameters mapped to the caller's argument types (FlowsTo).
/// Mutual recursion without return annotations is a loud error.
fn callee_return_type(
    callee: &crate::FunctionDef,
    args: &[ExprType],
    collector: &mut Collector,
    param_types: &HashMap<String, TokenStream>,
) -> Result<TokenStream, String> {
    if collector.return_visiting.contains(&callee.name) {
        return Err(format!(
            "mutually recursive functions without return annotations are not \
             inferred yet (issue #109, M4); annotate a return type in the cycle"
        ));
    }
    let callee_params: Vec<&str> = callee
        .args
        .posonlyargs
        .iter()
        .chain(callee.args.args.iter())
        .map(|p| p.arg.as_str())
        .collect();
    let mut callee_map: HashMap<String, TokenStream> = HashMap::new();
    for (i, name) in callee_params.iter().enumerate() {
        let Some(arg) = args.get(i) else { continue };
        callee_map.insert(name.to_string(), return_type_of(arg, collector, param_types)?);
    }
    let mut returns = Vec::new();
    collect_return_exprs(&callee.body, &mut returns);
    if returns.is_empty() {
        return Err(format!(
            "cannot infer the return type of `{}`: it has no return statements; \
             annotate the callee's return type (issue #109, M4)",
            callee.name
        ));
    }
    let saved_fn = collector.current_fn.take();
    collector.current_fn = Some(callee.name.clone());
    collector.return_visiting.insert(callee.name.clone());
    // The callee's return expressions are analyzed in the CALLEE's own
    // scope (its name_types, params mapped to the caller's argument
    // types): a return of a local (`return result`) resolves through the
    // callee's analysis, while a return of a parameter stays in the
    // caller's terms.
    let fn_unannotated: HashSet<String> = callee
        .args
        .posonlyargs
        .iter()
        .chain(callee.args.args.iter())
        .chain(callee.args.kwonlyargs.iter())
        .filter(|p| p.arg != "self" && p.annotation.is_none())
        .map(|p| p.arg.clone())
        .collect();
    let mut inner_unannotated = fn_unannotated.clone();
    for e in loop_element_names(&callee.body, &fn_unannotated) {
        inner_unannotated.insert(e);
    }
    let info = crate::analyze_function_types(&callee.body);
    let result = {
        let mut inner = Collector {
            unannotated: &inner_unannotated,
            name_types: &info.name_types,
            symbols: collector.symbols,
            options: collector.options,
            reqs: HashMap::new(),
            alias: HashMap::new(),
            returns: Vec::new(),
            reassigned: HashSet::new(),
            duck_returns: HashMap::new(),
            duck_method_calls: HashMap::new(),
            error: None,
            current_fn: Some(callee.name.clone()),
            callee_cache: HashMap::new(),
            visiting: HashSet::new(),
            return_visiting: collector.return_visiting.clone(),
            loop_elements: HashMap::new(),
        };
        let mut inferred: Option<TokenStream> = None;
        for ret in &returns {
            let ty = return_type_of(ret, &mut inner, &callee_map)?;
            match &inferred {
                None => inferred = Some(ty),
                Some(prev) if prev.to_string() == ty.to_string() => {}
                Some(prev) if unifies_with_recursion(prev, &ty) => {}
                _ => {
                    return Err(format!(
                        "`{}`'s return statements have different types; annotate \
                         its return type (issue #109, M4)",
                        callee.name
                    ))
                }
            }
        }
        Ok(inferred.unwrap())
    };
    collector.current_fn = saved_fn;
    collector.return_visiting.remove(&callee.name);
    result
}

/// Collect every return expression in a body (walks nested statements).
fn collect_return_exprs(body: &[Statement], out: &mut Vec<ExprType>) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Return(Some(e)) => out.push(e.value.clone()),
            StatementType::If(s) => {
                collect_return_exprs(&s.body, out);
                collect_return_exprs(&s.orelse, out);
            }
            StatementType::For(s) => {
                collect_return_exprs(&s.body, out);
                collect_return_exprs(&s.orelse, out);
            }
            StatementType::While(s) => {
                collect_return_exprs(&s.body, out);
                collect_return_exprs(&s.orelse, out);
            }
            StatementType::Try(t) => {
                collect_return_exprs(&t.body, out);
                for h in &t.handlers {
                    collect_return_exprs(&h.body, out);
                }
                collect_return_exprs(&t.orelse, out);
                collect_return_exprs(&t.finalbody, out);
            }
            _ => {}
        }
    }
}

/// The leaf values of an expression: itself, or the branches of an
/// if-expression (recursively) — the branches a returned expression can
/// actually evaluate to.
fn collect_expr_branches<'e>(expr: &'e ExprType, out: &mut Vec<&'e ExprType>) {
    match expr {
        ExprType::IfExp(e) => {
            collect_expr_branches(&e.body, out);
            collect_expr_branches(&e.orelse, out);
        }
        other => out.push(other),
    }
}

// ---------------------------------------------------------------------------
// M5: call-site satisfiability. A call whose arguments are statically known
// (literals, typed locals) is checked against the callee's inferred bounds
// at CONVERSION time — an unsatisfiable call is a loud error naming the
// Python line, never a rustc surprise at build time. The table mirrors
// stdpython's actual trait impls (including Rust std's missing int/float
// cross-PartialOrd/PartialEq), and is deliberately permissive for types the
// table is unsure about (no false positives).
// ---------------------------------------------------------------------------

/// Python-facing type names for error messages.
fn type_display(ty: &TypeInfo) -> String {
    match ty {
        TypeInfo::Int => "int".to_string(),
        TypeInfo::Float => "float".to_string(),
        TypeInfo::Bool => "bool".to_string(),
        TypeInfo::StrRef | TypeInfo::String => "str".to_string(),
        TypeInfo::Bytes => "bytes".to_string(),
        TypeInfo::Vec(_) => "list".to_string(),
        TypeInfo::Dict(..) => "dict".to_string(),
        TypeInfo::Tuple(_) => "tuple".to_string(),
        TypeInfo::Option(_) => "optional".to_string(),
        TypeInfo::Range => "range".to_string(),
        TypeInfo::NdArray => "array".to_string(),
        TypeInfo::Class(c) => c.clone(),
        TypeInfo::Borrowed(_) => "borrowed".to_string(),
        TypeInfo::PyObject => "unknown".to_string(),
    }
}

/// The trait name of a requirement, for the definition-time warning text.
fn trait_name_of(req: &ParamReq) -> &str {
    match req {
        ParamReq::Op(t, _) | ParamReq::OpOutput(t, _) | ParamReq::Cmp(t, _)
        | ParamReq::CmpCond(t, _) | ParamReq::Conversion(t) => t,
        ParamReq::Truthy => "Truthy",
        ParamReq::Len => "Len",
        ParamReq::Display => "PyDisplay",
        ParamReq::Repr => "PyRepr",
        ParamReq::Hash => "PyHash",
        ParamReq::IsNone => "PyIsNone",
        ParamReq::Index(_) => "PyIndex",
        ParamReq::SetIndex(..) => "PySetIndex",
        ParamReq::Contains(_) => "PyContains",
        ParamReq::Method(t, _, _) => t.as_str(),
        ParamReq::PyFromInt => "PyFromInt",
        ParamReq::Iterate(_) => "IntoIterator",
        ParamReq::AsRefStr => "AsRef<str>",
        ParamReq::Identity(_) | ParamReq::Untranslatable(_) => "?",
    }
}

/// Whether SOME known rython type could satisfy the parameter's whole
/// bound set (M5 definition-time warning). Only the types whose stdpython
/// impl sets are fully enumerated are considered — i64/f64/bool/String/
/// Vec/Dict — so the check only fires on clear contradictions. A
/// duck-typed Has* bound makes the set satisfiable by a user class.
fn definitionally_satisfiable(reqs: &[ParamReq]) -> bool {
    if reqs
        .iter()
        .any(|r| matches!(r, ParamReq::Method(t, _, _) if t.starts_with("Has")))
    {
        return true;
    }
    let candidates = [
        TypeInfo::Int,
        TypeInfo::Float,
        TypeInfo::Bool,
        TypeInfo::String,
        TypeInfo::StrRef,
        TypeInfo::Vec(Box::new(TypeInfo::Int)),
        TypeInfo::Vec(Box::new(TypeInfo::String)),
        TypeInfo::Dict(Box::new(TypeInfo::String), Box::new(TypeInfo::Int)),
    ];
    candidates
        .iter()
        .any(|t| reqs.iter().all(|r| definition_req_satisfied(r, t)))
}

/// Whether the candidate type satisfies one requirement at DEFINITION
/// time: a parameter rhs instantiates as the candidate itself, a concrete
/// rhs as its type, and an unknown rhs skips the check.
fn definition_req_satisfied(req: &ParamReq, t: &TypeInfo) -> bool {
    let with_rhs = |rhs: &RhsType| -> Option<TypeInfo> {
        match rhs {
            RhsType::Concrete(tokens) => concrete_tokens_to_typeinfo(tokens),
            RhsType::Param(_) | RhsType::Same => Some(t.clone()),
            RhsType::Unknown => None,
        }
    };
    let ok = |trait_name: &str, rhs: Option<&TypeInfo>| match rhs {
        Some(r) => type_satisfies(t, trait_name, Some(r)),
        None => type_satisfies(t, trait_name, None),
    };
    match req {
        ParamReq::Op(tn, rhs) | ParamReq::OpOutput(tn, rhs) => {
            with_rhs(rhs).map_or(true, |r| ok(tn, Some(&r)))
        }
        ParamReq::Cmp(tn, rhs) | ParamReq::CmpCond(tn, rhs) => {
            with_rhs(rhs).map_or(true, |r| ok(tn, Some(&r)))
        }
        ParamReq::Conversion(tn) => type_satisfies(t, tn, None),
        ParamReq::Truthy => type_satisfies(t, "Truthy", None),
        ParamReq::Len => type_satisfies(t, "Len", None),
        ParamReq::Display | ParamReq::Repr | ParamReq::IsNone => true,
        ParamReq::Hash => type_satisfies(t, "PyHash", None),
        ParamReq::Index(idx) => with_rhs(idx).map_or(true, |r| ok("PyIndex", Some(&r))),
        ParamReq::SetIndex(idx, _) => with_rhs(idx).map_or(true, |r| ok("PySetIndex", Some(&r))),
        ParamReq::Contains(item) => with_rhs(item).map_or(true, |r| ok("PyContains", Some(&r))),
        ParamReq::Method(tn, _, rhs) => {
            let tn = tn.as_str();
            match rhs {
                Some(rhs) => with_rhs(rhs).map_or(true, |r| ok(tn, Some(&r))),
                None => type_satisfies(t, tn, None),
            }
        }
        ParamReq::PyFromInt => type_satisfies(t, "PyFromInt", None),
        ParamReq::Iterate(_) => type_satisfies(t, "IntoIterator", None),
        ParamReq::AsRefStr => type_satisfies(t, "AsRef<str>", None),
        ParamReq::Identity(_) | ParamReq::Untranslatable(_) => true,
    }
}

/// Whether a value of type `ty` satisfies the trait bound the inference
/// emits for an unannotated parameter. `rhs` is the other operand's type
/// (None for unary bounds). Permissive (true) for types whose stdpython
/// impl set the table does not enumerate — the check only ever rejects
/// clear mismatches.
fn type_satisfies(ty: &TypeInfo, trait_name: &str, rhs: Option<&TypeInfo>) -> bool {
    let uncertain = |t: &TypeInfo| {
        matches!(
            t,
            TypeInfo::PyObject
                | TypeInfo::NdArray
                | TypeInfo::Class(_)
                | TypeInfo::Tuple(_)
                | TypeInfo::Option(_)
                | TypeInfo::Range
                | TypeInfo::Bytes
                | TypeInfo::Borrowed(_)
        )
    };
    if uncertain(ty) {
        return true;
    }
    match trait_name {
        // Operators: numeric promotion like stdpython's numeric_add /
        // numeric_sub_mul; strings and lists concatenate with themselves.
        "PyAdd" | "PySub" | "PyMul" => match (ty, rhs) {
            (TypeInfo::Int, Some(TypeInfo::Int | TypeInfo::Float)) => true,
            (TypeInfo::Float, Some(TypeInfo::Int | TypeInfo::Float)) => true,
            (TypeInfo::String | TypeInfo::StrRef, Some(TypeInfo::String | TypeInfo::StrRef)) => {
                trait_name == "PyAdd"
            }
            (TypeInfo::Vec(_), Some(TypeInfo::Vec(_))) => trait_name == "PyAdd",
            _ => false,
        },
        // Comparisons: Rust std's PartialEq/PartialOrd blankets are
        // same-type only — there is NO int/float cross comparison in std
        // (which is why literals convert via PyFromInt instead).
        "PyEq" | "PyNe" | "PyLt" | "PyLe" | "PyGt" | "PyGe" => match (ty, rhs) {
            (TypeInfo::Int, Some(TypeInfo::Int)) => true,
            (TypeInfo::Float, Some(TypeInfo::Float)) => true,
            (TypeInfo::Bool, Some(TypeInfo::Bool)) => true,
            (TypeInfo::String | TypeInfo::StrRef, Some(TypeInfo::String | TypeInfo::StrRef)) => {
                true
            }
            (TypeInfo::Vec(_), Some(TypeInfo::Vec(_))) => true,
            (TypeInfo::Dict(..), Some(TypeInfo::Dict(..))) => {
                matches!(trait_name, "PyEq" | "PyNe")
            }
            _ => false,
        },
        // int()/float()/bool() accept strings too (Python parses them),
        // mirroring stdpython's PyInt for &str/String, PyFloat for
        // &str/String, PyBool for &str/String.
        "PyInt" => matches!(
            ty,
            TypeInfo::Int | TypeInfo::Bool | TypeInfo::Float | TypeInfo::String | TypeInfo::StrRef
        ),
        "PyFloat" => matches!(
            ty,
            TypeInfo::Int | TypeInfo::Float | TypeInfo::String | TypeInfo::StrRef
        ),
        "PyBool" => matches!(
            ty,
            TypeInfo::Int
                | TypeInfo::Float
                | TypeInfo::Bool
                | TypeInfo::String
                | TypeInfo::StrRef
        ),
        "PyToString" | "PyDisplay" | "PyRepr" => true,
        "PyAbs" => matches!(ty, TypeInfo::Int | TypeInfo::Float),
        "Truthy" => true,
        "Len" => matches!(
            ty,
            TypeInfo::String | TypeInfo::Vec(_) | TypeInfo::Dict(..)
        ),
        "PyHash" => matches!(
            ty,
            TypeInfo::Int | TypeInfo::Float | TypeInfo::Bool | TypeInfo::String | TypeInfo::StrRef
        ),
        "PyIsNone" => true,
        "PyIndex" => match (ty, rhs) {
            (TypeInfo::String, Some(TypeInfo::Int)) => true,
            (TypeInfo::Vec(_), Some(TypeInfo::Int)) => true,
            (TypeInfo::Dict(..), Some(_)) => true,
            _ => false,
        },
        "PySetIndex" => match (ty, rhs) {
            (TypeInfo::Vec(_), Some(TypeInfo::Int)) => true,
            (TypeInfo::Dict(..), Some(_)) => true,
            _ => false,
        },
        "PyContains" => matches!(
            ty,
            TypeInfo::String | TypeInfo::StrRef | TypeInfo::Vec(_) | TypeInfo::Dict(..)
        ),
        "PyStrOps" => matches!(ty, TypeInfo::String | TypeInfo::StrRef),
        "PyListOps" | "PyPop" => matches!(ty, TypeInfo::Vec(_)),
        "PyFromInt" => matches!(ty, TypeInfo::Int | TypeInfo::Float),
        // "sep".join(...) elements (issue #116).
        "AsRef<str>" => matches!(ty, TypeInfo::String | TypeInfo::StrRef),
        // for x in p: iterables (Vec by value, String by char, dict keys,
        // tuples, ranges).
        "IntoIterator" => matches!(
            ty,
            TypeInfo::Vec(_)
                | TypeInfo::String
                | TypeInfo::StrRef
                | TypeInfo::Dict(..)
                | TypeInfo::Tuple(_)
                | TypeInfo::Range
        ),
        // Generated Has* duck traits are satisfied by any user class (the
        // impl is generated for every defining class).
        _ if trait_name.starts_with("Has") => matches!(ty, TypeInfo::Class(_)),
        _ => false,
    }
}

/// Map a bound's concrete-rhs token stream back to a TypeInfo, for the
/// satisfiability check.
fn concrete_tokens_to_typeinfo(tokens: &TokenStream) -> Option<TypeInfo> {
    match tokens.to_string().as_str() {
        "i64" => Some(TypeInfo::Int),
        "f64" => Some(TypeInfo::Float),
        "bool" => Some(TypeInfo::Bool),
        "String" => Some(TypeInfo::String),
        "& str" | "& 'static str" => Some(TypeInfo::StrRef),
        s if s.starts_with("Vec <") => Some(TypeInfo::Vec(Box::new(TypeInfo::PyObject))),
        _ => None,
    }
}

/// The statically-known type of a call argument: a literal (including a
/// leading unary minus), or a name with a typed analysis entry. Parameters
/// and computed expressions are None (their requirements flow through
/// propagate_from_callee instead — M4).
fn static_arg_type(expr: &ExprType, collector: &Collector) -> Option<TypeInfo> {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => Some(TypeInfo::Int),
            Some(litrs::Literal::Float(_)) => Some(TypeInfo::Float),
            Some(litrs::Literal::String(_)) => Some(TypeInfo::StrRef),
            Some(litrs::Literal::Bool(_)) => Some(TypeInfo::Bool),
            _ => None,
        },
        ExprType::UnaryOp(u) if matches!(u.op, crate::Ops::USub) => {
            match static_arg_type(u.operand.as_ref(), collector) {
                Some(TypeInfo::Int) | Some(TypeInfo::Float) => {
                    Some(match u.operand.as_ref() {
                        ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::Integer(_))) =>
                        {
                            TypeInfo::Int
                        }
                        _ => TypeInfo::Float,
                    })
                }
                _ => None,
            }
        }
        ExprType::Name(n) => collector.name_types.get(&n.id).cloned(),
        _ => None,
    }
}

/// M5: run the call-site satisfiability check over a whole body (module
/// init, __main__ block, or a function without unannotated parameters).
/// Collectors already walk inferred functions, so those need no extra pass.
pub fn check_call_sites(
    body: &[Statement],
    symbols: &SymbolTableScopes,
    name_types: &HashMap<String, TypeInfo>,
    options: &crate::PythonOptions,
) -> Result<(), String> {
    let empty = HashSet::new();
    let mut collector = Collector {
        unannotated: &empty,
        name_types,
        symbols,
        options,
        reqs: HashMap::new(),
        alias: HashMap::new(),
        returns: Vec::new(),
        reassigned: HashSet::new(),
        duck_returns: HashMap::new(),
        duck_method_calls: HashMap::new(),
        error: None,
        current_fn: None,
        callee_cache: HashMap::new(),
        visiting: HashSet::new(),
        return_visiting: HashSet::new(),
        loop_elements: HashMap::new(),
    };
    collector.walk(body);
    match collector.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The index of the parameter a callee returns, when every bare-parameter
/// return names the SAME parameter (mixed non-param returns are allowed —
/// they unify via the recursion rule at the caller).
fn callee_returned_param(callee: &crate::FunctionDef) -> Option<usize> {
    let params: Vec<&str> = callee
        .args
        .posonlyargs
        .iter()
        .chain(callee.args.args.iter())
        .map(|p| p.arg.as_str())
        .collect();
    let mut found: Option<usize> = None;
    fn walk_returns(
        body: &[Statement],
        params: &[&str],
        found: &mut Option<usize>,
        conflict: &mut bool,
    ) {
        for stmt in body {
            match &stmt.statement {
                StatementType::Return(Some(e)) => {
                    // A bare-parameter return — also inside an if-expression
                    // (`return x if ... else ...`).
                    let mut expr_returns = Vec::new();
                    collect_expr_branches(&e.value, &mut expr_returns);
                    for expr in expr_returns {
                        if let ExprType::Name(n) = expr {
                            if let Some(i) = params.iter().position(|p| p == &n.id.as_str()) {
                                match found {
                                    None => *found = Some(i),
                                    Some(prev) if *prev != i => *conflict = true,
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                StatementType::If(s) => {
                    walk_returns(&s.body, params, found, conflict);
                    walk_returns(&s.orelse, params, found, conflict);
                }
                StatementType::For(s) => {
                    walk_returns(&s.body, params, found, conflict);
                    walk_returns(&s.orelse, params, found, conflict);
                }
                StatementType::While(s) => {
                    walk_returns(&s.body, params, found, conflict);
                    walk_returns(&s.orelse, params, found, conflict);
                }
                StatementType::Try(t) => {
                    walk_returns(&t.body, params, found, conflict);
                    for h in &t.handlers {
                        walk_returns(&h.body, params, found, conflict);
                    }
                    walk_returns(&t.orelse, params, found, conflict);
                    walk_returns(&t.finalbody, params, found, conflict);
                }
                _ => {}
            }
        }
    }
    let mut conflict = false;
    walk_returns(&callee.body, &params, &mut found, &mut conflict);
    if conflict {
        None
    } else {
        found
    }
}

/// `<X as PyOp<X>>::Output` unifies with `X` (the recursive-call fixpoint:
/// int/str/list all satisfy PyOp<Self>::Output == Self). The operator's
/// argument must be the SAME X — `<A as PyAdd<B>>::Output` does NOT unify
/// with A.
fn unifies_with_recursion(a: &TokenStream, b: &TokenStream) -> bool {
    let (a, b) = (a.to_string(), b.to_string());
    fn inner(x: &str, other: &str) -> bool {
        // other == "< X as Py<Op> < X > > :: Output"
        let prefix = format!("< {} as ", x);
        let suffix = "> :: Output";
        if !(other.starts_with(&prefix) && other.ends_with(suffix)) {
            return false;
        }
        let mid = &other[prefix.len()..other.len() - suffix.len()];
        match mid.strip_suffix(&format!(" < {} >", x)) {
            Some(op) => op.starts_with("Py"),
            None => false,
        }
    }
    inner(&a, &b) || inner(&b, &a)
}

fn pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut chars = p.chars();
            match chars.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect()
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
    options: &'a crate::PythonOptions,
    reqs: HashMap<String, Vec<ParamReq>>,
    /// Local names that alias a parameter (`x = p` → x ↦ p).
    alias: HashMap<String, String>,
    returns: Vec<ExprType>,
    reassigned: HashSet<String>,
    /// Duck-typed user-method names → their unified return type tokens.
    duck_returns: HashMap<String, TokenStream>,
    /// Parameters with duck-typed user-method calls: param → method names.
    duck_method_calls: HashMap<String, HashSet<String>>,
    /// The first duck-typing error, if any (the walker cannot return
    /// Result, so errors are collected and surfaced after the walk).
    error: Option<String>,
    /// The function being collected (for self-recursion detection, M4).
    current_fn: Option<String>,
    /// Callee requirement summaries, memoized (M4): fn name → param → reqs.
    callee_cache: HashMap<String, HashMap<String, Vec<ParamReq>>>,
    /// Functions whose requirements are being collected (cycle detection).
    visiting: HashSet<String>,
    /// Functions whose return types are being computed (cycle detection,
    /// M4).
    return_visiting: HashSet<String>,
    /// Loop variables bound by `for x in p` over an unannotated parameter
    /// (or an alias of one): element name → the parameter it iterates (M2).
    loop_elements: HashMap<String, String>,
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
                    // `x = p` makes x an alias of p; any other store to an
                    // unannotated parameter (or loop element) reassigns it,
                    // which an inferred generic type cannot model.
                    if let [ExprType::Name(target)] = a.targets.as_slice() {
                        let aliases_param = matches!(&a.value, ExprType::Name(src)
                            if self.unannotated.contains(&src.id)
                                || self.alias.contains_key(&src.id));
                        if aliases_param {
                            if let ExprType::Name(src) = &a.value {
                                self.alias.insert(target.id.clone(), src.id.clone());
                            }
                        } else if self.unannotated.contains(&target.id) {
                            self.reassigned.insert(target.id.clone());
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
                    // M2: `for x in p` over an unannotated parameter (or an
                    // alias of one) bounds it as `IntoIterator<Item = E>`
                    // and threads the element type into the loop variable,
                    // which becomes a virtual parameter whose own bounds
                    // come from the body's uses of `x`.
                    if let ExprType::Name(n) = &s.iter {
                        if let Some(root) = self.root_unannotated(&n.id) {
                            match &s.target {
                                ExprType::Name(elt) => {
                                    self.loop_elements
                                        .insert(elt.id.clone(), root.clone());
                                    self.add(&root, ParamReq::Iterate(elt.id.clone()));
                                }
                                _ => {
                                    self.add(
                                        &root,
                                        ParamReq::Untranslatable(
                                            "iterating a parameter with a tuple/attribute \
                                             loop target is not inferred yet (issue #109, \
                                             M2); annotate the parameter or bind the loop \
                                             target differently"
                                                .to_string(),
                                        ),
                                    );
                                }
                            }
                        }
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
                    // A SELF-recursive call as the receiver: its result has
                    // the returned parameter's type (the fixpoint), so
                    // `fibonacci(n-1) + fibonacci(n-2)` needs
                    // `T: PyAdd<Self>` (M4).
                    if let ExprType::Call(c) = b.left.as_ref()
                        && let ExprType::Name(f) = c.func.as_ref()
                        && self.current_fn.as_deref() == Some(f.id.as_str())
                        && let Some(crate::SymbolTableNode::FunctionDef(callee)) =
                            self.symbols.get(&f.id)
                        && let Some(param_index) = callee_returned_param(callee)
                        && let Some(param) = callee
                            .args
                            .posonlyargs
                            .iter()
                            .chain(callee.args.args.iter())
                            .nth(param_index)
                        && self.unannotated.contains(&param.arg)
                    {
                        let rhs = if self.is_self_call(&b.right)
                            || matches!(b.right.as_ref(), ExprType::Name(r) if r.id == param.arg)
                        {
                            RhsType::Same
                        } else {
                            self.rhs_of(&b.right)
                        };
                        self.add(&param.arg, ParamReq::Op(t, rhs));
                    }
                }
                self.walk_expr(&b.left, false);
                self.walk_expr(&b.right, false);
            }
            ExprType::Compare(c) => {
                let operands: Vec<&ExprType> =
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
                            // constrained: `left.py_cmp(&right)`. A
                            // comparison in a CONDITION must yield bool
                            // (CmpCond); elsewhere its Output is the
                            // comparison's type.
                            if let ExprType::Name(l) = left {
                                if self.unannotated.contains(&l.id) {
                                    let req = if truthy {
                                        ParamReq::CmpCond(trait_name, self.cmp_rhs_of(right))
                                    } else {
                                        ParamReq::Cmp(trait_name, self.cmp_rhs_of(right))
                                    };
                                    self.add(&l.id, req);
                                    // The integer literal converts to the
                                    // parameter's own type (`B: From<i64>`).
                                    if matches!(
                                        right,
                                        ExprType::Constant(c)
                                            if matches!(&c.0, Some(litrs::Literal::Integer(_)))
                                    ) {
                                        self.add(&l.id, ParamReq::PyFromInt);
                                    }
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
                // `"sep".join(arg)` on a string-literal receiver: the
                // argument must be an iterable of AsRef<str> (issue #116).
                // For a Name argument that is (or aliases) an unannotated
                // parameter, the requirement lands on it with a fresh
                // element bound `E: AsRef<str>`; for a comprehension over a
                // parameter, the element itself gets the AsRef<str> bound.
                if let ExprType::Attribute(a) = c.func.as_ref()
                    && a.attr == "join"
                    && matches!(
                        a.value.as_ref(),
                        ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::String(_)))
                    )
                    && let Some(arg) = c.args.first()
                {
                    match arg {
                        ExprType::Name(n) => {
                            if let Some(root) = self.root_unannotated(&n.id) {
                                let elt = format!("__rython_elt_{}", n.id);
                                self.add(&root, ParamReq::Iterate(elt.clone()));
                                self.reqs
                                    .entry(elt.clone())
                                    .or_default()
                                    .push(ParamReq::AsRefStr);
                            }
                        }
                        ExprType::GeneratorExp(g) => {
                            // The desugared comprehension materializes a
                            // Vec of the ELEMENT EXPRESSION's type (e.g.
                            // `str(v)` -> Vec<String>), which satisfies
                            // join's AsRef<str> itself — only the iterable
                            // bound is needed here.
                            if let Some(generator) = g.generators.first()
                                && let ExprType::Name(n) = &generator.iter
                                && let Some(root) = self.root_unannotated(&n.id)
                                && let ExprType::Name(elt) = &generator.target
                            {
                                self.add(&root, ParamReq::Iterate(elt.id.clone()));
                            }
                        }
                        _ => {}
                    }
                }
                // `p.method(args)` on a parameter: record the stdlib method
                // requirement (M2) — the trait bound; pop's bound carries
                // the index argument's type.
                if let ExprType::Attribute(a) = c.func.as_ref() {
                    if let ExprType::Name(n) = a.value.as_ref() {
                        if self.unannotated.contains(&n.id) {
                            if let Some((_, trait_name, mutates, _)) = STDLIB_METHOD_TABLE
                                .iter()
                                .find(|(m, ..)| *m == a.attr)
                            {
                                let rhs = if *trait_name == "PyPop" {
                                    match c.args.first() {
                                        Some(arg) => Some(self.rhs_of(arg)),
                                        None => Some(RhsType::Concrete(quote!(i64))),
                                    }
                                } else {
                                    None
                                };
                                self.add(
                                    &n.id,
                                    ParamReq::Method((*trait_name).to_string(), *mutates, rhs),
                                );
                            } else if let Some(trait_name) = self.duck_trait_for(&a.attr, c)
                            {
                                // M3: a method defined by user classes in this
                                // package. The trait is generated once; the
                                // bound is the requirement.
                                let mutates = self.duck_method_mutates(&a.attr);
                                self.duck_method_calls
                                    .entry(n.id.clone())
                                    .or_default()
                                    .insert(a.attr.clone());
                                self.add(&n.id, ParamReq::Method(trait_name, mutates, None));
                            }
                        }
                    }
                }
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
                // A parameter passed to a user function (M4): adopt the
                // callee's parameter requirements (FlowsTo) or the concrete
                // type of an annotated callee parameter. Statically-known
                // arguments are also checked against the callee's inferred
                // bounds (M5, call-site satisfiability).
                if let ExprType::Name(f) = c.func.as_ref() {
                    if let Some(crate::SymbolTableNode::FunctionDef(callee)) =
                        self.symbols.get(&f.id)
                    {
                        self.propagate_from_callee(callee, &c.args);
                        self.check_call_site(callee, &c.args);
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
                // `p.method(...)`: known stdlib methods are recorded by the
                // Call arm (pop's bound needs the index argument); user-class
                // methods generate a duck-typing trait (M3). An UNKNOWN
                // method is a loud error with the nearest known candidates.
                if let ExprType::Name(n) = a.value.as_ref() {
                    if self.unannotated.contains(&n.id)
                        && STDLIB_METHOD_TABLE.iter().all(|(m, ..)| *m != a.attr)
                    {
                        if let Some(trait_name) = self.duck_trait_for(&a.attr, &crate::Call {
                            func: Box::new(ExprType::Attribute(a.clone())),
                            args: Vec::new(),
                            keywords: Vec::new(),
                        }) {
                            let mutates = self.duck_method_mutates(&a.attr);
                            self.duck_method_calls
                                .entry(n.id.clone())
                                .or_default()
                                .insert(a.attr.clone());
                            self.add(&n.id, ParamReq::Method(trait_name, mutates, None));
                        } else if self.error.is_none() {
                            let candidates = nearest_methods(&a.attr);
                            self.add(
                                &n.id,
                                ParamReq::Untranslatable(format!(
                                    "no known stdlib type provides method `{}`{}; \
                                     annotate `{}` or define the method on a class in \
                                     this package (issue #109)",
                                    a.attr,
                                    candidates,
                                    n.id
                                )),
                            );
                        }
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

    /// M3: generate (once per module) a duck-typing trait for a method
    /// defined by user classes in this package, and return the trait name.
    fn duck_trait_for(
        &mut self,
        method: &str,
        _call: &crate::Call,
    ) -> Option<String> {
        let classes: Vec<crate::ClassDef> = self
            .symbols
            .all_classes()
            .into_iter()
            .filter(|c| c.methods().any(|m| m.name == method))
            .collect();
        if classes.is_empty() {
            return None;
        }
        // Unify every defining class's signature: same parameter types in
        // the same order, same return type.
        let mut signatures: Vec<(String, Vec<String>, Vec<String>, Vec<String>)> = Vec::new();
        for class in &classes {
            match class_method_signature(class, method) {
                Ok(sig) => signatures.push((class.name.clone(), sig.0, sig.1, sig.2)),
                Err(e) => {
                    self.error = Some(e);
                    return None;
                }
            }
        }
        let first = signatures[0].clone();
        for (class_name, _names, types, _idents) in &signatures {
            if types != &first.2 {
                self.error = Some(format!(
                    "duck typing `{method}`: class `{class_name}` has a conflicting \
                     signature (parameter types must match across every class \
                     defining the method so one trait can bound them all)"
                ));
                return None;
            }
        }
        let ret = match class_method_return(&classes[0], method) {
            Ok(r) => r,
            Err(e) => {
                self.error = Some(e);
                return None;
            }
        };

        let trait_name = format!("Has{}", pascal_case(method));
        let trait_ident = quote::format_ident!("{}", trait_name);
        let method_ident = crate::safe_ident(method);
        let mutates = self.duck_method_mutates(method);
        let self_param = if mutates {
            quote!(&mut self)
        } else {
            quote!(&self)
        };

        // Param declarations use the first class's names (Rust impls match
        // traits by type, not name).
        let first_class = &classes[0];
        let first_sig = match class_method_signature(first_class, method) {
            Ok(sig) => sig,
            Err(e) => {
                self.error = Some(e);
                return None;
            }
        };
        let mut param_decls = Vec::new();
        let mut param_names = Vec::new();
        for (p, ty) in first_sig.0.iter().zip(first_sig.1.iter()) {
            let name = crate::safe_ident(&p);
            let ty: TokenStream = ty.parse().unwrap_or_else(|_| quote!(i64));
            param_decls.push(quote!(#name: #ty));
            param_names.push(name);
        }

        if !self
            .options
            .generated_duck_traits
            .borrow()
            .contains(&trait_name)
        {
            let class_list = classes
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let impls = classes.iter().map(|class| {
                let class_ident = crate::safe_ident(&class.name);
                quote! {
                    impl #trait_ident for #class_ident {
                        fn #method_ident(#self_param #(, #param_decls)*) -> Result<#ret, PyException> {
                            #class_ident::#method_ident(self #(, #param_names)*)
                        }
                    }
                }
            });
            let doc = format!(
                "Generated by rython from the Python method `{}` on {}                  (duck typing, issue #109): any type with this method                  satisfies the bound.",
                method, class_list
            );
            let items = quote! {
                #[doc = #doc]
                pub trait #trait_ident {
                    fn #method_ident(#self_param #(, #param_decls)*) -> Result<#ret, PyException>;
                }
                #(#impls)*
            };
            self.options.generated_duck_traits.borrow_mut().insert(trait_name.clone());
            self.options.module_pending_items.borrow_mut().push(items);
        }
        self.duck_returns.insert(method.to_string(), ret.clone());
        Some(trait_name)
    }

    /// M4: adopt a callee's parameter requirements onto the caller's
    /// arguments — a concretely-annotated callee parameter identity-forces
    /// the argument; an unannotated callee parameter propagates its bounds
    /// with inter-parameter references remapped to the caller's arguments.
    /// M5: verify a call's statically-known arguments satisfy the callee's
    /// inferred bounds (a loud conversion error, never a rustc surprise).
    /// Arguments whose type is not statically known are skipped — their
    /// requirements flow through propagate_from_callee instead.
    fn check_call_site(&mut self, callee: &crate::FunctionDef, args: &[ExprType]) {
        if self.error.is_some() {
            return;
        }
        let callee_params: Vec<&crate::Parameter> = callee
            .args
            .posonlyargs
            .iter()
            .chain(callee.args.args.iter())
            .collect();
        let callee_reqs = match self.callee_requirements(callee) {
            Ok(r) => r,
            Err(_) => return, // the callee's own inference reports this
        };
        for (i, arg) in args.iter().enumerate() {
            let Some(param) = callee_params.get(i) else { continue };
            let Some(param_reqs) = callee_reqs.get(&param.arg) else { continue };
            let Some(arg_ty) = static_arg_type(arg, self) else { continue };
            for req in param_reqs {
                // Resolve the requirement's rhs to a concrete type.
                let rhs_ty = |rhs: &RhsType| -> Option<TypeInfo> {
                    match rhs {
                        RhsType::Concrete(t) => concrete_tokens_to_typeinfo(t),
                        RhsType::Param(name) => {
                            let idx = callee_params.iter().position(|p| p.arg == *name)?;
                            args.get(idx).and_then(|a| static_arg_type(a, self))
                        }
                        RhsType::Same => Some(arg_ty.clone()),
                        RhsType::Unknown => None,
                    }
                };
                let (trait_name, ok) = match req {
                    ParamReq::Op(t, rhs) => {
                        (*t, rhs_ty(rhs).map_or(true, |r| type_satisfies(&arg_ty, t, Some(&r))))
                    }
                    ParamReq::OpOutput(t, rhs) => {
                        (*t, rhs_ty(rhs).map_or(true, |r| type_satisfies(&arg_ty, t, Some(&r))))
                    }
                    ParamReq::Cmp(t, rhs) | ParamReq::CmpCond(t, rhs) => {
                        (*t, rhs_ty(rhs).map_or(true, |r| type_satisfies(&arg_ty, t, Some(&r))))
                    }
                    ParamReq::Conversion(t) => (*t, type_satisfies(&arg_ty, t, None)),
                    ParamReq::Truthy => ("Truthy", type_satisfies(&arg_ty, "Truthy", None)),
                    ParamReq::Len => ("Len", type_satisfies(&arg_ty, "Len", None)),
                    ParamReq::Display => ("PyDisplay", true),
                    ParamReq::Repr => ("PyRepr", true),
                    ParamReq::Hash => ("PyHash", type_satisfies(&arg_ty, "PyHash", None)),
                    ParamReq::IsNone => ("PyIsNone", true),
                    ParamReq::Index(idx) => (
                        "PyIndex",
                        rhs_ty(idx).map_or(true, |r| type_satisfies(&arg_ty, "PyIndex", Some(&r))),
                    ),
                    ParamReq::SetIndex(idx, _) => (
                        "PySetIndex",
                        rhs_ty(idx).map_or(true, |r| type_satisfies(&arg_ty, "PySetIndex", Some(&r))),
                    ),
                    ParamReq::Contains(item) => (
                        "PyContains",
                        rhs_ty(item)
                            .map_or(true, |r| type_satisfies(&arg_ty, "PyContains", Some(&r))),
                    ),
                    ParamReq::Method(t, _, rhs) => {
                        let t = t.as_str();
                        let ok = match rhs {
                            Some(rhs) => rhs_ty(rhs)
                                .map_or(true, |r| type_satisfies(&arg_ty, t, Some(&r))),
                            None => type_satisfies(&arg_ty, t, None),
                        };
                        (t, ok)
                    }
                    ParamReq::PyFromInt => ("PyFromInt", type_satisfies(&arg_ty, "PyFromInt", None)),
                    ParamReq::Iterate(_) => (
                        "IntoIterator",
                        type_satisfies(&arg_ty, "IntoIterator", None),
                    ),
                    ParamReq::AsRefStr => (
                        "AsRef<str>",
                        type_satisfies(&arg_ty, "AsRef<str>", None),
                    ),
                    ParamReq::Identity(_) | ParamReq::Untranslatable(_) => continue,
                };
                if !ok {
                    self.error = Some(format!(
                        "call to `{}` cannot satisfy parameter `{}`'s inferred bound `{}`: \
                         an argument of type `{}` does not implement it. Annotate `{}` or \
                         the argument (issue #109, M5)",
                        callee.name,
                        param.arg,
                        trait_name,
                        type_display(&arg_ty),
                        param.arg,
                    ));
                    return;
                }
            }
        }
    }

    fn propagate_from_callee(&mut self, callee: &crate::FunctionDef, args: &[ExprType]) {
        let callee_params: Vec<&crate::Parameter> = callee
            .args
            .posonlyargs
            .iter()
            .chain(callee.args.args.iter())
            .collect();
        // Self-recursion (M4): the fixpoint — every non-parameter argument
        // must BE the corresponding parameter, so an expression argument
        // like `n - 1` needs `N: PySub<i64, Output = N>`.
        if self.current_fn.as_deref() == Some(callee.name.as_str()) {
            for (i, arg) in args.iter().enumerate() {
                if callee_params.get(i).is_none() {
                    continue;
                }
                if let ExprType::BinOp(b) = arg
                    && let Some(t) = bin_op_trait(&b.op)
                    && let ExprType::Name(n) = b.left.as_ref()
                    && self.unannotated.contains(&n.id)
                {
                    self.add(&n.id, ParamReq::OpOutput(t, self.rhs_of(&b.right)));
                }
            }
        }
        for (i, arg) in args.iter().enumerate() {
            let Some(arg_param) = (match arg {
                ExprType::Name(n) if self.unannotated.contains(&n.id) => Some(n.id.clone()),
                ExprType::Name(n) => self.alias.get(&n.id).cloned(),
                _ => None,
            }) else {
                continue;
            };
            let Some(callee_param) = callee_params.get(i) else { continue };
            if let Some(ann) = callee_param.annotation.as_deref() {
                // Identity-forced: the argument takes the concrete type.
                if let Some(ty) = crate::python_annotation_to_rust_type(ann) {
                    self.add(&arg_param, ParamReq::Identity(ty));
                }
                continue;
            }
            if !callee_param.arg.contains("__") && callee_param.arg == callee_param.arg {
                // Unannotated callee parameter: adopt its requirements.
                let callee_reqs = match self.callee_requirements(callee) {
                    Ok(r) => r,
                    Err(e) => {
                        self.error = Some(e);
                        return;
                    }
                };
                let Some(param_reqs) = callee_reqs.get(&callee_param.arg) else {
                    continue;
                };
                for req in param_reqs {
                    let mapped = match req {
                        ParamReq::Op(t, RhsType::Param(callee_name)) => {
                            let mapped = self.map_callee_param(callee_name, &callee_params, args);
                            ParamReq::Op(t, mapped)
                        }
                        ParamReq::Cmp(t, RhsType::Param(callee_name)) => {
                            let mapped = self.map_callee_param(callee_name, &callee_params, args);
                            ParamReq::Cmp(t, mapped)
                        }
                        ParamReq::Method(t, m, Some(RhsType::Param(callee_name))) => {
                            let mapped = self.map_callee_param(callee_name, &callee_params, args);
                            ParamReq::Method(t.clone(), *m, Some(mapped))
                        }
                        ParamReq::Iterate(elt) => {
                            // The callee iterates this parameter: the
                            // caller's argument must be iterable too — a
                            // fresh element name declares the caller-side
                            // Item, and the callee's element requirements
                            // adopt under it (M2 flow-through). The fresh
                            // name is NOT in the caller's unannotated set
                            // (it is discovered after the walk), so its
                            // requirements are inserted directly.
                            let fresh = format!("__rython_elt_{}", arg_param);
                            if let Some(elt_reqs) = callee_reqs.get(elt) {
                                for er in elt_reqs {
                                    let mapped = self.map_adopted_req(er, &callee_params, args);
                                    self.reqs
                                        .entry(fresh.clone())
                                        .or_default()
                                        .push(mapped);
                                }
                            }
                            ParamReq::Iterate(fresh)
                        }
                        other => other.clone(),
                    };
                    self.add(&arg_param, mapped);
                }
            }
        }
    }

    /// Map a callee parameter NAME to the caller's argument at the same
    /// position: a parameter argument stays a parameter, anything else is
    /// its static type.
    fn map_callee_param(
        &self,
        callee_name: &str,
        callee_params: &[&crate::Parameter],
        args: &[ExprType],
    ) -> RhsType {
        let index = callee_params.iter().position(|p| p.arg == callee_name);
        match index.and_then(|i| args.get(i)) {
            Some(ExprType::Name(n)) if self.unannotated.contains(&n.id) => {
                RhsType::Param(n.id.clone())
            }
            Some(ExprType::Name(n)) if self.alias.contains_key(&n.id) => {
                RhsType::Param(self.alias.get(&n.id).unwrap().clone())
            }
            Some(other) => self.rhs_of(other),
            None => RhsType::Unknown,
        }
    }

    /// Map a callee's LOOP-ELEMENT requirement into the caller's terms
    /// (M2): parameter operands remap to the caller's arguments; a nested
    /// iteration declares a fresh element name (the element's own body
    /// requirements are not followed — one level of flow-through).
    fn map_adopted_req(
        &self,
        req: &ParamReq,
        callee_params: &[&crate::Parameter],
        args: &[ExprType],
    ) -> ParamReq {
        match req {
            ParamReq::Op(t, RhsType::Param(name)) => {
                ParamReq::Op(t, self.map_callee_param(name, callee_params, args))
            }
            ParamReq::Cmp(t, RhsType::Param(name)) => {
                ParamReq::Cmp(t, self.map_callee_param(name, callee_params, args))
            }
            ParamReq::CmpCond(t, RhsType::Param(name)) => {
                ParamReq::CmpCond(t, self.map_callee_param(name, callee_params, args))
            }
            ParamReq::Method(t, m, Some(RhsType::Param(name))) => {
                ParamReq::Method(t.clone(), *m, Some(self.map_callee_param(name, callee_params, args)))
            }
            ParamReq::Iterate(inner) => {
                ParamReq::Iterate(format!("__rython_elt_{}", inner))
            }
            other => other.clone(),
        }
    }

    /// The requirement summary of a callee function: collect its body's
    /// requirements once per function (memoized), recursing through ITS
    /// callees. Self-recursion and cycles resolve against the in-progress
    /// sets (a fixpoint; mutual recursion is a loud error).
    fn callee_requirements(
        &mut self,
        callee: &crate::FunctionDef,
    ) -> Result<HashMap<String, Vec<ParamReq>>, String> {
        if let Some(cached) = self.callee_cache.get(&callee.name) {
            return Ok(cached.clone());
        }
        if self.visiting.contains(&callee.name) {
            if self.current_fn.as_deref() == Some(callee.name.as_str()) {
                // Self-recursion: the callee IS the current function; its
                // requirements are the ones being collected.
                return Ok(self.reqs.clone());
            }
            return Err(format!(
                "mutually recursive functions with unannotated parameters are not \
                 inferred yet (issue #109, M4); annotate the parameters of the cycle"
            ));
        }
        self.visiting.insert(callee.name.clone());
        let fn_unannotated: HashSet<String> = callee
            .args
            .posonlyargs
            .iter()
            .chain(callee.args.args.iter())
            .chain(callee.args.kwonlyargs.iter())
            .filter(|p| p.arg != "self" && p.annotation.is_none())
            .map(|p| p.arg.clone())
            .collect();
        // M2: the callee's loop elements are virtual parameters too.
        let mut unannotated = fn_unannotated.clone();
        for e in loop_element_names(&callee.body, &fn_unannotated) {
            unannotated.insert(e);
        }
        let info = crate::analyze_function_types(&callee.body);
        let mut inner = Collector {
            unannotated: &unannotated,
            name_types: &info.name_types,
            symbols: self.symbols,
            options: self.options,
            reqs: HashMap::new(),
            alias: HashMap::new(),
            returns: Vec::new(),
            reassigned: HashSet::new(),
            duck_returns: HashMap::new(),
            duck_method_calls: HashMap::new(),
            error: None,
            current_fn: Some(callee.name.clone()),
            callee_cache: std::mem::take(&mut self.callee_cache),
            visiting: self.visiting.clone(),
            return_visiting: self.return_visiting.clone(),
            loop_elements: HashMap::new(),
        };
        inner.walk(&callee.body);
        let inner_error = inner.error.clone();
        let inner_reqs = inner.reqs.clone();
        let inner_cache = inner.callee_cache;
        let inner_visiting = inner.visiting;
        self.callee_cache = inner_cache;
        self.visiting = inner_visiting;
        if let Some(e) = inner_error {
            return Err(e);
        }
        self.callee_cache.insert(callee.name.clone(), inner_reqs.clone());
        Ok(inner_reqs)
    }

    fn duck_method_mutates(&self, method: &str) -> bool {
        self.symbols.all_classes().iter().any(|c| {
            c.methods().any(|m| {
                m.name == method
                    && c.own_method_mutates(method, &self.symbols, &self.options)
            })
        })
    }

    fn walk_comprehension(&mut self, elt: &ExprType, generators: &[crate::Comprehension]) {
        // `for x in p` inside a comprehension (issue #116): same as the
        // For-statement arm — an unannotated parameter iterable bounds
        // IntoIterator and the target becomes a virtual element parameter.
        for generator in generators {
            if let ExprType::Name(n) = &generator.iter
                && let Some(root) = self.root_unannotated(&n.id)
                && let ExprType::Name(elt_name) = &generator.target
            {
                self.loop_elements.insert(elt_name.id.clone(), root.clone());
                self.add(&root, ParamReq::Iterate(elt_name.id.clone()));
            }
            self.walk_expr(&generator.iter, false);
            for cond in &generator.ifs {
                self.walk_expr(cond, true);
            }
        }
        self.walk_expr(elt, false);
    }

    /// The RhsType of an operand: a literal, a param, a param alias, a
    /// typed local, or Unknown.
    /// The rhs of a COMPARISON with a parameter: an integer literal is
    /// typed as the parameter's OWN type — Python promotes ints to floats
    /// (`2.5 <= 0`), and Rust std has no int/float cross-PartialOrd, so the
    /// only bound both i64 and f64 satisfy is `T: PyLe<Self>` (the literal
    /// converts via `From<i64>`).
    fn cmp_rhs_of(&self, expr: &ExprType) -> RhsType {
        match expr {
            ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::Integer(_))) => {
                RhsType::Same
            }
            _ => self.rhs_of(expr),
        }
    }

    /// Whether the expression is a call to the function being collected
    /// (self-recursion, M4).
    fn is_self_call(&self, expr: &ExprType) -> bool {
        matches!(expr, ExprType::Call(c)
            if matches!(c.func.as_ref(), ExprType::Name(f)
                if self.current_fn.as_deref() == Some(f.id.as_str())))
    }

    /// The root unannotated parameter a name flows from, following the
    /// `x = p` alias chain (`a = p; for x in a` iterates p).
    fn root_unannotated(&self, name: &str) -> Option<String> {
        let mut cur = name.to_string();
        let mut seen = HashSet::new();
        while let Some(next) = self.alias.get(&cur) {
            if !seen.insert(cur.clone()) {
                return None;
            }
            cur = next.clone();
        }
        if self.unannotated.contains(&cur) {
            Some(cur)
        } else {
            None
        }
    }

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
            ExprType::Call(c) => {
                // M4: a call to a user function that returns one of its own
                // parameters — the operand's type IS that parameter's (for
                // self-recursion this is the fixpoint: `x + repeat(x, n-1)`
                // constrains only `T: PyAdd<T>`).
                if let ExprType::Name(f) = c.func.as_ref()
                    && let Some(crate::SymbolTableNode::FunctionDef(callee)) =
                        self.symbols.get(&f.id)
                    && let Some(param_index) = callee_returned_param(callee)
                    && let Some(arg) = c.args.get(param_index)
                {
                    self.rhs_of(arg)
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
