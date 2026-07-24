use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::{format_ident, quote};
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

        // The I/O builtins have no no_std lowering (stdpython gates them
        // behind std): fail at conversion time with the reason, rather than
        // let the generated crate fail with a bare unresolved-name error. A
        // user definition of the same name shadows the builtin as usual.
        if options.no_std {
            if let ExprType::Name(n) = self.func.as_ref() {
                if matches!(n.id.as_str(), "print" | "input" | "open")
                    && symbols.get(&n.id).is_none()
                {
                    return Err(format!(
                        "`{}()` requires OS I/O, which the no_std profile does not \
                         provide; remove the call or convert without the no_std \
                         profile",
                        n.id
                    )
                    .into());
                }
            }
        }

        // Multi-argument range() maps to the arity-specific runtime
        // functions (Rust has no overloading); the 3-argument form can
        // raise ValueError on a zero step, hence `?`. A user-defined
        // `range` shadows the builtin and skips this mapping.
        if let ExprType::Name(n) = self.func.as_ref() {
            if n.id == "range"
                && symbols.get("range").is_none()
                && self.keywords.is_empty()
                && matches!(self.args.len(), 2 | 3)
            {
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                return Ok(if rendered.len() == 2 {
                    let (a, b) = (&rendered[0], &rendered[1]);
                    quote!(range_start_stop(#a, #b))
                } else {
                    let (a, b, c) = (&rendered[0], &rendered[1], &rendered[2]);
                    quote!(range_start_stop_step(#a, #b, #c)?)
                });
            }
        }

        // Builtins with keyword variants or by-reference runtime shapes:
        // min/max (key=, default=, n-ary), sorted (key=, reverse=),
        // enumerate (start=), pow (3-arg modular), and the by-reference
        // len/repr/reversed. Each spelling maps to its runtime variant; a
        // user definition of the same name shadows the builtin, and
        // unknown or duplicate keywords are loud errors, as Python raises
        // TypeError for them.
        if let ExprType::Name(n) = self.func.as_ref() {
            let bname = n.id.as_str();
            if matches!(
                bname,
                "min" | "max" | "sorted" | "enumerate" | "pow" | "len" | "repr"
                    | "reversed" | "frozenset" | "map" | "filter" | "list"
                    | "isinstance" | "hash" | "print"
            ) && symbols.get(bname).is_none()
            {
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                let unexpected = |kw: Option<&str>| -> Box<dyn std::error::Error> {
                    format!(
                        "{}() got an unexpected or duplicate keyword argument '{}'",
                        bname,
                        kw.unwrap_or("**kwargs")
                    )
                    .into()
                };
                match bname {
                    "min" | "max" => {
                        let mut key = None;
                        let mut default = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("key") if key.is_none() => key = Some(kw.value.clone()),
                                Some("default") if default.is_none() => {
                                    default = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.is_empty() {
                            return Err(
                                format!("{}() expected at least 1 argument", bname).into()
                            );
                        }
                        if rendered.len() >= 2 {
                            if key.is_some() || default.is_some() {
                                return Err(format!(
                                    "{}() with multiple positional values and keywords \
                                     is not supported yet; pass a list instead",
                                    bname
                                )
                                .into());
                            }
                            // Python min(a, b, c) folds pairwise; ties keep
                            // the earlier argument.
                            let two = format_ident!("{}2", bname);
                            let mut acc = rendered[0].clone();
                            for next in &rendered[1..] {
                                acc = quote!(#two(#acc, #next));
                            }
                            return Ok(acc);
                        }
                        let a = &rendered[0];
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        return Ok(match (key, default) {
                            (None, None) => {
                                let f = format_ident!("{}", bname);
                                quote!(#f(&(#a))?)
                            }
                            (Some(k), None) => {
                                let k = render(k)?;
                                let f = format_ident!("{}_key", bname);
                                quote!(#f(&(#a), #k)?)
                            }
                            (None, Some(d)) => {
                                let d = render(d)?;
                                let f = format_ident!("{}_default", bname);
                                quote!(#f(&(#a), #d))
                            }
                            (Some(k), Some(d)) => {
                                let k = render(k)?;
                                let d = render(d)?;
                                let f = format_ident!("{}_key_default", bname);
                                quote!(#f(&(#a), #k, #d))
                            }
                        });
                    }
                    "sorted" => {
                        let mut key = None;
                        let mut reverse = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("key") if key.is_none() => key = Some(kw.value.clone()),
                                Some("reverse") if reverse.is_none() => {
                                    reverse = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.len() != 1 {
                            return Err("sorted() takes exactly one positional argument"
                                .to_string()
                                .into());
                        }
                        let a = &rendered[0];
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        return Ok(match (key, reverse) {
                            (None, None) => quote!(sorted(&(#a))),
                            (Some(k), None) => {
                                let k = render(k)?;
                                quote!(sorted_key(&(#a), #k))
                            }
                            (None, Some(r)) => {
                                let r = render(r)?;
                                quote!(sorted_reverse(&(#a), #r))
                            }
                            (Some(k), Some(r)) => {
                                let k = render(k)?;
                                let r = render(r)?;
                                quote!(sorted_key_reverse(&(#a), #k, #r))
                            }
                        });
                    }
                    "enumerate" => {
                        let mut start = self.args.get(1).cloned();
                        if self.args.len() > 2 {
                            return Err("enumerate() takes at most 2 arguments"
                                .to_string()
                                .into());
                        }
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("start") if start.is_none() => {
                                    start = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.is_empty() {
                            return Err("enumerate() expected an iterable".to_string().into());
                        }
                        let a = &rendered[0];
                        return Ok(match start {
                            None => quote!(enumerate(#a)),
                            Some(s) => {
                                let s =
                                    s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                                quote!(enumerate_start(#a, #s))
                            }
                        });
                    }
                    "pow" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(
                                self.keywords[0].arg.as_deref(),
                            ));
                        }
                        return match rendered.as_slice() {
                            [b, e] => Ok(quote!(pow(#b, #e))),
                            [b, e, m] => Ok(quote!(pow_mod(#b, #e, #m)?)),
                            _ => Err("pow() takes 2 or 3 arguments".to_string().into()),
                        };
                    }
                    // isinstance is statically decidable in a typed
                    // lowering: it becomes the constant true/false when the
                    // argument's type is known (annotation or literal), and
                    // a loud error when it is not.
                    "isinstance" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if self.args.len() != 2 {
                            return Err("isinstance() takes exactly 2 arguments"
                                .to_string()
                                .into());
                        }
                        let target = match &self.args[1] {
                            ExprType::Name(t)
                                if matches!(
                                    t.id.as_str(),
                                    "int" | "float" | "str" | "bool"
                                ) =>
                            {
                                t.id.clone()
                            }
                            other => {
                                return Err(format!(
                                    "isinstance() second argument must be int, float, \
                                     str, or bool (got `{:?}`); tuples of types are not \
                                     supported yet",
                                    other
                                )
                                .into());
                            }
                        };
                        let actual: Option<String> = match &self.args[0] {
                            ExprType::Name(n) => options.local_types.get(&n.id).cloned(),
                            lit => crate::ast::tree::function_def::simple_expr_type(lit)
                                .map(|ty| match ty.to_string().as_str() {
                                    "i64" => "int".to_string(),
                                    "f64" => "float".to_string(),
                                    "bool" => "bool".to_string(),
                                    _ => "str".to_string(),
                                }),
                        };
                        let Some(actual) = actual else {
                            return Err(format!(
                                "isinstance(): the type of `{:?}` is not statically \
                                 known; annotate it (or assign it a literal) so the \
                                 check can be decided at conversion time",
                                self.args[0]
                            )
                            .into());
                        };
                        // bool is a subclass of int in Python.
                        let result = actual == target
                            || (actual == "bool" && target == "int");
                        return Ok(if result { quote!(true) } else { quote!(false) });
                    }
                    // The by-reference builtins: their runtime functions
                    // borrow, and Python's calls never consume the value.
                    "print" => {
                        // print builds on py_display (Python's str
                        // semantics: True, 1e+16, unquoted strings) — the
                        // Display fallback would silently diverge.
                        let mut sep = None;
                        let mut end = None;
                        let mut flush = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("sep") if sep.is_none() => sep = Some(kw.value.clone()),
                                Some("end") if end.is_none() => end = Some(kw.value.clone()),
                                Some("flush") if flush.is_none() => {
                                    flush = Some(kw.value.clone())
                                }
                                Some("file") => {
                                    return Err("print(file=...) is not supported: \
                                                generated code writes to stdout only"
                                        .to_string()
                                        .into());
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        // sep=None / end=None mean the defaults in Python.
                        let sep = sep.filter(|s| !crate::is_none_expr(s));
                        let end = end.filter(|e| !crate::is_none_expr(e));
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        if sep.is_none() && end.is_none() && flush.is_none() {
                            match rendered.as_slice() {
                                [] => return Ok(quote!(println!())),
                                [a] => return Ok(quote!(print(&(#a)))),
                                _ => {}
                            }
                        }
                        let sep = match sep {
                            Some(s) => render(s)?,
                            None => quote!(" "),
                        };
                        let end = match end {
                            Some(e) => render(e)?,
                            None => quote!("\n"),
                        };
                        // print(end="") with no arguments still needs an
                        // element type for the empty parts slice.
                        let parts = if rendered.is_empty() {
                            quote!(&[] as &[&str])
                        } else {
                            quote!(&[#(py_display(&(#rendered))),*])
                        };
                        return Ok(match flush {
                            None => quote!(print_parts(#parts, #sep, #end)),
                            Some(f) => {
                                let f = render(f)?;
                                quote!(print_parts_flush(#parts, #sep, #end, #f))
                            }
                        });
                    }
                    "len" | "repr" | "reversed" | "hash" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err(
                                format!("{}() takes exactly one argument", bname).into()
                            );
                        }
                        let f = format_ident!("{}", bname);
                        let a = &rendered[0];
                        return Ok(quote!(#f(&(#a))));
                    }
                    "frozenset" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err(
                                "frozenset() requires an iterable argument in rython \
                                 (an empty frozenset has no inferable element type)"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let a = &rendered[0];
                        return Ok(quote!(frozenset(#a)));
                    }
                    // map/filter dispatch on the FUNCTION argument's shape:
                    // lambdas are plain closures, while user-defined
                    // functions return Result and route through the
                    // fallible variants so their exceptions propagate.
                    "map" | "filter" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        let fallible = matches!(self.args.first(), Some(ExprType::Name(f))
                            if matches!(symbols.get(&f.id), Some(SymbolTableNode::FunctionDef(_))));
                        if bname == "filter" {
                            if rendered.len() != 2 {
                                return Err("filter() takes a function and an iterable"
                                    .to_string()
                                    .into());
                            }
                            let (f, xs) = (&rendered[0], &rendered[1]);
                            // filter(None, xs) keeps the truthy elements.
                            if self
                                .args
                                .first()
                                .is_some_and(crate::is_none_expr)
                            {
                                return Ok(quote!(filter_truthy(#xs)));
                            }
                            return Ok(if fallible {
                                quote!(filter_fallible(#f, #xs)?)
                            } else {
                                quote!(filter(#f, #xs))
                            });
                        }
                        return match rendered.as_slice() {
                            [f, xs] => Ok(if fallible {
                                quote!(map_fallible(#f, #xs)?)
                            } else {
                                quote!(map(#f, #xs))
                            }),
                            [f, a, b] => {
                                if fallible {
                                    return Err(
                                        "map() over two iterables with a user-defined \
                                         function is not supported yet; use a lambda"
                                            .to_string()
                                            .into(),
                                    );
                                }
                                Ok(quote!(map2(#f, #a, #b)))
                            }
                            _ => Err("map() takes a function and 1-2 iterables"
                                .to_string()
                                .into()),
                        };
                    }
                    "list" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err(
                                "list() requires an iterable argument in rython (an \
                                 empty list has no inferable element type; use [])"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let a = &rendered[0];
                        return Ok(quote!(list(#a)));
                    }
                    _ => unreachable!(),
                }
            }
        }

        // datetime constructors imported via `from datetime import ...`:
        // date/datetime/timedelta calls resolve their positional and
        // keyword arguments against the Python signatures and lower to the
        // runtime ::new constructors (Option-typed for the defaulted
        // parameters). date/datetime validate and propagate with `?`.
        if let ExprType::Name(n) = self.func.as_ref() {
            let from_datetime = matches!(
                symbols.get(&n.id),
                Some(SymbolTableNode::ImportFrom(import))
                    if import.module == "datetime"
            );
            if from_datetime && matches!(n.id.as_str(), "date" | "datetime" | "timedelta") {
                let (params, required): (&[&str], usize) = match n.id.as_str() {
                    "date" => (&["year", "month", "day"], 3),
                    "datetime" => (
                        &["year", "month", "day", "hour", "minute", "second", "microsecond"],
                        3,
                    ),
                    _ => (
                        &[
                            "days",
                            "seconds",
                            "microseconds",
                            "milliseconds",
                            "minutes",
                            "hours",
                            "weeks",
                        ],
                        0,
                    ),
                };
                if self.args.len() > params.len() {
                    return Err(format!(
                        "{}() takes at most {} arguments ({} given)",
                        n.id,
                        params.len(),
                        self.args.len()
                    )
                    .into());
                }
                let mut slots: Vec<Option<crate::ExprType>> = vec![None; params.len()];
                for (i, arg) in self.args.iter().enumerate() {
                    slots[i] = Some(arg.clone());
                }
                for kw in &self.keywords {
                    let idx = kw
                        .arg
                        .as_deref()
                        .and_then(|k| params.iter().position(|p| *p == k));
                    match idx {
                        Some(i) if slots[i].is_none() => slots[i] = Some(kw.value.clone()),
                        Some(i) => {
                            return Err(format!(
                                "{}() got multiple values for argument '{}'",
                                n.id, params[i]
                            )
                            .into());
                        }
                        None => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                n.id,
                                kw.arg.as_deref().unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let mut rendered = Vec::new();
                for (i, slot) in slots.iter().enumerate() {
                    let tok = match slot {
                        Some(e) => {
                            let v = e.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            if i < required { v } else { quote!(Some(#v)) }
                        }
                        None if i < required => {
                            return Err(format!(
                                "{}() missing required argument: '{}'",
                                n.id, params[i]
                            )
                            .into());
                        }
                        None => quote!(None),
                    };
                    rendered.push(tok);
                }
                let ty = crate::safe_ident(&n.id);
                let call = quote!(#ty::new(#(#rendered),*));
                // timedelta::new is infallible; date/datetime validate.
                return Ok(if n.id == "timedelta" { call } else { quote!(#call?) });
            }
        }

        // itertools functions imported via `from itertools import ...`:
        // keyword spellings (initial=, repeat=, fillvalue=, key=) map to
        // arity-specific runtime variants, and iterable arguments are
        // borrowed (the runtime takes slices; Python calls never consume).
        if let ExprType::Name(n) = self.func.as_ref() {
            let from_itertools = matches!(
                symbols.get(&n.id),
                Some(SymbolTableNode::ImportFrom(import))
                    if import.module == "itertools"
            );
            let handled = matches!(
                n.id.as_str(),
                "accumulate"
                    | "product"
                    | "zip_longest"
                    | "groupby"
                    | "pairwise"
                    | "combinations"
                    | "combinations_with_replacement"
                    | "permutations"
                    | "starmap"
            );
            if from_itertools && handled {
                let name = n.id.as_str();
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                let render = |e: crate::ExprType| {
                    e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                };
                let kw_of = |allowed: &[&str]| -> Result<
                    Vec<Option<crate::ExprType>>,
                    Box<dyn std::error::Error>,
                > {
                    let mut out: Vec<Option<crate::ExprType>> = vec![None; allowed.len()];
                    for kw in &self.keywords {
                        let idx = kw
                            .arg
                            .as_deref()
                            .and_then(|k| allowed.iter().position(|a| *a == k));
                        match idx {
                            Some(i) if out[i].is_none() => out[i] = Some(kw.value.clone()),
                            _ => {
                                return Err(format!(
                                    "{}() got an unexpected or duplicate keyword \
                                     argument '{}'",
                                    name,
                                    kw.arg.as_deref().unwrap_or("**kwargs")
                                )
                                .into());
                            }
                        }
                    }
                    Ok(out)
                };
                match name {
                    "accumulate" => {
                        let kws = kw_of(&["initial"])?;
                        let initial = kws.into_iter().next().unwrap();
                        let (xs, func) = match rendered.as_slice() {
                            [xs] => (xs.clone(), None),
                            [xs, f] => (xs.clone(), Some(f.clone())),
                            _ => {
                                return Err("accumulate() takes 1 or 2 positional \
                                            arguments"
                                    .to_string()
                                    .into());
                            }
                        };
                        return Ok(match (func, initial) {
                            (None, None) => quote!(accumulate_sum(&(#xs))),
                            (Some(f), None) => quote!(accumulate_func(&(#xs), #f)),
                            (None, Some(init)) => {
                                let init = render(init)?;
                                quote!(accumulate_sum_initial(&(#xs), #init))
                            }
                            (Some(f), Some(init)) => {
                                let init = render(init)?;
                                quote!(accumulate_func_initial(&(#xs), #f, #init))
                            }
                        });
                    }
                    "product" => {
                        let kws = kw_of(&["repeat"])?;
                        let repeat = kws.into_iter().next().unwrap();
                        if let Some(r) = repeat {
                            if rendered.len() != 1 {
                                return Err("product(iterable, repeat=n) takes one \
                                            iterable"
                                    .to_string()
                                    .into());
                            }
                            let r = render(r)?;
                            let xs = &rendered[0];
                            return match r.to_string().as_str() {
                                "2" => Ok(quote!(product_repeat2(&(#xs)))),
                                "3" => Ok(quote!(product_repeat3(&(#xs)))),
                                other => Err(format!(
                                    "product() repeat must be the literal 2 or 3 \
                                     (tuple arity is a compile-time shape); got {}",
                                    other
                                )
                                .into()),
                            };
                        }
                        return match rendered.as_slice() {
                            [a, b] => Ok(quote!(product2(&(#a), &(#b)))),
                            [a, b, c] => Ok(quote!(product3(&(#a), &(#b), &(#c)))),
                            _ => Err("product() supports 2 or 3 iterables, or one \
                                      iterable with repeat=2/3"
                                .to_string()
                                .into()),
                        };
                    }
                    "zip_longest" => {
                        let kws = kw_of(&["fillvalue"])?;
                        let fill = kws.into_iter().next().unwrap();
                        if rendered.len() != 2 {
                            return Err("zip_longest() supports exactly 2 iterables"
                                .to_string()
                                .into());
                        }
                        let (a, b) = (&rendered[0], &rendered[1]);
                        return Ok(match fill {
                            Some(v) => {
                                let v = render(v)?;
                                quote!(zip_longest_fill(&(#a), &(#b), #v))
                            }
                            None => quote!(zip_longest(&(#a), &(#b))),
                        });
                    }
                    "groupby" => {
                        let kws = kw_of(&["key"])?;
                        let key = kws.into_iter().next().unwrap();
                        if rendered.len() != 1 {
                            return Err("groupby() takes one iterable".to_string().into());
                        }
                        let xs = &rendered[0];
                        return Ok(match key {
                            Some(f) => {
                                let f = render(f)?;
                                quote!(groupby_key(&(#xs), #f))
                            }
                            None => quote!(groupby(&(#xs))),
                        });
                    }
                    "pairwise" => {
                        kw_of(&[])?;
                        if rendered.len() != 1 {
                            return Err("pairwise() takes one iterable".to_string().into());
                        }
                        let xs = &rendered[0];
                        return Ok(quote!(pairwise(&(#xs))));
                    }
                    "combinations" | "combinations_with_replacement" => {
                        kw_of(&[])?;
                        if rendered.len() != 2 {
                            return Err(
                                format!("{}() takes an iterable and r", name).into()
                            );
                        }
                        let f = format_ident!("{}", name);
                        let (xs, r) = (&rendered[0], &rendered[1]);
                        // Negative r raises ValueError, hence the `?`.
                        return Ok(quote!(#f(&(#xs), #r)?));
                    }
                    "permutations" => {
                        kw_of(&[])?;
                        return match rendered.as_slice() {
                            [xs] => Ok(quote!(permutations(&(#xs), None)?)),
                            [xs, r] => Ok(quote!(permutations(&(#xs), Some(#r))?)),
                            _ => Err("permutations() takes an iterable and optional r"
                                .to_string()
                                .into()),
                        };
                    }
                    "starmap" => {
                        kw_of(&[])?;
                        if rendered.len() != 2 {
                            return Err("starmap() takes a function and an iterable"
                                .to_string()
                                .into());
                        }
                        let (f, xs) = (&rendered[0], &rendered[1]);
                        return Ok(quote!(starmap(#f, &(#xs))));
                    }
                    _ => unreachable!(),
                }
            }
        }

        // functools/heapq/copy/textwrap/re functions: their runtime shapes
        // borrow (or mutably borrow) arguments, reduce() splits by arity,
        // and the re functions validate patterns at runtime (hence `?`).
        // Handled for both `from X import f; f(...)` and `import X;
        // X.f(...)` spellings.
        {
            let target: Option<(String, Option<&'static str>)> = match self.func.as_ref() {
                ExprType::Name(n) => match symbols.get(&n.id) {
                    Some(SymbolTableNode::ImportFrom(import))
                        if matches!(
                            import.module.as_str(),
                            "functools" | "heapq" | "copy" | "textwrap" | "re" | "hashlib"
                                | "csv"
                        ) =>
                    {
                        Some((n.id.clone(), None))
                    }
                    _ => None,
                },
                ExprType::Attribute(attr) => match attr.value.as_ref() {
                    ExprType::Name(m)
                        if matches!(
                            m.id.as_str(),
                            "functools" | "heapq" | "textwrap" | "re" | "hashlib" | "csv"
                        ) =>
                    {
                        let module: &'static str = match m.id.as_str() {
                            "functools" => "functools",
                            "heapq" => "heapq",
                            "re" => "re",
                            "hashlib" => "hashlib",
                            "csv" => "csv",
                            _ => "textwrap",
                        };
                        Some((attr.attr.clone(), Some(module)))
                    }
                    _ => None,
                },
                _ => None,
            };
            let known = target.as_ref().is_some_and(|(f, _)| {
                matches!(
                    f.as_str(),
                    "reduce"
                        | "heappush"
                        | "heappop"
                        | "heapify"
                        | "heappushpop"
                        | "heapreplace"
                        | "nlargest"
                        | "nsmallest"
                        | "copy"
                        | "deepcopy"
                        | "dedent"
                        | "indent"
                        | "search"
                        | "match"
                        | "fullmatch"
                        | "findall"
                        | "finditer"
                        | "sub"
                        | "split"
                        | "md5"
                        | "sha1"
                        | "sha256"
                        | "sha512"
                        | "wrap"
                        | "fill"
                        | "reader"
                )
            });
            if let (Some((fname, module_prefix)), true) = (target, known) {
                // wrap/fill accept width=, the re functions accept
                // flags= (and sub also count=); everything else takes no
                // keywords.
                let is_re_fn = matches!(
                    fname.as_str(),
                    "search" | "match" | "fullmatch" | "findall" | "finditer" | "sub"
                        | "split"
                );
                let mut width_kw: Option<crate::ExprType> = None;
                let mut flags_kw: Option<crate::ExprType> = None;
                let mut count_kw: Option<crate::ExprType> = None;
                let mut maxsplit_kw: Option<crate::ExprType> = None;
                for kw in &self.keywords {
                    let slot = match kw.arg.as_deref() {
                        Some("width")
                            if matches!(fname.as_str(), "wrap" | "fill") =>
                        {
                            &mut width_kw
                        }
                        Some("flags") if is_re_fn => &mut flags_kw,
                        Some("count") if fname == "sub" => &mut count_kw,
                        Some("maxsplit") if fname == "split" => &mut maxsplit_kw,
                        _ => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                fname,
                                kw.arg.as_deref().unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    };
                    if slot.is_some() {
                        return Err(format!(
                            "{}() got multiple values for a keyword argument",
                            fname
                        )
                        .into());
                    }
                    *slot = Some(kw.value.clone());
                }
                // Python re flags lower to inline flag letters: single
                // constants or |-combinations of re.IGNORECASE/I,
                // re.MULTILINE/M, re.DOTALL/S. Anything else is loud.
                fn flag_letters(
                    e: &crate::ExprType,
                ) -> Result<String, Box<dyn std::error::Error>> {
                    let name_of = |id: &str| -> Result<String, Box<dyn std::error::Error>> {
                        match id {
                            "IGNORECASE" | "I" => Ok("i".to_string()),
                            "MULTILINE" | "M" => Ok("m".to_string()),
                            "DOTALL" | "S" => Ok("s".to_string()),
                            other => Err(format!(
                                "unsupported re flag `{}`; supported: IGNORECASE,                                  MULTILINE, DOTALL (and | combinations)",
                                other
                            )
                            .into()),
                        }
                    };
                    match e {
                        ExprType::Attribute(a)
                            if matches!(a.value.as_ref(), ExprType::Name(m) if m.id == "re") =>
                        {
                            name_of(&a.attr)
                        }
                        ExprType::Name(n) => name_of(&n.id),
                        ExprType::BinOp(b)
                            if matches!(b.op, crate::ast::tree::bin_ops::BinOps::BitOr) =>
                        {
                            Ok(format!(
                                "{}{}",
                                flag_letters(&b.left)?,
                                flag_letters(&b.right)?
                            ))
                        }
                        other => Err(format!(
                            "unsupported re flags expression `{:?}`; use re.IGNORECASE,                              re.MULTILINE, re.DOTALL, or | combinations of them",
                            other
                        )
                        .into()),
                    }
                }
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                // The heap mutators take their first argument by &mut, so
                // it must be lowered as a PLACE: `heappush(rows[i], v)`
                // through the Load path would clone the element and the
                // push would silently vanish (the same clone-mutation bug
                // fixed for mutating methods on subscripted receivers).
                // rendered[0] becomes the full mutable-borrow expression:
                // py_index_mut already yields &mut for subscripts, names
                // take a fresh &mut.
                let heap_mutator = matches!(
                    fname.as_str(),
                    "heappush" | "heappop" | "heapify" | "heappushpop" | "heapreplace"
                );
                if heap_mutator {
                    if let Some(first) = self.args.first() {
                        rendered[0] = if matches!(first, ExprType::Subscript(_)) {
                            crate::ast::tree::subscript::subscript_receiver_place(
                                first,
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?
                        } else {
                            let v = &rendered[0];
                            quote!(&mut (#v))
                        };
                    }
                }
                let qual = |name: &str| {
                    let f = crate::safe_ident(name);
                    match module_prefix {
                        Some(m) => {
                            let m = format_ident!("{}", m);
                            quote!(#m::#f)
                        }
                        None => quote!(#f),
                    }
                };
                let arity = |expected: &str| -> Box<dyn std::error::Error> {
                    format!("{}() takes {} arguments", fname, expected).into()
                };
                return match (fname.as_str(), rendered.as_slice()) {
                    ("reduce", [f, xs]) => {
                        let p = qual("reduce");
                        Ok(quote!(#p(#f, &(#xs))?))
                    }
                    ("reduce", [f, xs, init]) => {
                        let p = qual("reduce_initial");
                        Ok(quote!(#p(#f, &(#xs), #init)))
                    }
                    ("reduce", _) => Err(arity("2 or 3")),
                    ("heappush", [h, x]) => {
                        let p = qual("heappush");
                        Ok(quote!(#p(#h, #x)))
                    }
                    ("heappop", [h]) => {
                        let p = qual("heappop");
                        Ok(quote!(#p(#h)?))
                    }
                    ("heapify", [h]) => {
                        let p = qual("heapify");
                        Ok(quote!(#p(#h)))
                    }
                    ("heappushpop", [h, x]) => {
                        let p = qual("heappushpop");
                        Ok(quote!(#p(#h, #x)))
                    }
                    ("heapreplace", [h, x]) => {
                        let p = qual("heapreplace");
                        Ok(quote!(#p(#h, #x)?))
                    }
                    ("nlargest" | "nsmallest", [n_arg, xs]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(#n_arg, &(#xs))))
                    }
                    ("copy" | "deepcopy", [x]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(&(#x))))
                    }
                    ("dedent", [s]) => {
                        let p = qual("dedent");
                        Ok(quote!(#p(&(#s))))
                    }
                    // wrap/fill: width by position, keyword, or Python's
                    // default of 70. They validate width, hence `?`.
                    ("wrap" | "fill", [t]) | ("wrap" | "fill", [t, _]) => {
                        let p = qual(&fname);
                        let width = match (rendered.get(1), width_kw) {
                            (Some(_), Some(_)) => {
                                return Err(format!(
                                    "{}() got multiple values for argument 'width'",
                                    fname
                                )
                                .into());
                            }
                            (Some(w), None) => quote!(#w),
                            (None, Some(w)) => {
                                let w = w.to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                quote!(#w)
                            }
                            (None, None) => quote!(70),
                        };
                        Ok(quote!(#p(&(#t), #width)?))
                    }
                    (
                        "search" | "match" | "fullmatch" | "findall" | "finditer",
                        [pat, text, ..],
                    ) => {
                        if rendered.len() > 3 {
                            return Err(format!(
                                "{}() takes at most 3 positional arguments",
                                fname
                            )
                            .into());
                        }
                        if rendered.len() > 2 && flags_kw.is_some() {
                            return Err(format!(
                                "{}() got multiple values for argument 'flags'",
                                fname
                            )
                            .into());
                        }
                        let flags = match (self.args.get(2), flags_kw) {
                            (Some(e), None) => flag_letters(e)?,
                            (None, Some(e)) => flag_letters(&e)?,
                            (None, None) => String::new(),
                            _ => unreachable!(),
                        };
                        // findall's result SHAPE depends on the pattern's
                        // capture-group count (strings for 0-1 groups,
                        // tuples beyond), so a literal pattern is compiled
                        // here at conversion time to pick the variant —
                        // which also surfaces bad patterns before the
                        // program ever runs. Non-literal patterns keep the
                        // string shape; 2+ groups there stay a loud
                        // runtime error.
                        let mut target = fname.clone();
                        if fname == "findall" {
                            if let ExprType::Constant(c) = &self.args[0] {
                                if let Some(litrs::Literal::String(slit)) = &c.0 {
                                    let pattern = slit.value();
                                    let re = regex::Regex::new(&pattern).map_err(|e| {
                                        format!(
                                            "re.findall(): cannot compile pattern {:?}: {} \
                                             (the regex engine does not support Python's \
                                             backreferences or lookarounds)",
                                            pattern, e
                                        )
                                    })?;
                                    match re.captures_len() - 1 {
                                        0 | 1 => {}
                                        2 => target = "findall2".to_string(),
                                        3 => target = "findall3".to_string(),
                                        n => {
                                            return Err(format!(
                                                "re.findall() with {} capture groups is \
                                                 not supported yet (at most 3)",
                                                n
                                            )
                                            .into());
                                        }
                                    }
                                }
                            }
                        }
                        let p = qual(&target);
                        Ok(quote!(#p(&(#pat), &(#text), #flags)?))
                    }
                    // re.split(pattern, string, maxsplit=0, flags=0):
                    // the THIRD positional is maxsplit, unlike the other
                    // re functions where it is flags.
                    ("split", [pat, text, ..]) => {
                        if rendered.len() > 4 {
                            return Err("split() takes at most 4 positional arguments"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 2 && maxsplit_kw.is_some() {
                            return Err(
                                "split() got multiple values for argument 'maxsplit'"
                                    .to_string()
                                    .into(),
                            );
                        }
                        if rendered.len() > 3 && flags_kw.is_some() {
                            return Err(
                                "split() got multiple values for argument 'flags'"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let maxsplit = match (rendered.get(2), maxsplit_kw) {
                            (Some(m), None) => quote!(#m),
                            (None, Some(m)) => {
                                let m = m.to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                quote!(#m)
                            }
                            (None, None) => quote!(0),
                            _ => unreachable!(),
                        };
                        let flags = match (self.args.get(3), flags_kw) {
                            (Some(e), None) => flag_letters(e)?,
                            (None, Some(e)) => flag_letters(&e)?,
                            (None, None) => String::new(),
                            _ => unreachable!(),
                        };
                        let p = qual("split");
                        Ok(quote!(#p(&(#pat), &(#text), #maxsplit, #flags)?))
                    }
                    ("sub", [pat, repl, text, ..]) => {
                        if rendered.len() > 4 {
                            return Err("sub() takes at most 4 positional arguments"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 3 && count_kw.is_some() {
                            return Err(
                                "sub() got multiple values for argument 'count'"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let count = match (rendered.get(3), count_kw) {
                            (Some(c), None) => quote!(#c),
                            (None, Some(c)) => {
                                let c = c.to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                quote!(#c)
                            }
                            (None, None) => quote!(0),
                            _ => unreachable!(),
                        };
                        let flags = match flags_kw {
                            Some(e) => flag_letters(&e)?,
                            None => String::new(),
                        };
                        let p = qual("sub");
                        Ok(quote!(#p(&(#pat), &(#repl), &(#text), #count, #flags)?))
                    }
                    // hashlib constructors: with initial data, or the
                    // empty + update() idiom.
                    ("md5" | "sha1" | "sha256" | "sha512", [data]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(&(#data))))
                    }
                    ("reader", [lines]) => {
                        let p = qual("reader");
                        Ok(quote!(#p(&(#lines))?))
                    }
                    ("md5" | "sha1" | "sha256" | "sha512", []) => {
                        let p = qual(&format!("{}_new", fname));
                        Ok(quote!(#p()))
                    }
                    ("indent", [s, prefix]) => {
                        let p = qual("indent");
                        Ok(quote!(#p(&(#s), &(#prefix))))
                    }
                    ("heappush" | "heappushpop" | "heapreplace" | "indent"
                        | "nlargest" | "nsmallest", _) => Err(arity("2")),
                    _ => Err(arity("the documented number of")),
                };
            }
        }

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
            // A mutating method on a subscripted receiver must go through
            // the PLACE lowering: `xs[0].append(v)` has to mutate the real
            // element, where the Load lowering (py_index) yields a clone
            // and the write silently vanishes.
            let receiver = if matches!(attr.value.as_ref(), ExprType::Subscript(_))
                && crate::ast::tree::scope::MUTATING_METHODS.contains(&attr.attr.as_str())
            {
                crate::ast::tree::subscript::subscript_receiver_place(
                    attr.value.as_ref(),
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?
            } else {
                attr.value
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?
            };

            // list.sort(): in-place, stable, with Python's keyword-only
            // key=/reverse=. Vec's inherent sort demands a total order
            // (rejecting floats), so every shape routes through the
            // PySort variants, which share sorted()'s NaN-loud comparator
            // and run key exactly once per element.
            if attr.attr == "sort" {
                if !self.args.is_empty() {
                    return Err("sort() takes no positional arguments".to_string().into());
                }
                let mut key = None;
                let mut reverse = None;
                for kw in &self.keywords {
                    match kw.arg.as_deref() {
                        Some("key") if key.is_none() => key = Some(kw.value.clone()),
                        Some("reverse") if reverse.is_none() => {
                            reverse = Some(kw.value.clone())
                        }
                        other => {
                            return Err(format!(
                                "sort() got an unexpected or duplicate keyword \
                                 argument '{}'",
                                other.unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let render = |e: crate::ExprType| {
                    e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                };
                return Ok(match (key, reverse) {
                    (None, None) => quote!((#receiver).py_sort()),
                    (None, Some(r)) => {
                        let r = render(r)?;
                        quote!((#receiver).py_sort_reverse(#r))
                    }
                    (Some(k), None) => {
                        let k = render(k)?;
                        quote!((#receiver).py_sort_key(#k))
                    }
                    (Some(k), Some(r)) => {
                        let k = render(k)?;
                        let r = render(r)?;
                        quote!((#receiver).py_sort_key_reverse(#k, #r))
                    }
                });
            }

            // str.split / str.rsplit take sep and maxsplit by position or
            // keyword, with sep=None (or absent) meaning whitespace mode.
            // Normalized here so every spelling maps to the right runtime
            // variant; unknown or duplicate keywords are loud errors, as
            // Python raises TypeError for them.
            if matches!(attr.attr.as_str(), "split" | "rsplit") {
                if self.args.len() > 2 {
                    return Err(format!(
                        "{}() takes at most 2 arguments ({} given)",
                        attr.attr,
                        self.args.len()
                    )
                    .into());
                }
                let mut sep = self.args.first().cloned();
                let mut maxsplit = self.args.get(1).cloned();
                for kw in &self.keywords {
                    match kw.arg.as_deref() {
                        Some("sep") => {
                            if sep.is_some() {
                                return Err(format!(
                                    "{}() got multiple values for argument 'sep'",
                                    attr.attr
                                )
                                .into());
                            }
                            sep = Some(kw.value.clone());
                        }
                        Some("maxsplit") => {
                            if maxsplit.is_some() {
                                return Err(format!(
                                    "{}() got multiple values for argument 'maxsplit'",
                                    attr.attr
                                )
                                .into());
                            }
                            maxsplit = Some(kw.value.clone());
                        }
                        other => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                attr.attr,
                                other.unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let is_rsplit = attr.attr == "rsplit";
                let sep = sep.filter(|s| !crate::is_none_expr(s));
                return Ok(match (sep, maxsplit) {
                    (None, None) => quote!((#receiver).py_split_whitespace()),
                    (None, Some(m)) => {
                        let m = m.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit_whitespace_maxsplit(#m))
                        } else {
                            quote!((#receiver).py_split_whitespace_maxsplit(#m))
                        }
                    }
                    (Some(s), None) => {
                        let s = s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit(&(#s))?)
                        } else {
                            quote!((#receiver).py_split(&(#s))?)
                        }
                    }
                    (Some(s), Some(m)) => {
                        let s = s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        let m = m.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit_maxsplit(&(#s), #m)?)
                        } else {
                            quote!((#receiver).py_split_maxsplit(&(#s), #m)?)
                        }
                    }
                });
            }

            // str.format on a LITERAL template translates to format! at
            // conversion time: auto-numbering, {0} positions, {name}
            // keywords, {{ escaping, and format specs all map — and any
            // spec Rust cannot reproduce exactly is a loud conversion
            // error, never approximated output.
            if attr.attr == "format" {
                let template = match attr.value.as_ref() {
                    ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_))) =>
                    {
                        match &c.0 {
                            Some(litrs::Literal::String(s)) => s.value().to_string(),
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        return Err(
                            "str.format on a non-literal template is not supported yet: \
                             the template must be a string literal so the fields can be \
                             checked at conversion time"
                                .to_string()
                                .into(),
                        );
                    }
                };
                return lower_str_format(
                    &template,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                );
            }

            // The remaining builtin methods are positional-only in Python;
            // a keyword here would be silently dropped by the positional
            // pattern match below, so fall through to the generic path,
            // which rejects keywords without a resolvable signature.
            if !self.keywords.is_empty() {
                // fall through
            } else {
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
                // re Match: m.group() is m.group(0); Rust can't overload.
                ("group", []) => {
                    return Ok(quote!((#receiver).group(0)));
                }
                // m.group("name") for (?P<name>...) groups: Rust can't
                // overload on the argument type, so the string spelling
                // routes to group_name. Numeric group(i) falls through to
                // the plain method call.
                ("group", [g]) if g.to_string().starts_with('"') => {
                    return Ok(quote!((#receiver).group_name(#g)));
                }
                // str.encode() / encode("utf-8"): UTF-8 bytes, which is
                // exactly what Rust strings hold.
                ("encode", []) => {
                    return Ok(quote!((#receiver).as_bytes().to_vec()));
                }
                ("encode", [enc]) => {
                    if enc.to_string().trim_matches('"') != "utf-8" {
                        return Err(format!(
                            "str.encode({}): only \"utf-8\" is supported",
                            enc
                        )
                        .into());
                    }
                    return Ok(quote!((#receiver).as_bytes().to_vec()));
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
                    return Ok(quote!((#receiver).py_ljust(#width, " ")?));
                }
                ("ljust", [width, fill]) => {
                    return Ok(quote!((#receiver).py_ljust(#width, &(#fill))?));
                }
                ("rjust", [width]) => {
                    return Ok(quote!((#receiver).py_rjust(#width, " ")?));
                }
                ("rjust", [width, fill]) => {
                    return Ok(quote!((#receiver).py_rjust(#width, &(#fill))?));
                }
                // str.find returns -1 when absent; str::find an Option.
                ("find", [needle]) => {
                    return Ok(quote!((#receiver).py_find(&(#needle))));
                }
                _ => {}
            }
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

        // datetime.strptime parses and validates, so it raises ValueError
        // like Python; propagate rather than hand back a bare Result.
        if name_str.ends_with(":: strptime") {
            return Ok(quote!(#call_expr?));
        }
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

/// Lower a literal `template.format(args...)` call to a Rust `format!`.
///
/// Every argument (used or not) is evaluated exactly once, in Python's
/// order, into a local binding — Python evaluates unused arguments too.
/// Used bindings are referenced from the format string by name; unused
/// ones bind to `_` so no warning fires. Errors mirror Python's:
/// mixing auto and manual numbering, out-of-range indices, and missing
/// keywords are conversion-time failures.
fn lower_str_format(
    template: &str,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    use crate::pyformat::{parse_template, translate_format_spec, FieldRef, Piece};

    let pieces = parse_template(template).map_err(|e| format!("str.format: {}", e))?;

    for kw in keywords {
        if kw.arg.is_none() {
            return Err("str.format with **kwargs is not supported yet".to_string().into());
        }
    }

    // Resolve each field to an argument slot and build the format string.
    let mut fmt = String::new();
    let mut used_positions: std::collections::HashSet<usize> = Default::default();
    let mut used_names: std::collections::HashSet<String> = Default::default();
    let mut auto_next = 0usize;
    let mut saw_auto = false;
    let mut saw_manual = false;
    let mut field_bindings: Vec<TokenStream> = Vec::new();
    for piece in &pieces {
        match piece {
            Piece::Literal(text) => {
                fmt.push_str(&text.replace('{', "{{").replace('}', "}}"));
            }
            Piece::Field { arg, conversion, spec } => {
                let index_name = match arg {
                    FieldRef::Auto => {
                        saw_auto = true;
                        let i = auto_next;
                        auto_next += 1;
                        if i >= args.len() {
                            return Err(format!(
                                "str.format: not enough positional arguments (field {} of \
                                 template {:?})",
                                i, template
                            )
                            .into());
                        }
                        used_positions.insert(i);
                        format!("__rython_fmt{}", i)
                    }
                    FieldRef::Index(i) => {
                        saw_manual = true;
                        if *i >= args.len() {
                            return Err(format!(
                                "str.format: replacement index {} out of range for \
                                 template {:?}",
                                i, template
                            )
                            .into());
                        }
                        used_positions.insert(*i);
                        format!("__rython_fmt{}", i)
                    }
                    FieldRef::Name(name) => {
                        if !keywords
                            .iter()
                            .any(|k| k.arg.as_deref() == Some(name.as_str()))
                        {
                            return Err(format!(
                                "str.format: template {:?} refers to {:?}, which is not \
                                 among the keyword arguments",
                                template, name
                            )
                            .into());
                        }
                        used_names.insert(name.clone());
                        format!("__rython_fmt_{}", name)
                    }
                };
                if saw_auto && saw_manual {
                    return Err(
                        "str.format: cannot switch between automatic field numbering \
                         and manual field specification"
                            .to_string()
                            .into(),
                    );
                }
                let lowering = if matches!(conversion, Some('r') | Some('a')) {
                    crate::pyformat::conversion_lowering(spec)
                        .map_err(|e| format!("str.format: {}", e))?
                } else {
                    translate_format_spec(spec).map_err(|e| format!("str.format: {}", e))?
                };
                match lowering {
                    crate::pyformat::SpecLowering::Inline(suffix) => {
                        if suffix.is_empty() {
                            fmt.push_str(&format!("{{{}}}", index_name));
                        } else {
                            fmt.push_str(&format!("{{{}:{}}}", index_name, suffix));
                        }
                    }
                    // The operand coerces or converts per-field (one
                    // argument may be reused with different specs), via a
                    // field-local binding referencing the argument's.
                    crate::pyformat::SpecLowering::CastF64(suffix) => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(let #ident = (#src) as f64;));
                        if suffix.is_empty() {
                            fmt.push_str(&format!("{{{}}}", fld));
                        } else {
                            fmt.push_str(&format!("{{{}:{}}}", fld, suffix));
                        }
                    }
                    crate::pyformat::SpecLowering::IntRadix {
                        fill,
                        align,
                        plus,
                        alternate,
                        zero,
                        width,
                        radix,
                    } => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(
                            let #ident = py_int_radix_format(
                                #src, #fill, #align, #plus, #alternate, #zero, #width, #radix,
                            );
                        ));
                        fmt.push_str(&format!("{{{}}}", fld));
                    }
                }
            }
        }
    }

    // Bindings: every argument evaluates exactly once, in order.
    let mut bindings = TokenStream::new();
    for (i, arg) in args.iter().enumerate() {
        let value = arg.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        if used_positions.contains(&i) {
            let ident = crate::safe_ident(&format!("__rython_fmt{}", i));
            bindings.extend(quote!(let #ident = #value;));
        } else {
            bindings.extend(quote!(let _ = #value;));
        }
    }
    for kw in keywords {
        let name = kw.arg.as_deref().unwrap_or_default();
        let value = kw
            .value
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        if used_names.contains(name) {
            let ident = crate::safe_ident(&format!("__rython_fmt_{}", name));
            bindings.extend(quote!(let #ident = #value;));
        } else {
            bindings.extend(quote!(let _ = #value;));
        }
    }

    for fb in field_bindings {
        bindings.extend(fb);
    }

    Ok(quote!({
        #bindings
        format!(#fmt)
    }))
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
