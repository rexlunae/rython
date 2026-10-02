//! Typed lowering of the `collections` containers (`deque`,
//! `defaultdict`, `OrderedDict`): construction, the methods whose Python
//! spelling the generic list/dict lowerings cannot carry, and the
//! inference of the constructed type.
//!
//! The runtime counterparts live in `stdpython::collections`. Everything
//! here is the compile-time half of one contract: a construct either
//! lowers to a call that behaves like CPython's, or is a conversion error
//! naming the construct and the rewrite — never a silently different
//! program.

use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    Call, CodeGenContext, CollectionsType, ExprType, PythonOptions, SymbolTableNode,
    SymbolTableScopes, TypeInfo,
};
use crate::ast::tree::collections_types::{DefaultFactoryClass, DequeMethod, OrderedDictMethod};
use crate::ast::tree::type_ctx::{iterable_element_type, type_mentions_pyobject};

type Lowered = Result<TokenStream, Box<dyn std::error::Error>>;

fn unsupported(message: String) -> Box<dyn std::error::Error> {
    message.into()
}

/// The `collections` class a CALLEE constructs: `deque` / `defaultdict` /
/// `OrderedDict` from-imported from `collections` (aliases followed), or
/// the module-qualified `collections.deque`. `None` for anything else —
/// including the `typing` spellings, which cannot be instantiated.
pub(crate) fn ctor_of(func: &ExprType, symbols: &SymbolTableScopes) -> Option<CollectionsType> {
    match func {
        // `from collections import deque as dq` binds only `dq`; the import
        // knows the defining name.
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::ImportFrom(i))
                if crate::StdModule::from_name(&i.module)
                    == Some(crate::StdModule::Collections) =>
            {
                CollectionsType::from_class_name(&i.defining_name(&n.id))
            }
            _ => None,
        },
        ExprType::Attribute(a) => {
            let ExprType::Name(m) = a.value.as_ref() else {
                return None;
            };
            if crate::StdModule::from_name(&m.id) == Some(crate::StdModule::Collections)
                && !crate::ast::tree::call::module_name_shadowed(&m.id, symbols)
            {
                CollectionsType::from_class_name(&a.attr)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Whether a constructor call carries no data (`deque()`,
/// `OrderedDict()`, `defaultdict(list)`, `deque(maxlen=5)`): the container
/// starts EMPTY, so later uses can pin the element / key / value types the
/// way they pin an empty `[]` / `{}` literal. With `symbols` the class is
/// resolved through imports and aliases (`OD()` after `from collections
/// import OrderedDict as OD`); without them the bare class name decides.
pub(crate) fn is_empty_construction(call: &Call, symbols: Option<&SymbolTableScopes>) -> bool {
    let kind = match symbols {
        Some(symbols) => ctor_of(&call.func, symbols),
        None => match call.func.as_ref() {
            ExprType::Name(n) => CollectionsType::from_class_name(&n.id),
            ExprType::Attribute(a) => match a.value.as_ref() {
                ExprType::Name(m) if m.id == "collections" => {
                    CollectionsType::from_class_name(&a.attr)
                }
                _ => None,
            },
            _ => None,
        },
    };
    match kind {
        // The factory is the one positional argument.
        Some(CollectionsType::DefaultDict) => call.args.len() <= 1,
        Some(_) => call.args.is_empty(),
        None => false,
    }
}

/// An owned rendering of a type for storage in a container: string
/// literals are owned `String`s there.
fn owned(t: TypeInfo) -> TypeInfo {
    match t {
        TypeInfo::StrRef => TypeInfo::String,
        TypeInfo::Vec(e) => TypeInfo::Vec(Box::new(owned(*e))),
        TypeInfo::PyTuple(e) => TypeInfo::PyTuple(Box::new(owned(*e))),
        TypeInfo::Tuple(ts) => TypeInfo::Tuple(ts.into_iter().map(owned).collect()),
        TypeInfo::Dict(k, v) => TypeInfo::Dict(Box::new(owned(*k)), Box::new(owned(*v))),
        other => other,
    }
}

/// The value type a `defaultdict` factory produces, when statically
/// known: the builtin class, or a zero-parameter lambda's body.
fn factory_value_type(
    factory: &ExprType,
    ctx: Option<&CodeGenContext>,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> TypeInfo {
    if let Some(class) = factory_class(factory, symbols) {
        return factory_class_type(class);
    }
    match factory {
        ExprType::Lambda(lam) if lambda_is_nullary(lam) => {
            owned(crate::infer_type(ctx, &lam.body, options, symbols))
        }
        _ => TypeInfo::PyObject,
    }
}

/// The builtin class a `defaultdict` factory names, or `None` when it is not
/// one rython can lower. A builtin (`int`, `list`, ...) counts only while no
/// symbol shadows it; `deque` is not a builtin, so it resolves through
/// `ctor_of` — `from collections import deque` (aliases followed) or
/// `collections.deque` — and a local class or function named `deque`, or an
/// unimported bare `deque` (a NameError in CPython), is not the factory.
fn factory_class(factory: &ExprType, symbols: &SymbolTableScopes) -> Option<DefaultFactoryClass> {
    if ctor_of(factory, symbols) == Some(CollectionsType::Deque) {
        return Some(DefaultFactoryClass::Deque);
    }
    match factory {
        ExprType::Name(n) if symbols.get(&n.id).is_none() => {
            DefaultFactoryClass::from_builtin_name(&n.id)
        }
        _ => None,
    }
}

fn factory_class_type(class: DefaultFactoryClass) -> TypeInfo {
    match class {
        DefaultFactoryClass::Int => TypeInfo::Int,
        DefaultFactoryClass::Float => TypeInfo::Float,
        DefaultFactoryClass::Str => TypeInfo::String,
        DefaultFactoryClass::Bool => TypeInfo::Bool,
        DefaultFactoryClass::List => TypeInfo::Vec(Box::new(TypeInfo::PyObject)),
        DefaultFactoryClass::Dict => {
            TypeInfo::Dict(Box::new(TypeInfo::PyObject), Box::new(TypeInfo::PyObject))
        }
        DefaultFactoryClass::Set => TypeInfo::HashSet(Box::new(TypeInfo::PyObject)),
        DefaultFactoryClass::Deque => {
            TypeInfo::Collection(CollectionsType::Deque, vec![TypeInfo::PyObject])
        }
    }
}

fn lambda_is_nullary(lam: &crate::Lambda) -> bool {
    let a = &lam.args;
    a.posonlyargs.is_empty()
        && a.args.is_empty()
        && a.vararg.is_none()
        && a.kwonlyargs.is_empty()
        && a.kwarg.is_none()
}

/// The type a constructor call produces (`deque([1, 2])` is
/// `deque[int]`; `defaultdict(int)` a `defaultdict[?, int]` whose key the
/// uses pin; `OrderedDict()` an `OrderedDict[?, ?]`). Unknown parts are
/// `PyObject`, which renders `_` and is left to rustc.
pub(crate) fn construction_type(
    kind: CollectionsType,
    call: &Call,
    ctx: Option<&CodeGenContext>,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> TypeInfo {
    let unknown = TypeInfo::PyObject;
    match kind {
        CollectionsType::Deque => {
            let elem = call
                .args
                .first()
                .and_then(|a| iterable_element_type(&crate::infer_type(ctx, a, options, symbols)))
                .map(owned)
                .unwrap_or(unknown);
            TypeInfo::Collection(kind, vec![elem])
        }
        CollectionsType::DefaultDict => {
            let v = call
                .args
                .first()
                .map(|f| factory_value_type(f, ctx, options, symbols))
                .unwrap_or_else(|| unknown.clone());
            let (k, v2) = call
                .args
                .get(1)
                .and_then(|m| {
                    crate::infer_type(ctx, m, options, symbols)
                        .dict_kv()
                        .map(|(k, v)| (owned(k.clone()), owned(v.clone())))
                })
                .unwrap_or_else(|| (unknown.clone(), unknown.clone()));
            let v = if matches!(v, TypeInfo::PyObject) { v2 } else { v };
            TypeInfo::Collection(kind, vec![k, v])
        }
        CollectionsType::OrderedDict => {
            let (k, v) = call
                .args
                .first()
                .and_then(|a| {
                    let t = crate::infer_type(ctx, a, options, symbols);
                    if let Some((k, v)) = t.dict_kv() {
                        return Some((owned(k.clone()), owned(v.clone())));
                    }
                    match iterable_element_type(&t) {
                        Some(TypeInfo::Tuple(pair)) if pair.len() == 2 => {
                            Some((owned(pair[0].clone()), owned(pair[1].clone())))
                        }
                        _ => None,
                    }
                })
                .unwrap_or_else(|| (unknown.clone(), unknown.clone()));
            // `OrderedDict(boxed)` IS the boxed OrderedDict (`PyValue`),
            // not a typed container whose parameters are unknown.
            if matches!(
                call.args.first().map(|a| crate::infer_type(ctx, a, options, symbols)),
                Some(TypeInfo::PyValue)
            ) {
                return TypeInfo::PyValue;
            }
            TypeInfo::Collection(kind, vec![k, v])
        }
    }
}

/// A FIELD's type from a constructor call whose arguments leave parts
/// unknown (`self.queue = deque()`, `self.cache = OrderedDict()`): the
/// unknown parts take the boxed forms an untyped `[]` / `{}` field takes —
/// an unknown KEY is a `String` (rython's key convention), any other
/// unknown element / value the boxed `PyValue`. `None` when `call` is not
/// a collections constructor.
pub(crate) fn field_type(
    call: &Call,
    ctx: Option<&CodeGenContext>,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<TypeInfo> {
    fn boxed(t: TypeInfo, key: bool) -> TypeInfo {
        match t {
            TypeInfo::PyObject if key => TypeInfo::String,
            TypeInfo::PyObject => TypeInfo::PyValue,
            TypeInfo::Vec(e) => TypeInfo::Vec(Box::new(boxed(*e, false))),
            TypeInfo::HashSet(e) => TypeInfo::HashSet(Box::new(boxed(*e, false))),
            TypeInfo::Dict(k, v) => {
                TypeInfo::Dict(Box::new(boxed(*k, true)), Box::new(boxed(*v, false)))
            }
            TypeInfo::Collection(kind, args) => {
                let args = args
                    .into_iter()
                    .enumerate()
                    .map(|(i, a)| boxed(a, kind.is_mapping() && i == 0))
                    .collect();
                TypeInfo::Collection(kind, args)
            }
            other => other,
        }
    }
    let kind = ctor_of(&call.func, symbols)?;
    Some(boxed(construction_type(kind, call, ctx, options, symbols), false))
}

/// The type path to construct through, with the binding's KNOWN type
/// arguments spelled as a turbofish (`OrderedDict::<String, i64>`; an
/// unresolved argument is `_`, which rustc fills from the uses).
fn typed_path(path: &TokenStream, kind: CollectionsType, expected: Option<&TypeInfo>) -> TokenStream {
    match expected {
        Some(TypeInfo::Collection(k, args)) if *k == kind && args.len() == kind.arity() => {
            let args = args.iter().map(|t| t.to_rust_type());
            quote!(#path::<#(#args),*>)
        }
        _ => path.clone(),
    }
}

/// An iterable ARGUMENT as something the runtime's `IntoIterator<Item = T>`
/// parameters take: lists, sets, ranges and deques pass as they are; a
/// dict (or collections mapping) iterates its KEYS; a generator
/// expression is already an iterator; anything else (a str — one-char
/// strings, a boxed value, ...) goes through the runtime `list()` like
/// Python's own iteration protocol.
fn iterable_tokens(
    arg: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    let ty = crate::infer_type(Some(ctx), arg, options, symbols);
    let rendered = crate::render_reused(arg, ctx.clone(), options.clone(), symbols.clone())?;
    if matches!(arg, ExprType::GeneratorExp(_)) {
        return Ok(rendered);
    }
    if ty.dict_kv().is_some() {
        return Ok(quote!((#rendered).py_keys()));
    }
    Ok(match ty {
        TypeInfo::Vec(_)
        | TypeInfo::HashSet(_)
        | TypeInfo::Range
        | TypeInfo::Collection(CollectionsType::Deque, _) => rendered,
        _ => quote!(stdpython::list(#rendered)),
    })
}

/// A `maxlen=` expression as the runtime's `Option<i64>`.
fn maxlen_tokens(
    expr: Option<&ExprType>,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    let Some(e) = expr else {
        return Ok(quote!(None));
    };
    if crate::is_none_expr(e) {
        return Ok(quote!(None));
    }
    if matches!(
        crate::infer_type(Some(ctx), e, options, symbols),
        TypeInfo::Option(_)
    ) {
        return crate::render_reused(e, ctx.clone(), options.clone(), symbols.clone());
    }
    let t = crate::render_typed(
        e,
        ctx.clone(),
        options.clone(),
        symbols.clone(),
        Some(TypeInfo::Int),
    )?;
    Ok(quote!(Some(#t)))
}

/// A `defaultdict` factory argument as the constructor expression: the
/// builtin classes carry their Python name (for `repr`), a zero-parameter
/// lambda is an unnamed factory, `None` is no factory. Anything else is a
/// conversion error — a callable cannot be a runtime value here.
fn defaultdict_ctor(
    path: &TokenStream,
    factory: Option<&ExprType>,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    let Some(factory) = factory else {
        return Ok(quote!(#path::without_factory()));
    };
    if crate::is_none_expr(factory) {
        return Ok(quote!(#path::without_factory()));
    }
    if let Some(class) = factory_class(factory, symbols) {
        let make = match class {
            DefaultFactoryClass::Int => quote!(|| 0i64),
            DefaultFactoryClass::Float => quote!(|| 0.0f64),
            DefaultFactoryClass::Str => quote!(|| String::new()),
            DefaultFactoryClass::Bool => quote!(|| false),
            DefaultFactoryClass::List => quote!(|| Vec::new()),
            DefaultFactoryClass::Dict => quote!(|| stdpython::PyDict::default()),
            DefaultFactoryClass::Set => quote!(|| std::collections::HashSet::new()),
            DefaultFactoryClass::Deque => quote!(|| stdpython::collections::deque::new()),
        };
        let name = class.name();
        return Ok(quote!(#path::with_class(#make, #name)));
    }
    match factory {
        ExprType::Lambda(lam) if lambda_is_nullary(lam) => {
            let value_ty = factory_value_type(factory, Some(ctx), options, symbols);
            let expected = (!type_mentions_pyobject(&value_ty)).then(|| value_ty.clone());
            let body = crate::render_typed(
                &lam.body,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
                expected.clone(),
            )?;
            return Ok(match expected {
                Some(t) => {
                    let t = t.to_rust_type();
                    quote!(#path::new(|| -> #t { #body }))
                }
                None => quote!(#path::new(|| #body)),
            });
        }
        _ => {}
    }
    Err(unsupported(
        "defaultdict(...) takes only a builtin class (int, float, str, bool, list, dict, \
         set), `collections.deque` (imported from collections, not shadowed), None, or a \
         no-argument lambda as its default_factory; a named \
         function or any other callable is not supported yet (a callable cannot be stored \
         as a runtime value); rewrite it as `lambda: your_function()`, or use \
         `d.setdefault(key, ...)`. rython refuses to silently ignore it"
            .to_string(),
    ))
}

/// Lower a `collections` constructor call. `path` spells the runtime type
/// as the module sees it (`deque`, `OrderedDict`, `collections::deque`);
/// `expected` is the binding's final type when it is known (the turbofish
/// that pins the key / value / element types rustc cannot see).
pub(crate) fn lower_construction(
    kind: CollectionsType,
    path: &TokenStream,
    call: &Call,
    expected: Option<&TypeInfo>,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    let path = typed_path(path, kind, expected);
    match kind {
        CollectionsType::Deque => {
            let mut iterable: Option<&ExprType> = None;
            let mut maxlen: Option<&ExprType> = None;
            for (i, a) in call.args.iter().enumerate() {
                match i {
                    0 => iterable = Some(a),
                    1 => maxlen = Some(a),
                    _ => {
                        return Err(unsupported(
                            "deque() takes at most 2 arguments (iterable, maxlen)".to_string(),
                        ));
                    }
                }
            }
            for kw in &call.keywords {
                match kw.arg.as_deref() {
                    Some("maxlen") if maxlen.is_none() => maxlen = Some(&kw.value),
                    other => {
                        return Err(unsupported(format!(
                            "deque() got an unexpected or duplicate keyword argument `{}`; \
                             only `maxlen=` is supported",
                            other.unwrap_or("**kwargs")
                        )));
                    }
                }
            }
            if iterable.is_none() && maxlen.is_none() {
                return Ok(quote!(#path::new()));
            }
            let maxlen = maxlen_tokens(maxlen, ctx, options, symbols)?;
            let items = match iterable {
                Some(it) => iterable_tokens(it, ctx, options, symbols)?,
                None => quote!(Vec::new()),
            };
            Ok(quote!(#path::construct(#items, #maxlen)?))
        }
        CollectionsType::DefaultDict => {
            if !call.keywords.is_empty() {
                return Err(unsupported(
                    "defaultdict(..., key=value) keyword items are not supported yet; pass \
                     a dict as the second argument, or assign the items afterwards. rython \
                     refuses to silently ignore it"
                        .to_string(),
                ));
            }
            if call.args.len() > 2 {
                return Err(unsupported(
                    "defaultdict() takes at most 2 arguments (default_factory, mapping)"
                        .to_string(),
                ));
            }
            let ctor = defaultdict_ctor(&path, call.args.first(), ctx, options, symbols)?;
            match call.args.get(1) {
                None => Ok(ctor),
                Some(mapping) => {
                    let ty = crate::infer_type(Some(ctx), mapping, options, symbols);
                    if ty.dict_kv().is_none() {
                        return Err(unsupported(
                            "defaultdict(factory, items): the second argument must be a dict \
                             the compiler can see (a dict literal or a dict-typed name); \
                             build the defaultdict first and assign the items. rython \
                             refuses to silently ignore it"
                                .to_string(),
                        ));
                    }
                    let items = if matches!(ty, TypeInfo::Dict(..)) {
                        crate::render_reused(
                            mapping,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?
                    } else {
                        let m = crate::render_reused(
                            mapping,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?;
                        quote!(stdpython::PyDict::from_iter((#m).py_items()))
                    };
                    Ok(quote!(#ctor.with_items(#items)))
                }
            }
        }
        CollectionsType::OrderedDict => {
            if !call.keywords.is_empty() {
                return Err(unsupported(
                    "OrderedDict(key=value) keyword items are not supported yet; pass a \
                     dict or a list of (key, value) pairs. rython refuses to silently \
                     ignore it"
                        .to_string(),
                ));
            }
            match call.args.as_slice() {
                [] => Ok(quote!(#path::new())),
                [src] => {
                    let ty = crate::infer_type(Some(ctx), src, options, symbols);
                    if matches!(ty, TypeInfo::Dict(..)) {
                        let items = crate::render_reused(
                            src,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?;
                        return Ok(quote!(#path::from_dict(#items)));
                    }
                    if ty.dict_kv().is_some() {
                        let items = crate::render_reused(
                            src,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?;
                        return Ok(quote!(#path::from_pairs((#items).py_items())));
                    }
                    let pairs = matches!(
                        iterable_element_type(&ty),
                        Some(TypeInfo::Tuple(ref p)) if p.len() == 2
                    ) || matches!(src, ExprType::GeneratorExp(_));
                    if pairs {
                        // A LIST OF PAIR LITERALS (`[("z", 26), ("y", 25)]`)
                        // renders member by member against the types the
                        // container stores (owned String keys / values), so
                        // the pairs are `(String, i64)`s.
                        let stored: Option<&Vec<TypeInfo>> = match expected {
                            Some(TypeInfo::Collection(_, args)) if args.len() == 2 => Some(args),
                            _ => None,
                        };
                        let literal_pairs: Option<Vec<(&ExprType, &ExprType)>> = match src {
                            ExprType::List(items) => items
                                .iter()
                                .map(|it| match it {
                                    ExprType::Tuple(t) if t.elts.len() == 2 => {
                                        Some((&t.elts[0], &t.elts[1]))
                                    }
                                    _ => None,
                                })
                                .collect(),
                            _ => None,
                        };
                        let items = match literal_pairs {
                            Some(pairs) if !pairs.is_empty() => {
                                let member = |e: &ExprType, slot: usize| -> Lowered {
                                    let want = stored
                                        .and_then(|a| a.get(slot))
                                        .filter(|t| !type_mentions_pyobject(t))
                                        .cloned()
                                        .unwrap_or_else(|| {
                                            owned(crate::infer_type(Some(ctx), e, options, symbols))
                                        });
                                    let want = (!type_mentions_pyobject(&want)).then_some(want);
                                    crate::render_typed_reused(
                                        e,
                                        ctx.clone(),
                                        options.clone(),
                                        symbols.clone(),
                                        want,
                                    )
                                };
                                let mut rendered = Vec::with_capacity(pairs.len());
                                for (k, v) in pairs {
                                    let k = member(k, 0)?;
                                    let v = member(v, 1)?;
                                    rendered.push(quote!((#k, #v)));
                                }
                                quote!(vec![#(#rendered),*])
                            }
                            _ => iterable_tokens(src, ctx, options, symbols)?,
                        };
                        return Ok(quote!(#path::from_pairs(#items)));
                    }
                    // A BOXED argument (an untyped parameter — requests'
                    // `from_key_val_list(value)`): the runtime inspects the
                    // value as CPython's constructor does (a dict's items,
                    // or an iterable of 2-item pairs; CPython's own
                    // TypeError / ValueError otherwise) and builds the
                    // BOXED OrderedDict, which prints, compares and
                    // iterates as one.
                    if matches!(ty, TypeInfo::PyValue) {
                        let boxed = crate::render_reused(
                            src,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?;
                        return Ok(quote!(#path::from_boxed(#boxed)?));
                    }
                    Err(unsupported(
                        "OrderedDict(x): the argument must be a dict, a list of (key, \
                         value) pairs, or an untyped (boxed) value the compiler can \
                         hand to the runtime; build the OrderedDict first and assign \
                         the items. rython refuses to silently ignore it"
                            .to_string(),
                    ))
                }
                _ => Err(unsupported(
                    "OrderedDict() takes at most 1 positional argument".to_string(),
                )),
            }
        }
    }
}

/// The message for a BARE collections annotation (`q: deque`,
/// `-> OrderedDict`): the Rust type needs its element / key / value types,
/// which only the subscripted spelling names. `None` when `ann` is not
/// such an annotation.
pub(crate) fn bare_annotation_message(
    ann: &ExprType,
    symbols: &SymbolTableScopes,
    place: &str,
) -> Option<String> {
    if matches!(ann, ExprType::Subscript(_)) {
        return None;
    }
    let kind = crate::ast::tree::type_ctx::collections_class_of(ann, Some(symbols))?;
    let example = match kind {
        CollectionsType::Deque => "deque[int]",
        CollectionsType::DefaultDict => "defaultdict[str, int]",
        CollectionsType::OrderedDict => "OrderedDict[str, int]",
    };
    Some(format!(
        "{place} annotation `{}` has no element/key type; use a subscripted annotation \
         like `{example}`",
        kind.name()
    ))
}

/// The class a `collections` constructor-call EXPRESSION constructs.
pub(crate) fn construction_kind(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
) -> Option<CollectionsType> {
    match expr {
        ExprType::Call(call) => ctor_of(&call.func, symbols),
        _ => None,
    }
}

/// Lower `expr` when it is a `collections` constructor call, against the
/// binding's `expected` type (the turbofish that pins what rustc cannot
/// see). `None` — not such a call, so the caller renders it as usual.
pub(crate) fn lower_ctor_call(
    expr: &ExprType,
    expected: Option<&TypeInfo>,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<Lowered> {
    let ExprType::Call(call) = expr else {
        return None;
    };
    let kind = ctor_of(&call.func, symbols)?;
    let path = match call.func.as_ref() {
        ExprType::Name(n) => {
            let cname = crate::safe_ident(&n.id);
            quote!(#cname)
        }
        ExprType::Attribute(a) => {
            let ExprType::Name(m) = a.value.as_ref() else {
                return None;
            };
            let module = crate::safe_ident(&m.id);
            let cname = crate::safe_ident(&a.attr);
            quote!(#module::#cname)
        }
        _ => return None,
    };
    Some(lower_construction(kind, &path, call, expected, ctx, options, symbols))
}

/// `dict(m)` of a collections mapping (`return dict(counts)` — the
/// idiom that hands a defaultdict / OrderedDict out as a plain dict): the
/// insertion-ordered items collected into the runtime's dict type.
/// `None` for any other call (the generic `dict(...)` path keeps it).
pub(crate) fn lower_dict_of_mapping(
    call: &Call,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<Lowered> {
    let ExprType::Name(f) = call.func.as_ref() else {
        return None;
    };
    if f.id != "dict" || symbols.get("dict").is_some() || !call.keywords.is_empty() {
        return None;
    }
    let [src] = call.args.as_slice() else {
        return None;
    };
    let ty = crate::infer_type(Some(ctx), src, options, symbols);
    if !(matches!(ty, TypeInfo::Collection(..)) && ty.dict_kv().is_some()) {
        return None;
    }
    Some(
        crate::render_reused(src, ctx.clone(), options.clone(), symbols.clone())
            .map(|m| quote!(stdpython::PyDict::from_iter((#m).py_items()))),
    )
}

/// A dataclass `field(default_factory=F)` default of a collections
/// annotation (`entries: deque[str] = field(default_factory=deque)`,
/// `counts: defaultdict[str, int] = field(default_factory=lambda:
/// defaultdict(int))`): the container the factory builds, typed by the
/// annotation. The factory is the class itself (an empty container; a bare
/// `defaultdict` is one WITHOUT a default_factory, as in CPython) or a
/// no-argument lambda returning a construction; anything else is a
/// conversion error. Filling the slot with `Default::default()` instead
/// would silently drop a defaultdict's factory.
pub(crate) fn lower_field_factory(
    field_call: &ExprType,
    annotation: &TypeInfo,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    let TypeInfo::Collection(kind, _) = annotation else {
        return Err(unsupported("field(default_factory=...) needs a collections annotation".to_string()));
    };
    let ExprType::Call(c) = field_call else {
        return Err(unsupported("field(default_factory=...) expected".to_string()));
    };
    let factory = c
        .keywords
        .iter()
        .find(|k| k.arg.as_deref() == Some("default_factory"))
        .map(|k| &k.value);
    match factory {
        // `default_factory=deque` / `OrderedDict` / `defaultdict`: the class
        // called with no arguments.
        Some(f @ ExprType::Name(_)) if ctor_of(f, symbols) == Some(*kind) => {
            let call = Call {
                func: Box::new(f.clone()),
                args: Vec::new(),
                keywords: Vec::new(),
            };
            let call = ExprType::Call(call);
            match lower_ctor_call(&call, Some(annotation), ctx, options, symbols) {
                Some(lowered) => lowered,
                None => Err(unsupported("field(default_factory=...) expected a class".to_string())),
            }
        }
        // `default_factory=lambda: <construction>`.
        Some(ExprType::Lambda(lam)) if lambda_is_nullary(lam) => crate::render_typed(
            &lam.body,
            ctx.clone(),
            options.clone(),
            symbols.clone(),
            Some(annotation.clone()),
        ),
        _ => Err(unsupported(
            "field(default_factory=...) of a collections container takes the class itself \
             (`default_factory=deque`) or a no-argument lambda building it (`lambda: \
             defaultdict(int)`); another factory is not supported yet. rython refuses to \
             silently ignore it"
                .to_string(),
        )),
    }
}

/// Whether an expression is a str literal.
fn is_str_literal(e: &ExprType) -> bool {
    matches!(e, ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::String(_))))
}

/// Whether `index` is a str LITERAL used on a collections mapping whose
/// key type is still unknown: such a key is an owned `String` (rython's
/// key convention), which also keeps the runtime's `&str` / `K` impls from
/// being ambiguous for rustc.
pub(crate) fn owns_literal_key(recv_ty: &TypeInfo, index: &ExprType) -> bool {
    matches!(recv_ty, TypeInfo::Collection(..))
        && matches!(recv_ty.dict_kv(), Some((k, _)) if matches!(k, TypeInfo::PyObject))
        && is_str_literal(index)
}

/// A KEY argument as the container's key type: rendered against `K` when
/// it is known; a str literal into a container whose key is still unknown
/// is an owned `String` (rython's key convention — and what keeps the
/// runtime's `&str` / `K` index impls from being ambiguous).
pub(crate) fn key_tokens(
    key: &ExprType,
    key_ty: &TypeInfo,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    if !type_mentions_pyobject(key_ty) {
        return crate::render_typed_reused(
            key,
            ctx.clone(),
            options.clone(),
            symbols.clone(),
            Some(key_ty.clone()),
        );
    }
    if is_str_literal(key) {
        return crate::render_typed(
            key,
            ctx.clone(),
            options.clone(),
            symbols.clone(),
            Some(TypeInfo::String),
        );
    }
    crate::render_reused(key, ctx.clone(), options.clone(), symbols.clone())
}

fn arity_error(what: &str, expected: &str) -> Box<dyn std::error::Error> {
    unsupported(format!("{what} takes {expected}"))
}

/// Lower a deque method call on `recv` (already rendered as a place for
/// the mutators).
pub(crate) fn lower_deque_method(
    method: DequeMethod,
    recv: &TokenStream,
    elem: &TypeInfo,
    call: &Call,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    let what = format!("deque.{}()", method.name());
    if !call.keywords.is_empty() {
        return Err(unsupported(format!(
            "{what} does not take keyword arguments"
        )));
    }
    let elem_expected = (!type_mentions_pyobject(elem)).then(|| elem.clone());
    let element = |a: &ExprType| {
        crate::render_typed_reused(
            a,
            ctx.clone(),
            options.clone(),
            symbols.clone(),
            elem_expected.clone(),
        )
    };
    match (method, call.args.as_slice()) {
        (DequeMethod::Append, [a]) => {
            let a = element(a)?;
            Ok(quote!((#recv).append(#a)))
        }
        (DequeMethod::AppendLeft, [a]) => {
            let a = element(a)?;
            Ok(quote!((#recv).appendleft(#a)))
        }
        (DequeMethod::Pop, []) => Ok(quote!((#recv).pop()?)),
        (DequeMethod::PopLeft, []) => Ok(quote!((#recv).popleft()?)),
        (DequeMethod::Extend, [it]) => {
            let it = iterable_tokens(it, ctx, options, symbols)?;
            Ok(quote!((#recv).extend(#it)))
        }
        (DequeMethod::ExtendLeft, [it]) => {
            let it = iterable_tokens(it, ctx, options, symbols)?;
            Ok(quote!((#recv).extendleft(#it)))
        }
        (DequeMethod::Rotate, []) => Ok(quote!((#recv).rotate(1))),
        (DequeMethod::Rotate, [n]) => {
            let n = crate::render_typed(
                n,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
                Some(TypeInfo::Int),
            )?;
            Ok(quote!((#recv).rotate(#n)))
        }
        (DequeMethod::Remove, [a]) => {
            let a = element(a)?;
            Ok(quote!((#recv).remove(&(#a))?))
        }
        (DequeMethod::Append | DequeMethod::AppendLeft | DequeMethod::Remove, _) => {
            Err(arity_error(&what, "exactly one argument"))
        }
        (DequeMethod::Extend | DequeMethod::ExtendLeft, _) => {
            Err(arity_error(&what, "exactly one argument (an iterable)"))
        }
        (DequeMethod::Pop | DequeMethod::PopLeft, _) => Err(arity_error(&what, "no arguments")),
        (DequeMethod::Rotate, _) => Err(arity_error(&what, "at most one argument")),
    }
}

/// Lower an OrderedDict-only method call on `recv`.
pub(crate) fn lower_ordered_dict_method(
    method: OrderedDictMethod,
    recv: &TokenStream,
    key_ty: &TypeInfo,
    call: &Call,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Lowered {
    let what = format!("OrderedDict.{}()", method.name());
    // `last=` (positional or keyword; default True).
    let mut positional = call.args.iter();
    let key = match method {
        OrderedDictMethod::MoveToEnd => match positional.next() {
            Some(k) => Some(k),
            None => {
                return Err(arity_error(&what, "a key argument"));
            }
        },
        OrderedDictMethod::PopItem => None,
    };
    let mut last: Option<&ExprType> = positional.next();
    if positional.next().is_some() {
        return Err(arity_error(&what, "at most a key and `last`"));
    }
    for kw in &call.keywords {
        match kw.arg.as_deref() {
            Some("last") if last.is_none() => last = Some(&kw.value),
            other => {
                return Err(unsupported(format!(
                    "{what} got an unexpected or duplicate keyword argument `{}`",
                    other.unwrap_or("**kwargs")
                )));
            }
        }
    }
    let last = match last {
        None => quote!(true),
        Some(e) => crate::render_typed(
            e,
            ctx.clone(),
            options.clone(),
            symbols.clone(),
            Some(TypeInfo::Bool),
        )?,
    };
    match (method, key) {
        (OrderedDictMethod::MoveToEnd, Some(k)) => {
            let k = key_tokens(k, key_ty, ctx, options, symbols)?;
            Ok(quote!((#recv).move_to_end(&(#k), #last)?))
        }
        _ => Ok(quote!((#recv).popitem(#last)?)),
    }
}
