//! The closed-world class hierarchy (issue #137, the round-99 evaluation's
//! drift 2): every class that some other class in the crate derives from is
//! a POLYMORPHIC ROOT, and a slot declared with the root's type (`list[Shape]`,
//! `dict[str, Item]`, a parameter `item: Item`, a return `-> Shape`) can hold
//! any class in the root's subtree. Rust structs have no subtyping, so the
//! root's slot type is a generated SUM TYPE, `Any<Root>`, with one variant
//! per class in the subtree (the root itself included), dispatching every
//! method of the root's MRO to the variant's own implementation — CPython's
//! dynamic dispatch, decided by a `match` instead of a vtable.
//!
//! This replaces three mechanisms that each assumed a slot's static type was
//! its runtime type: the `From<Derived> for Base` slice (which dropped every
//! override and the identity), the loud refusal of an overriding derived
//! value into a base slot, and the constant fold of `isinstance` on a
//! class-typed name (which answered `false` for a Square in a `list[Shape]`
//! — the prime-directive violation the idiom corpus caught).
//!
//! The index is computed ONCE per module conversion over every module of
//! the crate (`options.module_defs`), keyed by bare class name: bases are
//! named bare in a `class Sub(Base):` header, and the crate's classes are
//! assumed unique by name (the same assumption `class_subclassed_crate_wide`
//! makes). Exception classes and Protocols never take part: they lower to
//! marker structs with no trait machinery.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{ClassDef, ExprType, PythonOptions};

/// One concrete class in a root's subtree, with the module that defines it
/// (`None` for the module being converted, whose items are in scope bare).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    pub name: String,
    pub module_path: Option<Vec<String>>,
}

/// Root name → the classes in its subtree, the root FIRST and the rest in
/// name order, so the enum's variant order (and every match over it) is
/// deterministic across runs and modules.
pub type HierarchyRoots = HashMap<String, Vec<Variant>>;

thread_local! {
    /// The index of the conversion in progress, mirrored from
    /// `options.hierarchy_roots` so `TypeInfo::to_rust_type`, `unify` and
    /// `coerce_tokens` — which have no options in hand and are called
    /// from dozens of sites — can ask about roots and subtrees. Set by
    /// `install_roots` when the module computes its index; empty outside
    /// module generation.
    static ROOTS: std::cell::RefCell<std::rc::Rc<HierarchyRoots>> =
        std::cell::RefCell::new(std::rc::Rc::new(HashMap::new()));
}

/// Whether `name` is a polymorphic root of the conversion in progress.
pub fn is_polymorphic_root(name: &str) -> bool {
    ROOTS.with(|r| r.borrow().contains_key(name))
}

/// Whether `class` is in `root`'s subtree, by the registry.
pub fn in_subtree_by_name(class: &str, root: &str) -> bool {
    ROOTS.with(|r| {
        r.borrow()
            .get(root)
            .is_some_and(|v| v.iter().any(|x| x.name == class))
    })
}

/// The nearest common root of two classes, by the registry.
pub fn common_root_by_name(a: &str, b: &str) -> Option<String> {
    ROOTS.with(|r| {
        let roots = r.borrow();
        let mut candidates: Vec<&String> = roots
            .iter()
            .filter(|(_, v)| v.iter().any(|x| x.name == a) && v.iter().any(|x| x.name == b))
            .map(|(root, _)| root)
            .collect();
        candidates.sort();
        candidates
            .iter()
            .find(|root| {
                candidates.iter().all(|other| {
                    other == *root
                        || roots
                            .get(*other)
                            .is_some_and(|v| v.iter().any(|x| x.name == ***root))
                })
            })
            .map(|r| (*r).clone())
    })
}

/// The sum type's identifier for a root: `AnyShape` for `Shape`.
pub fn any_ident(root: &str) -> proc_macro2::Ident {
    crate::safe_ident(&format!("Any{root}"))
}

/// Mirror the computed index's root names into the thread-local registry.
pub fn install_roots(roots: &HierarchyRoots) {
    ROOTS.with(|r| *r.borrow_mut() = std::rc::Rc::new(roots.clone()));
}

/// A class that takes part in hierarchies: not an exception, not a Protocol.
fn participates(c: &ClassDef) -> bool {
    !crate::ast::tree::class_def::is_exception_class(c)
        && !c.bases.iter().any(|b| match b {
            ExprType::Name(n) => n.id == "Protocol",
            ExprType::Subscript(s) => {
                matches!(s.value.as_ref(), ExprType::Name(n) if n.id == "Protocol")
            }
            _ => false,
        })
}

/// The bare-named real bases of a class header (`object` is not a base).
fn base_names(c: &ClassDef) -> Vec<String> {
    c.bases
        .iter()
        .filter_map(|b| match b {
            ExprType::Name(n) if n.id != "object" => Some(n.id.clone()),
            _ => None,
        })
        .collect()
}

/// Compute the index over the module being converted (`this_classes`, in
/// scope bare) and every other module of the crate (`options.module_defs`).
pub fn compute_roots(this_classes: &[ClassDef], options: &PythonOptions) -> HierarchyRoots {
    // name → (its direct bases, the module defining it)
    let mut classes: BTreeMap<String, (Vec<String>, Option<Vec<String>>)> = BTreeMap::new();
    for c in this_classes {
        if participates(c) {
            classes.insert(c.name.clone(), (base_names(c), None));
        }
    }
    for (path, module) in options.module_defs.iter() {
        if path[..] == options.this_module_path[..] {
            continue;
        }
        // The other module's scope, as its own emission sees it (an
        // __init__ module is its own package — mirrors
        // cross_module_chain), so its gates and relative imports fold
        // exactly as they do when it is converted.
        let mut module_opts = options.clone();
        let is_package = options
            .module_defs
            .keys()
            .any(|k| k.len() > path.len() && k[..path.len()] == path[..]);
        module_opts.module_path = if is_package {
            path.clone()
        } else {
            path[..path.len().saturating_sub(1)].to_vec()
        };
        module_opts.this_module_path = path.clone();
        let defs = crate::ast::tree::module::emitted_class_defs(module, &module_opts);
        for c in defs {
            if participates(&c) {
                classes
                    .entry(c.name.clone())
                    .or_insert((base_names(&c), Some(path.clone())));
            }
        }
    }
    // Every class whose base chain (within the crate) reaches `root` is in
    // the root's subtree. A chain is followed only through classes the
    // crate defines; a base the crate does not define (a stdlib class) ends
    // it, and a cycle is cut by the visited set.
    let ancestors_of = |name: &str| -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut frontier = vec![name.to_string()];
        while let Some(cur) = frontier.pop() {
            let Some((bases, _)) = classes.get(&cur) else {
                continue;
            };
            for b in bases {
                if seen.insert(b.clone()) {
                    out.push(b.clone());
                    frontier.push(b.clone());
                }
            }
        }
        out
    };
    let mut roots: HierarchyRoots = HashMap::new();
    for (name, (_, _)) in classes.iter() {
        for anc in ancestors_of(name) {
            if let Some((_, module_path)) = classes.get(&anc) {
                let subtree = roots.entry(anc.clone()).or_insert_with(|| {
                    vec![Variant {
                        name: anc.clone(),
                        module_path: module_path.clone(),
                    }]
                });
                let (_, this_path) = &classes[name];
                subtree.push(Variant {
                    name: name.clone(),
                    module_path: this_path.clone(),
                });
            }
        }
    }
    // The root first, then the descendants in name order (the BTreeMap
    // iteration order above is by name, but subtrees fill as the
    // descendants are visited — sort to be safe).
    for subtree in roots.values_mut() {
        let root = subtree.remove(0);
        subtree.sort_by(|a, b| a.name.cmp(&b.name));
        subtree.dedup();
        subtree.insert(0, root);
    }
    roots
}

/// The subtree of `root` (root first), or `None` when `root` is not a root.
pub fn subtree<'a>(options: &'a PythonOptions, root: &str) -> Option<&'a Vec<Variant>> {
    options.hierarchy_roots.get(root)
}

/// Whether `class` is in the subtree of `root` (the root itself included).
pub fn in_subtree(options: &PythonOptions, class: &str, root: &str) -> bool {
    subtree(options, root).is_some_and(|v| v.iter().any(|x| x.name == class))
}

/// The nearest common root of two classes, if they share one: the root
/// whose subtree contains both and which is itself in every other such
/// root's subtree (the deepest). Drives `unify` for `[Circle(), Rect()]`.
pub fn common_root(options: &PythonOptions, a: &str, b: &str) -> Option<String> {
    let mut candidates: Vec<&String> = options
        .hierarchy_roots
        .iter()
        .filter(|(r, _)| in_subtree(options, a, r) && in_subtree(options, b, r))
        .map(|(r, _)| r)
        .collect();
    candidates.sort();
    // The deepest candidate is in every other candidate's subtree.
    candidates
        .iter()
        .find(|r| candidates.iter().all(|o| o == *r || in_subtree(options, r, o)))
        .map(|r| (*r).clone())
}

/// The crate path tokens naming a variant's struct from the root's module:
/// bare for the same module, `crate::a::b::Name` otherwise.
pub fn variant_path(v: &Variant) -> proc_macro2::TokenStream {
    let ident = crate::safe_ident(&v.name);
    match &v.module_path {
        None => quote::quote!(#ident),
        Some(path) => {
            let segs: Vec<_> = path.iter().map(|p| crate::safe_ident(p)).collect();
            quote::quote!(crate #(::#segs)* :: #ident)
        }
    }
}

/// A rendered `fn` item split for delegation: the head (everything up to
/// the body: visibility, `fn name<...>(params) -> ret` and any `where`),
/// the method name, and the parameter names to forward (the receiver
/// excluded). `None` when the tokens are not a single fn item.
pub fn split_fn(rendered: &proc_macro2::TokenStream) -> Option<(proc_macro2::TokenStream, proc_macro2::Ident, Vec<proc_macro2::Ident>)> {
    use proc_macro2::{Delimiter, TokenTree};
    let trees: Vec<TokenTree> = rendered.clone().into_iter().collect();
    let fn_pos = trees
        .iter()
        .position(|t| matches!(t, TokenTree::Ident(i) if i == "fn"))?;
    let name = match trees.get(fn_pos + 1)? {
        TokenTree::Ident(i) => i.clone(),
        _ => return None,
    };
    let params_pos = trees
        .iter()
        .enumerate()
        .skip(fn_pos + 2)
        .find(|(_, t)| matches!(t, TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis))
        .map(|(i, _)| i)?;
    let body_pos = trees
        .iter()
        .enumerate()
        .skip(params_pos + 1)
        .find(|(_, t)| matches!(t, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace))
        .map(|(i, _)| i)?;
    let head: proc_macro2::TokenStream = trees[..body_pos].iter().cloned().collect();
    // Parameter names: at depth 0 of the parameter group, the identifier
    // immediately before each `:`, skipping the receiver (`self`, `&self`,
    // `&mut self`) and a `mut` pattern prefix.
    let TokenTree::Group(params) = &trees[params_pos] else {
        return None;
    };
    let ptrees: Vec<TokenTree> = params.stream().into_iter().collect();
    let mut names = Vec::new();
    for (i, t) in ptrees.iter().enumerate() {
        if matches!(t, TokenTree::Punct(p) if p.as_char() == ':')
            && i > 0
            && let TokenTree::Ident(id) = &ptrees[i - 1]
            && id != "self"
        {
            // `x: T` — but not the `:` of a path (`a::b`) or a bound
            // inside a nested group (those sit at depth > 0, unreached).
            let is_path_sep = i + 1 < ptrees.len()
                && matches!(&ptrees[i + 1], TokenTree::Punct(p) if p.as_char() == ':')
                || i >= 2
                    && matches!(&ptrees[i - 2], TokenTree::Punct(p) if p.as_char() == ':');
            if !is_path_sep {
                names.push(id.clone());
            }
        }
    }
    Some((head, name, names))
}
