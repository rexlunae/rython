//! The `collections` classes the compiler models as typed containers.
//!
//! Python names arrive from CPython's parser as strings, so ONE string
//! comparison at the AST boundary is unavoidable — it happens exactly once,
//! in [`CollectionsType::from_class_name`] (the constructor / import name)
//! and [`CollectionsType::from_annotation_name`] (the annotation spelling,
//! which also accepts the `typing` aliases). Every consumer — the import
//! class registry, the annotation mapping, the construction lowering, the
//! method lowerings — works with the enum, so the set has one source of
//! truth.
//!
//! `Counter` and `ChainMap` are deliberately NOT here: they have no typed
//! lowering (their construction is a conversion error).

use proc_macro2::TokenStream;
use quote::quote;

/// A `collections` container class with a typed runtime counterpart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectionsType {
    /// `collections.deque` -> `stdpython::collections::deque<T>`
    Deque,
    /// `collections.defaultdict` -> `stdpython::collections::defaultdict<K, V>`
    DefaultDict,
    /// `collections.OrderedDict` -> `stdpython::collections::OrderedDict<K, V>`
    OrderedDict,
}

impl CollectionsType {
    /// Parse a CONSTRUCTOR / import name (`deque(...)`, `from collections
    /// import OrderedDict`).
    pub(crate) fn from_class_name(name: &str) -> Option<CollectionsType> {
        match name {
            "deque" => Some(CollectionsType::Deque),
            "defaultdict" => Some(CollectionsType::DefaultDict),
            "OrderedDict" => Some(CollectionsType::OrderedDict),
            _ => None,
        }
    }

    /// Parse an ANNOTATION name: the class names plus the `typing`
    /// spellings (`Deque[int]`, `DefaultDict[str, int]`).
    pub(crate) fn from_annotation_name(name: &str) -> Option<CollectionsType> {
        match name {
            "Deque" => Some(CollectionsType::Deque),
            "DefaultDict" => Some(CollectionsType::DefaultDict),
            other => CollectionsType::from_class_name(other),
        }
    }

    /// The class's Python name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            CollectionsType::Deque => "deque",
            CollectionsType::DefaultDict => "defaultdict",
            CollectionsType::OrderedDict => "OrderedDict",
        }
    }

    /// Whether the class is a MAPPING (two type arguments, the dict method
    /// surface) rather than a sequence (one element type argument).
    pub(crate) fn is_mapping(self) -> bool {
        !matches!(self, CollectionsType::Deque)
    }

    /// How many type arguments the Rust type takes.
    pub(crate) fn arity(self) -> usize {
        if self.is_mapping() { 2 } else { 1 }
    }

    /// The runtime type's path.
    pub(crate) fn rust_path(self) -> TokenStream {
        match self {
            CollectionsType::Deque => quote!(stdpython::collections::deque),
            CollectionsType::DefaultDict => quote!(stdpython::collections::defaultdict),
            CollectionsType::OrderedDict => quote!(stdpython::collections::OrderedDict),
        }
    }
}

/// A builtin CLASS usable as a `defaultdict` default factory
/// (`defaultdict(int)`, `defaultdict(list)`): the factory's Python name
/// is what `repr(defaultdict)` prints (`<class 'int'>`), so the runtime
/// keeps it. Anything else (a user function, a method, a lambda with
/// parameters) has no static lowering and is a conversion error; a
/// zero-parameter lambda lowers as an unnamed factory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultFactoryClass {
    Int,
    Float,
    Str,
    Bool,
    List,
    Dict,
    Set,
    Deque,
}

impl DefaultFactoryClass {
    /// Parse a factory NAME at the AST boundary (the caller has checked the
    /// builtin is not shadowed).
    pub(crate) fn from_name(name: &str) -> Option<DefaultFactoryClass> {
        match name {
            "int" => Some(DefaultFactoryClass::Int),
            "float" => Some(DefaultFactoryClass::Float),
            "str" => Some(DefaultFactoryClass::Str),
            "bool" => Some(DefaultFactoryClass::Bool),
            "list" => Some(DefaultFactoryClass::List),
            "dict" => Some(DefaultFactoryClass::Dict),
            "set" => Some(DefaultFactoryClass::Set),
            "deque" => Some(DefaultFactoryClass::Deque),
            _ => None,
        }
    }

    /// The class's Python name (`repr` prints it as `<class 'int'>`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            DefaultFactoryClass::Int => "int",
            DefaultFactoryClass::Float => "float",
            DefaultFactoryClass::Str => "str",
            DefaultFactoryClass::Bool => "bool",
            DefaultFactoryClass::List => "list",
            DefaultFactoryClass::Dict => "dict",
            DefaultFactoryClass::Set => "set",
            DefaultFactoryClass::Deque => "deque",
        }
    }
}

/// The deque methods whose lowering the compiler owns. The rest of
/// the deque surface (`clear`, `reverse`, `copy`, `count`, `index`,
/// `insert`) lowers through the shared list/container arms, which the
/// runtime's `deque` implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DequeMethod {
    Append,
    AppendLeft,
    Pop,
    PopLeft,
    Extend,
    ExtendLeft,
    Rotate,
    Remove,
}

impl DequeMethod {
    pub(crate) fn from_name(name: &str) -> Option<DequeMethod> {
        match name {
            "append" => Some(DequeMethod::Append),
            "appendleft" => Some(DequeMethod::AppendLeft),
            "pop" => Some(DequeMethod::Pop),
            "popleft" => Some(DequeMethod::PopLeft),
            "extend" => Some(DequeMethod::Extend),
            "extendleft" => Some(DequeMethod::ExtendLeft),
            "rotate" => Some(DequeMethod::Rotate),
            "remove" => Some(DequeMethod::Remove),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            DequeMethod::Append => "append",
            DequeMethod::AppendLeft => "appendleft",
            DequeMethod::Pop => "pop",
            DequeMethod::PopLeft => "popleft",
            DequeMethod::Extend => "extend",
            DequeMethod::ExtendLeft => "extendleft",
            DequeMethod::Rotate => "rotate",
            DequeMethod::Remove => "remove",
        }
    }
}

/// The OrderedDict-only methods (`move_to_end`, `popitem(last)`); the rest
/// of its surface is the dict surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderedDictMethod {
    MoveToEnd,
    PopItem,
}

impl OrderedDictMethod {
    pub(crate) fn from_name(name: &str) -> Option<OrderedDictMethod> {
        match name {
            "move_to_end" => Some(OrderedDictMethod::MoveToEnd),
            "popitem" => Some(OrderedDictMethod::PopItem),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            OrderedDictMethod::MoveToEnd => "move_to_end",
            OrderedDictMethod::PopItem => "popitem",
        }
    }
}

/// The builtins whose runtime form takes its iterable as a SLICE (`&[T]`)
/// or a `Vec<T>`, which a deque (ring-buffer storage) is not: a
/// deque argument is converted to a `Vec` first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceBuiltin {
    Min,
    Max,
    Sorted,
    Reversed,
    Enumerate,
    Zip,
    Map,
    Filter,
    Tuple,
    Set,
    Frozenset,
}

impl SliceBuiltin {
    pub(crate) fn from_name(name: &str) -> Option<SliceBuiltin> {
        match name {
            "min" => Some(SliceBuiltin::Min),
            "max" => Some(SliceBuiltin::Max),
            "sorted" => Some(SliceBuiltin::Sorted),
            "reversed" => Some(SliceBuiltin::Reversed),
            "enumerate" => Some(SliceBuiltin::Enumerate),
            "zip" => Some(SliceBuiltin::Zip),
            "map" => Some(SliceBuiltin::Map),
            "filter" => Some(SliceBuiltin::Filter),
            "tuple" => Some(SliceBuiltin::Tuple),
            "set" => Some(SliceBuiltin::Set),
            "frozenset" => Some(SliceBuiltin::Frozenset),
            _ => None,
        }
    }
}
