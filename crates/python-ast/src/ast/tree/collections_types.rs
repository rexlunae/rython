//! The `collections`-module names the compiler knows, as a typed enum.
//!
//! Python identifiers arrive from CPython's parser as strings, so ONE
//! string comparison at the AST boundary is unavoidable — but it happens
//! exactly once, in [`CollectionsType::from_name`]. Every consumer (the
//! import item and class registries, the `collections.X(...)` construction
//! lowering, the map-field classifier, the namedtuple-base check, and the
//! conversion-time rejection of the classes with no lowering) works with
//! the enum, so the set of known names has a single source of truth
//! instead of parallel string lists that can drift.

use quote::quote;

/// A name of the stdpython `collections` runtime module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectionsType {
    Counter,
    Deque,
    Defaultdict,
    OrderedDict,
    ChainMap,
    /// `namedtuple(...)`: a class factory handled at conversion time
    /// (the namedtuple-base divergence), not a runtime struct.
    Namedtuple,
}

impl CollectionsType {
    /// Parse a Python identifier (an attribute or imported name) at the
    /// AST boundary. The caller is responsible for having established that
    /// the name resolves against the `collections` module — except where a
    /// consumer documents that it classifies the bare name.
    pub(crate) fn from_name(name: &str) -> Option<CollectionsType> {
        match name {
            "Counter" => Some(CollectionsType::Counter),
            "deque" => Some(CollectionsType::Deque),
            "defaultdict" => Some(CollectionsType::Defaultdict),
            "OrderedDict" => Some(CollectionsType::OrderedDict),
            "ChainMap" => Some(CollectionsType::ChainMap),
            "namedtuple" => Some(CollectionsType::Namedtuple),
            _ => None,
        }
    }

    /// An ANNOTATION spelling. CPython's `collections` names are
    /// case-sensitive, so `from_name` stays exact; an annotation may also use
    /// the CamelCase alias (`typing.Deque[int]`, `DefaultDict[str, int]`),
    /// which is why this is a separate parse and not the same function.
    pub(crate) fn from_annotation_name(name: &str) -> Option<CollectionsType> {
        match name {
            "Deque" => Some(CollectionsType::Deque),
            "DefaultDict" => Some(CollectionsType::Defaultdict),
            other => CollectionsType::from_name(other),
        }
    }

    /// Whether the class is a MAPPING (two type arguments, the dict method
    /// surface) rather than a sequence (one element type argument). Only the
    /// classes with a typed lowering take part; the rest are rejected at
    /// conversion before a field or parameter is typed.
    pub(crate) fn is_mapping(self) -> bool {
        matches!(self, CollectionsType::Defaultdict | CollectionsType::OrderedDict)
    }

    /// How many type arguments the Rust type takes.
    pub(crate) fn arity(self) -> usize {
        if self.is_mapping() { 2 } else { 1 }
    }

    /// The runtime type's path.
    pub(crate) fn rust_path(self) -> proc_macro2::TokenStream {
        match self {
            CollectionsType::Counter => quote!(stdpython::collections::Counter),
            CollectionsType::Deque => quote!(stdpython::collections::deque),
            CollectionsType::Defaultdict => quote!(stdpython::collections::defaultdict),
            CollectionsType::OrderedDict => quote!(stdpython::collections::OrderedDict),
            // Every other name is rejected at conversion time
            // (`unlowered_rewrite`), so there is no runtime path to name.
            _ => quote!(()),
        }
    }

    /// The name's Python spelling.
    pub(crate) fn name(self) -> &'static str {
        match self {
            CollectionsType::Counter => "Counter",
            CollectionsType::Deque => "deque",
            CollectionsType::Defaultdict => "defaultdict",
            CollectionsType::OrderedDict => "OrderedDict",
            CollectionsType::ChainMap => "ChainMap",
            CollectionsType::Namedtuple => "namedtuple",
        }
    }

    /// Whether a construction (`X(...)` / `collections.X(...)`) lowers to
    /// the runtime struct's `X::new(...)` constructor.
    pub(crate) fn has_construction_lowering(self) -> bool {
        matches!(
            self,
            CollectionsType::Counter
                | CollectionsType::Deque
                | CollectionsType::Defaultdict
                | CollectionsType::OrderedDict
        )
    }

    /// Whether `from collections import X` emits a `use` of a runtime item.
    /// Only the classes with a construction lowering have one. The rest
    /// drop the import with a warning: `namedtuple` is a conversion-time
    /// class factory with no runtime counterpart (a `use` of it failed
    /// E0432 even for the supported `class P(namedtuple(...))` base), and
    /// a Counter/ChainMap construction is rejected by
    /// [`CollectionsType::unlowered_rewrite`].
    pub(crate) fn is_module_item(self) -> bool {
        self.has_construction_lowering()
    }

    /// Whether a field initialized by `X()` is a map: the boxed PyDict,
    /// matching `dict[str, Any]` lowering (urllib3's
    /// RecentlyUsedContainer._container).
    pub(crate) fn is_map_field(self) -> bool {
        matches!(
            self,
            CollectionsType::Defaultdict | CollectionsType::OrderedDict
        )
    }

    /// For a class the runtime defines but the converter cannot construct,
    /// the rewrite a conversion error recommends; `None` for every name
    /// that converts. Rejecting at conversion is the loud half of "correct
    /// or loud": without it the call renders as a bare `X(...)` and fails
    /// in rustc (E0425 / E0423), far from the Python source.
    pub(crate) fn unlowered_rewrite(self) -> Option<&'static str> {
        match self {
            // CPython's ChainMap holds REFERENCES to its maps (a write to
            // `d1` after `ChainMap(d1, d2)` is visible through the chain),
            // which a copy cannot reproduce; the merged dict is exact only
            // when neither the chain nor its maps change afterwards.
            CollectionsType::ChainMap => Some(
                "merge the maps into one dict instead — `{**d2, **d1}` for \
                 `ChainMap(d1, d2)` (later maps first, so earlier ones win) — \
                 when neither the chain nor its maps change afterwards",
            ),
            CollectionsType::Counter
            | CollectionsType::Deque
            | CollectionsType::Defaultdict
            | CollectionsType::OrderedDict
            | CollectionsType::Namedtuple => None,
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
    /// Parse a BUILTIN factory name at the AST boundary (the caller has
    /// checked the builtin is not shadowed). `deque` is not a builtin: it
    /// resolves through the `collections` import (`factory_class` in
    /// `collections_lower.rs`), never from a bare unbound name.
    pub(crate) fn from_builtin_name(name: &str) -> Option<DefaultFactoryClass> {
        match name {
            "int" => Some(DefaultFactoryClass::Int),
            "float" => Some(DefaultFactoryClass::Float),
            "str" => Some(DefaultFactoryClass::Str),
            "bool" => Some(DefaultFactoryClass::Bool),
            "list" => Some(DefaultFactoryClass::List),
            "dict" => Some(DefaultFactoryClass::Dict),
            "set" => Some(DefaultFactoryClass::Set),
            _ => None,
        }
    }

    /// The class's qualified Python name (`repr` prints it as
    /// `<class 'int'>`, `<class 'collections.deque'>`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            DefaultFactoryClass::Int => "int",
            DefaultFactoryClass::Float => "float",
            DefaultFactoryClass::Str => "str",
            DefaultFactoryClass::Bool => "bool",
            DefaultFactoryClass::List => "list",
            DefaultFactoryClass::Dict => "dict",
            DefaultFactoryClass::Set => "set",
            DefaultFactoryClass::Deque => "collections.deque",
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

/// The Counter methods whose lowering the compiler owns.
///
/// Both are here for CPython's optional arguments: `most_common()` means
/// "every entry" (`None`), not an empty list, and `get` on a Counter is
/// `dict.get`, so a missing key is `None` where only `c[k]` answers 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterMethod {
    MostCommon,
    Get,
}

impl CounterMethod {
    pub(crate) fn from_name(name: &str) -> Option<CounterMethod> {
        match name {
            "most_common" => Some(CounterMethod::MostCommon),
            "get" => Some(CounterMethod::Get),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            CounterMethod::MostCommon => "most_common",
            CounterMethod::Get => "get",
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
#[cfg(test)]
mod tests {
    use super::CollectionsType;

    const ALL: [CollectionsType; 6] = [
        CollectionsType::Counter,
        CollectionsType::Deque,
        CollectionsType::Defaultdict,
        CollectionsType::OrderedDict,
        CollectionsType::ChainMap,
        CollectionsType::Namedtuple,
    ];

    #[test]
    fn name_round_trips_through_from_name() {
        for item in ALL {
            assert_eq!(CollectionsType::from_name(item.name()), Some(item));
        }
        // Parsing is exact: CPython's names are case-sensitive.
        assert_eq!(CollectionsType::from_name("Deque"), None);
        assert_eq!(CollectionsType::from_name("chainmap"), None);
    }

    #[test]
    fn every_name_either_converts_or_is_rejected_with_a_rewrite() {
        // No name may fall between the two: a runtime class with neither a
        // construction lowering nor a rejection renders as a bare call
        // that fails in rustc.
        for item in ALL {
            let converts = item.has_construction_lowering() || item == CollectionsType::Namedtuple;
            assert_ne!(
                converts,
                item.unlowered_rewrite().is_some(),
                "{:?} must either lower or be rejected with a rewrite",
                item
            );
        }
    }
}
