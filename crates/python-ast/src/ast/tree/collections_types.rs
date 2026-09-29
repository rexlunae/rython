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
            CollectionsType::Deque | CollectionsType::Defaultdict | CollectionsType::OrderedDict
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
            CollectionsType::Counter => Some(
                "count into a plain dict instead — `counts: dict[str, int] = {}` \
                 and `counts[x] = counts.get(x, 0) + 1` per element, read with \
                 `counts.get(x, 0)` (a Counter's missing key counts 0)",
            ),
            CollectionsType::Deque
            | CollectionsType::Defaultdict
            | CollectionsType::OrderedDict
            | CollectionsType::Namedtuple => None,
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
