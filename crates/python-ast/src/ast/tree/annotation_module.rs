//! The annotation/decorator modules the compiler recognizes but that have
//! NO stdpython runtime counterpart — the complement of
//! [`StdModule`](super::std_module::StdModule).
//!
//! These modules are consumed at CONVERSION time (annotations map to Rust
//! types, decorators rewrite the definitions, `typing.cast` lowers to its
//! value argument), so their imports lower to nothing and their names never
//! resolve under the runtime crate. They are NOT in `StdModule` for a
//! reason: `is_stdpython_module` routes imports under the runtime crate,
//! and no `stdpython::typing` exists. The AGENTS.md boundary rule applies
//! the same way — one `from_name` match per name set, every consumer works
//! with the enum.

/// A module the compiler knows only at conversion time (annotations,
/// decorators, TYPE_CHECKING imports), not backed by a stdpython runtime
/// module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnnotationModule {
    Typing,
    TypingExtensions,
    Contextlib,
    Abc,
    Dataclasses,
}

impl AnnotationModule {
    /// Parse a module name at the AST boundary — the ONE place this name
    /// set exists as strings.
    pub(crate) fn from_name(name: &str) -> Option<AnnotationModule> {
        Some(match name {
            "typing" => AnnotationModule::Typing,
            "typing_extensions" => AnnotationModule::TypingExtensions,
            "contextlib" => AnnotationModule::Contextlib,
            "abc" => AnnotationModule::Abc,
            "dataclasses" => AnnotationModule::Dataclasses,
            _ => return None,
        })
    }
}

/// `from_name(name) == Some(Typing)` — the check the overwhelming majority
/// of annotation sites want; the other variants are reached at their few
/// dedicated sites. One predicate instead of scattered `== "typing"`
/// literals.
pub(crate) fn is_typing(name: &str) -> bool {
    AnnotationModule::from_name(name) == Some(AnnotationModule::Typing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The from_name boundary, pinned: each known name maps to exactly its
    /// variant, and unknown names are rejected.
    #[test]
    fn from_name_boundary() {
        assert_eq!(
            AnnotationModule::from_name("typing"),
            Some(AnnotationModule::Typing)
        );
        assert_eq!(
            AnnotationModule::from_name("typing_extensions"),
            Some(AnnotationModule::TypingExtensions)
        );
        assert_eq!(
            AnnotationModule::from_name("contextlib"),
            Some(AnnotationModule::Contextlib)
        );
        assert_eq!(
            AnnotationModule::from_name("abc"),
            Some(AnnotationModule::Abc)
        );
        assert_eq!(
            AnnotationModule::from_name("dataclasses"),
            Some(AnnotationModule::Dataclasses)
        );
        assert_eq!(AnnotationModule::from_name("typo"), None);
        assert_eq!(AnnotationModule::from_name(""), None);
        assert!(is_typing("typing"));
        assert!(!is_typing("typing_extensions"));
        assert!(!is_typing("functools"));
    }
}
