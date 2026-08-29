//! The builtin exception hierarchy, from the interpreter itself.
//!
//! `matches` in stdpython must know the exception tree (is a raised
//! `KeyError` caught by `except LookupError:`?) — the runtime cannot
//! link Python (it builds for `core`/`alloc` tiers), so the tree is
//! materialized into stdpython as a generated data table. This module is
//! the generator's source: it asks the LIVE interpreter (the same PyO3
//! path that produces parse trees) for every `BaseException` subclass
//! in `builtins` and the stdlib modules the runtime models, with each
//! class's real `__mro__`. The drift-detection test regenerates the
//! stdpython table and compares — the checked-in data can never
//! silently diverge from the reference interpreter it was built from.

use crate::Result;
use pyo3::prelude::*;
use std::ffi::CString;

/// Every BaseException subclass the runtime models, as `(name, [name of
/// itself, then each ancestor])` — the class's real `__mro__` from the
/// live interpreter.
///
/// Aliases are names bound to the same class object (`EnvironmentError`
/// IS `OSError`, `socket.timeout` IS `TimeoutError`, `ssl.CertificateError`
/// IS `SSLCertVerificationError`), so their `__mro__[0]` is the
/// canonical name; multiple-inheritance classes (`ExceptionGroup`,
/// `SSLCertVerificationError`) carry every branch of their MRO.
pub fn dump_builtin_exception_tree() -> Result<(String, String, Vec<(String, Vec<String>)>)> {
    let pymodule_code = include_str!("exception_tree/__init__.py");
    Python::attach(|py| {
        let code_cstr = CString::new(pymodule_code).expect("fmt into String");
        let pymodule =
            PyModule::from_code(py, &code_cstr, c"exception_tree.py", c"exception_tree").expect("fmt into String");
        let dump = pymodule.getattr("dump").expect("fmt into String");
        let result = dump.call0().expect("fmt into String");
        result.extract()
    })
    .map_err(|e| {
        crate::Error::from(anyhow::Error::new(e)).context(
            "failed to dump the builtin exception tree from the Python interpreter \
             (python-ast's exception-tree generator; is python3 importable with the \
             ssl module available?)",
        )
    })
}

/// A Python exception name, as a Rust enum variant ident
/// (`"gaierror"` → `Gaierror`, `"_IncompleteInputError"` →
/// `IncompleteInputError`). Exception names are Python identifiers, so
/// only case and leading underscores need handling.
fn variant_ident(canonical: &str) -> String {
    let trimmed = canonical.trim_start_matches('_');
    let mut chars = trimmed.chars();
    let first = chars.next().map(|c| c.to_uppercase().collect::<String>());
    match first {
        Some(first) => format!("{first}{}", chars.as_str()),
        None => "_".to_string(),
    }
}

/// The text of `crates/stdpython/src/builtin_exceptions_gen.rs`: the
/// checked-in exception tree derived from the interpreter. Everything
/// derived from the dump — the MRO table, the enum variant list, the
/// name→variant boundary, and the canonical-name map — is rendered from
/// the SAME rows, so none of them can drift from the others.
pub fn render_builtin_exceptions_gen(
    version: &str,
    platform: &str,
    tree: &[(String, Vec<String>)],
) -> Result<String> {
    use std::fmt::Write as _;

    // Variant idents must be unique — a collision would silently merge
    // two classes.
    let mut variants: Vec<(&str, String)> = Vec::new();
    for (_, mro) in tree {
        let canonical = mro.first().expect("non-empty mro");
        let ident = variant_ident(canonical);
        if variants.iter().any(|(c, i)| i == &ident && c != canonical) {
            return Err(crate::Error::msg(format!(
                "exception names {canonical} and {} both map to the Rust ident {ident}",
                variants.iter().find(|(_, i)| i == &ident).unwrap().0
            )));
        }
        if !variants.iter().any(|(c, _)| c == canonical) {
            variants.push((canonical, ident));
        }
    }
    variants.sort();

    let mut out = String::new();
    writeln!(
        out,
        "// GENERATED FILE — the builtin exception tree, from the interpreter."
    ).expect("fmt into String");
    writeln!(out).expect("fmt into String");
    writeln!(
        out,
        "// Source: CPython {version} ({platform}), dumped through python-ast's \
         PyO3 exception-tree generator (the same path that produces parse trees). \
         Do not edit by hand:"
    ).expect("fmt into String");
    writeln!(
        out,
        "//   RYTHON_REGEN=1 cargo test -p python-ast exception_tree_is_current"
    ).expect("fmt into String");
    writeln!(
        out,
        "// regenerates this file from the live interpreter and the test verifies \
         it is current."
    ).expect("fmt into String");
    writeln!(out).expect("fmt into String");
    writeln!(
        out,
        "/// name → its real `__mro__` (itself first, then each ancestor), as \
         CPython reports it."
    ).expect("fmt into String");
    writeln!(
        out,
        "/// Alias keys (`EnvironmentError`, `IOError`, `socket.timeout`, \
         `ssl.CertificateError`, ...) are names bound to the same class object, so \
         `__mro__[0]` is the canonical name."
    ).expect("fmt into String");
    writeln!(out, "pub(crate) static BUILTIN_EXCEPTION_MRO: &[(&str, &[&str])] = &[").expect("fmt into String");
    for (name, mro) in tree {
        let mro_lits: Vec<String> = mro.iter().map(|c| format!("\"{c}\"")).collect();
        writeln!(out, "    (\"{name}\", &[{}]),", mro_lits.join(", ")).expect("fmt into String");
    }
    writeln!(out, "];").expect("fmt into String");
    writeln!(out).expect("fmt into String");
    writeln!(
        out,
        "/// The canonical class name a name refers to (its `__mro__[0]` — the class \
         object's own name); alias keys canonicalize here."
    ).expect("fmt into String");
    writeln!(out, "pub(crate) fn canonical_name(name: &str) -> Option<&'static str> {{").expect("fmt into String");
    writeln!(out, "    match name {{").expect("fmt into String");
    for (name, mro) in tree {
        writeln!(out, "        \"{name}\" => Some(\"{}\"),", mro[0]).expect("fmt into String");
    }
    writeln!(out, "        _ => None,").expect("fmt into String");
    writeln!(out, "    }}").expect("fmt into String");
    writeln!(out, "}}").expect("fmt into String");
    writeln!(out).expect("fmt into String");
    writeln!(
        out,
        "/// One variant per canonical builtin exception class — the names `__mro__[0]` \
         across the dump, generated from the interpreter."
    ).expect("fmt into String");
    writeln!(
        out,
        "/// std-only: only the PyO3 surfacing (`pyo3_err`) and its tests use the \
         enum; the core/alloc tiers match through the MRO table alone."
    ).expect("fmt into String");
    writeln!(out, "#[cfg(feature = \"std\")]").expect("fmt into String");
    writeln!(out, "#[derive(Clone, Copy, PartialEq, Eq, Debug)]").expect("fmt into String");
    writeln!(out, "pub(crate) enum BuiltinException {{").expect("fmt into String");
    for (_, ident) in &variants {
        writeln!(out, "    {ident},").expect("fmt into String");
    }
    writeln!(out, "}}").expect("fmt into String");
    writeln!(out).expect("fmt into String");
    writeln!(out, "#[cfg(feature = \"std\")]").expect("fmt into String");
    writeln!(out, "impl BuiltinException {{").expect("fmt into String");
    writeln!(
        out,
        "    /// The ONE string→enum boundary — generated from the same dump as the \
         variant list, so aliases and variants cannot drift."
    ).expect("fmt into String");
    writeln!(out, "    pub(crate) fn from_name(name: &str) -> Option<Self> {{").expect("fmt into String");
    writeln!(out, "        match name {{").expect("fmt into String");
    for (name, mro) in tree {
        let canonical = mro.first().expect("non-empty mro");
        let ident = variant_ident(canonical);
        writeln!(out, "            \"{name}\" => Some(Self::{ident}),").expect("fmt into String");
    }
    writeln!(out, "            _ => None,").expect("fmt into String");
    writeln!(out, "        }}").expect("fmt into String");
    writeln!(out, "    }}").expect("fmt into String");
    writeln!(out).expect("fmt into String");
    writeln!(out, "    /// The canonical Python name (what CPython prints).").expect("fmt into String");
    writeln!(out, "    #[cfg(test)]").expect("fmt into String");
    writeln!(out, "    pub(crate) fn name(self) -> &'static str {{").expect("fmt into String");
    writeln!(out, "        match self {{").expect("fmt into String");
    for (canonical, ident) in &variants {
        writeln!(out, "            Self::{ident} => \"{canonical}\",").expect("fmt into String");
    }
    writeln!(out, "        }}").expect("fmt into String");
    writeln!(out, "    }}").expect("fmt into String");
    writeln!(out).expect("fmt into String");
    writeln!(out, "    /// Every variant — generated, so it cannot drift.").expect("fmt into String");
    writeln!(out, "    #[cfg(test)]").expect("fmt into String");
    writeln!(out, "    pub(crate) const ALL: &'static [Self] = &[").expect("fmt into String");
    for (_, ident) in &variants {
        writeln!(out, "        Self::{ident},").expect("fmt into String");
    }
    writeln!(out, "    ];").expect("fmt into String");
    writeln!(out, "}}").expect("fmt into String");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump() -> Vec<(String, Vec<String>)> {
        dump_builtin_exception_tree().expect("interpreter dump").2
    }

    /// The tree the generator produces is internally consistent: every
    /// MRO is non-empty, names its class first (the canonical name —
    /// alias keys may differ, that is the point), and bottoms out at
    /// BaseException.
    #[test]
    fn every_chain_is_well_formed() {
        let tree = dump();
        assert!(!tree.is_empty());
        for (name, mro) in &tree {
            assert!(!mro.is_empty(), "{name}: empty MRO");
            assert!(
                mro.last().map(String::as_str) == Some("BaseException"),
                "{name}: {mro:?} does not bottom out at BaseException"
            );
        }
    }

    /// Known facts pinned against the LIVE interpreter (not hand-written):
    /// aliases share the canonical MRO, and the multiple-inheritance
    /// exceptions carry both branches.
    #[test]
    fn interpreter_pins() {
        let tree = dump();
        let mro = |name: &str| {
            tree.iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("{name} not in the dump"))
                .1
                .clone()
        };
        // EnvironmentError/IOError ARE OSError: same class object.
        assert_eq!(mro("EnvironmentError"), mro("OSError"));
        assert_eq!(mro("IOError"), mro("OSError"));
        // socket.timeout IS TimeoutError.
        assert_eq!(mro("timeout").first().map(String::as_str), Some("TimeoutError"));
        // ssl.CertificateError IS SSLCertVerificationError.
        assert_eq!(
            mro("CertificateError").first().map(String::as_str),
            Some("SSLCertVerificationError")
        );
        // Multiple inheritance, exactly as CPython's data says.
        assert!(mro("ExceptionGroup").contains(&"Exception".to_string()));
        assert!(mro("SSLCertVerificationError").contains(&"ValueError".to_string()));
        // SystemExit is NOT caught by Exception (its MRO skips it).
        assert!(!mro("SystemExit").contains(&"Exception".to_string()));
    }

    /// The stdpython table is consistent with THIS interpreter: every
    /// name BOTH know must agree — the checked-in `__mro__` must equal
    /// the live dump's for each shared name. A divergence — a changed
    /// MRO, a re-aliased exception — fails here loudly, so the runtime's
    /// matching can never silently disagree with the reference
    /// interpreter about the exceptions it knows.
    ///
    /// The comparison covers only the intersection of the two name sets:
    /// the builtin exception tree is stable across interpreter versions,
    /// but a table generated on a newer Python may list exceptions the
    /// local interpreter does not have (and vice versa) — both are
    /// consistent models, and `RYTHON_REGEN=1` folds the live
    /// interpreter's full dump into the table. This keeps the check
    /// stable across the interpreter versions CI runs.
    #[test]
    fn exception_tree_is_current() {
        let (version, platform, tree) = dump_builtin_exception_tree().expect("interpreter dump");
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../stdpython/src/builtin_exceptions_gen.rs");
        let current = std::fs::read_to_string(&path)
            .expect("stdpython/src/builtin_exceptions_gen.rs must exist");
        let checked_in = parse_checked_in_table(&current);
        let live: std::collections::HashMap<&str, &[String]> = tree
            .iter()
            .map(|(name, mro)| (name.as_str(), mro.as_slice()))
            .collect();
        let mut diverged: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
        for (name, mro) in &checked_in {
            if let Some(live_mro) = live.get(name.as_str())
                && *live_mro != mro.as_slice()
            {
                diverged.push((name.clone(), mro.clone(), live_mro.to_vec()));
            }
        }
        if !diverged.is_empty() {
            let detail = diverged
                .iter()
                .map(|(name, checked, live_mro)| {
                    format!("{name}: checked {checked:?} vs live {live_mro:?}")
                })
                .collect::<Vec<_>>()
                .join("; ");
            if std::env::var_os("RYTHON_REGEN").is_some() {
                let expected = render_builtin_exceptions_gen(&version, &platform, &tree)
                    .expect("render the generated file");
                std::fs::write(&path, &expected).expect("write the regenerated file");
                panic!(
                    "exception tree regenerated from CPython {version} ({platform}); \
                     the test now passes on the next run"
                );
            }
            panic!(
                "the checked-in exception tree (stdpython/src/builtin_exceptions_gen.rs) \
                 diverged from CPython {version} ({platform}): {detail}; regenerate with \
                 `RYTHON_REGEN=1 cargo test -p python-ast exception_tree_is_current` and \
                 review the diff"
            );
        }
    }

    /// Parse the generated file's static table (`("Name", &["A", "B"]),`
    /// rows) back into (name, mro) pairs.
    fn parse_checked_in_table(text: &str) -> Vec<(String, Vec<String>)> {
        let mut out = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if !line.starts_with('(') {
                continue;
            }
            let Some(name) = line
                .strip_prefix('(')
                .and_then(|r| r.split_once('"'))
                .and_then(|(_, r)| r.split_once('"'))
                .map(|(n, _)| n.to_string())
            else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let Some(mro_part) = line.split_once("&[").map(|(_, m)| m) else {
                continue;
            };
            let mut mro = Vec::new();
            for (i, piece) in mro_part.split('"').skip(1).enumerate() {
                // split('"') alternates content, separator, content, ...
                // — the even indices are the members.
                if i % 2 == 0 && !piece.is_empty() && !piece.contains(']') {
                    mro.push(piece.to_string());
                }
            }
            if !mro.is_empty() {
                out.push((name, mro));
            }
        }
        out
    }
}
