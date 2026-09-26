//! Python package metadata and layout discovery, the way Python's own
//! tooling resolves them: PEP 621 `pyproject.toml` `[project]` +
//! `[tool.setuptools]`, legacy `setup.cfg`, and `setup.py` (read statically
//! first; executed through a `python3` shim only when the static read fails
//! AND execution is opted into — see [`SetupPyExec`]).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// The importable packages and metadata of a Python project.
#[derive(Debug, Clone, Default)]
pub struct ProjectMetadata {
    /// `[project] name` / `[metadata] name` / setup.py name.
    pub name: Option<String>,
    /// `[project] version` / `[metadata] version` / setup.py version.
    pub version: Option<String>,
    /// PEP 508 requirement strings from `[project] dependencies`,
    /// `install_requires`, or setup.py's install_requires.
    pub dependencies: Vec<String>,
    /// Importable package names (dotted, relative to the project root),
    /// each mapping to a directory containing `__init__.py`. The
    /// [`RYTHON_FIND_SENTINEL`] marker means "discover with setuptools
    /// find_packages semantics".
    pub packages: Vec<String>,
    /// Single-file modules (py-modules), e.g. "spam" for spam.py.
    pub py_modules: Vec<String>,
    /// package-dir mapping (package name → directory relative to root);
    /// the "" key sets the base for all packages.
    pub package_dir: HashMap<String, String>,
    /// `[tool.setuptools.packages.find] where` directories (relative to the
    /// project root) to search for packages; defaults to ["."].
    pub find_where: Vec<String>,
}

/// Marker used by the setup.py shim and find-configs for `find_packages()`.
pub const RYTHON_FIND_SENTINEL: &str = "__RYTHON_FIND_PACKAGES__";

/// Whether metadata resolution may EXECUTE a setup.py (through the python3
/// shim) when the static reader cannot determine its metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupPyExec {
    /// Never run setup.py (the default).
    Skip,
    /// Run it under the shim: its top-level code executes on this host.
    Allow,
}

impl SetupPyExec {
    /// The process-wide opt-in: `Allow` exactly when
    /// `RYPIP_ALLOW_SETUP_PY_EXEC=1`.
    pub fn from_env() -> Self {
        if std::env::var("RYPIP_ALLOW_SETUP_PY_EXEC").ok().as_deref() == Some("1") {
            SetupPyExec::Allow
        } else {
            SetupPyExec::Skip
        }
    }
}

/// The `setup()` keyword arguments metadata resolution reads; every other
/// keyword is ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SetupKwarg {
    Name,
    Version,
    InstallRequires,
    Packages,
    PyModules,
    PackageDir,
}

impl SetupKwarg {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "name" => SetupKwarg::Name,
            "version" => SetupKwarg::Version,
            "install_requires" => SetupKwarg::InstallRequires,
            "packages" => SetupKwarg::Packages,
            "py_modules" => SetupKwarg::PyModules,
            "package_dir" => SetupKwarg::PackageDir,
            _ => return None,
        })
    }
}

/// Read the project's packaging metadata, merging pyproject.toml (PEP 621,
/// authoritative), then setup.cfg, then setup.py for whatever is missing.
/// Whether setup.py may be executed comes from `RYPIP_ALLOW_SETUP_PY_EXEC`
/// ([`SetupPyExec::from_env`]).
pub fn read_project_metadata(root: &Path) -> Result<ProjectMetadata> {
    read_project_metadata_with(root, SetupPyExec::from_env())
}

/// [`read_project_metadata`] with the setup.py execution opt-in passed
/// explicitly rather than read from the environment.
pub fn read_project_metadata_with(root: &Path, exec: SetupPyExec) -> Result<ProjectMetadata> {
    let mut meta = ProjectMetadata::default();

    // PEP 621 + setuptools config in pyproject.toml.
    let pyproject = root.join("pyproject.toml");
    if pyproject.is_file() {
        let text = fs::read_to_string(&pyproject)
            .with_context(|| format!("reading {}", pyproject.display()))?;
        let value: toml::Value = text
            .parse()
            .with_context(|| format!("parsing {}", pyproject.display()))?;
        if let Some(project) = value.get("project").and_then(|v| v.as_table()) {
            meta.name = project
                .get("name")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            meta.version = project
                .get("version")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            if let Some(deps) = project.get("dependencies").and_then(|v| v.as_array()) {
                meta.dependencies = deps
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect();
            }
        }
        if let Some(tool) = value
            .get("tool")
            .and_then(|v| v.get("setuptools"))
            .and_then(|v| v.as_table())
        {
            if let Some(packages) = tool.get("packages").and_then(|v| v.as_array()) {
                meta.packages = packages
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect();
            }
            if let Some(py_modules) = tool.get("py-modules").and_then(|v| v.as_array()) {
                meta.py_modules = py_modules
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect();
            }
            if let Some(pkg_dir) = tool.get("package-dir").and_then(|v| v.as_table()) {
                for (pkg, dir) in pkg_dir {
                    if let Some(dir) = dir.as_str() {
                        meta.package_dir.insert(pkg.clone(), dir.to_string());
                    }
                }
            }
            if let Some(find) = tool.get("packages").and_then(|v| v.get("find")) {
                apply_find_config(find, &mut meta);
            }
        }
    }

    // Legacy setup.cfg.
    let setup_cfg = root.join("setup.cfg");
    if setup_cfg.is_file() {
        let text = fs::read_to_string(&setup_cfg)
            .with_context(|| format!("reading {}", setup_cfg.display()))?;
        let cfg = parse_config_ini(&text);
        if let Some(metadata) = cfg.get("metadata") {
            if meta.name.is_none() {
                meta.name = metadata.get("name").cloned();
            }
            if meta.version.is_none() {
                meta.version = metadata.get("version").cloned();
            }
        }
        if let Some(options) = cfg.get("options") {
            if meta.packages.is_empty() {
                if let Some(packages) = options.get("packages") {
                    meta.packages = split_comma_list(packages);
                }
            }
            if meta.py_modules.is_empty() {
                if let Some(py_modules) = options.get("py_modules") {
                    meta.py_modules = split_comma_list(py_modules);
                }
            }
            if meta.dependencies.is_empty() {
                if let Some(requires) = options.get("install_requires") {
                    meta.dependencies = requires
                        .lines()
                        .map(str::trim)
                        .filter(|s| !s.is_empty() && !s.starts_with('#'))
                        .map(str::to_string)
                        .collect();
                }
            }
        }
        if let Some(find) = cfg.get("options.packages.find") {
            // Same keys as [tool.setuptools.packages.find], INI-flavoured.
            let mut find_toml = toml::map::Map::new();
            for (k, v) in find {
                find_toml.insert(k.clone(), toml::Value::String(v.clone()));
            }
            apply_find_config(&toml::Value::Table(find_toml), &mut meta);
        }
    }

    // setup.py: the STATIC parse runs FIRST — executing a downloaded
    // sdist's setup.py is arbitrary third-party code at conversion time
    // (the supply-chain vector flagged in review). The python3 shim (which
    // stubs setuptools/distutils but still execs the file) only runs when
    // the static parse yields nothing — matching pip's own ordering, where
    // static metadata wins and setup.py execution is the legacy fallback.
    let setup_py = root.join("setup.py");
    if setup_py.is_file() {
        let captured = match statically_parse_setup_py(&setup_py) {
            Ok(captured) => captured,
            Err(static_err) => run_setup_py_shim(&setup_py, exec, &static_err).unwrap_or_default(),
        };
        if meta.name.is_none() {
            meta.name = captured.get(&SetupKwarg::Name).cloned();
        }
        if meta.version.is_none() {
            meta.version = captured.get(&SetupKwarg::Version).cloned();
        }
        if meta.dependencies.is_empty() {
            if let Some(deps) = captured.get(&SetupKwarg::InstallRequires) {
                meta.dependencies = split_python_list(deps);
            }
        }
        if meta.packages.is_empty() {
            match captured.get(&SetupKwarg::Packages).map(String::as_str) {
                Some(RYTHON_FIND_SENTINEL) => {
                    meta.packages.push(RYTHON_FIND_SENTINEL.to_string());
                }
                Some(list) => meta.packages = split_python_list(list),
                None => {}
            }
        }
        if meta.py_modules.is_empty() {
            if let Some(list) = captured.get(&SetupKwarg::PyModules) {
                meta.py_modules = split_python_list(list);
            }
        }
        if meta.package_dir.is_empty() {
            if let Some(dir) = captured.get(&SetupKwarg::PackageDir) {
                if let Ok(map) = parse_python_dict(dir) {
                    meta.package_dir = map;
                }
            }
        }
    }

    Ok(meta)
}

/// Apply a `packages.find` config table (where/include/exclude) to the
/// metadata.
fn apply_find_config(find: &toml::Value, meta: &mut ProjectMetadata) {
    if meta.packages.is_empty() {
        meta.packages.push(RYTHON_FIND_SENTINEL.to_string());
    }
    if let Some(where_list) = find.get("where").and_then(|v| v.as_array()) {
        meta.find_where = where_list
            .iter()
            .filter_map(|v| v.as_str())
            .map(str::to_string)
            .collect();
    }
}

/// Resolve the metadata's package list into concrete importable package
/// directories relative to `root` (each containing __init__.py), applying
/// setuptools' find semantics for the sentinel.
pub fn resolve_package_dirs(root: &Path, meta: &ProjectMetadata) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for package in &meta.packages {
        if package == RYTHON_FIND_SENTINEL {
            for found in find_packages_dir(root, &meta.find_where)? {
                if seen.insert(found.clone()) {
                    out.push(PathBuf::from(found));
                }
            }
            continue;
        }
        // Dotted package names nest: `a.b` is the package at a/b.
        let package_path = package.replace('.', "/");
        // package-dir mapping: `{"": "src"}` or `{"pkg": "lib/pkg"}`.
        let dir = meta
            .package_dir
            .get(package)
            .or_else(|| meta.package_dir.get(""))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&package_path));
        let abs = root.join(&dir);
        if !abs.join("__init__.py").is_file() {
            bail!(
                "package `{}` declared by the project metadata has no __init__.py at {}",
                package,
                abs.display()
            );
        }
        if seen.insert(dir.to_string_lossy().to_string()) {
            out.push(dir);
        }
    }

    Ok(out)
}

/// setuptools find_packages(): recursively collect directories containing
/// __init__.py (skipping hidden and underscore-prefixed dirs), returning
/// FILESYSTEM-relative paths (with `/` separators, `src/` container kept):
/// a `where = ["src"]` layout yields `src/mylib`, `src/mylib/sub`. The
/// caller joins them onto the project root for the path and derives the
/// import name by stripping the layout container.
pub fn find_packages_dir(root: &Path, where_dirs: &[String]) -> Result<Vec<String>> {
    let bases: Vec<PathBuf> = if where_dirs.is_empty() {
        vec![root.to_path_buf()]
    } else {
        where_dirs.iter().map(|w| root.join(w)).collect()
    };
    let mut found = Vec::new();
    for base in bases {
        if !base.is_dir() {
            bail!(
                "[tool.setuptools.packages.find] where={} is not a directory",
                base.display()
            );
        }
        collect_find(root, &base, &mut found)?;
    }
    found.sort();
    found.dedup();
    Ok(found)
}

fn collect_find(root: &Path, dir: &Path, found: &mut Vec<String>) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name.starts_with('_') {
            continue;
        }
        if path.join("__init__.py").is_file() {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            found.push(rel);
        }
        collect_find(root, &path, found)?;
    }
    Ok(())
}

/// Run setup.py under a python3 shim that records the setup() call without
/// executing any real setuptools code. Returns the captured kwargs, or None
/// when python3 is unavailable or the file cannot be executed.
///
/// EXECUTION IS OPT-IN: the shim runs the sdist's own top-level code on
/// this host, so it only fires under [`SetupPyExec::Allow`]
/// (`RYPIP_ALLOW_SETUP_PY_EXEC=1`). Without it the call is skipped with a
/// loud note naming why the static reader gave up (`static_err`), and
/// resolution falls back to the layout heuristics (correct-or-loud).
fn run_setup_py_shim(
    setup_py: &Path,
    exec: SetupPyExec,
    static_err: &anyhow::Error,
) -> Option<HashMap<SetupKwarg, String>> {
    if exec == SetupPyExec::Skip {
        eprintln!(
            "note: skipping setup.py execution for `{}`: {static_err:#} (set \
             RYPIP_ALLOW_SETUP_PY_EXEC=1 to allow running legacy sdists' setup.py during \
             metadata resolution)",
            setup_py.display()
        );
        return None;
    }
    let shim = r#"
import json, sys, types
captured = {}
def _setup(**kw):
    captured.update({k: (v if isinstance(v, str) else repr(v)) for k, v in kw.items()})
def _find_packages(*a, **k):
    return "__RYTHON_FIND_PACKAGES__"
def _find_namespace_packages(*a, **k):
    return "__RYTHON_FIND_PACKAGES__"
st = types.ModuleType("setuptools")
st.setup = _setup
st.find_packages = _find_packages
st.find_namespace_packages = _find_namespace_packages
st.__version__ = "0.0.0"
sys.modules["setuptools"] = st
try:
    import distutils.core as dc
    dc.setup = _setup
except Exception:
    pass
ns = {"__file__": sys.argv[1], "__name__": "__main__"}
try:
    src = open(sys.argv[1], encoding="utf-8").read()
    exec(compile(src, sys.argv[1], "exec"), ns)
except SystemExit:
    pass
except BaseException:
    # setup.py may import real setuptools submodules we cannot shim; the
    # caller falls back to the static parse.
    captured = {}
print("RYTHON_SETUP_RESULT " + json.dumps(captured, sort_keys=True))
"#;
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(shim)
        .arg(setup_py)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("RYTHON_SETUP_RESULT "))?;
    let payload = line.trim_start_matches("RYTHON_SETUP_RESULT ");
    let captured: HashMap<String, String> = serde_json::from_str(payload).ok()?;
    Some(
        captured
            .into_iter()
            .filter_map(|(k, v)| SetupKwarg::from_name(&k).map(|k| (k, v)))
            .collect(),
    )
}

/// Pull the setup() keyword arguments out of the file without executing
/// anything. Only LITERAL values are read; a consumed keyword whose value
/// is computed (`version=__version__`) fails the static read — loudly,
/// through the caller's note — rather than being taken verbatim.
fn statically_parse_setup_py(setup_py: &Path) -> Result<HashMap<SetupKwarg, String>> {
    let src = fs::read_to_string(setup_py)
        .with_context(|| format!("reading {}", setup_py.display()))?;
    let mut out = HashMap::new();

    // Locate `setup(` (identifier `setup` immediately followed by `(`).
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i + 5 < bytes.len() {
        if bytes[i..i + 5] == ['s', 'e', 't', 'u', 'p']
            && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != '_')
        {
            let mut j = i + 5;
            while j < bytes.len() && bytes[j].is_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == '(' {
                if let Some((kw, end)) = skim_setup_call(&bytes, j) {
                    for (name, value) in kw {
                        let Some(kwarg) = SetupKwarg::from_name(&name) else {
                            continue;
                        };
                        let Some(value) = value else {
                            bail!(
                                "setup()'s `{name}=` in {} is not a literal, so it cannot be \
                                 read without executing the file",
                                setup_py.display()
                            );
                        };
                        out.entry(kwarg).or_insert(value);
                    }
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }

    if out.is_empty() {
        bail!(
            "could not determine the package from {} (no setuptools.setup(...) call with \
             literal metadata keywords found); convert a directory with pyproject.toml or \
             setup.cfg instead",
            setup_py.display()
        );
    }
    Ok(out)
}

/// A skimmed call's keyword arguments in source order: `None` marks a
/// computed (non-literal) value.
type SkimmedKwargs = Vec<(String, Option<String>)>;

/// Skim one `setup(...)` call (`open` is the index of its `(`), returning
/// its keyword arguments and the index just past the call's closing paren.
/// A value is `Some(text)` when it is a literal — a string (quotes dropped),
/// a list/dict of strings (element quotes kept, as in the shim's `repr`, so
/// a comma inside an element stays inside it), or a bare `find_packages()`
/// — and `None` when it is computed.
fn skim_setup_call(bytes: &[char], open: usize) -> Option<(SkimmedKwargs, usize)> {
    // Nesting INSIDE the argument list: the call's own `(` is not a level,
    // so scanning starts just past it (counting it made every top-level
    // `,` and the closing `)` look nested, and no call was ever read).
    let mut depth = 0usize;
    let mut i = open + 1;
    let mut in_str: Option<char> = None;
    let mut out = Vec::new();
    let mut kw: Option<String> = None;
    let mut value_chars: Vec<char> = Vec::new();
    let mut seen_equals = false;
    // Whether the value so far is literal, and whether the last token was
    // a string (a second one adjacent to it is implicit concatenation).
    let mut literal = true;
    let mut after_string = false;

    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = in_str {
            if c == '\\' {
                if i + 1 < bytes.len() {
                    value_chars.push(bytes[i + 1]);
                    i += 2;
                    continue;
                }
            }
            if c == q {
                // Closing quote: part of the value only inside a list/dict.
                in_str = None;
                after_string = true;
                if depth > 0 {
                    value_chars.push(c);
                }
            } else {
                value_chars.push(c);
            }
            i += 1;
            continue;
        }
        let adjacent_string = after_string;
        if !c.is_whitespace() {
            after_string = false;
        }
        match c {
            '#' => {
                // A comment runs to the end of the line.
                while i < bytes.len() && bytes[i] != '\n' {
                    i += 1;
                }
            }
            '\'' | '"' => {
                if seen_equals {
                    in_str = Some(c);
                    literal &= !adjacent_string;
                    if depth > 0 {
                        value_chars.push(c);
                    }
                }
                i += 1;
            }
            '=' if kw.is_some() && !seen_equals => {
                seen_equals = true;
                i += 1;
            }
            '(' | '[' | '{' => {
                depth += 1;
                if seen_equals {
                    value_chars.push(c);
                    literal &= c != '(';
                }
                i += 1;
            }
            ')' => {
                if depth == 0 {
                    if let (Some(k), true) = (kw, seen_equals) {
                        out.push((k, finalize_value(&value_chars, literal)));
                    }
                    return Some((out, i + 1));
                }
                depth -= 1;
                if seen_equals {
                    value_chars.push(c);
                    literal = false;
                }
                i += 1;
            }
            ']' | '}' => {
                depth = depth.saturating_sub(1);
                if seen_equals {
                    value_chars.push(c);
                }
                i += 1;
            }
            ',' if depth == 0 => {
                if let (Some(k), true) = (kw, seen_equals) {
                    out.push((k, finalize_value(&value_chars, literal)));
                }
                kw = None;
                seen_equals = false;
                literal = true;
                value_chars.clear();
                i += 1;
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                if seen_equals {
                    value_chars.push(c);
                    literal = false;
                    i += 1;
                } else if kw.is_none() {
                    let start = i;
                    while i < bytes.len()
                        && (bytes[i].is_ascii_alphanumeric() || bytes[i] == '_')
                    {
                        i += 1;
                    }
                    kw = Some(bytes[start..i].iter().collect());
                } else {
                    value_chars.push(c);
                    i += 1;
                }
            }
            _ => {
                if seen_equals {
                    value_chars.push(c);
                    // List/dict separators are literal; anything else
                    // outside a string (a number, an operator) is not.
                    literal &= c.is_whitespace() || c == ',' || c == ':';
                }
                i += 1;
            }
        }
    }
    None
}

fn finalize_value(chars: &[char], literal: bool) -> Option<String> {
    let s: String = chars.iter().collect();
    let s = s.trim();
    // `packages=find_packages()` / `find_namespace_packages()` lower to the
    // discovery sentinel. Only the bare call: a `where=` or `exclude=`
    // argument changes what they find, so that form stays computed.
    if s == "find_packages()" || s == "find_namespace_packages()" {
        return Some(RYTHON_FIND_SENTINEL.to_string());
    }
    literal.then(|| s.to_string())
}

/// Split a Python list/tuple literal (`['a>=1,<2', 'b']`, as the static
/// reader and the shim's `repr` both render it) into its string elements.
/// Commas inside a quoted element belong to it: a PEP 508 specifier set is
/// itself comma-separated.
fn split_python_list(s: &str) -> Vec<String> {
    let s = s.trim();
    let inner = s
        .strip_prefix(['[', '('])
        .and_then(|s| s.strip_suffix([']', ')']))
        .unwrap_or(s);
    let mut items = Vec::new();
    let mut cur = String::new();
    let mut in_str: Option<char> = None;
    for c in inner.chars() {
        match in_str {
            Some(q) if c == q => in_str = None,
            Some(_) => cur.push(c),
            None => match c {
                '\'' | '"' => in_str = Some(c),
                ',' => items.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            },
        }
    }
    items.push(cur);
    items
        .into_iter()
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

fn split_comma_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .map(|s| s.trim_matches(['\'', '"']).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse a Python dict literal repr like `{'': 'src', 'pkg': 'lib'}`.
fn parse_python_dict(s: &str) -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    let s = s.trim();
    let inner = s
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .context("package_dir is not a dict literal")?;
    let mut depth = 0usize;
    let mut in_str: Option<char> = None;
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in inner.chars() {
        if let Some(q) = in_str {
            cur.push(c);
            if c == q {
                in_str = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => {
                in_str = Some(c);
                cur.push(c);
            }
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            ',' if depth == 0 => parts.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    for part in parts {
        let (k, v) = part.split_once(':').context("package_dir entry has no ':'")?;
        out.insert(
            k.trim().trim_matches(['\'', '"']).to_string(),
            v.trim().trim_matches(['\'', '"']).to_string(),
        );
    }
    Ok(out)
}

/// Minimal INI parser for setup.cfg (section headers and key = value lines,
/// with indented continuation lines appended to the previous key).
fn parse_config_ini(text: &str) -> HashMap<String, HashMap<String, String>> {
    let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current: Option<String> = None;
    let mut last_key: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current = Some(trimmed[1..trimmed.len() - 1].to_string());
            sections.entry(current.clone().unwrap()).or_default();
            last_key = None;
            continue;
        }
        if let Some(section) = &current {
            // An INDENTED line is a continuation of the previous value —
            // checked before key parsing because a continued dependency
            // (`    dep1>=1.0`) contains an `=` of its own.
            if line.starts_with(' ') || line.starts_with('\t') {
                if let Some(key) = &last_key {
                    if let Some(entry) = sections.get_mut(section).unwrap().get_mut(key) {
                        entry.push('\n');
                        entry.push_str(trimmed);
                    }
                }
            } else if let Some((k, v)) = trimmed.split_once('=') {
                let key = k.trim().to_string();
                let val = v.trim().to_string();
                sections.get_mut(section).unwrap().insert(key.clone(), val);
                last_key = Some(key);
            }
        }
    }
    sections
}
