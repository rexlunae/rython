//! Tests for pip-style package discovery (pyproject.toml / setup.cfg /
//! setup.py) and PEP 440/508 dependency resolution.

use std::fs;
use std::path::{Path, PathBuf};

use rypip::package::discover;
use rypip::packaging::{read_project_metadata, resolve_package_dirs};
use rypip::resolve::{
    matches_specifier, parse_requirement, parse_version, version_cmp, version_satisfies,
};

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rypip-pkgtest-{}-{}",
            tag,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("creating scratch dir");
        Scratch(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// PEP 440 version comparison and specifiers
// ---------------------------------------------------------------------------

#[test]
fn version_comparison_follows_pep440() {
    let v = |s: &str| parse_version(s).unwrap_or_else(|| panic!("parse {s}"));
    use std::cmp::Ordering;
    // release segments zero-pad
    assert_eq!(version_cmp(&v("1.2"), &v("1.2.0")), Ordering::Equal);
    assert_eq!(version_cmp(&v("1.2.1"), &v("1.2.0")), Ordering::Greater);
    // pre < final < post
    assert_eq!(version_cmp(&v("1.0a1"), &v("1.0")), Ordering::Less);
    assert_eq!(version_cmp(&v("1.0rc1"), &v("1.0")), Ordering::Less);
    assert_eq!(version_cmp(&v("1.0rc1"), &v("1.0b1")), Ordering::Greater);
    assert_eq!(version_cmp(&v("1.0"), &v("1.0.post1")), Ordering::Less);
    // dev sorts before everything of its base
    assert_eq!(version_cmp(&v("1.0.dev1"), &v("1.0a1")), Ordering::Less);
    assert_eq!(version_cmp(&v("1.0.dev2"), &v("1.0.dev1")), Ordering::Greater);
    // a .devN release sorts BEFORE its own base (PEP 440)
    assert_eq!(version_cmp(&v("1.0.dev1"), &v("1.0")), Ordering::Less);
    assert_eq!(version_cmp(&v("1.0a1.dev1"), &v("1.0a1")), Ordering::Less);
    assert_eq!(
        version_cmp(&v("1.0.post1.dev1"), &v("1.0.post1")),
        Ordering::Less
    );
    // epochs
    assert_eq!(version_cmp(&v("2!1.0"), &v("1!9.9")), Ordering::Greater);
}

#[test]
fn specifiers_match_pep440() {
    let v = |s: &str| parse_version(s).unwrap();
    assert!(matches_specifier(&v("1.4.5"), ">=", "1.4.5"));
    assert!(!matches_specifier(&v("1.4.4"), ">=", "1.4.5"));
    assert!(matches_specifier(&v("2.0"), "<", "3.0"));
    assert!(matches_specifier(&v("1.2.3"), "==", "1.2.3"));
    assert!(matches_specifier(&v("1.2.9"), "==", "1.2.*"));
    assert!(!matches_specifier(&v("1.3.0"), "==", "1.2.*"));
    assert!(matches_specifier(&v("2.1.0"), "~=", "2.0"));
    assert!(matches_specifier(&v("2.0.1"), "~=", "2.0"));
    assert!(!matches_specifier(&v("3.0.0"), "~=", "2.0"));
    assert!(matches_specifier(&v("1.4.9"), "~=", "1.4.5"));
    assert!(!matches_specifier(&v("1.5.0"), "~=", "1.4.5"));
    assert!(matches_specifier(&v("1.5.0"), "!=", "1.4.5"));
    assert!(matches_specifier(&v("2.0.0"), ">=", "1.0"));
    assert!(!matches_specifier(&v("0.9.0"), ">=", "1.0"));
}

#[test]
fn requirements_parse_names_and_specifiers() {
    let req = parse_requirement("requests>=2.0,<3").unwrap();
    assert_eq!(req.name, "requests");
    assert_eq!(req.specifiers, vec![(">=".into(), "2.0".into()), ("<".into(), "3".into())]);

    let req = parse_requirement("python-dateutil[tz]>=2.8 ; python_version < '3.12'").unwrap();
    assert_eq!(req.name, "python_dateutil");
    assert_eq!(req.extras, vec!["tz".to_string()]);
    assert!(req.marker.is_some());

    let req = parse_requirement("numpy==1.26.*").unwrap();
    assert_eq!(req.specifiers, vec![("==".into(), "1.26.*".into())]);
}

#[test]
fn version_satisfies_combines_specifiers() {
    let v = parse_version("2.5.1").unwrap();
    assert!(version_satisfies(
        &v,
        &[(">=".into(), "2.0".into()), ("<".into(), "3.0".into())]
    ));
    assert!(!version_satisfies(
        &v,
        &[(">=".into(), "2.0".into()), ("<".into(), "2.5".into())]
    ));
}

// ---------------------------------------------------------------------------
// pyproject.toml (PEP 621 + [tool.setuptools])
// ---------------------------------------------------------------------------

#[test]
fn pyproject_pep621_metadata_and_layout() {
    let scratch = Scratch::new("pep621");
    fs::write(
        scratch.path().join("pyproject.toml"),
        concat!(
            "[project]\n",
            "name = \"greeter\"\n",
            "version = \"1.2.3\"\n",
            "dependencies = [\"requests>=2.0\", \"textlib\"]\n",
            "\n",
            "[tool.setuptools]\n",
            "packages = [\"greeter\", \"greeter.util\"]\n",
        ),
    )
    .unwrap();
    let pkg = scratch.path().join("greeter");
    fs::create_dir_all(pkg.join("util")).unwrap();
    fs::write(pkg.join("__init__.py"), "from util import helper\n").unwrap();
    fs::write(pkg.join("util/__init__.py"), "def helper() -> int:\n    return 1\n").unwrap();
    fs::write(
        pkg.join("main.py"),
        "def main() -> str:\n    return \"hi\"\n",
    )
    .unwrap();

    let pkg = discover(scratch.path()).expect("discover");
    assert_eq!(pkg.name, "greeter");
    assert_eq!(pkg.version, "1.2.3");
    assert_eq!(pkg.dependencies, vec!["requests>=2.0".to_string(), "textlib".to_string()]);
    // The explicit packages list + recursion collect the tree once.
    let paths: Vec<String> = pkg
        .modules
        .iter()
        .map(|m| m.path.join("."))
        .collect();
    assert!(paths.contains(&"greeter".to_string()), "{:?}", paths);
    assert!(paths.contains(&"greeter.util".to_string()), "{:?}", paths);
    assert!(paths.contains(&"greeter.main".to_string()), "{:?}", paths);
    // No duplicates.
    assert_eq!(paths.len(), 3, "{:?}", paths);
}

#[test]
fn pyproject_find_packages_with_src_layout() {
    let scratch = Scratch::new("find-src");
    fs::write(
        scratch.path().join("pyproject.toml"),
        concat!(
            "[project]\n",
            "name = \"mylib\"\n",
            "version = \"0.5.0\"\n",
            "\n",
            "[tool.setuptools.packages.find]\n",
            "where = [\"src\"]\n",
        ),
    )
    .unwrap();
    let pkg_dir = scratch.path().join("src").join("mylib");
    fs::create_dir_all(pkg_dir.join("sub")).unwrap();
    fs::write(pkg_dir.join("__init__.py"), "").unwrap();
    fs::write(pkg_dir.join("sub/__init__.py"), "").unwrap();
    fs::write(pkg_dir.join("core.py"), "def f() -> int:\n    return 1\n").unwrap();

    let pkg = discover(scratch.path()).expect("discover");
    assert_eq!(pkg.name, "mylib");
    let paths: Vec<String> = pkg.modules.iter().map(|m| m.path.join(".")).collect();
    assert!(paths.contains(&"mylib".to_string()), "{:?}", paths);
    assert!(paths.contains(&"mylib.core".to_string()), "{:?}", paths);
    assert!(paths.contains(&"mylib.sub".to_string()), "{:?}", paths);
}

#[test]
fn pyproject_src_layout_without_explicit_packages_falls_back() {
    // A pyproject with only [project] name and a src/ package: the
    // historical heuristics still locate it.
    let scratch = Scratch::new("src-fallback");
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"greeter\"\nversion = \"1.0.0\"\n",
    )
    .unwrap();
    let pkg_dir = scratch.path().join("src").join("greeter");
    fs::create_dir_all(&pkg_dir).unwrap();
    fs::write(pkg_dir.join("__init__.py"), "").unwrap();

    let pkg = discover(scratch.path()).expect("discover");
    assert_eq!(pkg.name, "greeter");
    assert_eq!(pkg.version, "1.0.0");
}

// ---------------------------------------------------------------------------
// setup.cfg
// ---------------------------------------------------------------------------

#[test]
fn setup_cfg_metadata_and_install_requires() {
    let scratch = Scratch::new("setupcfg");
    fs::write(
        scratch.path().join("setup.cfg"),
        concat!(
            "[metadata]\n",
            "name = legacy_pkg\n",
            "version = 0.3.1\n",
            "\n",
            "[options]\n",
            "packages = legacy_pkg\n",
            "install_requires =\n",
            "    dep1>=1.0\n",
            "    dep2\n",
        ),
    )
    .unwrap();
    let pkg_dir = scratch.path().join("legacy_pkg");
    fs::create_dir_all(&pkg_dir).unwrap();
    fs::write(pkg_dir.join("__init__.py"), "").unwrap();

    let pkg = discover(scratch.path()).expect("discover");
    assert_eq!(pkg.name, "legacy_pkg");
    assert_eq!(pkg.version, "0.3.1");
    assert_eq!(pkg.dependencies, vec!["dep1>=1.0".to_string(), "dep2".to_string()]);
}

// ---------------------------------------------------------------------------
// setup.py (python3 shim)
// ---------------------------------------------------------------------------

#[test]
fn setup_py_shim_extracts_metadata() {
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: python3 not available");
        return;
    }
    let scratch = Scratch::new("setuppy");
    fs::write(
        scratch.path().join("setup.py"),
        concat!(
            "from setuptools import setup, find_packages\n",
            "\n",
            "setup(\n",
            "    name='shim_pkg',\n",
            "    version='2.0.0',\n",
            "    packages=find_packages(),\n",
            "    install_requires=['depx>=1.0', 'depy'],\n",
            ")\n",
        ),
    )
    .unwrap();
    let pkg_dir = scratch.path().join("shim_pkg");
    fs::create_dir_all(&pkg_dir).unwrap();
    fs::write(pkg_dir.join("__init__.py"), "").unwrap();
    fs::write(pkg_dir.join("mod.py"), "def f() -> int:\n    return 1\n").unwrap();

    let meta = read_project_metadata(scratch.path()).unwrap();
    assert_eq!(meta.name.as_deref(), Some("shim_pkg"));
    assert_eq!(meta.version.as_deref(), Some("2.0.0"));
    assert_eq!(
        meta.dependencies,
        vec!["depx>=1.0".to_string(), "depy".to_string()]
    );
    // find_packages() lowered to the discovery sentinel; resolve it.
    assert!(meta.packages.contains(&rypip::packaging::RYTHON_FIND_SENTINEL.to_string()));

    let dirs = resolve_package_dirs(scratch.path(), &meta).unwrap();
    assert_eq!(dirs.len(), 1, "{:?}", dirs);
    let pkg = discover(scratch.path()).expect("discover");
    let paths: Vec<String> = pkg.modules.iter().map(|m| m.path.join(".")).collect();
    assert!(paths.contains(&"shim_pkg".to_string()), "{:?}", paths);
    assert!(paths.contains(&"shim_pkg.mod".to_string()), "{:?}", paths);
}

#[test]
fn setup_py_static_fallback_without_python3() {
    // The static parser must handle a plain setup(...) call even without
    // executing it (simulated by parsing the file directly).
    let scratch = Scratch::new("setuppy-static");
    fs::write(
        scratch.path().join("setup.py"),
        concat!(
            "from setuptools import setup\n",
            "setup(\n",
            "    name='static_pkg',\n",
            "    version='3.1.4',\n",
            "    packages=['static_pkg'],\n",
            "    install_requires=['onlydep'],\n",
            ")\n",
        ),
    )
    .unwrap();
    let pkg_dir = scratch.path().join("static_pkg");
    fs::create_dir_all(&pkg_dir).unwrap();
    fs::write(pkg_dir.join("__init__.py"), "").unwrap();

    let pkg = discover(scratch.path()).expect("discover");
    assert_eq!(pkg.name, "static_pkg");
    assert_eq!(pkg.version, "3.1.4");
    assert_eq!(pkg.dependencies, vec!["onlydep".to_string()]);
}

// ---------------------------------------------------------------------------
// Dependency resolution (offline cache + network-gated E2E)
// ---------------------------------------------------------------------------

#[test]
fn resolve_requirement_from_cache_requires_network_or_cache() {
    // Offline resolution of an uncached dependency is a loud error.
    unsafe { std::env::set_var("RYPIP_OFFLINE", "1") };
    let req = parse_requirement("this-package-definitely-does-not-exist-rython-test").unwrap();
    let err = rypip::resolve::resolve_dependency(&req, true).expect_err("offline + uncached");
    assert!(err.to_string().contains("offline"), "{:?}", err);
}

#[test]
fn resolve_dependency_end_to_end_from_pypi() {
    // Gated: requires network + curl. Run with RYPIP_TEST_NETWORK=1.
    if std::env::var_os("RYPIP_TEST_NETWORK").is_none() {
        eprintln!("skipping network test (set RYPIP_TEST_NETWORK=1 to run)");
        return;
    }
    let scratch = Scratch::new("resolve-e2e");
    let cache = scratch.path().join("cache");
    unsafe {
        std::env::set_var("RYPIP_CACHE_DIR", &cache);
        std::env::remove_var("RYPIP_OFFLINE");
    }

    // pylev: a tiny pure-Python package with a single module.
    let req = parse_requirement("pylev>=1.3").unwrap();
    let dep = rypip::resolve::resolve_dependency(&req, false).expect("resolve pylev");
    assert_eq!(dep.import_name, "pylev");
    assert!(dep.path.join("__init__.py").is_file() || dep.path.is_file(), "{:?}", dep.path);

    // A second resolution hits the cache (no fetch).
    let dep2 = rypip::resolve::resolve_dependency(&req, false).expect("resolve pylev cached");
    assert_eq!(dep2.path, dep.path);

    // And the cached distribution satisfies the specifier.
    let version = parse_version(&dep.version).unwrap();
    assert!(version_satisfies(&version, &req.specifiers));
}

// ---------------------------------------------------------------------------
// convert-level integration: metadata deps + vendoring merge
// ---------------------------------------------------------------------------

#[test]
fn pyproject_dependencies_merge_with_vendored_python_modules_offline() {
    // A project declares `dependencies` in pyproject.toml and vendors them
    // via rython.toml [python-modules]. RYPIP_OFFLINE=1 + the explicit
    // manifest must satisfy the resolution without any network: the
    // vendored copy wins and the conversion succeeds.
    let scratch = Scratch::new("deps-offline");
    fs::write(
        scratch.path().join("pyproject.toml"),
        concat!(
            "[project]\n",
            "name = \"depapp\"\n",
            "version = \"0.1.0\"\n",
            "dependencies = [\"pylev>=1.3\"]\n",
            "\n",
            "[tool.setuptools]\n",
            "packages = [\"depapp\"]\n",
        ),
    )
    .unwrap();
    fs::create_dir_all(scratch.path().join("depapp")).unwrap();
    fs::write(scratch.path().join("depapp/__init__.py"), "").unwrap();
    fs::write(
        scratch.path().join("depapp/main.py"),
        concat!(
            "import pylev\n",
            "\n",
            "def dist(a: str, b: str) -> int:\n",
            "    return pylev.wf_levenshtein(a, b)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(dist(\"kitten\", \"sitting\"))\n",
        ),
    )
    .unwrap();
    fs::create_dir_all(scratch.path().join("vendor")).unwrap();
    fs::write(
        scratch.path().join("vendor/pylev.py"),
        concat!(
            "def wf_levenshtein(a: str, b: str) -> int:\n",
            "    n = len(a)\n",
            "    m = len(b)\n",
            "    if n == 0:\n",
            "        return m\n",
            "    if m == 0:\n",
            "        return n\n",
            "    prev = [0] * (m + 1)\n",
            "    for j in range(m + 1):\n",
            "        prev[j] = j\n",
            "    for i in range(1, n + 1):\n",
            "        cur = [0] * (m + 1)\n",
            "        cur[0] = i\n",
            "        for j in range(1, m + 1):\n",
            "            cost = 0 if a[i - 1] == b[j - 1] else 1\n",
            "            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)\n",
            "        prev = cur\n",
            "    return prev[m]\n",
        ),
    )
    .unwrap();
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\npylev = { path = \"vendor/pylev.py\" }\n",
    )
    .unwrap();

    unsafe {
        std::env::set_var("RYPIP_OFFLINE", "1");
        std::env::set_var("RYPIP_CACHE_DIR", scratch.path().join("empty-cache"));
    }
    let pkg = discover(scratch.path()).expect("discover");
    assert_eq!(pkg.dependencies, vec!["pylev>=1.3".to_string()]);
    let out = scratch.path().join("crate");
    let krate = rypip::convert(&pkg, &out, &rypip::convert::ConvertOptions::default())
        .expect("offline conversion with a vendored dependency");

    // The vendored module was transpiled as a sibling module.
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(lib.contains("mod pylev"), "lib.rs: {}", lib);
    let _ = krate;
}

#[test]
fn no_deps_skips_resolution_entirely() {
    // --no-deps: a project with unfulfilled dependencies converts without
    // fetching (the dependency is simply not vendored).
    let scratch = Scratch::new("no-deps");
    fs::write(
        scratch.path().join("pyproject.toml"),
        concat!(
            "[project]\n",
            "name = \"nodeps\"\n",
            "version = \"0.1.0\"\n",
            "dependencies = [\"pylev>=1.3\"]\n",
        ),
    )
    .unwrap();
    fs::create_dir_all(scratch.path().join("nodeps")).unwrap();
    fs::write(scratch.path().join("nodeps/__init__.py"), "").unwrap();
    fs::write(
        scratch.path().join("nodeps/main.py"),
        "def hello() -> str:\n    return \"hi\"\n",
    )
    .unwrap();

    unsafe { std::env::set_var("RYPIP_OFFLINE", "1") };
    let pkg = discover(scratch.path()).expect("discover");
    let out = scratch.path().join("crate");
    rypip::convert(
        &pkg,
        &out,
        &rypip::convert::ConvertOptions {
            no_deps: true,
            ..Default::default()
        },
    )
    .expect("no-deps conversion skips the fetch");
}


#[test]
fn resolve_dependency_tree_pulls_transitives() {
    // Issue #113: resolving `requests` must also resolve its transitive
    // requirements (urllib3, certifi, idna, charset-normalizer) — the
    // gate on the whole library-conversion use case. Gated like the other
    // network test: run with RYPIP_TEST_NETWORK=1.
    if std::env::var_os("RYPIP_TEST_NETWORK").is_none() {
        eprintln!("skipping network test (set RYPIP_TEST_NETWORK=1 to run)");
        return;
    }
    let scratch = Scratch::new("resolve-tree");
    let cache = scratch.path().join("cache");
    unsafe {
        std::env::set_var("RYPIP_CACHE_DIR", &cache);
        std::env::remove_var("RYPIP_OFFLINE");
    }

    let req = parse_requirement("requests>=2.0").unwrap();
    let tree = rypip::resolve::resolve_dependency_tree(&req, false).expect("resolve tree");
    let names: Vec<&str> = tree.iter().map(|d| d.import_name.as_str()).collect();
    for expected in ["requests", "urllib3", "certifi", "idna", "charset_normalizer"] {
        assert!(
            names.contains(&expected),
            "transitive dependency `{expected}` must be resolved; got {names:?}"
        );
    }
    // Every resolved package is vendorable (has a path on disk).
    for dep in &tree {
        assert!(dep.path.exists(), "path for `{}` missing: {}", dep.import_name, dep.path.display());
    }
}

#[test]
fn parse_requirement_handles_parenthesized_specifiers() {
    // PEP 508 parenthesized form used by botocore/boto3 metadata:
    // `jmespath (<2.0.0,>=0.7.1)` — the paren must not leak into the name.
    let req = parse_requirement("jmespath (<2.0.0,>=0.7.1)").unwrap();
    assert_eq!(req.name, "jmespath");
    assert_eq!(req.specifiers, vec![("<".to_string(), "2.0.0".to_string()), (">=".to_string(), "0.7.1".to_string())]);

    let req = parse_requirement("urllib3 (!=2.2.0,<3,>=1.25.4)").unwrap();
    assert_eq!(req.name, "urllib3");
    assert_eq!(req.specifiers.len(), 3);
}
