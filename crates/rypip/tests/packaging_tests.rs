//! Tests for pip-style package discovery (pyproject.toml / setup.cfg /
//! setup.py) and PEP 440/508 dependency resolution.

use std::fs;
use std::path::{Path, PathBuf};

mod common;

use common::Scratch;
use rypip::package::discover;
use rypip::packaging::{read_project_metadata, resolve_package_dirs};
use rypip::resolve::{
    matches_specifier, parse_requirement, parse_version, version_cmp, version_satisfies,
};

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
    // The shim is OPT-IN (RYPIP_ALLOW_SETUP_PY_EXEC=1): executing a
    // downloaded sdist's setup.py runs third-party code on this host, so
    // the default path must never fire. This test exercises the opt-in.
    // SAFETY: single-threaded test body; no other thread reads the env.
    unsafe {
        std::env::set_var("RYPIP_ALLOW_SETUP_PY_EXEC", "1");
    }
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

/// Write a pure-Python wheel (a zip) with the given `path -> contents`
/// entries into `dir`, named `{dist}-{version}-py3-none-any.whl`.
fn write_wheel(dir: &Path, dist: &str, version: &str, files: &[(&str, &str)]) -> PathBuf {
    use std::io::Write;
    fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{dist}-{version}-py3-none-any.whl"));
    let mut zip = zip::ZipWriter::new(fs::File::create(&path).unwrap());
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
    for (name, contents) in files {
        zip.start_file(*name, options).unwrap();
        zip.write_all(contents.as_bytes()).unwrap();
    }
    let info = format!("{dist}-{version}.dist-info");
    zip.start_file(format!("{info}/METADATA"), options).unwrap();
    zip.write_all(format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n").as_bytes())
        .unwrap();
    zip.start_file(format!("{info}/top_level.txt"), options).unwrap();
    zip.write_all(format!("{dist}\n").as_bytes()).unwrap();
    zip.finish().unwrap();
    record_digest(&path);
    path
}

/// Write the digest sidecar a verified download leaves beside a cached
/// artifact (the offline path reuses an artifact only through it).
fn record_digest(artifact: &Path) {
    use sha2::{Digest, Sha256};
    let digest: String = Sha256::digest(fs::read(artifact).unwrap())
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    fs::write(sidecar_of(artifact), format!("{digest}\n")).unwrap();
}

fn sidecar_of(artifact: &Path) -> PathBuf {
    let mut name = artifact.file_name().unwrap().to_os_string();
    name.push(".sha256");
    artifact.with_file_name(name)
}

/// Write an sdist tarball `{dist}-{version}.tar.gz` whose top directory
/// holds PKG-INFO, pyproject.toml and the package, plus its digest.
fn write_sdist(dir: &Path, dist: &str, version: &str, package: &str) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let artifact = dir.join(format!("{dist}-{version}.tar.gz"));
    {
        let gz = flate2::write::GzEncoder::new(
            fs::File::create(&artifact).unwrap(),
            flate2::Compression::default(),
        );
        let mut tar = tar::Builder::new(gz);
        let mut add = |name: &str, body: &str| {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, body.as_bytes()).unwrap();
        };
        let top = format!("{dist}-{version}");
        add(
            &format!("{top}/PKG-INFO"),
            &format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n"),
        );
        add(
            &format!("{top}/pyproject.toml"),
            &format!("[project]\nname = \"{dist}\"\nversion = \"{version}\"\n"),
        );
        add(
            &format!("{top}/{package}/__init__.py"),
            &format!("VERSION = \"{version}\"\n"),
        );
        tar.into_inner().unwrap().finish().unwrap();
    }
    record_digest(&artifact);
    artifact
}

#[test]
fn cached_versions_of_one_distribution_extract_apart() {
    // Two versions of one distribution in the cache: each resolves to its
    // OWN extracted tree. They used to extract into one shared directory,
    // so a module the newer version added (idna 3.19's cli.py) survived
    // into the older version's tree and was converted as part of it.
    let scratch = Scratch::new("cache-versions");
    let cache = scratch.path().join("cache");
    let dist_dir = cache.join("idna");
    write_wheel(
        &dist_dir,
        "idna",
        "1.0",
        &[("idna/__init__.py", "VERSION = \"1.0\"\n")],
    );
    write_wheel(
        &dist_dir,
        "idna",
        "2.0",
        &[
            ("idna/__init__.py", "VERSION = \"2.0\"\n"),
            ("idna/cli.py", "def main() -> None:\n    pass\n"),
        ],
    );

    // A cached artifact satisfying the requirement resolves offline: the
    // cache is the artifact, extraction is on demand.
    let newer = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("idna==2.0").unwrap(),
        true,
    )
    .expect("idna 2.0 from the cache");
    let older = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("idna==1.0").unwrap(),
        true,
    )
    .expect("idna 1.0 from the cache");
    assert_eq!(newer.version, "2.0");
    assert_eq!(older.version, "1.0");
    assert_eq!(newer.import_name, "idna");
    assert_ne!(newer.path, older.path);
    assert!(newer.path.join("cli.py").is_file(), "{}", newer.path.display());
    assert!(!older.path.join("cli.py").exists(), "{}", older.path.display());
    assert_eq!(fs::read_to_string(older.path.join("__init__.py")).unwrap(), "VERSION = \"1.0\"\n");
    // Each artifact extracts under its own stem.
    assert!(newer.path.starts_with(dist_dir.join("extracted/idna-2.0-py3-none-any")));
    assert!(older.path.starts_with(dist_dir.join("extracted/idna-1.0-py3-none-any")));

    // The extracted trees are reused on the next resolution.
    let again = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("idna==1.0").unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(again.path, older.path);

    // A version the cache does not hold is still the loud offline error.
    let err = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("idna==3.0").unwrap(),
        true,
    )
    .expect_err("idna 3.0 is not cached");
    assert!(err.to_string().contains("offline"), "{err:?}");

    // An UNPINNED requirement resolves the NEWEST satisfying cached
    // version — the choice online resolution makes — whatever order the
    // file system lists the cache in (Devin review on #342).
    for _ in 0..3 {
        let newest = rypip::resolve::resolve_dependency_in(
            &cache,
            &parse_requirement("idna>=1").unwrap(),
            true,
        )
        .unwrap();
        assert_eq!(newest.version, "2.0");
        assert_eq!(newest.path, newer.path);
    }
    // A complete extraction carries its marker, naming the artifact.
    let marker = dist_dir.join("extracted/idna-2.0-py3-none-any/.rypip-complete");
    assert_eq!(
        fs::read_to_string(&marker).unwrap().trim(),
        "idna-2.0-py3-none-any.whl"
    );
}

#[test]
fn an_interrupted_extraction_is_never_a_cache_hit() {
    // A tree at the extraction path WITHOUT the completion marker (an
    // interrupted run that got as far as the .dist-info) is not reused:
    // the artifact is extracted again, atomically, and the complete tree
    // replaces the partial one (Devin review on #342).
    let scratch = Scratch::new("cache-partial");
    let cache = scratch.path().join("cache");
    let dist_dir = cache.join("idna");
    write_wheel(&dist_dir, "idna", "1.0", &[("idna/__init__.py", "VERSION = \"1.0\"\n")]);
    let partial = dist_dir.join("extracted/idna-1.0-py3-none-any");
    fs::create_dir_all(partial.join("idna-1.0.dist-info")).unwrap();
    fs::write(partial.join("idna-1.0.dist-info/METADATA"), "truncated").unwrap();
    // No package directory, no marker.

    let dep = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("idna==1.0").unwrap(),
        true,
    )
    .expect("the artifact is extracted again");
    assert_eq!(dep.path, partial.join("idna"));
    assert!(dep.path.join("__init__.py").is_file(), "{}", dep.path.display());
    assert!(partial.join(".rypip-complete").is_file());
    assert_eq!(
        fs::read_to_string(partial.join("idna-1.0.dist-info/top_level.txt")).unwrap(),
        "idna\n"
    );
    // No in-progress directory is left behind.
    let leftovers: Vec<String> = fs::read_dir(dist_dir.join("extracted"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains(".partial-"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn a_cached_artifact_is_reused_only_through_its_verified_digest() {
    // The offline path trusts an artifact only through the digest a
    // verified download recorded beside it: no sidecar (an older cache)
    // means the artifact is not a cache hit, a mismatch is a loud error
    // naming the file, never a silent extraction (Devin review on #342).
    let scratch = Scratch::new("cache-digest");
    let cache = scratch.path().join("cache");
    let dist_dir = cache.join("idna");
    let artifact = write_wheel(&dist_dir, "idna", "1.0", &[("idna/__init__.py", "")]);
    let req = parse_requirement("idna==1.0").unwrap();

    fs::remove_file(sidecar_of(&artifact)).unwrap();
    let err = rypip::resolve::resolve_dependency_in(&cache, &req, true)
        .expect_err("an unverified artifact is not reused offline");
    assert!(err.to_string().contains("offline"), "{err:?}");

    fs::write(sidecar_of(&artifact), format!("{}\n", "0".repeat(64))).unwrap();
    let err = rypip::resolve::resolve_dependency_in(&cache, &req, true)
        .expect_err("an altered artifact is loud");
    let text = err.to_string();
    assert!(text.contains("sha256 mismatch"), "{text}");
    assert!(text.contains("idna-1.0-py3-none-any.whl"), "{text}");
    assert!(!dist_dir.join("extracted/idna-1.0-py3-none-any").exists());

    record_digest(&artifact);
    let dep = rypip::resolve::resolve_dependency_in(&cache, &req, true).unwrap();
    assert_eq!(dep.version, "1.0");
}

#[test]
fn cached_sdist_prereleases_keep_their_version() {
    // An sdist stem's version is everything after the distribution name
    // — a hyphen-separated prerelease included — so `pkg-1.0-rc1` is
    // 1.0rc1, not the final 1.0 (Devin review on #342); a wheel's version
    // is the one component before its tags.
    let scratch = Scratch::new("cache-prerelease");
    let cache = scratch.path().join("cache");
    let dist_dir = cache.join("pre");
    write_sdist(&dist_dir, "pre", "1.0-rc1", "pre");
    write_sdist(&dist_dir, "pre", "0.9", "pre");

    let rc = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("pre==1.0rc1").unwrap(),
        true,
    )
    .expect("the prerelease resolves under its own version");
    assert_eq!(rc.version, "1.0rc1");
    assert!(rc.path.join("__init__.py").is_file(), "{}", rc.path.display());
    assert!(rc.path.starts_with(dist_dir.join("extracted/pre-1.0-rc1")));

    let err = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("pre==1.0").unwrap(),
        true,
    )
    .expect_err("the prerelease is not the final release");
    assert!(err.to_string().contains("offline"), "{err:?}");

    let stable = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("pre==0.9").unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(stable.version, "0.9");
}

#[test]
fn cached_sdist_resolves_to_its_package_directory() {
    // An sdist tarball extracts to `{stem}/{dist}-{version}/{package}`;
    // the resolved path is the importable package directory.
    let scratch = Scratch::new("cache-sdist");
    let cache = scratch.path().join("cache");
    let dist_dir = cache.join("charset_normalizer");
    write_sdist(&dist_dir, "charset-normalizer", "3.4.0", "charset_normalizer");
    let dep = rypip::resolve::resolve_dependency_in(
        &cache,
        &parse_requirement("charset-normalizer>=3").unwrap(),
        true,
    )
    .expect("sdist from the cache");
    assert_eq!(dep.version, "3.4.0");
    assert_eq!(dep.import_name, "charset_normalizer");
    assert!(dep.path.join("__init__.py").is_file(), "{}", dep.path.display());
    assert!(dep.path.starts_with(dist_dir.join("extracted/charset-normalizer-3.4.0")));
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
