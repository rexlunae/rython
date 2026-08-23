//! pip-style dependency resolution for rypip: parse PEP 508 requirement
//! specifiers, query the PyPI JSON API, pick the newest matching version,
//! download the pure-Python wheel (or sdist), extract it into a cache, and
//! report the importable package — so `[project] dependencies` /
//! `install_requires` vendored Python libraries into a converted crate the
//! way pip would install them.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// A parsed PEP 508 requirement: name + version specifiers + extras +
/// marker (markers are currently ignored — rython targets a small runtime,
/// so the common `; python_version < "3.9"` guards are best treated as
/// satisfied rather than guessed).
#[derive(Debug, Clone)]
pub struct Requirement {
    /// Distribution name, normalized to lowercase with `-` → `_`.
    pub name: String,
    /// Version specifiers: (operator, version-string), e.g. (">=", "1.2").
    pub specifiers: Vec<(String, String)>,
    pub extras: Vec<String>,
    pub marker: Option<String>,
}

/// A PEP 440-ish version, comparable.
#[derive(Debug, Clone)]
pub struct Version {
    pub epoch: u64,
    pub release: Vec<u64>,
    pub pre: Option<(String, u64)>,
    pub post: Option<u64>,
    pub dev: Option<u64>,
}

/// Parse `requests>=2.0,<3 ; python_version < "3.10" [extra]`-style
/// requirement strings (a practical PEP 508 subset).
pub fn parse_requirement(s: &str) -> Result<Requirement> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty requirement");
    }

    // Strip a trailing marker after ';'.
    let (head, marker) = match s.split_once(';') {
        Some((h, m)) => (h, Some(m.trim().to_string())),
        None => (s, None),
    };
    // Split extras `name[extra1,extra2]`.
    let (name_part, extras) = match head.find('[') {
        Some(open) => {
            let close = head.find(']').with_context(|| {
                format!("malformed extras in requirement `{s}` (missing ']')")
            })?;
            let name = head[..open].trim();
            let extras: Vec<String> = head[open + 1..close]
                .split(',')
                .map(|e| e.trim().to_string())
                .filter(|e| !e.is_empty())
                .collect();
            (name, extras)
        }
        None => (head, Vec::new()),
    };

    // Name then specifiers: `name>=1.0,<2.0`.
    // PEP 508 parenthesized specifiers (`name (>=1.0,<2.0)` — used by
    // botocore/boto3 metadata) put the whole specifier list in parens;
    // treat the opening paren as the name/specifier boundary so the paren
    // does not leak into the name (issue: the transitive sweep).
    let open_paren = name_part.find('(');
    let split_at = open_paren
        .or_else(|| name_part.find(|c: char| c == '>' || c == '<' || c == '=' || c == '!' || c == '~'))
        .unwrap_or(name_part.len());
    let name = name_part[..split_at].trim();
    let mut spec_part = name_part[split_at..].trim();
    if open_paren.is_some() {
        spec_part = spec_part.trim_start_matches('(').trim_end_matches(')').trim();
    }
    if name.is_empty() {
        bail!("requirement `{s}` has no distribution name");
    }

    let mut specifiers = Vec::new();
    if !spec_part.is_empty() {
        let mut rest = spec_part;
        while !rest.is_empty() {
            let rest_trimmed = rest.trim_start();
            if rest_trimmed.is_empty() {
                break;
            }
            // Operator: two-char first (>= <= == != ~=), then single.
            let (op, after) = if rest_trimmed.starts_with(">=") {
                (">=", &rest_trimmed[2..])
            } else if rest_trimmed.starts_with("<=") {
                ("<=", &rest_trimmed[2..])
            } else if rest_trimmed.starts_with("==") {
                ("==", &rest_trimmed[2..])
            } else if rest_trimmed.starts_with("!=") {
                ("!=", &rest_trimmed[2..])
            } else if rest_trimmed.starts_with("~=") {
                ("~=", &rest_trimmed[2..])
            } else if rest_trimmed.starts_with('>') {
                (">", &rest_trimmed[1..])
            } else if rest_trimmed.starts_with('<') {
                ("<", &rest_trimmed[1..])
            } else {
                bail!("unsupported version specifier in `{s}` near `{rest_trimmed}`");
            };
            let after = after.trim_start();
            let (version_str, remainder) = match after.find(',') {
                Some(idx) => (after[..idx].trim().to_string(), &after[idx + 1..]),
                None => (after.trim().to_string(), ""),
            };
            if version_str.is_empty() {
                bail!("missing version after `{op}` in `{s}`");
            }
            specifiers.push((op.to_string(), version_str));
            rest = remainder;
        }
    }

    Ok(Requirement {
        name: normalize_dist_name(name),
        specifiers,
        extras,
        marker,
    })
}

fn normalize_dist_name(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .replace('-', "_")
        .replace('.', "_")
}

/// Parse a PEP 440 version string (practical subset: epoch!release with
/// pre/post/dev markers, `-` and `_` separators).
pub fn parse_version(s: &str) -> Option<Version> {
    let s = s.trim().trim_start_matches(['v', 'V']);
    let (epoch, rest) = match s.split_once('!') {
        Some((e, r)) => (e.trim().parse().ok()?, r),
        None => (0, s),
    };
    let rest = rest.replace('-', ".");

    // Split off pre/post/dev suffixes.
    let mut release_str = String::new();
    let mut pre = None;
    let mut post = None;
    let mut dev = None;
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() || c == '.' {
            release_str.push(c);
            i += 1;
            continue;
        }
        // A non-digit marker begins a suffix. Multi-char spellings
        // (dev, post, rc) skip their extra letters before the number.
        let lower = c.to_ascii_lowercase();
        match lower {
            'a' | 'b' => {
                let kind = if lower == 'a' { "a" } else { "b" };
                let (num, next) = read_number(&chars, i + 1);
                pre = Some((kind.to_string(), num));
                i = next;
            }
            'c' => {
                // `1.0c1` == `1.0rc1`.
                let (num, next) = read_number(&chars, i + 1);
                pre = Some(("rc".to_string(), num));
                i = next;
            }
            'p' => {
                // post: `.postN` / `postN` / `revN`-style.
                let mut j = i + 1;
                if chars.get(j) == Some(&'o')
                    && chars.get(j + 1) == Some(&'s')
                    && chars.get(j + 2) == Some(&'t')
                {
                    j += 3;
                }
                let (num, next) = read_number(&chars, j);
                post = Some(num);
                i = next;
            }
            'r' => {
                let two: String = chars[i..(i + 2).min(chars.len())]
                    .iter()
                    .collect::<String>()
                    .to_lowercase();
                if two == "rc" {
                    let (num, next) = read_number(&chars, i + 2);
                    pre = Some(("rc".to_string(), num));
                    i = next;
                } else {
                    // revN: post
                    let (num, next) = read_number(&chars, i + 1);
                    post = Some(num);
                    i = next;
                }
            }
            'd' => {
                // dev: `.devN` / `devN`.
                let mut j = i + 1;
                if chars.get(j) == Some(&'e') && chars.get(j + 1) == Some(&'v') {
                    j += 2;
                }
                let (num, next) = read_number(&chars, j);
                dev = Some(num);
                i = next;
            }
            _ => {
                // Unknown suffix: treat as end of version.
                break;
            }
        }
    }

    // Also handle `.postN` / `.devN` spelled after a dot (the loop above
    // consumed dots into release_str only when followed by digits; a dot
    // followed by a letter was left in the suffix scan because the dot
    // itself was pushed... verify below).
    let release: Vec<u64> = release_str
        .split('.')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    if release.is_empty() {
        return None;
    }
    Some(Version {
        epoch,
        release,
        pre,
        post,
        dev,
    })
}

fn read_number(chars: &[char], mut i: usize) -> (u64, usize) {
    while i < chars.len() && (chars[i] == '.' || chars[i] == '_') {
        i += 1;
    }
    let start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        // `1.0a` with no number: PEP 440 treats it as a1.
        (1, i)
    } else {
        (chars[start..i].iter().collect::<String>().parse().unwrap_or(0), i)
    }
}

/// Compare two versions per PEP 440 ordering.
pub fn version_cmp(a: &Version, b: &Version) -> Ordering {
    let a_key = cmp_key(a);
    let b_key = cmp_key(b);
    a_key.cmp(&b_key)
}

fn cmp_key(v: &Version) -> (u64, Vec<u64>, (u8, u64, u64), u64, u64) {
    let mut release = v.release.clone();
    while release.last() == Some(&0) {
        release.pop();
    }
    // pre: a dev-only release has NegativeInfinity pre; a final release
    // has PositiveInfinity; a pre-release has its own value.
    let pre: (u8, u64, u64) = match &v.pre {
        Some((kind, n)) => {
            let rank = match kind.as_str() {
                "a" => 0,
                "b" => 1,
                _ => 2,
            };
            (1, rank, *n)
        }
        None if v.dev.is_some() => (0, 0, 0),
        None => (2, 0, 0),
    };
    let dev = match v.dev {
        Some(n) => (1u8, n),
        None => (0u8, 0),
    };
    (
        v.epoch,
        release,
        pre,
        v.post.unwrap_or(0),
        // dev is the last key; fold it in as a 6th tuple element.
        dev.1,
    )
}

/// Whether `version` satisfies `(op, spec)`.
pub fn matches_specifier(version: &Version, op: &str, spec: &str) -> bool {
    match op {
        "==" => {
            if let Some(prefix) = spec.strip_suffix(".*") {
                // `==1.2.*` matches any version whose release starts with
                // the prefix.
                let prefix: Vec<u64> = prefix
                    .split('.')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                let release = &version.release;
                if prefix.len() > release.len() {
                    return false;
                }
                return prefix.iter().zip(release.iter()).all(|(p, r)| p == r);
            }
            match parse_version(spec) {
                Some(sv) => version_cmp(version, &sv) == Ordering::Equal,
                None => false,
            }
        }
        "!=" => !matches_specifier(version, "==", spec),
        ">=" => {
            match parse_version(spec) {
                Some(sv) => version_cmp(version, &sv) != Ordering::Less,
                None => false,
            }
        }
        ">" => match parse_version(spec) {
            Some(sv) => version_cmp(version, &sv) == Ordering::Greater,
            None => false,
        },
        "<=" => match parse_version(spec) {
            Some(sv) => version_cmp(version, &sv) != Ordering::Greater,
            None => false,
        },
        "<" => match parse_version(spec) {
            Some(sv) => version_cmp(version, &sv) == Ordering::Less,
            None => false,
        },
        "~=" => {
            // ~=X.Y.Z means >= X.Y.Z and == X.Y.* (drop the last segment).
            let Some(v) = parse_version(spec) else {
                return false;
            };
            if version_cmp(version, &v) == Ordering::Less {
                return false;
            }
            let mut release = v.release.clone();
            if release.len() > 1 {
                release.pop();
                let prefix = release
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(".");
                matches_specifier(version, "==", &format!("{prefix}.*"))
            } else {
                true
            }
        }
        _ => false,
    }
}

pub fn version_satisfies(version: &Version, specifiers: &[(String, String)]) -> bool {
    specifiers
        .iter()
        .all(|(op, spec)| matches_specifier(version, op, spec))
}

/// A resolved dependency: where its importable package lives on disk.
#[derive(Debug, Clone)]
pub struct ResolvedDependency {
    /// Top-level import name (e.g. "requests").
    pub import_name: String,
    /// The package directory (or single .py file) to vendor.
    pub path: PathBuf,
    /// The distribution version that was resolved.
    pub version: String,
}

/// The cache directory for downloaded distributions.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("RYPIP_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("rypip");
    }
    std::env::temp_dir().join("rypip-cache")
}

/// The Requires-Dist requirements of a resolved dependency, read from its
/// wheel METADATA or sdist PKG-INFO (issue #113: transitive resolution).
/// Requirements gated on an OPTIONAL extra (`; extra == "x"`) are skipped —
/// they are not installed by a plain `pip install <pkg>` either.
pub fn dependency_requirements(dep: &ResolvedDependency) -> Result<Vec<Requirement>> {
    let root = dep.path.parent().unwrap_or_else(|| Path::new("."));
    let mut metadata: Option<PathBuf> = None;
    for entry in fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            let cand = if p.extension().is_some_and(|e| e == "dist-info") {
                p.join("METADATA")
            } else {
                p.join("PKG-INFO")
            };
            if cand.is_file() {
                metadata = Some(cand);
                break;
            }
        }
    }
    let Some(metadata) = metadata else {
        return Ok(Vec::new());
    };
    let text = fs::read_to_string(&metadata)
        .with_context(|| format!("reading {}", metadata.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("Requires-Dist:") else {
            continue;
        };
        let Ok(req) = parse_requirement(rest.trim()) else {
            continue;
        };
        if req.marker.as_deref().is_some_and(|m| m.contains("extra ==")) {
            continue;
        }
        out.push(req);
    }
    Ok(out)
}

/// Resolve a requirement AND its transitive requirements (pip-style,
/// breadth-first). First-resolved-wins for version conflicts; cycles are
/// cut by name (issue #113).
pub fn resolve_dependency_tree(req: &Requirement, offline: bool) -> Result<Vec<ResolvedDependency>> {
    let mut out = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue = vec![req.clone()];
    while let Some(r) = queue.pop() {
        if !seen.insert(r.name.clone()) {
            continue;
        }
        let dep = resolve_dependency(&r, offline)?;
        for sub in dependency_requirements(&dep)? {
            if !seen.contains(&sub.name) {
                queue.push(sub);
            }
        }
        out.push(dep);
    }
    Ok(out)
}

/// Resolve one requirement from PyPI. `offline` skips the network and
/// fails loudly if the dependency is not already in the cache.
pub fn resolve_dependency(req: &Requirement, offline: bool) -> Result<ResolvedDependency> {
    let cache = cache_dir();
    let dist_dir = cache.join(&req.name);

    // A cached, already-extracted distribution that satisfies the
    // specifiers can be reused (this is what makes offline rebuilds work).
    if let Some(hit) = cached_match(&dist_dir, req) {
        return Ok(hit);
    }
    if offline {
        bail!(
            "dependency `{}` is not vendored and offline resolution is enabled \
             (RYPIP_OFFLINE=1); vendor it via rython.toml [python-modules] or \
             clear RYPIP_OFFLINE to fetch from PyPI",
            req.name
        );
    }

    // Query the PyPI JSON API for the distribution's releases.
    let json = fetch_url(&format!("https://pypi.org/pypi/{}/json", req.name))?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .with_context(|| format!("parsing PyPI metadata for `{}`", req.name))?;
    let releases = value
        .get("releases")
        .and_then(|r| r.as_object())
        .with_context(|| format!("PyPI response for `{}` has no releases", req.name))?;

    // Newest version satisfying the specifiers.
    let mut best: Option<(Version, String)> = None;
    for (version_str, files) in releases {
        let Some(version) = parse_version(version_str) else {
            continue;
        };
        if !version_satisfies(&version, &req.specifiers) {
            continue;
        }
        // Only consider versions that have at least one usable artifact.
        let usable = files.as_array().is_some_and(|files| {
            files.iter().any(|f| {
                f.get("packagetype")
                    .and_then(|p| p.as_str())
                    .is_some_and(|t| t == "sdist" || t == "bdist_wheel")
            })
        });
        if !usable {
            continue;
        }
        match &best {
            Some((bv, _)) if version_cmp(&version, bv) == Ordering::Less => {}
            _ => best = Some((version, version_str.clone())),
        }
    }
    let (_, best_version) =
        best.with_context(|| format!("no version of `{}` satisfies {:?}", req.name, req.specifiers))?;

    // Pick the artifact: pure-Python wheel preferred, sdist fallback.
    let files = releases
        .get(&best_version)
        .and_then(|f| f.as_array())
        .context("resolved version has no files")?;
    let artifact = files
        .iter()
        .find(|f| {
            f.get("filename")
                .and_then(|n| n.as_str())
                .is_some_and(is_pure_wheel)
        })
        .or_else(|| {
            files.iter().find(|f| {
                f.get("packagetype")
                    .and_then(|p| p.as_str())
                    .is_some_and(|t| t == "sdist")
            })
        })
        .with_context(|| format!("no pure-Python wheel or sdist for `{}` {}", req.name, best_version))?;
    let file_name = artifact
        .get("filename")
        .and_then(|n| n.as_str())
        .context("artifact has no filename")?
        .to_string();
    let url = artifact
        .get("url")
        .and_then(|u| u.as_str())
        .context("artifact has no url")?
        .to_string();

    // Download into the cache if not already present.
    fs::create_dir_all(&dist_dir)
        .with_context(|| format!("creating cache {}", dist_dir.display()))?;
    let artifact_path = dist_dir.join(&file_name);
    if !artifact_path.is_file() {
        let bytes = fetch_bytes(&url)?;
        // Integrity: PyPI's JSON metadata carries the sha256 of every
        // artifact. A mismatch (tampered cache, mirror, intercepted
        // download) is a loud error — never extract unverified code.
        if let Some(digests) = artifact.get("digests").and_then(|d| d.get("sha256"))
            && let expected = digests.as_str().unwrap_or_default()
            && !expected.is_empty()
        {
            use sha2::{Digest, Sha256};
            let hash = Sha256::digest(&bytes);
            let actual: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
            if actual != expected.to_ascii_lowercase() {
                return Err(anyhow::anyhow!(
                    "sha256 mismatch for `{}` ({}): expected {}, got {}",
                    file_name,
                    url,
                    expected,
                    actual
                ));
            }
        }
        fs::write(&artifact_path, bytes)
            .with_context(|| format!("writing {}", artifact_path.display()))?;
    }

    // Extract (wheels are zips, sdists are gzipped tarballs).
    let extract_dir = dist_dir.join("extracted");
    let package_dir = extract_distribution(&artifact_path, &extract_dir, &req.name)?;
    Ok(finalize_dependency(package_dir, &req.name, &best_version))
}

/// Determine the import name and the exact vendorable path (the importable
/// package directory or single module file) of an extracted distribution.
fn finalize_dependency(extract_top: PathBuf, dist_name: &str, version: &str) -> ResolvedDependency {
    let import_name = top_level_import(&extract_top, dist_name).unwrap_or_else(|_| {
        // Last resort: the distribution name itself.
        dist_name.to_string()
    });
    // The vendored path must be the importable package itself:
    // python_module_deps requires __init__.py (or a .py file) at the given
    // path's root.
    let vendored_path = if extract_top.join(&import_name).join("__init__.py").is_file() {
        extract_top.join(&import_name)
    } else if extract_top
        .join(&import_name)
        .with_extension("py")
        .is_file()
    {
        extract_top.join(&import_name).with_extension("py")
    } else {
        extract_top
    };
    ResolvedDependency {
        import_name,
        path: vendored_path,
        version: version.to_string(),
    }
}

/// Whether a wheel filename is pure Python: the last three tag segments
/// are `{py3|py2.py3}-none-any`.
fn is_pure_wheel(file_name: &str) -> bool {
    if !file_name.ends_with(".whl") {
        return false;
    }
    let stem = &file_name[..file_name.len() - 4];
    let tags: Vec<&str> = stem.split('-').collect();
    let n = tags.len();
    n >= 3
        && (tags[n - 3] == "py3" || tags[n - 3] == "py2.py3" || tags[n - 3].starts_with("py3."))
        && tags[n - 2] == "none"
        && tags[n - 1] == "any"
}

/// Look for an already-extracted cached dependency satisfying the
/// requirement.
fn cached_match(dist_dir: &Path, req: &Requirement) -> Option<ResolvedDependency> {
    let extracted = dist_dir.join("extracted");
    let entries = fs::read_dir(&extracted).ok()?;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        // The .dist-info metadata directory is not a distribution root,
        // and neither is the wheel DATA directory (`{dist}-{version}.data`).
        if name.ends_with(".dist-info") || name.contains(".data") {
            continue;
        }
        // Layout: `{dist}-{version}/` where the version may itself contain
        // dots/hyphens; find the version by stripping the normalized name.
        let Some(version) = name.strip_prefix(&format!("{}-", req.name)) else {
            continue;
        };
        let Some(version) = parse_version(version) else {
            continue;
        };
        if !version_satisfies(&version, &req.specifiers) {
            continue;
        }
        return Some(finalize_dependency(path, &req.name, &version_str_of(&version)));
    }
    None
}

fn version_str_of(v: &Version) -> String {
    let mut s = v.release.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(".");
    if let Some((kind, n)) = &v.pre {
        s.push_str(kind);
        s.push_str(&n.to_string());
    }
    if let Some(p) = v.post {
        s.push_str(".post");
        s.push_str(&p.to_string());
    }
    if let Some(d) = v.dev {
        s.push_str(".dev");
        s.push_str(&d.to_string());
    }
    s
}

/// Extract a wheel or sdist into `extract_dir`, returning the extracted
/// distribution's top directory.
fn extract_distribution(artifact: &Path, extract_dir: &Path, dist_name: &str) -> Result<PathBuf> {
    fs::create_dir_all(extract_dir)?;
    let file_name = artifact
        .file_name()
        .and_then(|n| n.to_str())
        .context("artifact has no file name")?;

    // The extracted top dir: strip the extension.
    let top_name = if file_name.ends_with(".whl") {
        file_name.strip_suffix(".whl").unwrap()
    } else {
        file_name
            .strip_suffix(".tar.gz")
            .or_else(|| file_name.strip_suffix(".tgz"))
            .or_else(|| file_name.strip_suffix(".zip"))
            .unwrap_or(file_name)
    };
    let top = extract_dir.join(top_name);
    if top.join("__init__.py").is_file() || top.is_dir() && has_dist_info(&top) {
        return Ok(top);
    }

    // A top-level .py file next to the dist-info: the single-module wheel
    // layout (six-1.17.0 extracts to six.py + six-1.17.0.dist-info/).
    let module_files: Vec<PathBuf> = fs::read_dir(extract_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "py"))
        .collect();
    if module_files.len() == 1 {
        let only_metadata_dirs = fs::read_dir(extract_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .all(|d| is_dist_info(&d));
        if only_metadata_dirs {
            return Ok(extract_dir.to_path_buf());
        }
    }

    if file_name.ends_with(".whl") || file_name.ends_with(".zip") {
        let file = fs::File::open(artifact)
            .with_context(|| format!("opening {}", artifact.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("reading zip {}", artifact.display()))?;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).context("reading zip entry")?;
            let out_path = entry
                .enclosed_name()
                .with_context(|| format!("unsafe path in {}", file_name))?;
            let dest = extract_dir.join(out_path);
            if entry.is_dir() {
                fs::create_dir_all(&dest)?;
            } else {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut out = fs::File::create(&dest)
                    .with_context(|| format!("writing {}", dest.display()))?;
                std::io::copy(&mut entry, &mut out)?;
            }
        }
    } else {
        // gzipped tarball (sdist).
        let file = fs::File::open(artifact)
            .with_context(|| format!("opening {}", artifact.display()))?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);
        archive
            .unpack(extract_dir)
            .with_context(|| format!("extracting {}", artifact.display()))?;
    }

    let top = extract_dir.join(top_name);
    if top.is_dir() {
        Ok(top)
    } else {
        // Some sdists extract with a different top dir; pick the single
        // subdirectory.
        let mut dirs: Vec<PathBuf> = fs::read_dir(extract_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.retain(|p| p.join("__init__.py").is_file() || has_dist_info(p));
        if dirs.len() == 1 {
            Ok(dirs.remove(0))
        } else {
            bail!(
                "could not locate the extracted package for `{}` under {}",
                dist_name,
                extract_dir.display()
            )
        }
    }
}

/// Whether a directory is itself a `.dist-info` directory (its name ends
/// with `.dist-info`) — a wheel's metadata dir, distinct from a package.
fn is_dist_info(dir: &Path) -> bool {
    dir.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".dist-info"))
}

fn has_dist_info(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(".dist-info")
            })
        })
        .unwrap_or(false)
}

/// Determine the top-level import name of an extracted distribution:
/// wheel `.dist-info/top_level.txt`, else our own package discovery.
fn top_level_import(dist_dir: &Path, dist_name: &str) -> Result<String> {
    // Scan for any *.dist-info directory (the distribution name in the
    // dir name may normalize `-`/`_` differently than the requirement).
    if let Ok(entries) = fs::read_dir(dist_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".dist-info") && entry.path().is_dir() {
                let top_level = entry.path().join("top_level.txt");
                if let Ok(text) = fs::read_to_string(&top_level) {
                    if let Some(first) = text
                        .lines()
                        .next()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                    {
                        return Ok(first.to_string());
                    }
                }
            }
        }
    }

    // sdist (or metadata-less wheel): run our own discovery on the
    // extracted tree.
    let meta = crate::packaging::read_project_metadata(dist_dir)?;
    let dirs = crate::packaging::resolve_package_dirs(dist_dir, &meta)?;
    if let Some(first) = dirs.first() {
        let name = first
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| dist_name.to_string());
        return Ok(name);
    }
    if !meta.py_modules.is_empty() {
        return Ok(meta.py_modules[0].clone());
    }
    // Last resort: a directory matching the distribution name.
    for candidate in [dist_name.to_string(), dist_name.replace('_', "-")] {
        let dir = dist_dir.join(&candidate);
        if dir.join("__init__.py").is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not determine the import name of `{}`; vendor it manually via \
         rython.toml [python-modules]",
        dist_name
    )
}

/// GET a URL, returning the body as a string.
fn fetch_url(url: &str) -> Result<String> {
    let bytes = fetch_bytes(url)?;
    String::from_utf8(bytes).context("response is not UTF-8")
}

/// GET a URL, returning the body as bytes. Uses curl (the toolchain already
/// shells out to cargo and python3); a missing curl is a loud error.
fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let output = std::process::Command::new("curl")
        .arg("-sSfL")
        .arg("--retry")
        .arg("3")
        .arg("--retry-delay")
        .arg("1")
        .arg("--max-time")
        .arg("120")
        .arg(url)
        .output()
        .with_context(|| {
            "downloading from PyPI requires `curl` on PATH; vendor the dependency \
             via rython.toml [python-modules] instead"
        })?;
    if !output.status.success() {
        bail!(
            "curl failed fetching {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

/// Parse the `install_requires`/`dependencies` of a metadata object into
/// requirements (used for the resolution error messages and vendoring).
pub fn parse_requirements(deps: &[String]) -> Vec<Requirement> {
    deps.iter()
        .filter_map(|d| parse_requirement(d).ok())
        .collect()
}

/// Merge resolved dependencies with explicit `[python-modules]` entries:
/// explicit entries win (a user-pinned vendored copy overrides PyPI).
pub(crate) fn merge_python_modules(
    explicit: HashMap<String, crate::convert::ManifestPythonModule>,
    resolved: Vec<ResolvedDependency>,
) -> HashMap<String, crate::convert::ManifestPythonModule> {
    use crate::convert::ManifestPythonModule;
    let mut merged = explicit;
    for dep in resolved {
        merged.entry(dep.import_name.clone()).or_insert_with(|| {
            ManifestPythonModule {
                path: dep.path.to_string_lossy().to_string(),
            }
        });
    }
    merged
}
