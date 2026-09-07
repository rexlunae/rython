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
    /// The PEP 440 local version label (`1.0+cpu` → `cpu`), normalized
    /// to lowercase with `.` separators; kept through the cache so an
    /// offline resolution reports the artifact's own version.
    pub local: Option<String>,
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
    // The local label (`+cpu`, `+ubuntu.1`): after everything else,
    // normalized like the rest (PEP 440 local versions).
    let (rest, local) = match rest.split_once('+') {
        Some((public, label)) => {
            let label = label.trim().to_ascii_lowercase().replace(['-', '_'], ".");
            if label.is_empty() {
                return None;
            }
            (public, Some(label))
        }
        None => (rest, None),
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
        local,
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
/// The local label's comparison key (PEP 440): no label sorts before
/// any label; segment-wise, numeric segments compare numerically and
/// after alphanumeric ones, alphanumeric ones lexically.
fn local_key(v: &Version) -> Vec<(u8, u64, String)> {
    v.local
        .as_deref()
        .map(|label| {
            label
                .split('.')
                .map(|seg| match seg.parse::<u64>() {
                    Ok(n) => (1, n, String::new()),
                    Err(_) => (0, 0, seg.to_string()),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `version_cmp` against a SPECIFIER's version: a specifier without a
/// local label matches any local variant of that public version (PEP
/// 440), so the candidate's label is ignored then.
fn cmp_for_specifier(version: &Version, spec: &Version) -> Ordering {
    if spec.local.is_none() && version.local.is_some() {
        let public = Version {
            local: None,
            ..version.clone()
        };
        return version_cmp(&public, spec);
    }
    version_cmp(version, spec)
}

pub fn version_cmp(a: &Version, b: &Version) -> Ordering {
    let a_key = cmp_key(a);
    let b_key = cmp_key(b);
    a_key.cmp(&b_key)
}

fn cmp_key(v: &Version) -> (u64, Vec<u64>, (u8, u64, u64), u64, u64, Vec<(u8, u64, String)>) {
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
    // PEP 440: a `.devN` release sorts BEFORE its base (`1.0.dev1 < 1.0`,
    // `1.0a1.dev1 < 1.0a1`). At this point every earlier key component is
    // equal, so an ABSENT dev must sort after any PRESENT dev — encode
    // absent as the u64 sentinel (a real dev number never reaches it).
    let dev_key = match v.dev {
        Some(n) => n,
        None => u64::MAX,
    };
    (
        v.epoch,
        release,
        pre,
        v.post.unwrap_or(0),
        dev_key,
        local_key(v),
    )
}

/// The PEP 440 prefix match (`==1.2.*`): the candidate's release starts
/// with the prefix's release, and a pre/post/dev marker the prefix
/// spells (`==1.0a1.*`, `==1.0.post1.*`) must match exactly; the
/// candidate's local label is ignored, and a prefix that is not a
/// version (`==1.x.*`, a local label) matches nothing — loud in the
/// sense that such a requirement never resolves, never silently
/// widened (Devin review on #342, round 9).
fn matches_prefix(version: &Version, prefix: &str) -> bool {
    let Some(parsed) = parse_version(prefix) else {
        return false;
    };
    // The lenient parser reads `1.x` as `1`: a prefix is a version only
    // if it spells one — its normalized form round-trips.
    let normalized = prefix
        .trim()
        .trim_start_matches(['v', 'V'])
        .to_ascii_lowercase()
        .replace(['-', '_'], ".");
    if version_str_of(&parsed) != normalized {
        return false;
    }
    let prefix = parsed;
    if prefix.local.is_some() {
        return false;
    }
    let mut release = version.release.clone();
    while release.len() < prefix.release.len() {
        release.push(0);
    }
    if !prefix
        .release
        .iter()
        .zip(release.iter())
        .all(|(p, r)| p == r)
    {
        return false;
    }
    (prefix.pre.is_none() || prefix.pre == version.pre)
        && (prefix.post.is_none() || prefix.post == version.post)
        && (prefix.dev.is_none() || prefix.dev == version.dev)
}

/// Whether `version` satisfies `(op, spec)`.
pub fn matches_specifier(version: &Version, op: &str, spec: &str) -> bool {
    match op {
        "==" => {
            if let Some(prefix) = spec.strip_suffix(".*") {
                return matches_prefix(version, prefix);
            }
            match parse_version(spec) {
                Some(sv) => cmp_for_specifier(version, &sv) == Ordering::Equal,
                None => false,
            }
        }
        "!=" => !matches_specifier(version, "==", spec),
        ">=" => {
            match parse_version(spec) {
                Some(sv) => cmp_for_specifier(version, &sv) != Ordering::Less,
                None => false,
            }
        }
        ">" => match parse_version(spec) {
            Some(sv) => cmp_for_specifier(version, &sv) == Ordering::Greater,
            None => false,
        },
        "<=" => match parse_version(spec) {
            Some(sv) => cmp_for_specifier(version, &sv) != Ordering::Greater,
            None => false,
        },
        "<" => match parse_version(spec) {
            Some(sv) => cmp_for_specifier(version, &sv) == Ordering::Less,
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
    resolve_dependency_in(&cache_dir(), req, offline)
}

/// `resolve_dependency` against an explicit cache directory (the
/// `RYPIP_CACHE_DIR` / `~/.cache/rypip` default is `cache_dir()`).
pub fn resolve_dependency_in(cache: &Path, req: &Requirement, offline: bool) -> Result<ResolvedDependency> {
    let dist_dir = cache.join(&req.name);

    // A cached, already-extracted distribution that satisfies the
    // specifiers can be reused (this is what makes offline rebuilds work).
    if let Some(hit) = cached_match(&dist_dir, req)? {
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

    // Integrity: PyPI's JSON metadata carries the sha256 of every
    // artifact. FAIL CLOSED — a missing digest means the response is not
    // from real PyPI (mirror/intercepted), and a mismatch means tampering;
    // either way the artifact is never extracted or transpiled. An
    // artifact already in the cache is checked against the SAME digest
    // before it is reused: a cache file that no longer matches the index
    // is replaced (and its extraction with it).
    let expected = artifact
        .get("digests")
        .and_then(|d| d.get("sha256"))
        .and_then(|d| d.as_str())
        .filter(|d| !d.is_empty())
        .with_context(|| {
            format!(
                "`{}` ({}) has no sha256 digest in the index response; \
                 refusing to extract an unverified artifact",
                file_name, url
            )
        })?
        .to_ascii_lowercase();
    fs::create_dir_all(&dist_dir)
        .with_context(|| format!("creating cache {}", dist_dir.display()))?;
    let artifact_path = dist_dir.join(&file_name);
    let cached_matches = match fs::read(&artifact_path) {
        Ok(bytes) => sha256_hex(&bytes) == expected,
        Err(_) => false,
    };
    if !cached_matches {
        // A stale or tampered cache file is replaced below; its
        // extraction (if any) is tied to the OLD digest by its marker,
        // so `extract_distribution` sets it aside and rebuilds it under
        // the rename-and-recheck protocol — nothing is removed in place
        // here, where a concurrent resolver's freshly published tree
        // could be the casualty (Devin review on #342, round 9).
        let bytes = fetch_bytes(&url)?;
        let actual = sha256_hex(&bytes);
        if actual != expected {
            return Err(anyhow::anyhow!(
                "sha256 mismatch for `{}` ({}): expected {}, got {}",
                file_name,
                url,
                expected,
                actual
            ));
        }
        publish_file(&artifact_path, &bytes)?;
    }
    // The verified digest, beside the artifact: the offline path reuses
    // the artifact only through it.
    let sidecar = digest_sidecar(&artifact_path);
    publish_file(&sidecar, format!("{expected}\n").as_bytes())?;

    // Extract (wheels are zips, sdists are gzipped tarballs). The
    // extraction verifies the published artifact against its sidecar; a
    // torn pair (a concurrent publisher of another download interleaved
    // with ours) is healed by publishing our verified pair again, once.
    let package_dir = match extract_distribution(&artifact_path, &dist_dir, &req.name) {
        Ok(dir) => dir,
        Err(e) if e.to_string().contains("sha256 mismatch") => {
            let bytes = fetch_bytes(&url)?;
            if sha256_hex(&bytes) != expected {
                return Err(e);
            }
            publish_file(&artifact_path, &bytes)?;
            publish_file(&sidecar, format!("{expected}\n").as_bytes())?;
            extract_distribution(&artifact_path, &dist_dir, &req.name)?
        }
        Err(e) => return Err(e),
    };
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

/// The file inside an extraction directory that marks it COMPLETE: written
/// after the whole archive is unpacked and the distribution root was
/// found, then the directory is renamed into place. It holds the artifact
/// file name the tree was extracted from (the version and the artifact
/// kind derive from it). A directory without it — an interrupted
/// extraction, an older cache layout — is never a cache hit.
const COMPLETE_MARKER: &str = ".rypip-complete";
/// The infix of an in-progress extraction directory
/// (`<stem>.partial-<pid>-<n>`).
const PARTIAL_INFIX: &str = ".partial-";
/// The infix of an incomplete tree set aside from the extraction path
/// before it is removed (`<stem>.stale-<pid>-<n>`).
const STALE_INFIX: &str = ".stale-";
/// Distinguishes the temporary directories of one process's threads.
static EXTRACTION_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn temporary_suffix() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        EXTRACTION_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// The sidecar recording a cached artifact's VERIFIED sha256: written
/// beside the artifact once a download was checked against PyPI's digest.
/// The offline path reuses an artifact only through it.
fn digest_sidecar(artifact: &Path) -> PathBuf {
    let mut name = artifact
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".sha256");
    artifact.with_file_name(name)
}

/// Whether a cached artifact matches its recorded digest. Ok(false): no
/// sidecar (an older cache, or a file that was never verified) — the
/// artifact is not reused offline; a mismatch is a loud error naming the
/// file, never a silent skip.
fn cached_artifact_verified(artifact: &Path) -> Result<bool> {
    let sidecar = digest_sidecar(artifact);
    let Ok(recorded) = fs::read_to_string(&sidecar) else {
        return Ok(false);
    };
    let recorded = recorded.trim().to_ascii_lowercase();
    let bytes =
        fs::read(artifact).with_context(|| format!("reading {}", artifact.display()))?;
    let actual = sha256_hex(&bytes);
    if actual != recorded {
        bail!(
            "sha256 mismatch for the cached artifact {}: its recorded digest is {}, the file \
             hashes to {} — the cache file was altered; delete it (and {}) to fetch it again",
            artifact.display(),
            recorded,
            actual,
            sidecar.display()
        );
    }
    Ok(true)
}

/// The directory an artifact extracts into: `<dist_dir>/extracted/<stem>`.
fn extraction_dir(dist_dir: &Path, artifact_file: &str) -> PathBuf {
    let stem = artifact_stem(artifact_file).unwrap_or(artifact_file);
    dist_dir.join("extracted").join(stem)
}

/// The artifact file name a COMPLETE extraction directory records.
fn extracted_artifact_name(dir: &Path) -> Option<String> {
    let text = fs::read_to_string(dir.join(COMPLETE_MARKER)).ok()?;
    let name = text.lines().next()?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// The sha256 digest of the artifact an extraction was built from (the
/// completion marker's second line): an extraction is COMPLETE only for
/// that artifact — a replaced artifact (same file name, another verified
/// digest) rebuilds it (Devin review on #342, round 5). A marker without
/// a digest (an older layout) matches nothing.
fn extracted_artifact_digest(dir: &Path) -> Option<String> {
    let text = fs::read_to_string(dir.join(COMPLETE_MARKER)).ok()?;
    let digest = text.lines().nth(1)?.trim().to_ascii_lowercase();
    (!digest.is_empty()).then_some(digest)
}

/// The digest recorded beside an artifact (its sidecar), lowercase.
fn recorded_digest(artifact: &Path) -> Result<String> {
    let sidecar = digest_sidecar(artifact);
    let recorded = fs::read_to_string(&sidecar)
        .with_context(|| format!("reading the recorded digest {}", sidecar.display()))?;
    Ok(recorded.trim().to_ascii_lowercase())
}

/// What a cache candidate is: a complete extraction (its root), or a
/// verified artifact still to extract.
enum Cached {
    /// A complete extraction's root, and whether its artifact is a wheel.
    Extracted { root: PathBuf, wheel: bool },
    Artifact(PathBuf),
}

/// Look for a cached distribution satisfying the requirement: the NEWEST
/// satisfying version among the complete extractions under `extracted/`
/// and the verified artifacts in the distribution's cache directory (the
/// same choice online resolution makes among the index's releases, so an
/// unpinned requirement resolves the same version from the cache in any
/// directory order); an artifact is extracted now. An extraction without
/// its completion marker and an artifact without its recorded digest are
/// not candidates; an artifact whose digest no longer matches is skipped
/// — a loud error only when no valid candidate remains.
fn cached_match(dist_dir: &Path, req: &Requirement) -> Result<Option<ResolvedDependency>> {
    let mut candidates: Vec<(Version, Cached)> = Vec::new();
    // A candidate whose artifact no longer hashes to its sidecar is
    // CORRUPT: it is skipped so a valid alternative still resolves, and
    // its error surfaces only when nothing valid remains (Devin review
    // on #342, round 10) — loud where it blocks, silent where it does not.
    let mut corrupt: Vec<anyhow::Error> = Vec::new();
    if let Ok(entries) = fs::read_dir(dist_dir.join("extracted")) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if !path.is_dir() || name.contains(PARTIAL_INFIX) || name.contains(STALE_INFIX) {
                continue;
            }
            let Some(artifact_name) = extracted_artifact_name(&path) else {
                continue;
            };
            let Some(version) = cached_version_of(&artifact_name, &req.name) else {
                continue;
            };
            if !version_satisfies(&version, &req.specifiers) {
                continue;
            }
            // The extraction is tied to its ARTIFACT: the artifact must
            // be present and verified, and the marker's digest must be
            // the artifact's — an extraction of a missing, unverified or
            // replaced artifact is not a candidate (the artifact, if
            // verified, extracts anew below).
            let artifact = dist_dir.join(&artifact_name);
            if !artifact.is_file() {
                continue;
            }
            match cached_artifact_verified(&artifact) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(e) => {
                    corrupt.push(e);
                    continue;
                }
            }
            if extracted_artifact_digest(&path) != Some(recorded_digest(&artifact)?) {
                continue;
            }
            let Ok(root) = locate_extracted_root(&path) else {
                continue;
            };
            let wheel = artifact_name.ends_with(".whl");
            candidates.push((version, Cached::Extracted { root, wheel }));
        }
    }
    if let Ok(entries) = fs::read_dir(dist_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(version) = cached_version_of(&name, &req.name) else {
                continue;
            };
            if !version_satisfies(&version, &req.specifiers) {
                continue;
            }
            match cached_artifact_verified(&path) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(e) => {
                    corrupt.push(e);
                    continue;
                }
            }
            candidates.push((version, Cached::Artifact(path)));
        }
    }
    // Ascending by version; at one version a wheel sorts after an sdist
    // and, per artifact, its complete extraction after the artifact, so
    // the last element is the newest version in the online preference,
    // extracted if it already is.
    candidates.sort_by(|(va, ca), (vb, cb)| {
        version_cmp(va, vb).then_with(|| {
            // The online preference, in any directory order: a wheel
            // over an sdist FIRST (whatever was extracted so far), then a
            // complete extraction over its own unextracted artifact
            // (Devin review on #342, rounds 6 and 7).
            let rank = |c: &Cached| match c {
                Cached::Extracted { wheel, .. } => (*wheel as u8) * 2 + 1,
                Cached::Artifact(p) => (p.extension().is_some_and(|e| e == "whl") as u8) * 2,
            };
            rank(ca).cmp(&rank(cb))
        })
    });
    let Some((version, cached)) = candidates.pop() else {
        if let Some(e) = corrupt.into_iter().next() {
            return Err(e);
        }
        return Ok(None);
    };
    let root = match cached {
        Cached::Extracted { root, .. } => root,
        Cached::Artifact(path) => extract_distribution(&path, dist_dir, &req.name)?,
    };
    Ok(Some(finalize_dependency(root, &req.name, &version_str_of(&version))))
}

/// The artifact file name without its archive extension: the name of the
/// directory the artifact extracts into.
fn artifact_stem(file_name: &str) -> Option<&str> {
    file_name
        .strip_suffix(".whl")
        .or_else(|| file_name.strip_suffix(".tar.gz"))
        .or_else(|| file_name.strip_suffix(".tgz"))
        .or_else(|| file_name.strip_suffix(".zip"))
}

/// The version an artifact FILE NAME (`idna-3.10-py3-none-any.whl`,
/// `charset-normalizer-3.4.0.tar.gz`, `pkg-1.0-rc1.tar.gz`) names for the
/// distribution `dist_name`. The kind decides the grammar: a wheel is
/// `{name}-{version}[-{build}]-{python}-{abi}-{platform}`, so the version
/// is the one component after the name; an sdist is `{name}-{version}`,
/// so everything after the name is the version, hyphen-separated
/// pre/post/dev suffixes included (a legacy name may itself contain `-`:
/// every split whose leading part normalizes to the wanted name is
/// tried). A file of another distribution, or not an archive, is None.
fn cached_version_of(file_name: &str, dist_name: &str) -> Option<Version> {
    let stem = artifact_stem(file_name)?;
    let is_wheel = file_name.ends_with(".whl");
    let want = normalize_dist_name(dist_name);
    for (index, _) in stem.match_indices('-') {
        let (name, rest) = (&stem[..index], &stem[index + 1..]);
        if normalize_dist_name(name) != want {
            continue;
        }
        let version = if is_wheel {
            let parts: Vec<&str> = rest.split('-').collect();
            // version + three tags, or version + build tag + three tags.
            if parts.len() != 4 && parts.len() != 5 {
                continue;
            }
            parts[0]
        } else {
            rest
        };
        if let Some(version) = parse_version(version) {
            return Some(version);
        }
    }
    None
}

fn version_str_of(v: &Version) -> String {
    let mut s = String::new();
    if v.epoch > 0 {
        s.push_str(&v.epoch.to_string());
        s.push('!');
    }
    s.push_str(&v.release.iter().map(|n| n.to_string()).collect::<Vec<_>>().join("."));
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
    if let Some(local) = &v.local {
        s.push('+');
        s.push_str(local);
    }
    s
}

/// Whether an extraction directory is COMPLETE for the artifact with
/// `digest`: its marker records that digest and its distribution root
/// is locatable. An extraction of another digest — the artifact was
/// replaced — is incomplete for this one and rebuilds.
fn extraction_complete(dir: &Path, digest: &str) -> bool {
    extracted_artifact_digest(dir).as_deref() == Some(digest) && locate_extracted_root(dir).is_ok()
}

/// Extract a wheel or sdist into its own directory under
/// `{dist_dir}/extracted/{artifact stem}/`, returning the extracted
/// distribution's root: the directory holding the package (and, for a
/// wheel, its `.dist-info`). Each artifact gets its own directory: two
/// versions of one distribution extracted into a shared directory merge
/// their files (a module the newer version added survives into the older
/// version's tree), which is not any version of the package. The
/// extraction is ATOMIC: the archive unpacks into a
/// `<stem>.partial-<pid>-<n>` sibling (unique per invocation, threads of
/// one process included), the root is located there, the completion
/// marker is written, and only then is the directory renamed into place
/// — an interrupted extraction leaves nothing the cache lookup accepts.
/// A COMPLETE tree at the path is never removed: an incomplete one is
/// set aside by rename first (`set_aside_incomplete`), so a tree another
/// resolver publishes between the check and the replacement survives
/// and is used; a concurrent resolver that wins the final rename wins.
/// Every failure after the sibling was created removes it; a sibling
/// whose owning process is provably dead (a crashed run's) is cleared
/// first — never one whose owner may still be extracting.
fn extract_distribution(artifact: &Path, dist_dir: &Path, dist_name: &str) -> Result<PathBuf> {
    let file_name = artifact
        .file_name()
        .and_then(|n| n.to_str())
        .context("artifact has no file name")?;
    let extract_dir = extraction_dir(dist_dir, file_name);
    let extracted = dist_dir.join("extracted");
    let stem = artifact_stem(file_name).unwrap_or(file_name);
    // The artifact's bytes, read ONCE and verified against its sidecar:
    // the bytes that unpack are the bytes the marker's digest names,
    // and a torn artifact/sidecar pair (two publishers of different
    // downloads interleaving) is a loud mismatch, never an unverified
    // extraction (Devin review on #342, round 6).
    let bytes = fs::read(artifact).with_context(|| format!("reading {}", artifact.display()))?;
    let digest = recorded_digest(artifact)?;
    let actual = sha256_hex(&bytes);
    if actual != digest {
        bail!(
            "sha256 mismatch for the cached artifact {}: its recorded digest is {}, the file \
             hashes to {} — the artifact and its digest were published apart; resolve again \
             (a concurrent resolver completes the pair) or delete both to fetch anew",
            artifact.display(),
            digest,
            actual
        );
    }
    if extraction_complete(&extract_dir, &digest) {
        return locate_extracted_root(&extract_dir);
    }
    // Anything else at that path (a partial extraction, an extraction
    // of a replaced artifact, an older cache layout) is set aside and
    // removed — unless it became complete for this artifact meanwhile,
    // in which case it is the answer.
    set_aside_incomplete(&extract_dir, &extracted, stem, &digest)?;
    if extraction_complete(&extract_dir, &digest) {
        return locate_extracted_root(&extract_dir);
    }
    remove_dead_temporaries(&extracted, stem);
    fs::create_dir_all(&extracted)?;
    let partial = extracted.join(format!("{stem}{PARTIAL_INFIX}{}", temporary_suffix()));
    fs::create_dir_all(&partial)?;
    let published =
        publish_extraction(&bytes, file_name, dist_name, &digest, &partial, &extract_dir);
    // Whatever happened, this invocation's sibling is gone: on success it
    // was renamed away (or the other resolver's tree won), on failure
    // its unpacked files are not left for the cache to skip forever.
    if partial.exists() {
        let _ = fs::remove_dir_all(&partial);
    }
    published?;
    locate_extracted_root(&extract_dir)
}

/// Move an INCOMPLETE tree at `extract_dir` out of the way and remove it.
/// The move is a rename, so a complete tree is never deleted in place: if
/// the tree turns out complete once set aside (another resolver published
/// it between the caller's check and the rename), it is renamed back —
/// or, when the path was filled again meanwhile by another complete
/// tree, the duplicate is dropped. A path that vanished is fine.
fn set_aside_incomplete(
    extract_dir: &Path,
    extracted: &Path,
    stem: &str,
    digest: &str,
) -> Result<()> {
    if !extract_dir.exists() {
        return Ok(());
    }
    let aside = extracted.join(format!("{stem}{STALE_INFIX}{}", temporary_suffix()));
    match fs::rename(extract_dir, &aside) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(anyhow::Error::new(e).context(format!(
            "setting aside the incomplete extraction {}",
            extract_dir.display()
        ))),
        Ok(()) => {
            if extraction_complete(&aside, digest) {
                if fs::rename(&aside, extract_dir).is_err() {
                    let _ = fs::remove_dir_all(&aside);
                }
                return Ok(());
            }
            fs::remove_dir_all(&aside)
                .with_context(|| format!("removing the incomplete extraction {}", aside.display()))
        }
    }
}

/// Unpack `artifact` into `partial`, validate, mark complete, and rename
/// into `extract_dir`. A rename that fails because another resolver
/// published a COMPLETE tree there meanwhile is that tree's success.
fn publish_extraction(
    bytes: &[u8],
    file_name: &str,
    dist_name: &str,
    digest: &str,
    partial: &Path,
    extract_dir: &Path,
) -> Result<()> {
    if file_name.ends_with(".whl") || file_name.ends_with(".zip") {
        let file = std::io::Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("reading zip {file_name}"))?;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).context("reading zip entry")?;
            let out_path = entry
                .enclosed_name()
                .with_context(|| format!("unsafe path in {}", file_name))?;
            let dest = partial.join(out_path);
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
        let file = std::io::Cursor::new(bytes);
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);
        archive
            .unpack(partial)
            .with_context(|| format!("extracting {file_name}"))?;
    }
    locate_extracted_root(partial).with_context(|| {
        format!(
            "could not locate the extracted package for `{}` under {}",
            dist_name,
            partial.display()
        )
    })?;
    let marker = partial.join(COMPLETE_MARKER);
    fs::write(&marker, format!("{file_name}\n{digest}\n"))
        .with_context(|| format!("writing {}", marker.display()))?;
    match fs::rename(partial, extract_dir) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Another resolver published the same artifact between our
            // check and our rename: its complete tree is the cache entry.
            if extraction_complete(extract_dir, digest)
                && locate_extracted_root(extract_dir).is_ok()
            {
                return Ok(());
            }
            Err(anyhow::Error::new(e).context(format!(
                "moving the extraction {} into place at {}",
                partial.display(),
                extract_dir.display()
            )))
        }
    }
}

/// Remove `<stem>.partial-*` and `<stem>.stale-*` siblings whose OWNING
/// PROCESS is provably dead (a crashed run's leftovers): the owner's pid
/// is in the name (`temporary_suffix`), and a sibling is removed only
/// when that pid is not this process and the platform can say it is not
/// running. Age is no evidence of liveness (Devin review on #342, round
/// 4: a long extraction is not a crashed one), so a sibling whose owner
/// is alive, or whose liveness cannot be determined, is left alone; it
/// costs disk, never correctness — the cache lookup skips it. Best
/// effort — a failure to remove one only leaves it for the next run.
fn remove_dead_temporaries(extracted: &Path, stem: &str) {
    let Ok(entries) = fs::read_dir(extracted) else {
        return;
    };
    let prefixes = [format!("{stem}{PARTIAL_INFIX}"), format!("{stem}{STALE_INFIX}")];
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(suffix) = prefixes.iter().find_map(|p| name.strip_prefix(p.as_str())) else {
            continue;
        };
        let Some(owner) = owner_pid(suffix) else {
            continue;
        };
        if owner == std::process::id() || process_alive(owner) != Some(false) {
            continue;
        }
        let _ = fs::remove_dir_all(entry.path());
    }
}

/// The owning pid of a temporary suffix (`<pid>-<n>`, or the older
/// `<pid>` spelling).
fn owner_pid(suffix: &str) -> Option<u32> {
    suffix.split('-').next()?.parse().ok()
}

/// Whether a process is running: `Some(true)`/`Some(false)` when the
/// platform can tell, `None` when it cannot (then nothing is removed).
/// Unix: procfs where it exists, else `kill -0` (a permission error is a
/// live process of another user).
#[cfg(unix)]
fn process_alive(pid: u32) -> Option<bool> {
    if Path::new("/proc/self").exists() {
        return Some(Path::new(&format!("/proc/{pid}")).exists());
    }
    let output = std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .ok()?;
    if output.status.success() {
        return Some(true);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Some(stderr.contains("permitted") || stderr.contains("Operation not"))
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> Option<bool> {
    None
}

/// Publish a whole file at `path` atomically: the bytes go to a unique
/// sibling (`<name>.tmp-<pid>-<n>`) first and are renamed into place, so
/// a concurrent reader — another resolver's digest check or extraction —
/// sees either no file or the complete one, never a truncated one (Devin
/// review on #342, round 4). A rename another resolver's publish of the
/// same path beat is not an error: the path holds their complete file.
fn publish_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("cache path has no parent")?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("cache path has no file name")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!("{name}.tmp-{}", temporary_suffix()));
    let written = fs::write(&tmp, bytes)
        .with_context(|| format!("writing {}", tmp.display()))
        .and_then(|()| match fs::rename(&tmp, path) {
            Ok(()) => Ok(()),
            Err(_) if path.is_file() => Ok(()),
            Err(e) => Err(anyhow::Error::new(e)
                .context(format!("publishing {}", path.display()))),
        });
    if tmp.exists() {
        let _ = fs::remove_file(&tmp);
    }
    written
}

/// The distribution root inside an extraction directory: the directory
/// itself when it is a wheel's contents (a `.dist-info` beside the
/// package) or a single-module layout (one `.py` file beside the
/// metadata), else the one subdirectory an sdist tarball unpacked to, or
/// the directory itself when it carries the sdist's own metadata (an
/// older cache layout that unpacked the tarball's top directly).
fn locate_extracted_root(extract_dir: &Path) -> Result<PathBuf> {
    if !extract_dir.is_dir() {
        bail!("{} is not a directory", extract_dir.display());
    }
    if has_dist_info(extract_dir) {
        return Ok(extract_dir.to_path_buf());
    }
    let is_sdist_top = |dir: &Path| {
        dir.join("PKG-INFO").is_file()
            || dir.join("pyproject.toml").is_file()
            || dir.join("setup.py").is_file()
            || dir.join("setup.cfg").is_file()
    };
    let mut dirs: Vec<PathBuf> = fs::read_dir(extract_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.retain(|p| p.join("__init__.py").is_file() || has_dist_info(p) || is_sdist_top(p));
    if dirs.len() == 1 && !is_sdist_top(extract_dir) && !dirs[0].join("__init__.py").is_file() {
        return Ok(dirs.remove(0));
    }
    if is_sdist_top(extract_dir) || dirs.iter().any(|d| d.join("__init__.py").is_file()) {
        return Ok(extract_dir.to_path_buf());
    }
    bail!("no distribution root under {}", extract_dir.display())
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

#[cfg(test)]
mod extraction_tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rypip-extraction-{tag}-{}-{}",
            std::process::id(),
            temporary_suffix()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn complete_tree(dir: &Path) {
        fs::create_dir_all(dir.join("idna")).unwrap();
        fs::write(dir.join("idna/__init__.py"), "").unwrap();
        fs::create_dir_all(dir.join("idna-1.0.dist-info")).unwrap();
        fs::write(dir.join(COMPLETE_MARKER), format!("idna-1.0-py3-none-any.whl\n{DIGEST}\n")).unwrap();
    }

    const DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000abc";

    #[test]
    fn a_complete_tree_is_never_removed_when_set_aside() {
        // Devin review on #342, round 3: a tree that became complete
        // between the caller's check and the replacement survives the
        // set-aside (it is renamed back), intact.
        let extracted = scratch("complete");
        let target = extracted.join("idna-1.0-py3-none-any");
        complete_tree(&target);
        set_aside_incomplete(&target, &extracted, "idna-1.0-py3-none-any", DIGEST).unwrap();
        assert!(extraction_complete(&target, DIGEST), "the complete tree stays at its path");
        assert!(target.join("idna/__init__.py").is_file());
        let leftovers: Vec<String> = fs::read_dir(&extracted)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(STALE_INFIX))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = fs::remove_dir_all(&extracted);
    }

    #[test]
    fn an_incomplete_tree_is_set_aside_and_removed() {
        let extracted = scratch("incomplete");
        let target = extracted.join("idna-1.0-py3-none-any");
        fs::create_dir_all(target.join("idna-1.0.dist-info")).unwrap();
        set_aside_incomplete(&target, &extracted, "idna-1.0-py3-none-any", DIGEST).unwrap();
        assert!(!target.exists());
        let leftovers: Vec<String> = fs::read_dir(&extracted)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        // A vanished path is fine.
        set_aside_incomplete(&target, &extracted, "idna-1.0-py3-none-any", DIGEST).unwrap();
        // A complete tree of ANOTHER digest (a replaced artifact) is
        // incomplete for this one: set aside and removed.
        complete_tree(&target);
        let other = "1111111111111111111111111111111111111111111111111111111111111111";
        set_aside_incomplete(&target, &extracted, "idna-1.0-py3-none-any", other).unwrap();
        assert!(!target.exists(), "an extraction of a replaced artifact is rebuilt");
        let _ = fs::remove_dir_all(&extracted);
    }

    #[test]
    fn dead_owners_temporaries_are_removed_and_live_ones_kept() {
        // Devin review on #342, round 4: liveness, not age, decides.
        let extracted = scratch("liveness");
        let stem = "idna-1.0-py3-none-any";
        let dead = extracted.join(format!("{stem}{PARTIAL_INFIX}{}-0", u32::MAX));
        let live = extracted.join(format!("{stem}{PARTIAL_INFIX}{}-999999", std::process::id()));
        let stale_dead = extracted.join(format!("{stem}{STALE_INFIX}{}-1", u32::MAX));
        for d in [&dead, &live, &stale_dead] {
            fs::create_dir_all(d.join("idna")).unwrap();
        }
        remove_dead_temporaries(&extracted, stem);
        if cfg!(unix) {
            assert!(!dead.exists(), "a dead owner's partial is removed");
            assert!(!stale_dead.exists(), "a dead owner's set-aside tree is removed");
        }
        assert!(live.exists(), "this process's live temporary stays");
        let _ = fs::remove_dir_all(&extracted);
    }

    #[test]
    fn a_file_publishes_whole_under_concurrent_publishers() {
        // Devin review on #342, round 4: N publishers of one cache path
        // never expose a truncated file — every observation is the whole
        // content, and no temporary is left behind.
        let dir = scratch("publish");
        let path = dir.join("pkg-1.0-py3-none-any.whl");
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let bytes = std::sync::Arc::new(bytes);
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (path, bytes) = (path.clone(), bytes.clone());
                std::thread::spawn(move || {
                    for _ in 0..5 {
                        publish_file(&path, &bytes).unwrap();
                        if let Ok(seen) = fs::read(&path) {
                            assert_eq!(seen.len(), bytes.len(), "a reader saw a partial file");
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(fs::read(&path).unwrap(), *bytes);
        let leftovers: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_artifact_sidecar_pair_never_extracts() {
        // Devin review on #342, round 6: an artifact whose bytes do not
        // hash to its sidecar's digest is a loud mismatch, never an
        // extraction under a digest it does not have.
        let dist_dir = scratch("torn");
        let artifact = dist_dir.join("idna-1.0-py3-none-any.whl");
        fs::write(&artifact, b"not the bytes the sidecar names").unwrap();
        fs::write(digest_sidecar(&artifact), format!("{DIGEST}\n")).unwrap();
        let err = extract_distribution(&artifact, &dist_dir, "idna").unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err:?}");
        assert!(!dist_dir.join("extracted/idna-1.0-py3-none-any").exists());
        let _ = fs::remove_dir_all(&dist_dir);
    }

    #[test]
    fn concurrent_publishers_of_different_contents_never_yield_a_silent_mismatch() {
        // Devin review on #342, round 7: publishers of DIFFERENT
        // downloads for one cache path (each its own consistent
        // artifact/sidecar pair) race with readers; every read either
        // verifies a consistent pair, finds no sidecar yet, or fails
        // LOUDLY on a torn pair — never accepts one silently — and a
        // final consistent publish heals the path.
        let dir = scratch("publish-mixed");
        let path = dir.join("pkg-1.0-py3-none-any.whl");
        let sidecar = digest_sidecar(&path);
        let contents: Vec<Vec<u8>> = (0..4u8)
            .map(|i| (0..50_000u32).map(|k| ((k % 200) as u8).wrapping_add(i)).collect())
            .collect();
        let digests: Vec<String> = contents.iter().map(|b| sha256_hex(b)).collect();
        let contents = std::sync::Arc::new(contents);
        let digests = std::sync::Arc::new(digests);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let (path, stop) = (path.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut verified = 0u32;
                let mut torn = 0u32;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    match cached_artifact_verified(&path) {
                        Ok(true) => verified += 1,
                        Ok(false) => {}
                        Err(e) => {
                            assert!(e.to_string().contains("sha256 mismatch"), "{e:?}");
                            torn += 1;
                        }
                    }
                }
                (verified, torn)
            })
        };
        let writers: Vec<_> = (0..4usize)
            .map(|i| {
                let (path, sidecar, contents, digests) =
                    (path.clone(), sidecar.clone(), contents.clone(), digests.clone());
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        publish_file(&path, &contents[i]).unwrap();
                        publish_file(&sidecar, format!("{}\n", digests[i]).as_bytes()).unwrap();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        // The reader's counts are what the race happened to show; the
        // invariant is that every read was verified, empty, or loud.
        let (_verified, _torn) = reader.join().unwrap();
        // Whatever the race left, a consistent publish heals the path.
        publish_file(&path, &contents[0]).unwrap();
        publish_file(&sidecar, format!("{}\n", digests[0]).as_bytes()).unwrap();
        assert!(cached_artifact_verified(&path).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wildcard_specifiers_match_the_release_prefix_and_exact_markers() {
        // Devin review on #342, round 9: the supported wildcard forms.
        let v = |s: &str| parse_version(s).unwrap();
        assert!(matches_specifier(&v("1.0.3"), "==", "1.0.*"));
        assert!(matches_specifier(&v("1.0"), "==", "1.0.*"));
        assert!(matches_specifier(&v("1"), "==", "1.0.*"), "trailing zeros are implicit");
        assert!(!matches_specifier(&v("1.1"), "==", "1.0.*"));
        assert!(!matches_specifier(&v("10.0"), "==", "1.*"), "a prefix is a segment, not text");
        assert!(matches_specifier(&v("1.0a1"), "==", "1.0.*"), "a prerelease shares the release");
        assert!(matches_specifier(&v("1.0a1"), "==", "1.0a1.*"));
        assert!(!matches_specifier(&v("1.0a2"), "==", "1.0a1.*"), "the marker must match exactly");
        assert!(matches_specifier(&v("1.0.post1"), "==", "1.0.post1.*"));
        assert!(matches_specifier(&v("1.0.3+cpu"), "==", "1.0.*"), "the local label is ignored");
        assert!(!matches_specifier(&v("1.0.3"), "==", "1.0+cpu.*"), "a local prefix matches nothing");
        assert!(!matches_specifier(&v("1.0.3"), "==", "1.x.*"), "a non-version prefix matches nothing");
        assert!(matches_specifier(&v("1.1"), "!=", "1.0.*"));
    }

    #[test]
    fn temporary_names_are_unique_per_invocation() {
        let a = temporary_suffix();
        let b = temporary_suffix();
        assert_ne!(a, b);
        assert!(a.starts_with(&format!("{}-", std::process::id())));
    }
}
