//! Python glob module implementation
//! 
//! This module provides Unix shell-style pathname pattern expansion.
//! Implementation matches Python's glob module API.

use crate::PyException;
use std::path::Path;

/// Find all pathnames matching a given pattern
#[cfg(feature = "std")]
pub fn glob<P: AsRef<str>>(pathname: P) -> Result<Vec<String>, PyException> {
    glob_impl(pathname.as_ref(), false)
}

/// glob with recursive `**` support (CPython's `recursive=True`).
#[cfg(feature = "std")]
fn glob_impl(pattern: &str, recursive: bool) -> Result<Vec<String>, PyException> {
    let mut results = Vec::new();

    // Handle absolute vs relative paths. Results keep the pattern's shape:
    // a relative pattern yields relative paths (issue #82) — the old code
    // resolved the base to current_dir() and pushed absolute paths.
    let is_absolute = pattern.starts_with('/')
        || pattern.starts_with("\\")
        || (pattern.len() > 1 && pattern.chars().nth(1) == Some(':'));
    let base: std::path::PathBuf = if is_absolute {
        std::path::PathBuf::from("/")
    } else {
        std::env::current_dir()
            .map_err(|e| crate::runtime_error(format!("Failed to get current directory: {}", e)))?
    };

    glob_recursive(&base, pattern, recursive, &mut results)?;

    if is_absolute {
        results.sort();
        return Ok(results);
    }
    // Strip the cwd prefix back off so relative patterns stay relative.
    let cwd = std::env::current_dir()
        .map_err(|e| crate::runtime_error(format!("Failed to get current directory: {}", e)))?;
    let cwd_str = cwd.to_string_lossy();
    let mut relative = Vec::new();
    for r in results {
        if let Some(stripped) = r.strip_prefix(&*cwd_str) {
            let stripped = stripped.trim_start_matches('/');
            if stripped.is_empty() {
                // A match of the cwd itself: CPython renders it as ".".
                relative.push(".".to_string());
            } else {
                relative.push(stripped.to_string());
            }
        } else {
            relative.push(r);
        }
    }
    relative.sort();
    Ok(relative)
}

/// Recursive glob implementation
#[cfg(feature = "std")]
fn glob_recursive(
    base_path: &Path,
    pattern: &str,
    recursive: bool,
    results: &mut Vec<String>,
) -> Result<(), PyException> {
    let pattern_parts: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    if pattern_parts.is_empty() {
        return Ok(());
    }

    glob_recursive_helper(base_path, &pattern_parts, 0, recursive, results)
}

#[cfg(feature = "std")]
fn glob_recursive_helper(
    current_path: &Path,
    pattern_parts: &[&str],
    part_index: usize,
    recursive: bool,
    results: &mut Vec<String>,
) -> Result<(), PyException> {
    if part_index >= pattern_parts.len() {
        if current_path.exists() {
            results.push(current_path.to_string_lossy().to_string());
        }
        return Ok(());
    }

    let current_pattern = pattern_parts[part_index];

    // Handle ** (recursive wildcard). CPython's default is
    // recursive=False, where ** behaves exactly like * (issue #82);
    // recursive=True descends (skipping hidden directories, like Python).
    if current_pattern == "**" && recursive {
        // Try matching current directory first
        glob_recursive_helper(current_path, pattern_parts, part_index + 1, recursive, results)?;

        // Then recursively search subdirectories. Python's ** does not
        // descend into hidden directories.
        if let Ok(entries) = std::fs::read_dir(current_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                let hidden = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'));
                if path.is_dir() && !hidden {
                    glob_recursive_helper(&path, pattern_parts, part_index, recursive, results)?;
                }
            }
        }
        return Ok(());
    }

    if !current_path.is_dir() {
        return Ok(());
    }

    let entries = std::fs::read_dir(current_path)
        .map_err(|e| crate::runtime_error(format!("Failed to read directory: {}", e)))?;

    for entry in entries.flatten() {
        let path = entry.path();
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Python's glob never matches a leading dot with a wildcard: a
        // hidden file only matches a pattern component that itself starts
        // with a literal dot.
        if filename.starts_with('.') && !current_pattern.starts_with('.') {
            continue;
        }

        if matches_pattern(filename, current_pattern) {
            if part_index == pattern_parts.len() - 1 {
                // Last part - add to results
                results.push(path.to_string_lossy().to_string());
            } else {
                // Continue with next part
                glob_recursive_helper(&path, pattern_parts, part_index + 1, recursive, results)?;
            }
        }
    }

    Ok(())
}

/// Check if filename matches pattern
fn matches_pattern(filename: &str, pattern: &str) -> bool {
    match_glob(filename, pattern)
}

/// Pattern matching with support for *, ?, [], and {}
fn match_glob(text: &str, pattern: &str) -> bool {
    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    
    match_glob_recursive(&text_chars, &pattern_chars, 0, 0)
}

fn match_glob_recursive(text: &[char], pattern: &[char], text_idx: usize, pattern_idx: usize) -> bool {
    // Base cases
    if pattern_idx >= pattern.len() {
        return text_idx >= text.len();
    }
    
    if text_idx >= text.len() {
        // Check if remaining pattern is only '*'
        return pattern[pattern_idx..].iter().all(|&c| c == '*');
    }
    
    match pattern[pattern_idx] {
        '*' => {
            // Try matching zero or more characters
            match_glob_recursive(text, pattern, text_idx, pattern_idx + 1) ||
            match_glob_recursive(text, pattern, text_idx + 1, pattern_idx)
        }
        '?' => {
            // Match exactly one character
            match_glob_recursive(text, pattern, text_idx + 1, pattern_idx + 1)
        }
        '[' => {
            // Character class matching
            if let Some((matched, next_pattern_idx)) = match_char_class(text[text_idx], pattern, pattern_idx) {
                if matched {
                    match_glob_recursive(text, pattern, text_idx + 1, next_pattern_idx)
                } else {
                    false
                }
            } else {
                false
            }
        }
        c => {
            // Match exact character. NOTE: `{a,b}` brace expansion is
            // deliberately NOT implemented — CPython's glob has no brace
            // expansion, so `{` is an ordinary literal (issue #82).
            if text[text_idx] == c {
                match_glob_recursive(text, pattern, text_idx + 1, pattern_idx + 1)
            } else {
                false
            }
        }
    }
}

/// Match character class like [abc], [a-z], [!abc]
fn match_char_class(ch: char, pattern: &[char], start_idx: usize) -> Option<(bool, usize)> {
    if start_idx >= pattern.len() || pattern[start_idx] != '[' {
        return None;
    }
    
    let mut idx = start_idx + 1;
    if idx >= pattern.len() {
        return None;
    }
    
    // Check for negation
    let negated = pattern[idx] == '!';
    if negated {
        idx += 1;
        if idx >= pattern.len() {
            return None;
        }
    }
    
    let mut matched = false;
    
    while idx < pattern.len() && pattern[idx] != ']' {
        if idx + 2 < pattern.len() && pattern[idx + 1] == '-' {
            // Range like a-z
            let start_char = pattern[idx];
            let end_char = pattern[idx + 2];
            if ch >= start_char && ch <= end_char {
                matched = true;
            }
            idx += 3;
        } else {
            // Single character
            if ch == pattern[idx] {
                matched = true;
            }
            idx += 1;
        }
    }
    
    if idx < pattern.len() && pattern[idx] == ']' {
        idx += 1; // Skip closing bracket
        Some((if negated { !matched } else { matched }, idx))
    } else {
        None
    }
}

/// Parse brace expansion like {a,b,c}
///
/// Intentionally absent: CPython's glob has no brace expansion, so `{` is
/// an ordinary literal character (issue #82). Kept as a tombstone so nobody
/// re-adds shell-style braces in the mistaken belief glob supports them.
#[allow(dead_code)]
fn parse_brace_expansion_removed() {}

/// Find all pathnames matching pattern (with recursive search)
#[cfg(feature = "std")]
pub fn rglob<P: AsRef<str>>(pathname: P) -> Result<Vec<String>, PyException> {
    let pattern = pathname.as_ref();
    // Add ** to make it recursive if not already present
    let recursive_pattern = if pattern.contains("**") {
        pattern.to_string()
    } else {
        format!("**/{}", pattern)
    };

    glob_impl(&recursive_pattern, true)
}

/// Determine if pathname is a hidden file
pub fn is_hidden<P: AsRef<str>>(pathname: P) -> bool {
    let path_str = pathname.as_ref();
    if let Some(filename) = Path::new(path_str).file_name() {
        if let Some(name) = filename.to_str() {
            return name.starts_with('.') && name != "." && name != "..";
        }
    }
    false
}

/// Escape glob metacharacters with the bracket escapes CPython uses:
/// `*` -> `[*]`, `?` -> `[?]`, `[` -> `[[]`; `]`, `{`, `}` are not magic
/// and stay untouched (issue #82). The old backslash escapes were not
/// readable by this module's own matcher, so `glob(escape(name))` never
/// round-tripped.
pub fn escape<P: AsRef<str>>(pathname: P) -> String {
    let path_str = pathname.as_ref();
    let mut result = String::new();

    for ch in path_str.chars() {
        match ch {
            '*' => result.push_str("[*]"),
            '?' => result.push_str("[?]"),
            '[' => result.push_str("[[]"),
            c => result.push(c),
        }
    }

    result
}

/// Check if string has glob metacharacters
pub fn has_magic<P: AsRef<str>>(pathname: P) -> bool {
    let path_str = pathname.as_ref();
    path_str.chars().any(|c| matches!(c, '*' | '?' | '['))
}

/// Same as glob but returns iterator (simplified - returns Vec for now)
#[cfg(feature = "std")]
pub fn iglob<P: AsRef<str>>(pathname: P) -> Result<Vec<String>, PyException> {
    glob(pathname)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_match_glob() {
        assert!(match_glob("hello.txt", "*.txt"));
        assert!(match_glob("hello.py", "hello.*"));
        assert!(match_glob("test", "t?st"));
        assert!(match_glob("file", "f[aeiou]le"));
        assert!(!match_glob("file.py", "*.txt"));
        assert!(!match_glob("hello", "h[xyz]llo"));
    }
    
    #[test]
    fn test_char_class() {
        assert!(matches!(match_char_class('a', &['[', 'a', 'b', 'c', ']'], 0), Some((true, 5))));
        assert!(matches!(match_char_class('d', &['[', 'a', 'b', 'c', ']'], 0), Some((false, 5))));
        assert!(matches!(match_char_class('b', &['[', 'a', '-', 'z', ']'], 0), Some((true, 5))));
        assert!(matches!(match_char_class('a', &['[', '!', 'b', 'c', ']'], 0), Some((true, 5))));
    }
    
    #[test]
    fn test_escape() {
        // CPython's bracket escapes, readable by this module's own matcher.
        assert_eq!(escape("file*.txt"), "file[*].txt");
        assert_eq!(escape("test[123].py"), "test[[]123].py");
        assert_eq!(escape("a?b{c}d]e"), "a[?]b{c}d]e");
        assert_eq!(escape("normal_file.txt"), "normal_file.txt");
        // The matcher round-trips what escape produces.
        assert!(match_glob("file*.txt", &escape("file*.txt")));
        assert!(match_glob("test[123].py", &escape("test[123].py")));
        assert!(!match_glob("fileX.txt", &escape("file*.txt")));
    }
    
    #[test]
    fn test_has_magic() {
        assert!(has_magic("*.txt"));
        assert!(has_magic("file?.py"));
        assert!(has_magic("test[123]"));
        assert!(!has_magic("{a,b,c}"));
        assert!(!has_magic("normal_file.txt"));
    }
    
    #[test]
    fn test_is_hidden() {
        assert!(is_hidden(".bashrc"));
        assert!(is_hidden(".hidden"));
        assert!(!is_hidden("visible.txt"));
        assert!(!is_hidden("."));
        assert!(!is_hidden(".."));
    }
}