//! Python string module implementation
//! 
//! This module provides string-related constants and template classes.
//! Implementation matches Python's string module API.

use crate::{PyException, python_function};
use alloc::{format, string::String, string::ToString, vec::Vec};

// String constants
pub const ascii_lowercase: &str = "abcdefghijklmnopqrstuvwxyz";
pub const ascii_uppercase: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const ascii_letters: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const digits: &str = "0123456789";
pub const hexdigits: &str = "0123456789abcdefABCDEF";
pub const octdigits: &str = "01234567";
pub const punctuation: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
pub const printable: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c";
pub const whitespace: &str = " \t\n\r\x0b\x0c";

python_function! {
    /// string.capwords - capitalize words in string
    pub fn capwords<S>(s: S, sep: Option<String>) -> String
    where [S: AsRef<str>]
    [signature: (s, sep=None)]
    [concrete_types: (String, Option<String>) -> String]
    {
        let s = s.as_ref();
        match sep {
            Some(separator) => {
                s.split(&separator)
                    .map(|word| {
                        let mut chars: Vec<char> = word.chars().collect();
                        if let Some(first_char) = chars.first_mut() {
                            *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
                        }
                        for ch in chars.iter_mut().skip(1) {
                            *ch = ch.to_lowercase().next().unwrap_or(*ch);
                        }
                        chars.into_iter().collect::<String>()
                    })
                    .collect::<Vec<String>>()
                    .join(&separator)
            }
            None => {
                s.split_whitespace()
                    .map(|word| {
                        let mut chars: Vec<char> = word.chars().collect();
                        if let Some(first_char) = chars.first_mut() {
                            *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
                        }
                        for ch in chars.iter_mut().skip(1) {
                            *ch = ch.to_lowercase().next().unwrap_or(*ch);
                        }
                        chars.into_iter().collect::<String>()
                    })
                    .collect::<Vec<String>>()
                    .join(" ")
            }
        }
    }
}

/// Template - simple template substitution
#[derive(Debug, Clone)]
pub struct Template {
    template: String,
    delimiter: char,
}

impl Template {
    /// Create a new template
    pub fn new<S: AsRef<str>>(template: S) -> Self {
        Self {
            template: template.as_ref().to_string(),
            delimiter: '$',
        }
    }
    
    /// Create template with custom delimiter
    pub fn with_delimiter<S: AsRef<str>>(template: S, delimiter: char) -> Self {
        Self {
            template: template.as_ref().to_string(),
            delimiter,
        }
    }
    
    /// Substitute variables from mapping. A single left-to-right scan
    /// (issue #82): `$$` is an escaped delimiter, `$name`/`${name}` resolve
    /// to the longest identifier, and substituted VALUES are never
    /// re-scanned (CPython's regex-based sub). A missing variable raises
    /// KeyError('name') — message is the quoted key, not a prose string.
    pub fn substitute<K, V>(&self, mapping: &[(K, V)]) -> Result<String, PyException>
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        self.substitute_impl(mapping, false)
    }
    
    /// Safe substitute - leave unmatched and invalid placeholders as-is
    pub fn safe_substitute<K, V>(&self, mapping: &[(K, V)]) -> String
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        self.substitute_impl(mapping, true).expect("safe_substitute never errors")
    }

    fn substitute_impl<K, V>(&self, mapping: &[(K, V)], safe: bool) -> Result<String, PyException>
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let chars: Vec<char> = self.template.chars().collect();
        let mut result = String::with_capacity(self.template.len());
        let mut i = 0;
        while i < chars.len() {
            if chars[i] != self.delimiter {
                result.push(chars[i]);
                i += 1;
                continue;
            }
            // At a delimiter. CPython's scanner treats anything after a
            // delimiter that is not an escape, an identifier, or a braced
            // identifier as an invalid placeholder (ValueError) — or, in
            // safe mode, leaves it literal.
            let invalid = |col: usize| {
                crate::value_error(format!(
                    "Invalid placeholder in string: line 1, col {}",
                    col
                ))
            };
            let Some(&next) = chars.get(i + 1) else {
                if safe {
                    result.push(self.delimiter);
                    i += 1;
                    continue;
                }
                return Err(invalid(i + 1));
            };
            if next == self.delimiter {
                // Escaped delimiter: $$ -> $.
                result.push(self.delimiter);
                i += 2;
                continue;
            }
            if next == '{' {
                let mut end = i + 2;
                while end < chars.len()
                    && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }
                let name: String = chars[i + 2..end].iter().collect();
                let well_formed = !name.is_empty()
                    && name
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    && chars.get(end) == Some(&'}');
                if !well_formed {
                    if safe {
                        result.push(self.delimiter);
                        result.push('{');
                        result.push_str(&name);
                        i = end;
                        continue;
                    }
                    return Err(invalid(i + 1));
                }
                match lookup(mapping, &name) {
                    Some(v) => {
                        result.push_str(v.as_ref());
                        i = end + 1;
                    }
                    None if safe => {
                        result.push_str(&format!("{}{{{}}}", self.delimiter, name));
                        i = end + 1;
                    }
                    None => {
                        return Err(crate::key_error(format!("'{}'", name)));
                    }
                }
                continue;
            }
            if next.is_ascii_alphabetic() || next == '_' {
                // Simple identifier: scan the longest run of identifier
                // characters so `$ab` never shadows `$ab` with a prefix
                // key `a` (issue #82).
                let mut end = i + 2;
                while end < chars.len()
                    && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }
                let name: String = chars[i + 1..end].iter().collect();
                match lookup(mapping, &name) {
                    Some(v) => {
                        result.push_str(v.as_ref());
                        i = end;
                    }
                    None if safe => {
                        result.push_str(&format!("{}{}", self.delimiter, name));
                        i = end;
                    }
                    None => {
                        return Err(crate::key_error(format!("'{}'", name)));
                    }
                }
                continue;
            }
            // Invalid placeholder: digit, space, punctuation after $.
            if safe {
                result.push(self.delimiter);
                i += 1;
                continue;
            }
            return Err(invalid(i + 1));
        }
        Ok(result)
    }
    
    /// Get template string
    pub fn template_string(&self) -> &str {
        &self.template
    }
    
    /// Get delimiter
    pub fn delimiter(&self) -> char {
        self.delimiter
    }
}

impl core::fmt::Display for Template {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{}", self.template)
    }
}

/// First pair whose key matches (dict semantics: the caller's mapping
/// slice should already be deduplicated by construction).
fn lookup<'a, K, V>(mapping: &'a [(K, V)], name: &str) -> Option<&'a V>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    mapping
        .iter()
        .find(|(k, _)| k.as_ref() == name)
        .map(|(_, v)| v)
}

/// Formatter - string formatting operations
#[derive(Debug)]
pub struct Formatter;

impl Formatter {
    /// Format string with positional arguments
    pub fn format<S: AsRef<str>>(template: S, args: &[&dyn core::fmt::Display]) -> String {
        let template = template.as_ref();
        let mut result = template.to_string();
        
        for (i, arg) in args.iter().enumerate() {
            let placeholder = format!("{{{}}}", i);
            result = result.replace(&placeholder, &format!("{}", arg));
        }
        
        result
    }
    
    /// Format string with named arguments
    pub fn format_map<S: AsRef<str>, K: AsRef<str>, V: core::fmt::Display>(
        template: S, 
        kwargs: &[(K, V)]
    ) -> String {
        let template = template.as_ref();
        let mut result = template.to_string();
        
        for (key, value) in kwargs {
            let placeholder = format!("{{{}}}", key.as_ref());
            result = result.replace(&placeholder, &format!("{}", value));
        }
        
        result
    }
    
    /// Validate format string
    pub fn vformat<S: AsRef<str>>(template: S) -> Result<Vec<String>, PyException> {
        let template = template.as_ref();
        let mut placeholders = Vec::new();
        let mut chars = template.chars().peekable();
        
        while let Some(ch) = chars.next() {
            if ch == '{' {
                if chars.peek() == Some(&'{') {
                    chars.next(); // Skip escaped brace
                    continue;
                }
                
                let mut placeholder = String::new();
                let mut found_end = false;
                
                while let Some(ch) = chars.next() {
                    if ch == '}' {
                        found_end = true;
                        break;
                    }
                    placeholder.push(ch);
                }
                
                if !found_end {
                    return Err(crate::value_error("Unmatched '{' in format string"));
                }
                
                placeholders.push(placeholder);
            } else if ch == '}' {
                if chars.peek() != Some(&'}') {
                    return Err(crate::value_error("Unmatched '}' in format string"));
                }
                chars.next(); // Skip escaped brace
            }
        }
        
        Ok(placeholders)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_capwords() {
        assert_eq!(capwords("hello world", None), "Hello World");
        assert_eq!(capwords("hello-world", Some("-".to_string())), "Hello-World");
        assert_eq!(capwords("HELLO WORLD", None), "Hello World");
    }
    
    #[test]
    fn test_template() {
        let tmpl = Template::new("Hello $name! Welcome to $place.");
        let mapping = [("name", "Alice"), ("place", "Wonderland")];
        
        assert_eq!(
            tmpl.substitute(&mapping).unwrap(),
            "Hello Alice! Welcome to Wonderland."
        );
        
        let tmpl2 = Template::new("Hello ${name}! Welcome to ${place}.");
        assert_eq!(
            tmpl2.substitute(&mapping).unwrap(),
            "Hello Alice! Welcome to Wonderland."
        );
    }
    
    #[test]
    fn test_template_missing_var() {
        let tmpl = Template::new("Hello $name! Welcome to $place.");
        let mapping = [("name", "Alice")]; // missing 'place'
        
        assert!(tmpl.substitute(&mapping).is_err());
    }
    
    #[test]
    fn test_safe_substitute() {
        let tmpl = Template::new("Hello $name! Welcome to $place.");
        let mapping = [("name", "Alice")]; // missing 'place'
        
        assert_eq!(
            tmpl.safe_substitute(&mapping),
            "Hello Alice! Welcome to $place."
        );
    }
}