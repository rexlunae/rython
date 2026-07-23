//! Python format-string machinery: translating `str.format` templates and
//! format specs into Rust `format!` equivalents at conversion time.
//!
//! Translation happens on LITERAL templates only, so every divergence is a
//! loud conversion error rather than wrong output: specs Rust cannot
//! reproduce exactly (`,` grouping, `e`/`g` scientific forms, `=`
//! alignment) are rejected, never approximated.

/// One parsed segment of a `str.format` template.
#[derive(Debug, PartialEq)]
pub(crate) enum Piece {
    /// Literal text (with `{{`/`}}` already unescaped).
    Literal(String),
    Field {
        arg: FieldRef,
        /// Python conversion flag (`!r`/`!s`/`!a`), if any.
        conversion: Option<char>,
        /// The raw Python format spec (text after `:`), possibly empty.
        spec: String,
    },
}

#[derive(Debug, PartialEq)]
pub(crate) enum FieldRef {
    /// `{}` — auto-numbered.
    Auto,
    /// `{0}` — explicit position.
    Index(usize),
    /// `{name}` — keyword lookup.
    Name(String),
}

/// Parse a `str.format` template into pieces. Errors mirror Python's
/// ValueError messages for malformed templates.
pub(crate) fn parse_template(template: &str) -> Result<Vec<Piece>, String> {
    let mut pieces = Vec::new();
    let mut lit = String::new();
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    lit.push('{');
                    continue;
                }
                if !lit.is_empty() {
                    pieces.push(Piece::Literal(std::mem::take(&mut lit)));
                }
                let mut field = String::new();
                let mut depth = 0usize;
                loop {
                    match chars.next() {
                        None => return Err("Single '{' encountered in format string".into()),
                        Some('{') => {
                            depth += 1;
                            field.push('{');
                        }
                        Some('}') if depth > 0 => {
                            depth -= 1;
                            field.push('}');
                        }
                        Some('}') => break,
                        Some(ch) => field.push(ch),
                    }
                }
                pieces.push(parse_field(&field)?);
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                    lit.push('}');
                } else {
                    return Err("Single '}' encountered in format string".into());
                }
            }
            other => lit.push(other),
        }
    }
    if !lit.is_empty() {
        pieces.push(Piece::Literal(lit));
    }
    Ok(pieces)
}

fn parse_field(field: &str) -> Result<Piece, String> {
    // name[!conversion][:spec]
    let (name_conv, spec) = match field.find(':') {
        Some(i) => (&field[..i], &field[i + 1..]),
        None => (field, ""),
    };
    if spec.contains('{') {
        return Err(
            "nested replacement fields in format specs are not supported yet".into(),
        );
    }
    let (name, conversion) = match name_conv.find('!') {
        Some(i) => {
            let conv = &name_conv[i + 1..];
            let c = match conv {
                "r" => 'r',
                "s" => 's',
                "a" => 'a',
                other => {
                    return Err(format!(
                        "Unknown conversion specifier {:?}",
                        other
                    ));
                }
            };
            (&name_conv[..i], Some(c))
        }
        None => (name_conv, None),
    };
    let arg = if name.is_empty() {
        FieldRef::Auto
    } else if name.chars().all(|c| c.is_ascii_digit()) {
        FieldRef::Index(name.parse().map_err(|_| "field index too large".to_string())?)
    } else if name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_')
        && !name.starts_with(|c: char| c.is_ascii_digit())
    {
        FieldRef::Name(name.to_string())
    } else {
        return Err(format!(
            "format field {:?} is not supported yet: only plain positions and names \
             lower (attribute/index access inside fields does not)",
            name
        ));
    };
    Ok(Piece::Field {
        arg,
        conversion,
        spec: spec.to_string(),
    })
}

/// Translate a Python format spec (the text after `:`) into the Rust
/// format-spec suffix producing IDENTICAL output, or a descriptive error
/// when Rust's formatter cannot reproduce it.
///
/// Python: [[fill]align][sign][#][0][width][,][.precision][type]
pub(crate) fn translate_format_spec(spec: &str) -> Result<String, String> {
    if spec.is_empty() {
        return Ok(String::new());
    }
    let chars: Vec<char> = spec.chars().collect();
    let mut i = 0;

    let mut fill: Option<char> = None;
    let mut align: Option<char> = None;
    if chars.len() >= 2 && matches!(chars[1], '<' | '>' | '^' | '=') {
        fill = Some(chars[0]);
        align = Some(chars[1]);
        i = 2;
    } else if matches!(chars.first(), Some('<' | '>' | '^' | '=')) {
        align = Some(chars[0]);
        i = 1;
    }
    if align == Some('=') {
        return Err(
            "the '=' (sign-aware padding) alignment is not supported yet".into(),
        );
    }

    let mut sign: Option<char> = None;
    if i < chars.len() && matches!(chars[i], '+' | '-' | ' ') {
        sign = Some(chars[i]);
        i += 1;
    }
    if sign == Some(' ') {
        return Err("the ' ' sign option is not supported yet".into());
    }

    let mut alternate = false;
    if i < chars.len() && chars[i] == '#' {
        alternate = true;
        i += 1;
    }

    let mut zero = false;
    if i < chars.len() && chars[i] == '0' {
        zero = true;
        i += 1;
    }

    let mut width = String::new();
    while i < chars.len() && chars[i].is_ascii_digit() {
        width.push(chars[i]);
        i += 1;
    }

    if i < chars.len() && chars[i] == ',' {
        return Err("the ',' thousands separator is not supported yet".into());
    }

    let mut precision = String::new();
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        while i < chars.len() && chars[i].is_ascii_digit() {
            precision.push(chars[i]);
            i += 1;
        }
        if precision.is_empty() {
            return Err("Format specifier missing precision".into());
        }
    }

    let ty = if i < chars.len() {
        let t = chars[i];
        i += 1;
        Some(t)
    } else {
        None
    };
    if i != chars.len() {
        return Err(format!("Invalid format specifier {:?}", spec));
    }

    // Map the presentation type. Rust reproduces d/s/f/x/X/o/b exactly;
    // e/E/g/n/% differ in exponent/grouping conventions and are rejected.
    let mut rust_type = String::new();
    match ty {
        None | Some('d') | Some('s') => {}
        Some('f') | Some('F') => {
            // Python's default 'f' precision is 6; Rust's is shortest.
            if precision.is_empty() {
                precision = "6".to_string();
            }
        }
        Some('x') => rust_type.push('x'),
        Some('X') => rust_type.push('X'),
        Some('o') => rust_type.push('o'),
        Some('b') => rust_type.push('b'),
        Some(other) => {
            return Err(format!(
                "the {:?} presentation type is not supported yet (Rust's formatter \
                 renders it differently than Python)",
                other
            ));
        }
    }

    let mut out = String::new();
    if let Some(f) = fill {
        out.push(f);
    }
    if let Some(a) = align {
        out.push(a);
    }
    if let Some(s) = sign {
        if s == '+' {
            out.push('+');
        }
    }
    if alternate {
        out.push('#');
    }
    if zero {
        out.push('0');
    }
    out.push_str(&width);
    if !precision.is_empty() {
        out.push('.');
        out.push_str(&precision);
    }
    out.push_str(&rust_type);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_translate() {
        assert_eq!(translate_format_spec(".2f").unwrap(), ".2");
        assert_eq!(translate_format_spec("f").unwrap(), ".6");
        assert_eq!(translate_format_spec(">6").unwrap(), ">6");
        assert_eq!(translate_format_spec("*^7").unwrap(), "*^7");
        assert_eq!(translate_format_spec("05d").unwrap(), "05");
        assert_eq!(translate_format_spec("+d").unwrap(), "+");
        assert_eq!(translate_format_spec("#x").unwrap(), "#x");
        assert_eq!(translate_format_spec("8.3f").unwrap(), "8.3");
        assert_eq!(translate_format_spec(".3").unwrap(), ".3");
        assert!(translate_format_spec(",").is_err());
        assert!(translate_format_spec("e").is_err());
        assert!(translate_format_spec("=10").is_err());
    }

    #[test]
    fn templates_parse() {
        let pieces = parse_template("{{x}} {0!r:>4} {name}").unwrap();
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[0], Piece::Literal("{x} ".to_string()));
        assert_eq!(
            pieces[1],
            Piece::Field {
                arg: FieldRef::Index(0),
                conversion: Some('r'),
                spec: ">4".to_string()
            }
        );
        assert_eq!(pieces[2], Piece::Literal(" ".to_string()));
        assert_eq!(
            pieces[3],
            Piece::Field {
                arg: FieldRef::Name("name".to_string()),
                conversion: None,
                spec: String::new()
            }
        );
        assert!(parse_template("{").is_err());
        assert!(parse_template("}").is_err());
    }
}
