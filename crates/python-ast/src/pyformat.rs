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

/// How a translated spec lowers into generated code.
#[derive(Debug, PartialEq)]
pub(crate) enum SpecLowering {
    /// A Rust format-spec suffix (after `:`) producing identical output.
    Inline(String),
    /// Same, but the operand must first coerce to f64: Python's `f` type
    /// formats integers as floats ("{:.2f}".format(5) == "5.00"), while
    /// Rust ignores precision on integers.
    CastF64(String),
    /// Radix types (x/X/o/b): Rust renders negative integers as their
    /// two's-complement bit pattern where Python uses sign+magnitude, so
    /// these route through the stdpython runtime formatter.
    IntRadix {
        fill: char,
        align: char,
        plus: bool,
        alternate: bool,
        zero: bool,
        width: usize,
        radix: char,
    },
}

/// The lowering for a field that carries a `!r`/`!a` conversion: the
/// translated spec's fill/align/width/precision apply to the debug
/// rendering (Rust `{:>10?}`), matching Python's repr-then-format order.
/// Numeric presentation types cannot combine with a conversion — Python
/// applies them to the repr STRING and raises — so those stay loud.
pub(crate) fn conversion_lowering(spec: &str) -> Result<SpecLowering, String> {
    match translate_format_spec(spec)? {
        SpecLowering::Inline(suffix) => Ok(SpecLowering::Inline(format!("{}?", suffix))),
        _ => Err(
            "numeric presentation types cannot combine with !r/!a (Python applies \
             the spec to the repr string and raises)"
                .into(),
        ),
    }
}

/// Translate a Python format spec (the text after `:`) into a lowering
/// producing IDENTICAL output, or a descriptive error when Rust's
/// formatter cannot reproduce it.
///
/// Python: [[fill]align][sign][#][0][width][,][.precision][type]
pub(crate) fn translate_format_spec(spec: &str) -> Result<SpecLowering, String> {
    if spec.is_empty() {
        return Ok(SpecLowering::Inline(String::new()));
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

    // Map the presentation type. Rust reproduces d/s and (with the f64
    // coercion) f/F exactly; radix types go through the runtime formatter
    // (Python renders negatives as sign+magnitude, Rust as two's
    // complement); e/E/g/n/% differ in exponent/grouping conventions and
    // are rejected.
    if ty.is_none() && !precision.is_empty() {
        return Err(
            "a precision with no presentation type is ambiguous: Python renders \
             floats in 'general' format there (significant figures, possibly \
             scientific notation), which Rust cannot reproduce — use .Ns for \
             string truncation or .Nf for fixed-point"
                .into(),
        );
    }

    let mut cast_f64 = false;
    match ty {
        None | Some('d') | Some('s') => {}
        Some('f') | Some('F') => {
            cast_f64 = true;
            // Python's default 'f' precision is 6; Rust's is shortest.
            if precision.is_empty() {
                precision = "6".to_string();
            }
        }
        Some(radix @ ('x' | 'X' | 'o' | 'b')) => {
            if !precision.is_empty() {
                return Err("precision not allowed in integer format specifier".into());
            }
            return Ok(SpecLowering::IntRadix {
                fill: fill.unwrap_or(' '),
                align: align.unwrap_or('\0'),
                plus: sign == Some('+'),
                alternate,
                zero,
                width: width.parse().unwrap_or(0),
                radix,
            });
        }
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
    if cast_f64 {
        Ok(SpecLowering::CastF64(out))
    } else {
        Ok(SpecLowering::Inline(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_translate() {
        use SpecLowering::*;
        assert_eq!(translate_format_spec(".2f").unwrap(), CastF64(".2".into()));
        assert_eq!(translate_format_spec("f").unwrap(), CastF64(".6".into()));
        assert_eq!(translate_format_spec(">6").unwrap(), Inline(">6".into()));
        assert_eq!(translate_format_spec("*^7").unwrap(), Inline("*^7".into()));
        assert_eq!(translate_format_spec("05d").unwrap(), Inline("05".into()));
        assert_eq!(translate_format_spec("+d").unwrap(), Inline("+".into()));
        assert_eq!(
            translate_format_spec("#x").unwrap(),
            IntRadix { fill: ' ', align: '\0', plus: false, alternate: true, zero: false, width: 0, radix: 'x' }
        );
        assert_eq!(translate_format_spec("8.3f").unwrap(), CastF64("8.3".into()));
        assert_eq!(translate_format_spec(".3s").unwrap(), Inline(".3".into()));
        // Bare precision is Python's general float format — ambiguous
        // without the operand type, so it is loud.
        assert!(translate_format_spec(".3").is_err());
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
