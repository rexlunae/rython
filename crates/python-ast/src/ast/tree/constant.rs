use std::fmt::*;

use litrs::Literal;
use tracing::debug;
use proc_macro2::*;
use pyo3::{Borrowed, Bound, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;

use crate::{CodeGen, CodeGenContext, PythonOptions, SymbolTableScopes};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub trait PyConstantTrait: Clone + Debug + PartialEq {
    type RustType;
}

#[derive(Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Constant(pub Option<Literal<String>>);

impl Serialize for Constant {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

impl<'de> Deserialize<'de> for Constant {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let l = Literal::parse(s).expect("[3] Parsing the literal");
        Ok(Self(Some(l)))
    }
}

impl std::string::ToString for Constant {
    fn to_string(&self) -> String {
        match self.0.clone() {
            Some(c) => c.to_string(),
            None => "None".to_string(),
        }
    }
}

pub fn try_string(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    let v: String = value.extract()?;
    // Debug-format the string so quotes, backslashes, and control characters
    // come out as valid Rust escape sequences.
    let l = Literal::parse(format!("{:?}", v)).expect("[4] Parsing the literal");

    Ok(Some(l))
}

pub fn try_bytes(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    let v: &[u8] = value.extract()?;
    // Rust byte-string literals only allow ASCII plus escapes, so escape every
    // byte that needs it (non-ASCII bytes become \xNN).
    let escaped: String = v
        .iter()
        .flat_map(|b| std::ascii::escape_default(*b))
        .map(char::from)
        .collect();
    let l = Literal::parse(format!("b\"{}\"", escaped)).expect("[4] Parsing the literal");

    Ok(Some(l))
}

pub fn try_int(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    let v: isize = value.extract()?;
    let l = Literal::parse(format!("{}", v)).expect("[4] Parsing the literal");

    Ok(Some(l))
}

pub fn try_float(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    let v: f64 = value.extract()?;
    // A NON-FINITE float constant (`1e1000` → inf, `0/0`-free literals →
    // nan) has no Rust float-LITERAL form: "inf"/"nan" are not valid Rust
    // literals, so Literal::parse would panic. Carry it as a NUL-prefixed
    // sentinel (like complex/Ellipsis) and render as the f64 EXPRESSION at
    // codegen (issue #372).
    if !v.is_finite() {
        let kind = if v.is_nan() {
            "nan"
        } else if v.is_sign_negative() {
            "ninf"
        } else {
            "inf"
        };
        return Ok(Some(Literal::parse(
            format!("\"\u{0}RYTHON_NONFINITE:{}\"", kind)
        )
        .expect("non-finite sentinel literal")));
    }
    // Rust's Display for integral floats drops the ".0" ("2.0" becomes
    // "2"), which would re-parse as an INTEGER literal and silently change
    // the generated type (and semantics — Python's 2.0 / 4 is 0.5).
    let mut s = format!("{}", v);
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    let l = Literal::parse(s).expect("[4] Parsing the literal");

    Ok(Some(l))
}

/// True when `l` is a non-finite float constant sentinel (issue #372).
pub fn is_nonfinite_literal(l: &Literal<String>) -> bool {
    l.to_string().starts_with("\"\u{0}RYTHON_NONFINITE:")
}

/// The Rust EXPRESSION tokens for a non-finite float constant
/// (`f64::INFINITY` / `f64::NEG_INFINITY` / `f64::NAN`), or None when `l`
/// is not a non-finite sentinel.
pub fn nonfinite_expression(l: &Literal<String>) -> Option<TokenStream> {
    let s = l.to_string();
    let raw: &str = s.as_ref();
    if !raw.starts_with("\"\u{0}RYTHON_NONFINITE:") {
        return None;
    }
    match raw {
        _ if raw.contains("\u{0}RYTHON_NONFINITE:nan") => Some(quote!(f64::NAN)),
        _ if raw.contains("\u{0}RYTHON_NONFINITE:ninf") => {
            Some(quote!(f64::NEG_INFINITY))
        }
        _ => Some(quote!(f64::INFINITY)),
    }
}

pub fn try_bool(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    let v: bool = value.extract()?;
    let l = Literal::parse(format!("{}", v)).expect("[4] Parsing the literal");

    Ok(Some(l))
}

/// A Rust float-literal token for an `f64`, mirroring `try_float`: Rust's
/// Display drops the ".0" of an integral float, which would re-parse as an
/// INTEGER literal and change the type.
pub fn f64_token(v: f64) -> String {
    let mut s = format!("{}", v);
    if v.is_finite() && !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    s
}

/// The NUL-prefixed sentinel marker used to carry a Python `complex` value
/// through the `Literal<String>` (there is no `Literal` complex variant).
/// A NUL byte cannot appear in a real Python source literal, so this cannot
/// collide with a genuine string constant (the same rationale as the
/// Ellipsis sentinel).
const COMPLEX_MARKER: &'static str = "\u{0}RYTHON_COMPLEX:";

pub fn complex_sentinel_literal(re: f64, im: f64) -> Literal<String> {
    Literal::parse(
        format!("\"{}{}:{}\"", COMPLEX_MARKER, re.to_bits(), im.to_bits())
    )
    .expect("complex sentinel literal")
}

/// True when `l` is a complex sentinel (a real Python string can never
/// start with the NUL marker plus its opening quote).
pub fn is_complex_literal(l: &Literal<String>) -> bool {
    l.to_string().starts_with("\"\u{0}RYTHON_COMPLEX:")
}

/// The stored (re, im) f64s from a complex sentinel, or None when `l` is
/// not a complex literal.
pub fn complex_parts(l: &Literal<String>) -> Option<(f64, f64)> {
    let s = l.to_string();
    let raw: &str = s.as_ref();
    // raw == "\u{0}RYTHON_COMPLEX:<re_bits>:<im_bits>"
    let marker = "\"\u{0}RYTHON_COMPLEX:";
    if !raw.starts_with(marker) {
        return None;
    }
    // Strip the leading quote + marker (marker includes the opening quote).
    let inner = &raw[marker.len()..];
    let body = if inner.ends_with('"') {
        &inner[..inner.len() - 1]
    } else {
        inner
    };
    // body == "<re_bits>:<im_bits>"
    let Some(i) = body.find(':') else {
        return None;
    };
    let re: u64 = (&body[..i]).parse::<u64>().unwrap();
    let im: u64 = (&body[i + 1..]).parse::<u64>().unwrap();
    Some((f64::from_bits(re), f64::from_bits(im)))
}

/// Python `complex` constant extraction. The caller's `if let Ok` chain
/// treats an `Err` as "not this kind", so a value without `.real`/`.imag`
/// (a string, Ellipsis, None, ...) simply falls through; only a real
/// `complex` has both, plus the earlier try_* arms already rejected
/// int/float/bool (which also answer `.real`).
pub fn try_complex(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    let re: f64 = value
        .getattr("real")
        .map_err(|e| crate::extraction_failure("complex real", value, e))?
        .extract::<f64>()
        .map_err(|e| crate::extraction_failure("complex real", value, e))?;
    let im: f64 = value
        .getattr("imag")
        .map_err(|e| crate::extraction_failure("complex imag", value, e))?
        .extract::<f64>()
        .map_err(|e| crate::extraction_failure("complex imag", value, e))?;
    Ok(Some(complex_sentinel_literal(re, im)))
}

// Sentinel literal stored for Python's `...` (Ellipsis): the extraction
// must succeed (a Protocol stub `def f(...) -> None: ...` is everywhere),
// and the value/statement codegen then decides — a bare `...` statement is
// a no-op (like `pass`), while using Ellipsis as a value is a loud error.
// A NUL character cannot appear in real Python source literals, so the
// sentinel string cannot collide with a genuine string constant.
pub fn ellipsis_sentinel() -> Literal<String> {
    Literal::parse("\"\u{0}RYTHON_ELLIPSIS\"".to_string()).expect("ellipsis sentinel literal")
}

pub fn is_ellipsis_literal(l: &Literal<String>) -> bool {
    l.to_string() == ellipsis_sentinel().to_string()
}

/// Python's `...` (Ellipsis) singleton: extracted as the sentinel literal so
/// parsing succeeds; codegen decides between no-op (bare statement) and loud
/// error (value use).
pub fn try_ellipsis(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    if value.is(&pyo3::types::PyEllipsis::get(value.py())) {
        Ok(Some(ellipsis_sentinel()))
    } else {
        Err(crate::extraction_failure(
            "constant",
            value,
            "expected Ellipsis",
        ))
    }
}

// This will mostly be invoked when the input is None.
pub fn try_option(value: &Bound<PyAny>) -> PyResult<Option<Literal<String>>> {
    let v: Option<Bound<PyAny>> = value.extract()?;
    // If we got None as a constant, return None; anything else is a constant
    // kind we don't know how to render as a Rust literal.
    match v {
        None => Ok(None),
        Some(other) => Err(crate::extraction_failure(
            "constant",
            &other,
            "this constant kind cannot be represented as a Rust literal",
        )),
    }
}

// This is the fun bit of code that is responsible from converting from Python constants to Rust ones.
impl<'a, 'py> FromPyObject<'a, 'py> for Constant {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extracts the values as a PyAny.
        let value = ob
            .getattr("value")
            .map_err(|e| crate::extraction_failure("constant value", &ob, e))?;
        debug!("[2] constant value: {}", value);

        let l = if let Ok(l) = try_string(&value) {
            l
        } else if let Ok(l) = try_bytes(&value) {
            l
        // We have to evaluaet bool before int because if a bool is evaluated as it, it will be cooerced to an in.
        } else if let Ok(l) = try_bool(&value) {
            l
        // Ints must be tried before floats: extracting f64 from a Python int
        // succeeds, and would silently lose precision above 2^53.
        } else if let Ok(l) = try_int(&value) {
            l
        } else if let Ok(l) = try_float(&value) {
            l
        } else if let Ok(l) = try_complex(&value) {
            l
        } else if let Ok(l) = try_ellipsis(&value) {
            l
        } else if let Ok(l) = try_option(&value) {
            l
        } else {
            return Err(crate::extraction_failure(
                "constant",
                &value,
                format!("unsupported constant value `{}`", value),
            ));
        };

        Ok(Self(l))
    }
}

impl CodeGen for Constant {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        _ctx: Self::Context,
        _options: Self::Options,
        _symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        match self.0 {
            Some(c) if is_nonfinite_literal(&c) => {
                // `1e1000` → f64::INFINITY etc.: the f64 EXPRESSION, not a
                // literal (safe then, is_finite true for typical constants).
                Ok(nonfinite_expression(&c).expect(
                    "matched a non-finite sentinel, so the expression exists"
                ))
            }
            Some(c) if is_complex_literal(&c) => {
                let (re, im) = complex_parts(&c).expect("complex literal parts");
                let re_tok: TokenStream = f64_token(re).parse().map_err(
                    |e| format!("cannot render complex real `{}` as Rust tokens: {}", re, e),
                )?;
                let im_tok: TokenStream = f64_token(im).parse().map_err(
                    |e| format!("cannot render complex imag `{}` as Rust tokens: {}", im, e),
                )?;
                Ok(quote!(Complex::new(#re_tok, #im_tok)))
            }
            Some(c) if is_ellipsis_literal(&c) => Err(
                "`...` (Ellipsis) as a VALUE is not supported by rython; it is only \
                 accepted as a bare statement (a no-op, like `pass`) — Protocol \
                 stubs such as `def f(...) -> None: ...` are the common use"
                    .to_string()
                    .into(),
            ),
            Some(c) => {
                let v: TokenStream = c
                    .to_string()
                    .parse()
                    .map_err(|e| format!("cannot render constant `{}` as Rust tokens: {}", c, e))?;
                // A bytes literal is an OWNED value in Python; the typed
                // paths already declare it Vec<u8> (simple_expr_type), so
                // the rendering must agree — a bare `b".."` is a
                // `&[u8; N]` that fails every Vec-typed position
                // (`return Ok(b"")` — urllib3's emscripten response).
                // AsRef<[u8]> receivers accept the Vec the same.
                if matches!(&c, Literal::ByteString(_)) {
                    return Ok(quote!(#v.to_vec()));
                }
                Ok(quote!(#v))
            }
            None => Ok(quote!(None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;
    //use super::*;
    use crate::{symbols::SymbolTableScopes, CodeGen};
    use tracing::debug;

    #[test]
    fn parse_string() {
        let s = crate::parse("'I ate a bug'", "test.py").unwrap();
        let ast = s
            .to_rust(
                crate::CodeGenContext::Module("test".to_string()),
                crate::PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();
        debug!("ast: {}", ast.to_string());

        assert_eq!("use stdpython :: * ; \"I ate a bug\"", ast.to_string());
    }

    #[test]
    fn parse_bytes() {
        let s = crate::parse("b'I ate a bug'", "test.py").unwrap();
        let ast = s
            .to_rust(
                crate::CodeGenContext::Module("test".to_string()),
                crate::PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();

        // A bytes literal renders OWNED (issue #137 round 21): the typed
        // paths declare it Vec<u8>, and the rendering agrees.
        assert_eq!(
            "use stdpython :: * ; b\"I ate a bug\" . to_vec ()",
            ast.to_string()
        );
    }

    #[test]
    fn parse_number_int() {
        let s = crate::parse("871234234", "test.py").unwrap();
        let ast = s
            .to_rust(
                crate::CodeGenContext::Module("test".to_string()),
                crate::PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();

        assert_eq!("use stdpython :: * ; 871234234", ast.to_string());
    }

    #[test]
    fn parse_number_neg_int() {
        let s = crate::parse("-871234234", "test.py").unwrap();
        let ast = s
            .to_rust(
                crate::CodeGenContext::Module("test".to_string()),
                crate::PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();

        assert_eq!("use stdpython :: * ; - 871234234", ast.to_string());
    }

    #[test]
    fn parse_number_float() {
        let s = crate::parse("87123.4234", "test.py").unwrap();
        let ast = s
            .to_rust(
                crate::CodeGenContext::Module("test".to_string()),
                crate::PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();

        assert_eq!("use stdpython :: * ; 87123.4234", ast.to_string());
    }

    #[test]
    fn parse_bool() {
        let s = crate::parse("True", "test.py").unwrap();
        let ast = s
            .to_rust(
                crate::CodeGenContext::Module("test".to_string()),
                crate::PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();

        assert_eq!("use stdpython :: * ; true", ast.to_string());
    }

    #[test]
    fn parse_none() {
        let s = crate::parse("None", "test.py").unwrap();
        let ast = s
            .to_rust(
                crate::CodeGenContext::Module("test".to_string()),
                crate::PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();

        assert_eq!("use stdpython :: * ; None", ast.to_string());
    }
}
