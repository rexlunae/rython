use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes,
    extract_list,
};

/// Joined string (f-string, e.g., f"Hello {name}")
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JoinedStr {
    /// The values that make up the f-string (mix of strings and expressions)
    pub values: Vec<ExprType>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// Formatted value within an f-string (e.g., the {name} part)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FormattedValue {
    /// The expression to be formatted
    pub value: Box<ExprType>,
    /// Conversion flag (None, 's', 'r', 'a') - represented as optional integer
    pub conversion: Option<i32>,
    /// Format specifier (optional)
    pub format_spec: Option<Box<ExprType>>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for JoinedStr {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract values
        let values: Vec<ExprType> = extract_list(&ob, "values", "joined string values")?;
        
        Ok(JoinedStr {
            values,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FormattedValue {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract value
        let value: ExprType = ob.getattr("value")?.extract()?;
        
        // Extract conversion (optional)
        let conversion: Option<i32> = if let Ok(conv_attr) = ob.getattr("conversion") {
            let conv_val: i32 = conv_attr.extract()?;
            if conv_val == -1 {
                None // -1 means no conversion
            } else {
                Some(conv_val)
            }
        } else {
            None
        };
        
        // Extract format_spec (optional)
        let format_spec: Option<Box<ExprType>> = if let Ok(spec_attr) = ob.getattr("format_spec") {
            if spec_attr.is_none() {
                None
            } else {
                Some(Box::new(spec_attr.extract()?))
            }
        } else {
            None
        };
        
        Ok(FormattedValue {
            value: Box::new(value),
            conversion,
            format_spec,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for JoinedStr {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl Node for FormattedValue {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for JoinedStr {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        self.values.into_iter().fold(symbols, |acc, val| val.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Build a single format! call: literal parts go into the format
        // string (with `{`/`}` escaped), interpolated parts each get a
        // placeholder and become an argument.
        let mut fmt = String::new();
        let mut args: Vec<TokenStream> = Vec::new();

        for val in self.values {
            match val {
                ExprType::Constant(c) => {
                    fmt.push_str(&escape_format_braces(&constant_text(&c)));
                }
                ExprType::FormattedValue(fv) => {
                    let placeholder = fv.rust_placeholder()?;
                    let expr = (*fv.value).to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    fmt.push_str(&placeholder);
                    args.push(expr);
                }
                other => {
                    let expr = other.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    fmt.push_str("{}");
                    args.push(expr);
                }
            }
        }

        if fmt.is_empty() && args.is_empty() {
            Ok(quote! { String::new() })
        } else {
            Ok(quote! { format!(#fmt #(, #args)*) })
        }
    }
}

/// Recover the unescaped text of a string constant (its stored form is a
/// quoted, escaped Rust literal).
fn constant_text(c: &crate::Constant) -> String {
    match &c.0 {
        Some(litrs::Literal::String(s)) => s.value().to_string(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Escape literal braces so they survive inside a format! string.
fn escape_format_braces(s: &str) -> String {
    s.replace('{', "{{").replace('}', "}}")
}

impl FormattedValue {
    /// The Rust format placeholder for this interpolation: `!r`/`!a`
    /// conversions map to `{:?}`, and format specs translate through the
    /// shared Python-spec translator. Specs Rust cannot reproduce exactly
    /// (or that interpolate other values) are LOUD conversion errors —
    /// falling back to `{}` would silently change the output.
    fn rust_placeholder(&self) -> Result<String, Box<dyn std::error::Error>> {
        // Python conversion codes are the ASCII values of 's', 'r', 'a'.
        if matches!(self.conversion, Some(114) | Some(97)) {
            return Ok("{:?}".to_string());
        }

        match &self.format_spec {
            None => Ok("{}".to_string()),
            Some(spec) => match static_spec_text(spec) {
                None => Err(
                    "f-string format specs that interpolate other values (e.g. \
                     f\"{x:{width}}\") are not supported yet"
                        .to_string()
                        .into(),
                ),
                Some(spec_text) => {
                    let rust_spec = crate::pyformat::translate_format_spec(spec_text.trim())
                        .map_err(|e| format!("f-string: {}", e))?;
                    if rust_spec.is_empty() {
                        Ok("{}".to_string())
                    } else {
                        Ok(format!("{{:{}}}", rust_spec))
                    }
                }
            },
        }
    }
}

/// If a format spec is a purely constant expression, return its text.
fn static_spec_text(spec: &ExprType) -> Option<String> {
    match spec {
        ExprType::Constant(c) => Some(constant_text(c)),
        ExprType::JoinedStr(js) => {
            let mut out = String::new();
            for part in &js.values {
                if let ExprType::Constant(c) = part {
                    out.push_str(&constant_text(c));
                } else {
                    return None;
                }
            }
            Some(out)
        }
        _ => None,
    }
}

impl CodeGen for FormattedValue {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let symbols = (*self.value).find_symbols(symbols);
        if let Some(format_spec) = self.format_spec {
            (*format_spec).find_symbols(symbols)
        } else {
            symbols
        }
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let placeholder = self.rust_placeholder()?;
        let value_tokens = (*self.value).to_rust(ctx, options, symbols)?;
        Ok(quote! {
            format!(#placeholder, #value_tokens)
        })
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_fstring, "f'Hello {name}'", "test.py");
}