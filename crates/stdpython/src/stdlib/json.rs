//! Python json module implementation
//! 
//! This module provides JSON encoder and decoder functionality.
//! Implementation matches Python's json module API.

use crate::PyException;
use crate::python_function;
use std::collections::HashMap;
use std::fmt;

/// JSONValue - represents any JSON value
#[derive(Debug, Clone, PartialEq)]
pub enum JSONValue {
    Null,
    Bool(bool),
    /// Integer number. Kept separate from `Float` so `json.dumps(1)` renders
    /// as `1` while `json.dumps(1.0)` renders as `1.0`, like Python.
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<JSONValue>),
    Object(HashMap<String, JSONValue>),
}

impl JSONValue {
    /// Check if this is null
    pub fn is_null(&self) -> bool {
        matches!(self, JSONValue::Null)
    }
    
    /// Check if this is a boolean
    pub fn is_bool(&self) -> bool {
        matches!(self, JSONValue::Bool(_))
    }
    
    /// Check if this is a number
    pub fn is_number(&self) -> bool {
        matches!(self, JSONValue::Int(_) | JSONValue::Float(_))
    }
    
    /// Check if this is a string
    pub fn is_string(&self) -> bool {
        matches!(self, JSONValue::String(_))
    }
    
    /// Check if this is an array
    pub fn is_array(&self) -> bool {
        matches!(self, JSONValue::Array(_))
    }
    
    /// Check if this is an object
    pub fn is_object(&self) -> bool {
        matches!(self, JSONValue::Object(_))
    }
    
    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        if let JSONValue::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }
    
    /// Get as number
    pub fn as_number(&self) -> Option<f64> {
        match self {
            JSONValue::Int(n) => Some(*n as f64),
            JSONValue::Float(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        if let JSONValue::Int(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    
    /// Get as string
    pub fn as_string(&self) -> Option<&String> {
        if let JSONValue::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
    
    /// Get as array
    pub fn as_array(&self) -> Option<&Vec<JSONValue>> {
        if let JSONValue::Array(arr) = self {
            Some(arr)
        } else {
            None
        }
    }
    
    /// Get as object
    pub fn as_object(&self) -> Option<&HashMap<String, JSONValue>> {
        if let JSONValue::Object(obj) = self {
            Some(obj)
        } else {
            None
        }
    }
}

impl fmt::Display for JSONValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            JSONValue::Null => write!(f, "null"),
            JSONValue::Bool(b) => write!(f, "{}", if *b { "true" } else { "false" }),
            JSONValue::Int(n) => write!(f, "{}", n),
            JSONValue::Float(n) => write!(f, "{}", format_json_float(*n)),
            JSONValue::String(s) => write!(f, "\"{}\"", escape_json_string(s, true)),
            JSONValue::Array(arr) => {
                write!(f, "[")?;
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            },
            JSONValue::Object(obj) => {
                write!(f, "{{")?;
                let mut first = true;
                for (key, value) in obj {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(f, "\"{}\": {}", escape_json_string(key, true), value)?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// JSONEncoder - encodes Python objects to JSON
#[derive(Debug)]
pub struct JSONEncoder {
    pub skipkeys: bool,
    pub ensure_ascii: bool,
    pub check_circular: bool,
    pub allow_nan: bool,
    pub sort_keys: bool,
    pub indent: Option<usize>,
    pub separators: Option<(String, String)>,
}

impl JSONEncoder {
    /// Create new encoder with default settings
    pub fn new() -> Self {
        Self {
            skipkeys: false,
            ensure_ascii: true,
            check_circular: true,
            allow_nan: true,
            sort_keys: false,
            indent: None,
            separators: None,
        }
    }
    
    /// Create encoder with custom settings
    pub fn with_options(
        skipkeys: bool,
        ensure_ascii: bool,
        check_circular: bool,
        allow_nan: bool,
        sort_keys: bool,
        indent: Option<usize>,
        separators: Option<(String, String)>,
    ) -> Self {
        Self {
            skipkeys,
            ensure_ascii,
            check_circular,
            allow_nan,
            sort_keys,
            indent,
            separators,
        }
    }
    
    /// Encode JSONValue to string
    pub fn encode(&self, obj: &JSONValue) -> String {
        if let Some(indent) = self.indent {
            self.encode_pretty(obj, 0, indent)
        } else {
            self.encode_compact(obj)
        }
    }
    
    fn encode_compact(&self, obj: &JSONValue) -> String {
        // Python's json.dumps default separators are (", ", ": ").
        let (item_sep, key_sep) = self.separators.as_ref()
            .map(|(is, ks)| (is.as_str(), ks.as_str()))
            .unwrap_or((", ", ": "));

        match obj {
            JSONValue::Null => "null".to_string(),
            JSONValue::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
            JSONValue::Int(n) => format!("{}", n),
            JSONValue::Float(n) => {
                if !self.allow_nan && (n.is_nan() || n.is_infinite()) {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "ValueError",
                            "Out of range float values are not JSON compliant"
                        )
                    );
                }
                format_json_float(*n)
            },
            JSONValue::String(s) => format!("\"{}\"", escape_json_string(s, self.ensure_ascii)),
            JSONValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|item| self.encode_compact(item)).collect();
                format!("[{}]", items.join(item_sep))
            },
            JSONValue::Object(obj) => {
                let mut pairs: Vec<(String, String)> = obj.iter()
                    .map(|(k, v)| (k.clone(), self.encode_compact(v)))
                    .collect();
                    
                if self.sort_keys {
                    pairs.sort_by(|a, b| a.0.cmp(&b.0));
                }
                
                let items: Vec<String> = pairs.into_iter()
                    .map(|(k, v)| {
                        format!("\"{}\"{}{}", escape_json_string(&k, self.ensure_ascii), key_sep, v)
                    })
                    .collect();
                format!("{{{}}}", items.join(item_sep))
            }
        }
    }
    
    fn encode_pretty(&self, obj: &JSONValue, depth: usize, indent_size: usize) -> String {
        let indent = " ".repeat(depth * indent_size);
        let next_indent = " ".repeat((depth + 1) * indent_size);
        
        match obj {
            JSONValue::Null => "null".to_string(),
            JSONValue::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
            JSONValue::Int(n) => format!("{}", n),
            JSONValue::Float(n) => {
                if !self.allow_nan && (n.is_nan() || n.is_infinite()) {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "ValueError",
                            "Out of range float values are not JSON compliant"
                        )
                    );
                }
                format_json_float(*n)
            },
            JSONValue::String(s) => format!("\"{}\"", escape_json_string(s, self.ensure_ascii)),
            JSONValue::Array(arr) => {
                if arr.is_empty() {
                    return "[]".to_string();
                }
                
                let mut result = "[\n".to_string();
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        result.push_str(",\n");
                    }
                    result.push_str(&next_indent);
                    result.push_str(&self.encode_pretty(item, depth + 1, indent_size));
                }
                result.push('\n');
                result.push_str(&indent);
                result.push(']');
                result
            },
            JSONValue::Object(obj) => {
                if obj.is_empty() {
                    return "{}".to_string();
                }
                
                let mut pairs: Vec<(String, String)> = obj.iter()
                    .map(|(k, v)| (k.clone(), self.encode_pretty(v, depth + 1, indent_size)))
                    .collect();
                    
                if self.sort_keys {
                    pairs.sort_by(|a, b| a.0.cmp(&b.0));
                }
                
                let mut result = "{\n".to_string();
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        result.push_str(",\n");
                    }
                    result.push_str(&next_indent);
                    result.push_str(&format!(
                        "\"{}\": {}",
                        escape_json_string(key, self.ensure_ascii),
                        value
                    ));
                }
                result.push('\n');
                result.push_str(&indent);
                result.push('}');
                result
            }
        }
    }
}

impl Default for JSONEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// JSONDecoder - decodes JSON to Python objects
#[derive(Debug)]
pub struct JSONDecoder {
    pub strict: bool,
}

impl JSONDecoder {
    /// Create new decoder
    pub fn new() -> Self {
        Self { strict: true }
    }
    
    /// Create decoder with custom settings
    pub fn with_strict(strict: bool) -> Self {
        Self { strict }
    }
    
    /// Decode JSON string to JSONValue
    pub fn decode<S: AsRef<str>>(&self, s: S) -> Result<JSONValue, PyException> {
        let s = s.as_ref().trim();
        if s.is_empty() {
            return Err(crate::value_error("Empty JSON string"));
        }
        
        let mut parser = JSONParser::new(s);
        let value = parser.parse_value()?;
        parser.skip_whitespace();
        if parser.pos < parser.input.len() {
            return Err(crate::value_error(format!(
                "Extra data: char {}",
                parser.pos
            )));
        }
        Ok(value)
    }
}

impl Default for JSONDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON Parser
struct JSONParser {
    input: Vec<char>,
    pos: usize,
}

impl JSONParser {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }
    
    fn parse_value(&mut self) -> Result<JSONValue, PyException> {
        self.skip_whitespace();
        
        if self.pos >= self.input.len() {
            return Err(crate::value_error("Unexpected end of JSON input"));
        }
        
        match self.input[self.pos] {
            '"' => self.parse_string(),
            '[' => self.parse_array(),
            '{' => self.parse_object(),
            't' | 'f' => self.parse_boolean(),
            'n' => self.parse_null(),
            c if c.is_ascii_digit() || c == '-' => self.parse_number(),
            _ => Err(crate::value_error(format!("Unexpected character: {}", self.input[self.pos]))),
        }
    }
    
    fn parse_string(&mut self) -> Result<JSONValue, PyException> {
        if self.input[self.pos] != '"' {
            return Err(crate::value_error("Expected '\"'"));
        }
        
        self.pos += 1; // Skip opening quote
        let mut result = String::new();
        
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            
            if ch == '"' {
                self.pos += 1; // Skip closing quote
                return Ok(JSONValue::String(result));
            }
            
            if ch == '\\' {
                self.pos += 1;
                if self.pos >= self.input.len() {
                    return Err(crate::value_error("Unterminated string"));
                }
                
                match self.input[self.pos] {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    '/' => result.push('/'),
                    'b' => result.push('\u{0008}'),
                    'f' => result.push('\u{000C}'),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    'u' => {
                        // Unicode escape
                        if self.pos + 4 >= self.input.len() {
                            return Err(crate::value_error("Invalid unicode escape"));
                        }
                        let hex_chars: String = self.input[self.pos+1..=self.pos+4].iter().collect();
                        let Ok(code_point) = u32::from_str_radix(&hex_chars, 16) else {
                            return Err(crate::value_error("Invalid unicode escape"));
                        };
                        self.pos += 4;
                        if (0xD800..0xDC00).contains(&code_point) {
                            // High surrogate: must be followed by \uDC00-\uDFFF,
                            // together encoding one astral code point.
                            if self.pos + 6 < self.input.len()
                                && self.input[self.pos + 1] == '\\'
                                && self.input[self.pos + 2] == 'u'
                            {
                                let lo_hex: String =
                                    self.input[self.pos + 3..=self.pos + 6].iter().collect();
                                if let Ok(lo) = u32::from_str_radix(&lo_hex, 16) {
                                    if (0xDC00..0xE000).contains(&lo) {
                                        let combined = 0x10000
                                            + ((code_point - 0xD800) << 10)
                                            + (lo - 0xDC00);
                                        if let Some(ch) = char::from_u32(combined) {
                                            result.push(ch);
                                            self.pos += 6;
                                        } else {
                                            return Err(crate::value_error(
                                                "Invalid unicode code point",
                                            ));
                                        }
                                    } else {
                                        return Err(crate::value_error("Unpaired surrogate"));
                                    }
                                } else {
                                    return Err(crate::value_error("Invalid unicode escape"));
                                }
                            } else {
                                return Err(crate::value_error("Unpaired surrogate"));
                            }
                        } else if let Some(ch) = char::from_u32(code_point) {
                            result.push(ch);
                        } else {
                            return Err(crate::value_error("Invalid unicode code point"));
                        }
                    },
                    _ => return Err(crate::value_error("Invalid escape sequence")),
                }
            } else {
                result.push(ch);
            }
            
            self.pos += 1;
        }
        
        Err(crate::value_error("Unterminated string"))
    }
    
    fn parse_array(&mut self) -> Result<JSONValue, PyException> {
        if self.input[self.pos] != '[' {
            return Err(crate::value_error("Expected '['"));
        }
        
        self.pos += 1; // Skip '['
        self.skip_whitespace();
        
        let mut result = Vec::new();
        
        if self.pos < self.input.len() && self.input[self.pos] == ']' {
            self.pos += 1;
            return Ok(JSONValue::Array(result));
        }
        
        loop {
            result.push(self.parse_value()?);
            self.skip_whitespace();
            
            if self.pos >= self.input.len() {
                return Err(crate::value_error("Unterminated array"));
            }
            
            match self.input[self.pos] {
                ',' => {
                    self.pos += 1;
                    self.skip_whitespace();
                },
                ']' => {
                    self.pos += 1;
                    break;
                },
                _ => return Err(crate::value_error("Expected ',' or ']'")),
            }
        }
        
        Ok(JSONValue::Array(result))
    }
    
    fn parse_object(&mut self) -> Result<JSONValue, PyException> {
        if self.input[self.pos] != '{' {
            return Err(crate::value_error("Expected '{'"));
        }
        
        self.pos += 1; // Skip '{'
        self.skip_whitespace();
        
        let mut result = HashMap::new();
        
        if self.pos < self.input.len() && self.input[self.pos] == '}' {
            self.pos += 1;
            return Ok(JSONValue::Object(result));
        }
        
        loop {
            // Parse key
            if self.pos >= self.input.len() || self.input[self.pos] != '"' {
                return Err(crate::value_error("Expected string key"));
            }
            
            let key = match self.parse_string()? {
                JSONValue::String(s) => s,
                _ => return Err(crate::value_error("Key must be string")),
            };
            
            self.skip_whitespace();
            
            // Parse colon
            if self.pos >= self.input.len() || self.input[self.pos] != ':' {
                return Err(crate::value_error("Expected ':'"));
            }
            self.pos += 1;
            self.skip_whitespace();
            
            // Parse value
            let value = self.parse_value()?;
            result.insert(key, value);
            
            self.skip_whitespace();
            
            if self.pos >= self.input.len() {
                return Err(crate::value_error("Unterminated object"));
            }
            
            match self.input[self.pos] {
                ',' => {
                    self.pos += 1;
                    self.skip_whitespace();
                },
                '}' => {
                    self.pos += 1;
                    break;
                },
                _ => return Err(crate::value_error("Expected ',' or '}'")),
            }
        }
        
        Ok(JSONValue::Object(result))
    }
    
    fn parse_boolean(&mut self) -> Result<JSONValue, PyException> {
        if self.matches("true") {
            self.pos += 4;
            Ok(JSONValue::Bool(true))
        } else if self.matches("false") {
            self.pos += 5;
            Ok(JSONValue::Bool(false))
        } else {
            Err(crate::value_error("Invalid boolean value"))
        }
    }
    
    fn parse_null(&mut self) -> Result<JSONValue, PyException> {
        if self.matches("null") {
            self.pos += 4;
            Ok(JSONValue::Null)
        } else {
            Err(crate::value_error("Invalid null value"))
        }
    }
    
    fn parse_number(&mut self) -> Result<JSONValue, PyException> {
        let start = self.pos;
        
        // Handle negative sign
        if self.pos < self.input.len() && self.input[self.pos] == '-' {
            self.pos += 1;
        }
        
        // Parse integer part
        if self.pos >= self.input.len() || !self.input[self.pos].is_ascii_digit() {
            return Err(crate::value_error("Invalid number"));
        }
        
        if self.input[self.pos] == '0' {
            self.pos += 1;
        } else {
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        
        // Parse fractional part
        if self.pos < self.input.len() && self.input[self.pos] == '.' {
            self.pos += 1;
            if self.pos >= self.input.len() || !self.input[self.pos].is_ascii_digit() {
                return Err(crate::value_error("Invalid number"));
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        
        // Parse exponent
        if self.pos < self.input.len() && (self.input[self.pos] == 'e' || self.input[self.pos] == 'E') {
            self.pos += 1;
            if self.pos < self.input.len() && (self.input[self.pos] == '+' || self.input[self.pos] == '-') {
                self.pos += 1;
            }
            if self.pos >= self.input.len() || !self.input[self.pos].is_ascii_digit() {
                return Err(crate::value_error("Invalid number"));
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        
        let number_str: String = self.input[start..self.pos].iter().collect();
        // A literal with no fraction or exponent is an int, like Python's
        // json.loads("1") -> 1 vs json.loads("1.0") -> 1.0.
        if !number_str.contains(['.', 'e', 'E']) {
            if let Ok(n) = number_str.parse::<i64>() {
                return Ok(JSONValue::Int(n));
            }
        }
        match number_str.parse::<f64>() {
            Ok(n) => Ok(JSONValue::Float(n)),
            Err(_) => Err(crate::value_error("Invalid number")),
        }
    }
    
    fn matches(&self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        if self.pos + chars.len() > self.input.len() {
            return false;
        }
        
        for (i, ch) in chars.iter().enumerate() {
            if self.input[self.pos + i] != *ch {
                return false;
            }
        }
        true
    }
    
    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }
}

// Module-level functions (Python API)
python_function! {
    /// json.loads - deserialize JSON string
    pub fn loads<S>(s: S) -> Result<JSONValue, PyException>
    where [S: AsRef<str>]
    [signature: (s)]
    [concrete_types: (String) -> Result<JSONValue, crate::PyException>]
    {
        JSONDecoder::new().decode(s)
    }
}

python_function! {
    /// json.dumps - serialize object to JSON string
    pub fn dumps(obj: &JSONValue, indent: Option<usize>) -> String
    [signature: (obj, indent=None)]
    [concrete_types: (&JSONValue, Option<usize>) -> String]
    {
        let encoder = JSONEncoder::with_options(false, true, true, true, false, indent, None);
        encoder.encode(obj)
    }
}

#[cfg(feature = "std")]
python_function! {
    /// json.dump - serialize object to JSON and write to file
    pub fn dump<P>(obj: &JSONValue, fp: P) -> Result<(), PyException>
    where [P: AsRef<std::path::Path>]
    [signature: (obj, fp)]
    [concrete_types: (&JSONValue, String) -> Result<(), crate::PyException>]
    {
        let json_str = dumps(obj, None);
        std::fs::write(fp, json_str).map_err(|e| crate::runtime_error(format!("Failed to write JSON: {}", e)))
    }
}

#[cfg(feature = "std")]
python_function! {
    /// json.load - deserialize JSON from file
    pub fn load<P>(fp: P) -> Result<JSONValue, PyException>
    where [P: AsRef<std::path::Path>]
    [signature: (fp)]
    [concrete_types: (String) -> Result<JSONValue, crate::PyException>]
    {
        let content = std::fs::read_to_string(fp).map_err(|e| crate::runtime_error(format!("Failed to read JSON: {}", e)))?;
        loads(content)
    }
}

// Helper functions

/// Format a float the way Python's json module does: integer-valued floats
/// keep their ".0", and non-finite values use Python's JSON spellings.
fn format_json_float(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else if n.fract() == 0.0 && n.abs() < 1e16 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

fn escape_json_string(s: &str, ensure_ascii: bool) -> String {
    let mut result = String::new();
    for ch in s.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\u{0008}' => result.push_str("\\b"),
            '\u{000C}' => result.push_str("\\f"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => result.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && !c.is_ascii() => {
                let code = c as u32;
                if code > 0xFFFF {
                    // Encode astral characters as a UTF-16 surrogate pair.
                    let v = code - 0x10000;
                    result.push_str(&format!(
                        "\\u{:04x}\\u{:04x}",
                        0xD800 + (v >> 10),
                        0xDC00 + (v & 0x3FF)
                    ));
                } else {
                    result.push_str(&format!("\\u{:04x}", code));
                }
            }
            c => result.push(c),
        }
    }
    result
}