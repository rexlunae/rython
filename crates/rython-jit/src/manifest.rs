//! What a compiled library says about itself: its module, whether it has a
//! program entry, and each public module-level function — with the shim
//! symbol and M1 signature when it has one, or the reason it has none.
//!
//! The converter's cdylib mode renders a [`Manifest`] into the library
//! (`ry_manifest`), and the host parses it back on load. Both sides use this
//! one type, so the format cannot drift; a manifest that does not parse is
//! a corrupt library, rebuilt rather than trusted.

use std::fmt::Write as _;

/// The Python types M1 marshals across the boundary (docs/jit.md, "The
/// boundary"): `int` as i64, `float` as f64, `bool`, `None` and `str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarshalType {
    None,
    Bool,
    Int,
    Float,
    Str,
}

impl MarshalType {
    /// The one parse from a type's Python spelling — an annotation's name
    /// (`int`) or a manifest field. Anything else is not marshallable in M1.
    pub fn from_name(name: &str) -> Option<MarshalType> {
        match name {
            "None" => Some(MarshalType::None),
            "bool" => Some(MarshalType::Bool),
            "int" => Some(MarshalType::Int),
            "float" => Some(MarshalType::Float),
            "str" => Some(MarshalType::Str),
            _ => None,
        }
    }

    /// The annotation spelling (`None`, `bool`, `int`, `float`, `str`).
    pub fn name(self) -> &'static str {
        match self {
            MarshalType::None => "None",
            MarshalType::Bool => "bool",
            MarshalType::Int => "int",
            MarshalType::Float => "float",
            MarshalType::Str => "str",
        }
    }

    /// `type(value).__name__` for a value of this type (`NoneType` for
    /// `None`).
    pub fn type_name(self) -> &'static str {
        match self {
            MarshalType::None => "NoneType",
            other => other.name(),
        }
    }

    /// The boundary tag a value of this type carries.
    pub fn tag(self) -> u32 {
        match self {
            MarshalType::None => crate::abi::RY_TAG_NONE,
            MarshalType::Bool => crate::abi::RY_TAG_BOOL,
            MarshalType::Int => crate::abi::RY_TAG_INT,
            MarshalType::Float => crate::abi::RY_TAG_FLOAT,
            MarshalType::Str => crate::abi::RY_TAG_STR,
        }
    }

    /// The tag's type, for decoding a value the library wrote.
    pub fn from_tag(tag: u32) -> Option<MarshalType> {
        match tag {
            crate::abi::RY_TAG_NONE => Some(MarshalType::None),
            crate::abi::RY_TAG_BOOL => Some(MarshalType::Bool),
            crate::abi::RY_TAG_INT => Some(MarshalType::Int),
            crate::abi::RY_TAG_FLOAT => Some(MarshalType::Float),
            crate::abi::RY_TAG_STR => Some(MarshalType::Str),
            _ => None,
        }
    }
}

/// One parameter of a shimmed function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: MarshalType,
}

/// A public module-level function with a shim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSig {
    /// The Python name.
    pub name: String,
    /// The exported shim symbol (`ryfn_<n>`).
    pub symbol: String,
    pub params: Vec<Param>,
    /// The return type the converter resolved for the generated function.
    pub returns: MarshalType,
}

impl FunctionSig {
    /// `add(a: int, b: int) -> int`, for messages.
    pub fn signature(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, p.ty.name()))
            .collect();
        format!("{}({}) -> {}", self.name, params.join(", "), self.returns.name())
    }
}

/// Why a public module-level function has no shim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnshimmedKind {
    /// A parameter or the result has a type M1 cannot marshal
    /// (`JitError::Marshal` when called).
    Marshal,
    /// The function's shape is outside what M1 calls: defaults, `*args`,
    /// keyword-only parameters, decorators, `async def`, a generator
    /// (`JitError::Unsupported` when called).
    Unsupported,
}

impl UnshimmedKind {
    fn name(self) -> &'static str {
        match self {
            UnshimmedKind::Marshal => "marshal",
            UnshimmedKind::Unsupported => "unsupported",
        }
    }

    fn from_name(name: &str) -> Option<UnshimmedKind> {
        match name {
            "marshal" => Some(UnshimmedKind::Marshal),
            "unsupported" => Some(UnshimmedKind::Unsupported),
            _ => None,
        }
    }
}

/// A public module-level function without a shim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unshimmed {
    pub name: String,
    pub kind: UnshimmedKind,
    /// What keeps it from crossing, naming the type or the construct.
    pub reason: String,
}

/// Whether the library can run its module as a program (`ry_entry`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// It can: the module has an entry point the binary would run.
    Program,
    /// It cannot, and the converter said why (no entry point, an async
    /// one, ...).
    Refused(String),
}

/// Whether function calls can import the module faithfully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Calls {
    Allowed,
    /// They cannot (an import under CPython would run something the
    /// library treats as its program entry), and the converter said why.
    Refused(String),
}

/// A compiled library's description of itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The Python module name.
    pub module: String,
    pub entry: Entry,
    pub calls: Calls,
    /// The shimmed functions.
    pub functions: Vec<FunctionSig>,
    /// Public module-level functions without a shim, and why.
    pub unshimmed: Vec<Unshimmed>,
    /// Classes defined at the module's top level (CPython names an
    /// uncaught exception of one module-qualified when it escapes an
    /// imported module).
    pub classes: Vec<String>,
}

const HEADER: &str = "rython-jit manifest 1";

impl Manifest {
    /// The text form, one record per line. Reasons are flattened to one
    /// line; names never contain whitespace.
    pub fn render(&self) -> String {
        let one_line = |text: &str| text.replace(['\n', '\r'], " ");
        let mut out = String::new();
        let _ = writeln!(out, "{}", HEADER);
        let _ = writeln!(out, "module {}", self.module);
        match &self.entry {
            Entry::Program => {
                let _ = writeln!(out, "entry program");
            }
            Entry::Refused(why) => {
                let _ = writeln!(out, "entry-refused {}", one_line(why));
            }
        }
        match &self.calls {
            Calls::Allowed => {
                let _ = writeln!(out, "calls allowed");
            }
            Calls::Refused(why) => {
                let _ = writeln!(out, "calls-refused {}", one_line(why));
            }
        }
        for f in &self.functions {
            let _ = write!(out, "fn {} {} {}", f.name, f.symbol, f.returns.name());
            for p in &f.params {
                let _ = write!(out, " {}:{}", p.name, p.ty.name());
            }
            out.push('\n');
        }
        for u in &self.unshimmed {
            let _ = writeln!(out, "unshimmed {} {} {}", u.name, u.kind.name(), one_line(&u.reason));
        }
        for class in &self.classes {
            let _ = writeln!(out, "class {}", class);
        }
        out
    }

    /// Parse [`Manifest::render`]'s text. Anything unexpected is an error:
    /// a manifest that does not parse is not trusted.
    pub fn parse(text: &str) -> Result<Manifest, String> {
        let mut lines = text.lines();
        if lines.next() != Some(HEADER) {
            return Err(format!("the manifest does not start with `{}`", HEADER));
        }
        let mut module = None;
        let mut entry = None;
        let mut calls = None;
        let mut functions = Vec::new();
        let mut unshimmed = Vec::new();
        let mut classes = Vec::new();
        for line in lines {
            let (kind, rest) = line.split_once(' ').unwrap_or((line, ""));
            match kind {
                "module" => module = Some(word(rest, "module")?),
                "entry" if rest == "program" => entry = Some(Entry::Program),
                "entry-refused" => entry = Some(Entry::Refused(rest.to_string())),
                "calls" if rest == "allowed" => calls = Some(Calls::Allowed),
                "calls-refused" => calls = Some(Calls::Refused(rest.to_string())),
                "fn" => {
                    let mut fields = rest.split(' ');
                    let name = word(fields.next().unwrap_or(""), "function name")?;
                    let symbol = word(fields.next().unwrap_or(""), "shim symbol")?;
                    let returns = marshal_type(fields.next().unwrap_or(""))?;
                    let mut params = Vec::new();
                    for field in fields {
                        let (name, ty) = field
                            .split_once(':')
                            .ok_or_else(|| format!("malformed parameter `{}`", field))?;
                        params.push(Param {
                            name: word(name, "parameter name")?,
                            ty: marshal_type(ty)?,
                        });
                    }
                    functions.push(FunctionSig {
                        name,
                        symbol,
                        params,
                        returns,
                    });
                }
                "unshimmed" => {
                    let mut fields = rest.splitn(3, ' ');
                    let name = word(fields.next().unwrap_or(""), "function name")?;
                    let kind = fields.next().unwrap_or("");
                    let kind = UnshimmedKind::from_name(kind)
                        .ok_or_else(|| format!("unknown unshimmed kind `{}`", kind))?;
                    unshimmed.push(Unshimmed {
                        name,
                        kind,
                        reason: fields.next().unwrap_or("").to_string(),
                    });
                }
                "class" => classes.push(word(rest, "class name")?),
                _ => return Err(format!("unexpected manifest line `{}`", line)),
            }
        }
        Ok(Manifest {
            module: module.ok_or("the manifest names no module")?,
            entry: entry.ok_or("the manifest has no entry record")?,
            calls: calls.ok_or("the manifest has no calls record")?,
            functions,
            unshimmed,
            classes,
        })
    }
}

fn word(text: &str, what: &str) -> Result<String, String> {
    if text.is_empty() || text.contains(char::is_whitespace) {
        return Err(format!("malformed {} `{}`", what, text));
    }
    Ok(text.to_string())
}

fn marshal_type(text: &str) -> Result<MarshalType, String> {
    MarshalType::from_name(text).ok_or_else(|| format!("unknown marshal type `{}`", text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manifest_round_trips_through_its_text() {
        let manifest = Manifest {
            module: "prog".to_string(),
            entry: Entry::Refused("no entry\npoint".to_string()),
            calls: Calls::Allowed,
            functions: vec![FunctionSig {
                name: "add".to_string(),
                symbol: "ryfn_0".to_string(),
                params: vec![
                    Param { name: "a".to_string(), ty: MarshalType::Int },
                    Param { name: "b".to_string(), ty: MarshalType::Str },
                ],
                returns: MarshalType::None,
            }],
            unshimmed: vec![Unshimmed {
                name: "f".to_string(),
                kind: UnshimmedKind::Marshal,
                reason: "parameter `xs` is a list".to_string(),
            }],
            classes: vec!["MyError".to_string()],
        };
        let parsed = Manifest::parse(&manifest.render()).unwrap();
        // Reasons are flattened to one line; everything else survives.
        assert_eq!(parsed.entry, Entry::Refused("no entry point".to_string()));
        assert_eq!(parsed.functions, manifest.functions);
        assert_eq!(parsed.unshimmed, manifest.unshimmed);
        assert_eq!(parsed.classes, manifest.classes);
        assert_eq!(parsed.functions[0].signature(), "add(a: int, b: str) -> None");
        assert!(Manifest::parse("garbage").is_err());
        assert!(Manifest::parse(&format!("{}\nmodule prog\nbogus x\n", HEADER)).is_err());
    }

    #[test]
    fn marshal_types_parse_from_their_python_spelling_only() {
        for ty in [
            MarshalType::None,
            MarshalType::Bool,
            MarshalType::Int,
            MarshalType::Float,
            MarshalType::Str,
        ] {
            assert_eq!(MarshalType::from_name(ty.name()), Some(ty));
            assert_eq!(MarshalType::from_tag(ty.tag()), Some(ty));
        }
        assert_eq!(MarshalType::from_name("list"), None);
        assert_eq!(MarshalType::None.type_name(), "NoneType");
    }
}
