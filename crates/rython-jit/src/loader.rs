//! Loading a compiled library and calling across its boundary.
//!
//! Libraries are never unloaded: a Rust `cdylib` with thread-locals and
//! statics is not safe to `dlclose`, so a loaded library lives for the
//! process. One process-wide registry maps an artifact path to its loaded
//! library, so two handles on one entry share one copy of its state (as
//! the dynamic loader would hand them anyway).

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::abi::*;
use crate::error::JitError;
use crate::manifest::{Calls, Entry, FunctionSig, Manifest, MarshalType, UnshimmedKind};
use crate::value::{Value, article};

/// A loaded library: its manifest and resolved entry points.
pub(crate) struct Library {
    pub(crate) path: PathBuf,
    pub(crate) manifest: Manifest,
    free: RyFree,
    entry: Option<RyEntryShim>,
    /// `ry_set_argv` (same signature as the entry), present whenever the
    /// library has a function or an entry to run.
    set_argv: Option<RyEntryShim>,
    functions: HashMap<String, (FunctionSig, RyFunctionShim)>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, &'static Library>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, &'static Library>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load the library at `path` (once per process), check its boundary
/// version, and resolve every symbol its manifest names. Any failure means
/// the artifact cannot be trusted; the caller rebuilds it.
pub(crate) fn load(path: &Path) -> Result<&'static Library, String> {
    let mut loaded = registry().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(library) = loaded.get(path) {
        return Ok(library);
    }
    // SAFETY: loading runs the library's initializers. The library is one
    // the JIT built into its private cache, keyed on everything that went
    // into it (docs/jit.md, "Risks": compiling and loading source IS
    // executing it).
    let raw = unsafe { libloading::Library::new(path) }.map_err(|e| e.to_string())?;
    // Never unloaded (see the module doc).
    let raw: &'static libloading::Library = Box::leak(Box::new(raw));
    // SAFETY (each `get`): the symbol types are the boundary's own
    // definitions, which the library compiled from the same text.
    let version = unsafe { raw.get::<RyAbiVersion>(b"ry_abi_version\0") }
        .map_err(|e| e.to_string())?;
    let version = unsafe { version() };
    if version != RY_ABI_VERSION {
        return Err(format!(
            "it speaks boundary version {}, not {}",
            version, RY_ABI_VERSION
        ));
    }
    let manifest_text = unsafe { raw.get::<RyManifestText>(b"ry_manifest\0") }
        .map_err(|e| e.to_string())?;
    let manifest = {
        let mut len = 0usize;
        let ptr = unsafe { manifest_text(&mut len) };
        // SAFETY: ry_manifest returns a static buffer of `len` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        let text = std::str::from_utf8(bytes).map_err(|e| format!("its manifest is not UTF-8: {}", e))?;
        Manifest::parse(text).map_err(|e| format!("its manifest is malformed: {}", e))?
    };
    let free = *unsafe { raw.get::<RyFree>(b"ry_free\0") }.map_err(|e| e.to_string())?;
    let entry = match manifest.entry {
        Entry::Program => Some(
            *unsafe { raw.get::<RyEntryShim>(b"ry_entry\0") }.map_err(|e| e.to_string())?,
        ),
        Entry::Refused(_) => None,
    };
    let set_argv = match (&manifest.entry, manifest.functions.is_empty()) {
        (Entry::Refused(_), true) => None,
        _ => Some(*unsafe { raw.get::<RyEntryShim>(b"ry_set_argv\0") }.map_err(|e| e.to_string())?),
    };
    let mut functions = HashMap::new();
    for sig in &manifest.functions {
        let mut symbol = sig.symbol.clone().into_bytes();
        symbol.push(0);
        let shim = *unsafe { raw.get::<RyFunctionShim>(&symbol) }.map_err(|e| e.to_string())?;
        functions.insert(sig.name.clone(), (sig.clone(), shim));
    }
    let library: &'static Library = Box::leak(Box::new(Library {
        path: path.to_path_buf(),
        manifest,
        free,
        entry,
        set_argv,
        functions,
    }));
    loaded.insert(path.to_path_buf(), library);
    Ok(library)
}

/// The host's stdout, flushed before control crosses the boundary: each
/// library has its own `std` and its own stdout buffer, so output stays in
/// program order only if each side flushes before the other writes.
fn flush_host_stdout() {
    let _ = std::io::stdout().flush();
}

impl Library {
    /// Copy a buffer the library handed out and release it.
    fn take_text(&self, ptr: *mut u8, len: usize) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        // SAFETY: the library wrote `len` bytes at `ptr` and keeps them
        // until ry_free.
        let text = String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) }).into_owned();
        unsafe { (self.free)(ptr, len) };
        Some(text)
    }

    fn qualified(&self, name: &str) -> String {
        format!("{}.{}", self.manifest.module, name)
    }

    /// Call `function` with `args`, checked against its manifest signature
    /// before anything runs.
    pub(crate) fn call(&self, function: &str, args: &[Value]) -> Result<Value, JitError> {
        if let Calls::Refused(why) = &self.manifest.calls {
            return Err(JitError::Unsupported(why.clone()));
        }
        let Some((sig, shim)) = self.functions.get(function) else {
            return Err(self.not_callable(function));
        };
        if args.len() != sig.params.len() {
            return Err(JitError::Marshal(format!(
                "`{}` takes {} argument{} ({}), but {} {} passed",
                self.qualified(&sig.name),
                sig.params.len(),
                if sig.params.len() == 1 { "" } else { "s" },
                sig.signature(),
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            )));
        }
        for (i, (arg, param)) in args.iter().zip(&sig.params).enumerate() {
            if arg.marshal_type() != param.ty {
                return Err(JitError::Marshal(format!(
                    "argument {} (`{}`) of `{}` is annotated `{}`, but {} `{}` was passed; \
                     the compiled function takes exactly its annotated type and the JIT \
                     never converts an argument (CPython would run it on the `{}` itself)",
                    i + 1,
                    param.name,
                    self.qualified(&sig.name),
                    param.ty.name(),
                    article(arg.type_name()),
                    arg.type_name(),
                    arg.type_name()
                )));
            }
        }
        let ry_args: Vec<RyValue> = args.iter().map(to_ry).collect();
        let mut out = RyOutcome::empty();
        flush_host_stdout();
        // SAFETY: the shim has the boundary's one function signature; the
        // argument strings outlive the call.
        let status = unsafe { shim(ry_args.as_ptr(), ry_args.len(), &mut out) };
        let text = self.take_text(out.text_ptr, out.text_len);
        match status {
            RY_STATUS_OK => self.take_value(&out.value),
            RY_STATUS_RAISED => Err(JitError::Python(decode_exception(text.as_deref().unwrap_or(""))?)),
            RY_STATUS_EXIT => Err(JitError::Exit {
                code: out.code as i32,
                message: text,
            }),
            RY_STATUS_PANIC => Err(JitError::Panic(text.unwrap_or_default())),
            RY_STATUS_MARSHAL => Err(JitError::Marshal(text.unwrap_or_default())),
            RY_STATUS_ROLE => Err(JitError::Unsupported(text.unwrap_or_default())),
            other => Err(JitError::Build(format!(
                "the library returned an unknown status {} from `{}`",
                other,
                self.qualified(function)
            ))),
        }
    }

    /// Why `function` has no shim, as the loud error its call is.
    fn not_callable(&self, function: &str) -> JitError {
        let qualified = self.qualified(function);
        if let Some(u) = self.manifest.unshimmed.iter().find(|u| u.name == function) {
            let message = format!("`{}` cannot be called through the JIT: {}", qualified, u.reason);
            return match u.kind {
                UnshimmedKind::Marshal => JitError::Marshal(message),
                UnshimmedKind::Unsupported => JitError::Unsupported(message),
            };
        }
        if self.manifest.classes.iter().any(|c| c == function) {
            return JitError::Unsupported(format!(
                "`{}` is a class; the JIT calls module-level functions only in M1 \
                 (call a function that constructs and uses it)",
                qualified
            ));
        }
        JitError::Unsupported(format!(
            "`{}` is not a public module-level function of `{}`; the JIT calls functions \
             defined with `def` at the module's top level whose names do not start with \
             `_` (M1)",
            qualified, self.manifest.module
        ))
    }

    /// Decode a result the library wrote, releasing its string.
    fn take_value(&self, value: &RyValue) -> Result<Value, JitError> {
        match MarshalType::from_tag(value.tag) {
            Some(MarshalType::None) => Ok(Value::None),
            Some(MarshalType::Bool) => Ok(Value::Bool(value.boolean != 0)),
            Some(MarshalType::Int) => Ok(Value::Int(value.int)),
            Some(MarshalType::Float) => Ok(Value::Float(value.float)),
            Some(MarshalType::Str) => {
                let text = self.take_text(value.ptr, value.len).unwrap_or_default();
                Ok(Value::Str(text))
            }
            None => Err(JitError::Build(format!(
                "the library returned a value with unknown tag {}",
                value.tag
            ))),
        }
    }

    /// Install the `sys.argv` function calls see, before anything reads it.
    pub(crate) fn set_argv(&self, argv: &[String]) -> Result<(), JitError> {
        // A library with no function and no entry runs nothing that could
        // read sys.argv.
        let Some(set_argv) = self.set_argv else {
            return Ok(());
        };
        let ry_argv: Vec<RyStr> = argv
            .iter()
            .map(|a| RyStr {
                ptr: a.as_ptr(),
                len: a.len(),
            })
            .collect();
        let mut out = RyOutcome::empty();
        // SAFETY: ry_set_argv has the entry's signature; the strings
        // outlive the call.
        let status = unsafe { set_argv(ry_argv.as_ptr(), ry_argv.len(), &mut out) };
        let text = self.take_text(out.text_ptr, out.text_len);
        match status {
            RY_STATUS_OK => Ok(()),
            RY_STATUS_ROLE => Err(JitError::Unsupported(text.unwrap_or_default())),
            RY_STATUS_MARSHAL => Err(JitError::Marshal(text.unwrap_or_default())),
            other => Err(JitError::Build(format!(
                "the library returned an unknown status {} from ry_set_argv",
                other
            ))),
        }
    }

    /// Run the program: `argv` becomes its `sys.argv`. The program's exit
    /// status comes back; the library never ends the process.
    pub(crate) fn run_entry(&self, argv: &[String]) -> Result<i32, JitError> {
        let entry = match (&self.manifest.entry, self.entry) {
            (Entry::Program, Some(entry)) => entry,
            (Entry::Refused(why), _) => return Err(JitError::Conversion(why.clone())),
            (Entry::Program, None) => unreachable!("load resolves ry_entry for a program"),
        };
        let ry_argv: Vec<RyStr> = argv
            .iter()
            .map(|a| RyStr {
                ptr: a.as_ptr(),
                len: a.len(),
            })
            .collect();
        let mut out = RyOutcome::empty();
        flush_host_stdout();
        // SAFETY: ry_entry has the boundary's entry signature; the
        // argument strings outlive the call.
        let status = unsafe { entry(ry_argv.as_ptr(), ry_argv.len(), &mut out) };
        let text = self.take_text(out.text_ptr, out.text_len);
        match status {
            RY_STATUS_OK => Ok(out.code as i32),
            RY_STATUS_ENTRY_ALREADY_RUN => Err(JitError::EntryAlreadyRun {
                module: self.manifest.module.clone(),
            }),
            RY_STATUS_ROLE => Err(JitError::Unsupported(text.unwrap_or_default())),
            RY_STATUS_PANIC => Err(JitError::Panic(text.unwrap_or_default())),
            RY_STATUS_MARSHAL => Err(JitError::Marshal(text.unwrap_or_default())),
            other => Err(JitError::Build(format!(
                "the library returned an unknown status {} from its entry",
                other
            ))),
        }
    }
}

/// An argument as the boundary carries it (strings borrowed for the call).
fn to_ry(value: &Value) -> RyValue {
    let mut ry = RyValue::none();
    ry.tag = value.marshal_type().tag();
    match value {
        Value::None => {}
        Value::Bool(b) => ry.boolean = u8::from(*b),
        Value::Int(n) => ry.int = *n,
        Value::Float(x) => ry.float = *x,
        Value::Str(s) => {
            ry.ptr = s.as_ptr().cast_mut();
            ry.len = s.len();
        }
    }
    ry
}

/// Rebuild a `PyException` from the library's encoding (see
/// `RY_STATUS_RAISED`): type, message, argument reprs, ancestry.
fn decode_exception(text: &str) -> Result<stdpython::PyException, JitError> {
    let malformed = || JitError::Build(format!("the library encoded an exception malformedly: {:?}", text));
    let mut fields = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let (len, after) = rest.split_once(':').ok_or_else(malformed)?;
        let len: usize = len.parse().map_err(|_| malformed())?;
        if after.len() < len || !after.is_char_boundary(len) {
            return Err(malformed());
        }
        fields.push(&after[..len]);
        rest = &after[len..];
    }
    let mut fields = fields.into_iter();
    let mut next = || fields.next().ok_or_else(malformed);
    let type_name = next()?.to_string();
    let message = next()?.to_string();
    let args = match next()? {
        "-" => None,
        count => {
            let count: usize = count.parse().map_err(|_| malformed())?;
            let mut args = Vec::with_capacity(count);
            for _ in 0..count {
                args.push(next()?.to_string());
            }
            Some(args)
        }
    };
    let ancestor_count: usize = next()?.parse().map_err(|_| malformed())?;
    let mut ancestors = Vec::with_capacity(ancestor_count);
    for _ in 0..ancestor_count {
        ancestors.push(next()?.to_string());
    }
    let exception = stdpython::PyException::new_with_attrs_and_ancestors(type_name, message, Vec::new(), ancestors);
    Ok(match args {
        Some(args) => exception.with_args_repr(args),
        None => exception,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_encoded_exception_decodes_to_its_type_message_arguments_and_ancestry() {
        let e = decode_exception("7:MyError13:('custom', 0)1:28:'custom'1:01:19:Exception").unwrap();
        assert_eq!(e.exception_type, "MyError");
        assert_eq!(e.message, "('custom', 0)");
        assert_eq!(e.args_repr, Some(vec!["'custom'".to_string(), "0".to_string()]));
        assert_eq!(e.user_ancestors, vec!["Exception".to_string()]);
        let e = decode_exception("10:ValueError3:bad1:-1:0").unwrap();
        assert_eq!(e.args_repr, None);
        assert!(decode_exception("3:abc").is_err());
        assert!(decode_exception("99:short").is_err());
    }
}
