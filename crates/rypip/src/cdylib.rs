//! The converter's cdylib mode: the library form the JIT loads (issue
//! #166, docs/jit.md). The crate is what `rypip convert` emits for the
//! same module, plus one generated module, `src/__rython_jit.rs`, holding
//! the C-ABI boundary:
//!
//! - `ry_abi_version`, `ry_manifest` and `ry_free`;
//! - one `ryfn_<n>` shim per public module-level function whose
//!   parameters and result are M1-marshallable (`int`, `float`, `bool`,
//!   `None`, `str`), each importing the module first (its
//!   `__module_init__`, as `import` runs the body), catching unwinds,
//!   flushing the library's stdout, and returning a status;
//! - `ry_entry`, which installs the program's `sys.argv` in this library's
//!   runtime and runs what the binary's `main` runs (the module's
//!   `__rython_main__`), reporting an uncaught exception as the binary does
//!   and returning the exit status instead of ending the host.
//!
//! Every function without a shim is listed in the manifest with the reason,
//! so calling it is a loud error naming the type or construct.

use python_ast::{ExprType, FunctionDef, StatementType, safe_ident};
use rython_jit::manifest::{
    Calls, Entry, FunctionSig, Manifest, MarshalType, Param, Unshimmed, UnshimmedKind,
};

/// The generated boundary module's name (`src/__rython_jit.rs`).
pub(crate) const BOUNDARY_MODULE: &str = "__rython_jit";

/// What the boundary generation needs to know about the one module.
pub(crate) struct CdylibModule<'a> {
    /// The module's Python name (the manifest's `module`).
    pub python_name: &'a str,
    /// Its Rust module path under the crate root.
    pub rust_path: &'a [String],
    pub source: &'a str,
    pub ast: &'a python_ast::Module,
    /// The transpiled module (the lib copy), as token text: which names
    /// the lowering emitted (`python_main`, the program body function).
    pub code: &'a str,
    /// Whether the converter selected the module as the program's entry.
    pub is_entry: bool,
    /// Whether the module's entry is the bare module-level `main()` call
    /// (issue #431): an `import` under CPython would run it.
    pub entry_is_bare_main_call: bool,
    /// Whether the module uses async constructs.
    pub uses_async: bool,
}

/// The boundary module's source and the manifest it carries.
pub(crate) fn generate(m: &CdylibModule<'_>) -> (String, Manifest) {
    let (functions, unshimmed) = classify_functions(m);
    let classes = m
        .ast
        .raw
        .body
        .iter()
        .filter_map(|s| match &s.statement {
            StatementType::ClassDef(c) => Some(c.name.clone()),
            _ => None,
        })
        .collect();
    let has_program_body = m
        .code
        .contains(&format!("fn {} (", python_ast::PROGRAM_BODY_FN));
    let entry = if m.is_entry && has_program_body {
        Entry::Program
    } else if m.is_entry && m.uses_async {
        Entry::Refused(format!(
            "`{}` has an async entry point, which the JIT cannot run yet (M1 runs \
             synchronous programs; the async runtime's entry is not part of the library \
             boundary); run it with `rypip run`, or make `main` synchronous",
            m.python_name
        ))
    } else {
        Entry::Refused(format!(
            "`{}` has no entry point; add an `if __name__ == \"__main__\":` block to run \
             it as a program (call its functions with `rypip jit call` instead)",
            m.python_name
        ))
    };
    let calls = if m.entry_is_bare_main_call {
        Calls::Refused(format!(
            "`{}` ends with a bare `main()` call, which CPython runs whenever the module is \
             imported, but which rython compiles as the program's entry; calling its \
             functions would import it without running `main()`, and rython refuses to \
             silently skip it — wrap the call in `if __name__ == \"__main__\":` to call \
             the module's functions through the JIT",
            m.python_name
        ))
    } else {
        Calls::Allowed
    };
    let manifest = Manifest {
        module: m.python_name.to_string(),
        entry,
        calls,
        functions: functions.iter().map(|(sig, _)| sig.clone()).collect(),
        unshimmed,
        classes,
    };
    let source = render_boundary(m, &manifest, &functions);
    (source, manifest)
}

/// A shimmed function: its manifest signature and the Rust function the
/// shim calls.
type Shimmed = (FunctionSig, proc_macro2::Ident);

/// Sort the module's public top-level functions into shims and refusals.
fn classify_functions(m: &CdylibModule<'_>) -> (Vec<Shimmed>, Vec<Unshimmed>) {
    let mut shimmed: Vec<Shimmed> = Vec::new();
    let mut unshimmed: Vec<Unshimmed> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for stmt in &m.ast.raw.body {
        let (func, is_async) = match &stmt.statement {
            StatementType::FunctionDef(f) => (f, false),
            StatementType::AsyncFunctionDef(f) => (f, true),
            _ => continue,
        };
        if func.name.starts_with('_') {
            continue;
        }
        if seen.contains(&func.name) {
            // Python's last definition wins; M1 refuses the name instead.
            shimmed.retain(|(sig, _)| sig.name != func.name);
            unshimmed.retain(|u| u.name != func.name);
            unshimmed.push(Unshimmed {
                name: func.name.clone(),
                kind: UnshimmedKind::Unsupported,
                reason: "it is defined more than once at module level (M1 calls a \
                         function defined once)"
                    .to_string(),
            });
            continue;
        }
        seen.push(func.name.clone());
        let refusal = |kind, reason: String| Unshimmed {
            name: func.name.clone(),
            kind,
            reason,
        };
        if is_async {
            unshimmed.push(refusal(
                UnshimmedKind::Unsupported,
                "it is an `async def` (a coroutine function), which M1 does not call".to_string(),
            ));
            continue;
        }
        match signature(func, m.source) {
            Ok((params, returns)) => {
                let symbol = format!("ryfn_{}", shimmed.len());
                let rust_name = if func.name == "main" && m.code.contains("fn python_main (") {
                    // The lowering moved the user's `main` aside for the
                    // generated entry (issue #479).
                    safe_ident("python_main")
                } else {
                    safe_ident(&func.name)
                };
                shimmed.push((
                    FunctionSig {
                        name: func.name.clone(),
                        symbol,
                        params,
                        returns,
                    },
                    rust_name,
                ));
            }
            Err((kind, reason)) => unshimmed.push(refusal(kind, reason)),
        }
    }
    // Shim symbols are numbered by position; renumber after any removal.
    for (i, (sig, _)) in shimmed.iter_mut().enumerate() {
        sig.symbol = format!("ryfn_{}", i);
    }
    (shimmed, unshimmed)
}

/// The function's M1 signature, or why it has none.
fn signature(
    func: &FunctionDef,
    source: &str,
) -> Result<(Vec<Param>, MarshalType), (UnshimmedKind, String)> {
    let unsupported = |what: &str| {
        Err((
            UnshimmedKind::Unsupported,
            format!("it {}, which M1 does not marshal (positional parameters without defaults only)", what),
        ))
    };
    if !func.decorator_list.is_empty() {
        return Err((
            UnshimmedKind::Unsupported,
            "it is decorated, and M1 calls undecorated functions only".to_string(),
        ));
    }
    if func.is_generator() {
        return Err((
            UnshimmedKind::Unsupported,
            "it is a generator function, which M1 does not call".to_string(),
        ));
    }
    let args = &func.args;
    if args.vararg.is_some() {
        return unsupported("takes `*args`");
    }
    if args.kwarg.is_some() {
        return unsupported("takes `**kwargs`");
    }
    if !args.kwonlyargs.is_empty() {
        return unsupported("has keyword-only parameters");
    }
    if !args.posonlyargs.is_empty() {
        return unsupported("has positional-only parameters");
    }
    if !args.defaults.is_empty() {
        return unsupported("has parameters with default values");
    }
    let mut params = Vec::new();
    for param in &args.args {
        let Some(annotation) = param.annotation.as_deref() else {
            return Err((
                UnshimmedKind::Marshal,
                format!(
                    "parameter `{}` has no annotation; the JIT marshals a parameter by its \
                     annotated type (int, float, bool, str or None)",
                    param.arg
                ),
            ));
        };
        let Some(ty) = annotation_type(annotation) else {
            return Err((
                UnshimmedKind::Marshal,
                format!(
                    "parameter `{}` is annotated `{}`, a type M1 cannot marshal across the \
                     JIT boundary (int, float, bool, str and None cross)",
                    param.arg,
                    annotation_text(annotation, source)
                ),
            ));
        };
        params.push(Param {
            name: param.arg.clone(),
            ty,
        });
    }
    // The return type the generated function carries: the one authority
    // call-through code uses (as the PyO3 bindings do).
    let resolved = func.resolved_return_type(
        &python_ast::SymbolTableScopes::new(),
        &python_ast::PythonOptions::default(),
    );
    let returns = match resolved {
        None => match func.returns.as_deref() {
            None => MarshalType::None,
            Some(ann) if python_ast::is_none_expr(ann) => MarshalType::None,
            Some(ann) => {
                return Err((
                    UnshimmedKind::Marshal,
                    format!(
                        "its result is annotated `{}`, which M1 cannot marshal back from \
                         the compiled function",
                        annotation_text(ann, source)
                    ),
                ));
            }
        },
        Some(tokens) => match return_type(&tokens.to_string()) {
            Some(ty) => ty,
            None => {
                let described = match func.returns.as_deref() {
                    Some(ann) => format!("annotated `{}`", annotation_text(ann, source)),
                    None => format!("inferred as the Rust type `{}`", tokens),
                };
                return Err((
                    UnshimmedKind::Marshal,
                    format!(
                        "its result is {}, a type M1 cannot marshal back across the JIT \
                         boundary (int, float, bool, str and None cross)",
                        described
                    ),
                ));
            }
        },
    };
    Ok((params, returns))
}

/// An annotation's M1 type: a bare `int` / `float` / `bool` / `str`, or
/// `None`.
fn annotation_type(annotation: &ExprType) -> Option<MarshalType> {
    if python_ast::is_none_expr(annotation) {
        return Some(MarshalType::None);
    }
    match annotation {
        ExprType::Name(name) => MarshalType::from_name(&name.id),
        _ => None,
    }
}

/// The M1 type of a generated function's resolved Rust return type.
fn return_type(rust: &str) -> Option<MarshalType> {
    match rust.replace(' ', "").as_str() {
        "()" => Some(MarshalType::None),
        "bool" => Some(MarshalType::Bool),
        "i64" => Some(MarshalType::Int),
        "f64" => Some(MarshalType::Float),
        "String" | "&'staticstr" | "&str" => Some(MarshalType::Str),
        _ => None,
    }
}

/// An annotation as the user wrote it, for messages: the source text of a
/// subscript (`list[int]`), else a rendering of its parts.
fn annotation_text(annotation: &ExprType, source: &str) -> String {
    match annotation {
        ExprType::Name(name) => name.id.clone(),
        ExprType::Attribute(attr) => format!("{}.{}", annotation_text(&attr.value, source), attr.attr),
        ExprType::Subscript(sub) => {
            let text = match (sub.lineno, sub.col_offset, sub.end_lineno, sub.end_col_offset) {
                (Some(line), Some(start), Some(end_line), Some(end)) if line == end_line => source
                    .lines()
                    .nth(line.saturating_sub(1))
                    .and_then(|l| l.get(start..end))
                    .map(str::to_string),
                _ => None,
            };
            text.unwrap_or_else(|| format!("{}[...]", annotation_text(&sub.value, source)))
        }
        ExprType::BinOp(op) if matches!(op.op, python_ast::BinOps::BitOr) => format!(
            "{} | {}",
            annotation_text(&op.left, source),
            annotation_text(&op.right, source)
        ),
        other if python_ast::is_none_expr(other) => "None".to_string(),
        _ => "this annotation".to_string(),
    }
}

/// The Rust type a shim passes for a parameter of an M1 type: the type the
/// lowering gives a parameter with that annotation (`str` arrives through
/// `impl Into<String>`, a `None` annotation is `Option<()>`).
fn param_rust_type(ty: MarshalType) -> &'static str {
    match ty {
        MarshalType::None => "Option<()>",
        MarshalType::Bool => "bool",
        MarshalType::Int => "i64",
        MarshalType::Float => "f64",
        MarshalType::Str => "String",
    }
}

/// The tag constant a parameter's argument must carry.
fn tag_const(ty: MarshalType) -> &'static str {
    match ty {
        MarshalType::None => "RY_TAG_NONE",
        MarshalType::Bool => "RY_TAG_BOOL",
        MarshalType::Int => "RY_TAG_INT",
        MarshalType::Float => "RY_TAG_FLOAT",
        MarshalType::Str => "RY_TAG_STR",
    }
}

/// A Rust string literal for `text`.
fn rust_str(text: &str) -> String {
    format!("{:?}", text)
}

fn render_boundary(m: &CdylibModule<'_>, manifest: &Manifest, functions: &[Shimmed]) -> String {
    let module_path = format!("crate::{}", m.rust_path.iter().map(|p| safe_ident(p).to_string()).collect::<Vec<_>>().join("::"));
    let has_entry = matches!(manifest.entry, Entry::Program);
    let has_shims = !functions.is_empty();
    let mut out = String::new();
    out.push_str(BOUNDARY_HEADER);
    out.push_str("#[allow(dead_code)]\nmod abi {\n");
    out.push_str(rython_jit::abi::ABI_SOURCE);
    out.push_str("}\nuse abi::*;\n\n");
    out.push_str(&format!(
        "/// The manifest the host reads on load (rython_jit::manifest).\nstatic MANIFEST: &str = {};\n\n",
        rust_str(&manifest.render())
    ));
    out.push_str(BOUNDARY_CORE);
    if has_entry || has_shims {
        out.push_str(&BOUNDARY_DISPATCH.replace("@@MODULE@@", &rust_str(m.python_name)));
    }
    if has_shims {
        out.push_str(&BOUNDARY_CALLS.replace("@@MODULE_PATH@@", &module_path));
        if functions.iter().any(|(sig, _)| !sig.params.is_empty()) {
            out.push_str(BOUNDARY_FROM_RY);
        }
    }
    if has_entry {
        out.push_str(&BOUNDARY_ENTRY.replace(
            "@@PROGRAM_BODY@@",
            &format!("{}::{}", module_path, python_ast::PROGRAM_BODY_FN),
        ));
    }
    for (sig, rust_name) in functions {
        let tags: Vec<&str> = sig.params.iter().map(|p| tag_const(p.ty)).collect();
        let call_args: Vec<String> = sig
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| format!("<{} as FromRy>::from_ry(&args[{}])", param_rust_type(p.ty), i))
            .collect();
        // The checked arguments, bound only when the function takes some.
        let checked = format!("unsafe {{ arguments(args, nargs, &[{}]) }}", tags.join(", "));
        let check = if sig.params.is_empty() {
            format!(
                "    if let Err(message) = {} {{\n        return unsafe {{ refuse(out, message) }};\n    }}\n",
                checked
            )
        } else {
            format!(
                "    let args = match {} {{\n        Ok(args) => args,\n        Err(message) => return unsafe {{ refuse(out, message) }},\n    }};\n",
                checked
            )
        };
        out.push_str(&format!(
            "\n/// `{signature}`\n\
             #[no_mangle]\n\
             pub unsafe extern \"C\" fn {symbol}(args: *const RyValue, nargs: usize, out: *mut RyOutcome) -> u32 {{\n\
             {check}\
             \x20   unsafe {{\n\
             \x20       call_function(out, {name}, move || {{\n\
             \x20           {path}::{rust_name}({call_args}).map(|value| (&mut Return(Some(value))).to_ry())\n\
             \x20       }})\n\
             \x20   }}\n\
             }}\n",
            signature = sig.signature(),
            symbol = sig.symbol,
            check = check,
            name = rust_str(&sig.name),
            path = module_path,
            rust_name = rust_name,
            call_args = call_args.join(", "),
        ));
    }
    out
}

const BOUNDARY_HEADER: &str = "\
//! The JIT boundary (rython-jit, issue #166; docs/jit.md): C-ABI shims over
//! this crate's module, generated by rypip's cdylib mode. Only C-compatible
//! values cross: this library links its own `std` and `stdpython`, which
//! the host does not share. Every shim catches unwinds, flushes this
//! library's stdout before returning, and reports how the call ended as a
//! status; nothing here ends the host process.

// The boundary's vocabulary, verbatim from rython-jit (the host compiles
// the same text). A library uses the subset its shims need.
";

const BOUNDARY_CORE: &str = r#"#[no_mangle]
pub extern "C" fn ry_abi_version() -> u32 {
    RY_ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn ry_manifest(len: *mut usize) -> *const u8 {
    unsafe { *len = MANIFEST.len() };
    MANIFEST.as_ptr()
}

/// Release a buffer this library handed to the host.
#[no_mangle]
pub unsafe extern "C" fn ry_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}
"#;

const BOUNDARY_DISPATCH: &str = r#"
/// The module's Python name, for messages.
const MODULE: &str = @@MODULE@@;

/// What this library's one copy of the module state serves: nothing yet,
/// function calls (the body ran as an `import`), or the program entry (it
/// ran as `__main__`). CPython would run those as two separate modules, so
/// one library serves one of them.
static ROLE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(ROLE_NONE);
const ROLE_NONE: u8 = 0;
const ROLE_IMPORTED: u8 = 1;
const ROLE_ENTRY: u8 = 2;

fn claim_role(role: u8, out: &mut RyOutcome) -> Result<(), u32> {
    use std::sync::atomic::Ordering::SeqCst;
    match ROLE.compare_exchange(ROLE_NONE, role, SeqCst, SeqCst) {
        Ok(_) => Ok(()),
        Err(current) if current == role && role == ROLE_IMPORTED => Ok(()),
        Err(ROLE_ENTRY) if role == ROLE_ENTRY => Err(RY_STATUS_ENTRY_ALREADY_RUN),
        Err(ROLE_ENTRY) => {
            set_text(out, format!(
                "`{}` already ran as a program (`__main__`) in this process, so its \
                 functions cannot be called through the same library: CPython would import \
                 `{}` as a second, fresh module, and one loaded library holds one copy of \
                 the module's state (M1)",
                MODULE, MODULE
            ));
            Err(RY_STATUS_ROLE)
        }
        Err(_) => {
            set_text(out, format!(
                "`{}` was already imported by a function call in this process, so it cannot \
                 run as a program through the same library: CPython would run `__main__` as \
                 a second, fresh module, and one loaded library holds one copy of the \
                 module's state (M1)",
                MODULE
            ));
            Err(RY_STATUS_ROLE)
        }
    }
}

/// Hand `text` to the host in the outcome (released with `ry_free`).
fn set_text(out: &mut RyOutcome, text: String) {
    let bytes: Box<[u8]> = text.into_bytes().into_boxed_slice();
    out.text_len = bytes.len();
    out.text_ptr = Box::into_raw(bytes).cast::<u8>();
}

/// The program arguments the host passed, as strings.
unsafe fn program_args(argv: *const RyStr, argc: usize) -> Result<Vec<String>, String> {
    let mut args = Vec::with_capacity(argc);
    for i in 0..argc {
        let arg = unsafe { *argv.add(i) };
        match std::str::from_utf8(unsafe { std::slice::from_raw_parts(arg.ptr, arg.len) }) {
            Ok(text) => args.push(text.to_string()),
            Err(_) => return Err(format!("program argument {} is not UTF-8", i)),
        }
    }
    Ok(args)
}

/// Install `argv` as this library's `sys.argv`, before anything reads it.
fn install_argv(args: Vec<String>, out: &mut RyOutcome) -> Result<(), u32> {
    stdpython::sys::install_program_args(args).map_err(|why| {
        set_text(out, format!(
            "`{}`'s sys.argv was already read or installed in this library ({:?}); \
             sys.argv is initialized once, so other arguments cannot be installed",
            MODULE, why
        ));
        RY_STATUS_ROLE
    })
}

/// Install the arguments function calls see as `sys.argv` (a host that
/// is not the program: `rypip jit call` installs `['-c']`, which `python3
/// -c` has). Refused once anything read or installed them.
#[no_mangle]
pub unsafe extern "C" fn ry_set_argv(argv: *const RyStr, argc: usize, out: *mut RyOutcome) -> u32 {
    let out = unsafe { &mut *out };
    let args = match unsafe { program_args(argv, argc) } {
        Ok(args) => args,
        Err(message) => {
            set_text(out, message);
            return RY_STATUS_MARSHAL;
        }
    };
    match install_argv(args, out) {
        Ok(()) => RY_STATUS_OK,
        Err(status) => status,
    }
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => message.to_string(),
            Err(_) => "Box<dyn Any>".to_string(),
        },
    }
}
"#;

const BOUNDARY_CALLS: &str = r#"
/// The arguments, checked against the shim's parameter tags before
/// anything runs.
unsafe fn arguments<'a>(
    args: *const RyValue,
    nargs: usize,
    tags: &[u32],
) -> Result<&'a [RyValue], String> {
    if nargs != tags.len() {
        return Err(format!("expected {} arguments, got {}", tags.len(), nargs));
    }
    if nargs == 0 {
        return Ok(&[]);
    }
    let args = unsafe { std::slice::from_raw_parts(args, nargs) };
    for (i, (arg, tag)) in args.iter().zip(tags).enumerate() {
        if arg.tag != *tag {
            return Err(format!("argument {} carries tag {}, not {}", i + 1, arg.tag, tag));
        }
        if arg.tag == RY_TAG_STR
            && std::str::from_utf8(unsafe { std::slice::from_raw_parts(arg.ptr, arg.len) }).is_err()
        {
            return Err(format!("argument {} is not UTF-8", i + 1));
        }
    }
    Ok(args)
}

unsafe fn refuse(out: *mut RyOutcome, message: String) -> u32 {
    set_text(unsafe { &mut *out }, message);
    RY_STATUS_MARSHAL
}

/// A value this library returns across the boundary.
trait IntoRy {
    fn into_ry(self) -> RyValue;
}

fn ry_value(tag: u32) -> RyValue {
    RyValue { tag, boolean: 0, int: 0, float: 0.0, ptr: std::ptr::null_mut(), len: 0 }
}

impl IntoRy for () {
    fn into_ry(self) -> RyValue {
        ry_value(RY_TAG_NONE)
    }
}

impl IntoRy for bool {
    fn into_ry(self) -> RyValue {
        RyValue { boolean: u8::from(self), ..ry_value(RY_TAG_BOOL) }
    }
}

impl IntoRy for i64 {
    fn into_ry(self) -> RyValue {
        RyValue { int: self, ..ry_value(RY_TAG_INT) }
    }
}

impl IntoRy for f64 {
    fn into_ry(self) -> RyValue {
        RyValue { float: self, ..ry_value(RY_TAG_FLOAT) }
    }
}

impl IntoRy for String {
    fn into_ry(self) -> RyValue {
        let bytes: Box<[u8]> = self.into_bytes().into_boxed_slice();
        let len = bytes.len();
        RyValue { ptr: Box::into_raw(bytes).cast::<u8>(), len, ..ry_value(RY_TAG_STR) }
    }
}

impl IntoRy for &str {
    fn into_ry(self) -> RyValue {
        self.to_string().into_ry()
    }
}

/// A function's result on its way out. Method resolution picks
/// `MarshalReturn` when the result's type crosses the boundary and falls
/// back to `OpaqueReturn` (one auto-reference further) when it does not,
/// so a result type the converter's prediction got wrong is a loud
/// marshal error at the call, never a library that fails to build.
struct Return<T>(Option<T>);

trait MarshalReturn {
    fn to_ry(&mut self) -> Result<RyValue, String>;
}

impl<T: IntoRy> MarshalReturn for Return<T> {
    fn to_ry(&mut self) -> Result<RyValue, String> {
        Ok(self.0.take().expect("a result converts once").into_ry())
    }
}

// Resolution reaches this trait only for a result type that does not
// cross; when every result does, it is unused by design.
#[allow(dead_code)]
trait OpaqueReturn {
    fn to_ry(&mut self) -> Result<RyValue, String>;
}

impl<T> OpaqueReturn for &mut Return<T> {
    fn to_ry(&mut self) -> Result<RyValue, String> {
        Err(std::any::type_name::<T>().to_string())
    }
}

/// Encode a raised exception for the host (see `RY_STATUS_RAISED`).
fn encode_exception(e: &stdpython::PyException) -> String {
    let mut fields: Vec<&str> = vec![&e.exception_type, &e.message];
    let arg_count;
    match &e.args_repr {
        None => fields.push("-"),
        Some(args) => {
            arg_count = args.len().to_string();
            fields.push(&arg_count);
            fields.extend(args.iter().map(String::as_str));
        }
    }
    let ancestor_count = e.user_ancestors.len().to_string();
    fields.push(&ancestor_count);
    fields.extend(e.user_ancestors.iter().map(String::as_str));
    let mut text = String::new();
    for field in fields {
        text.push_str(&field.len().to_string());
        text.push(':');
        text.push_str(field);
    }
    text
}

/// How an escaping exception leaves a function call: `SystemExit` as its
/// exit status (the host decides; the library never ends the process),
/// anything else encoded for the host to rebuild.
fn raised(out: &mut RyOutcome, e: &stdpython::PyException) -> u32 {
    match e.system_exit_status() {
        Ok(Some((code, message))) => {
            out.code = i64::from(code);
            if let Some(message) = message {
                set_text(out, message);
            }
            RY_STATUS_EXIT
        }
        Ok(None) => {
            set_text(out, encode_exception(e));
            RY_STATUS_RAISED
        }
        Err(error) => {
            out.code = 1;
            set_text(out, format!("rython: an uncaught SystemExit has no recorded code: {}", error));
            RY_STATUS_EXIT
        }
    }
}

/// Import the module (its body runs once, as `import` runs it), then run
/// `body`, catching an unwind; flush stdout; report the outcome.
unsafe fn call_function(
    out: *mut RyOutcome,
    name: &str,
    body: impl FnOnce() -> Result<Result<RyValue, String>, stdpython::PyException>,
) -> u32 {
    let out = unsafe { &mut *out };
    if let Err(status) = claim_role(ROLE_IMPORTED, out) {
        return status;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Result<RyValue, String>, stdpython::PyException> {
            @@MODULE_PATH@@::__module_init__()?;
            body()
        },
    ));
    let _ = std::io::Write::flush(&mut std::io::stdout());
    match result {
        Ok(Ok(Ok(value))) => {
            out.value = value;
            RY_STATUS_OK
        }
        Ok(Ok(Err(rust_type))) => {
            set_text(out, format!(
                "`{}.{}` returned a value of the Rust type `{}`, which M1 cannot marshal \
                 back across the JIT boundary (int, float, bool, str and None cross)",
                MODULE, name, rust_type
            ));
            RY_STATUS_MARSHAL
        }
        Ok(Err(e)) => raised(out, &e),
        Err(payload) => {
            set_text(out, panic_text(payload));
            RY_STATUS_PANIC
        }
    }
}
"#;

const BOUNDARY_FROM_RY: &str = r#"
/// An argument as the shim passes it: the parameter's lowered type.
trait FromRy {
    fn from_ry(value: &RyValue) -> Self;
}

impl FromRy for Option<()> {
    fn from_ry(_: &RyValue) -> Option<()> {
        None
    }
}

impl FromRy for bool {
    fn from_ry(value: &RyValue) -> bool {
        value.boolean != 0
    }
}

impl FromRy for i64 {
    fn from_ry(value: &RyValue) -> i64 {
        value.int
    }
}

impl FromRy for f64 {
    fn from_ry(value: &RyValue) -> f64 {
        value.float
    }
}

impl FromRy for String {
    fn from_ry(value: &RyValue) -> String {
        // Checked as UTF-8 by `arguments`.
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(value.ptr, value.len) })
            .into_owned()
    }
}
"#;

const BOUNDARY_ENTRY: &str = r#"
/// Run the program as its binary would — module initialization, then the
/// `__main__` body — with `argv` installed as this library's `sys.argv`.
/// An uncaught exception is reported exactly as the binary reports it, and
/// the exit status comes back in `out.code`; the host's process is never
/// ended here.
#[no_mangle]
pub unsafe extern "C" fn ry_entry(argv: *const RyStr, argc: usize, out: *mut RyOutcome) -> u32 {
    let out = unsafe { &mut *out };
    if let Err(status) = claim_role(ROLE_ENTRY, out) {
        return status;
    }
    let args = match unsafe { program_args(argv, argc) } {
        Ok(args) => args,
        Err(message) => {
            set_text(out, message);
            return RY_STATUS_MARSHAL;
        }
    };
    if let Err(status) = install_argv(args, out) {
        return status;
    }
    let result = std::panic::catch_unwind(@@PROGRAM_BODY@@);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    match result {
        Ok(Ok(())) => {
            out.code = 0;
            RY_STATUS_OK
        }
        Ok(Err(e)) => {
            out.code = i64::from(stdpython::report_uncaught(&e));
            RY_STATUS_OK
        }
        Err(payload) => {
            set_text(out, panic_text(payload));
            RY_STATUS_PANIC
        }
    }
}
"#;
