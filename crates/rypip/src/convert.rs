//! Conversion of a discovered Python package into a Cargo crate: one Rust
//! module per Python module, an optional binary entry point, and optional
//! PyO3 bindings so the crate can be imported from Python again.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use proc_macro2::TokenStream;
use python_ast::{
    CodeGen, CodeGenContext, PythonOptions, StatementType, SymbolTableScopes, parse_enhanced,
    python_annotation_to_rust_type, safe_ident,
};
use quote::quote;
use rust_format::{Formatter, RustFmt};

use crate::package::{PyModule, PyPackage, sanitize_name};

/// Lint allowances for generated code: transpiled Python legitimately
/// produces unused imports/variables and similar noise, and the generated
/// crate must still build under a consumer's `-D warnings`. `deprecated` is
/// allowed *within* the generated crate because rython uses #[deprecated]
/// notes to warn about lossy conversions (e.g. dropped parameter defaults) —
/// internal call sites are the faithfully-transpiled Python, while external
/// consumers still get the warning at their call sites.
/// Rustc lints the generated code interacts with. Most surface genuine
/// weaknesses in the source Python — unused imports and variables, dead and
/// unreachable code, dead stores (a None seed that is never read),
/// non-snake-case names, calls to lossily-converted (#[deprecated])
/// functions — so they are NOT suppressed by default: surfacing them is
/// part of the point of the tooling. The generated crate's lint posture
/// follows the warning mode: warn leaves rustc's default (warnings at
/// build time), deny promotes them to hard errors, allow suppresses them.
const GENERATED_LINTS: &str = "unused_imports, unused_variables, unused_mut, unused_assignments, dead_code, unreachable_code, non_snake_case, non_upper_case_globals, deprecated, noop_method_call";

fn generated_lint_attrs(mode: WarningMode) -> String {
    match mode {
        WarningMode::Warn => String::new(),
        WarningMode::Deny => format!("#![deny({})]\n", GENERATED_LINTS),
        WarningMode::Allow => format!("#![allow({})]\n", GENERATED_LINTS),
    }
}

/// The crate-level attribute for the no_std profile. Only the crate root
/// carries it; each module brings its own alloc imports (emitted by the
/// module lowering itself).
fn no_std_attr(opts: &ConvertOptions) -> &'static str {
    if opts.no_std || opts.kernel_module {
        "#![no_std]\n"
    } else {
        ""
    }
}

/// How lossy-conversion warnings are treated during conversion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum WarningMode {
    /// Report warnings and bake #[deprecated] notes into the generated
    /// code; the generated crate keeps rustc's default lint warnings, which
    /// surface source-Python weaknesses at build time (the default).
    #[default]
    Warn,
    /// Promote warnings to errors: fail the conversion if any conversion is
    /// lossy, and deny the surfaced lints in the generated crate so its
    /// build fails on them.
    Deny,
    /// Suppress warnings entirely — nothing reported, no #[deprecated]
    /// notes, and the surfaced lints are allowed in the generated crate.
    Allow,
}

/// Options controlling crate generation.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Add PyO3 bindings (a `python` cargo feature, cdylib output, and a
    /// #[pymodule] exposing bindable functions).
    pub pyo3: bool,
    /// Path to the stdpython runtime crate the generated crate depends on.
    pub stdpython_path: Option<PathBuf>,
    /// How lossy-conversion warnings are treated.
    pub warnings: WarningMode,
    /// Generate a `#![no_std]` crate on stdpython's alloc tier (no OS
    /// dependency). Python constructs that need the OS — print/input/open,
    /// os/datetime/random/… imports, `__main__` blocks — fail the
    /// conversion loudly instead of surfacing later as build errors in the
    /// generated crate.
    pub no_std: bool,
    /// Generate a Linux kernel module crate. Implies no_std and sets
    /// panic = "abort", cdylib output, kernel-specific entry points
    /// (module_init/module_exit), and printk lowering. The stdpython
    /// dependency is dropped entirely — only core Rust is available.
    pub kernel_module: bool,
    /// Generate a userspace driver crate for a rython byte-ring misc device
    /// (UIO style): the Python driver logic is compiled to a library, and a
    /// syscall-glue binary (open/read/write/ioctl) is generated around it —
    /// the only Rust in the driver is this generated boilerplate. The
    /// Python declares the device manifest: `__device_path__`,
    /// `__ioc_reset__`, and `__ioc_stats__` module-level constants.
    pub driver: bool,
    /// Generate a rust-for-linux kernel module (issue #88): a `module!`-
    /// macro crate implementing `kernel::Module`, built with the
    /// rust-for-linux toolchain inside a rust-for-linux kernel tree.
    /// Requires `kernel_module`; overrides the raw-FFI output.
    pub rust_for_linux: bool,
    /// Skip resolving the packaging metadata's dependencies (`[project]
    /// dependencies` / install_requires) against PyPI. Vendored
    /// `[python-modules]` entries always win either way.
    pub no_deps: bool,
}

/// A converted crate on disk.
#[derive(Debug)]
pub struct ConvertedCrate {
    pub root: PathBuf,
    pub name: String,
    /// Whether a binary entry point (src/main.rs) was generated.
    pub has_binary: bool,
    /// Human-readable warnings about lossy conversions (e.g. dropped
    /// parameter defaults). These are also baked into the generated code as
    /// #[deprecated] notes so consumers see them at their call sites.
    pub warnings: Vec<String>,
}

/// Which kernel module backend to generate (issue #88).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelTarget {
    /// Raw C FFI: a `#![no_std]` staticlib with `#[no_mangle] extern "C"`
    /// init_module/cleanup_module, a kmalloc-backed global allocator, and
    /// `.modinfo` ELF sections. Builds on any kernel with headers installed
    /// (no CONFIG_RUST needed).
    #[default]
    RawFfi,
    /// The rust-for-linux `kernel` crate: a `module!`-macro module
    /// implementing `kernel::Module`, built with the rust-for-linux
    /// toolchain inside a rust-for-linux kernel tree.
    RustForLinux,
}

/// Result of lowering a Python module to a kernel module: the generated
/// `lib.rs` plus which module entry points exist (the build recipe needs to
/// know, so the linker keeps exactly the entry points that are defined).
pub struct GeneratedKernel {
    pub code: String,
    pub has_init: bool,
    pub has_exit: bool,
    /// The generated misc device (`src/device.rs`) when the Python declares
    /// a device manifest — `None` for plain kernel modules.
    pub device: Option<String>,
    /// The module links `rykernel-shim` (the Rust compatibility layer):
    /// either because it generated a device, or because the Python imports
    /// kernel resources from it (`from rykernel_shim import ...`). The
    /// generated Cargo.toml must declare the shim, and the module must not
    /// define its own panic handler (the shim provides one).
    pub uses_shim: bool,
}

/// Kernel resources importable from kernel-module Python. These are safe
/// wrappers in the rykernel-shim crate — the Rust compatibility layer that
/// declares the kernel symbols a module needs. Each name is a `pub fn` on
/// the shim with an integer return, callable with integer arguments, safe
/// from module init/exit context.
const KERNEL_SHIM_FNS: &[&str] = &["ktime_get_real_seconds"];

/// Scan top-level assignments for module metadata (`__module_license__`,
/// `__module_author__`, `__module_description__`, `__module_version__`).
/// The license defaults to GPL — the kernel's modpost requires one.
fn scan_modinfo(ast: &python_ast::Module) -> std::collections::BTreeMap<String, String> {
    use python_ast::ast::tree::StatementType;
    let mut modinfo: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::from([("license".into(), "GPL".into())]);
    for stmt in &ast.raw.body {
        if let StatementType::Assign(assign) = &stmt.statement {
            if let (Some(target), Some(val_str)) =
                (assign.targets.first(), expr_str_literal(&assign.value))
            {
                if let python_ast::ExprType::Name(name) = target {
                    let key = match name.id.as_str() {
                        "__module_license__" => Some("license"),
                        "__module_author__" => Some("author"),
                        "__module_description__" => Some("description"),
                        "__module_version__" => Some("version"),
                        _ => None,
                    };
                    if let Some(k) = key {
                        modinfo.insert(k.into(), val_str);
                    }
                }
            }
        }
    }
    modinfo
}

/// Emit a `.modinfo` section entry for each metadata key-value pair.
/// Entries are null-terminated `key=value\0` strings in an ELF .modinfo
/// section; each gets its own static with a generated symbol name.
/// `#[used]` keeps the entries alive across `ld -r --gc-sections` (the
/// C-free module link step): nothing references them from the entry
/// points, so without it the license would be GC'd out of the module.
///
/// The `.modinfo` placement is gated to Linux (`cfg_attr`): Mach-O rejects
/// dot-prefixed section specifiers outright, and a module authored on a
/// non-Linux host must still type-check there (`cargo check`) — only an
/// actual Linux kernel build needs the strings in `.modinfo`.
fn modinfo_statics(modinfo: &std::collections::BTreeMap<String, String>) -> String {
    let mut out = String::new();
    for (idx, (key, value)) in modinfo.iter().enumerate() {
        let entry = format!("{key}={value}");
        let len = entry.len();
        let padded = (len + 7) & !7; // align to 8 bytes
        out.push_str("#[used]\n#[no_mangle]\n#[cfg_attr(target_os = \"linux\", link_section = \".modinfo\")]\n");
        out.push_str(&format!(
            "static __mod_info_{idx}: [u8; {padded}] = *b\"{entry}"
        ));
        for _ in len..padded {
            out.push_str("\\0");
        }
        out.push_str("\";\n\n");
    }
    out
}

/// Generate a kernel module lib.rs from Python source. This bypasses the
/// full transpiler in favour of a minimal lowering that produces kernel
/// entry points, printk lowering, and module metadata — either raw `no_std`
/// Rust (RawFfi) or a rust-for-linux `module!` crate (RustForLinux). When
/// the Python declares a device manifest, the module becomes a generated
/// misc device (device.rs + entry points that register it).
fn generate_kernel_lib_rs(
    source: &str,
    target: KernelTarget,
    package_name: &str,
    manifest: &DeviceManifest,
) -> Result<GeneratedKernel> {
    use python_ast::ast::tree::StatementType;
    use python_ast::parse_enhanced;

    let ast =
        parse_enhanced(source, "<kernel>".to_string()).map_err(|e| anyhow::anyhow!("{}", e))?;

    // The kernel runs with the FPU in a lazy-save state: reject
    // floating-point usage loudly instead of emitting code that can corrupt
    // userspace state (issue #87). Unconditional — the device-manifest
    // sub-mode (ioctl handlers, buffer logic) must not be an unguarded path
    // (issue #108). The scan is target-independent, so run it before the
    // has_device early return.
    check_kernel_no_floats(&ast)?;

    if manifest.has_device {
        return generate_kernel_device(&ast, manifest, target, package_name);
    }

    let mut init_body = String::new();
    let mut exit_body = String::new();
    let mut has_printk = false;
    let modinfo = scan_modinfo(&ast);

    // Kernel-resource imports (`from rykernel_shim import ...`): resolved
    // against the shim's curated allowlist and lowered to direct calls into
    // the rykernel-shim crate, exactly like rust-modules imports in user
    // space. The import must be declared in the Python; calls to names that
    // were never imported are loud errors.
    let mut shim_imports: Vec<String> = Vec::new();
    for stmt in &ast.raw.body {
        match &stmt.statement {
            StatementType::ImportFrom(imp) if imp.module == "rykernel_shim" => {
                for alias in &imp.names {
                    let name = alias.name.clone();
                    if !KERNEL_SHIM_FNS.contains(&name.as_str()) {
                        return Err(anyhow::anyhow!(
                            "kernel module: `rykernel_shim.{name}` is not an importable \
                             kernel resource (available: {})",
                            KERNEL_SHIM_FNS.join(", ")
                        ));
                    }
                    if !shim_imports.contains(&name) {
                        shim_imports.push(name);
                    }
                }
            }
            StatementType::Import(imp) => {
                for alias in &imp.names {
                    if alias.name == "rykernel_shim" {
                        for name in KERNEL_SHIM_FNS {
                            if !shim_imports.contains(&name.to_string()) {
                                shim_imports.push((*name).to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if !shim_imports.is_empty() && target == KernelTarget::RustForLinux {
        return Err(anyhow::anyhow!(
            "kernel module: `rykernel_shim` imports are not supported with \
             --rust-for-linux — the raw-FFI target owns the shim; the \
             rust-for-linux target binds the `kernel` crate instead"
        ));
    }

    // Scan top-level statements for entry points.
    for stmt in &ast.raw.body {
        match &stmt.statement {
            StatementType::FunctionDef(func) => {
                let is_init = func.name == "module_init";
                let is_exit = func.name == "module_exit";
                if !is_init && !is_exit {
                    continue;
                }
                if !kernel_args_empty(&func.args) {
                    return Err(anyhow::anyhow!(
                        "kernel target: `{}` must take no parameters — the kernel calls \
                         module entry points with none",
                        func.name
                    ));
                }
                let body = lower_kernel_body(
                    &func.body,
                    &mut has_printk,
                    target,
                    is_init,
                    &func.name,
                    &shim_imports,
                )?;
                if is_init {
                    init_body = body;
                } else {
                    exit_body = body;
                }
            }
            _ => {}
        }
    }

    // Error if no entry point found.
    if init_body.is_empty() && exit_body.is_empty() {
        return Err(anyhow::anyhow!(
            "kernel module requires at least one of `module_init()` or `module_exit()`"
        ));
    }

    match target {
        KernelTarget::RawFfi => {
            generate_kernel_raw_ffi(
                &init_body,
                &exit_body,
                &modinfo,
                has_printk,
                !shim_imports.is_empty(),
            )
            .map(|code| GeneratedKernel {
                code,
                has_init: !init_body.is_empty(),
                has_exit: !exit_body.is_empty(),
                device: None,
                uses_shim: !shim_imports.is_empty(),
            })
        }
        KernelTarget::RustForLinux => {
            generate_kernel_rust_for_linux(&init_body, &exit_body, &modinfo, package_name).map(
                |code| GeneratedKernel {
                    code,
                    has_init: !init_body.is_empty(),
                    has_exit: !exit_body.is_empty(),
                    device: None,
                    uses_shim: false,
                },
            )
        }
    }
}

/// Generate a kernel module for a device-manifest Python: `src/device.rs`
/// (a misc byte-ring device parameterized by the manifest) and `src/lib.rs`
/// (modinfo + entry points that register/deregister it). The Python is
/// declarative here — only the manifest constants and modinfo metadata are
/// consumed; the driver logic stays in user space (UIO/vfio pattern).
fn generate_kernel_device(
    ast: &python_ast::Module,
    manifest: &DeviceManifest,
    target: KernelTarget,
    module_name: &str,
) -> Result<GeneratedKernel> {
    use python_ast::ast::tree::StatementType;

    if let KernelTarget::RustForLinux = target {
        bail!(
            "kernel device generation (from __device_name__ etc.) is not supported \
             with --rust-for-linux — device generation targets the raw-FFI pipeline"
        );
    }

    // The generated device owns the module entry points; a Python
    // module_init/module_exit would be silently dropped.
    for stmt in &ast.raw.body {
        if let StatementType::FunctionDef(func) = &stmt.statement {
            if func.name == "module_init" || func.name == "module_exit" {
                bail!(
                    "kernel device generation: `{}` conflicts with the generated device \
                     entry points — the misc device owns init/exit; remove it or drop \
                     __device_name__",
                    func.name
                );
            }
        }
    }

    let modinfo = scan_modinfo(ast);

    let lib = kernel_device_lib_template()
        .replace("@@MODINFO@@", &modinfo_statics(&modinfo))
        .replace("@@MODULE@@", module_name)
        .replace("@@DEVICE_NAME@@", &manifest.device_name);
    let device = kernel_device_template()
        .replace("@@DEVICE_NAME@@", &manifest.device_name)
        .replace("@@BUFSZ@@", &manifest.bufsz.to_string())
        .replace("@@MAGIC@@", &format!("{:X}", manifest.magic))
        .replace("@@IOC_RESET@@", &format!("{:x}", manifest.ioc_reset))
        .replace("@@IOC_STATS@@", &format!("{:x}", manifest.ioc_stats))
        .replace("@@DEVICE_MODE@@", &format!("{:o}", manifest.device_mode));

    Ok(GeneratedKernel {
        code: lib,
        has_init: true,
        has_exit: true,
        device: Some(device),
        uses_shim: false,
    })
}

/// The raw-FFI kernel module: `#![no_std]` staticlib with
/// `#[no_mangle] extern "C"` entry points, kmalloc-backed allocator,
/// `.modinfo` sections, and a panic handler.
fn generate_kernel_raw_ffi(
    init_body: &str,
    exit_body: &str,
    modinfo: &std::collections::BTreeMap<String, String>,
    has_printk: bool,
    uses_shim: bool,
) -> Result<String> {
    let printk_decl = if has_printk {
        // The kernel exports `_printk`, not `printk` — `printk` is a C macro
        // around it. Modules must call the symbol directly.
        "extern \"C\" {\n    fn _printk(fmt: *const core::ffi::c_char, ...);\n}\n\n"
    } else {
        ""
    };

    // Kernel resources imported from Python (`from rykernel_shim import
    // ktime_get_real_seconds`) resolve to fully-qualified calls into the
    // shim — the Rust compatibility layer that declares the kernel symbols
    // (the import itself is a compile-time declaration; rypip validated it
    // against the shim's allowlist, so no `use` is needed). The shim also
    // provides the panic handler, so the module must not define its own.
    let mut out = format!(
        "// Generated by rypip --kernel-module. Edit freely.\n\
         #![no_std]\n\n\
         extern crate alloc;\n\n\
         use core::ffi::c_int;\n\
         {printk_decl}"
    );

    // kmalloc-backed global allocator: allows String, Vec, HashMap, and the
    // full stdpython alloc tier to work in kernel context. Same pattern as
    // rust-for-linux's Kmalloc allocator. The exported symbol is
    // `__kmalloc_noprof` on kernels >= 7.0 (the kmalloc() inline in slab.h
    // lowers to it; the _noprof suffix comes from the allocation-profiling
    // rework). On 6.x kernels this was `kmalloc` — adjust if needed.
    out.push_str(
        "use core::alloc::{GlobalAlloc, Layout};\n\n\
         extern \"C\" {\n    fn __kmalloc_noprof(size: usize, flags: core::ffi::c_uint) -> *mut u8;\n    fn kfree(ptr: *mut u8);\n}\n\n\
         const GFP_KERNEL: core::ffi::c_uint = 0xCC0;\n\n\
         struct KernelAllocator;\n\n\
         unsafe impl GlobalAlloc for KernelAllocator {\n    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {\n        unsafe { __kmalloc_noprof(layout.size(), GFP_KERNEL) }\n    }\n    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {\n        unsafe { kfree(ptr) }\n    }\n}\n\n\
         #[global_allocator]\n\
         static ALLOCATOR: KernelAllocator = KernelAllocator;\n\n",
    );

    // Emit a .modinfo section entry for each metadata key-value pair
    // (see `modinfo_statics`).
    out.push_str(&modinfo_statics(modinfo));

    // Kernel panic handler. The rykernel-shim crate provides one when the
    // module links it (it prints the fault message through _printk); a
    // module that does not use the shim defines its own.
    if !uses_shim {
        out.push_str(
            "#[panic_handler]\n\
             fn panic(_info: &core::panic::PanicInfo) -> ! {\n    loop {}\n}\n\n",
        );
    }

    // module_init entry point.
    if !init_body.is_empty() {
        out.push_str(&format!(
            "#[no_mangle]\n\
             pub extern \"C\" fn init_module() -> c_int {{\n\
             {init_body}\
             }}\n\n"
        ));
    }

    // module_exit entry point.
    if !exit_body.is_empty() {
        out.push_str(&format!(
            "#[no_mangle]\n\
             pub extern \"C\" fn cleanup_module() {{\n\
             {exit_body}\
             }}\n"
        ));
    }

    Ok(out)
}

/// Stdlib modules whose public surface uses floating-point arithmetic.
/// Importing them in kernel context is a loud conversion error (issue #87):
/// the kernel runs with the FPU in a lazy-save state, and executing FP
/// instructions without `kernel_fpu_begin()`/`kernel_fpu_end()` guards can
/// corrupt userspace FPU state and cause silent data corruption.
const KERNEL_FP_STDLIB: &[&str] = &[
    "cmath",
    "datetime",
    "decimal",
    "fractions",
    "math",
    "random",
    "statistics",
];

/// Build the loud conversion error for floating-point usage in kernel code.
fn kernel_fp_err(what: &str, line: Option<usize>) -> anyhow::Error {
    let at = line.map(|l| format!(" (line {l})")).unwrap_or_default();
    anyhow::anyhow!(
        "kernel target forbids floating-point{at}: {what}. The kernel runs with the \
         FPU in a lazy-save state; use integer or fixed-point math, or guard genuine \
         FP work with kernel_fpu_begin()/kernel_fpu_end()"
    )
}

/// Scan the whole module for floating-point usage and reject it loudly
/// (issue #87). Covers float literals in any expression, `float` type
/// annotations, `float()` calls, and imports of float-using stdlib modules —
/// including inside statements the kernel lowering does not otherwise touch,
/// so nothing is silently dropped.
fn check_kernel_no_floats(ast: &python_ast::Module) -> Result<()> {
    for stmt in &ast.raw.body {
        check_kernel_stmt_no_floats(stmt)?;
    }
    Ok(())
}

fn check_kernel_stmt_no_floats(stmt: &python_ast::Statement) -> Result<()> {
    use python_ast::ast::tree::StatementType as S;
    let line = stmt.lineno;
    match &stmt.statement {
        S::Import(imp) => {
            for alias in &imp.names {
                let root = alias.name.split('.').next().unwrap_or("");
                if KERNEL_FP_STDLIB.contains(&root) {
                    return Err(kernel_fp_err(
                        &format!(
                            "import of `{}` (a floating-point stdlib module)",
                            alias.name
                        ),
                        line,
                    ));
                }
            }
            Ok(())
        }
        S::ImportFrom(imp) => {
            let root = imp.module.split('.').next().unwrap_or("");
            if KERNEL_FP_STDLIB.contains(&root) {
                return Err(kernel_fp_err(
                    &format!(
                        "import of `{}` (a floating-point stdlib module)",
                        imp.module
                    ),
                    line,
                ));
            }
            Ok(())
        }
        S::FunctionDef(func) | S::AsyncFunctionDef(func) => {
            check_kernel_funcdef_no_floats(func, line)
        }
        S::ClassDef(class) => {
            for s in &class.body {
                check_kernel_stmt_no_floats(s)?;
            }
            Ok(())
        }
        S::Assign(assign) => {
            check_kernel_expr_no_floats(&assign.value, line)?;
            for target in &assign.targets {
                check_kernel_expr_no_floats(target, line)?;
            }
            Ok(())
        }
        S::AugAssign(aug) => {
            check_kernel_expr_no_floats(&aug.target, line)?;
            check_kernel_expr_no_floats(&aug.value, line)
        }
        S::Call(call) => check_kernel_call_no_floats(call, line),
        S::Expr(expr) => check_kernel_expr_no_floats(&expr.value, line),
        S::Return(Some(val)) => check_kernel_expr_no_floats(&val.value, line),
        S::Return(None) | S::Pass | S::Break | S::Continue => Ok(()),
        S::Raise(r) => {
            if let Some(exc) = &r.exc {
                check_kernel_expr_no_floats(exc, line)?;
            }
            if let Some(cause) = &r.cause {
                check_kernel_expr_no_floats(cause, line)?;
            }
            Ok(())
        }
        S::Assert { test, msg } => {
            check_kernel_expr_no_floats(test, line)?;
            if let Some(m) = msg {
                check_kernel_expr_no_floats(m, line)?;
            }
            Ok(())
        }
        S::Global(_) => Ok(()),
        S::Nonlocal(_) => Ok(()),
        // A bare annotated declaration (`x: int`) has no value to scan.
        S::AnnotatedName { .. } => Ok(()),
        S::Delete(targets) => {
            for t in targets {
                check_kernel_expr_no_floats(t, line)?;
            }
            Ok(())
        }
        S::If(i) => {
            check_kernel_expr_no_floats(&i.test, line)?;
            for s in i.body.iter().chain(&i.orelse) {
                check_kernel_stmt_no_floats(s)?;
            }
            Ok(())
        }
        S::For(f) => {
            check_kernel_expr_no_floats(&f.target, line)?;
            check_kernel_expr_no_floats(&f.iter, line)?;
            for s in f.body.iter().chain(&f.orelse) {
                check_kernel_stmt_no_floats(s)?;
            }
            Ok(())
        }
        S::While(w) => {
            check_kernel_expr_no_floats(&w.test, line)?;
            for s in w.body.iter().chain(&w.orelse) {
                check_kernel_stmt_no_floats(s)?;
            }
            Ok(())
        }
        S::AsyncFor(f) => {
            check_kernel_expr_no_floats(&f.target, line)?;
            check_kernel_expr_no_floats(&f.iter, line)?;
            for s in f.body.iter().chain(&f.orelse) {
                check_kernel_stmt_no_floats(s)?;
            }
            Ok(())
        }
        S::Try(t) => {
            for s in t.body.iter().chain(&t.orelse).chain(&t.finalbody) {
                check_kernel_stmt_no_floats(s)?;
            }
            for handler in &t.handlers {
                if let Some(et) = &handler.exception_type {
                    check_kernel_expr_no_floats(et, line)?;
                }
                for s in &handler.body {
                    check_kernel_stmt_no_floats(s)?;
                }
            }
            Ok(())
        }
        S::With(w) => {
            check_kernel_with_items_no_floats(&w.items, line)?;
            for s in &w.body {
                check_kernel_stmt_no_floats(s)?;
            }
            Ok(())
        }
        S::AsyncWith(w) => {
            check_kernel_with_items_no_floats(&w.items, line)?;
            for s in &w.body {
                check_kernel_stmt_no_floats(s)?;
            }
            Ok(())
        }
        S::Unimplemented(_) => Ok(()),
    }
}

fn check_kernel_with_items_no_floats(
    items: &[python_ast::WithItem],
    line: Option<usize>,
) -> Result<()> {
    for item in items {
        check_kernel_expr_no_floats(&item.context_expr, line)?;
        if let Some(vars) = &item.optional_vars {
            check_kernel_expr_no_floats(vars, line)?;
        }
    }
    Ok(())
}

fn check_kernel_funcdef_no_floats(
    func: &python_ast::FunctionDef,
    line: Option<usize>,
) -> Result<()> {
    check_kernel_args_no_floats(&func.args, line)?;
    if let Some(returns) = &func.returns {
        if annotation_mentions_float(returns) {
            return Err(kernel_fp_err("`float` return annotation", line));
        }
    }
    for dec in &func.decorator_list {
        check_kernel_expr_no_floats(dec, line)?;
    }
    for s in &func.body {
        check_kernel_stmt_no_floats(s)?;
    }
    Ok(())
}

fn check_kernel_args_no_floats(
    args: &python_ast::ParameterList,
    line: Option<usize>,
) -> Result<()> {
    for p in args
        .posonlyargs
        .iter()
        .chain(&args.args)
        .chain(&args.kwonlyargs)
    {
        if let Some(ann) = &p.annotation {
            if annotation_mentions_float(ann) {
                return Err(kernel_fp_err(
                    &format!("`float` annotation on parameter `{}`", p.arg),
                    line,
                ));
            }
        }
    }
    for p in [&args.vararg, &args.kwarg].into_iter().flatten() {
        if let Some(ann) = &p.annotation {
            if annotation_mentions_float(ann) {
                return Err(kernel_fp_err(
                    &format!("`float` annotation on parameter `{}`", p.arg),
                    line,
                ));
            }
        }
    }
    for d in args.defaults.iter() {
        check_kernel_expr_no_floats(d, line)?;
    }
    for d in args.kw_defaults.iter().flatten() {
        check_kernel_expr_no_floats(d, line)?;
    }
    Ok(())
}

/// Does this annotation expression mention the `float` type — bare, or
/// nested in a subscript such as `list[float]`?
fn annotation_mentions_float(expr: &python_ast::ExprType) -> bool {
    match expr {
        python_ast::ExprType::Name(n) => n.id == "float",
        python_ast::ExprType::Subscript(s) => {
            annotation_mentions_float(&s.value)
                || match &s.kind {
                    python_ast::SubscriptKind::Index(index) => annotation_mentions_float(index),
                    python_ast::SubscriptKind::Slice { lower, upper, step } => {
                        lower.as_deref().is_some_and(annotation_mentions_float)
                            || upper.as_deref().is_some_and(annotation_mentions_float)
                            || step.as_deref().is_some_and(annotation_mentions_float)
                    }
                }
        }
        python_ast::ExprType::Attribute(a) => annotation_mentions_float(&a.value),
        _ => false,
    }
}

/// Walk an expression for float literals and `float()` calls.
fn check_kernel_expr_no_floats(expr: &python_ast::ExprType, line: Option<usize>) -> Result<()> {
    use python_ast::ExprType as E;
    match expr {
        E::Constant(c) => {
            if matches!(c.0, Some(litrs::Literal::Float(_))) {
                return Err(kernel_fp_err(
                    &format!("float literal `{}`", c.to_string()),
                    line,
                ));
            }
            Ok(())
        }
        E::BoolOp(b) => b
            .values
            .iter()
            .try_for_each(|v| check_kernel_expr_no_floats(v, line)),
        E::NamedExpr(n) => {
            check_kernel_expr_no_floats(&n.left, line)?;
            check_kernel_expr_no_floats(&n.right, line)
        }
        E::BinOp(b) => {
            check_kernel_expr_no_floats(&b.left, line)?;
            check_kernel_expr_no_floats(&b.right, line)
        }
        E::UnaryOp(u) => check_kernel_expr_no_floats(&u.operand, line),
        E::Lambda(l) => {
            check_kernel_args_no_floats(&l.args, line)?;
            check_kernel_expr_no_floats(&l.body, line)
        }
        E::IfExp(i) => {
            check_kernel_expr_no_floats(&i.test, line)?;
            check_kernel_expr_no_floats(&i.body, line)?;
            check_kernel_expr_no_floats(&i.orelse, line)
        }
        E::Dict(d) => {
            for k in d.keys.iter().flatten() {
                check_kernel_expr_no_floats(k, line)?;
            }
            for v in &d.values {
                check_kernel_expr_no_floats(v, line)?;
            }
            Ok(())
        }
        E::Set(s) => s
            .elts
            .iter()
            .try_for_each(|e| check_kernel_expr_no_floats(e, line)),
        E::ListComp(lc) => check_kernel_comprehension_no_floats(&lc.elt, &lc.generators, line),
        E::SetComp(sc) => check_kernel_comprehension_no_floats(&sc.elt, &sc.generators, line),
        E::GeneratorExp(ge) => check_kernel_comprehension_no_floats(&ge.elt, &ge.generators, line),
        E::DictComp(dc) => {
            check_kernel_expr_no_floats(&dc.key, line)?;
            check_kernel_expr_no_floats(&dc.value, line)?;
            for g in &dc.generators {
                check_kernel_comprehension_gen_no_floats(g, line)?;
            }
            Ok(())
        }
        E::Await(a) => check_kernel_expr_no_floats(&a.value, line),
        E::Yield(y) => {
            if let Some(v) = &y.value {
                check_kernel_expr_no_floats(v, line)?;
            }
            Ok(())
        }
        E::YieldFrom(y) => check_kernel_expr_no_floats(&y.value, line),
        E::Compare(c) => {
            check_kernel_expr_no_floats(&c.left, line)?;
            c.comparators
                .iter()
                .try_for_each(|v| check_kernel_expr_no_floats(v, line))
        }
        E::Call(call) => check_kernel_call_no_floats(call, line),
        E::FormattedValue(fv) => check_kernel_expr_no_floats(&fv.value, line),
        E::JoinedStr(js) => js
            .values
            .iter()
            .try_for_each(|v| check_kernel_expr_no_floats(v, line)),
        E::Attribute(a) => check_kernel_expr_no_floats(&a.value, line),
        E::Subscript(s) => {
            check_kernel_expr_no_floats(&s.value, line)?;
            match &s.kind {
                python_ast::SubscriptKind::Index(index) => check_kernel_expr_no_floats(index, line),
                python_ast::SubscriptKind::Slice { lower, upper, step } => {
                    for part in [lower, upper, step].into_iter().flatten() {
                        check_kernel_expr_no_floats(part, line)?;
                    }
                    Ok(())
                }
            }
        }
        E::Starred(s) => check_kernel_expr_no_floats(&s.value, line),
        E::List(items) => items
            .iter()
            .try_for_each(|e| check_kernel_expr_no_floats(e, line)),
        E::Tuple(t) => t
            .elts
            .iter()
            .try_for_each(|e| check_kernel_expr_no_floats(e, line)),
        E::Name(_) | E::NoneType(_) | E::Unimplemented(_) | E::Unknown => Ok(()),
    }
}

fn check_kernel_call_no_floats(call: &python_ast::Call, line: Option<usize>) -> Result<()> {
    if let python_ast::ExprType::Name(n) = call.func.as_ref() {
        if n.id == "float" {
            return Err(kernel_fp_err("call to `float()`", line));
        }
    }
    check_kernel_expr_no_floats(&call.func, line)?;
    for arg in &call.args {
        check_kernel_expr_no_floats(arg, line)?;
    }
    for kw in &call.keywords {
        check_kernel_expr_no_floats(&kw.value, line)?;
    }
    Ok(())
}

fn check_kernel_comprehension_no_floats(
    elt: &python_ast::ExprType,
    generators: &[python_ast::Comprehension],
    line: Option<usize>,
) -> Result<()> {
    check_kernel_expr_no_floats(elt, line)?;
    for g in generators {
        check_kernel_comprehension_gen_no_floats(g, line)?;
    }
    Ok(())
}

fn check_kernel_comprehension_gen_no_floats(
    g: &python_ast::Comprehension,
    line: Option<usize>,
) -> Result<()> {
    check_kernel_expr_no_floats(&g.target, line)?;
    check_kernel_expr_no_floats(&g.iter, line)?;
    for c in &g.ifs {
        check_kernel_expr_no_floats(c, line)?;
    }
    Ok(())
}

/// Does an entry-function parameter list declare any parameters?
fn kernel_args_empty(args: &python_ast::ParameterList) -> bool {
    args.posonlyargs.is_empty()
        && args.args.is_empty()
        && args.vararg.is_none()
        && args.kwonlyargs.is_empty()
        && args.kwarg.is_none()
}

/// The rust-for-linux kernel module: a `module!`-macro crate implementing
/// `kernel::Module`. The module! macro emits the .modinfo metadata, so no
/// manual sections are needed; printk lowers to pr_info!, and module_init /
/// module_exit map to `init` / `Drop`.
fn generate_kernel_rust_for_linux(
    init_body: &str,
    exit_body: &str,
    modinfo: &std::collections::BTreeMap<String, String>,
    package_name: &str,
) -> Result<String> {
    let license = modinfo
        .get("license")
        .cloned()
        .unwrap_or_else(|| "GPL".to_string());
    let author = modinfo.get("author").cloned().unwrap_or_default();
    let description = modinfo.get("description").cloned().unwrap_or_default();
    let ko_name = package_name;
    let type_name = kernel_type_name(package_name);

    // The module! macro requires type/name/license; author/description are
    // optional but nice to have. module! emits .modinfo itself.
    let mut meta = format!("module! {{\n    type: {type_name},\n    name: \"{ko_name}\",\n");
    if !author.is_empty() {
        meta.push_str(&format!("    author: \"{author}\",\n"));
    }
    if !description.is_empty() {
        meta.push_str(&format!("    description: \"{description}\",\n"));
    }
    meta.push_str(&format!("    license: \"{license}\",\n}}\n"));

    let indent = |body: &str| -> String {
        // lower_kernel_body emits 4-space-indented lines; nest them one
        // more level inside fn init / fn drop.
        body.lines().map(|l| format!("    {l}\n")).collect()
    };

    let mut out = format!(
        "// Generated by rypip --kernel-module --rust-for-linux. Edit freely.\n\
         //\n\
         // Build with the rust-for-linux toolchain inside a rust-for-linux\n\
         // kernel tree (CONFIG_RUST=y): copy this crate into the tree, add\n\
         // `rust-obj-m += {ko_name}.o` to the Kbuild Makefile, and build the\n\
         // kernel. The `kernel` crate (path dep in Cargo.toml) provides the\n\
         // module! macro and kernel::prelude.\n\
         #![no_std]\n\
         #![no_main]\n\n\
         use kernel::prelude::*;\n\n\
         {meta}\n\
         struct {type_name};\n\n"
    );

    if !init_body.is_empty() {
        out.push_str(&format!(
            "impl kernel::Module for {type_name} {{\n\
             \x20   fn init(_module: &'static ThisModule) -> Result<Self> {{\n\
             {}\x20   \x20   Ok({type_name})\n\
             \x20   }}\n\
             }}\n\n",
            indent(init_body)
        ));
    }

    if !exit_body.is_empty() {
        out.push_str(&format!(
            "impl Drop for {type_name} {{\n\
             \x20   fn drop(&mut self) {{\n\
             {}\x20   }}\n\
             }}\n",
            indent(exit_body)
        ));
    }

    Ok(out)
}

/// The module skeleton for a generated device: modinfo statics, the
/// `mod device;` declaration, and entry points that register/deregister
/// the misc device. `@@MODINFO@@` is replaced with `modinfo_statics`
/// output, `@@MODULE@@` with the module name, `@@DEVICE_NAME@@` with the
/// misc device name.
fn kernel_device_lib_template() -> &'static str {
    r#"// Generated by rypip --kernel-module (device manifest). Edit the Python,
// not this file: the device constants come from __device_name__,
// __bufsz__, __magic__, __device_mode__, __ioc_reset__, __ioc_stats__ and
// the modinfo entries from __module_license__ etc. in the driver Python.
#![no_std]

mod device;

use core::ffi::c_int;

use rykernel_shim::printk;

@@MODINFO@@
// ---------------------------------------------------------------------------
// module entry points — owned by the generated device
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn init_module() -> c_int {
    match device::register() {
        Ok(()) => {
            printk!(
                "@@MODULE@@: loaded — /dev/@@DEVICE_NAME@@ ready (magic 0x%lx)\n",
                device::MAGIC as u64
            );
            0
        }
        Err(err) => {
            printk!("@@MODULE@@: misc_register failed: %ld\n", err as i64);
            err
        }
    }
}

#[no_mangle]
pub extern "C" fn cleanup_module() {
    device::deregister();
    printk!("@@MODULE@@: unloaded\n");
}
"#
}

/// The generated misc device — a byte ring with transaction counters and a
/// reset ioctl, parameterized by the device manifest. Deliberately
/// contains NO driver logic: it is a dumb byte pipe, an emulated MMIO
/// region with no hardware behind it. All semantics live in the driver
/// Python, compiled to Rust and run in user space against this device
/// (the UIO/vfio pattern).
fn kernel_device_template() -> &'static str {
    r#"//! The kernel-side "device" — generated by rypip --kernel-module from the
//! device manifest in the driver Python (__device_name__, __bufsz__,
//! __magic__, __device_mode__, __ioc_reset__, __ioc_stats__). Edit those,
//! not this file.
//!
//! A byte ring with transaction counters and a reset ioctl, exposed as the
//! misc device /dev/@@DEVICE_NAME@@.
//!
//! This device deliberately contains NO driver logic: it is a dumb byte
//! pipe with counters, an emulated MMIO region with no hardware behind it.
//! All semantics live in the driver Python, which rython compiles to Rust
//! and runs in user space against this device.
//!
//! The fops/ioctl numbers below are the ABI the user-space driver speaks
//! (rypip --driver emits the syscall glue from the same manifest):
//!   - write(2) — append up to @@BUFSZ@@ bytes to the ring, dropping the
//!     oldest bytes when full
//!   - read(2)  — drain bytes from the ring
//!   - ioctl 0x@@IOC_RESET@@ (RYTHON_IOC_RESET) — clear ring, keep byte
//!     counters, bump the reset counter
//!   - ioctl 0x@@IOC_STATS@@ (RYTHON_IOC_STATS) — return `Stats`
//!
//! Locking: a spinlock serialises ring access. `copy_to_user` /
//! `copy_from_user` are always called with the lock released — they can
//! page-fault, and sleeping while holding a spinlock is a kernel bug.

use core::ffi::{c_int, c_long, c_uint, c_ulong, c_void};
use core::mem::size_of;
use core::cell::UnsafeCell;

use rykernel_shim::device::{
    copy_from_user, copy_to_user, File, FileOperations, MiscDevice,
};
use rykernel_shim::ffi::{EFAULT, ENOTTY};
use rykernel_shim::sync::SpinLock;

/// Ring capacity — must match the user-space driver's expectations.
pub const BUFSZ: usize = @@BUFSZ@@;
/// Device magic (e.g. "RYHT"), echoed in the load message.
pub const MAGIC: u32 = 0x@@MAGIC@@;

/// `_IO(RYTHON_IOC_MAGIC, 1)` — clear the ring.
const IOC_RESET: c_uint = 0x@@IOC_RESET@@;
/// `_IOR(RYTHON_IOC_MAGIC, 2, struct rython_stats)` — fetch counters.
const IOC_STATS: c_uint = 0x@@IOC_STATS@@;

/// ioctl stats record, `struct rython_stats` (40 bytes).
#[repr(C)]
struct Stats {
    wr_bytes: u64,
    rd_bytes: u64,
    wr_calls: u64,
    rd_calls: u64,
    resets: u64,
}

/// Device state: the emulated "device memory" plus transaction counters.
struct Dev {
    buf: [u8; BUFSZ],
    head: usize,
    tail: usize,
    wr_bytes: u64,
    rd_bytes: u64,
    wr_calls: u64,
    rd_calls: u64,
    resets: u64,
}

const fn dev_init() -> Dev {
    Dev {
        buf: [0; BUFSZ],
        head: 0,
        tail: 0,
        wr_bytes: 0,
        rd_bytes: 0,
        wr_calls: 0,
        rd_calls: 0,
        resets: 0,
    }
}

/// `&mut Dev` access is exclusive under `LOCK`; the cell makes that
/// discipline explicit instead of relying on `static mut`.
struct SyncDev(UnsafeCell<Dev>);
unsafe impl Sync for SyncDev {}

static LOCK: SpinLock = SpinLock::new();
static DEV: SyncDev = SyncDev(UnsafeCell::new(dev_init()));

/// Run `f` with the device state locked.
fn with_dev<R>(f: impl FnOnce(&mut Dev) -> R) -> R {
    let _guard = LOCK.lock();
    f(unsafe { &mut *DEV.0.get() })
}

// ---------------------------------------------------------------------------
// file operations
// ---------------------------------------------------------------------------

/// Zero `len` bytes with volatile stores.
///
/// Deliberately volatile: a plain `*b = 0` loop (or an array literal) is
/// recognised by LLVM and emitted as a `memset` libcall, and references to
/// the interposable `memset` symbol are GOTPCREL relocations that the kernel
/// module loader rejects. Volatile stores are not fused, so no libcall is
/// emitted.
unsafe fn zero_volatile(p: *mut u8, len: usize) {
    for i in 0..len {
        unsafe { core::ptr::write_volatile(p.add(i), 0u8) };
    }
}

unsafe extern "C" fn dev_read(
    _file: *mut File,
    ubuf: *mut c_void,
    count: usize,
    _ppos: *mut i64,
) -> isize {
    // Drain the ring into the staging area (the ring buffer itself) under
    // the lock, then copy out with the lock released.
    let n = with_dev(|dev| {
        let mut n = 0usize;
        while n < count && dev.tail != dev.head {
            // SAFETY: `n < BUFSZ` — the ring holds at most BUFSZ-1 bytes, so
            // the drain loop runs at most BUFSZ times; `dev.tail < BUFSZ` is
            // maintained by the modulo below. No panics may reachable code
            // (panic formatting pulls GOTPCREL relocations the module loader
            // rejects), so these are unchecked on purpose.
            unsafe {
                *dev.buf.get_unchecked_mut(n) = *dev.buf.get_unchecked(dev.tail);
            }
            n += 1;
            dev.tail = (dev.tail + 1) % BUFSZ;
        }
        dev.rd_bytes += n as u64;
        if n > 0 {
            dev.rd_calls += 1;
        }
        n
    });
    if n > 0 {
        if let Err(e) = unsafe { copy_to_user(ubuf, dev_buf_ptr(), n) } {
            return e as isize;
        }
    }
    n as isize
}

fn dev_buf_ptr() -> *const u8 {
    // Read-only peek at the staging bytes just written; called after the
    // lock is released, exactly like the C shim's post-unlock copy.
    unsafe { &(*DEV.0.get()).buf as *const [u8; BUFSZ] as *const u8 }
}

unsafe extern "C" fn dev_write(
    _file: *mut File,
    ubuf: *const c_void,
    count: usize,
    _ppos: *mut i64,
) -> isize {
    if count == 0 {
        return 0;
    }
    let count = count.min(BUFSZ);
    // Copy the user bytes into a stack staging buffer first — copy_from_user
    // may fault, so it must not run under the spinlock. (MaybeUninit + an
    // explicit volatile zero: the `[0u8; N]` literal would lower to a memset
    // libcall, whose GOTPCREL relocation the module loader rejects.)
    let mut kbuf = core::mem::MaybeUninit::<[u8; BUFSZ]>::uninit();
    let kbuf_ptr = kbuf.as_mut_ptr() as *mut u8;
    unsafe { zero_volatile(kbuf_ptr, BUFSZ) };
    if let Err(e) = unsafe { copy_from_user(kbuf_ptr, ubuf, count) } {
        return e as isize;
    }
    with_dev(|dev| {
        for i in 0..count {
            // SAFETY: `i < count <= BUFSZ` (count was clamped above);
            // `dev.head < BUFSZ` is maintained by the modulo below. Unchecked
            // on purpose — see the note in dev_read.
            unsafe {
                *dev.buf.get_unchecked_mut(dev.head) = *kbuf_ptr.add(i);
            }
            dev.head = (dev.head + 1) % BUFSZ;
            if dev.head == dev.tail {
                // Full: drop the oldest byte.
                dev.tail = (dev.tail + 1) % BUFSZ;
            }
        }
        dev.wr_bytes += count as u64;
        dev.wr_calls += 1;
    });
    count as isize
}

unsafe extern "C" fn dev_ioctl(
    _file: *mut File,
    cmd: c_uint,
    arg: c_ulong,
) -> c_long {
    match cmd {
        IOC_RESET => {
            with_dev(|dev| {
                unsafe { zero_volatile(dev.buf.as_mut_ptr(), BUFSZ) };
                dev.head = 0;
                dev.tail = 0;
                dev.resets += 1;
            });
            0
        }
        IOC_STATS => {
            let st = with_dev(|dev| Stats {
                wr_bytes: dev.wr_bytes,
                rd_bytes: dev.rd_bytes,
                wr_calls: dev.wr_calls,
                rd_calls: dev.rd_calls,
                resets: dev.resets,
            });
            let ret = unsafe { copy_to_user(arg as *mut c_void, &st as *const Stats as *const u8, size_of::<Stats>()) };
            if ret.is_err() {
                EFAULT as c_long
            } else {
                0
            }
        }
        _ => ENOTTY as c_long,
    }
}

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------

static FOPS: FileOperations = FileOperations {
    owner: unsafe { &rykernel_shim::__this_module },
    read: Some(dev_read),
    write: Some(dev_write),
    unlocked_ioctl: Some(dev_ioctl),
    ..FileOperations::empty()
};

/// The registered device record. `mode 0600`: root-only, like the C shim;
/// install the udev rule to open it unprivileged.
///
/// Wrapped in `UnsafeCell`: `misc_register` initialises the embedded
/// `list_head` (`INIT_LIST_HEAD(&misc->list)`), so the record must live in
/// writable memory — a plain `static` would be placed in read-only
/// `.data.rel.ro` and fault on that write.
struct SyncDevice(UnsafeCell<MiscDevice>);
unsafe impl Sync for SyncDevice {}

static DEVICE: SyncDevice =
    SyncDevice(UnsafeCell::new(MiscDevice::new(b"@@DEVICE_NAME@@\0", &FOPS, 0o@@DEVICE_MODE@@)));

pub fn register() -> Result<(), c_int> {
    let dev: &'static MiscDevice = unsafe { &*DEVICE.0.get() };
    dev.register()
}

pub fn deregister() {
    let dev: &'static MiscDevice = unsafe { &*DEVICE.0.get() };
    dev.deregister();
}
"#
}

/// CamelCase a crate name into a Rust type name: `hello_kernel` ->
/// `HelloKernel`. Falls back to `K`-prefixed names for identifiers that
/// would not be valid Rust type names.
fn kernel_type_name(package_name: &str) -> String {
    let mut out = String::new();
    let mut upper = true;
    for ch in package_name.chars() {
        if ch.is_ascii_alphanumeric() {
            if upper {
                out.extend(ch.to_uppercase());
                upper = false;
            } else {
                out.push(ch);
            }
        } else {
            upper = true;
        }
    }
    if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
        format!("K{out}")
    } else {
        out
    }
}

/// Try to extract a string literal from an expression.
fn expr_str_literal(expr: &python_ast::ExprType) -> Option<String> {
    if let python_ast::ExprType::Constant(c) = expr {
        if let Some(litrs::Literal::String(slit)) = &c.0 {
            return Some(slit.value().to_string());
        }
    }
    None
}

/// Try to extract an integer literal from an expression.
fn expr_int_literal(expr: &python_ast::ExprType) -> Option<i64> {
    if let python_ast::ExprType::Constant(c) = expr {
        if let Some(litrs::Literal::Integer(ilit)) = &c.0 {
            return ilit.value::<i64>();
        }
    }
    None
}

/// Lower a kernel-function body: printk calls, integer-literal local
/// bindings, and return statements. Anything the kernel ABI cannot express
/// is a loud conversion error — never silently dropped (issue #84).
fn lower_kernel_body(
    body: &[python_ast::ast::tree::Statement],
    has_printk: &mut bool,
    target: KernelTarget,
    is_init: bool,
    func_name: &str,
    shim_imports: &[String],
) -> Result<String> {
    let mut out = String::new();
    for stmt in body {
        let line = stmt.lineno;
        match &stmt.statement {
            python_ast::StatementType::Expr(expr) => {
                // Docstrings: the canonical kernel module opens module_init
                // with a docstring (issue #83's example). They carry no
                // runtime meaning — drop them like CPython does.
                if expr_str_literal(&expr.value).is_some() {
                    continue;
                }
                let Some(call) = extract_kernel_call(expr) else {
                    return Err(kernel_body_err(
                        "unsupported expression statement (only printk(...) calls are supported)",
                        line,
                    ));
                };
                if call.name == "printk" {
                    if !call.keywords.is_empty() {
                        return Err(kernel_body_err(
                            "printk takes positional arguments only",
                            line,
                        ));
                    }
                    *has_printk = true;
                    let style = printk_style(target);
                    let (fmt, values) = lower_printk_args(call.args, line, style)?;
                    match target {
                        KernelTarget::RawFfi => {
                            out.push_str("    unsafe {\n        _printk(");
                            out.push_str(&format!(
                                "b\"{fmt}\\0\".as_ptr() as *const core::ffi::c_char"
                            ));
                            if !values.is_empty() {
                                out.push_str(&format!(", {}", values.join(", ")));
                            }
                            out.push_str(");\n    }\n");
                        }
                        KernelTarget::RustForLinux => {
                            out.push_str(&format!("    pr_info!(\"{fmt}\""));
                            if !values.is_empty() {
                                out.push_str(&format!(", {}", values.join(", ")));
                            }
                            out.push_str(");\n");
                        }
                    }
                    continue;
                }
                // A bare call to an imported kernel resource (e.g.
                // `ktime_get_real_seconds()` on its own line): discard the
                // value, matching Python's statement semantics.
                if shim_imports.contains(&call.name) {
                    if !call.keywords.is_empty() {
                        return Err(kernel_body_err(
                            &format!("`{}()` takes positional arguments only", call.name),
                            line,
                        ));
                    }
                    let args = lower_kernel_call_args(call.args, line)?;
                    out.push_str(&format!(
                        "    let _ = rykernel_shim::{}({});\n",
                        safe_ident(&call.name),
                        args.join(", ")
                    ));
                    continue;
                }
                return Err(kernel_body_err(
                    &format!(
                        "unsupported call `{}()` (only printk and imported rykernel_shim \
                         resources are available in kernel modules)",
                        call.name
                    ),
                    line,
                ));
            }
            python_ast::StatementType::Assign(assign) => {
                // Integer-literal local binding, e.g. `addr = 0x1fff0000`.
                // These are the only locals the kernel ABI supports; they
                // exist so f-string printk interpolations have values to
                // reference.
                let target_is_name =
                    matches!(assign.targets.first(), Some(python_ast::ExprType::Name(_)));
                if assign.targets.len() == 1 && target_is_name {
                    if let python_ast::ExprType::Constant(c) = &assign.value {
                        if let Some(litrs::Literal::Integer(ilit)) = &c.0 {
                            if let Some(val) = ilit.value::<i64>() {
                                if let python_ast::ExprType::Name(name) = &assign.targets[0] {
                                    out.push_str(&format!(
                                        "    let {}: i64 = {val};\n",
                                        safe_ident(&name.id)
                                    ));
                                    continue;
                                }
                            }
                        }
                    }
                    // Kernel-resource call binding: `now =
                    // ktime_get_real_seconds()` lowers to a direct call
                    // into the shim, with the value kept as an i64 local
                    // that printk interpolations can reference.
                    if let python_ast::ExprType::Call(call) = &assign.value {
                        if let python_ast::ExprType::Name(fname) = call.func.as_ref() {
                            if shim_imports.contains(&fname.id) {
                                if !call.keywords.is_empty() {
                                    return Err(kernel_body_err(
                                        &format!(
                                            "`{}()` takes positional arguments only",
                                            fname.id
                                        ),
                                        line,
                                    ));
                                }
                                let args = lower_kernel_call_args(&call.args, line)?;
                                if let python_ast::ExprType::Name(name) = &assign.targets[0] {
                                    out.push_str(&format!(
                                        "    let {}: i64 = rykernel_shim::{}({});\n",
                                        safe_ident(&name.id),
                                        safe_ident(&fname.id),
                                        args.join(", ")
                                    ));
                                    continue;
                                }
                            }
                        }
                    }
                }
                return Err(kernel_body_err(
                    "unsupported assignment (only `name = <integer literal>` and \
                     `name = <imported rykernel_shim call>` are supported)",
                    line,
                ));
            }
            python_ast::StatementType::Return(Some(val)) => {
                match target {
                    // Raw init returns an errno to the kernel; any literal
                    // works, though 0 (success) is the normal case.
                    KernelTarget::RawFfi if is_init => out.push_str(&format!(
                        "    return {};\n",
                        lower_kernel_expr(&val.value, line)?
                    )),
                    // rust-for-linux init maps the Python `return 0` to
                    // `Ok(ModuleType)`; other return values have no honest
                    // mapping to kernel::error::Error yet.
                    KernelTarget::RustForLinux if is_init => {
                        let is_zero = matches!(
                            &val.value,
                            python_ast::ExprType::Constant(c)
                                if matches!(
                                    &c.0,
                                    Some(litrs::Literal::Integer(ilit))
                                        if ilit.value::<i64>() == Some(0)
                                )
                        );
                        if !is_zero {
                            return Err(kernel_body_err(
                                "rust-for-linux module_init can only `return 0` (success); \
                                 error numbers have no mapping to kernel::error::Error yet",
                                line,
                            ));
                        }
                        // Success: nothing here; `Ok(<Type>)` is emitted by
                        // the module skeleton.
                    }
                    _ => {
                        return Err(kernel_body_err(
                            &format!(
                                "`{func_name}` cannot return a value — the kernel ABI gives \
                                 cleanup_module()/Drop::drop a void return"
                            ),
                            line,
                        ));
                    }
                }
            }
            python_ast::StatementType::Return(None) => {
                if target == KernelTarget::RawFfi {
                    out.push_str("    return;\n");
                }
                // rust-for-linux: bare `return` is a no-op in init (Ok is
                // emitted by the skeleton) and in Drop.
            }
            python_ast::StatementType::Pass => {}
            _ => {
                return Err(kernel_body_err(
                    "unsupported statement (only printk calls, integer-literal \
                     assignments, pass, and return are supported)",
                    line,
                ));
            }
        }
    }
    Ok(out)
}

/// Build a kernel-body lowering error with the source line when known.
fn kernel_body_err(what: &str, line: Option<usize>) -> anyhow::Error {
    let at = line.map(|l| format!(" (line {l})")).unwrap_or_default();
    anyhow::anyhow!("kernel target{at}: {what}")
}

/// Which format-string dialect printk lowers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrintkStyle {
    /// The kernel's C format parser (raw FFI): `%ld` conversions, `%`
    /// doubled for literal percent signs.
    C,
    /// Rust's `format!` parser (pr_info!): `{}` conversions, braces doubled.
    RustFmt,
}

fn printk_style(target: KernelTarget) -> PrintkStyle {
    match target {
        KernelTarget::RawFfi => PrintkStyle::C,
        KernelTarget::RustForLinux => PrintkStyle::RustFmt,
    }
}

/// Lower the arguments of a printk call to a format string (dialect chosen
/// by the target: percent-escaped C for the raw printk FFI, brace-escaped
/// Rust for pr_info!) and the list of value expressions. printk's only
/// argument must be a string literal or an f-string; each f-string
/// interpolation lowers to a conversion with the interpolated value passed
/// alongside.
fn lower_printk_args(
    args: &[python_ast::ExprType],
    line: Option<usize>,
    style: PrintkStyle,
) -> Result<(String, Vec<String>)> {
    if args.len() != 1 {
        return Err(kernel_body_err(
            "printk takes exactly one argument: a format string literal or f-string",
            line,
        ));
    }
    let mut fmt = String::new();
    let mut values: Vec<String> = Vec::new();
    match &args[0] {
        python_ast::ExprType::Constant(c) => {
            if let Some(litrs::Literal::String(slit)) = &c.0 {
                fmt.push_str(&escape_kernel_fmt(slit.value(), style));
            } else {
                return Err(kernel_body_err(
                    "printk format must be a string literal or f-string",
                    line,
                ));
            }
        }
        python_ast::ExprType::JoinedStr(js) => {
            for part in &js.values {
                match part {
                    python_ast::ExprType::Constant(c) => {
                        if let Some(litrs::Literal::String(slit)) = &c.0 {
                            fmt.push_str(&escape_kernel_fmt(slit.value(), style));
                        } else {
                            return Err(kernel_body_err(
                                "printk f-string parts must be strings or interpolations",
                                line,
                            ));
                        }
                    }
                    python_ast::ExprType::FormattedValue(fv) => {
                        if fv.conversion.is_some() || fv.format_spec.is_some() {
                            return Err(kernel_body_err(
                                "printk f-string conversions (!s/!r) and format specs are not \
                                 supported; interpolate plain integer values",
                                line,
                            ));
                        }
                        match style {
                            PrintkStyle::C => fmt.push_str("%ld"),
                            PrintkStyle::RustFmt => fmt.push_str("{}"),
                        }
                        values.push(lower_kernel_value(&fv.value, line)?);
                    }
                    _ => {
                        return Err(kernel_body_err(
                            "printk f-string parts must be strings or interpolations",
                            line,
                        ));
                    }
                }
            }
        }
        _ => {
            return Err(kernel_body_err(
                "printk format must be a string literal or f-string",
                line,
            ));
        }
    }
    Ok((fmt, values))
}

/// Escape a Python string for use inside a kernel format literal: control
/// characters become escapes, then the dialect-specific literal delimiters
/// are doubled — `%` for the C format parser (printk), `{`/`}` for Rust's
/// format! parser (pr_info!).
fn escape_kernel_fmt(s: &str, style: PrintkStyle) -> String {
    let mut escaped = String::new();
    for ch in s.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\t' => escaped.push_str("\\t"),
            '\r' => escaped.push_str("\\r"),
            '\0' => escaped.push_str("\\0"),
            '%' if style == PrintkStyle::C => escaped.push_str("%%"),
            '{' if style == PrintkStyle::RustFmt => escaped.push_str("{{"),
            '}' if style == PrintkStyle::RustFmt => escaped.push_str("}}"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Escape a Python string for a C byte-string literal (used for string
/// return values in raw-FFI entry bodies).
fn escape_c_format(s: &str) -> String {
    escape_kernel_fmt(s, PrintkStyle::C)
}

/// A simplified call representation for kernel lowering.
struct KernelCall<'a> {
    name: String,
    args: &'a [python_ast::ExprType],
    keywords: &'a [python_ast::Keyword],
}

/// Try to extract a named call from an expression.
fn extract_kernel_call(expr: &python_ast::Expr) -> Option<KernelCall<'_>> {
    if let python_ast::ExprType::Call(call) = &expr.value {
        if let python_ast::ExprType::Name(name) = call.func.as_ref() {
            return Some(KernelCall {
                name: name.id.clone(),
                args: &call.args,
                keywords: &call.keywords,
            });
        }
    }
    None
}

/// Lower a kernel-mode expression that must stand on its own (a return
/// value): string literals lower to C-string pointers, integer literals to
/// i64. Anything else is a loud error.
fn lower_kernel_expr(expr: &python_ast::ExprType, line: Option<usize>) -> Result<String> {
    match expr {
        python_ast::ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::String(slit)) => {
                let escaped = escape_c_format(slit.value());
                Ok(format!(
                    "b\"{escaped}\\0\".as_ptr() as *const core::ffi::c_char"
                ))
            }
            Some(litrs::Literal::Integer(ilit)) => ilit
                .value::<i64>()
                .map(|v| v.to_string())
                .ok_or_else(|| kernel_body_err("integer literal is too large", line)),
            _ => Err(kernel_body_err(
                "unsupported literal in module entry body (only string and integer literals)",
                line,
            )),
        },
        _ => Err(kernel_body_err(
            "unsupported expression in module entry body (only string and integer literals)",
            line,
        )),
    }
}

/// Lower an f-string interpolation value to a Rust expression. The kernel
/// ABI has no heap strings in argument position, so only names (resolved to
/// the i64 locals introduced by the body lowering) and integer literals are
/// valid interpolation values.
fn lower_kernel_value(expr: &python_ast::ExprType, line: Option<usize>) -> Result<String> {
    match expr {
        python_ast::ExprType::Name(name) => Ok(safe_ident(&name.id).to_string()),
        python_ast::ExprType::Constant(c) => {
            if let Some(litrs::Literal::Integer(ilit)) = &c.0 {
                ilit.value::<i64>()
                    .map(|v| v.to_string())
                    .ok_or_else(|| kernel_body_err("integer literal is too large", line))
            } else {
                Err(kernel_body_err(
                    "printk interpolations support integer values only",
                    line,
                ))
            }
        }
        _ => Err(kernel_body_err(
            "printk interpolations support integer values only",
            line,
        )),
    }
}

/// Lower positional arguments of a kernel-resource call: integer literals
/// and previously-bound i64 locals — the two value kinds the kernel ABI
/// supports.
fn lower_kernel_call_args(
    args: &[python_ast::ExprType],
    line: Option<usize>,
) -> Result<Vec<String>> {
    args.iter().map(|a| lower_kernel_value(a, line)).collect()
}

/// Convert `package` into a Cargo crate under `out_dir`.
pub fn convert(
    package: &PyPackage,
    out_dir: &Path,
    opts: &ConvertOptions,
) -> Result<ConvertedCrate> {
    if opts.no_std && opts.pyo3 {
        bail!(
            "PyO3 bindings require std (pyo3 links the Python runtime); drop one of --pyo3 / --no-std"
        );
    }
    if opts.kernel_module && opts.pyo3 {
        bail!("PyO3 bindings cannot be used in kernel modules");
    }
    let src_dir = out_dir.join("src");
    fs::create_dir_all(&src_dir).with_context(|| format!("creating {}", src_dir.display()))?;

    if opts.rust_for_linux && !opts.kernel_module {
        bail!("rust_for_linux requires kernel_module (pass --kernel-module with --rust-for-linux)");
    }
    if opts.driver && (opts.kernel_module || opts.pyo3 || opts.no_std) {
        bail!("--driver cannot be combined with --kernel-module, --pyo3, or --no-std");
    }

    let entry_file = package.entry_module().map(|m| m.file.clone());

    // Async usage across the package: any async construct (async def/await/
    // async for/async with) needs an executor, and any `import asyncio`
    // needs the tokio-backed stdpython module. Binary conversions link the
    // runtime themselves (tokio behind the `async-tokio` feature); library
    // conversions expose plain async fns and only pull stdpython's
    // asyncio module when they import it.
    let (package_uses_async, package_imports_asyncio) = package_async_flags(package);
    if opts.kernel_module && package_uses_async {
        bail!(
            "kernel modules cannot use async/await: the kernel has no user-space \
             executor, and the generated entry points are synchronous"
        );
    }
    if opts.no_std && (package_uses_async || package_imports_asyncio) {
        bail!(
            "async/await (and asyncio) require the tokio runtime, which the no_std \
             profile does not provide; convert without --no-std"
        );
    }
    let async_runtime_dep = entry_file.is_some() && package_uses_async;

    // `[python-modules]` vendors Python libraries into the generated crate
    // as a module tree; kernel modules compile a single entry file with no
    // tree, so a manifest that declares any is a loud error up front.
    let python_manifest = read_python_module_manifest(package)?;
    if opts.kernel_module && !python_manifest.is_empty() {
        bail!(
            "[python-modules] cannot be used with --kernel-module: kernel modules \
             compile a single entry file with no module tree"
        );
    }

    // The packaging metadata's own dependencies ([project] dependencies,
    // install_requires) are resolved pip-style — a PyPI fetch when they are
    // not already vendored via [python-modules] — so a pyproject.toml
    // project converts with its libraries wired in the same way Python
    // would install them. Explicit manifest entries override the fetch.
    // Kernel modules compile a single entry file and reject all python
    // modules, so they never resolve dependencies.
    let python_manifest = if opts.no_deps
        || opts.kernel_module
        || package.dependencies.is_empty()
    {
        python_manifest
    } else {
        resolve_packaging_dependencies(package, python_manifest)?
    };

    // Kernel modules use a dedicated lowering path: no stdpython, no alloc,
    // raw FFI entry points with printk and module metadata.
    if opts.kernel_module {
        let source = if let Some(entry) = &entry_file {
            let entry_module = package.modules.iter().find(|m| &m.file == entry);
            if let Some(m) = entry_module {
                m.source.clone()
            } else {
                bail!("kernel module entry file not found");
            }
        } else if let Some(first) = package.modules.first() {
            first.source.clone()
        } else {
            bail!("kernel module requires at least one Python source file");
        };
        let target = if opts.rust_for_linux {
            KernelTarget::RustForLinux
        } else {
            KernelTarget::RawFfi
        };
        let manifest = parse_device_manifest(&source)?;
        let crate_name = manifest
            .module_name
            .clone()
            .unwrap_or_else(|| package.name.clone());
        let kernel = generate_kernel_lib_rs(&source, target, &crate_name, &manifest)?;
        fs::write(src_dir.join("lib.rs"), &kernel.code)?;
        if let Some(device) = &kernel.device {
            fs::write(src_dir.join("device.rs"), device)?;
        }
        let has_binary = false;
        let warnings = Vec::new();
        // Kernel modules take the alloc-tier branch below, which carries no
        // surface features, so there are no vendored deps to consider.
        write_cargo_toml(
            package,
            &[],
            &std::collections::HashSet::new(),
            out_dir,
            opts,
            has_binary,
            &manifest,
            kernel.uses_shim,
        )?;
        if !opts.rust_for_linux {
            // rust-for-linux modules are registered in the kernel tree's
            // Kbuild Makefile (rust-obj-m += <name>.o), not a standalone one.
            write_kernel_makefile(&crate_name, out_dir, kernel.has_init, kernel.has_exit)?;
        }
        return Ok(ConvertedCrate {
            root: out_dir.to_path_buf(),
            name: crate_name,
            has_binary,
            warnings,
        });
    }

    // Userspace drivers use the full stdpython transpile for the logic and
    // wrap it in generated syscall glue — the only Rust shipped is the
    // template below; everything else comes from the Python.
    if opts.driver {
        return convert_driver(package, out_dir, opts, &python_manifest);
    }

    // Rust-module imports (`import crc32c`): resolve the manifest once and
    // thread the registry through every module's conversion so import
    // statements and call sites lower to direct crate calls.
    let rust_manifest = read_rust_module_manifest(package)?;
    let rust_modules = rust_module_registry(package, &rust_manifest)?;

    // Python-library imports (`[python-modules]`): vendored PyPI-style
    // packages transpiled into the crate beside the package's own modules.
    let python_deps = python_module_deps(package, &python_manifest)?;

    // Transpile every module, collecting lossy-conversion warnings.
    // Dependency modules come first so their declarations are written
    // before any module that imports them (order in `transpiled` only
    // matters for the written file order, not correctness).
    let python_module_names: std::collections::HashSet<String> =
        python_deps.iter().map(|(name, _)| name.clone()).collect();
    // ASTs of every module of the generated crate (vendored deps + the
    // package itself), keyed by module path.
    // Reachability-based module selection: a module outside the
    // import-reachable set (a CLI helper like idna/cli.py, a __main__
    // block, test utilities) is SKIPPED, so a divergent module the program
    // never imports no longer blocks conversion.
    let reachable = reachable_module_paths(package, &python_deps);
    // Only the modules the crate EMITS enter the shared map: a crate-wide
    // index over it (the class hierarchy's sum-type variants) must never
    // name a module that was skipped as unreachable (`crate::urllib3::
    // contrib::socks::...` in a crate with no contrib module — E0433).
    let module_defs = module_def_map(
        python_deps
            .iter()
            .flat_map(|(_, dep)| dep.modules.iter())
            .chain(package.modules.iter())
            .filter(|m| reachable.contains(&m.path)),
    );
    // One options object per conversion: the shared `module_defs` and the
    // cross-module trait-mut cache (computed once, reused by every module's
    // transpile via the Rc clone).
    let base_options = conversion_base_options(
        opts,
        &rust_modules,
        &python_module_names,
        &module_defs,
        async_runtime_dep,
        &package.name,
    );
    let mut transpiled: Vec<(&PyModule, String)> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    for (_name, dep) in &python_deps {
        for module in &dep.modules {
            if !reachable.contains(&module.path) {
                continue;
            }
            let code = transpile(module, &mut warnings, &base_options)?;
            transpiled.push((module, code));
        }
    }
    for module in &package.modules {
        if !reachable.contains(&module.path) {
            continue;
        }
        let code = transpile(module, &mut warnings, &base_options)?;
        transpiled.push((module, code));
    }
    // An entry module named `main` (path ["main"]) is bin-only: its module
    // file would be src/main.rs, which is the bin crate root, so it must
    // not be declared as a lib submodule (rustc would compile the bin
    // content — with its sibling `mod vendored;` decls — as a nested module
    // of the lib) and its functions are not reachable from the lib (the
    // PyO3 bindings live in the lib and must not reference them). An entry
    // module under any OTHER name keeps its lib-side module (src/<name>.rs
    // coexists with the src/main.rs bin).
    let entry_is_bin_only = entry_file
        .as_ref()
        .map(|f| {
            transpiled
                .iter()
                .find(|(m, _)| &m.file == f)
                .is_some_and(|(m, _)| m.path == ["main"])
        })
        .unwrap_or(false);
    // Bindings are generated before files are written so their warnings
    // (e.g. forced Python-side renames) participate in the warning mode.
    // Bindings describe the PACKAGE's public API only: vendored
    // [python-modules] dependencies are internal implementation (they are
    // transpiled and written, but must not surface as #[pyfunction]s in the
    // generated #[pymodule], and their bare-name collisions must not rename
    // or fail the package's own functions). A bin-only entry module is not
    // part of the lib either, so it is skipped the same way.
    let bindings_text = if opts.pyo3 {
        let package_modules: Vec<(&PyModule, String)> = package
            .modules
            .iter()
            .filter(|m| {
                !(entry_is_bin_only
                    && entry_file
                        .as_ref()
                        .is_some_and(|f| &m.file == f))
            })
            .filter_map(|m| {
                transpiled
                    .iter()
                    .find(|(tm, _)| tm.path == m.path)
                    .map(|(tm, code)| (*tm, code.clone()))
            })
            .collect();
        Some(generate_bindings(package, &package_modules, &mut warnings)?)
    } else {
        None
    };

    match opts.warnings {
        WarningMode::Deny if !warnings.is_empty() => bail!(
            "lossy conversion (warnings denied):\n  {}",
            warnings.join("\n  ")
        ),
        WarningMode::Allow => warnings.clear(),
        _ => {}
    }

    // Parent -> children map for `pub mod` declarations. The entry module
    // still gets a lib-side module (harmless), except a dedicated
    // `__main__.py`, which is bin-only by convention, and a NAMED entry
    // module (`main.py` with a `__main__` guard), which is also bin-only
    // (see `entry_is_bin_only` above). Non-root __init__ modules register
    // too: a sub-package whose only file is __init__.py must still be
    // declared by its parent or its code is silently dropped.
    let mut children: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
    for (module, _) in &transpiled {
        if module.path.is_empty() || is_dunder_main(module) {
            continue;
        }
        // Skip the bin-only entry module: it must not be declared as a lib
        // module (its file is overwritten by the bin below) nor by its
        // parent (the parent's decls are for lib modules).
        if entry_is_bin_only
            && entry_file
                .as_ref()
                .is_some_and(|f| &module.file == f)
        {
            continue;
        }
        let (parent, name) = module.path.split_at(module.path.len() - 1);
        children
            .entry(parent.to_vec())
            .or_default()
            .push(name[0].clone());
        // Intermediate packages ensure their ancestors know about them.
        for depth in 1..parent.len() + 1 {
            let (ancestor, child) = parent.split_at(depth - 1);
            let list = children.entry(ancestor.to_vec()).or_default();
            if !list.contains(&child[0].to_string()) {
                list.push(child[0].clone());
            }
        }
    }

    // Write module files. The lint allowances lead each file (they're inner
    // attributes), then the transpiled code (which may itself start with
    // inner doc attributes), then the `pub mod` declarations.
    // Two modules with the same package path would overwrite each other's
    // file — impossible within one package, but a vendored `[python-modules]`
    // dep can collide with a package module of the same name.
    {
        let mut seen: HashSet<&[String]> = HashSet::new();
        for (module, _) in &transpiled {
            if !seen.insert(module.path.as_slice()) {
                bail!(
                    "duplicate module path `{}`: a [python-modules] dependency \
                     collides with a module of the package",
                    module.path.join(".")
                );
            }
        }
    }

    let mut written: HashSet<Vec<String>> = HashSet::new();
    for (module, code) in &transpiled {
        if is_dunder_main(module) {
            continue; // handled as the binary below
        }
        let is_root = module.path.is_empty();
        let decls = mod_decls(&children, &module.path, module.is_init || is_root);
        let allows = if is_root {
            format!(
                "{}{}",
                no_std_attr(opts),
                generated_lint_attrs(opts.warnings)
            )
        } else {
            String::new()
        };
        let contents = format!("{}{}\n{}", allows, code, decls);
        let file = module_file_path(&src_dir, module);
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&file, format_rust(&contents))
            .with_context(|| format!("writing {}", file.display()))?;
        if module.is_init {
            written.insert(module.path.clone());
        }
    }
    write_missing_containers(&src_dir, &children, &written)?;

    // Ensure lib.rs exists even when the package has no root __init__.py.
    let lib_rs = src_dir.join("lib.rs");
    if !lib_rs.exists() {
        let decls = mod_decls(&children, &[], true);
        fs::write(
            &lib_rs,
            format_rust(&format!(
                "{}{}{}",
                no_std_attr(opts),
                generated_lint_attrs(opts.warnings),
                decls
            )),
        )?;
    }

    // PyO3 bindings.
    if let Some(bindings) = &bindings_text {
        fs::write(src_dir.join("python_api.rs"), format_rust(bindings))?;
        let mut lib = fs::read_to_string(&lib_rs)?;
        lib.push_str("\n#[cfg(feature = \"python\")]\nmod python_api;\n");
        fs::write(&lib_rs, format_rust(&lib))?;
    }

    // Binary entry point.
    let mut has_binary = false;
    if let Some(entry_file) = &entry_file {
        let (entry, code) = transpiled
            .iter()
            .find(|(m, _)| &m.file == entry_file)
            .expect("entry module was transpiled");
        // The bin target declares the sibling modules itself so the entry
        // module's `use crate::...` imports resolve within the bin crate.
        // Order: lint allowances, entry code (may start with inner doc
        // attributes), then the sibling mod declarations.
        let decls = if !is_dunder_main(entry) && entry.path.len() == 1 {
            // Exclude the entry module's own name from the bin-side decls.
            let mut decls = String::new();
            if let Some(kids) = children.get(&Vec::new()) {
                for kid in kids {
                    if Some(kid) != entry.path.first() {
                        decls.push_str(&format!("mod {};\n", kid));
                    }
                }
            }
            decls
        } else {
            mod_decls(&children, &[], true).replace("pub mod", "mod")
        };
        let main_contents = format!("{}{}\n{}", generated_lint_attrs(opts.warnings), code, decls);
        fs::write(src_dir.join("main.rs"), format_rust(&main_contents))?;
        has_binary = true;
    }

    write_cargo_toml(
        package,
        &python_deps,
        &reachable,
        out_dir,
        opts,
        has_binary,
        &DeviceManifest::default(),
        false,
    )?;

    Ok(ConvertedCrate {
        root: out_dir.to_path_buf(),
        name: package.name.clone(),
        has_binary,
        warnings,
    })
}

fn is_dunder_main(module: &PyModule) -> bool {
    module.path.last().map(String::as_str) == Some("__main__")
}

/// Device manifest a driver Python may declare; consumed by `--driver`
/// (user-space syscall glue) and `--kernel-module` (kernel misc device) to
/// parameterize the generated code. Absent constants keep the documented
/// defaults, so a driver Python can stay pure logic and still get working
/// glue. Declaring a device field (`__device_name__`, `__bufsz__`,
/// `__magic__`, `__device_mode__`) switches `--kernel-module` into
/// device-generation mode: rypip emits the misc device too.
struct DeviceManifest {
    /// `__module_name__` — kernel module name (→ `{name}.ko`). Defaults to
    /// the package name when absent.
    module_name: Option<String>,
    /// `__device_path__` — device node the user-space glue opens.
    device_path: String,
    /// `__device_name__` — misc device name (the `/dev/` node suffix).
    device_name: String,
    /// `__bufsz__` — byte-ring capacity in the kernel device.
    bufsz: usize,
    /// `__magic__` — device magic, echoed in the load message.
    magic: u32,
    /// `__device_mode__` — device node permission bits.
    device_mode: u32,
    /// `__ioc_reset__` / `__ioc_stats__` — ioctl numbers (shared ABI).
    ioc_reset: i64,
    ioc_stats: i64,
    /// True when the Python declared any device-generation field.
    has_device: bool,
}

impl Default for DeviceManifest {
    fn default() -> Self {
        DeviceManifest {
            module_name: None,
            device_path: "/dev/rython0".to_string(),
            device_name: "rython0".to_string(),
            bufsz: 4096,
            magic: 0x52594854,
            device_mode: 0o600,
            ioc_reset: 0x5201,
            ioc_stats: 0x8028_5202,
            has_device: false,
        }
    }
}

/// Read the device manifest out of a Python module: module-level constants
/// `__device_path__` (string), `__device_name__` / `__module_name__`
/// (strings), `__bufsz__` / `__magic__` / `__device_mode__` /
/// `__ioc_reset__` / `__ioc_stats__` (integers). Absent constants keep the
/// documented defaults. Malformed values are loud conversion errors — never
/// silently ignored.
fn parse_device_manifest(source: &str) -> Result<DeviceManifest> {
    let ast =
        parse_enhanced(source, "<driver>".to_string()).map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut manifest = DeviceManifest::default();
    for stmt in &ast.raw.body {
        if let python_ast::ast::tree::StatementType::Assign(assign) = &stmt.statement {
            if let (Some(target), value) = (assign.targets.first(), &assign.value) {
                if let python_ast::ExprType::Name(name) = target {
                    let key = name.id.as_str();
                    match key {
                        "__module_name__" => {
                            if let Some(s) = expr_str_literal(value) {
                                manifest.module_name = Some(s);
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __module_name__ must be a string literal"
                                );
                            }
                        }
                        "__device_path__" => {
                            if let Some(s) = expr_str_literal(value) {
                                manifest.device_path = s;
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __device_path__ must be a string literal"
                                );
                            }
                        }
                        "__device_name__" => {
                            if let Some(s) = expr_str_literal(value) {
                                manifest.has_device = true;
                                manifest.device_name = s;
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __device_name__ must be a string literal"
                                );
                            }
                        }
                        "__bufsz__" => {
                            if let Some(v) = expr_int_literal(value) {
                                manifest.has_device = true;
                                manifest.bufsz = v as usize;
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __bufsz__ must be an integer literal"
                                );
                            }
                        }
                        "__magic__" => {
                            if let Some(v) = expr_int_literal(value) {
                                manifest.has_device = true;
                                manifest.magic = v as u32;
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __magic__ must be an integer literal"
                                );
                            }
                        }
                        "__device_mode__" => {
                            if let Some(v) = expr_int_literal(value) {
                                manifest.has_device = true;
                                manifest.device_mode = v as u32;
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __device_mode__ must be an integer literal"
                                );
                            }
                        }
                        "__ioc_reset__" => {
                            if let Some(v) = expr_int_literal(value) {
                                manifest.ioc_reset = v;
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __ioc_reset__ must be an integer literal"
                                );
                            }
                        }
                        "__ioc_stats__" => {
                            if let Some(v) = expr_int_literal(value) {
                                manifest.ioc_stats = v;
                            } else {
                                bail!(
                                    "--driver/--kernel-module: __ioc_stats__ must be an integer literal"
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    // The device name is embedded in a Rust byte-string literal; keep it
    // to a safe charset so a weird Python string cannot break the output.
    if manifest.has_device
        && !manifest
            .device_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        bail!(
            "--driver/--kernel-module: __device_name__ must be alphanumeric, '_' or '-' \
             (got {:?})",
            manifest.device_name
        );
    }
    Ok(manifest)
}

/// Generate a userspace driver crate for a rython byte-ring misc device:
/// the Python driver logic transpiles to `src/lib.rs` (full stdpython), and
/// `src/main.rs` is generated syscall glue — open/read/write/ioctl around
/// the device manifest. The only Rust in the output is this template.
fn convert_driver(
    package: &PyPackage,
    out_dir: &Path,
    opts: &ConvertOptions,
    python_manifest: &HashMap<String, ManifestPythonModule>,
) -> Result<ConvertedCrate> {
    let src_dir = out_dir.join("src");
    fs::create_dir_all(&src_dir).with_context(|| format!("creating {}", src_dir.display()))?;

    let module = match package.entry_module() {
        Some(entry) => package
            .modules
            .iter()
            .find(|m| m.file == entry.file)
            .or_else(|| package.modules.first())
            .context("driver requires at least one Python source file")?,
        None => package
            .modules
            .first()
            .context("driver requires at least one Python source file")?,
    };

    // Python-library imports (`[python-modules]`): transpile the vendored
    // deps and write them as sibling modules of the crate so the entry
    // code's `from pylev import ...` resolves to `crate::pylev::...`.
    let python_deps = python_module_deps(package, python_manifest)?;
    let python_module_names: std::collections::HashSet<String> =
        python_deps.iter().map(|(name, _)| name.clone()).collect();

    let mut warnings = Vec::new();
    let rust_manifest = read_rust_module_manifest(package)?;
    let rust_modules = rust_module_registry(package, &rust_manifest)?;
    // Driver mode still vendors python-modules deps as siblings; give the
    // same cross-module knowledge as the plain package path.
    // Reachability-based module selection: a module outside the
    // import-reachable set (a CLI helper like idna/cli.py, a __main__
    // block, test utilities) is SKIPPED, so a divergent module the program
    // never imports no longer blocks conversion.
    let reachable = reachable_module_paths(package, &python_deps);
    // Only the modules the crate EMITS enter the shared map: a crate-wide
    // index over it (the class hierarchy's sum-type variants) must never
    // name a module that was skipped as unreachable (`crate::urllib3::
    // contrib::socks::...` in a crate with no contrib module — E0433).
    let module_defs = module_def_map(
        python_deps
            .iter()
            .flat_map(|(_, dep)| dep.modules.iter())
            .chain(package.modules.iter())
            .filter(|m| reachable.contains(&m.path)),
    );
    let (package_uses_async, _) = package_async_flags(package);
    let base_options = conversion_base_options(
        opts,
        &rust_modules,
        &python_module_names,
        &module_defs,
        package_uses_async,
        &package.name,
    );
    let code = transpile(module, &mut warnings, &base_options)?;

    let mut dep_transpiled: Vec<(&PyModule, String)> = Vec::new();
    for (_name, dep) in &python_deps {
        for dep_module in &dep.modules {
            if !reachable.contains(&dep_module.path) {
                continue;
            }
            let dep_code = transpile(dep_module, &mut warnings, &base_options)?;
            dep_transpiled.push((dep_module, dep_code));
        }
    }
    match opts.warnings {
        WarningMode::Deny if !warnings.is_empty() => bail!(
            "lossy conversion (warnings denied):\n  {}",
            warnings.join("\n  ")
        ),
        WarningMode::Allow => warnings.clear(),
        _ => {}
    }

    // lib.rs: the driver logic, compiled from Python.
    let mut lib = format!("{}\n{}", generated_lint_attrs(opts.warnings), code);

    // The dep modules join the crate as real modules; lib.rs declares the
    // top-level ones so `use crate::pylev::...` resolves. Sub-package
    // declarations are written inside the dep modules themselves.
    if !dep_transpiled.is_empty() {
        let mut children: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
        for (dep_module, _) in &dep_transpiled {
            let (parent, name) = dep_module.path.split_at(dep_module.path.len() - 1);
            children
                .entry(parent.to_vec())
                .or_default()
                .push(name[0].clone());
            for depth in 1..parent.len() + 1 {
                let (ancestor, child) = parent.split_at(depth - 1);
                let list = children.entry(ancestor.to_vec()).or_default();
                if !list.contains(&child[0].to_string()) {
                    list.push(child[0].clone());
                }
            }
        }
        let mut written: HashSet<Vec<String>> = HashSet::new();
        for (dep_module, dep_code) in &dep_transpiled {
            let decls = mod_decls(&children, &dep_module.path, dep_module.is_init);
            let contents = format!("{}\n{}", dep_code, decls);
            let file = module_file_path(&src_dir, dep_module);
            if let Some(parent) = file.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&file, format_rust(&contents))
                .with_context(|| format!("writing {}", file.display()))?;
            if dep_module.is_init {
                written.insert(dep_module.path.clone());
            }
        }
        write_missing_containers(&src_dir, &children, &written)?;
        for name in children.get(&Vec::new()).cloned().unwrap_or_default() {
            lib.push_str(&format!("\npub mod {};", name));
        }
    }
    fs::write(src_dir.join("lib.rs"), format_rust(&lib))?;

    // main.rs: the generated syscall glue, parameterized by the manifest.
    let manifest = parse_device_manifest(&module.source)?;
    let main = driver_glue_template()
        .replace("@@CRATE@@", &package.name)
        .replace("@@DEVICE_PATH@@", &manifest.device_path)
        .replace("@@IOC_RESET@@", &format!("{:x}", manifest.ioc_reset))
        .replace("@@IOC_STATS@@", &format!("{:x}", manifest.ioc_stats));
    fs::write(src_dir.join("main.rs"), format_rust(&main))?;

    write_cargo_toml(
        package,
        &python_deps,
        &reachable,
        out_dir,
        opts,
        true,
        &manifest,
        false,
    )?;

    Ok(ConvertedCrate {
        root: out_dir.to_path_buf(),
        name: package.name.clone(),
        has_binary: true,
        warnings,
    })
}

/// The syscall glue template for a rython byte-ring misc device. Placeholder
/// tokens: `@@CRATE@@` (crate name), `@@DEVICE_PATH@@`, `@@IOC_RESET@@`,
/// `@@IOC_STATS@@` (hex, no 0x). Everything else is the standard driver
/// shell: open/read/write/ioctl, one command round-trip with CRC-8
/// verification, CLI/interactive modes, and unit tests for the generated
/// driver logic.
fn driver_glue_template() -> &'static str {
    r#"//! Generated by `rypip convert --driver` — the syscall glue for a rython
//! byte-ring misc device (UIO style). The driver logic lives in the
//! compiled Python module; this file only does open/read/write/ioctl.
//! Edit the Python (`driver.py`), not this file.

use std::ffi::CString;
use std::io::{self, BufRead, IsTerminal};
use std::os::fd::RawFd;

use @@CRATE@@::{crc8, parse_hex, Device};
use stdpython::PyDict;

/// ioctl numbers, declared in the driver Python as `__ioc_reset__` and
/// `__ioc_stats__` (they mirror the kernel side):
///   RYTHON_IOC_RESET = _IO(0x52, 1)              -> 0x5201
///   RYTHON_IOC_STATS = _IOR(0x52, 2, stats(40))  -> 0x80285202
const RYTHON_IOC_RESET: libc::c_ulong = 0x@@IOC_RESET@@;
const RYTHON_IOC_STATS: libc::c_ulong = 0x@@IOC_STATS@@;

const DEVICE_PATH: &str = "@@DEVICE_PATH@@";

/// Kernel-side device counters (40-byte _IOR struct, 5 x u64).
#[repr(C)]
struct DeviceStats {
    wr_bytes: u64,
    rd_bytes: u64,
    wr_calls: u64,
    rd_calls: u64,
    resets: u64,
}

/* ------------------------------------------------------------------ */
/* syscall plumbing                                                   */
/* ------------------------------------------------------------------ */

fn open_device(path: &str) -> RawFd {
    let cpath = CString::new(path).expect("device path contains NUL");
    // SAFETY: standard libc open on a caller-supplied path.
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR) };
    if fd < 0 {
        let err = io::Error::last_os_error();
        eprintln!(
            "{}: cannot open {}: {} (is the module loaded? try `make load`)",
            env!("CARGO_PKG_NAME"),
            path,
            err
        );
        std::process::exit(1);
    }
    fd
}

/// Program the device: write the raw command bytes into the ring.
fn device_write(fd: RawFd, bytes: &[u8]) -> io::Result<usize> {
    // SAFETY: fd is a valid open descriptor, bytes is a readable slice.
    let n = unsafe { libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len()) };
    if n < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

/// Read the device's echo back out of the ring.
fn device_read(fd: RawFd, buf: &mut [u8]) -> io::Result<usize> {
    // SAFETY: fd is a valid open descriptor, buf is a writable slice.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
    if n < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

fn device_reset(fd: RawFd) -> io::Result<()> {
    // SAFETY: ioctl with a command that takes no argument.
    let rc = unsafe { libc::ioctl(fd, RYTHON_IOC_RESET) };
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn device_stats(fd: RawFd) -> io::Result<DeviceStats> {
    let mut st = DeviceStats {
        wr_bytes: 0,
        rd_bytes: 0,
        wr_calls: 0,
        rd_calls: 0,
        resets: 0,
    };
    // SAFETY: st is a valid, correctly sized struct for the _IOR command.
    let rc = unsafe { libc::ioctl(fd, RYTHON_IOC_STATS, &mut st as *mut DeviceStats) };
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(st)
    }
}

/* ------------------------------------------------------------------ */
/* one command round-trip                                             */
/* ------------------------------------------------------------------ */

/// Run one driver command: Python logic -> device -> echo -> CRC-8.
fn run_command(dev: &mut Device, fd: RawFd, line: &str) {
    // 1. Driver logic (compiled from the driver Python): parse, validate, update state.
    let resp = match dev.handle(line) {
        Ok(r) => r,
        Err(e) => format!("ERR exception: {}", e),
    };

    // 2. Program the kernel device with the raw command bytes.
    let mut wire = line.as_bytes().to_vec();
    wire.push(b'\n');
    if let Err(e) = device_write(fd, &wire) {
        eprintln!("{}: write failed: {}", env!("CARGO_PKG_NAME"), e);
        std::process::exit(1);
    }

    // 3. Read the device echo.
    let mut echo = [0u8; 512];
    let n = match device_read(fd, &mut echo) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{}: read failed: {}", env!("CARGO_PKG_NAME"), e);
            std::process::exit(1);
        }
    };

    // 4. Checksum the device bytes with the generated CRC-8.
    let crc = crc8(echo[..n].to_vec()).unwrap_or(-1);

    println!("{}   [dev-echo {}B crc=0x{:02x}]", resp, n, crc);
}

fn print_usage() {
    println!(
        "usage: rython-driver [--device PATH] [--reset] [--stats] [COMMAND...]\n\
         \n\
         Opens the rython kernel device and drives it with the driver logic\n\
         compiled from the driver Python by rypip.\n\
         \n\
         options:\n\
         \x20 --device PATH   device node (default /dev/rython0)\n\
         \x20 --reset         reset the kernel device (ioctl, clears ring/counters)\n\
         \x20 --stats         print kernel device counters (ioctl)\n\
         \n\
         With COMMAND arguments each is run once; without them, an\n\
         interactive session is started. Commands are the driver protocol:\n\
         \n\
         \x20 READ <hex> | WRITE <hex> <hex> | DUMP | STATS | RESET | HELP"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut path = DEVICE_PATH.to_string();
    let mut do_reset = false;
    let mut do_stats = false;
    let mut command_line: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                i += 1;
                path = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--device needs a path");
                    std::process::exit(2);
                });
            }
            "--reset" => do_reset = true,
            "--stats" => do_stats = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            a if a.starts_with('-') => {
                eprintln!("unknown option: {}", a);
                print_usage();
                std::process::exit(2);
            }
            other => {
                // Positional arguments form a single command line, so
                // `rython-driver WRITE 2 2a` runs the command "WRITE 2 2a".
                match command_line.as_mut() {
                    Some(cl) => {
                        cl.push(' ');
                        cl.push_str(other);
                    }
                    None => command_line = Some(other.to_string()),
                }
            }
        }
        i += 1;
    }

    let fd = open_device(&path);

    if do_reset {
        match device_reset(fd) {
            Ok(()) => println!("device reset (ioctl)"),
            Err(e) => {
                eprintln!("reset failed: {}", e);
                std::process::exit(1);
            }
        }
    }

    if do_stats {
        match device_stats(fd) {
            Ok(st) => println!(
                "kernel stats: wr_bytes={} rd_bytes={} wr_calls={} rd_calls={} resets={}",
                st.wr_bytes, st.rd_bytes, st.wr_calls, st.rd_calls, st.resets
            ),
            Err(e) => {
                eprintln!("stats failed: {}", e);
                std::process::exit(1);
            }
        }
    }

    // The driver state machine, instantiated from compiled Python.
    let mut dev = Device::new(PyDict::new(), "dev0").expect("Device::new");

    match command_line {
        Some(cmd) => run_command(&mut dev, fd, &cmd),
        None if do_reset || do_stats => {} // flags only — nothing else to run
        None => {
            if io::stdin().is_terminal() {
                println!("{}: interactive mode (Ctrl-D to quit). Try: WRITE 2 2a", env!("CARGO_PKG_NAME"));
            }
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };
                if line.trim().is_empty() {
                    continue;
                }
                run_command(&mut dev, fd, line.trim());
            }
        }
    }

    // SAFETY: fd was opened by us and is no longer used after this.
    unsafe {
        libc::close(fd);
    }
}

/* ------------------------------------------------------------------ */
/* unit tests for the generated (Python-derived) logic                */
/* ------------------------------------------------------------------ */

#[cfg(test)]
mod tests {
    use super::*;

    fn new_dev() -> Device {
        Device::new(PyDict::new(), "test").expect("Device::new")
    }

    #[test]
    fn generated_write_read_roundtrip() {
        let mut d = new_dev();
        assert_eq!(d.handle("WRITE 2 2a").unwrap(), "OK 42");
        assert_eq!(d.handle("READ 2").unwrap(), "VAL 42");
        assert_eq!(d.handle("READ 7").unwrap(), "VAL 0");
    }

    #[test]
    fn generated_bounds_and_errors() {
        let mut d = new_dev();
        assert_eq!(d.handle("WRITE 8 1").unwrap(), "ERR bad write"); // off window
        assert_eq!(d.handle("WRITE 2 zz").unwrap(), "ERR bad write"); // bad hex
        assert_eq!(d.handle("NOPE").unwrap(), "ERR unknown cmd: NOPE");
    }

    #[test]
    fn generated_dump_stats_reset() {
        let mut d = new_dev();
        d.handle("WRITE 1 5").unwrap();
        d.handle("WRITE 3 6").unwrap();
        let dump = d.handle("DUMP").unwrap();
        assert!(dump.contains("1:5"), "dump={dump}");
        assert!(dump.contains("3:6"), "dump={dump}");
        assert_eq!(d.handle("STATS").unwrap(), "ops=2 reads=0");
        assert_eq!(d.handle("RESET").unwrap(), "OK reset");
        assert_eq!(d.handle("DUMP").unwrap(), "");
    }

    #[test]
    fn generated_crc8_matches_reference() {
        // Reference: CRC-8, poly 0x07, init 0, no reflection.
        fn ref_crc8(data: &[u8]) -> u8 {
            let mut crc = 0u8;
            for &b in data {
                crc ^= b;
                for _ in 0..8 {
                    crc = if crc & 1 != 0 { (crc >> 1) ^ 0x07 } else { crc >> 1 };
                }
            }
            crc
        }
        for data in [
            vec![0u8],
            vec![1, 2, 3, 4],
            b"WRITE 2 2a\n".to_vec(),
            vec![0xde, 0xad, 0xbe, 0xef],
        ] {
            let got = crc8(data.to_vec()).unwrap() as u8;
            assert_eq!(got, ref_crc8(&data), "data={data:02x?}");
        }
    }

    #[test]
    fn generated_parse_hex() {
        assert_eq!(parse_hex("0").unwrap(), 0);
        assert_eq!(parse_hex("2a").unwrap(), 42);
        assert_eq!(parse_hex("DEADBEEF").unwrap(), 0xDEADBEEF);
        assert_eq!(parse_hex("nope").unwrap(), -1);
    }
}
"#
}

/// A clean package-relative filename for the parser: it derives a module
/// identifier from the filename, and absolute temp paths contain characters
/// that aren't valid in identifiers.
fn parse_filename(module: &PyModule) -> String {
    if module.is_init {
        if module.path.is_empty() {
            "__init__.py".to_string()
        } else {
            format!("{}/__init__.py", module.path.join("/"))
        }
    } else {
        format!("{}.py", module.path.join("/"))
    }
}

/// Cross-module class knowledge for imports within the generated crate:
/// the parsed AST of every sibling module, keyed by module path. ImportFrom
/// lowering uses this to bring a hierarchy class's traits into scope when
/// the class is imported from another module (inherited methods live on
/// the traits, and Rust method resolution requires the trait to be in
/// scope at the call site) and to resolve imported classes for
/// construction (`Dog("Rex")` → `Dog::new("Rex")?`).
fn module_def_map<'a>(
    modules: impl Iterator<Item = &'a PyModule>,
) -> std::collections::HashMap<Vec<String>, std::rc::Rc<python_ast::Module>> {
    let mut map = std::collections::HashMap::new();
    for module in modules {
        let Ok(ast) = parse_enhanced(&module.source, parse_filename(module)) else {
            continue;
        };
        map.insert(module.path.clone(), std::rc::Rc::new(ast));
    }
    map
}

/// Transpile one Python module to Rust source text, appending
/// lossy-conversion warnings (which are also baked into the generated code
/// as #[deprecated] notes, unless the warning mode suppresses them).
/// `base_options` carries the conversion-level context shared by every
/// module (rust/python module registries, `module_defs`, and the shared
/// cross-module trait-mut cache — the one-time merged-table scan runs on
/// the first module that needs it and is reused by the rest); only the
/// module's package path is per-module.
fn transpile(
    module: &PyModule,
    warnings: &mut Vec<String>,
    base_options: &PythonOptions,
) -> Result<String> {
    let ast = parse_enhanced(&module.source, parse_filename(module))
        .map_err(|e| anyhow::anyhow!("{} ({})", e, module.file.display()))?;

    for stmt in &ast.raw.body {
        if let StatementType::FunctionDef(func) = &stmt.statement {
            for note in func.lossy_conversion_notes() {
                warnings.push(format!(
                    "{}: function `{}`: {}",
                    parse_filename(module),
                    func.name,
                    note.trim_start_matches("rython: "),
                ));
            }
        }
    }

    let symbols = ast.clone().find_symbols(SymbolTableScopes::new());
    let module_name = module
        .path
        .last()
        .cloned()
        .unwrap_or_else(|| "lib".to_string());
    let mut options = base_options.clone();
    // Relative imports resolve against the current module's package path;
    // empty at the crate root (the default when unset).
    options.module_path = module_package_path(module);
    // The module's OWN path (module_path is the package path): the
    // promotion pass uses it to learn which names sibling modules import
    // from this module (`from .constant import _THAI`), so cross-module
    // constants are promoted to importable statics.
    options.this_module_path = module.path.clone();
    // Register rust-module imports in the shared symbol table so call
    // lowering can resolve them (Import::to_rust only sees a clone).
    let mut symbols = symbols;
    python_ast::register_rust_module_imports(&ast.raw.body, &options, &mut symbols)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let tokens = ast
        .to_rust(CodeGenContext::Module(module_name), options, symbols)
        .map_err(|e| {
            anyhow::anyhow!(
                "compiling {}: {}",
                module.file.display(),
                python_ast::format_error_chain(e.as_ref())
            )
        })?;
    // M5 definition-time warnings collected during inference (a bound set
    // no known type satisfies) — reported like any lossy-conversion
    // warning: reported at -W warn, fatal at -W deny, suppressed at allow.
    for w in std::mem::take(&mut *base_options.definition_warnings.borrow_mut()) {
        warnings.push(format!("{}: {}", parse_filename(module), w));
    }
    Ok(tokens.to_string())
}

/// Whether any module of the package contains async constructs (async def/
/// await/async for/async with) and whether any imports `asyncio`. Parse
/// failures are ignored — the transpile pass reports them with context.
fn package_async_flags(package: &PyPackage) -> (bool, bool) {
    let mut uses_async = false;
    let mut imports_asyncio = false;
    for module in &package.modules {
        if uses_async && imports_asyncio {
            break;
        }
        if let Ok(ast) = parse_enhanced(&module.source, parse_filename(module)) {
            if !uses_async && python_ast::module_contains_async(&ast.raw.body) {
                uses_async = true;
            }
            if !imports_asyncio && python_ast::module_imports_asyncio(&ast.raw.body) {
                imports_asyncio = true;
            }
        }
    }
    (uses_async, imports_asyncio)
}

/// A stdpython platform surface: a heavyweight dependency that only some
/// converted packages need, gated behind its own Cargo feature (the
/// convention `crates/stdpython/Cargo.toml` states — one feature per
/// surface, so the default build stays dependency-light).
///
/// A package opts into a surface by importing the Python module it backs,
/// so the mapping is import-root -> feature and lives here alone: the
/// feature names appear in exactly one place rather than scattered across
/// the emission sites. Guessing too narrowly is safe in the prime
/// directive's sense — the generated crate then names a module that isn't
/// compiled in and fails loudly, it never silently loses the surface.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum SurfaceFeature {
    /// `re` — the regex crate (a third of the std tier's build cost).
    ReRegex,
    /// `ssl` — TLS over rustls. Not implied by `http-ureq`: ureq carries
    /// its own rustls, and stdpython's `ssl` module stands alone.
    SslRustls,
    /// `urllib.request` — the ureq-backed HTTP client.
    HttpUreq,
}

impl SurfaceFeature {
    /// Every surface, in the order features are emitted.
    const ALL: [SurfaceFeature; 3] = [
        SurfaceFeature::ReRegex,
        SurfaceFeature::SslRustls,
        SurfaceFeature::HttpUreq,
    ];

    /// The stdpython Cargo feature this surface turns on.
    fn cargo_feature(self) -> &'static str {
        match self {
            SurfaceFeature::ReRegex => "re-regex",
            SurfaceFeature::SslRustls => "ssl-rustls",
            SurfaceFeature::HttpUreq => "http-ureq",
        }
    }

    /// The Python import root that pulls the surface in.
    fn import_root(self) -> &'static str {
        match self {
            SurfaceFeature::ReRegex => "re",
            SurfaceFeature::SslRustls => "ssl",
            SurfaceFeature::HttpUreq => "urllib",
        }
    }
}

/// The surfaces the package's imports ask for. Parses each module once and
/// tests every root against it, rather than re-parsing per feature.
///
/// Discovery spans exactly what the conversion transpiles into the crate,
/// no more and no less. Vendored `[python-modules]` deps are written as
/// sibling modules, so an `import re` in one of them needs the surface
/// just as much as one in the entry module -- but only if the module is
/// import-reachable, since `convert` skips the rest (a dependency's CLI
/// helper or test utility). Every module of the root package is a
/// reachability seed, so only dependency modules can be filtered out.
fn package_surface_features(
    package: &PyPackage,
    python_deps: &[(String, PyPackage)],
    reachable: &std::collections::HashSet<Vec<String>>,
) -> Vec<SurfaceFeature> {
    let mut needed: Vec<SurfaceFeature> = Vec::new();
    let modules = package.modules.iter().chain(
        python_deps
            .iter()
            .flat_map(|(_, dep)| dep.modules.iter())
            .filter(|m| reachable.contains(&m.path)),
    );
    for module in modules {
        let Ok(ast) = parse_enhanced(&module.source, parse_filename(module)) else {
            continue;
        };
        for surface in SurfaceFeature::ALL {
            if !needed.contains(&surface)
                && python_ast::module_imports_root(&ast.raw.body, surface.import_root())
            {
                needed.push(surface);
            }
        }
        if needed.len() == SurfaceFeature::ALL.len() {
            break;
        }
    }
    needed.sort();
    needed
}

/// Conversion-level PythonOptions shared by every module of one conversion
/// (see `transpile`): the registries, `module_defs`, and the shared
/// cross-module trait-mut cache.
fn conversion_base_options(
    opts: &ConvertOptions,
    rust_modules: &HashMap<String, python_ast::RustModuleSpec>,
    python_module_names: &std::collections::HashSet<String>,
    module_defs: &std::collections::HashMap<Vec<String>, std::rc::Rc<python_ast::Module>>,
    async_runtime_dep: bool,
    package_name: &str,
) -> PythonOptions {
    PythonOptions {
        lossy_warnings: opts.warnings != WarningMode::Allow,
        no_std: opts.no_std,
        rust_modules: std::rc::Rc::new(rust_modules.clone()),
        python_modules: std::rc::Rc::new(python_module_names.clone()),
        module_defs: std::rc::Rc::new(module_defs.clone()),
        async_runtime_dep,
        // The package's own name: root-qualified absolute self-imports
        // (`from urllib3.connection import ...`, pip's src-layout paths)
        // strip exactly this leading segment when matching module_defs'
        // relative keys — never an external root like `h2`.
        python_namespace: package_name.to_string(),
        ..Default::default()
    }
}

/// `pub mod child;` declarations for a container module.
fn mod_decls(
    children: &BTreeMap<Vec<String>, Vec<String>>,
    at: &[String],
    _is_container: bool,
) -> String {
    let mut out = String::new();
    if let Some(kids) = children.get(at) {
        let mut sorted = kids.clone();
        sorted.sort();
        sorted.dedup();
        for kid in sorted {
            out.push_str(&format!("pub mod {};\n", kid));
        }
    }
    out
}

/// A package whose `__init__` the crate does not emit (nothing imports
/// it — the entry reaches `net.manager` straight through the package)
/// still needs its `mod.rs`, or rustc finds no file for `pub mod net;`
/// and every submodule under it is silently dropped: write one holding
/// only the `pub mod child;` declarations for each such container.
fn write_missing_containers(
    src_dir: &Path,
    children: &BTreeMap<Vec<String>, Vec<String>>,
    written: &HashSet<Vec<String>>,
) -> Result<()> {
    for path in children.keys() {
        if path.is_empty() || written.contains(path) {
            continue;
        }
        let mut dir = src_dir.to_path_buf();
        for part in path {
            dir = dir.join(part);
        }
        fs::create_dir_all(&dir)?;
        let file = dir.join("mod.rs");
        fs::write(&file, format_rust(&mod_decls(children, path, true)))
            .with_context(|| format!("writing {}", file.display()))?;
    }
    Ok(())
}

/// Where a module's Rust file lives within src/.
fn module_file_path(src_dir: &Path, module: &PyModule) -> PathBuf {
    if module.path.is_empty() {
        return src_dir.join("lib.rs");
    }
    if module.is_init {
        let mut dir = src_dir.to_path_buf();
        for part in &module.path {
            dir = dir.join(part);
        }
        return dir.join("mod.rs");
    }
    let (dirs, name) = module.path.split_at(module.path.len() - 1);
    let mut dir = src_dir.to_path_buf();
    for part in dirs {
        dir = dir.join(part);
    }
    dir.join(format!("{}.rs", name[0]))
}

/// Generate the PyO3 bindings module: wrappers for every function whose
/// signature is expressible in concrete Rust types. Wrapper identifiers are
/// qualified by module path so same-named functions in different modules
/// don't collide in the flat bindings file; the Python-visible name stays
/// the bare function name when it is unique across the package, and falls
/// back to the qualified name (with a conversion warning — it's a visible
/// rename) when it isn't.
fn generate_bindings(
    package: &PyPackage,
    transpiled: &[(&PyModule, String)],
    warnings: &mut Vec<String>,
) -> Result<String> {
    type Signature = (Vec<TokenStream>, Vec<TokenStream>, Option<TokenStream>);
    let mut candidates: Vec<(&PyModule, python_ast::FunctionDef, Signature)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for (module, _) in transpiled {
        if is_dunder_main(module) {
            continue;
        }
        let ast = parse_enhanced(&module.source, parse_filename(module))
            .map_err(|e| anyhow::anyhow!("{} ({})", e, module.file.display()))?;

        for stmt in &ast.raw.body {
            let StatementType::FunctionDef(func) = &stmt.statement else {
                continue;
            };
            if func.name.starts_with('_') {
                continue;
            }
            match bindable_signature(func) {
                Some(sig) => candidates.push((module, func.clone(), sig)),
                None => skipped.push(format!("{}.{}", module.path.join("."), func.name)),
            }
        }
    }

    let mut name_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, func, _) in &candidates {
        *name_counts.entry(func.name.as_str()).or_default() += 1;
    }

    let mut wrappers: Vec<TokenStream> = Vec::new();
    let mut registrations: Vec<TokenStream> = Vec::new();
    let mut collisions: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (module, func, (params, arg_names, ret)) in &candidates {
        let bare = func.name.as_str();
        let qualified = if module.path.is_empty() {
            bare.to_string()
        } else {
            format!("{}_{}", module.path.join("_"), bare)
        };
        let wrapper_name = safe_ident(&qualified);
        let target = safe_ident(bare);
        let path: Vec<_> = module.path.iter().map(|p| safe_ident(p)).collect();
        let call = quote!(crate::#(#path::)*#target(#(#arg_names),*));
        // Generated functions return Result<T, PyException>; the wrapper
        // maps a raised exception onto the corresponding real Python
        // exception class (From<PyException> for PyErr in stdpython).
        let ret_tokens = match ret {
            Some(ty) => quote!(-> pyo3::PyResult<#ty>),
            None => quote!(-> pyo3::PyResult<()>),
        };
        let body = quote!(#call.map_err(pyo3::PyErr::from));
        // Keep the bare Python-visible name when it's unambiguous; a
        // package-wide duplicate keeps the qualified name (registering two
        // same-named functions would silently shadow one of them).
        let py_name = if name_counts[bare] == 1 {
            quote!(#[pyo3(name = #bare)])
        } else {
            collisions.entry(bare).or_default().push(qualified.clone());
            quote!()
        };
        wrappers.push(quote! {
            #[pyfunction]
            #py_name
            fn #wrapper_name(#(#params),*) #ret_tokens {
                #body
            }
        });
        registrations.push(quote! {
            m.add_function(wrap_pyfunction!(#wrapper_name, m)?)?;
        });
    }

    for (bare, qualified) in &collisions {
        warnings.push(format!(
            "python bindings: {} modules define a function named `{}`; they are \
             exposed to Python under module-qualified names (`{}`) because a \
             module cannot hold two same-named functions",
            qualified.len(),
            bare,
            qualified.join("`, `"),
        ));
    }

    if wrappers.is_empty() {
        bail!(
            "no functions with bindable signatures found; annotate parameters \
             with int/float/str/bool types to expose them to Python{}",
            if skipped.is_empty() {
                String::new()
            } else {
                format!(" (skipped: {})", skipped.join(", "))
            }
        );
    }

    let module_name = safe_ident(&package.name);
    let skipped_note = if skipped.is_empty() {
        String::new()
    } else {
        format!(
            "//! Skipped (signature not expressible in concrete Rust types yet): {}\n",
            skipped.join(", ")
        )
    };
    let bindings = quote! {
        use pyo3::prelude::*;

        #(#wrappers)*

        #[pymodule]
        fn #module_name(m: &Bound<'_, PyModule>) -> PyResult<()> {
            #(#registrations)*
            Ok(())
        }
    };
    Ok(format!(
        "//! PyO3 bindings generated by rypip.\n{}{}",
        skipped_note, bindings
    ))
}

/// If the function's signature maps to concrete Rust types, return
/// (parameter tokens, argument names, return type tokens).
#[allow(clippy::type_complexity)]
fn bindable_signature(
    func: &python_ast::FunctionDef,
) -> Option<(Vec<TokenStream>, Vec<TokenStream>, Option<TokenStream>)> {
    // Keep to plain positional parameters without defaults: *args/**kwargs
    // and positional-only parameters add generated parameters this simple
    // wrapper doesn't model, and defaulted parameters would lose their
    // Python-side optionality (a #[pyo3(signature = ...)] attribute could
    // lift that restriction later).
    if func.args.vararg.is_some()
        || func.args.kwarg.is_some()
        || !func.args.kwonlyargs.is_empty()
        || !func.args.posonlyargs.is_empty()
        || !func.args.defaults.is_empty()
    {
        return None;
    }
    let mut params = Vec::new();
    let mut names = Vec::new();
    for param in &func.args.args {
        let annotation = param.annotation.as_deref()?;
        let ty = python_annotation_to_rust_type(annotation)?;
        let name = safe_ident(&param.arg);
        params.push(quote!(#name: #ty));
        names.push(quote!(#name));
    }
    // The wrapper's return type must be exactly what the generated function
    // carries — resolved_return_type is the single source of truth (it gates
    // annotations on all-paths-return, so a function that can fall through
    // binds as returning unit, matching the generated `()`).
    let ret = func.resolved_return_type(
        &python_ast::SymbolTableScopes::new(),
        &python_ast::PythonOptions::default(),
    );
    Some((params, names, ret))
}

/// Write the generated crate's Cargo.toml. `manifest` carries the device
/// manifest: kernel device generation swaps the stdpython dependency for
/// the rykernel-shim crate (the device code links it directly) and takes
/// its crate name from `__module_name__` when declared.
/// Normalize a Python package version into a valid Cargo semver:
/// pyproject "0.1" (valid for Python) becomes "0.1.0".
fn cargo_version(v: &str) -> String {
    let parts: Vec<&str> = v.trim().split('.').collect();
    let mut out = parts[..parts.len().min(3)].to_vec();
    while out.len() < 3 {
        out.push("0");
    }
    out.join(".")
}

fn write_cargo_toml(
    package: &PyPackage,
    python_deps: &[(String, PyPackage)],
    reachable: &std::collections::HashSet<Vec<String>>,
    out_dir: &Path,
    opts: &ConvertOptions,
    has_binary: bool,
    manifest: &DeviceManifest,
    uses_shim: bool,
) -> Result<()> {
    // Async usage drives two Cargo.toml decisions: a BINARY crate with async
    // code links tokio behind a default-on `async-tokio` feature (the codegen
    // gates its entry attribute and `use tokio;` on that feature), and any
    // crate that imports asyncio enables stdpython's tokio-backed asyncio
    // module.
    let (uses_async, imports_asyncio) = package_async_flags(package);
    let async_binary = has_binary && uses_async;
    // stdpython's tokio-backed asyncio module is only needed when the code
    // actually imports asyncio (plain async fns need no runtime module).
    let needs_async_stdpython = imports_asyncio;

    let crate_name = if opts.kernel_module {
        // Kernel modules take their name from the kernel manifest; the
        // userspace driver keeps the package name (the glue and the
        // integration tests reference it).
        manifest
            .module_name
            .clone()
            .unwrap_or_else(|| package.name.clone())
    } else {
        package.name.clone()
    };
    // The generated device links the shim; so does a plain kernel module
    // whose Python imports kernel resources from rykernel_shim.
    let kernel_device = opts.kernel_module && (manifest.has_device || uses_shim);
    let stdpython_source = resolve_stdpython_source(opts)?;
    let stdpython_dep = if kernel_device {
        // The shim is not published: resolve a checkout next to this tool,
        // an explicit override, or fail loudly.
        let shim = resolve_rykernel_shim_path().ok_or_else(|| {
            anyhow::anyhow!(
                "--kernel-module needs the rykernel-shim crate (device generation or \
                 rykernel_shim imports link it): set RYPIP_RYKERNEL_SHIM_PATH to \
                 its checkout, or run rypip from the rython workspace \
                 (crates/rykernel-shim)"
            )
        })?;
        format!(
            "rykernel-shim = {{ path = \"{}\" }}",
            shim.display().to_string().replace('\\', "/")
        )
    } else if opts.rust_for_linux {
        "# kernel = { path = \"/path/to/linux/rust/kernel\" }\n\
         # The kernel crate provides the module! macro and pr_info!; point\n\
         # the path at your rust-for-linux tree's rust/kernel directory."
            .to_string()
    } else if opts.kernel_module {
        match &stdpython_source {
            StdpythonSource::Path(path) => format!(
                "stdpython = {{ path = \"{}\", default-features = false, features = [\"alloc\"] }}",
                path.display().to_string().replace('\\', "/"),
            ),
            StdpythonSource::Registry(version) => format!(
                "stdpython = {{ version = \"{}\", default-features = false, features = [\"alloc\"] }}",
                version,
            ),
        }
    } else {
        // The std tier of stdpython plus exactly the feature-gated
        // surfaces the package actually uses: the tokio-backed asyncio
        // module (async code or `import asyncio`) and the surfaces
        // `package_surface_features` reads off the imports.
        //
        // The list is emitted with `default-features = false` rather than
        // riding stdpython's defaults, so a package that never imports
        // `ssl` or `re` does not compile rustls or the regex engine —
        // together about two thirds of the std tier's dependency build.
        // stdpython's own `default` still carries them, so nothing else
        // that depends on it is affected.
        let mut features: Vec<&str> = vec!["std"];
        if needs_async_stdpython {
            features.push("async-tokio");
        }
        // `--pyo3` bindings call `.map_err(pyo3::PyErr::from)` on the
        // generated wrappers, which needs stdpython's `From<PyException>
        // for pyo3::PyErr`. Nothing else does, and pyo3 (with
        // auto-initialize) is the std tier's most expensive dependency.
        if opts.pyo3 {
            features.push("pyo3-interop");
        }
        features.extend(
            package_surface_features(package, python_deps, reachable)
                .into_iter()
                .map(SurfaceFeature::cargo_feature),
        );
        let feature_list = format!(
            ", default-features = false, features = [{}]",
            features
                .iter()
                .map(|f| format!("\"{}\"", f))
                .collect::<Vec<_>>()
                .join(", ")
        );
        match (&stdpython_source, opts.no_std) {
            (StdpythonSource::Path(path), true) => format!(
                "stdpython = {{ path = \"{}\", default-features = false, features = [\"alloc\"] }}",
                path.display().to_string().replace('\\', "/"),
            ),
            (StdpythonSource::Path(path), false) => format!(
                "stdpython = {{ path = \"{}\"{} }}",
                path.display().to_string().replace('\\', "/"),
                feature_list,
            ),
            (StdpythonSource::Registry(version), true) => format!(
                "stdpython = {{ version = \"{}\", default-features = false, features = [\"alloc\"] }}",
                version,
            ),
            (StdpythonSource::Registry(version), false) => {
                format!("stdpython = {{ version = \"{}\"{} }}", version, feature_list)
            }
        }
    };
    let mut toml = format!(
        "# Generated by rypip from a Python package. Edit freely.\n\
         [package]\n\
         name = \"{name}\"\n\
         version = \"{version}\"\n\
         edition = \"2021\"\n\n\
         [dependencies]\n\
         {stdpython_dep}\n",
        name = crate_name,
        version = cargo_version(&package.version),
    );
    // Userspace drivers link libc for their generated syscall glue.
    if opts.driver {
        toml.push_str("libc = \"0.2\"\n");
    }
    // no_std and kernel-module targets must not unwind — there is no
    // unwinding runtime in embedded, wasm, or kernel contexts.
    if opts.no_std || opts.kernel_module {
        toml.push_str(
            "\n[profile.dev]\npanic = \"abort\"\n\n[profile.release]\npanic = \"abort\"\n",
        );
    }
    // Crates declared via `from rython import rust` bindings become
    // dependencies of the generated crate. The declaration's `path=` /
    // `version=` spec is copied verbatim (paths are relative to the
    // generated crate's Cargo.toml).
    for (name, path, version) in rust_bind_dependencies(package) {
        let entry = match (path, version) {
            (Some(p), _) => format!("{name} = {{ path = \"{}\" }}", p),
            (_, Some(v)) => format!("{name} = \"{v}\""),
            (None, None) => continue,
        };
        toml.push_str(&format!("{entry}\n"));
    }
    // Import-based rust-module deps from the rython.toml manifest.
    let rust_manifest = read_rust_module_manifest(package)?;
    for (name, path, version) in rust_module_dependencies(package, &rust_manifest, out_dir) {
        let entry = match (path, version) {
            (Some(p), _) => format!("{name} = {{ path = \"{}\" }}", p),
            (_, Some(v)) => format!("{name} = \"{v}\""),
            (None, None) => continue,
        };
        toml.push_str(&format!("{entry}\n"));
    }
    if opts.pyo3 {
        // pyo3-build-config lets the generated build script request pyo3's
        // extension-module link args (see below): since pyo3 0.23+ its own
        // build scripts stopped emitting them, and on macOS the linker then
        // rejects the undefined `_Py_*` symbols that must resolve against
        // whichever interpreter loads the module.
        toml.push_str(
            "pyo3 = { version = \"0.29\", features = [\"extension-module\"], optional = true }\n\n\
             [build-dependencies]\n\
             pyo3-build-config = { version = \"0.29\", optional = true }\n\n\
             [features]\n\
             python = [\"dep:pyo3\", \"dep:pyo3-build-config\"]\n\n\
             [lib]\n\
             crate-type = [\"lib\", \"cdylib\"]\n",
        );
    }
    if opts.kernel_module {
        toml.push_str("\n[lib]\ncrate-type = [\"staticlib\"]\n");
    }
    // Async BINARY crates link the runtime (tokio) behind a default-on
    // `async-tokio` feature: the generated entry point carries
    // `#[cfg_attr(feature = "async-tokio", tokio::main)]` and a
    // compile_error that names the feature when it is off
    // (`--no-default-features`). Library conversions never declare tokio:
    // their async functions are plain async fns, driven by the consumer's
    // executor.
    if async_binary {
        toml.push_str(
            "tokio = { version = \"1\", features = [\"macros\", \"rt-multi-thread\"], optional = true }\n\n\
             [features]\n\
             default = [\"async-tokio\"]\n\
             async-tokio = [\"dep:tokio\"]\n\n",
        );
    }
    fs::write(out_dir.join("Cargo.toml"), toml)?;
    // PyO3 bindings need a build script: pyo3 requires the downstream
    // crate to ask for its extension-module link args (a macOS linker
    // rejects the cdylib's undefined `_Py_*` symbols otherwise). The
    // helper is a no-op on platforms that don't need it.
    if opts.pyo3 {
        fs::write(
            out_dir.join("build.rs"),
            concat!(
                "//! Linker setup for the PyO3 bindings (built with `--features python`).\n",
                "//!\n",
                "//! pyo3 requires the downstream build script to request its\n",
                "//! extension-module link args: on macOS the linker otherwise rejects\n",
                "//! the undefined `_Py_*` symbols that must resolve against the\n",
                "//! interpreter process loading the module.\n",
                "\n",
                "fn main() {\n",
                "    #[cfg(feature = \"python\")]\n",
                "    pyo3_build_config::add_extension_module_link_args();\n",
                "}\n",
            ),
        )?;
    }
    // Keep the generated crate out of any enclosing workspace.
    let manifest = out_dir.join("Cargo.toml");
    let mut text = fs::read_to_string(&manifest)?;
    text.push_str("\n[workspace]\n");
    fs::write(manifest, text)?;
    Ok(())
}

/// Crate dependencies declared by `from rython import rust` bindings across
/// the package's modules, deduplicated by crate name (first declaration's
/// spec wins). Parse failures here are ignored: the transpile pass reports
/// them with full context.
fn rust_bind_dependencies(package: &PyPackage) -> Vec<(String, Option<String>, Option<String>)> {
    let mut seen = std::collections::HashSet::new();
    let mut deps = Vec::new();
    for module in &package.modules {
        if let Ok(ast) = parse_enhanced(&module.source, parse_filename(module)) {
            for spec in python_ast::rust_bind_specs(&ast.raw.body) {
                if seen.insert(spec.crate_name.clone()) {
                    deps.push((spec.crate_name, spec.path, spec.version));
                }
            }
        }
    }
    deps
}

// ---------------------------------------------------------------------------
// Rust modules: `import` / `from ... import` bindings into Rust crates
// ---------------------------------------------------------------------------

/// A manifest entry from `rython.toml` `[rust-modules]`.
#[derive(Clone, Debug)]
struct ManifestRustModule {
    /// The actual crate name (Cargo dependency name). Defaults to the
    /// import name.
    crate_name: String,
    /// Dependency spec: at most one of these is set.
    path: Option<String>,
    version: Option<String>,
}

/// Read `rython.toml` from the package root, if present. Returns the
/// `[rust-modules]` table keyed by import name. Missing manifest is not an
/// error — it simply means no import-based Rust bindings.
fn read_rust_module_manifest(package: &PyPackage) -> Result<HashMap<String, ManifestRustModule>> {
    let manifest_path = package.root.join("rython.toml");
    let text = match fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(e) => bail!("reading {}: {}", manifest_path.display(), e),
    };
    let value: toml::Value =
        toml::from_str(&text).with_context(|| format!("parsing {}", manifest_path.display()))?;
    let Some(table) = value.get("rust-modules").and_then(|v| v.as_table()) else {
        return Ok(HashMap::new());
    };
    let mut out = HashMap::new();
    for (import_name, entry) in table {
        let t = entry.as_table().with_context(|| {
            format!(
                "{}: `{import_name}` must be a table",
                manifest_path.display()
            )
        })?;
        let crate_name = t
            .get("crate")
            .and_then(|v| v.as_str())
            .unwrap_or(import_name)
            .to_string();
        let path = t.get("path").and_then(|v| v.as_str()).map(str::to_string);
        let version = t
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        if path.is_some() && version.is_some() {
            bail!(
                "{}: `{import_name}` declares both path and version; give one",
                manifest_path.display()
            );
        }
        out.insert(
            import_name.clone(),
            ManifestRustModule {
                crate_name,
                path,
                version,
            },
        );
    }
    Ok(out)
}

/// A manifest entry from `rython.toml` `[python-modules]` (or a resolved
/// PyPI dependency): a vendored PyPI-style Python library, compiled into
/// the generated crate.
#[derive(Clone, Debug)]
pub(crate) struct ManifestPythonModule {
    /// Path to a single .py file or a package directory, relative to the
    /// package root (where the manifest lives).
    pub(crate) path: String,
}

/// Read `rython.toml` `[python-modules]`, keyed by the Python import name
/// (e.g. `pylev = { path = "vendor/pylev/wf.py" }`). Missing manifest or
/// table is not an error — it simply means no Python-library imports.
fn read_python_module_manifest(
    package: &PyPackage,
) -> Result<HashMap<String, ManifestPythonModule>> {
    let manifest_path = package.root.join("rython.toml");
    let text = match fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(e) => bail!("reading {}: {}", manifest_path.display(), e),
    };
    let value: toml::Value =
        toml::from_str(&text).with_context(|| format!("parsing {}", manifest_path.display()))?;
    let Some(table) = value.get("python-modules").and_then(|v| v.as_table()) else {
        return Ok(HashMap::new());
    };
    let mut out = HashMap::new();
    for (import_name, entry) in table {
        let t = entry.as_table().with_context(|| {
            format!(
                "{}: `{import_name}` must be a table",
                manifest_path.display()
            )
        })?;
        let path = t
            .get("path")
            .and_then(|v| v.as_str())
            .with_context(|| {
                format!(
                    "{}: `{import_name}` needs a `path` (a .py file or package directory)",
                    manifest_path.display()
                )
            })?
            .to_string();
        out.insert(
            import_name.clone(),
            ManifestPythonModule {
                path: path.clone(),
            },
        );
    }
    Ok(out)
}

/// Resolve the packaging metadata's declared dependencies (pyproject
/// `[project] dependencies` / setup.cfg `install_requires` / setup.py) into
/// `[python-modules]`-style entries, fetching from PyPI unless the
/// dependency is already vendored explicitly (explicit entries win) or
/// `RYPIP_OFFLINE=1` is set (then a missing dep is a loud error).
/// The module paths reachable from the program's roots (the package's own
/// modules) by following `import`/`from ... import` statements across the
/// package and its dependencies — absolute and relative. Unreachable
/// modules (CLI helpers, `__main__` blocks, test utilities) are skipped by
/// the converter, so a divergent module the program never imports cannot
/// block conversion.
fn reachable_module_paths(
    package: &PyPackage,
    deps: &[(String, PyPackage)],
) -> std::collections::HashSet<Vec<String>> {
    
    use std::collections::{HashMap, HashSet, VecDeque};

    let mut by_path: HashMap<Vec<String>, &PyModule> = HashMap::new();
    for (_, dep) in deps {
        for m in &dep.modules {
            by_path.insert(m.path.clone(), m);
        }
    }
    for m in &package.modules {
        by_path.insert(m.path.clone(), m);
    }

    let mut reachable: HashSet<Vec<String>> = HashSet::new();
    let mut queue: VecDeque<Vec<String>> = VecDeque::new();
    for m in &package.modules {
        queue.push_back(m.path.clone());
    }

    while let Some(path) = queue.pop_front() {
        if !reachable.insert(path.clone()) {
            continue;
        }
        let Some(module) = by_path.get(&path) else { continue };
        // Parse with the RELATIVE module filename (like `transpile` does):
        // parse_enhanced derives the module name from the filename, and an
        // absolute path produces an invalid identifier (`__home__tserica__...`)
        // which fails parsing and silently drops every import of the module.
        let Ok(ast) = python_ast::parse_enhanced(&module.source, parse_filename(module)) else {
            continue;
        };
        for stmt in &ast.raw.body {
            reachable_walk_imports(stmt, module, &path, &by_path, &mut queue);
        }
    }
    reachable
}

/// Walk one statement for imports to enqueue, recursing into nested
/// statement lists (if/while/for/with/try bodies) AND function bodies:
/// `from pip._internal.utils.entrypoints import _wrapper` inside `main()`
/// (a function-local import) still makes `pip._internal...` reachable.
/// Function bodies are walked with the function's own package path — a
/// function-local import is relative to the module's package, not to the
/// function.
fn reachable_walk_imports(
    stmt: &python_ast::ast::tree::Statement,
    module: &PyModule,
    path: &[String],
    by_path: &HashMap<Vec<String>, &PyModule>,
    queue: &mut VecDeque<Vec<String>>,
) {
    use python_ast::ast::tree::StatementType;

    match &stmt.statement {
        StatementType::Import(i) => {
            for alias in &i.names {
                let parts: Vec<&str> = alias.name.split('.').collect();
                for len in (1..=parts.len()).rev() {
                    let cand: Vec<String> =
                        parts[..len].iter().map(|s| s.to_string()).collect();
                    if by_path.contains_key(&cand) {
                        queue.push_back(cand);
                        break;
                    }
                }
            }
        }
        StatementType::ImportFrom(i) => {
            let package_path: Vec<String> = if module.is_init {
                path.to_vec()
            } else {
                path[..path.len().saturating_sub(1)].to_vec()
            };
            // Absolute imports (level == 0) resolve from the crate
            // root; relative imports walk up `level` packages from
            // the current package path.
            let mut base: Vec<String> = if i.level == 0 {
                Vec::new()
            } else {
                let cut = (package_path.len() + 1).saturating_sub(i.level);
                package_path[..cut.min(package_path.len())].to_vec()
            };
            if i.module.is_empty() {
                // `from . import utils` — the names are sibling
                // modules of the resolved package.
                for alias in &i.names {
                    let mut cand = base.clone();
                    cand.push(alias.name.clone());
                    if by_path.contains_key(&cand) {
                        queue.push_back(cand);
                    }
                }
            } else {
                base.extend(
                    i.module.split('.').filter(|p| !p.is_empty()).map(|s| s.to_string()),
                );
                for len in (1..=base.len()).rev() {
                    let cand = base[..len].to_vec();
                    if by_path.contains_key(&cand) {
                        queue.push_back(cand);
                        break;
                    }
                }
                // `from .http2 import probe as http2_probe` — the imported
                // NAMES may be SUBMODULES of the resolved package
                // (http2/probe.py), not just items of its __init__: enqueue
                // each `base + name` that is a module (urllib3's
                // connection.py uses probe's acquire_and_get). Without
                // this, probe.py is never transpiled and the call fails
                // E0433.
                for alias in &i.names {
                    let mut cand = base.clone();
                    cand.push(alias.name.clone());
                    if by_path.contains_key(&cand) {
                        queue.push_back(cand);
                    }
                }
            }
        }
        _ => {}
    }
    // Recurse into nested statement lists and function bodies.
    match &stmt.statement {
        StatementType::FunctionDef(f) => {
            for b in &f.body {
                reachable_walk_imports(b, module, path, by_path, queue);
            }
        }
        StatementType::If(s) => {
            for b in s.body.iter().chain(s.orelse.iter()) {
                reachable_walk_imports(b, module, path, by_path, queue);
            }
        }
        StatementType::While(s) => {
            for b in s.body.iter().chain(s.orelse.iter()) {
                reachable_walk_imports(b, module, path, by_path, queue);
            }
        }
        StatementType::For(s) => {
            for b in s.body.iter().chain(s.orelse.iter()) {
                reachable_walk_imports(b, module, path, by_path, queue);
            }
        }
        StatementType::With(s) => {
            for b in &s.body {
                reachable_walk_imports(b, module, path, by_path, queue);
            }
        }
        StatementType::Try(s) => {
            for b in s.body.iter().chain(s.orelse.iter()).chain(s.finalbody.iter()) {
                reachable_walk_imports(b, module, path, by_path, queue);
            }
            for h in &s.handlers {
                for b in &h.body {
                    reachable_walk_imports(b, module, path, by_path, queue);
                }
            }
        }
        StatementType::ClassDef(c) => {
            for b in &c.body {
                reachable_walk_imports(b, module, path, by_path, queue);
            }
        }
        _ => {}
    }
}

fn resolve_packaging_dependencies(
    package: &PyPackage,
    explicit: HashMap<String, ManifestPythonModule>,
) -> Result<HashMap<String, ManifestPythonModule>> {
    let requirements = crate::resolve::parse_requirements(&package.dependencies);
    if requirements.is_empty() {
        return Ok(explicit);
    }
    let offline = std::env::var_os("RYPIP_OFFLINE").is_some();
    let mut resolved = Vec::new();
    for req in &requirements {
        if explicit.contains_key(&req.name) {
            continue;
        }
        // Issue #113: resolve the dependency AND its transitive
        // requirements (pip-style), so `requests` brings in urllib3,
        // certifi, idna, charset-normalizer, ...
        let tree = crate::resolve::resolve_dependency_tree(req, offline)
            .with_context(|| format!("resolving dependency `{}`", req.name))?;
        resolved.extend(tree);
    }
    Ok(crate::resolve::merge_python_modules(explicit, resolved))
}

/// Load every `[python-modules]` dependency into an in-memory `PyPackage`,
/// with module paths remapped under the import name: a single file
/// `vendor/pylev/wf.py` becomes the one module `pylev`; a package directory
/// `vendor/pkg/` with `__init__.py` becomes `pkg`, `pkg.sub`, ... The
/// returned packages are keyed by import name.
fn python_module_deps(
    package: &PyPackage,
    manifest: &HashMap<String, ManifestPythonModule>,
) -> Result<Vec<(String, PyPackage)>> {
    let mut deps = Vec::new();
    for (import_name, entry) in manifest {
        let abs = package.root.join(&entry.path);
        let abs = abs.canonicalize().with_context(|| {
            format!(
                "python-module `{import_name}`: cannot access {}",
                abs.display()
            )
        })?;
        if abs.is_file() {
            let source = fs::read_to_string(&abs).with_context(|| {
                format!("python-module `{import_name}`: reading {}", abs.display())
            })?;
            let name = sanitize_name(import_name);
            let file = abs.clone();
            let module = PyModule {
                path: vec![name.clone()],
                source,
                file,
                is_init: false,
            };
            let pkg = PyPackage {
                name: name.clone(),
                version: "0.0.0".to_string(),
                root: abs
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from(".")),
                modules: vec![module],
                dependencies: Vec::new(),
            };
            deps.push((import_name.clone(), pkg));
        } else if abs.is_dir() {
            if !abs.join("__init__.py").is_file() {
                bail!(
                    "python-module `{import_name}`: {} is a directory without \
                     __init__.py; vendored packages must be importable packages",
                    abs.display()
                );
            }
            let mut pkg = crate::discover(&abs).with_context(|| {
                format!("python-module `{import_name}`: discovering {}", abs.display())
            })?;
            let prefix = sanitize_name(import_name);
            for module in &mut pkg.modules {
                module.path.insert(0, prefix.clone());
                // The package's own __init__ was discovered as the root
                // (empty path); after prefixing it IS the package init.
                if module.path.len() == 1 && module.file.file_name() == Some("__init__.py".as_ref())
                {
                    module.is_init = true;
                }
            }
            deps.push((import_name.clone(), pkg));
        } else {
            bail!(
                "python-module `{import_name}`: {} is neither a file nor a directory",
                abs.display()
            );
        }
    }
    Ok(deps)
}

/// The package path of a module for relative-import resolution:
/// `pkg/sub/mod.py` resolves `from . import x` against `pkg/sub`. An
/// `__init__.py` IS its package, so `pkg/sub/__init__.py` resolves against
/// `pkg/sub`. The crate root (`path == []`) resolves against itself.
fn module_package_path(module: &PyModule) -> Vec<String> {
    if module.is_init {
        module.path.clone()
    } else {
        module.path[..module.path.len().saturating_sub(1)].to_vec()
    }
}

/// The names a module imports that could be Rust modules: `import crc32c`
/// (and `import crc32c as c`) plus `from crc32c import ...`.
fn imported_names(source: &str, filename: &str) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(ast) = parse_enhanced(source, filename) else {
        return names;
    };
    for stmt in &ast.raw.body {
        match &stmt.statement {
            StatementType::Import(imp) => {
                for alias in &imp.names {
                    // Only the top-level name can be a Rust module.
                    let root = alias.name.split('.').next().unwrap_or(&alias.name);
                    if root != "rython" && root != "rypython" {
                        names.push(root.to_string());
                    }
                }
            }
            StatementType::ImportFrom(fr) => {
                if fr.level == 0 && fr.module != "rython" && fr.module != "rypython" {
                    names.push(fr.module.clone());
                }
            }
            _ => {}
        }
    }
    names
}

/// Build the rust-module registry for a package: every import name that the
/// package actually uses and that the manifest maps to a crate. Each name
/// resolves to a `RustModuleSpec` from a `.pyi` stub (preferred) or by
/// inference from the crate source. Names imported but absent from the
/// manifest are deliberately NOT resolved here — they fall through to the
/// normal import path (stdlib, sibling module, ...).
fn rust_module_registry(
    package: &PyPackage,
    manifest: &HashMap<String, ManifestRustModule>,
) -> Result<HashMap<String, python_ast::RustModuleSpec>> {
    // Which names are actually imported anywhere in the package?
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    for module in &package.modules {
        for name in imported_names(&module.source, &parse_filename(module)) {
            if manifest.contains_key(&name) {
                used.insert(name);
            }
        }
    }

    let mut registry = HashMap::new();
    for import_name in used {
        let entry = &manifest[&import_name];
        let spec = python_ast::resolve_rust_module(
            &import_name,
            &entry.crate_name,
            entry.path.as_deref(),
            entry.version.as_deref(),
            &package.root,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        registry.insert(import_name.clone(), spec);
    }
    Ok(registry)
}

/// Crate dependencies for the package's rust-module imports (path/version
/// from the manifest), deduplicated by crate name. Manifest `path`s are
/// relative to the package root (where the manifest lives); they are
/// rebased onto `out_dir` — the generated crate's Cargo.toml — so the
/// copied dep spec resolves from there.
fn rust_module_dependencies(
    package: &PyPackage,
    manifest: &HashMap<String, ManifestRustModule>,
    out_dir: &Path,
) -> Vec<(String, Option<String>, Option<String>)> {
    let mut seen = std::collections::HashSet::new();
    let mut deps = Vec::new();
    for module in &package.modules {
        for name in imported_names(&module.source, &parse_filename(module)) {
            if let Some(entry) = manifest.get(&name) {
                if seen.insert(entry.crate_name.clone()) {
                    let path = entry
                        .path
                        .as_deref()
                        .map(|p| rebase_path(&package.root, out_dir, p));
                    deps.push((entry.crate_name.clone(), path, entry.version.clone()));
                }
            }
        }
    }
    deps
}

/// Rebase `p` — a path relative to `base` — onto `to`, so the same file is
/// addressed from `to`'s directory. Falls back to the original when either
/// side cannot be absolutized.
fn rebase_path(base: &Path, to: &Path, p: &str) -> String {
    let abs = base.join(p);
    let abs = abs.canonicalize().unwrap_or(abs);
    let to = to.canonicalize().unwrap_or_else(|_| to.to_path_buf());
    let mut to_components: Vec<_> = to.components().collect();
    let mut abs_components: Vec<_> = abs.components().collect();
    while !to_components.is_empty()
        && !abs_components.is_empty()
        && to_components[0] == abs_components[0]
    {
        to_components.remove(0);
        abs_components.remove(0);
    }
    let mut out = PathBuf::new();
    for _ in &to_components {
        out.push("..");
    }
    for c in abs_components {
        out.push(c.as_os_str());
    }
    out.to_string_lossy().into_owned()
}

/// Where the generated crate's stdpython dependency comes from.
enum StdpythonSource {
    /// A local checkout (explicit flag, env var, or this tool's own
    /// source tree when running from it).
    Path(PathBuf),
    /// The crates.io release matching this rypip's version — the
    /// default for an INSTALLED rypip, where the source-tree path baked
    /// in at build time does not exist on the user's machine.
    Registry(String),
}

/// Locate the stdpython crate the generated code depends on: an explicit
/// option, the RYPIP_STDPYTHON_PATH environment variable, the copy that
/// ships alongside this tool's own source tree, or (for an installed
/// rypip) the published release with rypip's own version — the workspace
/// versions move in lockstep, so the pair is always compatible.
fn resolve_stdpython_source(opts: &ConvertOptions) -> Result<StdpythonSource> {
    if let Some(path) = &opts.stdpython_path {
        return path
            .canonicalize()
            .map(StdpythonSource::Path)
            .with_context(|| format!("stdpython path {} not found", path.display()));
    }
    if let Ok(env_path) = std::env::var("RYPIP_STDPYTHON_PATH") {
        return PathBuf::from(&env_path)
            .canonicalize()
            .map(StdpythonSource::Path)
            .with_context(|| format!("RYPIP_STDPYTHON_PATH {} not found", env_path));
    }
    let built_in = Path::new(env!("CARGO_MANIFEST_DIR")).join("../stdpython");
    if let Ok(path) = built_in.canonicalize() {
        return Ok(StdpythonSource::Path(path));
    }
    Ok(StdpythonSource::Registry(
        env!("CARGO_PKG_VERSION").to_string(),
    ))
}

/// Locate the rykernel-shim crate that generated device modules link: an
/// explicit `RYPIP_RYKERNEL_SHIM_PATH`, or the checkout that ships
/// alongside this tool's own source tree (crates/rykernel-shim). The crate
/// is not published, so an installed rypip without a rython checkout has
/// no shim to point at — generation fails loudly in that case.
fn resolve_rykernel_shim_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("RYPIP_RYKERNEL_SHIM_PATH") {
        let p = PathBuf::from(env_path);
        if p.join("Cargo.toml").is_file() {
            return Some(p);
        }
    }
    let built_in = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rykernel-shim");
    built_in.canonicalize().ok()
}

/// Generate a C-free Kbuild Makefile and a pahole wrapper that build the
/// kernel module purely from Rust: no `{name}_shim.c`, no C anywhere.
///
/// Pipeline:
///   1. `cargo build --release --target x86_64-unknown-none` — the generated
///      crate is a `#![no_std]` staticlib (kmalloc allocator, `.modinfo`
///      sections, `extern "C"` entry points).
///   2. `ld -r --gc-sections -u init_module ...` — merge the staticlib into a
///      single relocatable object the module loader accepts. `--gc-sections`
///      prunes everything unreachable from the entry points; that is what
///      drops the PIC/GOTPCREL relocations (R_X86_64_GOTPCREL is rejected by
///      the kernel loader) that rust-std's precompiled objects carry.
///   3. Kbuild does modpost, BTF, and the final link. The `pahole-wrapper`
///      works around pahole emitting no types (and therefore no `.BTF.1`)
///      for pure-Rust objects when the kernel passes `--lang_exclude=rust`.
fn write_kernel_makefile(name: &str, out_dir: &Path, has_init: bool, has_exit: bool) -> Result<()> {
    let entry_flags = {
        let mut flags = Vec::new();
        if has_init {
            flags.push("-u init_module");
        }
        if has_exit {
            flags.push("-u cleanup_module");
        }
        flags.join(" ")
    };
    let makefile = format!(
        r#"# Kbuild Makefile for {name}.ko — generated by rypip --kernel-module.
#
# Pure-Rust module: no C shim. The module is a Rust staticlib linked into a
# single relocatable object that Kbuild can consume.
#
# Prerequisites:
#   - linux-headers for the running kernel
#   - nightly Rust (for -Zbuild-std; stable cannot rebuild core/alloc, whose
#     precompiled objects carry GOTPCREL relocations the module loader
#     rejects) plus the freestanding target:
#         rustup toolchain install nightly --profile minimal
#         rustup +nightly target add x86_64-unknown-none
#   - GNU ld >= 2.36 (--gc-sections in relocatable links)
#
# Build:  make
# Load:   sudo insmod {name}.ko
# Unload: sudo rmmod {name}
# Logs:   sudo dmesg | tail -5

KDIR ?= /lib/modules/$(shell uname -r)/build
CARGO ?= cargo
RUST_TARGET ?= x86_64-unknown-none
KOBJ := target/$(RUST_TARGET)/release/lib{name}.a
# Kernel modules must not use PIC addressing: the module loader rejects
# R_X86_64_GOTPCREL ("Unknown rela relocation: 9"). The static model makes
# every reference direct. code-model=kernel matches the kernel's own model;
# embed-bitcode=no keeps the .llvmbc sections out of the staticlib.
# -Zbuild-std rebuilds core/alloc/compiler_builtins with these flags — the
# precompiled ones from rustup are built PIC and keep their GOTPCRELs in
# reachable code (stdpython pulls alloc + the panic-message machinery).
RUSTFLAGS := -C relocation-model=static -C code-model=kernel -C embed-bitcode=no
BUILD_STD := -Zbuild-std=core,alloc,compiler_builtins

obj-m += {name}.o

all: rust link kbuild

# 1. Compile the module to a freestanding staticlib (no libc, no std),
#    rebuilding core/alloc/compiler_builtins under the static model.
rust:
	RUSTFLAGS="$(RUSTFLAGS)" $(CARGO) +nightly build $(BUILD_STD) --release --target $(RUST_TARGET)

# 2. Link the staticlib into a single relocatable object. --gc-sections
#    prunes everything unreachable from the module entry points (this is what
#    removes the PIC/GOTPCREL relocations the module loader rejects), and -u
#    keeps exactly the entry points the generated module defines. The
#    generated Rust emits .modinfo entries, so the module keeps its
#    license/author/description without any C.
link:
	ld -r --gc-sections {entry_flags} --whole-archive $(KOBJ) -o {name}.o
	objcopy --remove-section .llvmbc --remove-section .llvmcmd --strip-debug {name}.o
	@touch .{name}.o.cmd

# 3. Kbuild: modpost, BTF, and the final module. PAHOLE is passed on the
#    command line (command-line variables override the kbuild Makefile's
#    `PAHOLE = pahole`) and points at the wrapper that works around pahole
#    emitting no BTF for pure-Rust objects. CURDIR (make's own working
#    directory — this directory, even when driven from a parent make) is
#    used for M=, not PWD: a top-level `make module` exports PWD as the
#    parent directory, which would point kbuild at the wrong tree.
kbuild:
	$(MAKE) -C $(KDIR) LLVM=1 M=$(CURDIR) PAHOLE=$(CURDIR)/pahole-wrapper modules

clean:
	$(MAKE) -C $(KDIR) LLVM=1 M=$(CURDIR) clean
	$(CARGO) clean

.PHONY: all rust link kbuild clean
"#
    );

    let wrapper = r#"#!/bin/sh
# pahole-wrapper — generated by rypip --kernel-module.
#
# Workaround for pahole + pure-Rust modules on kernels built with BTF support
# (CONFIG_DEBUG_INFO_BTF_MODULES=y): the kernel's BTF Makefile passes
# --lang_exclude=rust, and pahole then emits NO types for an object whose only
# debug info is Rust. The kbuild BTF step fails with
#   "FAILED: load BTF from ... .BTF.1: No such file or directory"
#
# This wrapper runs the real pahole; if it produced nothing, it fabricates a
# valid empty BTF header so resolve_btfids/objcopy can proceed. Modules with
# no BTF types are valid; they simply carry no type info.
#
# The empty header is the 24-byte struct btf_header:
#   magic 0xeb9f, version 1, flags 0, hdr_len 24, type_off/len 0, str_off/len 0
# (little-endian; printf octal escapes — \ddd is at most 3 digits, so
# \237 = 0x9f and \353 = 0xeb).
PAHOLE_REAL="${PAHOLE_REAL:-pahole}"
out=""
for arg in "$@"; do
    case "$arg" in
        --btf_encode_detached=*) out="${arg#--btf_encode_detached=}" ;;
    esac
done
"$PAHOLE_REAL" "$@" || exit $?
if [ -n "$out" ] && [ ! -s "$out" ]; then
    printf '\237\353\001\000\030\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000' > "$out"
fi
"#;

    fs::write(out_dir.join("Makefile"), &makefile)?;
    fs::write(out_dir.join("pahole-wrapper"), wrapper)?;
    // Make sure the wrapper is executable in the generated tree.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::metadata(out_dir.join("pahole-wrapper"))?.permissions();
        let mut new_perms = perms.clone();
        new_perms.set_mode(perms.mode() | 0o755);
        fs::set_permissions(out_dir.join("pahole-wrapper"), new_perms)?;
    }
    Ok(())
}

/// Format generated Rust; fall back to the unformatted text if rustfmt is
/// unavailable or rejects the input.
fn format_rust(source: &str) -> String {
    RustFmt::default()
        .format_str(source)
        .unwrap_or_else(|_| source.to_string())
}
