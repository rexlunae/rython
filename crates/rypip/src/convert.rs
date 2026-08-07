//! Conversion of a discovered Python package into a Cargo crate: one Rust
//! module per Python module, an optional binary entry point, and optional
//! PyO3 bindings so the crate can be imported from Python again.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use proc_macro2::TokenStream;
use python_ast::{
    parse_enhanced, python_annotation_to_rust_type, safe_ident, CodeGen, CodeGenContext,
    PythonOptions, StatementType, SymbolTableScopes,
};
use quote::quote;
use rust_format::{Formatter, RustFmt};

use crate::package::{PyModule, PyPackage};

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
    if opts.no_std || opts.kernel_module { "#![no_std]\n" } else { "" }
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

/// Generate a kernel module lib.rs from Python source. This bypasses the
/// full transpiler in favour of a minimal lowering that produces raw `no_std`
/// Rust with kernel entry points, printk FFI, and module metadata.
fn generate_kernel_lib_rs(source: &str) -> Result<String> {
    use python_ast::parse_enhanced;
    use python_ast::ast::tree::StatementType;

    let ast = parse_enhanced(source, "<kernel>".to_string())
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // The kernel runs with the FPU in a lazy-save state: reject floating-point
    // usage loudly instead of emitting code that can corrupt userspace state.
    check_kernel_no_floats(&ast)?;

    let mut init_body = String::new();
    let mut exit_body = String::new();
    let mut has_printk = false;
    // Module metadata: key -> value (e.g. "license" -> "GPL").
    let mut modinfo: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::from([("license".into(), "GPL".into())]);

    // Scan top-level statements for module metadata and entry points.
    for stmt in &ast.raw.body {
        match &stmt.statement {
            StatementType::Assign(assign) => {
                if let (Some(target), Some(val_str)) =
                    (assign.targets.first(), expr_str_literal(&assign.value))
                {
                    if let python_ast::ExprType::Name(name) = target {
                        let key = match name.id.as_str() {
                            "__module_license__" => Some("license"),
                            "__module_author__" => Some("author"),
                            "__module_description__" => Some("description"),
                            _ => None,
                        };
                        if let Some(k) = key {
                            modinfo.insert(k.into(), val_str);
                        }
                    }
                }
            }
            StatementType::FunctionDef(func) => {
                let is_init = func.name == "module_init";
                let is_exit = func.name == "module_exit";
                if !is_init && !is_exit {
                    continue;
                }
                let body = lower_kernel_body(&func.body, &mut has_printk)?;
                if is_init {
                    init_body = body;
                } else {
                    exit_body = body;
                }
            }
            _ => {}
        }
    }

    // Warn if no entry point found.
    if init_body.is_empty() && exit_body.is_empty() {
        return Err(anyhow::anyhow!(
            "kernel module requires at least one of `module_init()` or `module_exit()`"
        ));
    }

    let printk_decl = if has_printk {
        "extern \"C\" {\n    fn printk(fmt: *const core::ffi::c_char, ...);\n}\n\n"
    } else {
        ""
    };

    let mut out = format!(
        "// Generated by rypip --kernel-module. Edit freely.\n\
         #![no_std]\n\n\
         extern crate alloc;\n\n\
         use core::ffi::c_int;\n\
         {printk_decl}"
    );

    // kmalloc-backed global allocator: allows String, Vec, HashMap, and the
    // full stdpython alloc tier to work in kernel context. Same pattern as
    // rust-for-linux's Kmalloc allocator.
    out.push_str(
        "use core::alloc::{GlobalAlloc, Layout};\n\n\
         extern \"C\" {\n    fn kmalloc(size: usize, flags: core::ffi::c_uint) -> *mut u8;\n    fn kfree(ptr: *mut u8);\n}\n\n\
         const GFP_KERNEL: core::ffi::c_uint = 0xCC0;\n\n\
         struct KernelAllocator;\n\n\
         unsafe impl GlobalAlloc for KernelAllocator {\n    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {\n        unsafe { kmalloc(layout.size(), GFP_KERNEL) }\n    }\n    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {\n        unsafe { kfree(ptr) }\n    }\n}\n\n\
         #[global_allocator]\n\
         static ALLOCATOR: KernelAllocator = KernelAllocator;\n\n",
    );

    // Emit a .modinfo section entry for each metadata key-value pair.
    // Entries are null-terminated key=value\0 strings in an ELF .modinfo
    // section; each gets its own static with a generated symbol name.
    for (idx, (key, value)) in modinfo.iter().enumerate() {
        let entry = format!("{key}={value}");
        let len = entry.len();
        let padded = (len + 7) & !7; // align to 8 bytes
        out.push_str(&format!(
            "#[no_mangle]\n#[link_section = \".modinfo\"]\n"
        ));
        out.push_str(&format!(
            "static __mod_info_{idx}: [u8; {padded}] = *b\"{entry}"
        ));
        for _ in len..padded {
            out.push_str("\\0");
        }
        out.push_str("\";\n\n");
    }

    // Kernel panic handler.
    out.push_str(
        "#[panic_handler]\n\
         fn panic(_info: &core::panic::PanicInfo) -> ! {\n    loop {}\n}\n\n",
    );

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
                        &format!("import of `{}` (a floating-point stdlib module)", alias.name),
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
                    &format!("import of `{}` (a floating-point stdlib module)", imp.module),
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

fn check_kernel_with_items_no_floats(items: &[python_ast::WithItem], line: Option<usize>) -> Result<()> {
    for item in items {
        check_kernel_expr_no_floats(&item.context_expr, line)?;
        if let Some(vars) = &item.optional_vars {
            check_kernel_expr_no_floats(vars, line)?;
        }
    }
    Ok(())
}

fn check_kernel_funcdef_no_floats(func: &python_ast::FunctionDef, line: Option<usize>) -> Result<()> {
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

fn check_kernel_args_no_floats(args: &python_ast::ParameterList, line: Option<usize>) -> Result<()> {
    for p in args.posonlyargs.iter().chain(&args.args).chain(&args.kwonlyargs) {
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
fn _expr_int_literal(expr: &python_ast::ExprType) -> Option<i64> {
    if let python_ast::ExprType::Constant(c) = expr {
        if let Some(litrs::Literal::Integer(ilit)) = &c.0 {
            return ilit.value::<i64>();
        }
    }
    None
}

/// Lower a kernel-function body: printk calls and return statements.
fn lower_kernel_body(
    body: &[python_ast::ast::tree::Statement],
    has_printk: &mut bool,
) -> Result<String> {
    let mut out = String::new();
    for stmt in body {
        match &stmt.statement {
            python_ast::StatementType::Expr(expr) => {
                if let Some(call) = extract_kernel_call(expr) {
                    if call.name == "printk" {
                        *has_printk = true;
                        out.push_str("    unsafe {\n        printk(");
                        for (i, arg) in call.args.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(&lower_kernel_expr(arg));
                        }
                        out.push_str(");\n    }\n");
                    }
                }
            }
            python_ast::StatementType::Return(Some(val)) => {
                out.push_str(&format!("    return {};\n", lower_kernel_expr(&val.value)));
            }
            python_ast::StatementType::Return(None) => {
                out.push_str("    return;\n");
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A simplified call representation for kernel lowering.
struct KernelCall<'a> {
    name: String,
    args: &'a [python_ast::ExprType],
}

/// Try to extract a named call from an expression.
fn extract_kernel_call(expr: &python_ast::Expr) -> Option<KernelCall<'_>> {
    if let python_ast::ExprType::Call(call) = &expr.value {
        if let python_ast::ExprType::Name(name) = call.func.as_ref() {
            return Some(KernelCall {
                name: name.id.clone(),
                args: &call.args,
            });
        }
    }
    None
}

/// Lower a kernel-mode expression to a Rust token string.
fn lower_kernel_expr(expr: &python_ast::ExprType) -> String {
    match expr {
        python_ast::ExprType::Constant(c) => {
            match &c.0 {
                Some(litrs::Literal::String(slit)) => {
                    let s = slit.value();
                    let escaped = s
                        .replace('\\', "\\\\")
                        .replace('\"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\t', "\\t");
                    format!(
                        "b\"{}\\0\".as_ptr() as *const core::ffi::c_char",
                        escaped
                    )
                }
                Some(litrs::Literal::Integer(ilit)) => {
                    format!("{}", ilit.value::<i64>().unwrap_or(0))
                }
                _ => c.to_string(),
            }
        }
        _ => format!("/* TODO: lower {:?} */", expr),
    }
}

/// Convert `package` into a Cargo crate under `out_dir`.
pub fn convert(package: &PyPackage, out_dir: &Path, opts: &ConvertOptions) -> Result<ConvertedCrate> {
    if opts.no_std && opts.pyo3 {
        bail!("PyO3 bindings require std (pyo3 links the Python runtime); drop one of --pyo3 / --no-std");
    }
    if opts.kernel_module && opts.pyo3 {
        bail!("PyO3 bindings cannot be used in kernel modules");
    }
    let src_dir = out_dir.join("src");
    fs::create_dir_all(&src_dir)
        .with_context(|| format!("creating {}", src_dir.display()))?;

    let entry_file = package.entry_module().map(|m| m.file.clone());

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
        let kernel_code = generate_kernel_lib_rs(&source)?;
        fs::write(src_dir.join("lib.rs"), &kernel_code)?;
        let has_binary = false;
        let warnings = Vec::new();
        write_cargo_toml(package, out_dir, opts, has_binary)?;
        write_kernel_makefile(package, out_dir)?;
        return Ok(ConvertedCrate {
            root: out_dir.to_path_buf(),
            name: package.name.clone(),
            has_binary,
            warnings,
        });
    }

    // Transpile every module, collecting lossy-conversion warnings.
    let mut transpiled: Vec<(&PyModule, String)> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    for module in &package.modules {
        let code = transpile(module, &mut warnings, opts)?;
        transpiled.push((module, code));
    }
    // Bindings are generated before files are written so their warnings
    // (e.g. forced Python-side renames) participate in the warning mode.
    let bindings_text = if opts.pyo3 {
        Some(generate_bindings(package, &transpiled, &mut warnings)?)
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
    // `__main__.py`, which is bin-only by convention. Non-root __init__
    // modules register too: a sub-package whose only file is __init__.py
    // must still be declared by its parent or its code is silently dropped.
    let mut children: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
    for (module, _) in &transpiled {
        if module.path.is_empty() || is_dunder_main(module) {
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
    for (module, code) in &transpiled {
        if is_dunder_main(module) {
            continue; // handled as the binary below
        }
        let is_root = module.path.is_empty();
        let decls = mod_decls(&children, &module.path, module.is_init || is_root);
        let allows = if is_root {
            format!("{}{}", no_std_attr(opts), generated_lint_attrs(opts.warnings))
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
    }

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
        let main_contents = format!(
            "{}{}\n{}",
            generated_lint_attrs(opts.warnings),
            code,
            decls
        );
        fs::write(src_dir.join("main.rs"), format_rust(&main_contents))?;
        has_binary = true;
    }

    write_cargo_toml(package, out_dir, opts, has_binary)?;

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

/// Transpile one Python module to Rust source text, appending
/// lossy-conversion warnings (which are also baked into the generated code
/// as #[deprecated] notes, unless the warning mode suppresses them).
fn transpile(
    module: &PyModule,
    warnings: &mut Vec<String>,
    opts: &ConvertOptions,
) -> Result<String> {
    let mode = opts.warnings;
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
    let options = PythonOptions {
        lossy_warnings: mode != WarningMode::Allow,
        no_std: opts.no_std,
        ..Default::default()
    };
    let tokens = ast
        .to_rust(
            CodeGenContext::Module(module_name),
            options,
            symbols,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "compiling {}: {}",
                module.file.display(),
                python_ast::format_error_chain(e.as_ref())
            )
        })?;
    Ok(tokens.to_string())
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
    let ret = func.resolved_return_type();
    Some((params, names, ret))
}

/// Write the generated crate's Cargo.toml.
fn write_cargo_toml(
    package: &PyPackage,
    out_dir: &Path,
    opts: &ConvertOptions,
    _has_binary: bool,
) -> Result<()> {
    let stdpython_source = resolve_stdpython_source(opts)?;
    // Kernel modules use stdpython's alloc tier, backed by a kmalloc
    // global allocator — String, Vec, dicts, and the full stdlib work.
    // The no_std profile pins stdpython to its alloc tier: no OS, no libc,
    // suitable for embedded/wasm targets.
    let stdpython_dep = if opts.kernel_module {
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
        match (&stdpython_source, opts.no_std) {
        (StdpythonSource::Path(path), true) => format!(
            "stdpython = {{ path = \"{}\", default-features = false, features = [\"alloc\"] }}",
            path.display().to_string().replace('\\', "/"),
        ),
        (StdpythonSource::Path(path), false) => format!(
            "stdpython = {{ path = \"{}\" }}",
            path.display().to_string().replace('\\', "/"),
        ),
        (StdpythonSource::Registry(version), true) => format!(
            "stdpython = {{ version = \"{}\", default-features = false, features = [\"alloc\"] }}",
            version,
        ),
        (StdpythonSource::Registry(version), false) => {
            format!("stdpython = \"{}\"", version)
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
        name = package.name,
        version = package.version,
    );
    // no_std and kernel-module targets must not unwind — there is no
    // unwinding runtime in embedded, wasm, or kernel contexts.
    if opts.no_std || opts.kernel_module {
        toml.push_str(
            "\n[profile.dev]\npanic = \"abort\"\n\n[profile.release]\npanic = \"abort\"\n",
        );
    }
    if opts.pyo3 {
        toml.push_str(
            "pyo3 = { version = \"0.29\", features = [\"extension-module\"], optional = true }\n\n\
             [features]\n\
             python = [\"dep:pyo3\"]\n\n\
             [lib]\n\
             crate-type = [\"lib\", \"cdylib\"]\n",
        );
    }
    if opts.kernel_module {
        toml.push_str(
            "\n[lib]\ncrate-type = [\"staticlib\"]\n",
        );
    }
    fs::write(out_dir.join("Cargo.toml"), toml)?;
    // Keep the generated crate out of any enclosing workspace.
    let manifest = out_dir.join("Cargo.toml");
    let mut text = fs::read_to_string(&manifest)?;
    text.push_str("\n[workspace]\n");
    fs::write(manifest, text)?;
    Ok(())
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

/// Generate a Kbuild Makefile and C shim that builds the kernel module
/// from the Rust cdylib. The shim bridges the kernel module loader
/// with our Rust-compiled init_module/cleanup_module symbols.
fn write_kernel_makefile(package: &PyPackage, out_dir: &Path) -> Result<()> {
    let name = &package.name;
    let makefile = format!(
        r#"# Kbuild Makefile for {name}.ko — generated by rypip --kernel-module.
#
# Prerequisites: linux-headers, build-essential, Rust toolchain
#
# Build:  make
# Load:   sudo insmod {name}.ko
# Unload: sudo rmmod {name}
# Logs:   sudo dmesg | tail -5

KDIR ?= /lib/modules/$(shell uname -r)/build

obj-m += {name}.o
{name}-objs := {name}_shim.o target/release/lib{name}.a

all: rust
	$(MAKE) -C $(KDIR) M=$(PWD) modules

rust:
	cargo build --release

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	cargo clean

.PHONY: all rust clean
"#
    );

    let shim = format!(
        r#"// {name}_shim.c — generated by rypip. Bridges Kbuild and Rust.
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>

// Declared by the Rust lib.rs with #[no_mangle] pub extern "C".
extern int init_module(void);
extern void cleanup_module(void);

MODULE_LICENSE("GPL");
"#
    );

    fs::write(out_dir.join("Makefile"), &makefile)?;
    fs::write(out_dir.join(format!("{name}_shim.c")), &shim)?;
    Ok(())
}

/// Format generated Rust; fall back to the unformatted text if rustfmt is
/// unavailable or rejects the input.
fn format_rust(source: &str) -> String {
    RustFmt::default()
        .format_str(source)
        .unwrap_or_else(|_| source.to_string())
}
