use std::collections::HashMap;
use std::fs::{File, read_to_string};
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use clap::Parser;
use python_ast::{CodeGen, CodeGenContext, PythonOptions, parse, symbols::SymbolTableScopes};
use rust_format::{Formatter, RustFmt};

// Set up the fern logging facility.
fn setup_logger(
    level: log::LevelFilter,
    log_file: Option<String>,
) -> std::result::Result<(), fern::InitError> {
    let mut logger = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}:{}] {}",
                humantime::format_rfc3339_seconds(SystemTime::now()),
                record.level(),
                record.target(),
                record.line().unwrap(),
                message
            ))
        })
        .level(level)
        .chain(std::io::stdout());

    match log_file {
        None => {}
        Some(s) => logger = logger.chain(fern::log_file(s)?),
    };
    logger.apply()?;
    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "The output file.")]
    output: Option<PathBuf>,
    #[arg()]
    inputs: Vec<PathBuf>,

    #[clap(long, short, action, help = "Nicely format the output.")]
    pretty: bool,

    #[clap(
        long,
        short,
        action,
        help = "Don't actually compile, just output the ast represented as a Rust data structure."
    )]
    ast_only: bool,

    #[clap(
        long,
        short,
        action,
        help = "Don't actually compile, just output the symbols."
    )]
    symbols_only: bool,

    #[clap(long, short, action, help="Sets the log level. Values are: off,error,warn,info,debug,trace", default_value_t=log::LevelFilter::Warn)]
    log_level: log::LevelFilter,

    #[clap(long, action, help = "Write log events to this file.")]
    log_file: Option<String>,

    #[clap(long, short, action, help = "Compile without stdpython.")]
    nostd: bool,

    #[clap(
        long,
        action,
        value_name = "BACKEND",
        help = "Force a numpy execution backend for the generated program: scalar, \
                rayon, simd, cuda, or vulkan. Emits a startup numpy::set_backend call \
                (overrides RYPY_NUMPY_BACKEND and np.set_backend); build the generated \
                crate with the matching stdpython feature (numpy-rayon, numpy-simd, ...)."
    )]
    numpy_backend: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    setup_logger(args.log_level, args.log_file)?;

    let mut output_list = Vec::new();

    for input in args.inputs {
        // Rust-module imports resolve through the rython.toml manifest in
        // the input file's directory.
        let rust_modules = load_rust_modules(input.parent().unwrap_or_else(|| Path::new(".")))?;
        let options = PythonOptions {
            with_std_python: !args.nostd,
            numpy_backend: args.numpy_backend.clone(),
            rust_modules: std::rc::Rc::new(rust_modules),
            ..Default::default()
        };

        let module_name = format!("{}", input.to_string_lossy());
        let py = read_to_string(input.clone())?;
        let ast = parse(&py, &module_name)?;

        let output = if args.ast_only {
            if args.pretty {
                format!("{:#?}", ast)
            } else {
                format!("{:?}", ast)
            }
        } else {
            let symbols = ast.clone().find_symbols(SymbolTableScopes::new());
            if args.symbols_only {
                if args.pretty {
                    format!("{:#?}", symbols)
                } else {
                    format!("{:?}", symbols)
                }
            } else {
                let mut symbols = symbols;
                python_ast::register_rust_module_imports(&ast.raw.body, &options, &mut symbols)
                    .map_err(anyhow::Error::msg)?;
                let rust = ast
                    .to_rust(
                        CodeGenContext::Module(module_name.clone()),
                        options.clone(),
                        symbols.clone(),
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to compile {}: {}",
                            module_name,
                            python_ast::format_error_chain(e.as_ref())
                        )
                    })?;
                // The -W channel: lossy-conversion divergences collected
                // during lowering print to stderr, exactly like rypip's
                // multi-module converter (issue #209 — previously the
                // single-file CLI silently discarded them, so a dropped
                // call looked like a clean conversion).
                for w in options.definition_warnings.borrow().iter() {
                    eprintln!("warning: {}: {}", module_name, w);
                }
                if args.pretty {
                    let unformatted = rust.to_string();
                    RustFmt::default().format_str(unformatted)?
                } else {
                    format!("{}", rust)
                }
            }
        };
        output_list.push(output);
    }

    let file_output = output_list.join("");

    match args.output {
        Some(f) => {
            let mut file = File::create(f)?;
            file.write_all(file_output.as_bytes())?;
        }
        None => println!("{}", file_output),
    }

    Ok(())
}

/// Load the `[rust-modules]` table from `dir/rython.toml` (when present)
/// into a registry keyed by import name, resolving each entry from a `.pyi`
/// stub next to the manifest or by inference from the crate source. A
/// missing manifest is not an error: it simply means no import-based Rust
/// bindings. The caller must add the crates to the build's Cargo.toml.
fn load_rust_modules(dir: &Path) -> Result<HashMap<String, python_ast::RustModuleSpec>> {
    let manifest_path = dir.join("rython.toml");
    let text = match read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(e) => {
            return Err(anyhow::anyhow!(
                "reading {}: {}",
                manifest_path.display(),
                e
            ));
        }
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
            return Err(anyhow::anyhow!(
                "{}: `{import_name}` declares both path and version; give one",
                manifest_path.display()
            ));
        }
        let spec = python_ast::resolve_rust_module(
            import_name,
            &crate_name,
            path.as_deref(),
            version.as_deref(),
            dir,
        )
        .map_err(anyhow::Error::msg)?;
        out.insert(import_name.clone(), spec);
    }
    Ok(out)
}
