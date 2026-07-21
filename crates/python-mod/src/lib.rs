use std::fs;
use std::path::Path;

use proc_macro::TokenStream;
use proc_macro_error2::{abort, abort_call_site, proc_macro_error};

use python_ast::{
    parse_enhanced, tree::Module, CodeGen, CodeGenContext, PythonOptions, SymbolTableScopes,
};

use quote::{format_ident, quote};

/// Reads the named Python module from the root of the current crate and
/// parses it, returning the structured parse error on failure so the caller
/// can report it against the macro invocation.
fn load_module(mod_name: &str) -> Result<Module, python_ast::Error> {
    let mod_name_dir = format!("src/{}/__init__.py", mod_name);
    let mod_name_file = format!("src/{}.py", mod_name);

    let (path, python_str) = if Path::new(&mod_name_dir).exists() {
        (mod_name_dir.clone(), fs::read_to_string(&mod_name_dir))
    } else if Path::new(&mod_name_file).exists() {
        (mod_name_file.clone(), fs::read_to_string(&mod_name_file))
    } else {
        return Err(python_ast::parsing_error(
            python_ast::SourceLocation::new(mod_name_file.clone()),
            format!(
                "Python module `{}` not found (looked for {} and {})",
                mod_name, mod_name_file, mod_name_dir
            ),
            "create the module file, or check the module name passed to the macro",
        ));
    };

    let python_str = python_str.map_err(|io_err| {
        python_ast::parsing_error(
            python_ast::SourceLocation::new(path.clone()),
            format!("cannot read Python module file {}: {}", path, io_err),
            "check the file's permissions and encoding (it must be valid UTF-8)",
        )
    })?;

    parse_enhanced(&python_str, &path)
}

/// Render a structured python-ast error as a rustc diagnostic on `span_token`
/// and abort macro expansion. The location and message are the primary line;
/// any help text carried by the error becomes a rustc `help:` note.
fn abort_with_python_error(span_token: &proc_macro2::TokenTree, err: &python_ast::Error) -> ! {
    let location = err.get_field("location").unwrap_or_default().to_string();
    let message = {
        let m = err.get_field("message").unwrap_or_default().to_string();
        if m.is_empty() { err.to_string() } else { m }
    };
    let help = err.get_field("help").unwrap_or_default().to_string();

    let headline = if location.is_empty() {
        format!("rython: {}", message)
    } else {
        format!("rython: {}: {}", location, message)
    };

    if help.is_empty() {
        abort!(span_token, "{}", headline);
    } else {
        abort!(span_token, "{}", headline; help = "{}", help);
    }
}

fn module_options(input: TokenStream, options: PythonOptions) -> TokenStream {
    // We take the first token off of the stream, which is the module name, and
    // then any other tokens will be placed at the top of the generated module.
    let input2 = proc_macro2::TokenStream::from(input);
    let mut tokens = input2.into_iter();

    let Some(mod_name_token) = tokens.next() else {
        abort_call_site!(
            "rython: missing module name";
            help = "use the macro as `python_module!(my_module)`, where `src/my_module.py` \
                    (or `src/my_module/__init__.py`) contains the Python source"
        );
    };
    let mod_name = mod_name_token.to_string();

    let mut remaining_input: proc_macro2::TokenStream = tokens.collect();

    // Load and parse the Python module; report failures on the module-name
    // token so rustc points at the user's macro invocation.
    let py_mod = match load_module(&mod_name) {
        Ok(module) => module,
        Err(e) => abort_with_python_error(&mod_name_token, &e),
    };

    // Collect the module's symbols before code generation (an empty symbol
    // table changes how names resolve during codegen).
    let symbols = py_mod.clone().find_symbols(SymbolTableScopes::new());

    // Convert the parse tree to a Rust TokenStream.
    let py_output = py_mod
        .to_rust(CodeGenContext::Module(mod_name.clone()), options, symbols)
        .unwrap_or_else(|e| {
            // Recover the structured error if there is one; otherwise show
            // the full source chain.
            if let Some(structured) = e.downcast_ref::<python_ast::Error>() {
                abort_with_python_error(&mod_name_token, structured)
            } else {
                abort!(
                    &mod_name_token,
                    "rython: failed to compile Python module `{}`: {}",
                    mod_name,
                    python_ast::format_error_chain(e.as_ref())
                );
            }
        });

    // Add the output to the remaining tokens, which serve as a preamble.
    remaining_input.extend(py_output);

    // Add the module declaration.
    let new_mod_name = format_ident!("{}", mod_name);
    let result = quote!(mod #new_mod_name {
        #remaining_input
    });

    result.into()
}

/// Macro taking two parameters. The first is the name of the module, the second is the path to load it from.
///
/// ```Rust
/// use std::module_path;
///
/// python_module!(py_mod);
/// ```
#[proc_macro]
#[proc_macro_error]
pub fn python_module(input: TokenStream) -> TokenStream {
    let options = PythonOptions::default();
    module_options(input, options)
}

#[proc_macro]
#[proc_macro_error]
/// Loads a python module, but does not include the stdpython libraries.
pub fn python_module_nostd(input: TokenStream) -> TokenStream {
    let mut options = PythonOptions::default();
    options.with_std_python = false;
    module_options(input, options)
}
