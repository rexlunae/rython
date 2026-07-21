use std::path::Path;
use std::fs;

use proc_macro::{ Span, TokenStream  };
use proc_macro_error2::{
    proc_macro_error,
    abort,
};

use python_ast::{
    CodeGen, CodeGenContext,
    PythonOptions, PyResult,
    parse,
    tree::Module,
    SymbolTableScopes
};

use quote::{quote, format_ident};

/// Reads the named Python module from the root of the current crate.
fn load_module(mod_name: String) -> PyResult<Module> {
    let mod_path = Span::mixed_site().local_file();

    let mod_name_dir = format!("src/{}/__init__.py", &mod_name);
    let mod_name_file = format!("src/{}.py", &mod_name);

    let python_str = if Path::new(&mod_name_dir).exists() {
        fs::read_to_string(mod_name_dir)
    } else if Path::new(&mod_name_file).exists() {
        fs::read_to_string(mod_name_file)
    } else {
        abort!(mod_name, "Module not found {} in {:?}", mod_name, mod_path)
    }.expect("Reading python module");

    parse(&python_str, &mod_name)
}

fn module_options(input: TokenStream, options: PythonOptions) -> TokenStream {
    // We take the first token off of the stream, which is the module name, and then any other tokens
    // will be placed at the top of the generated module.
    let symbols = SymbolTableScopes::new();

    let mut new_input = input.into_iter();
    let mod_name = new_input.next()
        .expect("missing module name").to_string();

    let remaining_input = proc_macro::TokenStream::from_iter(new_input);
    let mut remaining_input2 = proc_macro2::TokenStream::from(remaining_input);

    //let mut output = quote!(#remaining_input);

    // Loads a Python module and stores it in a String
    let py_mod = load_module(mod_name.clone())
        .expect("loading python module ");

    // Convert the parse tree to Rust a TokenStream.
    let py_output = py_mod.to_rust(CodeGenContext::Module(mod_name.clone()), options, symbols)
        .expect("converting Python to Rust");

    // Add the output to the remaining tokens, which serve as a preamble.
    proc_macro2::TokenStream::extend::<proc_macro2::TokenStream>(&mut remaining_input2, py_output.into());


    // Add the module declaration.
    let new_mod_name = format_ident!("{}", mod_name);
    let result = quote!(mod #new_mod_name {
        #remaining_input2
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
