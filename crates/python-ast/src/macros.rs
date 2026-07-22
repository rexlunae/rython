/// Macros for reducing code duplication in the python-ast library.

/// Macro for generating test functions for AST parsing.
/// Reduces duplication in test code. Asserts that both parsing and code
/// generation succeed and that the generated stream is non-empty, so codegen
/// regressions actually fail the test.
#[macro_export]
macro_rules! create_parse_test {
    ($test_name:ident, $code:literal, $file_name:literal) => {
        #[test]
        fn $test_name() {
            let options = PythonOptions::default();
            let result = crate::parse($code, $file_name)
                .unwrap_or_else(|e| panic!("failed to parse {:?}: {}", $code, e));
            tracing::info!("Python tree: {:?}", result);

            let symbols = result.clone().find_symbols(SymbolTableScopes::new());
            let code = result
                .to_rust(
                    CodeGenContext::Module($file_name.replace(".py", "").to_string()),
                    options,
                    symbols,
                )
                .unwrap_or_else(|e| panic!("failed to generate code for {:?}: {}", $code, e));
            tracing::info!("Generated code: {}", code);
            assert!(
                !code.to_string().trim().is_empty(),
                "codegen produced empty output for {:?}",
                $code
            );
        }
    };
}

/// Macro for generating Node trait implementations with optional position fields.
/// This macro automatically implements the Node trait for types that have position fields.
#[macro_export]
macro_rules! impl_node_with_positions {
    ($type_name:ident { $($field:ident),* }) => {
        impl $crate::Node for $type_name {
            fn lineno(&self) -> Option<usize> {
                $(
                    if stringify!($field) == "lineno" {
                        return self.$field;
                    }
                )*
                None
            }

            fn col_offset(&self) -> Option<usize> {
                $(
                    if stringify!($field) == "col_offset" {
                        return self.$field;
                    }
                )*
                None
            }

            fn end_lineno(&self) -> Option<usize> {
                $(
                    if stringify!($field) == "end_lineno" {
                        return self.$field;
                    }
                )*
                None
            }

            fn end_col_offset(&self) -> Option<usize> {
                $(
                    if stringify!($field) == "end_col_offset" {
                        return self.$field;
                    }
                )*
                None
            }
        }
    };
    
    // Variant for types without position fields
    ($type_name:ident) => {
        impl $crate::Node for $type_name {
            // All methods return None (default implementation)
        }
    };
}

