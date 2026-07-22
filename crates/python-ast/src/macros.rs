/// Macros for reducing code duplication in the python-ast library.

/// Macro for generating FromPyObject implementations for operator enums.
/// This reduces the boilerplate for extracting Python operator objects.
#[macro_export]
macro_rules! impl_from_py_object_for_op_enum {
    ($enum_name:ident, $error_msg:literal) => {
        impl<'a, 'py> FromPyObject<'a, 'py> for $enum_name {
    type Error = pyo3::PyErr;
            fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                let err_msg = format!($error_msg, dump(&ob, None)?);
                Err(pyo3::exceptions::PyValueError::new_err(
                    ob.error_message("<unknown>", err_msg),
                ))
            }
        }
    };
}

/// Macro for generating standard CodeGen trait implementations.
/// This reduces boilerplate for the common pattern of CodeGen implementations.
#[macro_export]
macro_rules! impl_standard_codegen {
    ($type_name:ident) => {
        impl CodeGen for $type_name {
            type Context = CodeGenContext;
            type Options = PythonOptions;
            type SymbolTable = SymbolTableScopes;

            fn to_rust(
                self,
                ctx: Self::Context,
                options: Self::Options,
                symbols: Self::SymbolTable,
            ) -> Result<proc_macro2::TokenStream, Box<dyn std::error::Error>> {
                self.generate_rust_code(ctx, options, symbols)
            }
        }
    };
}

/// Macro for implementing CodeGen with custom code generation logic.
#[macro_export]
macro_rules! impl_codegen_with_custom {
    ($type_name:ident, $generate_fn:expr) => {
        impl CodeGen for $type_name {
            type Context = CodeGenContext;
            type Options = PythonOptions;
            type SymbolTable = SymbolTableScopes;

            fn to_rust(
                self,
                ctx: Self::Context,
                options: Self::Options,
                symbols: Self::SymbolTable,
            ) -> Result<proc_macro2::TokenStream, Box<dyn std::error::Error>> {
                $generate_fn(self, ctx, options, symbols)
            }
        }
    };
}

/// Macro for extracting PyAny attributes with consistent error handling.
/// Propagates a structured error (with the node's source position) instead of
/// panicking; must be used inside a function returning `PyResult`.
#[macro_export]
macro_rules! extract_py_attr {
    ($obj:expr, $attr:literal, $error_context:literal) => {
        $obj.getattr($attr)
            .map_err(|e| $crate::extraction_failure($error_context, &$obj, e))?
    };
}

/// Macro for extracting PyAny type names with error handling.
/// Propagates a structured error instead of panicking; must be used inside a
/// function returning `PyResult`.
#[macro_export]
macro_rules! extract_py_type_name {
    ($obj:expr, $context:literal) => {
        $obj.get_type()
            .name()
            .map_err(|e| $crate::extraction_failure($context, &$obj, e))?
    };
}

/// Macro for generating binary operator FromPyObject implementations.
/// This handles the common pattern of extracting left, right, and op from Python binary operators.
#[macro_export]
macro_rules! impl_binary_op_from_py {
    ($struct_name:ident, $enum_name:ident, $op_variants:tt) => {
        impl<'a, 'py> FromPyObject<'a, 'py> for $struct_name {
    type Error = pyo3::PyErr;
            fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                tracing::debug!("ob: {}", dump(&ob, None)?);
                
                let op = extract_py_attr!(ob, "op", "operator");
                let op_type = extract_py_type_name!(op, "binary operator");
                
                let left = extract_py_attr!(ob, "left", "binary operand");
                let right = extract_py_attr!(ob, "right", "binary operand");
                
                tracing::debug!("left: {}, right: {}", dump(&left, None)?, dump(&right, None)?);

                let op_type_str: String = op_type.extract()?;
                let op = match op_type_str.as_ref() {
                    $op_variants,
                    _ => {
                        tracing::debug!("Found unknown {} {:?}", stringify!($enum_name), op);
                        $enum_name::Unknown
                    }
                };

                let left = left
                    .extract()
                    .map_err(|e| $crate::extraction_failure("left operand", &ob, e))?;
                let right = right
                    .extract()
                    .map_err(|e| $crate::extraction_failure("right operand", &ob, e))?;

                Ok($struct_name {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
        }
    };
}

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

/// Macro for generating PyAny attribute extraction with error context.
#[macro_export]
macro_rules! extract_with_context {
    ($obj:expr, $attr:literal) => {
        $obj.getattr($attr).map_err(|e| {
            pyo3::exceptions::PyAttributeError::new_err(format!(
                "Failed to extract '{}': {}",
                $attr, e
            ))
        })?
    };
}

/// Generates repetitive match arms for operator conversions.
#[macro_export]
macro_rules! operator_match_arms {
    ($($variant:ident => $string:literal),* $(,)?) => {
        $(
            $string => Self::$variant,
        )*
    };
}

/// Macro for generating symbol table tests with consistent patterns.
#[macro_export]
macro_rules! symbol_table_test {
    ($test_name:ident, $setup:block, $assertion:block) => {
        #[test]
        fn $test_name() {
            $setup
            $assertion
        }
    };
}