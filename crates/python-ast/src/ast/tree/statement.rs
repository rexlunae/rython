use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{
    dump, err_from, extraction_failure, Assign, AsyncFor, AsyncWith, AugAssign, Call, ClassDef,
    CodeGen, CodeGenContext, Expr, For, FunctionDef, If, Import, ImportFrom, Node, PythonOptions,
    Raise, StatementNotYetImplemented, SymbolTableScopes, Try, While, With,
};

use tracing::debug;

use serde::{Deserialize, Serialize};

/// AST node types that can be used as a statement implement this type.
pub trait PyStatementTrait: Clone + PartialEq {
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Statement {
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
    pub statement: StatementType,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Statement {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
            statement: StatementType::extract(ob)?,
        })
    }
}

impl Node for Statement {
    fn lineno(&self) -> Option<usize> {
        self.lineno
    }
    fn col_offset(&self) -> Option<usize> {
        self.col_offset
    }
    fn end_lineno(&self) -> Option<usize> {
        self.end_lineno
    }
    fn end_col_offset(&self) -> Option<usize> {
        self.end_col_offset
    }
}

impl CodeGen for Statement {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        self.statement.clone().find_symbols(symbols)
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let (lineno, col_offset) = (self.lineno, self.col_offset);
        let (end_lineno, end_col_offset) = (self.end_lineno, self.end_col_offset);
        self.statement
            .clone()
            .to_rust(ctx, options, symbols)
            .map_err(|e| {
                let location = crate::SourceLocation::with_span(
                    "<module>",
                    lineno,
                    col_offset.map(|c| c + 1),
                    end_lineno,
                    end_col_offset,
                );
                Box::<dyn std::error::Error>::from(crate::codegen_error(
                    location,
                    crate::format_error_chain(e.as_ref()),
                    "",
                ))
            })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum StatementType {
    AsyncFunctionDef(FunctionDef),
    Assign(Assign),
    AugAssign(AugAssign),
    Break,
    Continue,
    ClassDef(ClassDef),
    Call(Call),
    Pass,
    Return(Option<Expr>),
    Import(Import),
    ImportFrom(ImportFrom),
    Expr(Expr),
    FunctionDef(FunctionDef),
    If(If),
    For(For),
    While(While),
    Try(Try),
    AsyncWith(AsyncWith),
    AsyncFor(AsyncFor),
    Raise(Raise),
    With(With),

    Unimplemented(String),
}

impl<'a, 'py> FromPyObject<'a, 'py> for StatementType {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let ob_type = ob
            .get_type()
            .name()
            .map_err(|e| extraction_failure("statement type", &ob, e))?;

        debug!("statement...ob_type: {}...{}", ob_type, dump(&ob, Some(4))?);
        match ob_type.extract::<String>()?.as_str() {
            "AsyncFunctionDef" => Ok(StatementType::AsyncFunctionDef(
                FunctionDef::extract(ob)
                    .map_err(|e| extraction_failure("async function definition", &ob, e))?,
            )),
            "Assign" => {
                let assignment = Assign::extract(ob)
                    .map_err(|e| extraction_failure("assignment", &ob, e))?;
                Ok(StatementType::Assign(assignment))
            }
            "AnnAssign" => {
                // An annotated assignment (`x: int = 5`) is an ordinary
                // assignment with a type annotation we don't yet consume; a
                // bare annotation (`x: int`) declares nothing at runtime.
                let value = ob
                    .getattr("value")
                    .map_err(|e| extraction_failure("annotated assignment value", &ob, e))?;
                if value.is_none() {
                    return Ok(StatementType::Pass);
                }
                let target = ob
                    .getattr("target")
                    .map_err(|e| extraction_failure("annotated assignment target", &ob, e))?
                    .extract()
                    .map_err(|e| extraction_failure("annotated assignment target", &ob, e))?;
                let value = value
                    .extract()
                    .map_err(|e| extraction_failure("annotated assignment value", &ob, e))?;
                Ok(StatementType::Assign(Assign {
                    targets: vec![target],
                    value,
                    type_comment: None,
                }))
            }
            "AugAssign" => {
                let aug_assignment = AugAssign::extract(ob)
                    .map_err(|e| extraction_failure("augmented assignment", &ob, e))?;
                Ok(StatementType::AugAssign(aug_assignment))
            }
            "Pass" => Ok(StatementType::Pass),
            "Call" => {
                let value = ob
                    .getattr("value")
                    .map_err(|e| extraction_failure("call statement value", &ob, e))?;
                let call = Call::extract(value.as_borrowed())
                    .map_err(|e| extraction_failure("call statement", &ob, e))?;
                debug!("call: {:?}", call);
                Ok(StatementType::Call(call))
            }
            "ClassDef" => Ok(StatementType::ClassDef(
                ClassDef::extract(ob)
                    .map_err(|e| extraction_failure("class definition", &ob, e))?,
            )),
            "Continue" => Ok(StatementType::Continue),
            "Break" => Ok(StatementType::Break),
            "FunctionDef" => Ok(StatementType::FunctionDef(
                FunctionDef::extract(ob)
                    .map_err(|e| extraction_failure("function definition", &ob, e))?,
            )),
            "Import" => Ok(StatementType::Import(
                Import::extract(ob).map_err(|e| extraction_failure("import", &ob, e))?,
            )),
            "ImportFrom" => Ok(StatementType::ImportFrom(
                ImportFrom::extract(ob)
                    .map_err(|e| extraction_failure("from-import", &ob, e))?,
            )),
            "Expr" => {
                let expr = ob
                    .extract()
                    .map_err(|e| extraction_failure("expression statement", &ob, e))?;
                Ok(StatementType::Expr(expr))
            }
            "Return" => {
                tracing::debug!("return expression: {}", dump(&ob, None)?);
                // Extract the return value from the Return statement's 'value' field
                let return_value = if let Ok(value_attr) = ob.getattr("value") {
                    if value_attr.is_none() {
                        // Bare 'return' statement - create a NoneType Expr
                        Some(Expr {
                            value: crate::tree::ExprType::NoneType(crate::tree::Constant(None)),
                            ctx: None,
                            lineno: ob.lineno(),
                            col_offset: ob.col_offset(),
                            end_lineno: ob.end_lineno(),
                            end_col_offset: ob.end_col_offset(),
                        })
                    } else {
                        // Return with actual expression - extract as ExprType then wrap in Expr
                        let expr_value: crate::tree::ExprType = value_attr
                            .extract()
                            .map_err(|e| extraction_failure("return value", &ob, e))?;
                        Some(Expr {
                            value: expr_value,
                            ctx: None,
                            lineno: ob.lineno(),
                            col_offset: ob.col_offset(),
                            end_lineno: ob.end_lineno(),
                            end_col_offset: ob.end_col_offset(),
                        })
                    }
                } else {
                    None
                };
                Ok(StatementType::Return(return_value))
            }
            "If" => {
                let if_stmt =
                    If::extract(ob).map_err(|e| extraction_failure("if statement", &ob, e))?;
                Ok(StatementType::If(if_stmt))
            }
            "For" => {
                let for_stmt =
                    For::extract(ob).map_err(|e| extraction_failure("for loop", &ob, e))?;
                Ok(StatementType::For(for_stmt))
            }
            "While" => {
                let while_stmt =
                    While::extract(ob).map_err(|e| extraction_failure("while loop", &ob, e))?;
                Ok(StatementType::While(while_stmt))
            }
            "Try" => {
                let try_stmt =
                    Try::extract(ob).map_err(|e| extraction_failure("try statement", &ob, e))?;
                Ok(StatementType::Try(try_stmt))
            }
            "AsyncWith" => {
                let async_with_stmt = AsyncWith::extract(ob)
                    .map_err(|e| extraction_failure("async with statement", &ob, e))?;
                Ok(StatementType::AsyncWith(async_with_stmt))
            }
            "AsyncFor" => {
                let async_for_stmt = AsyncFor::extract(ob)
                    .map_err(|e| extraction_failure("async for loop", &ob, e))?;
                Ok(StatementType::AsyncFor(async_for_stmt))
            }
            "Raise" => {
                let raise_stmt =
                    Raise::extract(ob).map_err(|e| extraction_failure("raise statement", &ob, e))?;
                Ok(StatementType::Raise(raise_stmt))
            }
            "With" => {
                let with_stmt =
                    With::extract(ob).map_err(|e| extraction_failure("with statement", &ob, e))?;
                Ok(StatementType::With(with_stmt))
            }
            other => Err(extraction_failure(
                "statement",
                &ob,
                format!(
                    "the `{}` statement is not yet supported by rython",
                    other
                ),
            )),
        }
    }
}

impl CodeGen for StatementType {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        match self {
            StatementType::Assign(a) => a.find_symbols(symbols),
            StatementType::AugAssign(a) => a.find_symbols(symbols),
            StatementType::ClassDef(c) => c.find_symbols(symbols),
            StatementType::FunctionDef(f) => f.find_symbols(symbols),
            StatementType::Import(i) => i.find_symbols(symbols),
            StatementType::ImportFrom(i) => i.find_symbols(symbols),
            StatementType::Expr(e) => e.find_symbols(symbols),
            StatementType::If(i) => i.find_symbols(symbols),
            StatementType::For(f) => f.find_symbols(symbols),
            StatementType::While(w) => w.find_symbols(symbols),
            StatementType::Try(t) => t.find_symbols(symbols),
            StatementType::AsyncWith(aw) => aw.find_symbols(symbols),
            StatementType::AsyncFor(af) => af.find_symbols(symbols),
            StatementType::Raise(r) => r.find_symbols(symbols),
            StatementType::With(w) => w.find_symbols(symbols),
            _ => symbols,
        }
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        match self {
            StatementType::AsyncFunctionDef(s) => {
                let func_def = s.to_rust(Self::Context::Async(Box::new(ctx)), options, symbols)?;
                Ok(quote!(#func_def))
            }
            StatementType::Assign(a) => a.to_rust(ctx, options, symbols),
            StatementType::AugAssign(a) => a.to_rust(ctx, options, symbols),
            StatementType::Break => Ok(quote! {break;}),
            StatementType::Call(c) => c.to_rust(ctx, options, symbols),
            StatementType::ClassDef(c) => c.to_rust(ctx, options, symbols),
            StatementType::Continue => Ok(quote! {continue;}),
            StatementType::Pass => Ok(quote! {}),
            StatementType::FunctionDef(s) => s.to_rust(ctx, options, symbols),
            StatementType::Import(s) => s.to_rust(ctx, options, symbols),
            StatementType::ImportFrom(s) => s.to_rust(ctx, options, symbols),
            StatementType::Expr(s) => s.to_rust(ctx, options, symbols),
            StatementType::Return(None) => Ok(quote!(return)),
            StatementType::Return(Some(e)) => {
                let exp = e.clone().to_rust(ctx, options, symbols)?;
                Ok(quote!(return #exp))
            }
            StatementType::If(i) => i.to_rust(ctx, options, symbols),
            StatementType::For(f) => f.to_rust(ctx, options, symbols),
            StatementType::While(w) => w.to_rust(ctx, options, symbols),
            StatementType::Try(t) => t.to_rust(ctx, options, symbols),
            StatementType::AsyncWith(aw) => aw.to_rust(ctx, options, symbols),
            StatementType::AsyncFor(af) => af.to_rust(ctx, options, symbols),
            StatementType::Raise(r) => r.to_rust(ctx, options, symbols),
            StatementType::With(w) => w.to_rust(ctx, options, symbols),
            _ => {
                let error = err_from(StatementNotYetImplemented(self));
                Err(error.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_pass_statement() {
        let statement = StatementType::Pass;
        let options = PythonOptions::default();
        let tokens = statement.clone().to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );

        debug!("statement: {:?}, tokens: {:?}", statement, tokens);
        assert_eq!(tokens.unwrap().is_empty(), true);
    }

    #[test]
    fn check_break_statement() {
        let statement = StatementType::Break;
        let options = PythonOptions::default();
        let tokens = statement.clone().to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );

        debug!("statement: {:?}, tokens: {:?}", statement, tokens);
        assert_eq!(tokens.unwrap().is_empty(), false);
    }

    #[test]
    fn check_continue_statement() {
        let statement = StatementType::Continue;
        let options = PythonOptions::default();
        let tokens = statement.clone().to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );

        debug!("statement: {:?}, tokens: {:?}", statement, tokens);
        assert_eq!(tokens.unwrap().is_empty(), false);
    }

    #[test]
    fn return_with_nothing() {
        let tree = crate::parse("return", "<none>").unwrap();
        assert_eq!(tree.raw.body.len(), 1);
        assert_eq!(
            tree.raw.body[0].statement,
            StatementType::Return(Some(Expr {
                value: crate::tree::ExprType::NoneType(crate::tree::Constant(None)),
                lineno: Some(1),
                col_offset: Some(0),
                end_lineno: Some(1),
                end_col_offset: Some(6),
                ..Default::default()
            }))
        );
    }

    #[test]
    fn return_with_expr() {
        let lit = litrs::Literal::Integer(litrs::IntegerLit::parse(String::from("8")).unwrap());
        let tree = crate::parse("return 8", "<none>").unwrap();
        assert_eq!(tree.raw.body.len(), 1);
        assert_eq!(
            tree.raw.body[0].statement,
            StatementType::Return(Some(Expr {
                value: crate::tree::ExprType::Constant(crate::tree::Constant(Some(lit))),
                lineno: Some(1),
                col_offset: Some(0),
                end_lineno: Some(1),
                end_col_offset: Some(8),
                ..Default::default()
            }))
        );
    }

    #[test]
    fn does_module_compile() {
        let options = PythonOptions::default();
        let result = crate::parse(
            "#test comment
def foo():
    continue
    pass
",
            "test_case",
        )
        .unwrap();
        tracing::info!("{:?}", result);
        let code = result.to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        tracing::info!("module: {:?}", code);
    }
}
