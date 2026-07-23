use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    dump, extraction_failure, err_from, BinOpNotYetImplemented, BinaryOperation, CodeGen, CodeGenContext, ExprType,
    FromPythonString, PyAttributeExtractor, PythonOperator, PythonOptions, SymbolTableScopes,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BinOps {
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
    Pow,
    LShift,
    RShift,
    BitOr,
    BitXor,
    BitAnd,
    MatMult,

    Unknown,
}

impl FromPythonString for BinOps {
    fn from_python_string(s: &str) -> Option<Self> {
        match s {
            "Add" => Some(BinOps::Add),
            "Sub" => Some(BinOps::Sub),
            "Mult" => Some(BinOps::Mult),
            "Div" => Some(BinOps::Div),
            "FloorDiv" => Some(BinOps::FloorDiv),
            "Mod" => Some(BinOps::Mod),
            "Pow" => Some(BinOps::Pow),
            "LShift" => Some(BinOps::LShift),
            "RShift" => Some(BinOps::RShift),
            "BitOr" => Some(BinOps::BitOr),
            "BitXor" => Some(BinOps::BitXor),
            "BitAnd" => Some(BinOps::BitAnd),
            "MatMult" => Some(BinOps::MatMult),
            _ => None,
        }
    }
    
    fn unknown() -> Self {
        BinOps::Unknown
    }
}

impl PythonOperator for BinOps {
    fn to_rust_op(&self) -> Result<TokenStream, Box<dyn std::error::Error>> {
        match self {
            BinOps::Add => Ok(quote!(+)),
            BinOps::Sub => Ok(quote!(-)),
            BinOps::Mult => Ok(quote!(*)),
            BinOps::Div => Ok(quote!(as f64 /)),
            BinOps::FloorDiv => Ok(quote!(/)),
            BinOps::Mod => Ok(quote!(%)),
            BinOps::Pow => Ok(quote!(.pow)),
            BinOps::LShift => Ok(quote!(<<)),
            BinOps::RShift => Ok(quote!(>>)),
            BinOps::BitOr => Ok(quote!(|)),
            BinOps::BitXor => Ok(quote!(^)),
            BinOps::BitAnd => Ok(quote!(&)),
            _ => Err(err_from(BinOpNotYetImplemented(BinOp {
                op: self.clone(),
                left: Box::new(ExprType::Name(crate::Name { id: "unknown".to_string() })),
                right: Box::new(ExprType::Name(crate::Name { id: "unknown".to_string() })),
            })).into()),
        }
    }
    
    fn is_unknown(&self) -> bool {
        matches!(self, BinOps::Unknown)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BinOp {
    pub op: BinOps,
    pub left: Box<ExprType>,
    pub right: Box<ExprType>,
}

impl BinaryOperation for BinOp {
    type OperatorType = BinOps;
    
    fn operator(&self) -> &Self::OperatorType {
        &self.op
    }
    
    fn left(&self) -> &ExprType {
        &self.left
    }
    
    fn right(&self) -> &ExprType {
        &self.right
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for BinOp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        tracing::debug!("ob: {}", dump(&ob, None)?);
        
        let op = ob.extract_attr_with_context("op", "binary operator")?;
        let op_type_str = op.extract_type_name("binary operator")?;
        
        let left = ob.extract_attr_with_context("left", "binary operand")?;
        let right = ob.extract_attr_with_context("right", "binary operand")?;
        
        tracing::debug!("left: {}, right: {}", dump(&left, None)?, dump(&right, None)?);

        let op = BinOps::parse_or_unknown(&op_type_str);
        if matches!(op, BinOps::Unknown) {
            tracing::debug!("Found unknown BinOp {:?}", op_type_str);
        }

        let left = left.extract().map_err(|e| extraction_failure("getting binary operator operand", &ob, e))?;
        let right = right.extract().map_err(|e| extraction_failure("getting binary operator operand", &ob, e))?;

        Ok(BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    }
}

impl CodeGen for BinOp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        // Python's ** promotes based on operand types; route through the
        // stdpython py_pow helper, which implements those semantics.
        if matches!(self.op, BinOps::Pow) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            return Ok(quote!(py_pow(#left, #right)));
        }
        
        // For Div, we need to cast to f64
        if matches!(self.op, BinOps::Div) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            return Ok(quote!((#left) as f64 / (#right) as f64));
        }

        // Python's // floors toward negative infinity and % takes the
        // divisor's sign; Rust's / and % truncate. Route through the
        // stdpython helpers, which implement the Python semantics.
        if matches!(self.op, BinOps::FloorDiv) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            return Ok(quote!(py_floordiv(#left, #right)));
        }

        if matches!(self.op, BinOps::Mod) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            return Ok(quote!(py_mod(#left, #right)));
        }
        
        // Python's * repeats sequences when one operand is a string:
        // "!" * 3 == "!!!". Route literal-string repetition through the
        // stdpython multiply_string helper (numeric multiplication keeps
        // the plain operator below).
        if matches!(self.op, BinOps::Mult) {
            let left_is_str = matches!(&*self.left, ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::String(_))))
                || matches!(&*self.left, ExprType::JoinedStr(_));
            let right_is_str = matches!(&*self.right, ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::String(_))))
                || matches!(&*self.right, ExprType::JoinedStr(_));
            if left_is_str || right_is_str {
                let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                let right = self.right.clone().to_rust(ctx, options, symbols)?;
                return Ok(if left_is_str {
                    quote!(multiply_string(#left, (#right) as i64))
                } else {
                    quote!(multiply_string(#right, (#left) as i64))
                });
            }
        }

        // Python `+` covers cases Rust's Add doesn't (String + String,
        // int/float promotion, list concatenation): lower through the
        // stdpython PyAdd trait, which borrows both operands.
        if matches!(self.op, BinOps::Add) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            return Ok(quote!((#left).py_add(&(#right))));
        }

        // Use the generic binary operation implementation for everything else
        self.generate_rust_code(ctx, options, symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_add, "1 + 2", "test_case.py");
    create_parse_test!(test_subtract, "1 - 2", "test_case.py");
    create_parse_test!(test_multiply, "3 * 4", "test_case.py");
    create_parse_test!(test_divide, "8 / 2", "test_case.py");
    create_parse_test!(test_power, "2 ** 3", "test_case.py");
    create_parse_test!(test_modulo, "10 % 3", "test_case.py");
    
    #[test]
    fn test_unknown_operator() {
        let unknown_op = BinOps::Unknown;
        assert!(unknown_op.is_unknown());
        assert!(unknown_op.to_rust_op().is_err());
    }
    
    #[test]
    fn test_from_python_string() {
        assert_eq!(BinOps::from_python_string("Add"), Some(BinOps::Add));
        assert_eq!(BinOps::from_python_string("Unknown"), None);
        assert_eq!(BinOps::parse_or_unknown("Invalid"), BinOps::Unknown);
    }
}
