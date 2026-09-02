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
            let left = clone_if_place(&self.left, left);
            let right = clone_if_place(&self.right, right);
            return Ok(quote!(py_pow(#left, #right)));
        }
        
        // For Div, Python semantics are elementwise/numeric true division.
        // Route through the stdpython py_div helper: numeric operands
        // divide to f64, and NdArray operands (numpy) divide elementwise.
        // The `?` propagates a catchable ZeroDivisionError instead of
        // silently yielding inf/nan (issue #107).
        if matches!(self.op, BinOps::Div) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            let left = clone_if_place(&self.left, left);
            let right = clone_if_place(&self.right, right);
            return Ok(quote!(py_div(#left, #right)?));
        }

        // Python's `@` is matrix multiplication (numpy semantics for
        // arrays); route through the stdpython py_matmul helper. Non-array
        // operands fail loudly at compile time (no impl), like CPython's
        // TypeError for unsupported types.
        if matches!(self.op, BinOps::MatMult) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            let left = clone_if_place(&self.left, left);
            let right = clone_if_place(&self.right, right);
            return Ok(quote!(py_matmul(#left, #right)));
        }

        // Python's // floors toward negative infinity and % takes the
        // divisor's sign; Rust's / and % truncate. Route through the
        // stdpython helpers, which implement the Python semantics. The `?`
        // propagates a catchable ZeroDivisionError (issue #75).
        if matches!(self.op, BinOps::FloorDiv) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            let left = clone_if_place(&self.left, left);
            let right = clone_if_place(&self.right, right);
            return Ok(quote!(py_floordiv(#left, #right)?));
        }

        if matches!(self.op, BinOps::Mod) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            let left = clone_if_place(&self.left, left);
            let right = clone_if_place(&self.right, right);
            return Ok(quote!(py_mod(#left, #right)?));
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
        // stdpython PyAdd trait, which borrows both operands. Bare numeric
        // literals get an explicit type: trait-method resolution on an
        // unanchored `{integer}` receiver fails before literal fallback
        // (e.g. `1 + 2` in an unannotated position).
        if matches!(self.op, BinOps::Add) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            let left = anchor_numeric_literal(&self.left, left);
            let right = anchor_numeric_literal(&self.right, right);
            return Ok(quote!((#left).py_add(&(#right))));
        }

        // `-` mirrors `+`: PySub borrows its operands (numeric promotion
        // for scalars, elementwise for NdArray), so `x - y` never moves
        // the variables.
        if matches!(self.op, BinOps::Sub) {
            // An OPTION-typed RHS (`x - y` where y is `int | None` —
            // urllib3's `self.chunk_left - amt` and
            // `time.monotonic() - self._start_connect`): the runtime
            // Option blanket unwraps an Option LHS, but a None RHS would
            // need `i64: PySub<Option<i64>>` (not implemented — the
            // blanket's bound runs the other way). Python raises TypeError
            // when either operand is None; unwrap the RHS with the loud
            // §12.2 panic (the `is not None` guard in real code prevents
            // it). Computed before the operand renders move ctx/options.
            let option_rhs = is_option_expr(&self.right, &ctx, &options, &symbols);
            let lhs_name = if option_rhs {
                py_operand_name(&self.left, &ctx, &options, &symbols)
            } else {
                ""
            };
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            let left = anchor_numeric_literal(&self.left, left);
            let right = anchor_numeric_literal(&self.right, right);
            let right = if option_rhs {
                let msg = format!(
                    "unsupported operand type(s) for -: '{}' and 'NoneType'",
                    lhs_name
                );
                quote!(match (#right).clone() {
                    Some(__rython_w) => __rython_w,
                    None => panic!(#msg),
                })
            } else {
                right
            };
            return Ok(quote!((#left).py_sub(&(#right))));
        }

        // `*` mirrors `+`/`-` for everything except the literal-string
        // repetition handled above: PyMul borrows both operands, so
        // `x * 2` never moves `x`.
        if matches!(self.op, BinOps::Mult) {
            let left = self.left.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let right = self.right.clone().to_rust(ctx, options, symbols)?;
            let left = anchor_numeric_literal(&self.left, left);
            let right = anchor_numeric_literal(&self.right, right);
            return Ok(quote!((#left).py_mul(&(#right))));
        }

        // Use the generic binary operation implementation for everything else
        self.generate_rust_code(ctx, options, symbols)
    }
}

/// Clone a VARIABLE operand handed to a by-value helper.
///
/// `py_add`/`py_sub`/`py_mul` borrow, so `x + y` never moves; the helpers
/// below (`py_div`, `py_floordiv`, `py_mod`, `py_pow`, `py_matmul`) take
/// ownership, which moved out of the variable and made any later use a
/// borrow-checker error in the generated crate — `print(b / a)` followed by
/// `print(a * 2.0)` on numpy arrays (issue #201). Python has no such
/// restriction, and rython's value semantics say a name survives being
/// used. Only place expressions are cloned: temporaries cannot be moved
/// from twice, and a bare numeric literal must stay unwrapped so its type
/// can still be anchored.
fn clone_if_place(expr: &ExprType, tokens: TokenStream) -> TokenStream {
    match expr {
        ExprType::Name(_) | ExprType::Attribute(_) | ExprType::Subscript(_) => {
            quote!((#tokens).clone())
        }
        _ => tokens,
    }
}

/// Give a bare numeric literal (possibly under unary +/-) a concrete type
/// so PyAdd's trait-method resolution has an anchored receiver: int
/// literals become i64, float literals f64. Anything else is left to its
/// own type.
fn anchor_numeric_literal(expr: &ExprType, tokens: TokenStream) -> TokenStream {
    fn literal_type(expr: &ExprType) -> Option<TokenStream> {
        match expr {
            ExprType::Constant(c) => match &c.0 {
                Some(litrs::Literal::Integer(_)) => Some(quote!(i64)),
                Some(litrs::Literal::Float(_)) => Some(quote!(f64)),
                _ => None,
            },
            ExprType::UnaryOp(u) => literal_type(&u.operand),
            _ => None,
        }
    }
    match literal_type(expr) {
        Some(ty) => quote!((#tokens) as #ty),
        None => tokens,
    }
}

/// Whether an expression lowers to an `Option<...>`: a name typed through
/// the per-function maps (`amt: int | None` — urllib3's `_handle_chunk`)
/// or a `self.<field>` whose field is Option (the field table, since
/// `infer_type` sees only the syntactic shape). Used to unwrap an
/// Option-typed RHS of `-` (the runtime Option blanket unwraps the LHS).
fn is_option_expr(
    expr: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    if matches!(crate::infer_type(Some(&ctx), expr, options, symbols), crate::TypeInfo::Option(_)) {
        return true;
    }
    if let ExprType::Attribute(attr) = expr
        && matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "self")
    {
        if let Some(t) = crate::ast::tree::aug_assign::self_field_rust_ty(
            &attr.attr,
            ctx,
            options,
            symbols,
        ) {
            return matches!(t, crate::TypeInfo::Option(_));
        }
    }
    false
}

/// The CPython type name of an operand for the loud TypeError message
/// when the OTHER operand is None (`x - None`): the LHS's inner type
/// (`Option<i64>` → `int`, `f64` → `float`). Resolves the same sources
/// as [`is_option_expr`] — the per-function type maps, a `self.<field>`
/// through the class table, and `time.monotonic()` (f64).
fn py_operand_name(
    expr: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> &'static str {
    fn operand_name(t: &crate::TypeInfo) -> &'static str {
        match t {
            crate::TypeInfo::Int => "int",
            crate::TypeInfo::Float => "float",
            crate::TypeInfo::Option(inner) => operand_name(inner.as_ref()),
            _ => "int",
        }
    }
    let from_infer = operand_name(&crate::infer_type(Some(&ctx), expr, options, symbols));
    if from_infer != "int" {
        return from_infer;
    }
    // infer_type sees a `self.<field>` only syntactically (PyObject): the
    // class-table type carries the inner numeric kind.
    if let ExprType::Attribute(attr) = expr
        && matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "self")
        && let Some(t) = crate::ast::tree::aug_assign::self_field_rust_ty(
            &attr.attr,
            ctx,
            options,
            symbols,
        )
    {
        if matches!(t, crate::TypeInfo::Float) {
            return "float";
        }
        if matches!(t, crate::TypeInfo::Int) {
            return "int";
        }
    }
    // `time.monotonic()` returns f64.
    if let ExprType::Call(call) = expr
        && let ExprType::Attribute(attr) = call.func.as_ref()
        && attr.attr == "monotonic"
    {
        return "float";
    }
    from_infer
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
