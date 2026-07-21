use proc_macro2::TokenStream;
use pyo3::{Borrowed, Bound, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    dump, err_from, extraction_failure, Attribute, Await, BinOp, BoolOp, Call, CodeGen, CodeGenContext, Compare,
    Constant, Dict, DictComp, ExprTypeNotYetImplemented, FormattedValue, GeneratorExp, IfExp,
    JoinedStr, Lambda, ListComp, Name, NamedExpr, Node, PythonOptions, Set, SetComp, Starred,
    Subscript, SymbolTableScopes, Tuple, UnaryOp, Yield, YieldFrom,
};

/// Mostly this shouldn't be used, but it exists so that we don't have to manually implement FromPyObject on all of ExprType
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[repr(transparent)]
pub struct Container<T>(pub T);

impl<'a, 'py> FromPyObject<'a, 'py> for Container<crate::pytypes::List<ExprType>> {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let list = crate::pytypes::List::<ExprType>::new();

        tracing::debug!("pylist: {}", dump(&ob, Some(4))?);
        let _converted_list: Vec<Bound<PyAny>> = ob.extract()?;
        for item in _converted_list.iter() {
            tracing::debug!("item: {:?}", item);
        }

        Ok(Self(list))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum ExprType {
    BoolOp(BoolOp),
    NamedExpr(NamedExpr),
    BinOp(BinOp),
    UnaryOp(UnaryOp),
    Lambda(Lambda),
    IfExp(IfExp),
    Dict(Dict),
    Set(Set),
    ListComp(ListComp),
    DictComp(DictComp),
    SetComp(SetComp),
    GeneratorExp(GeneratorExp),
    Await(Await),
    Yield(Yield),
    YieldFrom(YieldFrom),
    Compare(Compare),
    Call(Call),
    FormattedValue(FormattedValue),
    JoinedStr(JoinedStr),
    Constant(Constant),

    /// These can appear in a few places, such as the left side of an assignment.
    Attribute(Attribute),
    Subscript(Subscript),
    Starred(Starred),
    Name(Name),
    List(Vec<ExprType>),
    Tuple(Tuple),
    /*Slice(),*/
    NoneType(Constant),

    Unimplemented(String),
    #[default]
    Unknown,
}

impl<'a, 'py> FromPyObject<'a, 'py> for ExprType {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        tracing::debug!("exprtype ob: {}", dump(&ob, Some(4))?);

        let expr_type = ob
            .get_type()
            .name()
            .map_err(|e| extraction_failure("expression type name", &ob, e))?;
        tracing::debug!("expression type: {}, value: {}", expr_type, dump(&ob, None)?);

        let r = match expr_type.extract::<String>()?.as_str() {
            "Attribute" => {
                let a = ob.extract().map_err(|e| extraction_failure("extracting Attribute in expression", &ob, e))?;
                Ok(Self::Attribute(a))
            }
            "Await" => {
                //println!("await: {}", dump(&ob, None)?);
                let a = ob.extract().map_err(|e| extraction_failure("extracting await value in expression", &ob, e))?;
                Ok(Self::Await(a))
            }
            "BoolOp" => {
                let b = ob.extract().map_err(|e| extraction_failure("extracting BoolOp in expression", &ob, e))?;
                Ok(Self::BoolOp(b))
            }
            "Call" => {
                let et = ob.extract().map_err(|e| extraction_failure("parsing Call expression", &ob, e))?;
                Ok(Self::Call(et))
            }
            "Compare" => {
                let c = ob.extract().map_err(|e| extraction_failure("extracting Compare in expression", &ob, e))?;
                Ok(Self::Compare(c))
            }
            "Constant" => {
                tracing::debug!("constant: {}", dump(&ob, None)?);
                let c = ob.extract().map_err(|e| extraction_failure("extracting Constant in expression", &ob, e))?;
                Ok(Self::Constant(c))
            }
            "List" => {
                // Extract the list elements using the 'elts' attribute
                let elts_attr = ob
                    .getattr("elts")
                    .map_err(|e| extraction_failure("list elements", &ob, e))?;
                let elts_vec: Vec<Bound<PyAny>> = elts_attr
                    .extract()
                    .map_err(|e| extraction_failure("list elements", &ob, e))?;

                // Convert each element to ExprType
                let mut expr_list = Vec::new();
                for elt in elts_vec {
                    let expr: ExprType = elt
                        .extract()
                        .map_err(|e| extraction_failure("list element", &elt, e))?;
                    expr_list.push(expr);
                }
                
                Ok(Self::List(expr_list))
            }
            "ListComp" => {
                let lc = ob.extract().map_err(|e| extraction_failure("extracting ListComp in expression", &ob, e))?;
                Ok(Self::ListComp(lc))
            }
            "DictComp" => {
                let dc = ob.extract().map_err(|e| extraction_failure("extracting DictComp in expression", &ob, e))?;
                Ok(Self::DictComp(dc))
            }
            "SetComp" => {
                let sc = ob.extract().map_err(|e| extraction_failure("extracting SetComp in expression", &ob, e))?;
                Ok(Self::SetComp(sc))
            }
            "GeneratorExp" => {
                let ge = ob.extract().map_err(|e| extraction_failure("extracting GeneratorExp in expression", &ob, e))?;
                Ok(Self::GeneratorExp(ge))
            }
            "Name" => {
                let name = ob.extract().map_err(|e| extraction_failure("parsing Name expression", &ob, e))?;
                Ok(Self::Name(name))
            }
            "UnaryOp" => {
                let c = ob.extract().map_err(|e| extraction_failure("extracting UnaryOp in expression", &ob, e))?;
                Ok(Self::UnaryOp(c))
            }
            "BinOp" => {
                let c = ob.extract().map_err(|e| extraction_failure("extracting BinOp in expression", &ob, e))?;
                Ok(Self::BinOp(c))
            }
            "Lambda" => {
                let l = ob.extract().map_err(|e| extraction_failure("extracting Lambda in expression", &ob, e))?;
                Ok(Self::Lambda(l))
            }
            "IfExp" => {
                let i = ob.extract().map_err(|e| extraction_failure("extracting IfExp in expression", &ob, e))?;
                Ok(Self::IfExp(i))
            }
            "Dict" => {
                let d = ob.extract().map_err(|e| extraction_failure("extracting Dict in expression", &ob, e))?;
                Ok(Self::Dict(d))
            }
            "Set" => {
                let s = ob.extract().map_err(|e| extraction_failure("extracting Set in expression", &ob, e))?;
                Ok(Self::Set(s))
            }
            "Tuple" => {
                let t = ob.extract().map_err(|e| extraction_failure("extracting Tuple in expression", &ob, e))?;
                Ok(Self::Tuple(t))
            }
            "Subscript" => {
                let s = ob.extract().map_err(|e| extraction_failure("extracting Subscript in expression", &ob, e))?;
                Ok(Self::Subscript(s))
            }
            "Starred" => {
                let s = ob.extract().map_err(|e| extraction_failure("extracting Starred in expression", &ob, e))?;
                Ok(Self::Starred(s))
            }
            "Yield" => {
                let y = ob.extract().map_err(|e| extraction_failure("extracting Yield in expression", &ob, e))?;
                Ok(Self::Yield(y))
            }
            "YieldFrom" => {
                let yf = ob.extract().map_err(|e| extraction_failure("extracting YieldFrom in expression", &ob, e))?;
                Ok(Self::YieldFrom(yf))
            }
            "JoinedStr" => {
                let js = ob.extract().map_err(|e| extraction_failure("extracting JoinedStr in expression", &ob, e))?;
                Ok(Self::JoinedStr(js))
            }
            "FormattedValue" => {
                let fv = ob.extract().map_err(|e| extraction_failure("extracting FormattedValue in expression", &ob, e))?;
                Ok(Self::FormattedValue(fv))
            }
            _ => {
                let err_msg = format!(
                    "Unimplemented expression type {}, {}",
                    expr_type,
                    dump(&ob, None)?
                );
                Err(pyo3::exceptions::PyValueError::new_err(
                    ob.error_message("<unknown>", err_msg.as_str()),
                ))
            }
        };
        r
    }
}

impl<'a> CodeGen for ExprType {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        match self {
            ExprType::Attribute(attribute) => attribute.to_rust(ctx, options, symbols),
            ExprType::Await(func) => func.to_rust(ctx, options, symbols),
            ExprType::BinOp(binop) => binop.to_rust(ctx, options, symbols),
            ExprType::BoolOp(boolop) => boolop.to_rust(ctx, options, symbols),
            ExprType::Call(call) => call.to_rust(ctx, options, symbols),
            ExprType::Compare(c) => c.to_rust(ctx, options, symbols),
            ExprType::Constant(c) => c.to_rust(ctx, options, symbols),
            ExprType::Lambda(l) => l.to_rust(ctx, options, symbols),
            ExprType::IfExp(i) => i.to_rust(ctx, options, symbols),
            ExprType::Dict(d) => d.to_rust(ctx, options, symbols),
            ExprType::Set(s) => s.to_rust(ctx, options, symbols),
            ExprType::ListComp(lc) => lc.to_rust(ctx, options, symbols),
            ExprType::DictComp(dc) => dc.to_rust(ctx, options, symbols),
            ExprType::SetComp(sc) => sc.to_rust(ctx, options, symbols),
            ExprType::GeneratorExp(ge) => ge.to_rust(ctx, options, symbols),
            ExprType::Tuple(t) => t.to_rust(ctx, options, symbols),
            ExprType::Subscript(s) => s.to_rust(ctx, options, symbols),
            ExprType::Starred(s) => s.to_rust(ctx, options, symbols),
            ExprType::Yield(y) => y.to_rust(ctx, options, symbols),
            ExprType::YieldFrom(yf) => yf.to_rust(ctx, options, symbols),
            ExprType::JoinedStr(js) => js.to_rust(ctx, options, symbols),
            ExprType::FormattedValue(fv) => fv.to_rust(ctx, options, symbols),
            ExprType::List(l) => {
                let mut elements = Vec::new();
                let mut has_starred = false;
                
                for li in l {
                    let code = li
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    
                    // Check if this is a starred expression
                    if matches!(li, ExprType::Starred(_)) {
                        has_starred = true;
                        let code_str = code.to_string();
                        // Special handling for sys::argv unpacking
                        if code_str.contains("sys :: argv") {
                            // Mark that we need special sys::argv handling with a unique marker
                            elements.push(quote! { __STARRED_ARGV_MARKER__ });
                        } else {
                            elements.push(code);
                        }
                    } else {
                        elements.push(code);
                    }
                }
                
                // If we have starred expressions, handle them specially
                if has_starred {
                    let mut final_elements = Vec::new();
                    let mut has_argv_starred = false;
                    
                    for element in elements {
                        let elem_str = element.to_string();
                        if elem_str.contains("__STARRED_ARGV_MARKER__") {
                            has_argv_starred = true;
                            continue; // Skip the placeholder
                        } else {
                            final_elements.push(element);
                        }
                    }
                    
                    // Build the vector with proper unpacking
                    if has_argv_starred {
                        if final_elements.is_empty() {
                            // Only sys::argv unpacking
                            Ok(quote! {
                                (*sys::argv).clone()
                            })
                        } else {
                            // Mix of regular elements and sys::argv unpacking
                            // Clone each element to avoid ownership issues
                            Ok(quote! {
                                {
                                    let mut vec = Vec::new();
                                    #(vec.push((#final_elements).clone().to_string());)*
                                    vec.extend((*sys::argv).iter().cloned());
                                    vec
                                }
                            })
                        }
                    } else {
                        // Other starred expressions (not sys::argv)
                        Ok(quote! {
                            vec![#(#final_elements.to_string()),*]
                        })
                    }
                } else {
                    // For regular vector creation, ensure all elements are strings
                    // Always convert to String for consistency and compatibility
                    // Clone to avoid ownership issues
                    Ok(quote! {
                        vec![#((#elements).clone().to_string()),*]
                    })
                }
            }
            ExprType::Name(name) => name.to_rust(ctx, options, symbols),
            ExprType::NoneType(c) => c.to_rust(ctx, options, symbols),
            ExprType::UnaryOp(operand) => operand.to_rust(ctx, options, symbols),

            _ => {
                let error = err_from(ExprTypeNotYetImplemented(self));
                Err(error.into())
            }
        }
    }
}

/// An Expr only contains a single value key, which leads to the actual expression,
/// which is one of several types.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Expr {
    pub value: ExprType,
    pub ctx: Option<String>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Expr {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let err_msg = format!("extracting object value {} in expression", dump(&ob, None)?);

        let ob_value = ob
            .getattr("value")
            .map_err(|e| extraction_failure("expression value", &ob, format!("{}: {}", err_msg, e)))?;
        tracing::debug!("ob_value: {}", dump(&ob_value, None)?);

        // The context is Load, Store, etc. For some types of expressions such as Constants, it does not exist.
        let ctx: Option<String> = if let Ok(pyany) = ob_value.getattr("ctx") {
            pyany.get_type().extract().unwrap_or_default()
        } else {
            None
        };

        let mut r = Self {
            value: ExprType::Unknown,
            ctx: ctx,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        };

        let expr_type = ob_value
            .get_type()
            .name()
            .map_err(|e| extraction_failure("expression type name", &ob, e))?;
        tracing::debug!(
            "expression type: {}, value: {}",
            expr_type,
            dump(&ob_value, None)?
        );
        match expr_type.extract::<String>()?.as_str() {
            "Attribute" => {
                let a = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Attribute expression", &ob_value, e))?;
                r.value = ExprType::Attribute(a);
                Ok(r)
            }
            "Await" => {
                let a = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Await expression", &ob_value, e))?;
                r.value = ExprType::Await(a);
                Ok(r)
            }
            "BinOp" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("BinOp expression", &ob_value, e))?;
                r.value = ExprType::BinOp(c);
                Ok(r)
            }
            "BoolOp" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("BoolOp expression", &ob_value, e))?;
                r.value = ExprType::BoolOp(c);
                Ok(r)
            }
            "Call" => {
                let et = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Call expression", &ob_value, e))?;
                r.value = ExprType::Call(et);
                Ok(r)
            }
            "Constant" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Constant expression", &ob_value, e))?;
                r.value = ExprType::Constant(c);
                Ok(r)
            }
            "Compare" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Compare expression", &ob_value, e))?;
                r.value = ExprType::Compare(c);
                Ok(r)
            }
            "List" => {
                // Extract the list elements using the 'elts' attribute
                let elts_attr = ob_value
                    .getattr("elts")
                    .map_err(|e| extraction_failure("list elements", &ob_value, e))?;
                let elts_vec: Vec<Bound<PyAny>> = elts_attr
                    .extract()
                    .map_err(|e| extraction_failure("list elements", &ob_value, e))?;

                // Convert each element to ExprType
                let mut expr_list = Vec::new();
                for elt in elts_vec {
                    let expr: ExprType = elt
                        .extract()
                        .map_err(|e| extraction_failure("list element", &elt, e))?;
                    expr_list.push(expr);
                }
                
                r.value = ExprType::List(expr_list);
                Ok(r)
            }
            "Name" => {
                let name = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Name expression", &ob_value, e))?;
                r.value = ExprType::Name(name);
                Ok(r)
            }
            "UnaryOp" => {
                let c = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("UnaryOp expression", &ob_value, e))?;
                r.value = ExprType::UnaryOp(c);
                Ok(r)
            }
            "Lambda" => {
                let l = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Lambda expression", &ob_value, e))?;
                r.value = ExprType::Lambda(l);
                Ok(r)
            }
            "IfExp" => {
                let i = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("IfExp expression", &ob_value, e))?;
                r.value = ExprType::IfExp(i);
                Ok(r)
            }
            "Dict" => {
                let d = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Dict expression", &ob_value, e))?;
                r.value = ExprType::Dict(d);
                Ok(r)
            }
            "Set" => {
                let s = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Set expression", &ob_value, e))?;
                r.value = ExprType::Set(s);
                Ok(r)
            }
            "Tuple" => {
                let t = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Tuple expression", &ob_value, e))?;
                r.value = ExprType::Tuple(t);
                Ok(r)
            }
            "Subscript" => {
                let s = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Subscript expression", &ob_value, e))?;
                r.value = ExprType::Subscript(s);
                Ok(r)
            }
            "Yield" => {
                let y = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("Yield expression", &ob_value, e))?;
                r.value = ExprType::Yield(y);
                Ok(r)
            }
            "YieldFrom" => {
                let yf = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("YieldFrom expression", &ob_value, e))?;
                r.value = ExprType::YieldFrom(yf);
                Ok(r)
            }
            "JoinedStr" => {
                let js = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("JoinedStr expression", &ob_value, e))?;
                r.value = ExprType::JoinedStr(js);
                Ok(r)
            }
            "FormattedValue" => {
                let fv = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("FormattedValue expression", &ob_value, e))?;
                r.value = ExprType::FormattedValue(fv);
                Ok(r)
            }
            "GeneratorExp" => {
                let ge = ob_value
                    .extract()
                    .map_err(|e| extraction_failure("GeneratorExp expression", &ob_value, e))?;
                r.value = ExprType::GeneratorExp(ge);
                Ok(r)
            }
            // In sitations where an expression is optional, we may see a NoneType expressions.
            "NoneType" => {
                r.value = ExprType::NoneType(Constant(None));
                Ok(r)
            }
            _ => {
                let err_msg = format!(
                    "Unimplemented expression type {}, {}",
                    expr_type,
                    dump(&ob, None)?
                );
                Err(pyo3::exceptions::PyValueError::new_err(
                    ob.error_message("<unknown>", err_msg.as_str()),
                ))
            }
        }
    }
}

impl CodeGen for Expr {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        let _module_name = match ctx.clone() {
            CodeGenContext::Module(name) => name,
            _ => "unknown".to_string(),
        };

        match self.value.clone() {
            ExprType::Await(a) => a.to_rust(ctx.clone(), options, symbols),
            ExprType::BinOp(binop) => binop.to_rust(ctx.clone(), options, symbols),
            ExprType::BoolOp(boolop) => boolop.to_rust(ctx.clone(), options, symbols),
            ExprType::Call(call) => call.to_rust(ctx.clone(), options, symbols),
            ExprType::Constant(constant) => constant.to_rust(ctx, options, symbols),
            ExprType::Compare(compare) => compare.to_rust(ctx, options, symbols),
            ExprType::Lambda(l) => l.to_rust(ctx, options, symbols),
            ExprType::IfExp(i) => i.to_rust(ctx, options, symbols),
            ExprType::Dict(d) => d.to_rust(ctx, options, symbols),
            ExprType::Set(s) => s.to_rust(ctx, options, symbols),
            ExprType::GeneratorExp(ge) => ge.to_rust(ctx, options, symbols),
            ExprType::Tuple(t) => t.to_rust(ctx, options, symbols),
            ExprType::Subscript(s) => s.to_rust(ctx, options, symbols),
            ExprType::UnaryOp(operand) => operand.to_rust(ctx, options, symbols),
            ExprType::List(l) => {
                // Use the same logic as ExprType::List in the main match above
                let expr_type = ExprType::List(l);
                expr_type.to_rust(ctx, options, symbols)
            }
            ExprType::Name(name) => name.to_rust(ctx, options, symbols),
            ExprType::Yield(y) => y.to_rust(ctx, options, symbols),
            ExprType::YieldFrom(yf) => yf.to_rust(ctx, options, symbols),
            ExprType::JoinedStr(js) => js.to_rust(ctx, options, symbols),
            ExprType::FormattedValue(fv) => fv.to_rust(ctx, options, symbols),
            // NoneType expressions generate no code.
            ExprType::NoneType(_c) => Ok(quote!()),
            _ => {
                let error = err_from(ExprTypeNotYetImplemented(self.value));
                Err(error.into())
            }
        }
    }
}

impl Node for Expr {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_call_expression() {
        let expression = crate::parse("test()", "test.py").unwrap();
        let mut options = PythonOptions::default();
        options.with_std_python = false;
        let symbols = SymbolTableScopes::new();
        let tokens = expression
            .clone()
            .to_rust(CodeGenContext::Module("test".to_string()), options, symbols)
            .unwrap();
        assert_eq!(tokens.to_string(), "fn __module_init__ () { test () } fn main () { __module_init__ () ; }");
    }
}
