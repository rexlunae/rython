use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, PyAttributeExtractor, extract_list
};

use super::Statement;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct If {
    pub test: ExprType,
    pub body: Vec<Statement>,
    pub orelse: Vec<Statement>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for If {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let test = ob.extract_attr_with_context("test", "if test condition")?;
        let test = test
            .extract()
            .map_err(|e| crate::extraction_failure("if condition", &ob, e))?;
        
        let body: Vec<Statement> = extract_list(&ob, "body", "if body statements")?;
        let orelse: Vec<Statement> = extract_list(&ob, "orelse", "if else statements")?;
        
        Ok(If {
            test,
            body,
            orelse,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(If { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for If {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let symbols = self.test.find_symbols(symbols);
        let symbols = self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc));
        self.orelse.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // A VERSION-GATED branch (`if sys.version_info[0] < 3:` —
        // distlib's compat.py): rython targets Python 3, so the gate is
        // evaluated at conversion time and the dead branch is dropped.
        // The test never lowers to runtime code.
        if let Some(taken) = version_gate_taken(&self.test) {
            let (branch, label) = if taken {
                (&self.body, "true")
            } else {
                (&self.orelse, "false")
            };
            options.definition_warnings.borrow_mut().push(format!(
                "`if sys.version_info...` version gate evaluates to {label}; \
                 the other branch is dropped at conversion time"
            ));
            let stmts: Result<Vec<_>, _> = branch
                .iter()
                .cloned()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let stmts = stmts?;
            return Ok(quote! { #(#stmts;)* });
        }

        // A STATICALLY-DECIDED module-name gate (issue #137): `if brotli
        // is not None:` where `brotli = None` is the module's single
        // store (the folded handler of a failed import guard), or `if
        // HAS_ZSTD:` where the flag is a single-store False. CPython
        // never enters the dead branch; the guarded class definitions
        // and decoder branches fold away with it.
        if let Some(taken) = static_name_gate_taken(&self.test, &options) {
            let branch = if taken { &self.body } else { &self.orelse };
            let stmts: Result<Vec<_>, _> = branch
                .iter()
                .cloned()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let stmts = stmts?;
            return Ok(quote! { #(#stmts;)* });
        }

        // Regular if statement handling; the test is a condition position,
        // so Python truthiness applies.
        let test =
            crate::condition_to_rust(&self.test, ctx.clone(), options.clone(), symbols.clone())?;

        // Issue #125: `if x is not None:` narrows x to its inner type inside
        // the body — reads unwrap (Name::to_rust consults narrowed_names),
        // and the comprehension/iteration over x sees the inner element
        // type. Any other test narrows nothing.
        let mut body_options = options.clone();
        let mut else_options = options.clone();
        if let Some((narrowed, inner)) = crate::narrowing_from_test(&self.test, &options) {
            let mut narrowed_names = options.narrowed_names.as_ref().clone();
            // The narrowed type: the Option's inner type, or for a
            // str|bytes union narrowed by isinstance, the concrete branch
            // type carried in the map value (String/Bytes).
            let target = inner.clone().unwrap_or(crate::TypeInfo::StrOrBytes);
            narrowed_names.insert(narrowed.clone(), target);
            body_options.narrowed_names = std::rc::Rc::new(narrowed_names);
            if let Some(inner) = inner {
                let mut name_types = options.name_types.as_ref().clone();
                name_types.insert(narrowed.clone(), inner);
                body_options.name_types = std::rc::Rc::new(name_types);
            }
        }
        // Issue #121: `if isinstance(x, (bytes, bytearray)):` (or its
        // negation) narrows a str|bytes union to the CONCRETE branch in the
        // body AND the complementary branch in the else.
        if let Some((name, body_ty, else_ty)) =
            crate::isinstance_narrowing(&self.test, &options, &symbols)
        {
            let mut body_n = options.narrowed_names.as_ref().clone();
            body_n.insert(name.clone(), body_ty.clone());
            body_options.narrowed_names = std::rc::Rc::new(body_n);
            let mut else_n = options.narrowed_names.as_ref().clone();
            else_n.insert(name.clone(), else_ty.clone());
            else_options.narrowed_names = std::rc::Rc::new(else_n);
            // A CLASS narrowing (a root-typed name to a class of its
            // subtree — hierarchy.rs) also retypes the name for the branch,
            // so method and field resolution see the narrowed class, and
            // records the root the sum type's view is taken from.
            if let Some(crate::TypeInfo::Class(root)) = options.name_types.get(&name).cloned()
                && matches!(body_ty, crate::TypeInfo::Class(_))
            {
                for (opts, ty) in [(&mut body_options, &body_ty), (&mut else_options, &else_ty)] {
                    let mut nt = opts.name_types.as_ref().clone();
                    nt.insert(name.clone(), ty.clone());
                    opts.name_types = std::rc::Rc::new(nt);
                    let mut origin = opts.narrowed_class_origin.as_ref().clone();
                    origin.insert(name.clone(), root.clone());
                    opts.narrowed_class_origin = std::rc::Rc::new(origin);
                }
            }
        }

        // A test that folded to a compile-time CONSTANT — an isinstance
        // decided through the class tree, a version gate that reached here
        // as a literal, a bool constant: the dead branch is pruned and the
        // live one inlined, so the output carries no `if true { ... }`
        // noise. Constant tests have no side effects, so dropping them is
        // sound.
        if let Some(taken) = const_bool_tokens(&test) {
            let (branch, opts) = if taken {
                (self.body, body_options)
            } else {
                (self.orelse, else_options)
            };
            let stmts: Result<Vec<_>, _> = branch
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), opts.clone(), symbols.clone()))
                .collect();
            let stmts = stmts?;
            return Ok(quote! { #(#stmts;)* });
        }

        let body_stmts: Result<Vec<_>, _> = self
            .body
            .into_iter()
            .map(|stmt| stmt.to_rust(ctx.clone(), body_options.clone(), symbols.clone()))
            .collect();
        let body_stmts = body_stmts?;
        
        if self.orelse.is_empty() {
            Ok(quote! {
                if #test {
                    #(#body_stmts;)*
                }
            })
        } else {
            let else_stmts: Result<Vec<_>, _> = self.orelse
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), else_options.clone(), symbols.clone()))
                .collect();
            let else_stmts = else_stmts?;
            
            Ok(quote! {
                if #test {
                    #(#body_stmts;)*
                } else {
                    #(#else_stmts;)*
                }
            })
        }
    }
}

/// Whether rendered condition tokens are a compile-time boolean constant:
/// the bare literal, or the literal behind the condition position's
/// truthiness call (`(true).is_truthy()`).
fn const_bool_tokens(test: &TokenStream) -> Option<bool> {
    let s: String = test.to_string().split_whitespace().collect();
    match s.as_str() {
        "true" | "(true).is_truthy()" => Some(true),
        "false" | "(false).is_truthy()" => Some(false),
        _ => None,
    }
}

/// Statically evaluate a `sys.version_info` version gate
/// (`sys.version_info[0] < 3`, `sys.version_info >= (3,)`...): rython
/// targets Python 3, so the gate's truth value is known at conversion
/// time. Returns `Some(taken)` when the test is a recognized version
/// gate (whether the guarded branch is the one that runs), `None` when
/// the test is not one. `pub(crate)` so the module walker can splice the
/// taken branch at MODULE level (a version-gated `def` must be a module
/// item, not a nested function — certifi's core.py).
pub(crate) fn version_gate_taken(test: &ExprType) -> Option<bool> {
    // The simulated version rython reports (Python 3.11.0).
    const VERSION: [i64; 3] = [3, 11, 0];

    let ExprType::Compare(c) = test else {
        return None;
    };
    if c.ops.len() != 1 || c.comparators.len() != 1 {
        return None;
    }
    // Left side: `sys.version_info` (optionally subscripted, e.g. `[0]`).
    let (left_is_sys_version, left_index): (bool, Option<i64>) = match &*c.left {
        // The bare form (`sys.version_info >= (3, 11)` — certifi's
        // core.py): the ATTRIBUTE itself is the version tuple; passing
        // the receiver Name made every bare gate fall through (only the
        // subscripted `[0]` form ever fired).
        ExprType::Attribute(_) => (is_sys_version_info(&*c.left), None),
        ExprType::Subscript(s) => {
            if !is_sys_version_info(&s.value) {
                return None;
            }
            match &s.kind {
                crate::SubscriptKind::Index(i) => (
                    true,
                    int_literal(i).or_else(|| {
                        // `sys.version_info[0]` — a bare int index.
                        None
                    }),
                ),
                _ => return None,
            }
        }
        _ => return None,
    };
    if !left_is_sys_version {
        return None;
    }

    // Right side: an int literal (comparisons like `sys.version_info[0]
    // < 3`), or a tuple expression (`sys.version_info >= (3,)`).
    let right: Option<Vec<i64>> = match &c.comparators[0] {
        ExprType::Constant(cn) => match &cn.0 {
            Some(litrs::Literal::Integer(i)) => {
                Some(vec![i.value()?])
            }
            _ => return None,
        },
        ExprType::Tuple(t) => {
            let mut out = Vec::new();
            for e in &t.elts {
                let ExprType::Constant(cn) = e else {
                    return None;
                };
                let Some(litrs::Literal::Integer(i)) = &cn.0 else {
                    return None;
                };
                out.push(i.value()?);
            }
            Some(out)
        }
        _ => return None,
    };
    let right = right?;

    // Build the left-hand version vector: full `sys.version_info`, or a
    // slice of it from the subscript (only index 0 is statically common;
    // any other index falls back to the whole tuple comparison semantics
    // of `sys.version_info[0]` -> [major]).
    let left: Vec<i64> = match left_index {
        Some(0) => vec![VERSION[0]],
        Some(_) => return None,
        None => VERSION.to_vec(),
    };

    // Zip-compare, Python tuple semantics: the first differing element
    // decides. rython's [3, 11, 0] vs the gate's constants.
    let result = match c.ops[0] {
        crate::Compares::Lt => version_cmp(&left, &right) == std::cmp::Ordering::Less,
        crate::Compares::LtE => version_cmp(&left, &right) != std::cmp::Ordering::Greater,
        crate::Compares::Gt => version_cmp(&left, &right) == std::cmp::Ordering::Greater,
        crate::Compares::GtE => version_cmp(&left, &right) != std::cmp::Ordering::Less,
        crate::Compares::Eq => version_cmp(&left, &right) == std::cmp::Ordering::Equal,
        crate::Compares::NotEq => version_cmp(&left, &right) != std::cmp::Ordering::Equal,
        _ => return None,
    };
    Some(result)
}

/// Python tuple comparison: element-wise until one differs; a prefix is
/// less than a longer tuple.
fn version_cmp(left: &[i64], right: &[i64]) -> std::cmp::Ordering {
    for (a, b) in left.iter().zip(right.iter()) {
        match a.cmp(b) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    left.len().cmp(&right.len())
}

/// Whether an expression is `sys.version_info`.
fn is_sys_version_info(expr: &ExprType) -> bool {
    matches!(
        expr,
        ExprType::Attribute(a)
            if a.attr == "version_info"
                && matches!(&*a.value, ExprType::Name(n)
                    if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Sys))
    )
}

/// The integer value of a constant int expression, if it is one.
fn int_literal(expr: &ExprType) -> Option<i64> {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(i)) => i.value(),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_simple_if, "if x > 5:\n    print('big')", "if_test.py");
    create_parse_test!(test_if_else, "if x > 5:\n    print('big')\nelse:\n    print('small')", "if_test.py");
    create_parse_test!(test_if_elif, "if x > 10:\n    print('huge')\nelif x > 5:\n    print('big')\nelse:\n    print('small')", "if_test.py");
}
/// Conversion-time truth of a test over STATICALLY-DECIDED module names
/// (single-store None/False module constants — module.rs's
/// statically_none_names / statically_false_names, typically the folded
/// handler of a failed import guard). Some(truth) when the whole test
/// folds; None leaves the test to runtime. Issue #137.
pub(crate) fn static_name_gate_taken(
    test: &ExprType,
    options: &PythonOptions,
) -> Option<bool> {
    match test {
        ExprType::Name(n) => {
            // None and False are both falsy; a resolved module import is
            // always truthy.
            if options.statically_module_names.contains(&n.id) {
                return Some(true);
            }
            (options.statically_none_names.contains(&n.id)
                || options.statically_false_names.contains(&n.id))
            .then_some(false)
        }
        ExprType::UnaryOp(u) if matches!(u.op, crate::Ops::Not) => {
            static_name_gate_taken(&u.operand, options).map(|t| !t)
        }
        ExprType::Compare(c) => {
            let ExprType::Name(n) = c.left.as_ref() else {
                return None;
            };
            // `name is None` truth: a statically-None name → true; a
            // resolved module import (never None) → false.
            let is_none = if options.statically_none_names.contains(&n.id) {
                true
            } else if options.statically_module_names.contains(&n.id) {
                false
            } else {
                return None;
            };
            let rhs = c.comparators.first()?;
            if !crate::is_none_expr(rhs) {
                return None;
            }
            match c.ops.first() {
                Some(crate::Compares::Is) | Some(crate::Compares::Eq) => Some(is_none),
                Some(crate::Compares::IsNot) | Some(crate::Compares::NotEq) => {
                    Some(!is_none)
                }
                _ => None,
            }
        }
        ExprType::BoolOp(b) => {
            let vals: Vec<Option<bool>> = b
                .values
                .iter()
                .map(|v| static_name_gate_taken(v, options))
                .collect();
            match b.op {
                crate::BoolOps::And => {
                    if vals.iter().any(|v| *v == Some(false)) {
                        Some(false)
                    } else if vals.iter().all(|v| *v == Some(true)) {
                        Some(true)
                    } else {
                        None
                    }
                }
                crate::BoolOps::Or => {
                    if vals.iter().any(|v| *v == Some(true)) {
                        Some(true)
                    } else if vals.iter().all(|v| *v == Some(false)) {
                        Some(false)
                    } else {
                        None
                    }
                }
                crate::BoolOps::Unknown => None,
            }
        }
        _ => None,
    }
}
