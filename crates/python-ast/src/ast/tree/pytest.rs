//! pytest collection needs its Python runtime; it is not an inert external call.
//! Keep ordinary modules (including test_* definitions) as ordinary modules,
//! but refuse references to the unsupported runner before external-call drops.

use std::collections::HashSet;

use crate::{CodeGen, ExprType, PythonOptions, SymbolTableNode, SymbolTableScopes};

#[derive(Clone, Copy, PartialEq, Eq)]
enum PytestBinding {
    Module,
    Cmdline,
    Main,
    ConsoleMain,
}

impl PytestBinding {
    fn from_module_name(name: &str) -> Option<Self> {
        match name {
            "pytest" => Some(Self::Module),
            _ => None,
        }
    }

    fn member(self, name: &str) -> Option<Self> {
        match name {
            "main" if matches!(self, Self::Module | Self::Cmdline) => Some(Self::Main),
            "console_main" if self == Self::Module => Some(Self::ConsoleMain),
            "cmdline" if self == Self::Module => Some(Self::Cmdline),
            _ => None,
        }
    }

    fn of(
        expr: &ExprType,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
        seen: &mut HashSet<(Vec<String>, String)>,
    ) -> Option<Self> {
        match expr {
            ExprType::Name(name) => {
                if crate::ast::tree::type_ctx::name_is_locally_bound(&name.id, options)
                    || !seen.insert((options.this_module_path.clone(), name.id.clone()))
                {
                    return None;
                }
                // A known pytest runner can also arrive through a star
                // import. Explicit bindings and local parameters still win.
                match symbols.get(&name.id).or_else(|| symbols.get("*")) {
                    Some(SymbolTableNode::Alias(canonical)) => Self::of(
                        &ExprType::Name(crate::Name {
                            id: canonical.clone(),
                        }),
                        options,
                        symbols,
                        seen,
                    ),
                    Some(SymbolTableNode::Import(import)) => {
                        let alias = import.names.iter().find(|alias| {
                            alias.asname.as_deref().unwrap_or(&alias.name) == name.id
                                || alias.name == name.id
                        })?;
                        let binding = Self::from_module_name(&alias.name)?;
                        let path: Vec<_> = alias.name.split('.').map(str::to_string).collect();
                        if crate::module_defs_contains(options, &path)
                            || options.python_modules.contains(&alias.name)
                        {
                            return None;
                        }
                        Some(binding)
                    }
                    Some(SymbolTableNode::ImportFrom(import)) => {
                        let path = import.resolved_module_path(options);
                        let defining = if import.names.iter().any(|alias| alias.name == "*") {
                            name.id.clone()
                        } else {
                            import.defining_name(&name.id)
                        };
                        if let Some(key) = crate::module_defs_key(options, &path) {
                            // Follow a crate re-export in its OWN module's scope.
                            let module = options.module_defs.get(key)?;
                            let mut source_options = options.clone();
                            source_options.module_path = if options
                                .module_defs
                                .keys()
                                .any(|other| other.len() > key.len() && other.starts_with(key))
                            {
                                key.to_vec()
                            } else {
                                key[..key.len().saturating_sub(1)].to_vec()
                            };
                            source_options.this_module_path = key.to_vec();
                            source_options.name_types = Default::default();
                            source_options.local_types = Default::default();
                            source_options.called_params = Default::default();
                            source_options.param_type_vars = Default::default();
                            let source_symbols = module
                                .as_ref()
                                .clone()
                                .find_symbols(SymbolTableScopes::new());
                            return Self::of(
                                &ExprType::Name(crate::Name { id: defining }),
                                &source_options,
                                &source_symbols,
                                seen,
                            );
                        }
                        if import.level != 0 || options.python_modules.contains(&import.module) {
                            return None;
                        }
                        Self::from_module_name(&import.module)?.member(&defining)
                    }
                    _ => None,
                }
            }
            ExprType::Attribute(attr) => {
                Self::of(&attr.value, options, symbols, seen)?.member(&attr.attr)
            }
            _ => None,
        }
    }
}

pub(crate) fn reject_runner(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    if options.unindexed_crate_modules {
        // python_module! cannot rule out an unindexed sibling named pytest;
        // its normal Rust name resolution must decide that import.
        return Ok(());
    }
    let spelling = match PytestBinding::of(expr, options, symbols, &mut HashSet::new()) {
        Some(PytestBinding::Main) => "pytest.main()",
        Some(PytestBinding::ConsoleMain) => "pytest.console_main()",
        _ => return Ok(()),
    };
    Err(format!("{spelling} test execution is not supported yet: rython has no pytest collection or runner; run the suite with CPython (`python -m pytest`), or port it to the supported unittest subset with an explicit unittest.main() entry; rython refuses to silently ignore requested test execution (including runner function values)").into())
}

/// Module/cmdline objects cannot be copied into a boxed placeholder and
/// later used to bypass the runner guard. Import aliases remain supported;
/// this check runs only when a value is actually lowered, not while walking
/// a module receiver as part of a normal call.
pub(crate) fn reject_module_value(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    if !options.unindexed_crate_modules
        && matches!(
            PytestBinding::of(expr, options, symbols, &mut HashSet::new()),
            Some(PytestBinding::Module | PytestBinding::Cmdline)
        )
    {
        return Err("pytest module/cmdline values are not supported yet: copying an external pytest object would lose its runner; use import aliases for module names, and run tests with CPython (`python -m pytest`) or port to the supported unittest subset with an explicit unittest.main() entry; rython refuses to silently lose requested test execution through a module value".into());
    }
    Ok(())
}

/// Check operands before an enclosing call can discard them. Comprehension
/// targets belong to their own scope; a lambda body is checked when lowered
/// in its own scope, not against the enclosing call's imports.
pub(crate) fn reject_in_expr(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::ast::tree::visit::{self, Descend};
    reject_runner(expr, options, symbols)?;
    if matches!(expr, ExprType::Call(_)) {
        // The shared visitor lists the callee first, then the argument
        // values. Check module-valued arguments even when an enclosing
        // unmodeled call would never lower them (getattr(pytest, ...)).
        for argument in visit::subexprs_for(expr, Descend::OwnScope)
            .into_iter()
            .skip(1)
        {
            reject_module_value(argument, options, symbols)?;
        }
    }
    if let Some((elements, generators)) = visit::comprehension_parts(expr) {
        let mut inner = options.clone();
        let mut names = options.name_types.as_ref().clone();
        for generator in generators {
            reject_in_expr(&generator.iter, &inner, symbols)?;
            for name in visit::target_names(&generator.target) {
                names.insert(name.to_string(), crate::TypeInfo::PyValue);
            }
            inner.name_types = std::rc::Rc::new(names.clone());
            for filter in &generator.ifs {
                reject_in_expr(filter, &inner, symbols)?;
            }
        }
        for element in elements {
            reject_in_expr(element, &inner, symbols)?;
        }
        return Ok(());
    }
    for child in visit::subexprs_for(expr, Descend::OwnScope) {
        reject_in_expr(child, options, symbols)?;
    }
    Ok(())
}

/// The entry-point promotion path can consume a statement without lowering
/// it. Keep the same source span there as ordinary statement diagnostics.
pub(crate) fn reject_in_statement(
    statement: &crate::Statement,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::ast::tree::visit::stmt_exprs(statement)
        .into_iter()
        .try_for_each(|expr| reject_in_expr(expr, options, symbols))
        .map_err(|error| {
            crate::codegen_error(
                crate::SourceLocation::with_span(
                    "<module>",
                    statement.lineno,
                    statement.col_offset.map(|col| col + 1),
                    statement.end_lineno,
                    statement.end_col_offset,
                ),
                crate::format_error_chain(error.as_ref()),
                "",
            )
            .into()
        })
}
