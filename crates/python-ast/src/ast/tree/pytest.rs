//! pytest collection needs its Python runtime; it is not an inert external call.
//! Keep ordinary modules (including test_* definitions) as ordinary modules,
//! but refuse references to the unsupported runner before external-call drops.

use std::collections::HashSet;

use crate::{CodeGen, ExprType, PythonOptions, SymbolTableNode, SymbolTableScopes};

#[derive(Clone, Copy, PartialEq, Eq)]
enum PytestBinding {
    Module,
    InternalPackage,
    Config,
    Cmdline,
    Main,
    ConsoleMain,
}

impl PytestBinding {
    fn from_module_name(name: &str) -> Option<Self> {
        match name {
            "pytest" => Some(Self::Module),
            "_pytest" => Some(Self::InternalPackage),
            "_pytest.config" => Some(Self::Config),
            _ => None,
        }
    }

    fn member(self, name: &str) -> Option<Self> {
        match name {
            "main" if matches!(self, Self::Module | Self::Cmdline | Self::Config) => Some(Self::Main),
            "console_main" if matches!(self, Self::Module | Self::Config) => Some(Self::ConsoleMain),
            "cmdline" if self == Self::Module => Some(Self::Cmdline),
            "config" if self == Self::InternalPackage => Some(Self::Config),
            _ => None,
        }
    }

    fn of(
        expr: &ExprType,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
        seen: &mut HashSet<(Vec<String>, String)>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        crate::stack_guard::check()?;
        match expr {
            ExprType::Name(name) => {
                if crate::ast::tree::type_ctx::name_is_locally_bound(&name.id, options)
                    || !seen.insert((options.this_module_path.clone(), name.id.clone()))
                {
                    return Ok(None);
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
                        let Some(alias) = import.names.iter().find(|alias| {
                            alias.asname.as_deref().unwrap_or(&alias.name) == name.id
                                || alias.name == name.id
                                || (alias.asname.is_none()
                                    && alias.name.split('.').next() == Some(name.id.as_str()))
                        }) else {
                            return Ok(None);
                        };
                        let path: Vec<_> = alias.name.split('.').map(str::to_string).collect();
                        if crate::module_defs_contains(options, &path)
                            || options.python_modules.contains(&alias.name)
                        {
                            return Ok(None);
                        }
                        // An unaliased dotted import binds the package root;
                        // an alias (including the symbol table's canonical
                        // dotted spelling) binds the actual imported module.
                        let imported = if name.id == alias.name || alias.asname.is_some() {
                            alias.name.as_str()
                        } else {
                            alias.name.split('.').next().unwrap_or("")
                        };
                        Ok(Self::from_module_name(imported))
                    }
                    Some(SymbolTableNode::ImportFrom(import)) => {
                        let path = import.resolved_module_path(options);
                        let defining = if import.names.iter().any(|alias| alias.name == "*") {
                            name.id.clone()
                        } else {
                            import.defining_name(&name.id)
                        };
                        if crate::module_defs_contains(options, &path) {
                            return Self::in_module(&path, defining, options, seen);
                        }
                        if import.level != 0 || options.python_modules.contains(&import.module) {
                            return Ok(None);
                        }
                        Ok(Self::from_module_name(&import.module).and_then(|m| m.member(&defining)))
                    }
                    _ => Ok(None),
                }
            }
            ExprType::Attribute(attr) => {
                if let Some(path) = crate_module_expr(&attr.value, options, symbols) {
                    return Self::in_module(&path, attr.attr.clone(), options, seen);
                }
                Ok(Self::of(&attr.value, options, symbols, seen)?
                    .and_then(|binding| binding.member(&attr.attr)))
            }
            _ => Ok(None),
        }
    }

    /// Re-exports use the defining module's bindings, including when the
    /// caller reaches that module through a dotted attribute chain.
    fn in_module(
        path: &[String],
        name: String,
        options: &PythonOptions,
        seen: &mut HashSet<(Vec<String>, String)>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        crate::stack_guard::check()?;
        let Some(key) = crate::module_defs_key(options, path) else {
            return Ok(None);
        };
        let module = &options.module_defs[key];
        let mut source_options = options.clone();
        source_options.module_path = if options.module_defs.keys()
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
        let source_symbols = module.as_ref().clone().find_symbols(SymbolTableScopes::new());
        Self::of(&ExprType::Name(crate::Name { id: name }), &source_options, &source_symbols, seen)
    }
}

/// Resolve module receivers through the existing import authority. Peel
/// attributes iteratively so this lookup itself cannot exhaust the stack.
fn crate_module_expr(
    mut expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<Vec<String>> {
    let mut members = Vec::new();
    while let ExprType::Attribute(attr) = expr {
        members.push(attr.attr.clone());
        expr = &attr.value;
    }
    let ExprType::Name(name) = expr else { return None; };
    if crate::ast::tree::type_ctx::name_is_locally_bound(&name.id, options) {
        return None;
    }
    let mut path = super::module::crate_module_bound_to(&name.id, symbols, options)?;
    path.extend(members.into_iter().rev());
    crate::module_defs_key(options, &path).map(<[String]>::to_vec)
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
    let spelling = match PytestBinding::of(expr, options, symbols, &mut HashSet::new())? {
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
            PytestBinding::of(expr, options, symbols, &mut HashSet::new())?,
            Some(PytestBinding::Module | PytestBinding::Cmdline
                | PytestBinding::InternalPackage | PytestBinding::Config)
        )
    {
        return Err("pytest module/cmdline values are not supported yet: copying an external pytest object would lose its runner; use import aliases for module names, and run tests with CPython (`python -m pytest`) or port to the supported unittest subset with an explicit unittest.main() entry; rython refuses to silently lose requested test execution through a module value".into());
    }
    Ok(())
}

/// Check operands before an enclosing call can discard them, including
/// containers and lambda bodies, with their own parameter bindings.
pub(crate) fn reject_in_expr(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), Box<dyn std::error::Error>> {
    check_operand(expr, options, symbols, true)
}

fn check_operand(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
    value_slot: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::ast::tree::visit::{self, Descend};
    crate::stack_guard::check()?;
    reject_runner(expr, options, symbols)?;
    if value_slot {
        reject_module_value(expr, options, symbols)?;
    }
    if let ExprType::Attribute(attr) = expr {
        // A module receiver is a namespace, whereas an argument, element,
        // conditional result or subscript operand is a copied value.
        return check_operand(&attr.value, options, symbols, false);
    }
    if let ExprType::Lambda(lambda) = expr {
        for default in lambda.args.defaults.iter()
            .chain(lambda.args.kw_defaults.iter().flatten())
        {
            reject_in_expr(default, options, symbols)?;
        }
        let inner = crate::lambda_scope(lambda, &[], options);
        let inner_symbols = lambda.body_symbols(symbols.clone());
        return reject_in_expr(&lambda.body, &inner, &inner_symbols);
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
