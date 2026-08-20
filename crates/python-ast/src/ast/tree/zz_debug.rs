#[cfg(test)]
mod tests {
    use test_log::test;
    use crate::{CodeGen, CodeGenContext, PythonOptions, SymbolTableScopes, parse, ast::tree::type_ctx::annotation_type_info};

    #[test]
    fn debug_opt_annotation() {
        let m = parse("def f(cp_isolation: list[str] | None = None) -> list[str]:\n    return cp_isolation\n", "d.py").unwrap();
        eprintln!("ANNOTATION INFO: {:?}", annotation_type_info(&m.raw.body[0].statement));
    }
}
