# Test fixture: a module that IMPORTS A SIBLING and calls into it.
#
# `python_module!(sibling_caller)` converts one module with no index of its
# siblings, so `import sibling_helper` + `sibling_helper.greet()` must stay a
# real crate path (`use crate::sibling_helper`) and a real call -- not an
# external-module drop followed by a boxed None (issue #456 / Devin review
# on #460).
import sibling_helper


def call_greet() -> str:
    return sibling_helper.greet()


def call_add(a: int, b: int) -> int:
    return sibling_helper.add(a, b)
