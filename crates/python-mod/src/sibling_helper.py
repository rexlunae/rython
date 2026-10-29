# Test fixture: the sibling that sibling_caller.py imports.
def greet() -> str:
    return "hello from sibling"


def add(a: int, b: int) -> int:
    return a + b
