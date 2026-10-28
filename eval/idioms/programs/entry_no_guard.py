"""A `def main()` called at module level, with no `__name__` guard (#431).

The most common single-file script shape: Python runs the file as
`__main__`, so the bare call IS the program body. The conversion must emit
a binary (not a library whose `pub fn main` collides with a generated
`fn main`, rustc's E0428) and the binary must print what CPython prints.

Module-level state is set up BEFORE the call, so the output observes both
that the setup ran first and that `main` ran after it.
"""

BANNER = "start"


def helper(n: int) -> int:
    return n * n


def main() -> None:
    print(BANNER, helper(3), helper(4))
    total = 0
    for n in [1, 2, 3]:
        total += helper(n)
    print(total)


main()
