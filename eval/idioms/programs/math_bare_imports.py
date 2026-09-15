"""Bare `from math import ...` still threads exceptions (#369).

A direct and an aliased bare import of the FALLIBLE `math.isqrt` inside a
`try` must reach its `except ValueError:` block (a swallowed Result would
print nothing). math.isqrt is 3.8+, so this transcript also runs under the
CI oracle; math.fma (3.13+) is covered only by the runtime/codegen pins.
"""

from math import isqrt
from math import isqrt as iq


def main() -> None:
    print(isqrt(16))
    try:
        isqrt(-1)
    except ValueError:
        print("caught direct")
    try:
        iq(-1)
    except ValueError:
        print("caught aliased")


if __name__ == "__main__":
    main()
