"""math.gamma (#369).

The gamma of a POSITIVE INTEGER is the exact integer factorial-family value
(24.0 for 5!, 120.0 for 6!, 2.0 for 3!, 1.0 for 1!) — byte-identical across
platforms and CPython versions, so these are transcript-safe. Non-integer
gamma (e.g. gamma(0.5) / gamma(-2.5)) goes through CPython's platform lgamma
and differs by ulps across libms (the cbrt family), so those are pinned
within tolerance in the runtime test; the pole ValueError behavior is
runtime-pinned too (the message changed between 3.12 and 3.14).
"""

import math


def main() -> None:
    print(repr(math.gamma(1.0)))
    print(repr(math.gamma(2.0)))
    print(repr(math.gamma(3.0)))
    print(repr(math.gamma(5.0)))
    print(repr(math.gamma(6.0)))


if __name__ == "__main__":
    main()
