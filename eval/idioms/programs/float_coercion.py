"""float-coercion of a __float__-carrying class (#367).

A class with __float__ is float-coercible: it lowers to f64 in float
contexts (From<Class> for f64 via Into<f64>), so it feeds the math
functions AND a heterogeneous [float, FloatLike] list for math.fsum.
Each value prints so a lost/incorrect coercion shows up as a diff.
"""

import math


class FloatLike:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value


def main() -> None:
    print(math.ceil(FloatLike(42.5)))
    print(math.ceil(FloatLike(-1.0)))
    print(math.floor(FloatLike(41.9)))
    print(math.fabs(FloatLike(-2.25)))
    # A heterogeneous [float, FloatLike] list feeds fsum.
    print(repr(math.fsum([1e100, FloatLike(1.0), -1e100, 1e-100])))
    print(repr(math.fsum([0.5, FloatLike(0.25), 0.125])))


if __name__ == "__main__":
    main()
