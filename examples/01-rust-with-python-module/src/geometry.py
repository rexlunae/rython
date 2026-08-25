"""Plane-geometry primitives for the host Rust program.

Ordinary Python: classes, single inheritance with super(), method
overrides that dispatch dynamically, f-strings, and a helper function
whose parameter types are inferred from use. The `python_module!` macro
compiles this file to Rust at build time - there is no interpreter at
runtime.
"""


class Shape:
    def __init__(self, name: str):
        self.name = name

    def area(self) -> float:
        return 0.0

    def perimeter(self) -> float:
        return 0.0

    def describe(self) -> str:
        # self.area() / self.perimeter() dispatch to the subclass override,
        # exactly as they would under CPython.
        a = self.area()
        p = self.perimeter()
        return f"{self.name}: area={a} perimeter={p}"


class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        super().__init__("rectangle")
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2.0 * (self.width + self.height)


class Circle(Shape):
    PI = 3.141592653589793

    def __init__(self, radius: float):
        super().__init__("circle")
        self.radius = radius

    def area(self) -> float:
        return Circle.PI * self.radius * self.radius

    def perimeter(self) -> float:
        return 2.0 * Circle.PI * self.radius


def scale(value, factor):
    """Parameter types are inferred: rython derives a generic Rust
    signature from `value * factor`, so this one function serves every
    type Python's `*` serves - floats, ints, strings ("na" * 4), and
    lists ([1, 2] * 3)."""
    return value * factor


def clamp(value, low, high):
    """All three parameters can be returned, so inference unifies them
    into ONE type variable: clamp<T>(value: T, low: T, high: T) -> T,
    callable with floats, ints, or strings."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def lerp(start, end, t):
    """Chained arithmetic: the inferred signature carries the
    intermediate operator-output bounds for `start + (end - start) * t`."""
    return start + (end - start) * t
