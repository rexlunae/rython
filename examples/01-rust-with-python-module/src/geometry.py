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
    signature from how the parameters are used, so this one function
    works for floats and ints alike."""
    return value * factor
