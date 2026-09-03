import math


class Shape:
    def area(self) -> float:
        return 0.0

    def name(self) -> str:
        return "shape"


class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius * self.radius

    def name(self) -> str:
        return "circle"

    def same_kind(self, other: "Shape") -> bool:
        # `type(self)` on a Circle's own method is Circle; `other` is a
        # root-typed value, so the check is its runtime variant.
        return isinstance(other, type(self))


class Rect(Shape):
    def __init__(self, w: float, h: float):
        self.w = w
        self.h = h

    def area(self) -> float:
        return self.w * self.h

    def scale(self, k: float) -> None:
        self.w *= k
        self.h *= k


C = Circle


def describe(s: Shape) -> str:
    return f"{s.name()}:{s.area():.2f}"


def grown(s: Shape) -> float:
    # A mutation through a NARROWED name reaches the value the name
    # holds: the area read after it sees the change.
    if isinstance(s, C):
        s.radius = s.radius + 1.0
    if isinstance(s, Rect):
        s.scale(2.0)
    return s.area()


def main() -> None:
    shapes: list[Shape] = [Circle(1.0), Rect(2.0, 3.0), Shape()]
    for s in shapes:
        print(describe(s), f"{grown(s):8.3g}|")
    circles = sum(1 for s in shapes if isinstance(s, C))
    print(circles)
    # A TUPLE of class targets on a root-typed value: the OR of the
    # variant tests; an ancestor in the tuple answers for every value.
    quads = sum(1 for s in shapes if isinstance(s, (Rect, C)))
    anything = sum(1 for s in shapes if isinstance(s, (Shape, Rect)))
    print(quads, anything)
    probe = Circle(0.5)
    print(probe.same_kind(shapes[0]), probe.same_kind(shapes[1]), probe.same_kind(shapes[2]))
    for v in [3.14159, 1234567.0, 0.0001234, 1e16, 100.0, -0.0, 2.5e-5]:
        print(f"{v:g}|{v:10.4g}|{v:<8.2g}|")


if __name__ == "__main__":
    main()
