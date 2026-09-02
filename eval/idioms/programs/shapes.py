"""Inheritance, polymorphic dispatch through a base-typed list,
isinstance, __repr__, sorted by key, max, NotImplementedError."""
import math


class Shape:
    def area(self) -> float:
        raise NotImplementedError("area")

    def name(self) -> str:
        return "shape"

    def __repr__(self) -> str:
        return f"{self.name()}({self.area():.2f})"


class Circle(Shape):
    def __init__(self, r: float):
        self.r = r

    def area(self) -> float:
        return math.pi * self.r * self.r

    def name(self) -> str:
        return "circle"


class Rect(Shape):
    def __init__(self, w: float, h: float):
        self.w = w
        self.h = h

    def area(self) -> float:
        return self.w * self.h

    def name(self) -> str:
        return "rect"


class Square(Rect):
    def __init__(self, side: float):
        super().__init__(side, side)

    def name(self) -> str:
        return "square"


def describe(s: Shape) -> str:
    if isinstance(s, Square):
        return f"square of side {s.w:g}"
    if isinstance(s, Rect):
        return f"rect {s.w:g}x{s.h:g}"
    return f"{s.name()} with area {s.area():.1f}"


def main() -> None:
    shapes: list[Shape] = [Circle(1.0), Rect(2.0, 3.0), Square(2.5), Circle(0.5)]
    for s in shapes:
        print(describe(s))
    total = sum(s.area() for s in shapes)
    print(f"total {total:.3f}")
    biggest = max(shapes, key=lambda s: s.area())
    print("biggest:", biggest)
    print(sorted(shapes, key=lambda s: s.area()))
    squares = [s for s in shapes if isinstance(s, Rect)]
    print(len(squares), [s.name() for s in squares])
    try:
        print(Shape().area())
    except NotImplementedError as e:
        print("abstract:", e)


if __name__ == "__main__":
    main()
