"""An instance handed to a function or a loop variable is the SAME object
the container holds: a mutation through the parameter, the loop
variable, or a narrowed name is visible when the container is read
afterwards. The classes here are never mutated through a fetched alias
inside a method — only through parameters and loop variables — so the
sharing decision must see a mutation anywhere in the family."""


class Shape:
    def area(self) -> float:
        return 0.0


class Circle(Shape):
    def __init__(self, r: float):
        self.r = r

    def area(self) -> float:
        return 3.0 * self.r * self.r


class Rect(Shape):
    def __init__(self, w: float, h: float):
        self.w = w
        self.h = h

    def area(self) -> float:
        return self.w * self.h

    def scale(self, k: float) -> None:
        self.w *= k
        self.h *= k


class Counter:
    def __init__(self):
        self.n = 0


def grow(s: Shape) -> None:
    if isinstance(s, Circle):
        s.r = s.r + 1.0
    if isinstance(s, Rect):
        s.scale(2.0)


def bump(c: Counter) -> None:
    c.n += 1


def main() -> None:
    shapes: list[Shape] = [Circle(1.0), Rect(2.0, 3.0), Shape()]
    for s in shapes:
        grow(s)
    print([f"{s.area():.1f}" for s in shapes])
    first = shapes[0]
    grow(first)
    print(f"{shapes[0].area():.1f} {first.area():.1f}")
    counters = [Counter(), Counter()]
    for c in counters:
        bump(c)
        bump(c)
    bump(counters[1])
    print([c.n for c in counters])
    for c in counters:
        c.n += 10
    print(counters[0].n, counters[1].n)


if __name__ == "__main__":
    main()
