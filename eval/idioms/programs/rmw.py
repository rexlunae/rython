"""A read-modify-write through a container-held (shared) object reads the
target ONCE before the operand runs, as Python does: an operand that
mutates the same object never changes what the operation reads, for the
plain store, every augmented operator, and a property with a setter."""


class Counter:
    def __init__(self):
        self.n = 10
        self._x = 1

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, v: int) -> None:
        self._x = v


def take(c: Counter) -> int:
    c.n = c.n + 100
    return 3


def main() -> None:
    counters = [Counter(), Counter()]
    c = counters[0]
    c.n -= take(c)
    print(c.n, counters[0].n)
    c.n *= take(c)
    print(c.n)
    c.n += take(c)
    print(c.n)
    c.n &= take(c)
    print(c.n)
    d = counters[1]
    d.x = d.x + 1
    d.x += d.x
    print(d.x, counters[1].x)


if __name__ == "__main__":
    main()
