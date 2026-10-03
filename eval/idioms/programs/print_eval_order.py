"""print() and str.format evaluate every argument before rendering any of them."""

from collections import deque


class Counter:
    def __init__(self) -> None:
        self.n = 0

    def bump(self) -> int:
        self.n += 1
        return self.n

    def __str__(self) -> str:
        return f"Counter({self.n})"


class Holder:
    def __init__(self) -> None:
        self.items: list[int] = [1, 2, 3]

    def take(self) -> int:
        return self.items.pop()


def main() -> None:
    # A later argument mutates the object an earlier one names: print
    # renders AFTER evaluating all of its arguments.
    xs = [1, 2, 3]
    print(xs, xs.pop())
    print(xs.pop(), xs)
    d = {"a": 1, "b": 2}
    print(d, d.pop("a"))
    q = deque([1, 2, 3])
    print(q, q.popleft(), q)
    q2 = deque([5])
    print(q2, q2.popleft())

    # A shared class instance mutated through a method call.
    c = Counter()
    print(c, c.bump(), c.bump())
    print(c.bump(), c)

    # An attribute chain is a place too.
    h = Holder()
    print(h.items, h.take())

    # sep= and end= keep the same order.
    ys = [7, 8, 9]
    print(ys, ys.pop(), sep=" | ", end=" <\n")

    # Snapshots taken eagerly stay snapshots.
    zs = [1, 2, 3]
    print(zs[:2], len(zs), zs.pop(), zs)

    # f-strings and a literal-format % with a tuple literal render each
    # field as it is evaluated: the in-order cases.
    fs = [1, 2, 3]
    print(f"{fs} {fs.pop()}")
    pc = Counter()
    print("%s %s" % (pc.n, pc.bump()))

    # str.format evaluates all arguments, then renders. (Only scalars are
    # Display here: a list or class instance does not build, loudly.)
    wc = Counter()
    print("{} {}".format(wc.n, wc.bump()))
    print("{0} {1} {0}".format(wc.n, wc.bump()))

    # No later mutation: nothing changes.
    ok = [1, 2]
    print(ok, len(ok), sorted(ok))


if __name__ == "__main__":
    main()
