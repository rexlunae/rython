"""A list of shared objects mutated through an index, beside plain lists and
dicts mutated through theirs.

State is printed after every mutation, so a lost write or a copy-instead-of-
alias lowering shows up as a diff rather than a pass: `items[0].n += 1` must
be visible through `current`, `slots[1].n = 40` must be visible through the
list, and the plain `xs`/`d` updates must land in their containers.

The plain containers are mutated in the SAME function that reads them: a
container passed to a mutating function and read again afterwards is the
aliasing shape rython refuses loudly (issue #79), not what this program is
about.
"""


class Box:
    def __init__(self, n: int) -> None:
        self.n = n


current = Box(0)


def bump() -> None:
    current.n += 1


def run() -> None:
    items = [current]
    items[0].n += 1
    bump()
    print(current.n, items[0].n)

    xs: list[int] = [1, 2, 3]
    xs[0] += 10
    xs[1] = 42
    print(xs)

    d: dict[str, int] = {"a": 1}
    d["a"] += 5
    d["b"] = 7
    print(d["a"], d["b"], sorted(d))

    slots = [Box(10), Box(20), Box(30)]
    slots[0].n += 5
    slots[1].n = 40
    slots[2].n *= 2
    print([b.n for b in slots], len(slots))

    total = 0
    for b in slots:
        total += b.n
    print(total)


if __name__ == "__main__":
    run()
