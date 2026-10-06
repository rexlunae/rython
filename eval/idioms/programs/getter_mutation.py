class Child:
    def __init__(self) -> None:
        self.items: list[int] = [1, 2]
        self.n = 0


class Holder:
    def __init__(self) -> None:
        self.kid = Child()

    @property
    def child(self) -> Child:
        return self.kid

    def get_child(self) -> Child:
        return self.kid


def main() -> None:
    h = Holder()
    h.child.items.pop()
    print(h.child.items)
    h.get_child().items.append(9)
    print(h.kid.items)
    c = h.child
    c.items.append(7)
    print(h.kid.items)
    h.child.n += 4
    h.get_child().n = h.child.n + 1
    h.kid.items.append(3)
    print(h.kid.n, c.n, h.kid.items)
    print(c is h.kid, h.get_child() is h.child)


if __name__ == "__main__":
    main()
