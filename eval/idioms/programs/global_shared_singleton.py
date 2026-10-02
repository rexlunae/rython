class Counter:
    def __init__(self):
        self.n = 0

    def bump(self, k: int) -> int:
        self.n += k
        return self.n


class Registry:
    def __init__(self):
        self.items: list[Counter] = []

    def add(self, c: Counter) -> None:
        self.items.append(c)


_instance = None


def get_instance() -> Counter:
    global _instance
    if _instance is None:
        _instance = Counter()
    return _instance


reg = Registry()
reg.add(Counter())
held = Counter()
reg.add(held)


def main() -> None:
    a = get_instance()
    a.bump(2)
    b = get_instance()
    print(b.bump(3))
    print(a is b)
    print(get_instance().n)

    held.bump(7)
    reg.items[0].bump(1)
    total = 0
    for c in reg.items:
        total += c.n
        print(c.n)
    print(total)
    print(sum(c.n for c in reg.items))


if __name__ == "__main__":
    main()
