class Counter:
    def __init__(self):
        self.n = 0

    def bump(self, k: int) -> int:
        self.n += k
        return self.n


class Pool:
    def __init__(self):
        self.label = "pool"
        self.c = Counter()


_instance = None
_pool = None
current = Counter()


def get_instance() -> Counter:
    global _instance
    if _instance is None:
        _instance = Counter()
    return _instance


def get_pool() -> Pool:
    global _pool
    if _pool is None:
        _pool = Pool()
    return _pool


def swap() -> None:
    global current
    current = Counter()


def main() -> None:
    a = get_instance()
    a.bump(2)
    b = get_instance()
    print(b.bump(3))
    print(a is b)
    p = get_pool()
    p.label = "changed"
    p.c.bump(9)
    print(get_pool().label, get_pool().c.n)
    old = current
    old.bump(4)
    print(current.n)
    swap()
    print(current.n, old.n)


if __name__ == "__main__":
    main()
