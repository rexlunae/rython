class Box:
    def __init__(self):
        self.n = 0


current = Box()
DEFAULT = Box()


def bump() -> None:
    current.n += 1


def run_alias() -> None:
    x = current
    x.n = 5
    bump()
    print(current.n, x.n)
    current.n = 9
    print(x.n)


def run_identity() -> None:
    x = current
    y = current
    print(x is y, x is current)


def run_read_only() -> None:
    x = DEFAULT
    x.n = 3
    print(DEFAULT.n, x.n)
    DEFAULT.n += 1
    print(x.n)


def main() -> None:
    run_alias()
    run_identity()
    run_read_only()


if __name__ == "__main__":
    main()
