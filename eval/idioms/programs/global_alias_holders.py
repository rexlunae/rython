class Box:
    def __init__(self):
        self.n = 0


current = Box()


def bump() -> None:
    current.n += 1


def set5(b: Box) -> None:
    b.n = 5


def run_param() -> None:
    set5(current)
    print(current.n)


def run_list() -> None:
    items = [current]
    items[0].n += 1
    bump()
    print(current.n, items[0].n)


def main() -> None:
    run_param()
    run_list()


if __name__ == "__main__":
    main()
