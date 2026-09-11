"""Value `or`/`and`: the SELECTED operand is returned, never a bool.
Short-circuiting and evaluation order are observable through the log."""
log: list[str] = []


def push_it(v: str) -> None:
    log.append(v)


def note(s: str) -> str:
    push_it(s)
    return s


def pick_path(path: str) -> str:
    return path or "/default"


def pick_list(items: list[int], fallback: list[int]) -> list[int]:
    return items or fallback


def pick_both(a: str, b: str) -> str:
    return a and b


def main() -> None:
    print(pick_path(""))
    print(pick_path("index.html"))
    print(pick_list([], [1, 2]))
    print(pick_list([0], [1, 2]))
    print(pick_both("", "second"))
    print(pick_both("first", "second"))
    log.clear()
    r = note("") or note("SECOND")
    print("or-falsy", r, log)
    log.clear()
    r = note("x") or note("SECOND")
    print("or-truthy", r, log)
    log.clear()
    r = note("x") and note("SECOND")
    print("and-truthy", r, log)
    log.clear()
    r = note("") and note("SECOND")
    print("and-falsy", r, log)


if __name__ == "__main__":
    main()
