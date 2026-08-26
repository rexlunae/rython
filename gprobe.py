import os

label = "start"
tag = os.environ.get("NO_SUCH_VAR", "fallback")

def compute() -> int:
    return 2

limit = compute()

def bump_label(suffix: str) -> None:
    global label
    label = label + suffix

def extend_label() -> None:
    global label
    label += "!"

def raise_limit() -> None:
    global limit
    limit = limit + 10

def retag() -> None:
    global tag
    tag = tag + "-x"

def main() -> None:
    bump_label("-a")
    bump_label("-b")
    extend_label()
    raise_limit()
    retag()
    print(label)
    print(limit)
    print(tag)

if __name__ == "__main__":
    main()
