"""Values returned by dict / list literals.

Issue #137's requests/urllib3 `found unit type ()` sites: a returned
literal whose inferred type disagreed with the code emitted for it, so
the signature collapsed to `()`. The printed values make the CONTENT
observable, so a lowering that dropped or transmuted an element shows up
here and not merely as a compile that started working.
"""

from collections import OrderedDict


def counts() -> dict[str, int]:
    return {"a": 1, "b": 2}


def names() -> list[str]:
    return ["x", "y"]


def pair() -> tuple[str, int]:
    return ("a", 1)


def nested() -> dict[str, dict[str, int]]:
    return {"outer": {"inner": 1}}


def main() -> None:
    print(counts())
    print(names())
    print(pair())
    print(nested())
    print([{"k": 1}])
    print({"k": ["a"]})
    print(OrderedDict([("b", 1), ("a", 2)]))
    # A returned literal reached through a local, not just inline.
    total = {"n": 1 + 1}
    print(total)


if __name__ == "__main__":
    main()
