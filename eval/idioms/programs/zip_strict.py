"""zip(a, b, strict=True) — ValueError on unequal lengths (#369).

CPython 3.10+'s strict= keyword makes zip raise ValueError when the two
iterables differ in length ("zip() argument 2 is shorter than argument 1");
without it (or with strict=False) zip truncates like plain zip. The strict
form returns a Result that threads `?`, so a surrounding try/except catches
it exactly as CPython does.
"""


def main() -> None:
    print(list(zip([1, 2, 3], [4, 5, 6], strict=True)))
    try:
        for px, qx in zip([1, 2, 3], [10, 20], strict=True):
            pass
    except ValueError as e:
        print("caught:", e)
    try:
        for px, qx in zip([1, 2], [10, 20, 30], strict=True):
            pass
    except ValueError as e:
        print("caught:", e)
    # strict=False and the bare form truncate like plain zip
    print(list(zip([1, 2, 3], [10, 20])))
    print(list(zip([1, 2, 3], [10, 20], strict=False)))


if __name__ == "__main__":
    main()
