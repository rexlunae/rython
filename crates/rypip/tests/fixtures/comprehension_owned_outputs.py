from typing import Callable, cast as checked
import typing as types


def cast(value: list[int]) -> str:
    return str(value)


def stringify(value: int) -> str:
    return str(value)


def invoke(checked: Callable[[int], str]) -> list[str]:
    return [checked(value) for value in range(3)]


def main() -> None:
    rows = list([value] for value in range(3))
    rows[0].append(9)
    print(rows)
    print([[value] for value in range(3)])
    print([(value, str(value)) for value in range(3)])
    print([str(row) for row in [[1], [2]]])
    print([checked(int, value) for value in range(3)])
    print([types.cast(list[int], [value]) for value in range(3)])
    print([cast(row) for row in [[1], [2]]])
    print(invoke(stringify))


if __name__ == "__main__":
    main()
