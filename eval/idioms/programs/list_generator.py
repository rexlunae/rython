def collected(values: list[int]) -> list[str]:
    return list(str(value) for value in values)


def checked(value: int) -> int:
    print("item", value)
    if value == 2:
        raise ValueError("stop")
    return value


def main() -> None:
    print("strings", list(str(value) for value in [1]))
    print("filtered", list(value * value for value in range(6) if value % 2 != 0))
    print("empty", list(value for value in range(0)))
    print("nested", list(left * 10 + right for left in range(2) for right in range(3) if left != right))
    values = [1, 2, 3]
    print("returned", collected(values))
    print("retained", values)
    print("tuples", list((value, str(value)) for value in range(3)))
    try:
        result = list(checked(value) for value in [1, 2, 3])
        print(result)
    except ValueError as error:
        print("caught", str(error))
    print("after")


if __name__ == "__main__":
    main()
