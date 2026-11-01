counter = 1


def next_value() -> int:
    global counter
    counter += 1
    return counter


class AppError(Exception):
    pass


def main() -> None:
    try:
        raise ValueError("a", "b")
    except ValueError as error:
        print(str(error))
        print(error)
        print(repr(error))
    try:
        raise ValueError(True)
    except ValueError as error:
        print(str(error))
        print(repr(error))
    try:
        raise ValueError(1.0)
    except ValueError as error:
        print(str(error))
        print(repr(error))
    initial = counter
    try:
        raise TypeError(initial, next_value(), True)
    except TypeError as error:
        print(str(error))
        print(repr(error))
    print("counter", counter)
    try:
        raise KeyError("a", 3, False)
    except KeyError as error:
        print(str(error))
        print(repr(error))
    try:
        raise AppError("a", 3, 1.0)
    except AppError as error:
        print(str(error))
        print(repr(error))


if __name__ == "__main__":
    main()
