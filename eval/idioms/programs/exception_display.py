class AppError(Exception):
    pass


class Child(AppError):
    pass


class Pretty:
    def __str__(self) -> str:
        return "pretty"

    def __repr__(self) -> str:
        return "Pretty()"


calls = 0


def next_key() -> str:
    global calls
    calls += 1
    return "again"


def main() -> None:
    try:
        raise Child("k")
    except AppError as error:
        print(str(error))
        print(error)
        print(repr(error))
    try:
        raise KeyError("k")
    except KeyError as error:
        print(str(error))
        print(repr(error))
    try:
        raise KeyError(5)
    except KeyError as error:
        print(str(error))
        print(repr(error))
    try:
        raise KeyError((1, "a"))
    except KeyError as error:
        print(str(error))
        print(repr(error))
    try:
        raise KeyError(next_key())
    except KeyError as error:
        print(str(error))
        print(repr(error))
    print("calls", calls)
    item = Pretty()
    print(str(item))
    print(item)
    print(repr(item))


if __name__ == "__main__":
    main()
