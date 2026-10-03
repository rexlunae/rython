"""OrderedDict built from an untyped (boxed) argument: requests' from_key_val_list."""

from collections import OrderedDict


def from_key_val_list(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")
    return OrderedDict(value)


def main() -> None:
    pairs = from_key_val_list([("b", 1), ("a", 2), ("c", 3)])
    print(pairs)
    print(len(pairs))
    for key in pairs:
        print(key)
    print(list(pairs))

    from_dict = from_key_val_list({"x": 1, "y": 2})
    print(from_dict)
    print(len(from_dict))

    print(from_key_val_list(None))

    again = from_key_val_list([("b", 1), ("a", 2), ("c", 3)])
    swapped = from_key_val_list([("a", 2), ("b", 1), ("c", 3)])
    print(pairs == again)
    print(pairs == swapped)
    print(pairs == {"a": 2, "b": 1, "c": 3})
    print("a" in pairs, "q" in pairs)
    print(pairs["a"])

    try:
        print(pairs["nope"])
    except KeyError as exc:
        print("KeyError:", exc)

    try:
        from_key_val_list(5)
    except ValueError as exc:
        print("ValueError:", exc)


if __name__ == "__main__":
    main()
