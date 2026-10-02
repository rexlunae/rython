"""Insertion-ordered mapping with collections.OrderedDict."""

from collections import OrderedDict


def main() -> None:
    od = OrderedDict()
    od["b"] = 1
    od["a"] = 2
    od["c"] = 3
    print(list(od.keys()))
    print(list(od.values()))
    print(list(od.items()))
    print(len(od))
    print(od["a"], od.get("zz"), od.get("zz", 9))

    od["b"] = 10
    od["a"] += 5
    print(list(od.items()))

    od.move_to_end("b")
    print(list(od))
    od.move_to_end("c", last=False)
    print(list(od))
    print(od.popitem())
    print(od.popitem(last=False))
    print(list(od))
    print("a" in od, "q" in od)
    print(od)

    try:
        print(od["nope"])
    except KeyError as exc:
        print("KeyError:", exc)

    od["z"] = 26
    del od["a"]
    print(len(od), bool(od))
    for k in od:
        print(k)


if __name__ == "__main__":
    main()
