# A method's inferred return type must come from the method's own locals,
# not from a same-named local in whichever function calls it (#448).
class Labels:
    def keyed(self):
        s = "k"
        return {s: 1}

    def pair(self):
        n = "x"
        return [n, n]


def main() -> None:
    s = {1, 2}
    n = 3
    k = Labels().keyed()
    k["k"] += 4
    p = Labels().pair()
    p.append("y")
    print(k, len(s))
    print(p, n)


if __name__ == "__main__":
    main()
