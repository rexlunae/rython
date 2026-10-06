# A function's inferred return type must come from the function's own
# scope, not from a same-named local in whichever function calls it
# (#448): neither from the caller's binding of a name the callee binds
# itself, nor from a caller local that shadows a module name the callee
# reads.
LIMITS = [7, 8]


class Labels:
    def keyed(self):
        s = "k"
        return {s: 1}

    def pair(self):
        n = "x"
        return [n, n]

    def first(self):
        return [LIMITS[0], 1]


def first():
    return [LIMITS[0], 2]


def main() -> None:
    s = {1, 2}
    n = 3
    LIMITS = ["shadow"]
    k = Labels().keyed()
    k["k"] += 4
    p = Labels().pair()
    p.append("y")
    q = Labels().first()
    q.append(5)
    r = first()
    r.append(6)
    print(k, len(s))
    print(p, n)
    print(q, r, LIMITS)


if __name__ == "__main__":
    main()
