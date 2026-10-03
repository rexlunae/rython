def build_and_read(seed):
    d = {}
    d["a"] = "1"
    d["b"] = str(seed)
    got = d.get("a", "?")
    missing = d.get("zz", "D")
    keys = list(d.items())
    n = 0
    for k, v in keys:
        n = n + len(k) + len(v)
    return got + missing + str(n)


def pop_and_probe(seed):
    d = {"a": "1", "b": str(seed)}
    first = d.pop("a")
    left = len(d.items())
    return first + str(left) + str("b" in d)


print(build_and_read(2))
print(pop_and_probe(9))
