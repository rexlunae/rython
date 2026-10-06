"""String-literal locals returned inside list and dict literals."""


class Labeler:
    def __init__(self) -> None:
        self.count = 0

    def tags(self):
        s = "v"
        self.count += 1
        return [s, "w"]

    def mapping(self):
        s = "v"
        self.count += 1
        return {"k": s}

    def keyed(self):
        s = "k"
        self.count += 1
        return {s: 1}


def top_tags():
    s = "x"
    return [s]


def main() -> None:
    lab = Labeler()
    t = lab.tags()
    t.append("z")
    print(t, len(t))
    m = lab.mapping()
    m["j"] = "q"
    print(m, len(m))
    k = lab.keyed()
    k["k"] += 4
    print(k)
    print(top_tags())
    print(lab.count)


if __name__ == "__main__":
    main()
