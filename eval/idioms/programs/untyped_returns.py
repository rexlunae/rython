# Unannotated methods whose returns the converter must type: a bool fold
# (all), a list copy of a field, a try/except returning an int or a boxed
# default, and a boxed parameter returned on two paths.

class Session:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.closed = False

    def add(self, n: str) -> None:
        self.names.append(n)

    def all_short(self):
        return all([len(n) < 5 for n in self.names])

    def keys(self):
        return list(self.names)

    def find(self, n: str, default):
        try:
            return self.names.index(n)
        except ValueError:
            return default

    def first(self, r):
        if r:
            return r
        return r


def main() -> None:
    s = Session()
    s.add("ab")
    s.add("cdefgh")
    print(s.all_short(), s.keys(), s.find("cdefgh", -1), s.find("zz", -1))
    print(s.first(""), s.first("x"))


if __name__ == "__main__":
    main()
