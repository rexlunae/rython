"""A property setter must not change a later value in the same RHS tuple."""

x = "before"


class Pair:
    @property
    def a(self):
        return 0

    @a.setter
    def a(self, value):
        global x
        x = "after"

    def update(self):
        self.a, self.b = 1, x


def main():
    pair = Pair()
    pair.update()
    print(x, pair.b)


if __name__ == "__main__":
    main()
