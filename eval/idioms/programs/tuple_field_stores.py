"""Tuple-unpacked attribute stores infer fields and coerce their values."""


class Pair:
    def __init__(self):
        self.number, self.label = 1, "first"

    def update(self):
        self.number, self.label = 2, "updated"


def main():
    pair = Pair()
    pair.update()
    print(pair.number, pair.label)


if __name__ == "__main__":
    main()
