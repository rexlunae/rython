"""Boxed-value class construction with scalar args (#367).

A class whose __init__ stores an unannotated parameter into a field
(self.value = value) lowers the field to a boxed PyValue. Constructing it
with a scalar (Box(42), Box("hi")) must box the argument, or the generated
crate fails to build. The values print so a lost box is observable.
"""


class Box:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def main() -> None:
    a = Box(42)
    b = Box("hi")
    c = Box(2.5)
    print(a.get(), b.get(), c.get(), sep="|")
    # A boxed field read back as itself.
    print(a.value, b.value, c.value, sep="|")


if __name__ == "__main__":
    main()
