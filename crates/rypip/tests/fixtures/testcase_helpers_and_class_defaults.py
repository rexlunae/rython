import unittest


class Totals(unittest.TestCase):
    # A class data attribute that setUp overrides per instance.
    scale = 0

    def setUp(self) -> None:
        self.scale = 2
        self.offset = self.base()

    # Reached from setUp, so it may only read what it assigns itself.
    def base(self) -> int:
        self.seed = 1
        return self.seed

    # Reached only from tests, after setUp returned.
    def scaled(self, v: int) -> int:
        return v * self.scale + self.offset

    @staticmethod
    def describe(n: int) -> str:
        return "count=" + str(n)

    @classmethod
    def label(cls, n: int) -> str:
        return "label " + str(n)

    def test_scaled(self) -> None:
        self.assertEqual(self.scaled(3), 7)
        print("scaled", self.scaled(3))

    def test_offset(self) -> None:
        print("offset", self.offset)

    def test_static(self) -> None:
        print(self.describe(self.scale), self.label(self.offset))


if __name__ == "__main__":
    unittest.main()
