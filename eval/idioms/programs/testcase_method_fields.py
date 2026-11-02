"""A TestCase can create fields in the test body, without setUp or __init__."""
import unittest


class Totals(unittest.TestCase):
    def test_a(self):
        self.total = 3
        self.assertEqual(self.total, 3)
        self.total += 4
        print(self.total)

    def test_b(self):
        self.total = 10
        self.assertEqual(self.total, 10)
        print(self.total)


if __name__ == "__main__":
    unittest.main()
