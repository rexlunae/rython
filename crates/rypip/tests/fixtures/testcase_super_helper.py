import unittest


class Base(unittest.TestCase):
    def helper(self):
        self.value = 13


class Child(Base):
    def test_helper(self):
        super().helper()
        print("helper", self.value)


if __name__ == "__main__":
    unittest.main()
