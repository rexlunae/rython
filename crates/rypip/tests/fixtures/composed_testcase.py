import unittest
class Inner:
    def __init__(self):
        self.value = 7
class T(unittest.TestCase):
    def test_first(self) -> None:
        self.inner = Inner()
        print(self.inner.value)
    def test_second(self) -> None:
        self.inner = Inner()
        self.assertEqual(self.inner.value, 7)
if __name__ == '__main__':
    unittest.main()
