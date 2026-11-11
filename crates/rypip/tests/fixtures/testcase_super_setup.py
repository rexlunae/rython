import unittest


class Direct(unittest.TestCase):
    def setUp(self):
        print("setup before")
        super().setUp()
        self.ready = True
        print("setup after")

    def test_ready(self):
        self.assertTrue(self.ready)
        print("test ready", self.ready)


if __name__ == "__main__":
    unittest.main()
