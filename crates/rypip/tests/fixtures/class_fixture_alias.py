import unittest
class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print('setup')
    @classmethod
    def tearDownClass(cls) -> None:
        print('teardown')
    def test_one(self) -> None:
        print('test')
Alias = T
if __name__ == "__main__":
    unittest.main()
