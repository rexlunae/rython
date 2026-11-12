import unittest
class A(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print('setup')
    @classmethod
    def tearDownClass(cls) -> None:
        try:
            print('teardown exit')
            raise SystemExit(4)
        finally:
            print('finally')
    def test_one(self) -> None:
        print('test')
class Z(unittest.TestCase):
    def test_later(self) -> None:
        print('unexpected later suite')
if __name__ == '__main__':
    unittest.main()
