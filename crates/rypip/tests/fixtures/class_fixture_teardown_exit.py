import unittest
class A(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print('setup')
    @classmethod
    def tearDownClass(cls) -> None:
        print('teardown exit')
        raise SystemExit(4)
    def test_one(self) -> None:
        print('test')
class Z(unittest.TestCase):
    def test_later(self) -> None:
        print('unexpected later suite')
if __name__ == '__main__':
    try:
        unittest.main(exit=False)
    finally:
        print('finally')
