import unittest
class A(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            print('setup exit')
            raise SystemExit(3)
        finally:
            print('finally')
    @classmethod
    def tearDownClass(cls) -> None:
        print('unexpected teardown')
    def test_one(self) -> None:
        print('unexpected test')
class Z(unittest.TestCase):
    def test_later(self) -> None:
        print('unexpected later suite')
if __name__ == '__main__':
    unittest.main()
