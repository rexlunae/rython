import unittest
class A(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print('setup assertion')
        raise AssertionError('setup broken')
    def test_never(self) -> None:
        print('never')
Alias = A
class B(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print('teardown assertion')
        raise AssertionError('teardown broken')
    def test_one(self) -> None:
        print('test')
unittest.main()
