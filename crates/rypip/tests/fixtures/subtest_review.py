import unittest
class A(unittest.TestCase):
    @unittest.expectedFailure
    def test_expected(self) -> None:
        try:
            with self.subTest(n=1):
                with self.subTest(n=2):
                    self.assertEqual(1, 2)
            print('after expected')
        except AssertionError:
            print('caught assertion')
        finally:
            print('finally expected')
    def tearDown(self) -> None:
        print('expected teardown')
class B(unittest.TestCase):
    def setUp(self) -> None:
        with self.subTest(n=3):
            self.assertEqual(1, 2)
        print('after setup')
    def test_never(self) -> None:
        print('TEST')
    def tearDown(self) -> None:
        print('TEARDOWN')
class C(unittest.TestCase):
    def test_documented(self) -> None:
        """Documented test."""
        with self.subTest(n=4):
            self.assertEqual(1, 2)
class D(unittest.TestCase):
    def test_empty(self) -> None:
        with self.subTest():
            self.assertEqual(1, 2)
if __name__ == "__main__":
    unittest.main()
