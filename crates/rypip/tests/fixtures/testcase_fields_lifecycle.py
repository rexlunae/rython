import unittest


class Base(unittest.TestCase):
    def test_total(self, amount: int = 3):
        self.total = amount
        self.assertEqual(self.total, amount)
        print("base", self.total)


class Branches(unittest.TestCase):
    def test_both(self):
        flag = False
        if flag:
            self.total = 4
        else:
            self.total = 3
        self.test_value = 4
        self.assertEqual(self.total, 3)
        self.assertEqual(self.test_value, 4)
        print("branches", self.total, self.test_value)

    def test_raise(self):
        flag = False
        if flag:
            raise ValueError()
        else:
            self.total = 6
        print("raise", self.total)

    def test_return(self):
        flag = False
        if flag:
            return
        self.total = 5
        print("return", self.total)


class Child(Base):
    def test_total(self, amount: int = 7):
        self.total = amount
        self.assertEqual(self.total, 7)
        print("child", self.total)


class Descendant(Child):
    pass


class FixtureBase(unittest.TestCase):
    def setUp(self):
        flag = False
        if flag:
            self.start = 9
        else:
            self.start = 11
        print("setup", self.start)

    def tearDown(self):
        print("teardown", self.start)

    def test_a(self):
        self.total = 3
        self.assertEqual(self.total, 3)
        self.assertEqual(self.start, 11)
        self.start += 1
        print("body", self.total, self.start)


class FixtureChild(FixtureBase):
    def test_b(self):
        self.assertEqual(self.start, 11)
        self.total = 8
        print("body", self.total, self.start)


class ErrorAfterStore(unittest.TestCase):
    def setUp(self):
        self.seed = 4
        print("error setup", self.seed)

    def test_error(self):
        self.total = 3
        print("error body", self.total)
        raise ValueError("boom")

    def tearDown(self):
        print("error cleanup", self.seed)


class SetupFailure(unittest.TestCase):
    def setUp(self):
        self.seed = 1
        print("broken setup", self.seed)
        raise RuntimeError("setup broke")

    def test_never(self):
        self.total = 99
        print("unreachable body", self.total)

    def tearDown(self):
        print("unreachable cleanup", self.seed)


@unittest.skip("inherited gate")
class SkippedBase(unittest.TestCase):
    def test_total(self):
        self.total = 3
        print("unreachable skipped", self.total)


class SkippedChild(SkippedBase):
    pass


class TeardownFailure(unittest.TestCase):
    def setUp(self):
        self.seed = 2
        print("teardown setup", self.seed)

    def test_total(self):
        self.total = 4
        print("teardown body", self.total)

    def tearDown(self):
        print("broken teardown", self.seed)
        raise RuntimeError("teardown broke")


if __name__ == "__main__":
    unittest.main()
