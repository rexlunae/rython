import unittest
FLAG = True
class Base(unittest.TestCase):
    @unittest.skipIf(FLAG, 'x')
    def test_a(self) -> None: print('base')
class Child(Base):
    def test_a(self) -> None: print('child')
del Base
FLAG = False
if __name__ == '__main__':
    unittest.main()
