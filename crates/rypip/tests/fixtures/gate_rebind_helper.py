import unittest
FLAG = True
def change() -> None:
    global FLAG
    FLAG = False
@unittest.skipIf(FLAG, 'x')
class T(unittest.TestCase):
    def test_a(self) -> None: print('body')
if __name__ == '__main__':
    change()
    unittest.main()
