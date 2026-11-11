import unittest
FLAG = True
@unittest.skipIf(FLAG, 'x')
class T(unittest.TestCase):
    def test_a(self) -> None: print('body')
if __name__ == '__main__':
    unittest.main(exit=False)
    FLAG = False
    print('finished')
