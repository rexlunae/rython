import unittest
FLAG = True
@unittest.skipIf((seen := FLAG), 'x')
class T(unittest.TestCase):
    def test_a(self) -> None: print('body')
FLAG = False
if __name__ == '__main__':
    unittest.main()
