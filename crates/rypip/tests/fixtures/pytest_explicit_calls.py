def test_add() -> None:
    assert 1 + 1 == 2

def test_sub() -> None:
    assert 2 - 1 == 0

def main() -> None:
    test_add()
    print('called test_add')
    try:
        test_sub()
    except AssertionError:
        print('caught test_sub')
    print('after tests')

if __name__ == '__main__':
    main()
