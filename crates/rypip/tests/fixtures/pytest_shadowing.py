import pytest as pt
class Runner:
    def main(self) -> None:
        print('local main')
def run(pt: Runner) -> None:
    pt.main()
def test_helper() -> None:
    print('test helper')
def main() -> None:
    run(Runner())
    test_helper()
if __name__ == '__main__':
    main()
