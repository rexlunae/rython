import threading


def work() -> None:
    print("worker must not start")


class Runner:
    def __init__(self):
        self.value = 7

    def configure(self) -> None:
        t, t = threading.Thread(target=work), 0
        self.value = t


def main() -> None:
    runner = Runner()
    runner.configure()
    print(runner.value)


if __name__ == "__main__":
    main()
