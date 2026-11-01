import threading


def work() -> None:
    print("worked")


class Runner:
    def __init__(self) -> None:
        self.worker = threading.Thread(target=work)

    def go(self) -> None:
        print(self.worker.is_alive())
        self.worker.start()
        self.worker.join()
        print(self.worker.is_alive())


def main() -> None:
    Runner().go()


if __name__ == "__main__":
    main()
