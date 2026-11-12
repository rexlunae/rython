import threading


def work(n: int) -> None:
    print('work', n)


class Runner:
    def __init__(self) -> None:
        self.t = None

    def configure(self) -> None:
        t, n = threading.Thread(target=work, args=(7,)), 0
        self.t = t
        print('count', n)

    def run(self) -> None:
        if self.t is not None:
            self.t.start()
            self.t.join()
            print('alive', self.t.is_alive())


def main() -> None:
    runner = Runner()
    runner.run()
    runner.configure()
    runner.run()


if __name__ == '__main__':
    main()
