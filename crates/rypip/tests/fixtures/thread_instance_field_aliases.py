import threading
import threading as th
from threading import Thread as Worker


def work(label: str, n: int) -> None:
    print(label, n, threading.current_thread().name)


def argument(label: str) -> str:
    print("argument", label)
    return label


def execute(worker: th.Thread) -> None:
    worker.start()
    worker.join()


class Runner:
    def __init__(self, label: str) -> None:
        print("before", threading.active_count())
        self.label = label
        worker = Worker(target=work, args=(argument(self.label), 7))
        self.worker = worker
        print("created")

    def go(self) -> None:
        print(self.worker.is_alive())
        execute(self.worker)
        self.worker.join()
        print(self.worker.is_alive())


class Direct:
    def __init__(self) -> None:
        self.worker: th.Thread = th.Thread(target=work, args=("direct", 9))

    def go(self) -> None:
        self.worker.start()
        self.worker.join()


def main() -> None:
    runner = Runner("aliased")
    print("ready")
    runner.go()
    Direct().go()
    worker = Worker(target=work, args=("last", 11))
    execute(worker)


if __name__ == "__main__":
    main()
