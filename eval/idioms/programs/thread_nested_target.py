"""A nested def is a thread target, and a Thread field is returned from a method."""
import threading


class Runner:
    def __init__(self) -> None:
        self.label = "jobs"
        self.worker = threading.Thread(target=report, args=(self.label,))

    def handle(self) -> threading.Thread:
        return self.worker


def report(label: str) -> None:
    print("report", label, threading.current_thread().name)


def main() -> None:
    totals: list[int] = []

    def add(n: int) -> None:
        print("add", n, threading.current_thread().name)

    t = threading.Thread(target=add, args=(5,))
    t.start()
    t.join()
    totals.append(5)
    r = Runner()
    r.handle().start()
    r.handle().join()
    print("alive", r.handle().is_alive(), "total", sum(totals))


if __name__ == "__main__":
    main()
