import threading


def work(n: int) -> None:
    print("work", n, threading.current_thread().name)


class Pool:
    def __init__(self):
        self.t = None
        self.n = 3

    def start(self):
        self.t = threading.Thread(target=work, args=(self.n,))
        self.t.start()

    def stop(self):
        if self.t is not None:
            self.t.join()
            print("joined", self.t.is_alive())
            self.t = None
        print("stopped", self.t is None)

    def restart(self):
        self.stop()
        self.n = self.n + 1
        worker = threading.Thread(target=work, args=(self.n,))
        self.t = worker
        print("alive before start", worker.is_alive())
        worker.start()


def main():
    p = Pool()
    p.stop()
    p.start()
    p.stop()
    p.restart()
    p.stop()


if __name__ == "__main__":
    main()
