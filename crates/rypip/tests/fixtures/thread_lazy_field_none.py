import threading


def work(n: int) -> None:
    print("work", n, threading.current_thread().name)


class Eager:
    def __init__(self):
        self.t = None
        self.go()

    def go(self):
        self.t = threading.Thread(target=work, args=(1,))

    def finish(self):
        self.t.start()
        self.t.join()
        print("eager", self.t.is_alive())


class Lazy:
    def __init__(self):
        self.t = None

    def ensure(self):
        if self.t is None:
            self.t = threading.Thread(target=work, args=(2,))
            self.t.start()

    def safe_join(self):
        try:
            self.t.join()
            print("joined", self.t.is_alive())
        except AttributeError as e:
            print("caught", e)


class Gate:
    def __init__(self, go: bool):
        self.t = None
        if go:
            self.t = threading.Thread(target=work, args=(10,))

    def run(self):
        if self.t is None:
            print("no thread")
        else:
            self.t.start()
            self.t.join()


def main():
    e = Eager()
    e.finish()
    lazy = Lazy()
    lazy.safe_join()
    lazy.ensure()
    lazy.ensure()
    lazy.safe_join()
    print("none?", lazy.t is None)
    Gate(False).run()
    Gate(True).run()


if __name__ == "__main__":
    main()
