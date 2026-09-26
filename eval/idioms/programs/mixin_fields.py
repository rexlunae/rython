"""A mixin whose methods use an attribute only its subclasses define.

Python finds `self.retries` on the instance at runtime, whichever
subclass set it; the mixin itself never assigns it. Each subclass here
keeps its own count, and the printed totals after the mutations show
that every instance reads and writes its own attribute.
"""


class RetryMixin:
    def describe(self) -> str:
        return type(self).__name__ + " retries=" + str(self.retries)

    def retry(self) -> bool:
        if self.retries >= self.limit:
            return False
        self.retries += 1
        return True


class Fetcher(RetryMixin):
    def __init__(self, limit: int) -> None:
        self.retries = 0
        self.limit = limit


class Uploader(RetryMixin):
    def __init__(self) -> None:
        self.retries = 2
        self.limit = 3


def main() -> None:
    f = Fetcher(2)
    u = Uploader()
    attempts = [f.retry(), f.retry(), f.retry(), u.retry(), u.retry()]
    print(attempts)
    print(f.describe())
    print(u.describe())
    print(f.retries + u.retries)


if __name__ == "__main__":
    main()
