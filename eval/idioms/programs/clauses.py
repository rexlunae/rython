"""Mutations and definitions that sit only in the clauses a naive walker
skips: a for-loop's else, a while-loop's else, try's else and finally,
and a yield nested inside an expression."""
from typing import Iterator, Optional


class Tally:
    def __init__(self):
        self.hits: list[str] = []
        self.note: Optional[str] = None

    def scan(self, words: list[str], stop: str) -> str:
        for w in words:
            if w == stop:
                break
        else:
            # Only place `note` is ever stored: the loop's else clause.
            self.note = "exhausted"
        n = 0
        while n < 2:
            n += 1
        else:
            self.hits.append("while-else")
        try:
            value = int(stop)
        except ValueError:
            value = -1
        else:
            self.hits.append(f"else:{value}")
        finally:
            self.hits.append("finally")
        return f"{self.note} {self.hits}"


def counter(limit: int) -> Iterator[int]:
    total = 0
    while total < limit:
        # The yield sits inside a larger expression, not as a statement.
        step = (yield total) or 1
        total += step


def drive() -> list[int]:
    return list(counter(3))


def main() -> None:
    t = Tally()
    print(t.scan(["a", "b"], "z"))
    print(t.scan(["a", "b"], "7"))
    print(drive())


if __name__ == "__main__":
    main()
