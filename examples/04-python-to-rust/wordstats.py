"""wordstats: a tiny word-frequency report.

Ordinary Python - classes, inheritance with super(), overridden methods,
dicts, f-strings, and three fully unannotated (type-inferred) helpers -
used as the walkthrough program for converting Python to Rust with
rypip. See README.md next to this file; the crate it generates is
checked in under generated/ for reading.
"""


class Tally:
    """Base class: a named accumulator."""

    def __init__(self, label: str):
        self.label = label
        self.total = 0

    def add(self, n: int) -> None:
        self.total = self.total + n

    def summary(self) -> str:
        return f"{self.label}: {self.total}"


class WordTally(Tally):
    """Counts words and tracks per-word frequencies."""

    def __init__(self):
        super().__init__("words")
        self.freq: dict[str, int] = {}

    def add_text(self, text: str) -> None:
        words = text.split()
        self.add(len(words))
        for w in words:
            self.freq[w] = self.freq.get(w, 0) + 1

    def summary(self) -> str:
        # Extend the base summary rather than replacing it.
        base = super().summary()
        distinct = len(self.freq)
        return f"{base} ({distinct} distinct)"

    def top(self) -> str:
        best = ""
        best_count = 0
        for w in sorted(self.freq.keys()):
            count = self.freq.get(w, 0)
            if count > best_count:
                best = w
                best_count = count
        return f"{best} x{best_count}"


def longest(words):
    """No annotations: `for w in words` infers an iterable, and the
    accumulator's `best = ""` seed concretizes the element type - the
    signature is `longest<T: IntoIterator<Item = String>>(words: T) ->
    Result<String, _>`."""
    best = ""
    for w in words:
        if len(w) > len(best):
            best = w
    return best


def total_chars(words):
    """No annotations: an integer-seeded accumulator over an inferred
    iterable, with `len(w)` bounding the elements."""
    n = 0
    for w in words:
        n = n + len(w)
    return n


def within(value, low, high):
    """No annotations: rython infers a generic, comparison-bounded Rust
    signature from how the parameters are used."""
    return low <= value and value <= high


if __name__ == "__main__":
    tally = WordTally()
    tally.add_text("the quick brown fox jumps over the lazy dog")
    tally.add_text("the dog barks")
    print(tally.summary())
    print(f"top: {tally.top()}")
    print(f"longest: {longest('the quick brown fox'.split())}")
    print(f"chars: {total_chars('the quick brown fox'.split())}")
    print(f"tweet-sized: {within(tally.total, 1, 280)}")
