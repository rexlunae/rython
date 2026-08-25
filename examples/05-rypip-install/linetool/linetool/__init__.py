"""Counting core for linetool: plain Python, no CLI concerns."""


class Stats:
    """Accumulates line/word/character counts."""

    def __init__(self):
        self.lines = 0
        self.words = 0
        self.chars = 0

    def feed(self, text: str) -> None:
        self.lines = self.lines + 1
        self.words = self.words + len(text.split())
        self.chars = self.chars + len(text)

    def row(self, label: str) -> str:
        return f"{self.lines} {self.words} {self.chars} {label}"
