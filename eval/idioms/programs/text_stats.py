"""String methods, dict counting, sorting by a compound key, formatted
columns."""

TEXT = """The quick brown fox jumps over the lazy dog.
The dog sleeps; the fox does not! Quick, quick, quick."""


def words(text: str) -> list[str]:
    out = []
    for raw in text.split():
        w = raw.strip(".,;!?").lower()
        if w:
            out.append(w)
    return out


def frequencies(ws: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for w in ws:
        counts[w] = counts.get(w, 0) + 1
    return counts


def top(counts: dict[str, int], n: int) -> list[tuple[str, int]]:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def main() -> None:
    ws = words(TEXT)
    print(len(ws), "words,", len(set(ws)), "distinct")
    counts = frequencies(ws)
    for word, n in top(counts, 4):
        print(f"{word:<8}{n:>3} {'#' * n}")
    longest = max(ws, key=len)
    print("longest:", longest, len(longest))
    print("starts with q:", sorted(w for w in counts if w.startswith("q")))
    title = " ".join(w.capitalize() for w in ws[:4])
    print(title, "|", title.swapcase(), "|", title.replace(" ", "_"))
    print(TEXT.count("the"), TEXT.lower().count("the"), "the" in counts)


if __name__ == "__main__":
    main()
