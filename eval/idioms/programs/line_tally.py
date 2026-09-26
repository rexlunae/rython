"""Tally a log file line by line: `for line in f` over a text file."""

import os

PATH = "line_tally_input.txt"


def write_log() -> None:
    with open(PATH, "w") as out:
        out.write("INFO start\n")
        out.write("WARN disk 91%\n")
        out.write("# comment\n")
        out.write("\n")
        out.write("INFO request /a\n")
        out.write("ERROR timeout /b\n")
        out.write("INFO done")


def describe(level: str, count: int) -> str:
    text = "{:<5} {}".format(level, count)
    if level == "ERROR":
        return text + " !"
    return text


def main() -> None:
    write_log()
    counts: dict[str, int] = {}
    seen = 0
    with open(PATH) as f:
        for raw in f:
            seen += 1
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            level = line.split(" ", 1)[0]
            counts[level] = counts.get(level, 0) + 1
    for level in sorted(counts):
        print(describe(level, counts[level]))
    print("lines read:", seen, "total:", sum(counts.values()))
    g = open(PATH)
    for raw in g:
        if raw.startswith("WARN"):
            print("first warning:", repr(raw))
            break
    print("next:", repr(g.readline()))
    g.close()
    try:
        for raw in g:
            print("unreachable", raw)
    except ValueError as exc:
        print(type(exc).__name__ + ":", exc)
    os.remove(PATH)


if __name__ == "__main__":
    main()
