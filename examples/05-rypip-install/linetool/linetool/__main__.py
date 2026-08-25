"""The linetool CLI: wc-style counts for one text file."""

import argparse

from linetool import Stats


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="linetool", description="count lines, words, and characters"
    )
    parser.add_argument("path", help="text file to count")
    args = parser.parse_args()

    stats = Stats()
    with open(args.path) as f:
        for line in f.readlines():
            stats.feed(line)
    print(stats.row(args.path))


if __name__ == "__main__":
    main()
