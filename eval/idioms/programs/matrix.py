"""Nested lists, comprehensions over ranges, zip(*), tuple unpacking,
in-place mutation of a nested element, float formatting."""


def make(rows: int, cols: int) -> list[list[int]]:
    return [[r * cols + c for c in range(cols)] for r in range(rows)]


def transpose(m: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*m)]


def row_sums(m: list[list[int]]) -> list[int]:
    return [sum(row) for row in m]


def show(m: list[list[int]]) -> None:
    for row in m:
        print(" ".join(f"{x:3d}" for x in row))


def main() -> None:
    m = make(3, 4)
    show(m)
    print(row_sums(m), sum(row_sums(m)))
    t = transpose(m)
    print(len(t), len(t[0]), t[3])
    m[1][2] = -m[1][2]
    print(m[1], row_sums(m)[1])
    diag = [m[i][i] for i in range(min(len(m), len(m[0])))]
    print("diag", diag)
    pairs = [(r, c) for r in range(3) for c in range(4) if (r + c) % 3 == 0]
    print(pairs)
    total = 0
    for i, row in enumerate(m):
        for j, x in enumerate(row):
            if i == j:
                total += x
    print(total == sum(diag))
    avg = sum(sum(r) for r in m) / (len(m) * len(m[0]))
    print(f"avg {avg:.2f}", round(avg), int(avg))
    flat = [x for row in m for x in row]
    print(flat[::3], flat[-2:], max(flat), min(flat))


if __name__ == "__main__":
    main()
