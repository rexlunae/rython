"""A work queue and a sliding window built on collections.deque."""

from collections import deque


def drain(q: deque[int]) -> list[int]:
    out: list[int] = []
    while q:
        out.append(q.popleft())
    return out


def window_sums(values: list[int], size: int) -> list[int]:
    window: deque[int] = deque(maxlen=size)
    sums: list[int] = []
    for v in values:
        window.append(v)
        if len(window) == size:
            sums.append(sum(window))
    return sums


def main() -> None:
    d = deque([1, 2])
    d.append(3)
    d.appendleft(0)
    print(len(d))
    print(d)
    print(list(d))
    print(d[0], d[-1])
    print(d.pop(), d.popleft())
    print(list(d))

    d.extend([7, 8, 9])
    d.rotate(2)
    print(list(d))
    d.rotate(-1)
    print(list(d))
    print(2 in d, 5 in d)

    work = deque([10, 20, 30])
    print(drain(work))

    print(window_sums([1, 2, 3, 4, 5, 6], 3))

    recent = deque([1, 2, 3], maxlen=3)
    recent.append(4)
    print(list(recent), recent.maxlen)
    recent.appendleft(0)
    print(list(recent))
    print(d.maxlen)

    empty: deque[int] = deque()
    try:
        empty.pop()
    except IndexError as exc:
        print("IndexError:", exc)
    try:
        empty.popleft()
    except IndexError as exc:
        print("IndexError:", exc)

    for item in deque("abc"):
        print(item)


if __name__ == "__main__":
    main()
