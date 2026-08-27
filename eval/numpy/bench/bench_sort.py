"""Sort throughput. The input is rebuilt-free: a fixed unsorted array is
sorted each iteration and the result consumed by a sum."""
import time

import numpy as np

SIZES = [1000, 100000, 2000000]


def report(kernel: str, n: int, reps: int, seconds: float, check: float) -> None:
    print(f"{kernel}\t{n}\t{reps}\t{seconds}\t{check}")


def reps_for(n: int) -> int:
    r = 4000000 // n
    if r < 3:
        return 3
    return r


# NOTE: this would read better as a helper returning np.ndarray, but a
# local assigned from such a call is typed PyValue and the generated crate
# does not compile (REPORT.md finding C4), so it is inlined at each site.
def bench_sort(n: int) -> None:
    reps = reps_for(n)
    a = np.sin(np.multiply(np.linspace(0.0, 1.0, n), 1000.0))
    acc = np.sum(np.sort(a))
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(np.sort(a))
        i = i + 1
    t1 = time.perf_counter()
    report("sort+sum", n, reps, (t1 - t0) / reps, acc)


def bench_argsort(n: int) -> None:
    reps = reps_for(n)
    a = np.sin(np.multiply(np.linspace(0.0, 1.0, n), 1000.0))
    acc = np.sum(np.argsort(a))
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(np.argsort(a))
        i = i + 1
    t1 = time.perf_counter()
    report("argsort+sum", n, reps, (t1 - t0) / reps, acc)


def bench_sum_only(n: int) -> None:
    """The sum baseline to subtract from the two above."""
    reps = reps_for(n)
    a = np.sin(np.multiply(np.linspace(0.0, 1.0, n), 1000.0))
    acc = np.sum(a)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(a)
        i = i + 1
    t1 = time.perf_counter()
    report("sum-baseline", n, reps, (t1 - t0) / reps, acc)


def main() -> None:
    print("kernel\tn\treps\tseconds_per_rep\tchecksum")
    for n in SIZES:
        bench_sort(n)
        bench_argsort(n)
        bench_sum_only(n)


if __name__ == "__main__":
    main()
