"""Elementwise kernel throughput.

Each kernel is a dependency chain (`out = op(out, b)`) so no iteration can
be hoisted or eliminated; the result is consumed after the timed region.
Reported time is seconds per iteration.
"""
import time

import numpy as np

SIZES = [1000, 100000, 10000000]


def report(kernel: str, n: int, reps: int, seconds: float, check: float) -> None:
    print(f"{kernel}\t{n}\t{reps}\t{seconds}\t{check}")


def reps_for(n: int) -> int:
    r = 200000000 // n
    if r < 3:
        return 3
    return r


def bench_add(n: int) -> None:
    reps = reps_for(n)
    b = np.full(n, 1.0)
    out = np.zeros(n)
    out = np.add(out, b)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.add(out, b)
        i = i + 1
    t1 = time.perf_counter()
    report("add", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_multiply(n: int) -> None:
    reps = reps_for(n)
    b = np.full(n, 1.000001)
    out = np.ones(n)
    out = np.multiply(out, b)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.multiply(out, b)
        i = i + 1
    t1 = time.perf_counter()
    report("multiply", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_divide(n: int) -> None:
    reps = reps_for(n)
    b = np.full(n, 1.000001)
    out = np.ones(n)
    out = np.divide(out, b)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.divide(out, b)
        i = i + 1
    t1 = time.perf_counter()
    report("divide", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_sqrt(n: int) -> None:
    reps = reps_for(n)
    out = np.full(n, 2.0)
    out = np.sqrt(out)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.sqrt(out)
        i = i + 1
    t1 = time.perf_counter()
    report("sqrt", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_exp(n: int) -> None:
    reps = reps_for(n)
    out = np.full(n, -0.5)
    out = np.multiply(np.exp(out), -0.5)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.multiply(np.exp(out), -0.5)
        i = i + 1
    t1 = time.perf_counter()
    report("exp+multiply", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_compare(n: int) -> None:
    reps = reps_for(n)
    a = np.linspace(0.0, 1.0, n)
    out = np.zeros(n)
    out = np.where(np.greater(a, 0.5), a, out)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.where(np.greater(a, 0.5), a, out)
        i = i + 1
    t1 = time.perf_counter()
    report("greater+where", n, reps, (t1 - t0) / reps, np.sum(out))


def main() -> None:
    print("kernel\tn\treps\tseconds_per_rep\tchecksum")
    for n in SIZES:
        bench_add(n)
        bench_multiply(n)
        bench_divide(n)
        bench_sqrt(n)
        bench_exp(n)
        bench_compare(n)


if __name__ == "__main__":
    main()
