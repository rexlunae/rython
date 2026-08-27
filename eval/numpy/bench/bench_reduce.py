"""Reduction throughput. Each iteration's scalar result is accumulated, so
nothing can be hoisted out of the timed loop."""
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


def bench_sum(n: int) -> None:
    reps = reps_for(n)
    a = np.linspace(0.0, 1.0, n)
    acc = np.sum(a)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(a)
        i = i + 1
    t1 = time.perf_counter()
    report("sum", n, reps, (t1 - t0) / reps, acc)


def bench_mean(n: int) -> None:
    reps = reps_for(n)
    a = np.linspace(0.0, 1.0, n)
    acc = np.mean(a)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.mean(a)
        i = i + 1
    t1 = time.perf_counter()
    report("mean", n, reps, (t1 - t0) / reps, acc)


def bench_max(n: int) -> None:
    reps = reps_for(n)
    a = np.linspace(0.0, 1.0, n)
    acc = np.max(a)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.max(a)
        i = i + 1
    t1 = time.perf_counter()
    report("max", n, reps, (t1 - t0) / reps, acc)


def bench_std(n: int) -> None:
    reps = reps_for(n)
    a = np.linspace(0.0, 1.0, n)
    acc = np.std(a)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.std(a)
        i = i + 1
    t1 = time.perf_counter()
    report("std", n, reps, (t1 - t0) / reps, acc)


def bench_argmax(n: int) -> None:
    reps = reps_for(n)
    a = np.linspace(0.0, 1.0, n)
    iacc = np.argmax(a)
    iacc = 0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        iacc = iacc + np.argmax(a)
        i = i + 1
    t1 = time.perf_counter()
    report("argmax", n, reps, (t1 - t0) / reps, float(iacc))


def bench_dot(n: int) -> None:
    # np.vdot, not np.dot: rython's np.dot returns an NdArray even for the
    # 1-D x 1-D case, so `acc + np.dot(a, b)` does not compile
    # (REPORT.md finding C7). np.vdot returns f64 in both implementations.
    reps = reps_for(n)
    a = np.linspace(0.0, 1.0, n)
    b = np.linspace(1.0, 2.0, n)
    acc = np.vdot(a, b)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.vdot(a, b)
        i = i + 1
    t1 = time.perf_counter()
    report("vdot", n, reps, (t1 - t0) / reps, acc)


def main() -> None:
    print("kernel\tn\treps\tseconds_per_rep\tchecksum")
    for n in SIZES:
        bench_sum(n)
        bench_mean(n)
        bench_max(n)
        bench_std(n)
        bench_argmax(n)
        bench_dot(n)


if __name__ == "__main__":
    main()
