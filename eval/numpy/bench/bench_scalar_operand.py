"""Array-array vs array-scalar vs unary, at several sizes.

Separates the three ufunc paths so their costs can be attributed: in
rython an array-scalar op broadcasts the scalar into a full-size array
first, an array-array op copies both operands, and a unary op copies its
input.
"""
import time

import numpy as np

SIZES = [10000, 100000, 1000000, 10000000]


def report(kernel: str, n: int, reps: int, seconds: float, check: float) -> None:
    print(f"{kernel}\t{n}\t{reps}\t{seconds}\t{check}")


def reps_for(n: int) -> int:
    r = 100000000 // n
    if r < 5:
        return 5
    return r


def bench_array_array(n: int) -> None:
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
    report("add_array_array", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_array_scalar(n: int) -> None:
    reps = reps_for(n)
    out = np.zeros(n)
    out = np.add(out, 1.0)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.add(out, 1.0)
        i = i + 1
    t1 = time.perf_counter()
    report("add_array_scalar", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_unary(n: int) -> None:
    reps = reps_for(n)
    out = np.full(n, 2.0)
    out = np.sqrt(out)
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        out = np.sqrt(out)
        i = i + 1
    t1 = time.perf_counter()
    report("sqrt_unary", n, reps, (t1 - t0) / reps, np.sum(out))


def bench_sum(n: int) -> None:
    reps = reps_for(n)
    a = np.full(n, 1.0)
    acc = np.sum(a)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(a)
        i = i + 1
    t1 = time.perf_counter()
    report("sum", n, reps, (t1 - t0) / reps, acc)


def main() -> None:
    print("kernel\tn\treps\tseconds_per_rep\tchecksum")
    for n in SIZES:
        bench_array_array(n)
        bench_array_scalar(n)
        bench_unary(n)
        bench_sum(n)


if __name__ == "__main__":
    main()
