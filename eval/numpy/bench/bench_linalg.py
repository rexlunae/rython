"""Matrix kernels: matmul, inv, solve, det."""
import time

import numpy as np

SIZES = [16, 64, 256]


def report(kernel: str, n: int, reps: int, seconds: float, check: float) -> None:
    print(f"{kernel}\t{n}\t{reps}\t{seconds}\t{check}")


def reps_for(n: int) -> int:
    r = 40000000 // (n * n * n)
    if r < 3:
        return 3
    return r


# A well-conditioned n x n matrix (identity scaled by n, plus a smooth
# ramp), inlined rather than written as an np.ndarray-returning helper:
# a local assigned from such a call is typed PyValue and the generated
# crate does not compile (REPORT.md finding C4).
def bench_matmul(n: int) -> None:
    reps = reps_for(n)
    a = np.add(np.multiply(np.eye(n), float(n)),
               np.reshape(np.linspace(0.0, 1.0, n * n), (n, n)))
    b = np.transpose(a)
    acc = np.sum(np.matmul(a, b))
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(np.matmul(a, b))
        i = i + 1
    t1 = time.perf_counter()
    report("matmul", n, reps, (t1 - t0) / reps, acc)


def bench_inv(n: int) -> None:
    reps = reps_for(n)
    a = np.add(np.multiply(np.eye(n), float(n)),
               np.reshape(np.linspace(0.0, 1.0, n * n), (n, n)))
    acc = np.sum(np.linalg.inv(a))
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(np.linalg.inv(a))
        i = i + 1
    t1 = time.perf_counter()
    report("inv", n, reps, (t1 - t0) / reps, acc)


def bench_solve(n: int) -> None:
    reps = reps_for(n)
    a = np.add(np.multiply(np.eye(n), float(n)),
               np.reshape(np.linspace(0.0, 1.0, n * n), (n, n)))
    b = np.ones(n)
    acc = np.sum(np.linalg.solve(a, b))
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.sum(np.linalg.solve(a, b))
        i = i + 1
    t1 = time.perf_counter()
    report("solve", n, reps, (t1 - t0) / reps, acc)


def bench_det(n: int) -> None:
    reps = reps_for(n)
    a = np.add(np.multiply(np.eye(n), float(n)),
               np.reshape(np.linspace(0.0, 1.0, n * n), (n, n)))
    acc = np.linalg.det(a)
    acc = 0.0
    t0 = time.perf_counter()
    i = 0
    while i < reps:
        acc = acc + np.linalg.det(a)
        i = i + 1
    t1 = time.perf_counter()
    report("det", n, reps, (t1 - t0) / reps, acc)


def main() -> None:
    print("kernel\tn\treps\tseconds_per_rep\tchecksum")
    for n in SIZES:
        bench_matmul(n)
        bench_inv(n)
        bench_solve(n)
        bench_det(n)


if __name__ == "__main__":
    main()
