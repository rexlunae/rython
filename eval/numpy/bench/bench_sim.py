"""A realistic mixed workload: the harmonic-oscillator integration and the
quarter-circle integral from examples/02-gpu-numpy."""
import time

import numpy as np


def report(kernel: str, n: int, reps: int, seconds: float, check: float) -> None:
    print(f"{kernel}\t{n}\t{reps}\t{seconds}\t{check}")


def quarter_circle(n: int) -> float:
    xs = np.linspace(0.0, 1.0, n)
    ys = np.sqrt(np.subtract(1.0, np.multiply(xs, xs)))
    return np.mean(ys)


def oscillators(n: int, steps: int) -> float:
    pos = np.linspace(-1.0, 1.0, n)
    vel = np.zeros(n)
    i = 0
    while i < steps:
        vel = np.add(vel, np.multiply(np.multiply(pos, -1.0), 0.01))
        pos = np.add(pos, np.multiply(vel, 0.01))
        i = i + 1
    return np.mean(np.multiply(pos, pos))


def main() -> None:
    print("kernel\tn\treps\tseconds_per_rep\tchecksum")

    n = 1000000
    reps = 5
    acc = quarter_circle(1000)
    t0 = time.perf_counter()
    i = 0
    acc = 0.0
    while i < reps:
        acc = acc + quarter_circle(n)
        i = i + 1
    t1 = time.perf_counter()
    report("quarter_circle", n, reps, (t1 - t0) / reps, acc)

    n = 100000
    steps = 100
    reps = 3
    acc = oscillators(1000, 2)
    t0 = time.perf_counter()
    i = 0
    acc = 0.0
    while i < reps:
        acc = acc + oscillators(n, steps)
        i = i + 1
    t1 = time.perf_counter()
    report("oscillators_100steps", n, reps, (t1 - t0) / reps, acc)

    n = 1000
    steps = 2000
    reps = 3
    acc = oscillators(1000, 2)
    t0 = time.perf_counter()
    i = 0
    acc = 0.0
    while i < reps:
        acc = acc + oscillators(n, steps)
        i = i + 1
    t1 = time.perf_counter()
    report("oscillators_2000steps", n, reps, (t1 - t0) / reps, acc)


if __name__ == "__main__":
    main()
