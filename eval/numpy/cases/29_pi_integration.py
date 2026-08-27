"""The example workload, deterministic and fixed-size."""
import numpy as np


def quarter_circle(n: int) -> float:
    xs = np.linspace(0.0, 1.0, n)
    ys = np.sqrt(np.subtract(1.0, np.multiply(xs, xs)))
    return np.mean(ys)


def oscillator_spread(n_steps: int) -> float:
    pos = np.linspace(-1.0, 1.0, 1000)
    vel = np.zeros(1000)
    i = 0
    while i < n_steps:
        vel = np.add(vel, np.multiply(np.multiply(pos, -1.0), 0.01))
        pos = np.add(pos, np.multiply(vel, 0.01))
        i = i + 1
    return np.mean(np.multiply(pos, pos))


def main() -> None:
    print(4.0 * quarter_circle(10000))
    print(oscillator_spread(500))


if __name__ == "__main__":
    main()
