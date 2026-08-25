"""Array compute with a selectable execution backend (CPU today, GPU by
cargo feature): estimate pi by midpoint integration and run a wave of
harmonic oscillators, all through rython's numpy subset.

Every numpy operation in the compiled program funnels through one
execution engine, chosen once per process: np.set_backend("...") in the
program (or the rythonc --numpy-backend flag, which pins it at startup),
or "auto" - the best engine compiled into the binary.

"scalar" is the always-available sequential engine; "rayon", "simd",
"cuda", and "vulkan" name the accelerated engines behind the stdpython
cargo features numpy-rayon / numpy-simd / numpy-cuda / numpy-vulkan.
Requesting an engine the binary was not built with is a loud
RuntimeError - never a silent fallback. Run with --gpu to see it.
"""

import argparse
import numpy as np


def quarter_circle(n: int) -> float:
    """Mean of sqrt(1 - x^2) over [0, 1] - converges to pi/4."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.sqrt(np.subtract(1.0, np.multiply(xs, xs)))
    return np.mean(ys)


def accel(pos: np.ndarray) -> np.ndarray:
    """Restoring force of a unit harmonic oscillator: a = -x."""
    return np.multiply(pos, -1.0)


def oscillator_spread(n_steps: int) -> float:
    """Mean squared displacement of 1000 oscillators after n_steps of
    semi-implicit Euler integration of x'' = -x."""
    pos = np.linspace(-1.0, 1.0, 1000)
    vel = np.zeros(1000)
    i = 0
    while i < n_steps:
        vel = np.add(vel, np.multiply(accel(pos), 0.01))
        pos = np.add(pos, np.multiply(vel, 0.01))
        i = i + 1
    return np.mean(np.multiply(pos, pos))


def main() -> None:
    parser = argparse.ArgumentParser(prog="simulate", description="numpy backend demo")
    parser.add_argument("--samples", type=int, default=1000000)
    parser.add_argument("--gpu", action="store_true", help="run on the CUDA engine")
    args = parser.parse_args()

    if args.gpu:
        # Loud RuntimeError unless the binary was built with the
        # numpy-cuda feature and a CUDA driver is present.
        np.set_backend("cuda")

    pi = 4.0 * quarter_circle(args.samples)
    print(f"pi ~= {pi}")
    print(f"oscillator <x^2> after 500 steps: {oscillator_spread(500)}")


if __name__ == "__main__":
    main()
