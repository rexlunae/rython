"""Rows longer than numpy's 75-column linewidth wrap onto several lines."""
import numpy as np


def main() -> None:
    print(np.linspace(0.0, 1.0, 20))
    print(np.arange(40))
    print(np.reshape(np.linspace(0.0, 1.0, 40), (2, 20)))


if __name__ == "__main__":
    main()
