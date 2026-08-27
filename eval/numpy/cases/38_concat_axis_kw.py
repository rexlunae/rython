"""concatenate with axis as a keyword (rython requires the keyword form)."""
import numpy as np


def main() -> None:
    p = np.reshape(np.arange(4), (2, 2))
    q = np.reshape(np.arange(4, 8), (2, 2))
    print(np.concatenate([p, q], axis=0))
    print(np.concatenate([p, q], axis=1))
    print(np.concatenate([np.array([1.0]), np.array([2.0])], axis=0))


if __name__ == "__main__":
    main()
