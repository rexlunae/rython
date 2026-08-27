"""Arrays above numpy's 1000-element threshold print summarized."""
import numpy as np


def main() -> None:
    print(np.arange(1001))
    print(np.zeros(2000))


if __name__ == "__main__":
    main()
