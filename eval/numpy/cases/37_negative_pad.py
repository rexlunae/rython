"""Sign placement inside numpy's column padding."""
import numpy as np


def main() -> None:
    print(np.array([-1.0, 2.0]))
    print(np.array([-1.0, 22.0]))
    print(np.array([-1, 2]))


if __name__ == "__main__":
    main()
