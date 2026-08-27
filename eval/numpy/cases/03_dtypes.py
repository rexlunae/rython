"""dtype= on the constructors."""
import numpy as np


def main() -> None:
    print(np.zeros(3, dtype=np.int64))
    print(np.ones(3, dtype=np.int32))
    print(np.zeros(2, dtype=np.float32))
    print(np.ones(4, dtype=np.bool_))
    print(np.zeros((2, 2), dtype="int64"))
    print(np.ones(3, dtype="float32"))


if __name__ == "__main__":
    main()
