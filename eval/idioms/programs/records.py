"""property with a setter, classmethod constructors, staticmethod,
__eq__/__lt__ for sorting and membership, __str__."""


class Version:
    def __init__(self, major: int, minor: int, patch: int = 0):
        self.major = major
        self.minor = minor
        self._patch = patch

    @property
    def patch(self) -> int:
        return self._patch

    @patch.setter
    def patch(self, value: int) -> None:
        if value < 0:
            raise ValueError("patch must be >= 0")
        self._patch = value

    @classmethod
    def parse(cls, text: str) -> "Version":
        parts = [int(p) for p in text.split(".")]
        while len(parts) < 3:
            parts.append(0)
        return cls(parts[0], parts[1], parts[2])

    @staticmethod
    def is_valid(text: str) -> bool:
        return all(p.isdigit() for p in text.split("."))

    def bump_minor(self) -> None:
        self.minor += 1
        self.patch = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def main() -> None:
    vs = [Version.parse(s) for s in ["1.2.3", "1.10", "0.9.9", "1.2"]]
    print([str(v) for v in sorted(vs)])
    print(Version.parse("1.2") == Version(1, 2), Version(1, 2, 0) in vs)
    print(Version.is_valid("1.2.x"), Version.is_valid("3.4.5"))
    v = vs[0]
    v.bump_minor()
    print(v, v.patch, vs[0] is v)
    try:
        v.patch = -1
    except ValueError as e:
        print("bad patch:", e)
    print(v.patch, max(vs), min(vs) < max(vs))


if __name__ == "__main__":
    main()
