"""charset_normalizer's legacy.detect shape (round 99): a MODULE-level
function whose local is bound to a factory-call chain — `r =
from_bytes(byte_str).best()` where `best() -> Match | None` — then field
reads through `is not None` ternaries and and-chains. The Option binding
must seed the local (`optional_names`), the ternary TRUE branch must
narrow (reads unwrap), and the ternary's Option-ness must survive the
narrowing (`r.encoding if r is not None else None` is Option<String>,
not a String-vs-None if/else).
"""


class Match:
    def __init__(self, enc: str):
        self._e = enc
        self._l = "English"
        self._c = 0.05

    @property
    def encoding(self) -> str:
        return self._e

    @property
    def language(self) -> str:
        return self._l

    @property
    def chaos(self) -> float:
        return self._c


class Matches:
    def __init__(self, ms: list[Match]):
        self._ms: list[Match] = ms

    def best(self) -> Match | None:
        if len(self._ms):
            return self._ms[0]
        return None


def from_bytes(data: bytes) -> Matches:
    return Matches([Match("utf_8")])


def detect(byte_str: bytes) -> str:
    r = from_bytes(byte_str).best()
    encoding = r.encoding if r is not None else None
    language = r.language if r is not None and r.language != "Unknown" else ""
    confidence = 1.0 - r.chaos if r is not None else None
    if confidence is not None and confidence >= 0.9:
        confidence -= 0.2
    # The mutations stay observable through STRING rendering: the
    # `encoding`/`confidence` locals are Option bindings, so the ternary
    # forms render Some-wrapped values; the string concatenation forces
    # the String branch types.
    label = (encoding if encoding is not None else "none") + "/" + language
    if confidence is not None:
        label += "/" + str(confidence)
    print(label)
    return label


def main() -> None:
    result = detect(b"sample bytes")
    print("done:", result)


if __name__ == "__main__":
    main()
