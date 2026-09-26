"""IdnaTestV2.txt conformance vectors, run through idna's UTS-46 mapping.

An in-subset port of idna 3.10's tests/test_idna_uts46.py (its runTest body,
without the unittest load_tests protocol): every vector line is checked
against decode(), encode() and transitional encode(), and the run prints
each failure and the pass/skip/fail totals.

run_vectors.py writes this file into idna's package as `idna/vectors.py`,
replacing the marker line below with the upstream test module's own
`_SKIP_TESTS` list, and runs it under CPython and under rython.
"""

from . import core as idna

@@SKIP@@

class Skip(Exception):
    pass


def check(source: str, expected: str, status: str, op: str) -> str:
    """'' when the vector holds; otherwise the failure message. Raises Skip."""
    try:
        if op == "decode":
            output = idna.decode(source, uts46=True, strict=True)
        elif op == "encode":
            output = idna.encode(source, uts46=True, strict=True).decode("ascii")
        else:
            output = idna.encode(source, uts46=True, strict=True, transitional=True).decode("ascii")
        if status != "[]":
            return "{}() did not emit required error {} for {}".format(op, status, repr(source))
        if output != expected:
            return "unexpected {}() output: {} != {}".format(op, repr(output), repr(expected))
    except (idna.IDNAError, UnicodeError, ValueError) as exc:
        if str(exc).startswith("Unknown"):
            raise Skip()
        if status == "[]":
            return "{}() raised {}: {}".format(op, type(exc).__name__, str(exc))
    return ""


def run_vector(fields: list[str]) -> str:
    source = fields[0]
    to_unicode = fields[1]
    to_unicode_status = fields[2]
    to_ascii = fields[3]
    to_ascii_status = fields[4]
    to_ascii_t = fields[5]
    to_ascii_t_status = fields[6]
    if source in _SKIP_TESTS:
        return ""
    if not to_unicode:
        to_unicode = source
    if not to_unicode_status:
        to_unicode_status = "[]"
    if not to_ascii:
        to_ascii = to_unicode
    if not to_ascii_status:
        to_ascii_status = to_unicode_status
    if not to_ascii_t:
        to_ascii_t = to_ascii
    if not to_ascii_t_status:
        to_ascii_t_status = to_ascii_status
    failure = check(source, to_unicode, to_unicode_status, "decode")
    if failure:
        return failure
    failure = check(source, to_ascii, to_ascii_status, "encode")
    if failure:
        return failure
    return check(source, to_ascii_t, to_ascii_t_status, "transitional")


def main() -> None:
    ran = 0
    skipped = 0
    failed = 0
    with open("IdnaTestV2.txt", encoding="utf-8") as tests_file:
        lineno = 0
        for raw in tests_file:
            lineno += 1
            line = raw.strip()
            if "#" in line:
                line = line.split("#", 1)[0]
            if not line:
                continue
            fields = [field.strip() for field in line.split(";")]
            ran += 1
            try:
                failure = run_vector(fields)
            except Skip:
                skipped += 1
                continue
            if failure:
                failed += 1
                print("IdnaTestV2.txt line {}: {}".format(lineno, failure))
    print("ran {} vectors: {} skipped, {} failed".format(ran, skipped, failed))


if __name__ == "__main__":
    main()
