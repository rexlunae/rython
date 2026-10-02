"""`from m import X as Y` binds only `Y`: the plain `X` of another module stays.

Two modules each export an item of the same name (`os.path.split` /
`re.split`, `re.escape` / `glob.escape`) with outputs that differ, so a
call that resolves against the wrong module prints a different line. The
aliased import comes after the plain one for `split` and before it for
`escape`, so both orders are exercised.
"""

from glob import escape as gescape
from os.path import split
from re import escape
from re import split as rsplit


def main() -> None:
    print(split("a/b"))
    print(rsplit(",", "x,y,z"))
    print(escape("a.b*c"))
    print(gescape("a.b*c"))


if __name__ == "__main__":
    main()
