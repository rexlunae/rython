# An Optional parameter defaulted inside an `if x is None:` guard is
# narrowed afterwards; storing it into another Optional local, or passing it
# to an Optional parameter, must still wrap it, and str() of an Optional
# prints the value or None.

def run(flag: bool, release: bool | None = None) -> str:
    if release is None:
        release = flag
    keep = release
    if keep:
        keep = False
    return again(release=release) + " " + str(keep)


def again(release: bool | None = None) -> str:
    return "none" if release is None else str(release)


def main() -> None:
    print(run(True), run(False, True), run(True, False))


if __name__ == "__main__":
    main()
