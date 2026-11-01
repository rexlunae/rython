import sys
import sys as system
from sys import exit as stop
quit = sys.exit


def leave() -> None:
    stop(7)


def replace_exit() -> None:
    try:
        leave()
    finally:
        raise ValueError("replaced")


def return_exit() -> int:
    return system.exit(9)


def no_result() -> None:
    print("no result")


def optional_status(present: bool) -> int | None:
    if present:
        return 4
    return None


def main() -> int:
    kind = sys.argv[1]
    print("kind", kind)
    if kind == "zero":
        return 0
    if kind == "wrap":
        return 256
    if kind == "negative":
        return -1
    if kind == "max":
        return 9223372036854775807
    if kind == "none":
        sys.exit()
    if kind == "null":
        system.exit(None)
    if kind == "true":
        stop(True)
    if kind == "false":
        stop(False)
    if kind == "text":
        print("partial", end="")
        sys.exit("bad input")
    if kind == "float":
        sys.exit(1.25)
    if kind == "bytes":
        sys.exit(b"bye")
    if kind == "tuple":
        sys.exit(("bad", 3, None))
    if kind == "cleanup":
        try:
            leave()
        finally:
            print("cleanup")
    if kind == "caught":
        try:
            leave()
        except Exception:
            print("wrong handler")
        except SystemExit as error:
            print("caught", str(error), repr(error), error.code)
        print("after")
        return 0
    if kind == "override":
        try:
            replace_exit()
        except ValueError as error:
            print("replaced", str(error))
        return 0
    if kind == "return":
        return return_exit()
    if kind == "reraise":
        try:
            sys.exit(4)
        except SystemExit:
            print("handler")
            raise
        finally:
            print("cleanup")
    if kind == "constructed":
        error = SystemExit(5)
        print(str(error), repr(error), error.code)
        raise error
    if kind == "bare":
        raise SystemExit
    if kind == "multi":
        raise SystemExit("a", 2)
    if kind == "call_none":
        sys.exit(no_result())
    if kind == "option_none":
        sys.exit(optional_status(False))
    if kind == "option_some":
        sys.exit(optional_status(True))
    if kind == "alias":
        quit(6)
    return 3


if __name__ == "__main__":
    sys.exit(main())
