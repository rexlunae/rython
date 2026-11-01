import sys


def stop() -> None:
    print("before")
    try:
        sys.exit(3)
    finally:
        print("cleanup")


def main() -> None:
    try:
        stop()
    except Exception:
        print("wrong handler")
    except SystemExit as error:
        print("caught", str(error))
        print("code", error.code)
    print("after")


if __name__ == "__main__":
    main()
