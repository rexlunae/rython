Alias = ValueError
from builtins import ValueError as ImportedError
import builtins as builtin
from ssl import SSLError
calls = 0
state = 1


def argument(value: str) -> str:
    global calls
    calls += 1
    print("argument", value)
    return value


def make() -> BaseException:
    return TypeError("returned", 3)


def next_state() -> int:
    global state
    state += 1
    return state


def describe(error: BaseException | None) -> str:
    if error is None:
        return "absent"
    return str(error)


def optional(value: int | None) -> BaseException:
    return ValueError((1, value))


def main() -> None:
    error = ValueError("v")
    print(str(error))
    print(repr(error))
    print(isinstance(error, ValueError), isinstance(error, TypeError), isinstance(error, BaseException))
    print(describe(error))
    print(describe(None))
    imported = ImportedError("imported")
    qualified = builtin.TypeError("qualified")
    print(repr(imported), repr(qualified))
    try:
        raise error
    except ValueError as caught:
        print("caught", str(caught))
    print("retained", repr(error))
    aliased = Alias(argument("a"), argument("b"))
    print(str(aliased))
    print(repr(aliased))
    print("calls", calls)
    returned = make()
    print(str(returned))
    print(repr(returned))
    nested = Exception(ValueError("inner", 4), "outer")
    print(str(nested))
    print(repr(nested))
    empty = ValueError()
    empty_string = ValueError("")
    scalar = ValueError(True, 1.0, None)
    print(str(empty), repr(empty))
    print(str(empty_string), repr(empty_string))
    print(str(scalar))
    print(repr(scalar))
    pair = ValueError((1, "tuple"))
    print(str(pair), repr(pair))
    absent = optional(None)
    present = optional(3)
    print(str(absent), repr(absent))
    print(str(present), repr(present))
    none_result = Exception(None)
    print(str(none_result), repr(none_result))
    capture = TypeError(state, next_state())
    print(str(capture))
    print("state", state)
    try:
        saved = KeyError((1, "key"))
        raise saved
    except LookupError as caught:
        print(str(caught))
        print(repr(caught))
    syntax_empty = SyntaxError()
    syntax = SyntaxError("bad")
    ssl_empty = SSLError()
    ssl_one = SSLError("tls")
    print(str(syntax_empty), repr(syntax_empty))
    print(str(syntax), repr(syntax))
    print(str(ssl_empty), repr(ssl_empty))
    print(str(ssl_one), repr(ssl_one))


if __name__ == "__main__":
    main()
