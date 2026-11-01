from builtins import ValueError as ImportedError
import builtins as builtin
from socket import error as SocketError, gaierror, herror
from ssl import SSLError

Alias = ValueError
calls = 0
state = 1


class ParseFailure(IndentationError):
    pass


class NestedParseFailure(ParseFailure):
    pass


class TabFailure(TabError):
    pass


class SyntaxFailure(SyntaxError):
    pass


def returning_branch(flag: bool) -> None:
    if flag:
        error = ValueError("returning")
    else:
        return
    print("returning branch", str(error))


def raising_branch(flag: bool) -> None:
    if flag:
        error = ValueError("raising")
    else:
        raise TypeError("stopped")
    print("raising branch", str(error))


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


def owned_text() -> str:
    return "owned"


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
    text = owned_text()
    owned_tuple = ValueError((text,))
    print(str(owned_tuple), repr(owned_tuple))
    print("retained", text)
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
    parse_empty = ParseFailure()
    nested_empty = NestedParseFailure()
    tab_empty = TabFailure()
    custom_syntax_empty = SyntaxFailure()
    parse_message = ParseFailure("indent")
    print(str(parse_empty), repr(parse_empty))
    print(str(nested_empty), repr(nested_empty))
    print(str(tab_empty), repr(tab_empty))
    print(str(custom_syntax_empty), repr(custom_syntax_empty))
    print(str(parse_message), repr(parse_message))
    try:
        raise NestedParseFailure()
    except SyntaxError as parse_error:
        print("caught syntax", str(parse_error), repr(parse_error))
    returning_branch(True)
    returning_branch(False)
    raising_branch(True)
    try:
        raising_branch(False)
    except TypeError as branch_error:
        print("caught branch", str(branch_error))
    print(str(ssl_empty), repr(ssl_empty))
    print(str(ssl_one), repr(ssl_one))
    dns = gaierror("dns")
    host = herror("host")
    print(str(dns), repr(dns))
    print(str(host), repr(host))
    try:
        raise FileNotFoundError("missing")
    except gaierror:
        print("wrong gaierror match")
    except herror:
        print("wrong herror match")
    except OSError:
        print("OSError caught FileNotFoundError")
    try:
        raise dns
    except herror:
        print("wrong sibling match")
    except gaierror as network_error:
        print("gaierror caught", repr(network_error))
    try:
        raise SocketError("os")
    except gaierror:
        print("wrong socket.error match")
    except OSError as os_error:
        print("socket.error alias", repr(os_error))


if __name__ == "__main__":
    main()
