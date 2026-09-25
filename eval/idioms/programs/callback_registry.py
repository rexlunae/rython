"""A module-level registry of callbacks, registered and unregistered by
functions (issue #122: botocore's _INITIALIZERS)."""

from typing import Callable

_INITIALIZERS = []


def register_initializer(callback: Callable[[str], None]) -> None:
    _INITIALIZERS.append(callback)


def unregister_initializer(callback: Callable[[str], None]) -> None:
    _INITIALIZERS.remove(callback)


def invoke_initializers(session: str) -> None:
    for initializer in _INITIALIZERS:
        initializer(session)


def hello(s: str) -> None:
    print("hello", s)


def bye(s: str) -> None:
    print("bye", s)


if __name__ == "__main__":
    register_initializer(hello)
    register_initializer(bye)
    invoke_initializers("s1")
    print(len(_INITIALIZERS))
    unregister_initializer(hello)
    invoke_initializers("s2")
    print(len(_INITIALIZERS))
