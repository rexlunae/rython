# Exception-tree dump for python-ast (see exception_tree.rs).
#
# The interpreter is the source of truth for the builtin exception
# hierarchy: every BaseException subclass in `builtins` plus the stdlib
# modules whose exceptions the rython runtime models (urllib.error,
# socket, ssl, codeop) is recorded with its real `__mro__`, as name →
# [itself, then each ancestor]. Aliases fall out of the data
# (EnvironmentError IS OSError, socket.timeout IS TimeoutError,
# ssl.CertificateError IS SSLCertVerificationError — each is a name
# bound to the same class object, so `__mro__[0]` is the canonical
# name), and multiple inheritance (ExceptionGroup, ssl's
# SSLCertVerificationError) is exactly CPython's data rather than a
# hand-copied tree.
import builtins
import codeop
import socket
import ssl
import sys
import urllib.error

_MODULES = (builtins, urllib.error, socket, ssl, codeop)


def collect(mod):
    out = {}
    for name in dir(mod):
        value = getattr(mod, name)
        if isinstance(value, type) and issubclass(value, BaseException):
            # Every class's __mro__ ends in `object`; it is not an
            # exception and never a catchable target, so the tree stops
            # at BaseException. Non-exception bases (a plain mixin, or
            # urllib's addinfourl in HTTPError's MRO) always land AFTER
            # BaseException in the linearization — trim them too.
            mro = [c.__name__ for c in value.__mro__ if c is not object]
            if "BaseException" in mro:
                mro = mro[: mro.index("BaseException") + 1]
            out[name] = mro
    return out


def dump():
    tree = {}
    for mod in _MODULES:
        for name, mro in collect(mod).items():
            if name in tree and tree[name] != mro:
                raise ValueError(
                    "exception name {!r} is bound to different classes "
                    "in two modules: {} vs {}".format(name, tree[name], mro)
                )
            tree[name] = mro
    return sys.version.split()[0], sys.platform, sorted(tree.items())
