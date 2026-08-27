//! Tests pinning generated-Rust semantics to Python behavior for the
//! correctness fixes: operators, list literals, keyword escaping, assignment
//! mutability, loop else-clauses, with-statements, comprehensions, f-strings,
//! statement separators, await handling, and from-imports.

use python_ast::{CodeGen, CodeGenContext, PythonOptions, SymbolTableScopes, parse};

/// Two-module crate for the cross-module trait-mut cache tests: module A
/// defines a hierarchy whose trait widens (Dog's mutating `grow` override
/// widens Animal's trait), module B imports it and calls `grow` on a
/// Dog-typed parameter. The importing module's own per-module precompute
/// has no entry for Animal (the hierarchy lives in A), so call sites in B
/// must consult the merged cross-module table.
fn cross_module_fixture() -> (std::rc::Rc<python_ast::Module>, PythonOptions) {
    let a = parse(
        concat!(
            "class Animal:\n",
            "    def __init__(self, name: str):\n",
            "        self.name = name\n",
            "\n",
            "    def grow(self) -> None:\n",
            "        pass\n",
            "\n",
            "class Dog(Animal):\n",
            "    def grow(self) -> None:\n",
            "        self.name = self.name + \"!\"\n",
        ),
        "animals.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["animals".to_string()], std::rc::Rc::new(a));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    (options.module_defs.values().next().unwrap().clone(), options)
}

#[test]
fn ellipsis_statement_is_a_noop() {
    // Protocol stubs `def f(...) -> None: ...` are everywhere (pip's
    // build_env, typing Protocols). The bare `...` body is a no-op like
    // `pass`; using Ellipsis as a VALUE is a loud error.
    let out = compile(
        "def stub(x: int) -> None:\n    ...\n\ndef real(x: int) -> int:\n    return x + 1\n",
        "ellipsis.py",
    );
    assert!(
        !out.contains("RYTHON_ELLIPSIS"),
        "bare `...` statement must not leak into generated code: {}",
        out
    );
    assert!(out.contains("fn stub"), "stub must still be emitted: {}", out);
    assert!(
        out.contains("fn real"),
        "real must still be emitted: {}",
        out
    );

    let module = parse("y = ...", "ellipsis_val.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let err = module
        .to_rust(
            CodeGenContext::Module("ellipsis_val".to_string()),
            PythonOptions::default(),
            symbols,
        )
        .expect_err("Ellipsis as a value must be a loud error");
    assert!(
        err.to_string().contains("Ellipsis"),
        "error must mention Ellipsis: {}",
        err
    );
}

#[test]
fn future_import_is_a_noop() {
    // `from __future__ import annotations` is a compiler directive; it must
    // not lower to a `use crate::__future__::...` (an unresolved import in
    // the generated crate — pip/__init__.py hit exactly that).
    let out = compile(
        "from __future__ import annotations\n\ndef f(x: int) -> int:\n    return x\n",
        "future.py",
    );
    assert!(
        !out.contains("__future__"),
        "from __future__ must lower to nothing: {}",
        out
    );
    assert!(out.contains("fn f"), "function must still be emitted: {}", out);
}

#[test]
fn metaclass_abcmata_is_a_lossy_noop() {
    // `metaclass=abc.ABCMeta` only enforces abstract-method instantiation
    // at runtime; lowering the class as a plain class keeps data+methods
    // (pip's BuildEnvironment). ANY metaclass keyword is a lossy no-op now:
    // the class lowers as a plain struct and the -W channel reports the
    // dropped metaclass machinery.
    let src = concat!(
        "import abc\n",
        "\n",
        "class Shape(metaclass=abc.ABCMeta):\n",
        "    def area(self) -> float:\n",
        "        return 0.0\n",
    );
    let out = compile(src, "meta.py");
    assert!(
        out.contains("struct Shape"),
        "metaclass class must still emit its struct: {}",
        out
    );

    let (out, warnings) = compile_with_warnings(
        "class Bad(metaclass=SomeMeta):\n    pass\n",
        "meta_bad.py",
    );
    assert!(
        out.contains("struct Bad"),
        "a non-ABCMeta metaclass name must not block conversion: {}",
        out
    );
    assert!(
        warnings.iter().any(|w| w.contains("metaclass") && w.contains("dropped")),
        "the dropped-metaclass divergence must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn isinstance_accepts_a_tuple_of_types() {
    // `isinstance(x, (bytearray, bytes))` is the "accept either" idiom
    // (charset_normalizer's from_bytes). Both lower to Vec<u8>, so the
    // check against a bytes|bytearray argument is statically true.
    let out = compile(
        "def check(sequences: bytes | bytearray) -> bool:\n\
         \x20   if not isinstance(sequences, (bytearray, bytes)):\n\
         \x20       return False\n\
         \x20   return True\n",
        "istuple.py",
    );
    assert!(
        out.contains("true"),
        "isinstance against a bytes tuple must decide true: {}",
        out
    );
    // A tuple containing a type the argument can't be decides false.
    let out = compile(
        "def check(x: int) -> bool:\n\
         \x20   return isinstance(x, (str, float))\n",
        "istuple2.py",
    );
    assert!(
        out.contains("false"),
        "isinstance against a non-matching tuple must decide false: {}",
        out
    );
}

#[test]
fn empty_list_into_optional_name_renders_typed_empty() {
    // `cp_isolation: list[str] | None = None`, then `cp_isolation = []` on
    // the None path (charset_normalizer's from_bytes): the empty literal
    // must render as the INNER typed container (Vec::<String>::new()) and
    // the optional-store wrap adds the Some — the old code had no
    // Option(inner) arm and failed with "no inferable element type".
    let out = compile(
        "def f(cp_isolation: list[str] | None = None) -> list[str]:\n\
         \x20   if cp_isolation is not None:\n\
         \x20       cp_isolation = cp_isolation\n\
         \x20   else:\n\
         \x20       cp_isolation = []\n\
         \x20   return cp_isolation\n",
        "optempty.py",
    );
    assert!(
        out.contains("Vec :: < String > :: new ()") || out.contains("Vec::<String>::new()"),
        "empty literal into an Optional name must render the typed inner container: {}",
        out
    );
}

#[test]
fn dataclass_synthesizes_init_from_annotated_fields() {
    // @dataclass classes get a synthesized __init__: each annotated field
    // becomes a parameter, the body stores self.field = field, and a
    // defaulted field becomes a defaulted parameter. The dataclasses
    // import is a no-op (the decorator is consumed at conversion time).
    let out = compile(
        "from dataclasses import dataclass\n\
         \n\
         @dataclass(frozen=True, slots=True)\n\
         class Point:\n\
         \x20   x: float\n\
         \x20   y: float\n\
         \x20   label: str = \"origin\"\n",
        "dc.py",
    );
    assert!(
        !out.contains("dataclasses"),
        "the dataclasses import must lower to nothing: {}",
        out
    );
    assert!(
        out.contains("pub fn new"),
        "the constructor must be synthesized: {}",
        out
    );
    assert!(
        out.contains("self.x = x;") || out.contains("self . x = x ;"),
        "__init__ must store each field: {}",
        out
    );
    assert!(
        out.contains("self.label = label") || out.contains("self . label = label"),
        "defaulted field must be stored too: {}",
        out
    );
    // Constructing the dataclass works (pip's Scheme(platlib=...) pattern).
    let out = compile(
        "from dataclasses import dataclass\n\
         \n\
         @dataclass\n\
         class Scheme:\n\
         \x20   platlib: str\n\
         \x20   scripts: str\n\
         \n\
         def get() -> Scheme:\n\
         \x20   return Scheme(platlib=\"a\", scripts=\"b\")\n",
        "dc2.py",
    );
    assert!(
        out.contains("Scheme :: new"),
        "dataclass construction must lower through the synthesized new: {}",
        out
    );
    // A dataclass with no annotated fields is a loud error, not silent.
    let module = parse("@dataclass\nclass Empty:\n    pass\n", "dc3.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let err = module
        .to_rust(
            CodeGenContext::Module("dc3".to_string()),
            PythonOptions::default(),
            symbols,
        )
        .expect_err("an empty @dataclass must be a loud error");
    assert!(
        err.to_string().contains("no annotated fields"),
        "error must mention the missing fields: {}",
        err
    );
}

#[test]
fn is_not_none_narrows_option_names() {
    // `if x is not None:` narrows x to its inner type in the body (reads
    // unwrap, comprehension/iteration sees the inner element), and when
    // both branches leave x non-None the name stays narrowed for the rest
    // of the function (charset_normalizer's cp_isolation pattern).
    let out = compile(
        "def f(cp_isolation: list[str] | None = None) -> list[str]:\n\
         \x20   if cp_isolation is not None:\n\
         \x20       cp_isolation = [x + \"!\" for x in cp_isolation]\n\
         \x20   else:\n\
         \x20       cp_isolation = []\n\
         \x20   return cp_isolation\n",
        "narrow.py",
    );
    // Reads inside the narrowed body unwrap the Option.
    assert!(
        out.contains("clone () . unwrap ()") || out.contains("clone().unwrap()"),
        "narrowed reads must unwrap: {}",
        out
    );
    // The comprehension iterates the inner element type (String), not the
    // Option's single inner Vec as one element.
    assert!(
        out.contains("for x in (cp_isolation) . clone () . unwrap ()")
            || out.contains("for x in (cp_isolation).clone().unwrap()"),
        "comprehension must iterate the unwrapped list: {}",
        out
    );
    // The post-if return unwraps too (both branches left x non-None).
    assert!(
        out.contains("return Ok ((cp_isolation) . clone () . unwrap ())")
            || out.contains("return Ok((cp_isolation).clone().unwrap())"),
        "post-if reads must unwrap: {}",
        out
    );
    // The STORE target must NOT unwrap (it wraps in Some below).
    assert!(
        !out.contains("(cp_isolation) . clone () . unwrap () = Some")
            && !out.contains("(cp_isolation).clone().unwrap() = Some"),
        "store targets must not unwrap: {}",
        out
    );
}

#[test]
fn imported_function_keyword_args_resolve_cross_module() {
    // Issue #123: `from helpers import greet` + `greet(name, excited=True)`
    // needs the callee's signature, which lives in the DEFINING module —
    // the same cross-module lookup classes use. Without it, keyword
    // arguments on imported functions were a loud error.
    let helpers = parse(
        "def greet(name: str, *, excited: bool = False) -> str:\n\
         \x20   return name + (\"!\" if excited else \"\")\n",
        "helpers.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        vec!["helpers".to_string()],
        std::rc::Rc::new(helpers),
    );
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let caller = parse(
        "from helpers import greet\n\
         \ndef use() -> str:\n\
         \x20   return greet(\"hi\", excited=True)\n",
        "caller.py",
    )
    .unwrap();
    let symbols = caller.clone().find_symbols(SymbolTableScopes::new());
    let out = caller
        .to_rust(
            CodeGenContext::Module("caller".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    // The keyword maps to its parameter position (the bool, not a string).
    assert!(
        out.contains("greet (__rython_arg_0 , __rython_arg_1)"),
        "keyword must map to the parameter position: {}",
        out
    );
    assert!(
        out.contains("__rython_arg_1 = true") || out.contains("__rython_arg_1 = true ;"),
        "excited=True must bind true: {}",
        out
    );
}

#[test]
fn decorator_factory_expression_synthesizes_cached_function() {
    // Issue #127: `alias = lru_cache(maxsize=N)(fn)` — the decorator
    // FACTORY applied as an expression (charset_normalizer's
    // `cached_mess_ratio = lru_cache(maxsize=None)(mess_ratio)`). The
    // assignment emits a synthesized cached function named `alias` (the
    // same @lru_cache machinery a decorated definition gets), and call
    // sites route through the cache.
    let out = compile(
        "from functools import lru_cache\n\
         \n\
         def helper(x: int) -> int:\n\
         \x20   return x * 2\n\
         \n\
         cached_helper = lru_cache(maxsize=None)(helper)\n\
         \n\
         def use(n: int) -> int:\n\
         \x20   return cached_helper(n) + cached_helper(n)\n",
        "factory.py",
    );
    assert!(
        out.contains("fn cached_helper"),
        "the factory must synthesize a cached function: {}",
        out
    );
    assert!(
        out.contains("__LRU_CACHE"),
        "the synthesized function must carry the cache static: {}",
        out
    );
    assert!(
        out.contains("helper(x)") || out.contains("helper (x)"),
        "the wrapper must call the wrapped function: {}",
        out
    );
    // The assignment must NOT survive as a store.
    assert!(
        !out.contains("cached_helper ="),
        "the factory assignment must be consumed, not stored: {}",
        out
    );
}

#[test]
fn unsupported_decorator_error_is_consistent() {
    // The systematic registry reports any unknown decorator with the same
    // "unsupported" message whether it decorates a function or a class.
    let err = compile_err("@my_decorator\ndef f() -> int:\n    return 1\n", "d1.py");
    assert!(
        err.contains("not supported yet"),
        "function decorator error must come from the registry: {}",
        err
    );
    let err = compile_err("@my_decorator\nclass C:\n    pass\n", "d2.py");
    assert!(
        err.contains("not supported yet"),
        "class decorator error must come from the registry: {}",
        err
    );
}

#[test]
fn lru_cache_float_keys_use_python_semantics() {
    // @lru_cache on a function with a float parameter: the cache key wraps
    // the float in PyFloatKey (Python semantics: -0.0 == 0.0, NaN never
    // hits), while the inner fn still takes the raw f64.
    let out = compile(
        "from functools import lru_cache\n\
         \n\
         @lru_cache(maxsize=None)\n\
         def f(x: float, s: str) -> float:\n\
         \x20   return x\n",
        "lrufloat.py",
    );
    assert!(
        out.contains("PyFloatKey"),
        "float cache keys must wrap in PyFloatKey: {}",
        out
    );
    assert!(
        out.contains("fn __lru_uncached") || out.contains("fn __lru_uncached "),
        "the uncached fn must be emitted: {}",
        out
    );
}

#[test]
fn str_bytes_union_narrows_via_isinstance() {
    // Issue #121: a `str | bytes` parameter lowers to StrOrBytes and
    // isinstance checks narrow each branch to the concrete type — the
    // pattern requests' to_native_string and idna's ulabel use.
    let out = compile(
        "def to_native(string: str | bytes) -> str:\n\
         \x20   if isinstance(string, str):\n\
         \x20       out = string\n\
         \x20   else:\n\
         \x20       out = string.decode(\"ascii\")\n\
         \x20   return out\n",
        "union.py",
    );
    assert!(
        out.contains("StrOrBytes"),
        "str | bytes must lower to StrOrBytes: {}",
        out
    );
    assert!(
        out.contains("is_str ()") || out.contains("is_str()"),
        "isinstance(str) must dispatch at runtime: {}",
        out
    );
    assert!(
        out.contains("as_str () . unwrap ()") || out.contains("as_str().unwrap()"),
        "the str branch must read as_str().unwrap(): {}",
        out
    );
    assert!(
        out.contains("as_bytes () . unwrap ()") || out.contains("as_bytes().unwrap()"),
        "the bytes branch must read as_bytes().unwrap(): {}",
        out
    );
    assert!(
        out.contains("decode_ascii") || out.contains("decode_by_name"),
        "bytes branch must decode through the codec: {}",
        out
    );
}

#[test]
fn bytes_like_methods_on_narrowed_bytes() {
    // idna's ulabel: after `isinstance(label, (bytes, bytearray))`, the
    // bytes branch uses lower/startswith/endswith and str(bytes,
    // encoding=...).
    let out = compile(
        "def ulabel(label: str | bytes | bytearray) -> str:\n\
         \x20   if isinstance(label, (bytes, bytearray)):\n\
         \x20       b = bytes(label)\n\
         \x20       b = b.lower()\n\
         \x20       if b.startswith(b\"xn--\"):\n\
         \x20           b = b[4:]\n\
         \x20       return str(b, encoding=\"ascii\")\n\
         \x20   return label\n",
        "ulabel.py",
    );
    assert!(
        out.contains("into_bytes_like"),
        "bytes(label) must lower through into_bytes_like: {}",
        out
    );
    assert!(
        out.contains("lower ()") || out.contains(".lower()"),
        "bytes .lower() must dispatch: {}",
        out
    );
    assert!(
        out.contains("decode_by_name"),
        "str(bytes, encoding=...) must decode: {}",
        out
    );
}

#[test]
fn exception_class_isinstance_matches_by_name() {
    // `isinstance(e, LookupError)` where e is a caught exception tests the
    // PyException's name string (charset_normalizer's codec fallback).
    let out = compile(
        "def f() -> bool:\n\
         \x20   try:\n\
         \x20       x = 1\n\
         \x20   except (UnicodeDecodeError, LookupError) as e:\n\
         \x20       return isinstance(e, LookupError)\n\
         \x20   return False\n",
        "isexc.py",
    );
    assert!(
        out.contains("matches"),
        "isinstance on a caught exception must lower to .matches: {}",
        out
    );
}

#[test]
fn typing_calls_are_noops_and_type_aliases_emit_pub_types() {
    // typing-module calls (TypeVar, Protocol, TypeAlias, cast, ...) exist
    // only for the type system; a call lowers to nothing. A module-level
    // `name = str` (requests' compat `builtin_str = str`) is a TYPE ALIAS:
    // it emits a pub type (so re-exports resolve) and isinstance resolution
    // treats the name as the builtin type.
    let out = compile(
        "from typing import TypeVar, Protocol, TypeAlias, cast\n\
         \n\
         _T_co = TypeVar(\"_T_co\", covariant=True)\n\
         \n\
         class SupportsRead(Protocol[_T_co]):\n\
         \x20   def read(self, length: int = 1) -> _T_co: ...\n\
         \n\
         HookType: TypeAlias = str\n",
        "typing.py",
    );
    assert!(
        !out.contains("TypeVar"),
        "TypeVar call must lower to nothing: {}",
        out
    );
    assert!(
        !out.contains("Protocol"),
        "Protocol base must lower to nothing: {}",
        out
    );
    // The Protocol class body has a method with `...` stub — fine.

    let out = compile(
        "builtin_str = str\n\
         bytes_alias = bytes\n",
        "aliases.py",
    );
    assert!(
        out.contains("pub type builtin_str = String") || out.contains("pub type builtin_str"),
        "str alias must emit a pub type: {}",
        out
    );
    assert!(
        out.contains("pub type bytes_alias") || out.contains("pub type bytes_alias = Vec < u8 >"),
        "bytes alias must emit a pub type: {}",
        out
    );
    // The alias assignment must not survive as a runtime store.
    assert!(
        !out.contains("builtin_str = str;"),
        "the alias must not be a runtime store: {}",
        out
    );
}

#[test]
fn type_checking_blocks_are_skipped() {
    // `if TYPE_CHECKING:` never runs at runtime — the block (imports,
    // type-only class definitions) must be skipped entirely (requests'
    // _types.py `_ValidatedRequest(PreparedRequest)`).
    let out = compile(
        "if TYPE_CHECKING:\n\
         \x20   from .models import PreparedRequest\n\
         \x20   class _ValidatedRequest(PreparedRequest):\n\
         \x20       url: str\n\
         \n\
         def f() -> int:\n\
         \x20   return 1\n",
        "tc.py",
    );
    assert!(
        !out.contains("_ValidatedRequest"),
        "TYPE_CHECKING block must be skipped: {}",
        out
    );
    assert!(
        out.contains("fn f"),
        "code after the block must still emit: {}",
        out
    );
}

#[test]
fn exception_class_lowers_to_a_marker_struct() {
    // Custom exceptions (`class IDNAError(UnicodeError)`, `class
    // RequestException(IOError)`) are string-tagged PyException values at
    // runtime; the class definition is a marker struct so `raise
    // IDNAError(...)` / `except IDNAError` keep working. idna's IDNAError
    // has a *args __init__ — the marker lowering must not trip the
    // variadic-parameter guard.
    let out = compile(
        "class IDNAError(UnicodeError):\n\
         \x20   code: str | None\n\
         \x20   def __init__(self, *args: object, code=None) -> None:\n\
         \x20       super().__init__(*args)\n\
         \x20       self.code = code\n\
         \n\
         class IDNABidiError(IDNAError):\n\
         \x20   pass\n",
        "exc.py",
    );
    assert!(
        out.contains("pub struct IDNAError ;")
            || out.contains("pub struct IDNAError;"),
        "exception class must lower to a marker struct: {}",
        out
    );
    assert!(
        out.contains("pub struct IDNABidiError")
            || out.contains("pub struct IDNABidiError ;"),
        "custom exception inheriting a custom exception must be a marker: {}",
        out
    );
    assert!(
        !out.contains("fn __init__"),
        "the marker must not carry __init__: {}",
        out
    );
}

#[test]
fn composition_chain_store_goes_through_mut_accessors() {
    // A store through a composition chain (`self.inner.x = v`) inside a
    // generic trait default must write through the MUTABLE accessor of
    // every hop: `self.inner_mut().x`. The old lowering rendered the
    // receiver in load flavor (`self.inner().x`), which clones the inner
    // struct — the store silently vanished.
    let src = concat!(
        "class Inner:\n",
        "    def __init__(self, x: int):\n",
        "        self.x = x\n",
        "\n",
        "class Outer(Inner):\n",
        "    def __init__(self):\n",
        "        self.inner: Inner = Inner(0)\n",
        "\n",
        "    def mutate(self):\n",
        "        self.inner.x = 5\n",
        "\n",
        "class OuterChild(Outer):\n",
        "    pass\n",
    );
    let out = compile(src, "compose_store.py");
    // The mutation widens the trait, so the default runs on &mut self.
    assert!(
        out.contains("fn mutate (& mut self ,"),
        "chain store must widen the trait: {}",
        out
    );
    assert!(
        out.contains("self . inner_mut () . x"),
        "store must go through self.inner_mut().x, not the cloning load: {}",
        out
    );
    assert!(
        !out.contains("self . inner () . x ="),
        "store must not land on a clone (self.inner().x = ...): {}",
        out
    );
}

#[test]
fn composition_chain_mutating_method_uses_mut_receivers() {
    // `self.inner.bump()` (user-defined mutating callee) and
    // `self.inner.nums.append(v)` (builtin mutating method) on composition
    // fields must render the WHOLE receiver chain in place flavor inside a
    // generic trait default. The one-hop-only rewrite used to leave deeper
    // chains on the cloning load accessors.
    let src = concat!(
        "class Inner:\n",
        "    def __init__(self, nums: list[int]):\n",
        "        self.nums = nums\n",
        "\n",
        "    def bump(self):\n",
        "        self.nums.append(2)\n",
        "\n",
        "class Outer(Inner):\n",
        "    def __init__(self):\n",
        "        self.inner: Inner = Inner([1])\n",
        "\n",
        "    def mutate(self):\n",
        "        self.inner.bump()\n",
        "        self.inner.nums.append(3)\n",
        "\n",
        "class OuterChild(Outer):\n",
        "    pass\n",
    );
    let out = compile(src, "compose_call.py");
    assert!(
        out.contains("(self . inner_mut ()) . bump ()"),
        "user-defined mutating callee must go through self.inner_mut(): {}",
        out
    );
    assert!(
        out.contains("(self . inner_mut () . nums) . push"),
        "builtin mutating method must go through self.inner_mut().nums: {}",
        out
    );
    assert!(
        !out.contains("self . inner () . nums ()"),
        "must not mutate through the cloning load accessors: {}",
        out
    );
}

#[test]
fn tuple_destructuring_stores_through_mut_accessors() {
    // `self.x, self.y = ...` in a generic trait default destructures INTO
    // the fields: each attribute target must render through its mutable
    // accessor, not through the cloning load form.
    let src = concat!(
        "class Base:\n",
        "    def __init__(self):\n",
        "        self.x = 0\n",
        "        self.y = 0\n",
        "\n",
        "    def reset(self):\n",
        "        self.x, self.y = 1, 2\n",
        "\n",
        "class Child(Base):\n",
        "    pass\n",
    );
    let out = compile(src, "tuple_target.py");
    assert!(
        out.contains("* self . x_mut () , * self . y_mut ()"),
        "tuple targets must destructure through the mut accessors: {}",
        out
    );
}

#[test]
fn single_element_tuple_target_keeps_the_trailing_comma() {
    // `x, = f()` parses as a one-element TUPLE target (`x,` — the trailing
    // comma is what makes it one). The per-element rendering must keep it:
    // `(x,) = ...` destructures against the one-element tuple value, while
    // `(x) = ...` would be a parenthesized place and fail to type-check.
    let src = concat!(
        "def pair() -> tuple[int]:\n",
        "    return (1,)\n",
        "\n",
        "def unpack() -> int:\n",
        "    x, = pair()\n",
        "    return x\n",
    );
    let out = compile(src, "single_tuple_target.py");
    assert!(
        out.contains("(x ,) = pair () ?"),
        "single-element tuple target must keep the trailing comma: {}",
        out
    );
    assert!(
        !out.contains("(x) = "),
        "must not emit a parenthesized place for a tuple target: {}",
        out
    );
}

#[test]
fn cross_module_mut_table_computed_once_and_cached() {
    // The first fallback call builds the merged table over every module of
    // the crate; the scan finds Dog's mutating override and widens
    // Animal.grow — the root of the trait Dog's method re-emits into.
    let (_, options) = cross_module_fixture();
    assert!(
        python_ast::module_widens_method_cached(&options, "Animal", "grow"),
        "Dog's mutating override must widen Animal.grow across modules"
    );
    // The table is now cached: repeated lookups are HashMap hits, not
    // re-scans of the module ASTs.
    assert!(
        matches!(
            &*options.cross_module_mut_self.borrow(),
            python_ast::CrossModuleMutSelf::Computed(_)
        ),
        "first fallback must leave the merged table cached"
    );
    assert!(python_ast::module_widens_method_cached(&options, "Animal", "grow"));
    // A method no definition mutates stays un-widened.
    assert!(!python_ast::module_widens_method_cached(&options, "Animal", "describe"));
}

#[test]
fn cross_module_class_info_computed_once_and_cached() {
    // Resolving an imported class's construction/fields must build the
    // defining module's symbol table ONCE per conversion, not per call
    // site: the first `module_class_def` builds the table over every module
    // of the crate and caches it; later lookups (every attribute access,
    // method call, construction, trait import) are HashMap hits.
    let (_, options) = cross_module_fixture();
    // Module path of the fixture: ["animals"].
    let (dog, symbols) = python_ast::module_class_def(&options, &["animals".to_string()], "Dog")
        .expect("Dog must resolve through the defining module");
    assert_eq!(dog.name, "Dog");
    // The defining module's symbol table comes back with it, so the base
    // chain resolves inside the module that DECLARED Dog: `Animal` must be
    // resolvable there (the importer's scope does not name it).
    assert!(
        symbols.get("Animal").is_some(),
        "defining module's symbols must resolve Dog's base Animal"
    );
    // The table is now cached.
    assert!(
        matches!(
            &*options.cross_module_classes.borrow(),
            python_ast::CrossModuleClasses::Computed(_)
        ),
        "first cross-module class lookup must leave the cache populated"
    );
    // Repeated lookups are cache hits and stay consistent.
    let (dog2, _) = python_ast::module_class_def(&options, &["animals".to_string()], "Dog")
        .expect("cached Dog lookup");
    assert_eq!(dog2.name, "Dog");
    // Traits: Animal is the hierarchy root, so Dog carries DogTrait (own)
    // plus AnimalTrait (its methods re-emit there), nearest first.
    let traits =
        python_ast::module_class_traits(&options, &["animals".to_string()]);
    assert_eq!(
        traits.get("Dog").map(Vec::as_slice),
        Some(&["DogTrait".to_string(), "AnimalTrait".to_string()][..]),
        "Dog must import DogTrait (own) + AnimalTrait (methods re-emit there)"
    );
    // A class that does not exist resolves to None without poisoning the cache.
    assert!(python_ast::module_class_def(&options, &["animals".to_string()], "Nope").is_none());
}

#[test]
fn cross_module_mut_reentrant_guard_does_not_recompute() {
    // During the one-time scan, the scan's own per-method analysis consults
    // method_needs_mut_self again. The Computing sentinel must answer false
    // (the direct chain walk in method_needs_mut_self resolves the call)
    // instead of recursing into another scan.
    let (_, options) = cross_module_fixture();
    *options.cross_module_mut_self.borrow_mut() = python_ast::CrossModuleMutSelf::Computing;
    assert!(
        !python_ast::module_widens_method_cached(&options, "Animal", "grow"),
        "re-entrant fallback during a scan must not recompute"
    );
    assert!(
        matches!(
            &*options.cross_module_mut_self.borrow(),
            python_ast::CrossModuleMutSelf::Computing
        ),
        "the in-progress scan state must be left untouched"
    );
}
fn compile(src: &str, name: &str) -> String {
    let module = parse(src, name).unwrap_or_else(|e| panic!("parse failed for {:?}: {}", src, e));
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    module
        .to_rust(
            CodeGenContext::Module(name.replace(".py", "")),
            PythonOptions::default(),
            symbols,
        )
        .unwrap_or_else(|e| panic!("codegen failed for {:?}: {}", src, e))
        .to_string()
}

/// Compile and return (generated Rust, the `-W` definition warnings pushed
/// during inference/codegen). `definition_warnings` is an Rc shared across
/// option clones, so reading the original options after codegen collects
/// everything the transpiler would report through the -W channel.
fn compile_with_warnings(src: &str, name: &str) -> (String, Vec<String>) {
    let module = parse(src, name).unwrap_or_else(|e| panic!("parse failed for {:?}: {}", src, e));
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let options = PythonOptions::default();
    let out = module
        .to_rust(
            CodeGenContext::Module(name.replace(".py", "")),
            options.clone(),
            symbols,
        )
        .unwrap_or_else(|e| panic!("codegen failed for {:?}: {}", src, e))
        .to_string();
    let warnings = options.definition_warnings.borrow().clone();
    (out, warnings)
}

#[test]
fn power_uses_py_pow() {
    let out = compile("y = 2 ** 3", "pow.py");
    assert!(out.contains("py_pow"), "generated: {}", out);
    assert!(!out.contains(". pow"), "generated: {}", out);
}

#[test]
fn power_aug_assign_uses_py_pow() {
    let out = compile("x = 2\nx **= 3", "pow2.py");
    assert!(out.contains("py_pow"), "generated: {}", out);
}

#[test]
fn list_literals_keep_element_types() {
    let out = compile("nums = [1, 2, 3]", "list.py");
    assert!(out.contains("vec ! [1 , 2 , 3]"), "generated: {}", out);
    assert!(!out.contains("to_string"), "generated: {}", out);
}

#[test]
fn len_is_i64_everywhere() {
    // len() lowers to `len(&x) as i64` (Issue #100 follow-up): the runtime
    // length is usize, but Python ints are i64 everywhere else, and the
    // type inference must agree — an empty list pinned from
    // `xs.append(len(s))` must be Vec<i64>, and index/range positions must
    // NOT get a redundant (and wrong) `.try_into().unwrap()` on a value
    // that is already i64.
    let out = compile(
        "def forward(x: list[float]) -> list[float]:\n\
         \x20   result: list[float] = []\n\
         \x20   for j in range(len(x)):\n\
         \x20       result.append(x[j])\n\
         \x20   return result\n",
        "issue100.py",
    );
    assert!(
        out.contains("len (& (x)) as i64"),
        "len() must cast to i64: {}",
        out
    );
    assert!(
        !out.contains("try_into ()"),
        "i64 len() must not be re-coerced: {}",
        out
    );
    assert!(
        out.contains("Vec :: < f64 > :: new ()"),
        "empty list must be pinned from the append: {}",
        out
    );
}

#[test]
fn len_append_pins_empty_list_to_i64() {
    // Issue: len() infers Usize while codegen emits i64, so an empty list
    // whose only pinning use is `xs.append(len(s))` was declared
    // `Vec::<usize>::new()` and then rejected when the appended i64 did
    // not match. The pinning must agree with the emission.
    let out = compile(
        "def sizes(ss: list[str]) -> list[int]:\n\
         \x20   xs = []\n\
         \x20   for s in ss:\n\
         \x20       xs.append(len(s))\n\
         \x20   return xs\n",
        "lenpin.py",
    );
    assert!(
        out.contains("Vec :: < i64 > :: new ()"),
        "empty list must pin to i64 (matching len() as i64): {}",
        out
    );
    assert!(
        !out.contains("usize"),
        "no usize may appear in the generated code: {}",
        out
    );
}

#[test]
fn field_named_base_in_hierarchy_converts_with_a_collision_warning() {
    // A class that inherits and also stores an attribute named `base`
    // used to be a clean conversion-time error (the embedded-base accessor
    // and the field accessor would both be `fn base`). Now the collision is
    // a documented divergence: the field is a pub struct field, the
    // embedded-base accessor stays a trait item, the field's OWN trait
    // accessors are skipped, and the -W channel reports it.
    let (out, warnings) = compile_with_warnings(
        "class Animal:\n\
         \x20   def __init__(self):\n\
         \x20       self.name = 'x'\n\
         class Dog(Animal):\n\
         \x20   def __init__(self):\n\
         \x20       self.base = 1\n",
        "basefield.py",
    );
    assert!(
        out.contains("pub struct Dog") && out.contains("pub base : i64")
            && out.contains("pub __rython_base : Animal"),
        "the base field and the embedded base must coexist: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("field `base` of `Dog`") && w.contains("collides")),
        "the collision must be reported through -W: {:?}",
        warnings
    );
    // `base_mut` collides the same way; `base` on a BASE-LESS class is fine
    // (no embedded-base accessor is emitted).
    let (out, warnings) = compile_with_warnings(
        "class Animal:\n\
         \x20   def __init__(self):\n\
         \x20       self.name = 'x'\n\
         class Dog(Animal):\n\
         \x20   def __init__(self):\n\
         \x20       self.base_mut = 1\n",
        "basemutfield.py",
    );
    assert!(
        out.contains("pub base_mut : i64") && out.contains("pub __rython_base : Animal"),
        "the base_mut field and the embedded base must coexist: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("`base_mut`") && w.contains("collides")),
        "the base_mut collision must be reported through -W: {:?}",
        warnings
    );
    let out = compile(
        "class Base:\n\
         \x20   def __init__(self):\n\
         \x20       self.base = 1\n",
        "basefieldok.py",
    );
    assert!(
        out.contains("pub base : i64") && !out.contains("Trait"),
        "base field on a base-less class must compile: {}",
        out
    );
}

#[test]
fn mixed_numeric_list_unifies_to_float() {
    // [1, 2.0] is Vec<f64> in Rust; the int literal must be coerced.
    let out = compile("xs = [1, 2.0, 3]", "mixed.py");
    assert!(
        out.contains("as f64"),
        "int element must coerce to f64: {}",
        out
    );
}

#[test]
fn incompatible_list_elements_are_a_loud_error() {
    // Issue #130: primitive mixes ([1, "a"]) BOX to Vec<PyValue>; a mix
    // involving an UNBOXABLE element (a dict literal has no PyValue
    // variant) stays a loud conversion error.
    let err = compile_err("[1, {'a': 2}]", "badlist.py");
    assert!(
        err.contains("mixes incompatible element types"),
        "expected loud conversion error, got: {}",
        err
    );
}

#[test]
fn reused_name_argument_is_cloned() {
    // Issue #102: parameters lower to owned values, so a name passed to a
    // user function twice must be cloned on each call (Python shares by
    // reference; Rust moves).
    let out = compile(
        "def forward(x: list[float]) -> list[float]:\n\
         \x20   return x\n\n\
         def test() -> list[float]:\n\
         \x20   x: list[float] = [1.0, 2.0]\n\
         \x20   out1: list[float] = forward(x)\n\
         \x20   out2: list[float] = forward(x)\n\
         \x20   return out2\n",
        "issue102.py",
    );
    let clones = out.matches("clone ()").count();
    assert!(
        clones >= 2,
        "reused owned name must be cloned per call, got {} clones: {}",
        clones,
        out
    );
}

#[test]
fn single_use_name_is_not_cloned() {
    let out = compile(
        "def forward(x: list[float]) -> list[float]:\n\
         \x20   return x\n\n\
         def test() -> list[float]:\n\
         \x20   x: list[float] = [1.0, 2.0]\n\
         \x20   return forward(x)\n",
        "singleuse.py",
    );
    assert!(
        !out.contains("clone ()"),
        "single-use name must not be cloned: {}",
        out
    );
}

#[test]
fn unused_loop_index_lowers_to_underscore() {
    // Issue #101: an index that is never read in the body must not emit a
    // Rust binding rustc warns about.
    let out = compile(
        "def f() -> int:\n\
         \x20   total = 0\n\
         \x20   for i in range(3):\n\
         \x20       total += 1\n\
         \x20   return total\n",
        "issue101.py",
    );
    assert!(
        out.contains("for _ in range"),
        "unused index must lower to `_`: {}",
        out
    );
    // ...while a used index keeps its name.
    let out = compile(
        "def f(x: list[int]) -> int:\n\
         \x20   total = 0\n\
         \x20   for i in range(len(x)):\n\
         \x20       total += x[i]\n\
         \x20   return total\n",
        "issue101b.py",
    );
    assert!(
        out.contains("for i in range"),
        "used index keeps its name: {}",
        out
    );
}

#[test]
fn empty_list_pinned_by_append_is_typed() {
    // Issue #77 companion: `xs = []` + `xs.append(v)` pins the element
    // type from the append, so the empty literal is rendered typed.
    let out = compile(
        "def f() -> list[int]:\n\
         \x20   xs = []\n\
         \x20   xs.append(1)\n\
         \x20   return xs\n",
        "emptyappend.py",
    );
    assert!(
        out.contains("Vec :: < i64 > :: new ()"),
        "empty list must be pinned from append: {}",
        out
    );
}

#[test]
fn empty_list_pinned_by_extend_is_typed() {
    // `xs = []` + `xs.extend(ys)` pins the element type from the extended
    // container — including when append and extend both contribute.
    let out = compile(
        "def f(rows: list[list[int]]) -> list[int]:\n\
         \x20   out = []\n\
         \x20   out.append(1)\n\
         \x20   out.extend(rows[0])\n\
         \x20   return out\n",
        "emptyextend.py",
    );
    assert!(
        out.contains("Vec :: < i64 > :: new ()"),
        "empty list must be pinned from extend: {}",
        out
    );
}

#[test]
fn empty_dict_pinned_by_subscript_store_is_typed() {
    let out = compile(
        "def f() -> dict[str, int]:\n\
         \x20   d = {}\n\
         \x20   d[\"k\"] = 1\n\
         \x20   return d\n",
        "emptydict.py",
    );
    assert!(
        out.contains("PyDict :: < String , i64 > :: from"),
        "empty dict must be pinned from subscript store: {}",
        out
    );
}

#[test]
fn unpinned_empty_container_lowers_as_a_boxed_pyvalue_vec() {
    // Issue #77: `x = []` with no use that could pin the element type used
    // to be a conversion-time error. It now converts as the boxed-container
    // divergence: `Vec<PyValue>`, with the -W channel reporting the lossy
    // type.
    let (out, warnings) = compile_with_warnings("def f():\n    x = []\n    return x\n", "issue77.py");
    assert!(
        out.contains("Vec :: < stdpython :: PyValue > :: new ()"),
        "the empty list must lower as Vec<PyValue>: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("no inferable element type") && w.contains("Vec<PyValue>")),
        "the boxed-container divergence must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn bare_container_annotation_lowers_as_the_bare_name() {
    // Issue #76 companion: `def f(xs: list)` used to be a loud
    // conversion-time error directing the user to subscripted annotations.
    // It now converts, emitting the annotation name as-is (the
    // boxed-container divergence: the parameter's inferred type is the
    // boxed PyValue, so the body treats it as an untyped value).
    let out = compile(
        "def f(xs: list) -> int:\n    return len(xs)\n",
        "bareann.py",
    );
    assert!(
        out.contains("pub fn f (xs : list)"),
        "a bare `list` parameter must convert: {}",
        out
    );
    let out = compile(
        "def f(xs: dict) -> int:\n    return len(xs)\n",
        "bareann2.py",
    );
    assert!(
        out.contains("pub fn f (xs : dict)"),
        "a bare `dict` parameter must convert: {}",
        out
    );
    // ... but a bare RETURN annotation still fails loudly with the
    // subscripting hint (the return type would be invalid Rust).
    let err = compile_err("def f() -> list:\n    return [1]\n", "bareret.py");
    assert!(
        err.contains("return annotation") && err.contains("no element/key type"),
        "expected loud bare return-annotation error, got: {}",
        err
    );
    // ... and subscripted generics, including set[T], still work.
    let out = compile(
        "def f(a: list[int], b: dict[str, int], c: set[int]):\n    pass\n",
        "generics2.py",
    );
    assert!(
        out.contains("c : std :: collections :: HashSet < i64 >"),
        "generated: {}",
        out
    );
}

#[test]
fn rust_keywords_are_escaped() {
    let out = compile("type = 5", "kw.py");
    assert!(out.contains("r#type"), "generated: {}", out);

    let out = compile("def loop():\n    pass\n", "kw2.py");
    assert!(out.contains("fn r#loop"), "generated: {}", out);
}

#[test]
fn assignments_hoist_declaration_and_store() {
    // Assigned names are hoisted to a declaration and each assignment is a
    // plain store (a `let mut` per assignment would shadow inside nested
    // blocks instead of assigning). A single store needs no `mut`.
    // (A literal here would become a module constant static instead, so
    // use a computed value.)
    let out = compile("x = 1 + 1", "mut.py");
    assert!(out.contains("let x"), "generated: {}", out);
    assert!(
        !out.contains("let mut x"),
        "single store needs no mut: {}",
        out
    );
}

#[test]
fn mut_is_inferred_only_where_needed() {
    // Branch-exclusive initialization: no path assigns twice, so no mut —
    // rustc would warn unused_mut otherwise.
    let src =
        "def f(c) -> int:\n    if c:\n        x = 1\n    else:\n        x = 2\n    return x\n";
    let out = compile(src, "branches.py");
    assert!(out.contains("let x ;"), "generated: {}", out);
    assert!(!out.contains("let mut x"), "generated: {}", out);

    // A store inside a loop may execute repeatedly: mut required.
    let src = "def g(items: list[int]) -> int:\n    total = 0\n    for i in items:\n        total = total + i\n    return total\n";
    let out = compile(src, "loopmut.py");
    assert!(out.contains("let mut total"), "generated: {}", out);

    // A mutating method call requires a mutable binding.
    let out = compile(
        "def h():\n    items = []\n    items.append(1)\n",
        "append.py",
    );
    assert!(out.contains("let mut items"), "generated: {}", out);

    // A parameter that is only read is not rebound.
    let out = compile("def k(n: int) -> int:\n    return n\n", "readonly.py");
    assert!(!out.contains("let mut n"), "generated: {}", out);
}

#[test]
fn nested_block_assignment_stores_into_the_outer_variable() {
    // `x = 2` inside the if must update the function-scoped x, not create a
    // shadowing binding that dies at the end of the block.
    let src = "def pick(c) -> int:\n    x = 1\n    if c:\n        x = 2\n    return x\n";
    let out = compile(src, "scope.py");
    assert_eq!(
        out.matches("let mut x").count(),
        1,
        "one declaration, plain stores elsewhere: {}",
        out
    );
    assert!(
        out.contains("if (c) . is_truthy () { x = 2"),
        "generated: {}",
        out
    );
}

#[test]
fn assigned_parameters_are_rebound_mutably() {
    // Rust parameters are immutable; a parameter the body assigns to is
    // rebound as a mutable local first.
    let out = compile(
        "def f(n: int) -> int:\n    n = n + 1\n    return n\n",
        "param.py",
    );
    assert!(out.contains("let mut n = n"), "generated: {}", out);
}

#[test]
fn chained_module_string_constants_promote_both_names() {
    // urllib3/_version.py: `__version__ = version = '2.7.0'` — a chained
    // module-level constant. Both names become `pub static` (the importing
    // module's `from ._version import __version__` needs the item).
    let out = compile(
        "__version__ = version = '2.7.0'\n",
        "version.py",
    );
    assert!(
        out.contains("pub static __version__ : & 'static str = \"2.7.0\""),
        "generated: {}",
        out
    );
    assert!(
        out.contains("pub static version : & 'static str = \"2.7.0\""),
        "generated: {}",
        out
    );
}

#[test]
fn chained_assignment_assigns_each_target() {
    // A module-level chained constant (`a = b = 1`) promotes BOTH names to
    // `pub static` (each is a single-store module value — urllib3's
    // `__version__ = version = '2.7.0'`). Inside a function, the chain
    // keeps the ordinary lowering.
    let out = compile("a = b = 1", "chain.py");
    assert!(out.contains("pub static a : i64 = 1"), "generated: {}", out);
    assert!(out.contains("pub static b : i64 = 1"), "generated: {}", out);

    let out = compile(
        "def f(n: int) -> int:\n    a = b = n\n    return a + b\n",
        "chain_fn.py",
    );
    assert!(out.contains("__rython_chain"), "generated: {}", out);
    assert!(out.contains("a = __rython_chain"), "generated: {}", out);
    assert!(out.contains("b = __rython_chain"), "generated: {}", out);
}

#[test]
fn attribute_assignment_is_not_a_let() {
    // The param is annotated so this stays a test of ASSIGNMENT lowering
    // (unannotated params with attribute stores are an M2 inference gap).
    let out = compile(
        "class Point:\n    def __init__(self) -> None:\n        self.field = 0\n\ndef f(obj: Point) -> None:\n    obj.field = 1\n",
        "attr.py",
    );
    assert!(!out.contains("let obj . field"), "generated: {}", out);
    assert!(!out.contains("let mut obj . field"), "generated: {}", out);
}

#[test]
fn for_else_tracks_break() {
    let src = "for x in items:\n    break\nelse:\n    done()\n";
    let out = compile(src, "forelse.py");
    assert!(out.contains("__rython_broke = true"), "generated: {}", out);
    assert!(out.contains("if ! __rython_broke"), "generated: {}", out);
}

#[test]
fn plain_for_has_no_break_flag() {
    let out = compile("for x in items:\n    f(x)\n", "for.py");
    assert!(!out.contains("__rython_broke"), "generated: {}", out);
}

#[test]
fn loop_index_read_in_yield_body_is_not_unused() {    // `for x in chunks[1:-1]: yield x + b"\n"` (urllib3's response
    // __iter__): the index is used ONLY in the yield expression, which the
    // unused-index walker must still see — otherwise `x` lowers to `_`
    // while the body references it (E0425).
    let out = compile(
        "def gen(chunks: list[bytes]) -> None:\n    for x in chunks[1:-1]:\n        yield x + b\"\\n\"\n",
        "genyield.py",
    );
    assert!(
        out.contains("for x in"),
        "index used in yield must bind, not lower to _: {}",
        out
    );
    assert!(
        !out.contains("for _ in"),
        "index used in yield must bind, not lower to _: {}",
        out
    );
    assert!(
        out.contains("(x) . py_add"),
        "yield body must reference x: {}",
        out
    );
}

#[test]
fn while_else_tracks_break() {
    let src = "while cond:\n    break\nelse:\n    done()\n";
    let out = compile(src, "whileelse.py");
    assert!(out.contains("__rython_broke = true"), "generated: {}", out);
    assert!(out.contains("if ! __rython_broke"), "generated: {}", out);
}

#[test]
fn nested_loop_break_stays_plain() {
    // The inner loop's break belongs to the inner loop, so the outer
    // for/else needs no flag at all: its else runs unconditionally, and the
    // break stays plain.
    let src = "for x in items:\n    for y in inner:\n        break\nelse:\n    done()\n";
    let out = compile(src, "nested.py");
    assert!(!out.contains("__rython_broke"), "generated: {}", out);
    assert!(out.contains("done ()"), "generated: {}", out);
}

#[test]
fn loop_else_without_break_has_no_flag() {
    // No break in the body: declaring `let mut __rython_broke` would trip
    // deny-warnings builds with unused_mut, so the else runs unconditionally.
    let src = "for x in items:\n    f(x)\nelse:\n    done()\n";
    let out = compile(src, "forelse2.py");
    assert!(!out.contains("__rython_broke"), "generated: {}", out);
    assert!(out.contains("done ()"), "generated: {}", out);
}

#[test]
fn loop_else_break_inside_if_still_tracked() {
    // A break nested in an if still belongs to this loop.
    let src = "for x in items:\n    if x:\n        break\nelse:\n    done()\n";
    let out = compile(src, "forelse3.py");
    assert!(out.contains("__rython_broke = true"), "generated: {}", out);
    assert!(out.contains("if ! __rython_broke"), "generated: {}", out);
}

#[test]
fn with_binds_context_manager() {
    let src = "with open(name) as fh:\n    read(fh)\n";
    let out = compile(src, "with.py");
    assert!(out.contains("let mut fh"), "generated: {}", out);
    assert!(out.contains("open"), "generated: {}", out);
}

#[test]
fn with_without_target_still_evaluates() {
    let src = "with lock():\n    body()\n";
    let out = compile(src, "with2.py");
    assert!(out.contains("let _ = lock ()"), "generated: {}", out);
}

#[test]
fn comprehension_binds_target() {
    let out = compile("doubled = [x * 2 for x in items]", "comp.py");
    assert!(out.contains("for x in"), "generated: {}", out);
    assert!(!out.contains("_item"), "generated: {}", out);
    assert!(out.contains("push"), "generated: {}", out);
}

#[test]
fn comprehension_condition_uses_continue() {
    let out = compile("evens = [x for x in items if x % 2 == 0]", "comp2.py");
    assert!(out.contains("continue"), "generated: {}", out);
}

#[test]
fn multi_generator_comprehension_nests_loops() {
    let out = compile("pairs = [x + y for x in a for y in b]", "comp3.py");
    let for_count = out.matches("for ").count();
    assert!(for_count >= 2, "expected nested loops, generated: {}", out);
    assert!(!out.contains("vec ! []"), "generated: {}", out);
}

#[test]
fn dict_comprehension_inserts_pairs() {
    let out = compile("m = {k: v for k in keys}", "comp4.py");
    assert!(out.contains("insert"), "generated: {}", out);
    assert!(out.contains("PyDict"), "generated: {}", out);
}

#[test]
fn fstring_builds_single_format() {
    let out = compile("s = f\"Hello {name}\"", "fstr.py");
    assert!(out.contains("\"Hello {}\""), "generated: {}", out);
    // No string concatenation with `+`, which didn't even compile.
    assert!(!out.contains("\" + "), "generated: {}", out);
}

#[test]
fn fstring_maps_precision_spec() {
    let out = compile("s = f\"{pi:.2f}\"", "fstr2.py");
    assert!(out.contains("{:.2}"), "generated: {}", out);
}

#[test]
fn fstring_repr_conversion_uses_pythons_repr() {
    // Python's !r is repr(), not Rust's Debug: repr("ab") is 'ab' with
    // SINGLE quotes, where {:?} would render "ab".
    let out = compile("s = f\"{val!r}\"", "fstr3.py");
    assert!(out.contains("repr (& (val))"), "generated: {}", out);
    assert!(!out.contains("{:?}"), "generated: {}", out);
}

#[test]
fn statements_in_blocks_are_separated() {
    let src = "if cond:\n    first()\n    second()\n";
    let out = compile(src, "sep.py");
    let first = out.find("first ()").expect("first call present");
    let second = out.find("second ()").expect("second call present");
    let between = &out[first..second];
    assert!(between.contains(';'), "no separator between calls: {}", out);
}

#[test]
fn async_calls_do_not_guess_await() {
    let src = "async def f(x):\n    return abs(x)\n";
    let out = compile(src, "await.py");
    assert!(!out.contains(". await"), "generated: {}", out);
}

#[test]
fn explicit_await_still_awaits() {
    let src = "async def f(x: int) -> int:\n    return await g(x)\n";
    let out = compile(src, "await2.py");
    assert!(out.contains(". await"), "generated: {}", out);
}

#[test]
fn from_import_brings_name_into_scope() {
    let out = compile("from os import path", "imp.py");
    assert!(
        out.contains("use stdpython :: os :: path ;"),
        "generated: {}",
        out
    );
}

#[test]
fn from_import_with_alias() {
    let out = compile("from os import path as p", "imp2.py");
    assert!(
        out.contains("use stdpython :: os :: path as p ;"),
        "generated: {}",
        out
    );
}

#[test]
fn lambda_parameters_are_bare_names() {
    let out = compile("f = lambda x: x", "lam.py");
    assert!(out.contains("| x |"), "generated: {}", out);
    assert!(!out.contains("impl Into"), "generated: {}", out);
}

#[test]
fn return_type_inferred_from_int_constant() {
    let out = compile("def f():\n    return 42\n", "ret.py");
    assert!(
        out.contains("-> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn return_type_inferred_from_fstring() {
    let out = compile("def f():\n    return f\"x={x}\"\n", "ret2.py");
    assert!(
        out.contains("-> Result < String , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn return_type_inferred_from_string_literal() {
    let out = compile("def f():\n    return \"hi\"\n", "ret3.py");
    assert!(
        out.contains("-> Result < & 'static str , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn mixed_returns_box_to_pyvalue() {
    // `return 1` / `return "s"` has no single concrete type: the returns
    // box to PyValue and the signature agrees (issue #133 — the previous
    // `Result<(), _>` signature made the generated body unbuildable).
    let out = compile(
        "def f(c: bool):\n    if c:\n        return 1\n    return \"s\"\n",
        "ret4.py",
    );
    assert!(
        out.contains("-> Result < stdpython :: PyValue , PyException >"),
        "generated: {}",
        out
    );
    assert!(out.contains("PyValue :: from"), "generated: {}", out);
}

#[test]
fn bare_return_gets_no_annotation() {
    let out = compile("def f():\n    return\n", "ret5.py");
    assert!(
        out.contains("-> Result < () , PyException >"),
        "generated: {}",
        out
    );
    assert!(out.contains("return Ok (())"), "generated: {}", out);
}

#[test]
fn return_type_inferred_through_local_variable() {
    let out = compile("def f():\n    n = 5\n    n -= 1\n    return n\n", "ret6.py");
    assert!(
        out.contains("-> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn partial_return_gets_no_annotation() {
    // The fall-through path implicitly returns None, so annotating -> i64
    // would make the generated fn fail to compile.
    let out = compile("def f(c):\n    if c:\n        return 1\n", "ret7.py");
    assert!(!out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn return_in_loop_only_gets_no_annotation() {
    let out = compile(
        "def f(items: list[int]):\n    for x in items:\n        return 1\n",
        "ret8.py",
    );
    assert!(!out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn exhaustive_if_else_returns_get_annotation() {
    let src = "def f(c):\n    if c:\n        return 1\n    else:\n        return 2\n";
    let out = compile(src, "ret9.py");
    assert!(
        out.contains("-> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn annotated_parameters_map_to_rust_types() {
    let out = compile(
        "def f(a: int, b: float, c: str, d: bool):\n    pass\n",
        "ann_params.py",
    );
    assert!(out.contains("a : i64"), "generated: {}", out);
    assert!(out.contains("b : f64"), "generated: {}", out);
    assert!(out.contains("c : String"), "generated: {}", out);
    assert!(out.contains("d : bool"), "generated: {}", out);
    assert!(!out.contains(": int"), "generated: {}", out);
}

#[test]
fn return_annotation_used_when_inference_fails() {
    let out = compile("def f(x: int) -> int:\n    return x + 1\n", "ann_ret.py");
    assert!(
        out.contains("-> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn string_repetition_uses_multiply_string() {
    let out = compile("s = \"!\" * 3", "strmul.py");
    assert!(out.contains("multiply_string"), "generated: {}", out);
    let out = compile("s = 3 * \"!\"", "strmul2.py");
    assert!(out.contains("multiply_string"), "generated: {}", out);
    // Numeric multiplication is untouched.
    let out = compile("n = 3 * 4", "nummul.py");
    assert!(!out.contains("multiply_string"), "generated: {}", out);
}

#[test]
fn stdlib_from_import_anchors_to_stdpython() {
    let out = compile("from os import path", "imp3.py");
    assert!(
        out.contains("use stdpython :: os :: path ;"),
        "generated: {}",
        out
    );
}

#[test]
fn sibling_from_import_anchors_to_crate() {
    let out = compile("from helpers import util", "imp4.py");
    assert!(
        out.contains("use crate :: helpers :: util ;"),
        "generated: {}",
        out
    );
}

#[test]
fn defaulted_annotated_parameter_maps_type() {
    // Defaulted parameters lower to plain required parameters with mapped
    // types (never the raw Python name, and no Option wrapper, which
    // type-checked against neither bodies nor call sites).
    let out = compile("def f(x: int = 0):\n    return x\n", "def_param.py");
    assert!(out.contains("x : i64"), "generated: {}", out);
    assert!(!out.contains("Option"), "generated: {}", out);
    assert!(!out.contains(": int"), "generated: {}", out);
}

#[test]
fn kwonly_annotated_parameter_maps_type() {
    let out = compile("def f(*, x: int):\n    pass\n", "kwonly.py");
    assert!(out.contains("x : i64"), "generated: {}", out);
    assert!(!out.contains(": int"), "generated: {}", out);
}

#[test]
fn annotation_ignored_when_body_can_fall_through() {
    // A return annotation must not be applied when a path can reach the end
    // of the function without returning (the implicit tail is `()`) — but
    // ignoring it is a lossy conversion that likely marks a source bug, so
    // the generated function must carry a warning note saying so.
    let out = compile(
        "def f(c) -> int:\n    if c:\n        return 1\n",
        "ann_partial.py",
    );
    assert!(!out.contains("-> i64"), "generated: {}", out);
    assert!(out.contains("deprecated"), "generated: {}", out);
    assert!(
        out.contains("return annotation was ignored")
            || out.contains("return annotation `-> int`")
            || out.contains("`-> int` return annotation"),
        "warning note should name the ignored annotation: {}",
        out
    );

    // A function that honors its annotation carries no warning.
    let out = compile("def g() -> int:\n    return 1\n", "ann_honored.py");
    assert!(!out.contains("deprecated"), "generated: {}", out);

    // `-> None` on a fall-through body is accurate, not lossy.
    let out = compile("def h() -> None:\n    print(1)\n", "ann_none.py");
    assert!(!out.contains("deprecated"), "generated: {}", out);
}

#[test]
fn try_except_lowers_to_result_handling() {
    let src = concat!(
        "def f(n):\n",
        "    try:\n",
        "        raise ValueError(\"bad\")\n",
        "    except ValueError as e:\n",
        "        print(e)\n",
        "    except (TypeError, KeyError):\n",
        "        print(\"other\")\n",
    );
    let out = compile(src, "try.py");
    // The body runs in a closure returning Result<(), PyException>.
    assert!(
        out.contains("Result < () , PyException >"),
        "generated: {}",
        out
    );
    // raise inside the try returns an Err the handlers can match.
    assert!(
        out.contains("return Err (PyException :: new (\"ValueError\""),
        "generated: {}",
        out
    );
    // Handlers are guard-matched arms, in order; the tuple form ORs.
    assert!(
        out.contains("if __rython_exc . matches (\"ValueError\")"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("matches (\"TypeError\") || __rython_exc . matches (\"KeyError\")"),
        "generated: {}",
        out
    );
    // `as e` binds the caught exception.
    assert!(
        out.contains("let mut e = __rython_exc . clone ()"),
        "generated: {}",
        out
    );
    // An unmatched exception re-raises as an Err out of the function.
    assert!(
        out.contains("Err (__rython_exc) => { return Err (__rython_exc) ; }"),
        "generated: {}",
        out
    );
}

#[test]
fn try_handler_bodies_only_run_on_matching_error() {
    // The old lowering ran every handler body unconditionally after the try
    // body; the handler statements must now live inside match arms.
    let src = concat!(
        "def f():\n",
        "    try:\n",
        "        work()\n",
        "    except Exception:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "tryarm.py");
    let arm_pos = out.find("Err (__rython_exc)").expect("handler arm");
    let cleanup_pos = out.find("cleanup ()").expect("handler body");
    assert!(
        cleanup_pos > arm_pos,
        "handler body must be inside the Err arm: {}",
        out
    );
}

#[test]
fn nested_raise_propagates_to_outer_try() {
    // A try inside a try: the inner unmatched arm returns Err out of the
    // *outer* closure instead of panicking.
    let src = concat!(
        "def f():\n",
        "    try:\n",
        "        try:\n",
        "            raise KeyError(\"k\")\n",
        "        except ValueError:\n",
        "            pass\n",
        "    except KeyError:\n",
        "        pass\n",
    );
    let out = compile(src, "nested_try.py");
    assert!(
        out.contains("Err (__rython_exc) => { return Err (__rython_exc) ; }"),
        "inner unmatched exception must propagate as Err: {}",
        out
    );
}

#[test]
fn finally_runs_before_reraise() {
    let src = concat!(
        "def f():\n",
        "    try:\n",
        "        work()\n",
        "    except ValueError:\n",
        "        pass\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "finally.py");
    // finally body appears both after the match (normal paths) and in the
    // unmatched-reraise arm (before propagation).
    assert!(out.matches("cleanup ()").count() >= 2, "generated: {}", out);
}

#[test]
fn zero_division_raises_catchable_zero_division_error() {
    // Issue #75: py_floordiv/py_mod used to panic on a zero divisor, so
    // `except ZeroDivisionError` could never catch them. They now return
    // Result and the call sites thread `?`, keeping the error inside the
    // try-body closure where the handlers can match it. Issue #107: true
    // division `/` was still silently yielding inf/nan; it now goes through
    // the same Result-returning helper with `?`.
    for op in ["//", "%", "/"] {
        let src = format!(
            "def f():\n    try:\n        print(5 {op} 0)\n    except ZeroDivisionError:\n        print(\"caught\")\n"
        );
        let out = compile(&src, "zero_div.py");
        assert!(
            out.contains("py_mod (5 , 0) ?")
                || out.contains("py_floordiv (5 , 0) ?")
                || out.contains("py_div (5 , 0) ?"),
            "{} must route through the Result-returning helper with `?`: {}",
            op,
            out
        );
        assert!(
            out.contains("__rython_exc . matches (\"ZeroDivisionError\")"),
            "the handler must match ZeroDivisionError: {}",
            out
        );
    }
    // The augmented-assignment forms lower through the same helpers.
    for (op, helper) in [("//", "py_floordiv"), ("%", "py_mod"), ("/", "py_div")] {
        let src = format!(
            "def f():\n    try:\n        x = 10\n        x {op}= 0\n    except ZeroDivisionError:\n        print(\"caught\")\n"
        );
        let out = compile(&src, "zero_div_aug.py");
        assert!(
            out.contains(&format!("x = {helper} (x , 0) ?")),
            "aug {op}= must thread `?`: {}",
            out
        );
    }
}

#[test]
fn hoisted_variable_first_assigned_in_try_body_gets_default_init() {
    // Issue #78: a variable first assigned inside a try body is captured by
    // the try-body closure while possibly-uninitialized, which rustc rejects
    // (E0381). The hoist must give it a Default initializer instead of a
    // bare `let mut x;`.
    let src = concat!(
        "def f():\n",
        "    try:\n",
        "        x = compute()\n",
        "    finally:\n",
        "        cleanup()\n",
        "    print(x)\n",
    );
    let out = compile(src, "try_hoist.py");
    assert!(
        out.contains("let mut x = Default :: default () ;"),
        "hoisted capture needs a Default initializer: {}",
        out
    );
    assert!(
        !out.contains("let mut x ;"),
        "no bare uninitialized hoist for a closure-captured name: {}",
        out
    );
}

#[test]
fn finally_runs_before_handler_and_else_returns() {
    // Python: finally always executes before control leaves the try
    // statement — including when an except handler or else clause returns
    // or raises. Handler/else bodies must route through the finally, not
    // return straight out of the function.
    let src = concat!(
        "def f(n: int) -> int:\n",
        "    try:\n",
        "        check(n)\n",
        "    except ValueError:\n",
        "        return 0\n",
        "    else:\n",
        "        return 1\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "finally_handler.py");
    // Both the handler return and the else return thread out through a
    // PyFlow closure whose Return arm runs cleanup() first.
    assert_eq!(
        out.matches(
            "Ok (PyFlow :: Return (__rython_ret)) => { cleanup () ; return Ok (__rython_ret) ; }"
        )
        .count(),
        2,
        "handler and else returns must run the finally first: {}",
        out
    );

    // A raise inside a handler also runs the finally before propagating.
    let src = concat!(
        "def g(n: int):\n",
        "    try:\n",
        "        check(n)\n",
        "    except ValueError:\n",
        "        raise RuntimeError(\"rethrown\")\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "finally_reraise.py");
    assert!(
        out.contains("Err (__rython_reraise) => { cleanup () ; return Err (__rython_reraise) ; }"),
        "handler raise must run the finally first: {}",
        out
    );

    // Without a finally clause, handler bodies stay inline — no closure.
    let src = concat!(
        "def h(n: int) -> int:\n",
        "    try:\n",
        "        check(n)\n",
        "    except ValueError:\n",
        "        return 0\n",
        "    return 1\n",
    );
    let out = compile(src, "no_finally.py");
    assert!(!out.contains("__rython_inner"), "generated: {}", out);
}

#[test]
fn awaited_async_calls_propagate_exceptions() {
    // Async functions register in the symbol table like ordinary ones, so
    // calls to them get `?` — reordered after `.await` so it unwraps the
    // awaited Result, not the future.
    let src = concat!(
        "async def helper() -> int:\n",
        "    return 1\n",
        "\n",
        "async def caller() -> int:\n",
        "    return await helper()\n",
    );
    let out = compile(src, "async_prop.py");
    assert!(
        out.contains("helper () . await ?"),
        "awaited user call must unwrap the Result: {}",
        out
    );
}

#[test]
fn bare_trailing_return_gets_no_unreachable_tail() {
    // A bare `return` fully exits the function (it extracts as returning
    // None), so no Ok(()) tail may follow it — that would be unreachable
    // code, tripping deny-warnings builds.
    let out = compile("def f():\n    work()\n    return\n", "bareret.py");
    assert!(out.contains("return Ok (())"), "generated: {}", out);
    assert!(
        !out.contains("return Ok (()) ; Ok (())"),
        "no unreachable tail after a trailing bare return: {}",
        out
    );
}

#[test]
fn raise_returns_err_from_the_function() {
    // Functions return Result<T, PyException>, so raising anywhere is
    // returning Err — callers propagate it with `?`, as Python propagates
    // exceptions up the call stack.
    let out = compile("def f():\n    raise RuntimeError(\"boom\")\n", "raise.py");
    assert!(
        out.contains("return Err (PyException :: new (\"RuntimeError\""),
        "generated: {}",
        out
    );
    assert!(!out.contains("panic !"), "generated: {}", out);
}

#[test]
fn calls_to_user_functions_propagate_with_question_mark() {
    let src = concat!(
        "def helper() -> int:\n",
        "    return 1\n",
        "\n",
        "def caller() -> int:\n",
        "    return helper() + 1\n",
    );
    let out = compile(src, "prop.py");
    assert!(out.contains("helper () ?"), "generated: {}", out);

    // Builtins that don't raise stay plain (print takes its argument by
    // reference).
    let out = compile("def f(x: int):\n    print(x)\n", "plaincall.py");
    assert!(out.contains("print (& (x))"), "generated: {}", out);
    assert!(!out.contains("print (& (x)) ?"), "generated: {}", out);
}

#[test]
fn return_inside_try_threads_through_controlflow() {
    // A return in a try body must escape the closure, run the finally, and
    // return from the function.
    let src = concat!(
        "def f(n: int) -> int:\n",
        "    try:\n",
        "        return n\n",
        "    except ValueError:\n",
        "        return 0\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "trystmt_ret.py");
    assert!(out.contains("PyFlow :: Return (n)"), "generated: {}", out);
    assert!(
        out.contains(
            "Ok (PyFlow :: Return (__rython_ret)) => { cleanup () ; return Ok (__rython_ret) ; }"
        ),
        "finally must run before the returned value leaves: {}",
        out
    );
}

#[test]
fn assert_lowers_to_assertion_error() {
    let out = compile(
        "def f(n):\n    assert n > 0, \"need positive\"\n",
        "assert.py",
    );
    // Comparisons lower through the PyGt trait (borrowed operands); the
    // integer literal is converted to the parameter's own type (M4).
    assert!(
        out.contains("if ! ((n) . py_gt (& (T :: py_from_int (0))))"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("PyException :: new (\"AssertionError\""),
        "generated: {}",
        out
    );

    // Inside a try, a failed assert is catchable.
    let src = concat!(
        "def f(n):\n",
        "    try:\n",
        "        assert n > 0\n",
        "    except AssertionError:\n",
        "        pass\n",
    );
    let out = compile(src, "assert_try.py");
    assert!(
        out.contains("return Err (PyException :: new (\"AssertionError\""),
        "generated: {}",
        out
    );
}

#[test]
fn unary_plus_emits_no_invalid_operator() {
    // Rust has no unary +; `+x` is the identity.
    let out = compile("y = +x", "uadd.py");
    assert!(!out.contains("= + x"), "generated: {}", out);
    assert!(out.contains("y = (x)"), "generated: {}", out);
}

#[test]
fn conditions_apply_python_truthiness() {
    // Non-bool condition: wrapped in is_truthy (empty string/list and zero
    // are false, as in Python).
    let out = compile(
        "def f(items):\n    if items:\n        work()\n",
        "truthy.py",
    );
    assert!(
        out.contains("if (items) . is_truthy ()"),
        "generated: {}",
        out
    );

    let out = compile(
        "def f(n):\n    while n:\n        work()\n",
        "truthy_while.py",
    );
    assert!(
        out.contains("while (n) . is_truthy ()"),
        "generated: {}",
        out
    );

    // Comparisons already yield bool: no wrapping.
    let out = compile(
        "def f(n: int):\n    if n < 0:\n        work()\n",
        "truthy_cmp.py",
    );
    assert!(!out.contains("is_truthy"), "generated: {}", out);

    // Boolean operators recurse into operands; `not` negates a condition.
    let out = compile(
        "def f(a, b):\n    if a and not b:\n        work()\n",
        "truthy_bool.py",
    );
    assert!(
        out.contains("((a) . is_truthy ()) && (! ((b) . is_truthy ()))"),
        "generated: {}",
        out
    );
}

#[test]
fn is_none_lowers_to_py_is_none() {
    let out = compile(
        "def f(x):\n    if x is None:\n        work()\n",
        "isnone.py",
    );
    assert!(out.contains("(x) . py_is_none ()"), "generated: {}", out);

    let out = compile(
        "def f(x):\n    if x is not None:\n        work()\n",
        "isnotnone.py",
    );
    assert!(out.contains("! (x) . py_is_none ()"), "generated: {}", out);

    // `is` between two non-None values keeps the identity approximation.
    let out = compile("found = a is b", "isplain.py");
    assert!(out.contains("& a == & b"), "generated: {}", out);
}

#[test]
fn python_list_methods_map_to_correct_rust() {
    let src = concat!(
        "def f() -> int:\n",
        "    items = [1, 2, 3]\n",
        "    items.append(4)\n",
        "    items.remove(2)\n",
        "    items.insert(0, 9)\n",
        "    last = items.pop()\n",
        "    return last + items.count(9)\n",
    );
    let out = compile(src, "listops.py");
    // append pushes one element (Vec::append concatenates — wrong).
    assert!(out.contains("(items) . push (4)"), "generated: {}", out);
    // remove removes by value and raises ValueError when absent.
    assert!(out.contains("position"), "generated: {}", out);
    assert!(out.contains("\"ValueError\""), "generated: {}", out);
    // insert applies Python index rules (negatives, clamping) via py_insert.
    assert!(out.contains("py_insert (0 , 9)"), "generated: {}", out);
    // pop raises a catchable IndexError instead of returning an Option.
    assert!(out.contains("\"IndexError\""), "generated: {}", out);
    assert!(out.contains("pop () . ok_or_else"), "generated: {}", out);
    // count passes by reference to the PyListOps method.
    assert!(out.contains("count (& (9))"), "generated: {}", out);
}

#[test]
fn python_str_methods_map_through_pystrops() {
    let src = concat!(
        "def f(s: str) -> str:\n",
        "    parts = s.split()\n",
        "    head = s.split(\",\")\n",
        "    n = s.find(\"x\")\n",
        "    return \"-\".join(parts)\n",
    );
    let out = compile(src, "strops.py");
    assert!(out.contains("py_split_whitespace ()"), "generated: {}", out);
    assert!(out.contains("py_split (& (\",\")) ?"), "generated: {}", out);
    assert!(out.contains("py_find (& (\"x\"))"), "generated: {}", out);
    assert!(out.contains(". join (parts)"), "generated: {}", out);
}

#[test]
fn str_parameters_accept_borrowed_and_owned_strings() {
    let out = compile(
        "def shout(name: str) -> str:\n    return name.upper()\n",
        "strparam.py",
    );
    // The parameter is generic over Into<String>, converted once up front.
    assert!(
        out.contains("name : impl Into < String >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("let name : String = name . into ()"),
        "generated: {}",
        out
    );
}

#[test]
fn for_loop_target_leaks_to_the_enclosing_scope() {
    // Python's loop variable is function-scoped: a target also referenced
    // outside the loop keeps its value after it. The scope analysis hoists
    // the name; the loop must STORE into it instead of shadowing it with a
    // fresh `for` binding (issue #80).
    let out = compile(
        "def f(items: list[int]) -> int:\n    x = 0\n    for x in items:\n        pass\n    return x\n",
        "forleak.py",
    );
    assert!(
        out.contains("for __rython_elt in items") && !out.contains("for x in items"),
        "leaked loop target must lower to a store into the hoisted binding: {}",
        out
    );
    assert!(out.contains("x = __rython_elt ;"), "generated: {}", out);

    // A target only referenced inside the loop keeps its direct binding.
    let out = compile(
        "def f(items: list[int]) -> int:\n    s = 0\n    for v in items:\n        s = s + v\n    return s\n",
        "forlocal.py",
    );
    assert!(out.contains("for v in items"), "generated: {}", out);
    assert!(!out.contains("__rython_elt"), "generated: {}", out);

    // Unused index names still lower to `_` (issue #101) unless hoisted.
    let out = compile(
        "def f(items: list[int]) -> int:\n    n = 0\n    for _ in items:\n        n = n + 1\n    return n\n",
        "forunderscore.py",
    );
    assert!(out.contains("for _ in items"), "generated: {}", out);
}

#[test]
fn list_remove_evaluates_receiver_and_value_once() {
    // The old lowering spliced the receiver twice and the value inside the
    // position closure (once per scanned element); a side-effecting
    // receiver (`xs[which()].remove(2)`) ran twice. Both are now bound to
    // temps before the scan (issue #80).
    let out = compile(
        "def f(xs: list[list[int]]) -> None:\n    xs[0].remove(2)\n",
        "removeonce.py",
    );
    assert!(
        out.contains("let __rython_recv") && out.contains("let __rython_val = 2"),
        "receiver and value must be bound exactly once: {}",
        out
    );
    assert_eq!(out.matches("py_index_mut").count(), 1, "generated: {}", out);
}

#[test]
fn chained_assignment_to_a_container_literal_clones_per_target() {
    // `a = b = []` needs shared aliasing: each target gets its own copy and
    // later mutations through one name silently diverge from Python (issue
    // #80). The aliasing divergence (issues #79/#104): the literal is
    // built once into a temp and CLONED into each target, with the -W
    // channel reporting the lossy semantics.
    let (out, warnings) = compile_with_warnings("a = b = []\n", "chainlist.py");
    assert!(
        out.contains("__rython_chain . clone ()"),
        "each target must receive its own clone: {}",
        out
    );
    assert!(out.contains("Vec :: < stdpython :: PyValue > :: new ()"), "generated: {}", out);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("chained assignment to a container literal")
                && w.contains("shared aliasing")),
        "the aliasing divergence must be reported through -W: {:?}",
        warnings
    );

    // A tuple literal is immutable — copies are unobservable, so it stays.
    let module = crate::parse("a = b = (1, 2)\n", "chaintuple.py").unwrap();
    let symbols = module.clone().find_symbols(crate::SymbolTableScopes::new());
    let out = module
        .to_rust(
            crate::CodeGenContext::Module("chaintuple".to_string()),
            crate::PythonOptions::default(),
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(out.contains("__rython_chain"), "generated: {out}");
}

#[test]
fn omitted_defaults_must_be_constant() {
    // CPython evaluates defaults once at def time; rython inlines them at
    // the call site. A mutable default would also be SHARED across calls,
    // which owned values cannot express (issue #80) — loud error when a
    // call actually omits it.
    let module = crate::parse("def f(x=[]):\n    return x\n\nf()\n", "mutdefault.py").unwrap();
    let symbols = module.clone().find_symbols(crate::SymbolTableScopes::new());
    let err = module
        .to_rust(
            crate::CodeGenContext::Module("mutdefault".to_string()),
            crate::PythonOptions::default(),
            symbols,
        )
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("mutable default") && msg.contains("SHARES"),
        "error: {msg}"
    );

    // A non-constant default that is never omitted at any call site stays
    // out of the way: the deprecated-note signature path already forces
    // every argument to be passed explicitly.
    let module = crate::parse(
        "def g(x=compute()):\n    return x\n\ng(3)\n",
        "nonconstdefault.py",
    )
    .unwrap();
    let symbols = module.clone().find_symbols(crate::SymbolTableScopes::new());
    let out = module
        .to_rust(
            crate::CodeGenContext::Module("nonconstdefault".to_string()),
            crate::PythonOptions::default(),
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(out.contains("g (3)"), "generated: {out}");

    // A constant default inlined at an omitted call site is fine.
    let module = crate::parse("def h(x=7):\n    return x\n\nh()\n", "constdefault.py").unwrap();
    let symbols = module.clone().find_symbols(crate::SymbolTableScopes::new());
    let out = module
        .to_rust(
            crate::CodeGenContext::Module("constdefault".to_string()),
            crate::PythonOptions::default(),
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(out.contains("h (7)"), "generated: {out}");
}

#[test]
fn user_definitions_shadow_stdlib_module_spellings() {
    // `re = ...` then `re.search(...)` must call the user's object, not
    // the re module (issue #80). The module intercept defers to the
    // user-defined symbol.
    let out = compile(
        "def f() -> str:\n    re = \"x\"\n    return re.upper()\n",
        "shadowre.py",
    );
    assert!(out.contains("re . upper ()"), "generated: {out}");
    assert!(!out.contains("re :: upper"), "generated: {out}");

    // Same for the call-dispatch modules (heapq as a callable name).
    let out = compile(
        "def f() -> int:\n    heapq = 5\n    return heapq\n",
        "shadowheapq.py",
    );
    assert!(!out.contains("heapq :: "), "generated: {out}");
}

#[test]
fn subscripts_lower_through_py_index() {
    // Reads follow Python index rules (negatives, catchable IndexError).
    let out = compile(
        "def f(items: list[int], i: int) -> int:\n    return items[i]\n",
        "sub.py",
    );
    assert!(
        out.contains("(items) . py_index (i) ?"),
        "generated: {}",
        out
    );

    // Stores go through py_set_index, not the Load lowering. The value is
    // bound first: Python evaluates the RHS, then the receiver and index
    // (issue #80 — and a side-effecting RHS runs exactly once).
    let out = compile(
        "def f(items: list[int]):\n    items[0] = 5\n",
        "substore.py",
    );
    assert!(
        out.contains("let __rython_val = 5 ; (items) . py_set_index (0 , __rython_val) ?"),
        "generated: {}",
        out
    );
    assert!(!out.contains("py_index (0) ? ="), "generated: {}", out);

    // Dict stores insert; catchable KeyError on reads comes from PyIndex.
    // String-keyed dicts own the key at the store site (py_set_index
    // takes String for PyDict<String, V>); reads take &str. The value is
    // bound before the store (issue #80).
    let out = compile(
        "def f():\n    d = {\"a\": 1}\n    d[\"b\"] = 2\n    return d[\"a\"]\n",
        "dictsub.py",
    );
    assert!(
        out.contains(
            "let __rython_val = 2 ; (d) . py_set_index ((\"b\") . to_string () , __rython_val) ?"
        ),
        "generated: {}",
        out
    );
    assert!(out.contains("py_index (\"a\") ?"), "generated: {}", out);
}

#[test]
fn slices_lower_through_py_slice() {
    let out = compile(
        "def f(items: list[int]):\n    return items[1:3]\n",
        "slice1.py",
    );
    assert!(
        out.contains("py_slice (Some (1) , Some (3) , None)"),
        "generated: {}",
        out
    );

    let out = compile("def f(s: str) -> str:\n    return s[::-1]\n", "slice2.py");
    assert!(
        out.contains("py_slice (None , None , Some (- 1))"),
        "generated: {}",
        out
    );
}

#[test]
fn container_annotations_map_to_rust_types() {
    let out = compile(
        "def f(a: list[int], b: dict[str, int], c: set[int]):\n    pass\n",
        "generics.py",
    );
    assert!(out.contains("a : Vec < i64 >"), "generated: {}", out);
    assert!(
        out.contains("b : PyDict < String , i64 >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("c : std :: collections :: HashSet < i64 >"),
        "generated: {}",
        out
    );
}

#[test]
fn augmented_assignment_to_subscript_reads_and_stores() {
    // counts[k] += 1 is read-modify-write through py_index/py_set_index —
    // the Load lowering yields a temporary, not a place.
    let out = compile(
        "def f():\n    counts = {\"a\": 1}\n    counts[\"a\"] += 5\n",
        "augsub.py",
    );
    assert!(
        out.contains("py_index (__rython_idx . clone ()) ?"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("py_set_index (__rython_idx , (__rython_elem) . py_add (& (5))) ?"),
        "generated: {}",
        out
    );

    // Other operators combine with the read value too.
    let out = compile(
        "def f():\n    nums = [1, 2]\n    nums[-1] *= 2\n",
        "augsub2.py",
    );
    assert!(
        out.contains("py_set_index (__rython_idx , __rython_elem * 2) ?"),
        "generated: {}",
        out
    );
}

#[test]
fn bare_numeric_literals_are_anchored_in_addition() {
    // `1 + 2` with no type anchor: the PyAdd receiver must have a concrete
    // type, or trait resolution fails before integer-literal fallback.
    let out = compile("y = 1 + 2", "anchor.py");
    assert!(
        out.contains("((1) as i64) . py_add (& ((2) as i64))"),
        "generated: {}",
        out
    );

    let out = compile("y = 1.5 + 2.5", "anchor2.py");
    assert!(
        out.contains("((1.5) as f64) . py_add"),
        "generated: {}",
        out
    );
}

#[test]
fn addition_lowers_through_py_add() {
    // Python + covers String + String and list concat, which Rust's Add
    // doesn't; operands are borrowed so variables stay usable.
    let out = compile(
        "def f(a: str, b: str) -> str:\n    return a + b\n",
        "addstr.py",
    );
    assert!(out.contains("(a) . py_add (& (b))"), "generated: {}", out);

    let out = compile(
        "def f(n: int) -> int:\n    n += 1\n    return n\n",
        "addaug.py",
    );
    assert!(
        out.contains("n = (n) . py_add (& (1))"),
        "generated: {}",
        out
    );
}

#[test]
fn dict_literals_and_methods_lower_through_pydict() {
    // Dict literals are insertion-ordered PyDicts, not HashMaps.
    let out = compile("d = {\"a\": 1}", "dictlit.py");
    assert!(out.contains("PyDict :: from"), "generated: {}", out);
    assert!(!out.contains("HashMap :: from"), "generated: {}", out);

    // Method mappings: get/pop/setdefault/views.
    let src = concat!(
        "def f() -> int:\n",
        "    d = {\"a\": 1}\n",
        "    x = d.get(\"a\", 0)\n",
        "    y = d.pop(\"a\")\n",
        "    z = d.pop(\"gone\", 9)\n",
        "    d.setdefault(\"b\", 2)\n",
        "    ks = d.keys()\n",
        "    vs = d.values()\n",
        "    it = d.items()\n",
        "    return x + y + z\n",
    );
    let out = compile(src, "dictops.py");
    assert!(
        out.contains("py_get_default (& ((\"a\") . to_string ()) , 0)"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("py_pop ((\"a\") . to_string ()) ?"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("py_pop_default ((\"gone\") . to_string () , 9)"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("py_setdefault ((\"b\") . to_string () , 2)"),
        "generated: {}",
        out
    );
    assert!(out.contains("py_keys ()"), "generated: {}", out);
    assert!(out.contains("py_values ()"), "generated: {}", out);
    assert!(out.contains("py_items ()"), "generated: {}", out);

    // get with one argument returns an Option (value-or-None). Dict keys
    // are String (literal or annotation), so literal keys are owned.
    let out = compile(
        "def g(d: dict[str, int]):\n    v = d.get(\"k\")\n",
        "dictget.py",
    );
    assert!(
        out.contains("py_get (& ((\"k\") . to_string ()))"),
        "generated: {}",
        out
    );
}

#[test]
fn del_full_slice_clears_in_place() {
    // Python: `del xs[:]` on ["a","b","c"] leaves []; the lowering is the
    // container's in-place clear. Verified against python3.
    let out = compile(
        "xs = [\"a\", \"b\", \"c\"]\ndel xs[:]\n",
        "delclear.py",
    );
    assert!(out.contains(". clear ()"), "generated: {}", out);
}

#[test]
fn bounded_slice_delete_lowers_to_py_slice_delete() {
    // Python: `del xs[1:3]` on [1,2,3,4] leaves [1,4] - range removal
    // with clamped/negative-normalized bounds (issue #153).
    let out = compile("xs = [1, 2, 3, 4]\ndel xs[1:3]\n", "bounedel.py");
    assert!(
        out.contains("py_slice_delete (Some (1) , Some (3))"),
        "generated: {}",
        out
    );
}

#[test]
fn extended_step_slice_delete_lowers_to_the_step_variant() {
    // `del xs[a:b:c]` removes the strided selection via the runtime's
    // index-computing delete.
    let out = compile(
        "xs = [1, 2, 3, 4]\ndel xs[::2]\n",
        "stepdel.py",
    );
    assert!(
        out.contains("py_slice_delete_step"),
        "generated: {}",
        out
    );
}

#[test]
fn negative_step_slice_delete_removes_the_selection() {
    // Python: `del xs[::-1]` on [1,2,3,4] leaves []; `del xs[::-2]` on
    // [0,1,2] leaves [1]. Verified against python3 3.14 - the runtime's
    // extended index walk removes highest-slot-first.
    let out = compile("xs = [1, 2, 3, 4]\ndel xs[::-1]\n", "negdel.py");
    assert!(
        out.contains("py_slice_delete_step"),
        "generated: {}",
        out
    );
}

#[test]
fn slice_assignment_lowers_to_py_slice_assign() {
    // Issue #153: `xs[a:b] = R` replaces the clamped range in place -
    // a different-length RHS inserts or removes elements, exactly like
    // CPython's list_ass_subscript.
    let out = compile("xs = [1, 2, 3, 4]\nxs[1:3] = [9]\n", "sliceassign.py");
    assert!(
        out.contains("py_slice_assign (Some (1) , Some (3) , vec ! [9])"),
        "generated: {}",
        out
    );
}

#[test]
fn strided_slice_assignment_lowers_to_the_step_variant() {
    // `ys[::2] = [9, 9]` on [0,1,2] -> [9,1,9]: slots computed by the
    // runtime, replacement length-checked (ValueError on mismatch).
    let out = compile("ys = [0, 1, 2]\nys[::2] = [9, 9]\n", "strideassign.py");
    assert!(
        out.contains("py_slice_assign_step"),
        "generated: {}",
        out
    );
}

#[test]
fn negative_strided_slice_assignment_assigns_in_slot_order() {
    // Python: `ys[::-2] = [7, 8]` on [0,1,2] -> [8, 1, 7]: selection is
    // indices [2, 0]; values assign left-to-right onto those slots.
    // Verified against python3 3.14.
    let out = compile("ys = [0, 1, 2]\nys[::-2] = [7, 8]\n", "negassign.py");
    assert!(
        out.contains("py_slice_assign_step"),
        "generated: {}",
        out
    );
}

#[test]
fn isinstance_type_call_resolves_statically() {
    // Issue #134 (charset_normalizer): `isinstance(x, type(self))` —
    // `type(...)` of a statically-known instance resolves to that class.
    // Same class: true. A different known class: false. An UNANNOTATED
    // argument has unknown type: the documented class-as-value divergence
    // lowers to false and records a definition warning naming it.
    let src = concat!(
        "class Door:\n",
        "    def __init__(self, ok: bool):\n",
        "        self.ok = ok\n",
        "\n",
        "    def same(self, other: \"Door\") -> bool:\n",
        "        return isinstance(other, type(self))\n",
        "\n",
        "    def diff(self, other: \"Door\") -> bool:\n",
        "        return isinstance(other, type(int))\n",
        "\n",
        "    def unknown(self, other) -> bool:\n",
        "        return isinstance(other, type(self))\n"
    );
    let out = compile(src, "istype.py");

    let same_part = out.split("fn same").nth(1).expect("same fn");
    assert!(
        same_part.contains("return Ok (true)"),
        "same-class must be true: {}",
        out
    );
    let diff_part = out.split("fn diff").nth(1).expect("diff fn");
    assert!(
        diff_part.contains("return Ok (false)"),
        "different class must be false: {}",
        out
    );
}

#[test]
fn isinstance_type_call_unknown_argument_warns_divergence() {
    let src = concat!(
        "class Door:\n",
        "    def unknown(self, other) -> bool:\n",
        "        return isinstance(other, type(self))\n"
    );
    let (out, warnings) = compile_with_warnings(src, "istypeunknown.py");
    assert!(out.contains("return Ok (false)"), "generated: {}", out);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("class-as-value divergence")),
        "warnings: {:?}",
        warnings
    );
}

#[test]
fn nonself_receiver_is_renamed_through_the_body_and_nested_scopes() {
    // Python binds the instance to the FIRST parameter whatever its name
    // (issue #132, boto3's factory_self): body references rename to the
    // Rust `self` receiver — including through a nested function that
    // captures it — while a nested function binding its OWN receiver
    // keeps its scope untouched.
    let out = compile(
        concat!(
            "class W:\n",
            "    def __init__(self, base: int):\n",
            "        self.base = base\n",
            "\n",
            "    def scale(factory_self, times: int) -> int:\n",
            "        def twice():\n",
            "            return 2\n",
            "        return (factory_self.base * times) + twice()\n",
            "\n",
            "    def check(factory_self, ok: bool) -> int:\n",
            "        assert factory_self.base >= 0\n",
            "        if not ok:\n",
            "            raise ValueError(f\"bad {factory_self.base}\")\n",
            "        return 0\n",
            "\n",
            "    @staticmethod\n",
            "    def make_inner(self_arg: int) -> int:\n",
            "        return self_arg\n",
        ),
        "recvrename.py",
    );
    assert!(
        !out.contains("factory_self"),
        "receiver references must be renamed to self: {out}"
    );
    // The renamed receiver flows through normal method lowering.
    assert!(out.contains("self . base"), "generated: {out}");
    // The nested staticmethod's own first parameter is ITS receiver-name
    // binding: its body must not be rewritten to a `self` that does not
    // exist in an associated fn.
    let make_fn = out.split("fn make_inner").nth(1).expect("make_inner");
    assert!(make_fn.contains("self_arg"), "generated: {}", out);
}

#[test]
fn rebinding_a_nonself_receiver_is_a_loud_error() {
    // Python would rebind the local reference; rython's receiver is an
    // immutable &self, so the reassignment has no lowering — loud error
    // instead of silently different codegen.
    let err = compile_err(
        "class W:\n    def m(factory_self):\n        factory_self = 3\n        return factory_self\n",
        "recvrebind.py",
    );
    assert!(
        err.contains("rebinds its receiver"),
        "err: {err}"
    );
}

#[test]
fn keyword_arguments_map_to_parameter_positions() {
    let src = concat!(
        "def volume(w: int, h: int, d: int) -> int:\n",
        "    return w * h * d\n",
        "\n",
        "def f() -> int:\n",
        "    return volume(d=2, w=3, h=4)\n",
    );
    let out = compile(src, "kw.py");
    // Keywords land in signature order regardless of call order; the
    // arguments are bound to temps in SOURCE order first (d=2, w=3, h=4),
    // then referenced in parameter order (issue #80). The reordering block
    // is parenthesized before `?` so it is valid in any position (F9).
    assert!(
        out.contains("let __rython_arg_0 = 2 ; let __rython_arg_1 = 3 ; let __rython_arg_2 = 4 ; volume (__rython_arg_1 , __rython_arg_2 , __rython_arg_0) }) ?"),
        "generated: {}",
        out
    );
}

#[test]
fn omitted_defaults_fill_at_the_call_site() {
    let src = concat!(
        "def greet(name: str = \"world\", excited: bool = False) -> str:\n",
        "    return name\n",
        "\n",
        "def f() -> str:\n",
        "    return greet()\n",
        "\n",
        "def g() -> str:\n",
        "    return greet(excited=True)\n",
    );
    let out = compile(src, "kwdef.py");
    assert!(
        out.contains("greet (\"world\" , false) }) ?"),
        "generated: {}",
        out
    );
    // greet(excited=True): name's default is a constant, so it stays
    // inlined in parameter position while the keyword is bound to a temp
    // (evaluated in source order, then referenced in parameter order).
    assert!(
        out.contains("let __rython_arg_0 = true ; greet (\"world\" , __rython_arg_0) }) ?"),
        "keyword for the second param leaves the first defaulted: {}",
        out
    );
}

#[test]
fn keywords_on_unknown_callees_lower_positionally() {
    // Without a signature the keyword order can't be checked. The
    // dynamic-dispatch divergence: keywords lower POSITIONALLY (in source
    // order) so the call still converts — refusing to convert would break
    // whole modules over one dynamic call.
    let out = compile("unknown_func(a=1)\n", "kwunknown.py");
    assert!(
        out.contains("unknown_func (1)"),
        "the keyword value must lower positionally: {}",
        out
    );
}

#[test]
fn dict_comprehensions_build_ordered_pydicts() {
    // Comprehension-built dicts preserve insertion order like literals.
    let out = compile(
        "def f(items: list[int]):\n    return {x: x * 2 for x in items}\n",
        "dictcomp.py",
    );
    assert!(out.contains("PyDict :: new ()"), "generated: {}", out);
    assert!(!out.contains("HashMap :: new ()"), "generated: {}", out);
}

#[test]
fn none_lowers_to_option() {
    // x = None initializes an Option; later non-None stores wrap in Some
    // so both arms unify to Option<T>.
    let src = concat!(
        "def f(items: list[int]) -> int:\n",
        "    found = None\n",
        "    for x in items:\n",
        "        found = x\n",
        "    if found is None:\n",
        "        return -1\n",
        "    return 0\n",
    );
    let out = compile(src, "opt.py");
    assert!(out.contains("found = None"), "generated: {}", out);
    assert!(out.contains("found = Some (x)"), "generated: {}", out);
    assert!(
        out.contains("(found) . py_is_none ()"),
        "generated: {}",
        out
    );
}

#[test]
fn optional_annotations_map_to_option() {
    let out = compile(
        "def f(tag: Optional[int], n: int | None) -> int:\n    return 0\n",
        "optann.py",
    );
    assert!(out.contains("tag : Option < i64 >"), "generated: {}", out);
    assert!(out.contains("n : Option < i64 >"), "generated: {}", out);
}

#[test]
fn optional_parameters_wrap_arguments_at_call_sites() {
    let src = concat!(
        "def label(tag: Optional[int]) -> int:\n",
        "    return 0\n",
        "\n",
        "def f() -> int:\n",
        "    a = label(7)\n",
        "    b = label(None)\n",
        "    return a + b\n",
    );
    let out = compile(src, "optcall.py");
    // All-positional calls emit directly (no reordering): the `?` still
    // applies to the whole call, now wrapped in the lowering block and
    // parenthesized so it is valid in any position (F9).
    assert!(out.contains("({ label (Some (7)) }) ?"), "generated: {}", out);
    assert!(out.contains("({ label (None) }) ?"), "generated: {}", out);
}

#[test]
fn optional_stores_from_option_values_do_not_double_wrap() {
    // The RHS already yields an Option (dict.get, another optional name, an
    // Optional-returning call): wrapping it again would bury an absent value
    // as Some(None) and flip a later `is None` check.
    let src = concat!(
        "def probe(d: dict[str, int], keys: list[str]) -> int:\n",
        "    result = None\n",
        "    for k in keys:\n",
        "        result = d.get(k)\n",
        "    alias = None\n",
        "    alias = result\n",
        "    if alias is None:\n",
        "        return -1\n",
        "    return 0\n",
    );
    let out = compile(src, "optget.py");
    assert!(out.contains("result = (d) . py_get"), "generated: {}", out);
    assert!(
        !out.contains("Some ((d) . py_get"),
        "double-wrapped dict.get store, generated: {}",
        out
    );
    assert!(out.contains("alias = result"), "generated: {}", out);
    assert!(
        !out.contains("Some (result)"),
        "double-wrapped optional-name store, generated: {}",
        out
    );
}

#[test]
fn conditional_stores_into_optional_names_wrap_per_arm() {
    // `x if c else None` into a None-seeded name wraps each arm
    // independently: Some(x) / None. Wrapping the whole conditional would
    // bury the None arm as Some(None) and flip a later `is None` check.
    let src = concat!(
        "def f(n: int) -> int:\n",
        "    tag = None\n",
        "    tag = n if n > 0 else None\n",
        "    if tag is None:\n",
        "        return 0\n",
        "    return 1\n",
    );
    let out = compile(src, "optifexp.py");
    assert!(
        out.contains("tag = if") && out.contains("Some (n)"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains("Some (if"),
        "wrapped the whole conditional, generated: {}",
        out
    );
}

#[test]
fn conditional_with_option_arms_stores_without_rewrap() {
    // Both arms already yield an Option (dict.get / None): the conditional
    // is an Option and stores through unchanged.
    let src = concat!(
        "def f(d: dict[int, int], n: int) -> int:\n",
        "    choice = None\n",
        "    choice = d.get(n) if n > 0 else None\n",
        "    if choice is None:\n",
        "        return -1\n",
        "    return 0\n",
    );
    let out = compile(src, "optifexp2.py");
    assert!(out.contains("choice = if"), "generated: {}", out);
    assert!(
        !out.contains("Some (if") && !out.contains("Some ((d) . py_get"),
        "double-wrapped a conditional Option, generated: {}",
        out
    );
}

#[test]
fn conditional_arguments_to_optional_parameters_wrap_per_arm() {
    let src = concat!(
        "def label(tag: Optional[int]) -> int:\n",
        "    return 0\n",
        "\n",
        "def f(n: int) -> int:\n",
        "    return label(n if n > 0 else None)\n",
    );
    let out = compile(src, "optifexp3.py");
    assert!(
        out.contains("label (if") && out.contains("Some (n)"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains("Some (if"),
        "wrapped the whole conditional argument, generated: {}",
        out
    );
}

#[test]
fn optional_returning_calls_store_and_pass_without_rewrap() {
    // find() generates Result<Option<i64>, PyException>; the call site's `?`
    // leaves an Option, which must flow into optional names and Optional
    // parameters as-is.
    let src = concat!(
        "def find(d: dict[str, int], k: str) -> Optional[int]:\n",
        "    return d.get(k)\n",
        "\n",
        "def label(tag: Optional[int]) -> int:\n",
        "    return 0\n",
        "\n",
        "def f(d: dict[str, int]) -> int:\n",
        "    hit = None\n",
        "    hit = find(d, \"a\")\n",
        "    return label(find(d, \"b\"))\n",
    );
    let out = compile(src, "optret.py");
    assert!(out.contains("hit = find"), "generated: {}", out);
    assert!(
        !out.contains("hit = Some (find"),
        "double-wrapped Optional-returning call store, generated: {}",
        out
    );
    assert!(
        !out.contains("label (Some (find"),
        "double-wrapped Optional-returning call argument, generated: {}",
        out
    );
}

#[test]
fn typing_imports_lower_to_nothing() {
    let out = compile("from typing import Optional\nx = 1\n", "typing.py");
    assert!(!out.contains("typing"), "generated: {}", out);
}

#[test]
fn membership_uses_py_contains() {
    let out = compile("found = x in items", "in.py");
    assert!(out.contains("py_contains"), "generated: {}", out);

    let out = compile("missing = x not in items", "notin.py");
    assert!(
        out.contains("! (items) . py_contains"),
        "generated: {}",
        out
    );
}

#[test]
fn multiple_lossy_conversions_fold_into_one_attribute() {
    // Rust allows only one #[deprecated] per item, so a function with both a
    // dropped default and an ignored return annotation must fold both notes
    // into a single attribute.
    let out = compile(
        "def f(c, x: int = 3) -> int:\n    if c:\n        return x\n",
        "lossy_both.py",
    );
    assert_eq!(
        out.matches("deprecated").count(),
        1,
        "exactly one #[deprecated] attribute: {}",
        out
    );
    assert!(out.contains("were dropped"), "generated: {}", out);
    assert!(out.contains("return annotation"), "generated: {}", out);
}

#[test]
fn lossy_warnings_can_be_suppressed_by_options() {
    let src = "def f(x: int = 3) -> int:\n    if x:\n        return x\n";
    let module = parse(src, "suppress.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let options = PythonOptions {
        lossy_warnings: false,
        ..Default::default()
    };
    let out = module
        .to_rust(CodeGenContext::Module("suppress".into()), options, symbols)
        .unwrap()
        .to_string();
    assert!(!out.contains("deprecated"), "generated: {}", out);
}

#[test]
fn dropped_defaults_emit_call_site_warning() {
    // Dropping a Python default is a semantic change; the generated function
    // must carry a #[deprecated] note so consumer call sites are warned.
    let out = compile("def f(x: int = 3) -> int:\n    return x\n", "warn_def.py");
    assert!(out.contains("deprecated"), "generated: {}", out);
    assert!(out.contains("were dropped"), "generated: {}", out);

    // No defaults, no warning attribute.
    let out = compile("def g(x: int) -> int:\n    return x\n", "no_warn.py");
    assert!(!out.contains("deprecated"), "generated: {}", out);
}

// ---- Struct-based classes ----

fn compile_err(src: &str, name: &str) -> String {
    let module = parse(src, name).unwrap_or_else(|e| panic!("parse failed: {}", e));
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let err = module
        .to_rust(
            CodeGenContext::Module(name.replace(".py", "")),
            PythonOptions::default(),
            symbols,
        )
        .expect_err("conversion must fail loudly");
    format!("{}", err)
}

const COUNTER: &str = concat!(
    "class Counter:\n",
    "    def __init__(self, label: str, start: int = 0):\n",
    "        self.label = label\n",
    "        self.count = start\n",
    "\n",
    "    def bump(self, amount: int) -> int:\n",
    "        self.count += amount\n",
    "        return self.count\n",
    "\n",
    "    def double_bump(self, amount: int) -> int:\n",
    "        self.bump(amount)\n",
    "        self.bump(amount)\n",
    "        return self.count\n",
    "\n",
    "    def peek(self) -> int:\n",
    "        return self.count\n",
);

#[test]
fn classes_lower_to_structs_with_inferred_fields() {
    let out = compile(COUNTER, "counter.py");
    assert!(out.contains("pub struct Counter"), "generated: {}", out);
    assert!(out.contains("pub label : String"), "generated: {}", out);
    assert!(out.contains("pub count : i64"), "generated: {}", out);
    assert!(
        out.contains("pub fn new (label : impl Into < String > , start : i64) -> Result < Self , PyException >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("__rython_self . __init__ (label , start) ?"),
        "generated: {}",
        out
    );
}

#[test]
fn method_receivers_follow_mutation_including_transitive_calls() {
    let out = compile(COUNTER, "receivers.py");
    // __init__ and bump store through self; double_bump only via calling
    // bump; peek reads only.
    assert!(
        out.contains("fn __init__ (& mut self ,"),
        "generated: {}",
        out
    );
    assert!(out.contains("fn bump (& mut self ,"), "generated: {}", out);
    assert!(
        out.contains("fn double_bump (& mut self ,"),
        "transitive self-call must select &mut self: {}",
        out
    );
    assert!(out.contains("fn peek (& self ,"), "generated: {}", out);
}

#[test]
fn construction_and_method_calls_propagate_exceptions() {
    let src = format!(
        "{}\n\ndef run() -> int:\n    c = Counter(\"hits\")\n    c.bump(amount=2)\n    return c.peek()\n",
        COUNTER
    );
    let (out, warnings) = compile_with_warnings(&src, "classcalls.py");
    // Construction resolves defaults against __init__ (minus self) and
    // lowers to new()?; the omitted `start` fills with its default.
    assert!(
        out.contains("Counter :: new (\"hits\" , 0) ?"),
        "generated: {}",
        out
    );
    // A KEYWORD call on a user-class method maps the keyword to the
    // parameter positionally (`c.bump(amount=2)` → `(c).bump(2)`) — a
    // plain method call, never dropped.
    assert!(
        out.contains("(c) . bump (2)") || out.contains("(c) . bump (__rython_arg_0)"),
        "the keyword call must map to the method: {}",
        out
    );
    assert!(
        !warnings.iter().any(|w| w.contains("bump") && w.contains("is dropped")),
        "a keyword call on a real method must NOT be dropped: {:?}",
        warnings
    );
    assert!(out.contains("peek ()"), "generated: {}", out);
    // A local constructing a mutating class needs a mutable binding.
    assert!(out.contains("let mut c ;"), "generated: {}", out);
}

#[test]
fn user_methods_shadow_builtin_method_rewrites() {
    // A user-defined method named like a dict/list builtin must resolve to
    // the class, not the py_get rewrite.
    let src = concat!(
        "class Box:\n",
        "    def __init__(self, v: int):\n",
        "        self.v = v\n",
        "\n",
        "    def get(self, bonus: int) -> int:\n",
        "        return self.v + bonus\n",
        "\n",
        "def run() -> int:\n",
        "    b = Box(3)\n",
        "    return b.get(1)\n",
    );
    let out = compile(src, "shadow.py");
    assert!(out.contains("(b) . get (1) ?"), "generated: {}", out);
    assert!(!out.contains("py_get"), "generated: {}", out);
}

#[test]
fn composed_fields_type_and_resolve_through_chains() {
    let src = concat!(
        "class Point:\n",
        "    def __init__(self, x: int):\n",
        "        self.x = x\n",
        "\n",
        "    def shift(self, dx: int):\n",
        "        self.x += dx\n",
        "\n",
        "class Holder:\n",
        "    def __init__(self, p: Point):\n",
        "        self.p = p\n",
        "\n",
        "    def nudge(self):\n",
        "        self.p.shift(1)\n",
    );
    let out = compile(src, "compose.py");
    assert!(out.contains("pub p : Point"), "generated: {}", out);
    // shift mutates Point, so nudge mutates self through the field chain.
    assert!(out.contains("fn nudge (& mut self ,"), "generated: {}", out);
    assert!(
        out.contains(". shift (1) ?"),
        "field-chain method calls propagate exceptions: {}",
        out
    );
}

#[test]
fn unsupported_class_constructs_are_tolerated_metadata() {
    // A base rython cannot emit a struct for (an imported module, a
    // builtin) is tolerated as metadata: the class lowers as a plain
    // struct, losing the base (the foreign-base divergence).
    let out = compile("class C(str):\n    pass\n", "builtin_base.py");
    assert!(
        out.contains("pub struct C"),
        "a builtin base must not block conversion: {}",
        out
    );

    // A class-level attribute store is metadata too: it is dropped and the
    // class lowers with just its methods.
    let out = compile("class C:\n    VERSION = 3\n", "classattr.py");
    assert!(
        out.contains("pub struct C"),
        "a class attribute must not block conversion: {}",
        out
    );

    // A None-initialized field lowers as the boxed PyValue.
    let out = compile(
        "class C:\n    def __init__(self):\n        self.x = None\n",
        "noneattr.py",
    );
    assert!(
        out.contains("pub x : stdpython :: PyValue"),
        "a None field must lower as the boxed PyValue: {}",
        out
    );
}

// ---- Trait-based inheritance ----

#[test]
fn inheritance_emits_trait_machinery() {
    let src = concat!(
        "class Base:\n",
        "    def __init__(self, v: int):\n",
        "        self.v = v\n",
        "\n",
        "    def get(self) -> int:\n",
        "        return self.v\n",
        "\n",
        "class Child(Base):\n",
        "    pass\n",
    );
    let out = compile(src, "inherit.py");
    assert!(out.contains("trait BaseTrait"), "generated: {}", out);
    assert!(out.contains("impl BaseTrait for Child"), "generated: {}", out);
    assert!(
        out.contains("pub __rython_base : Base"),
        "derived struct must embed the direct base: {}",
        out
    );
}

#[test]
fn override_reemits_into_ancestor_trait_impl() {
    // Dog.describe overrides Animal.describe: it must be re-emitted inside
    // `impl AnimalTrait for Dog` (Animal's trait), not into Dog's own trait.
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "    def describe(self) -> str:\n",
        "        return self.name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name: str, breed: str):\n",
        "        super().__init__(name)\n",
        "        self.breed = breed\n",
        "\n",
        "    def describe(self) -> str:\n",
        "        return super().describe() + \" \" + self.breed\n",
        "\n",
        "    def bark(self) -> str:\n",
        "        return \"woof\"\n",
    );
    let out = compile(src, "override.py");
    let animal_impl = out.find("impl AnimalTrait for Dog");
    let dog_trait = out.find("trait DogTrait");
    let dog_impl = out.find("impl DogTrait for Dog");
    assert!(
        animal_impl.is_some() && dog_impl.is_some(),
        "missing impls: {}",
        out
    );
    let animal_impl = animal_impl.unwrap();
    let dog_trait = dog_trait.unwrap();
    let dog_impl = dog_impl.unwrap();
    // describe must NOT be a member of Dog's own trait (it belongs to
    // Animal's), and must appear in `impl AnimalTrait for Dog`.
    assert!(
        !out[dog_trait..dog_impl].contains("fn describe"),
        "describe must not live in DogTrait: {}",
        out
    );
    assert!(
        out[animal_impl..].contains("fn describe (& self"),
        "describe must be re-emitted in AnimalTrait impl: {}",
        out
    );
    // bark is a NEW method: it goes into Dog's own trait as a default.
    assert!(
        out[dog_trait..dog_impl].contains("fn bark (& self"),
        "bark must be a DogTrait default: {}",
        out
    );
}

#[test]
fn super_dispatches_through_the_definer_trampoline() {
    // `super().get()` must run the base's ORIGINAL body with the DERIVED
    // self: the call dispatches through the base's super trampoline
    // (`<Self as BaseTrait>::__rython_super_get(self)`), never through the
    // embedded base struct — a call on `self.__rython_base` would pin the
    // receiver to Base and resolve nested `self.x()` inside the body against
    // Base, silently diverging from CPython's MRO.
    let src = concat!(
        "class Base:\n",
        "    def __init__(self, v: int):\n",
        "        self.v = v\n",
        "\n",
        "    def get(self) -> int:\n",
        "        return self.v\n",
        "\n",
        "class Child(Base):\n",
        "    def get(self) -> int:\n",
        "        return super().get() + 1\n",
    );
    let out = compile(src, "super.py");
    assert!(
        out.contains("< Self as BaseTrait > :: __rython_super_get (self ,)"),
        "super().get() must dispatch through the Base super trampoline: {}",
        out
    );
    assert!(
        out.contains("fn __rython_super_get (& self ,)"),
        "BaseTrait must carry the __rython_super_get trampoline default: {}",
        out
    );
    assert!(
        !out.contains("__rython_base) . get ()"),
        "must not call through the embedded base struct: {}",
        out
    );
}

#[test]
fn mutating_override_widens_the_trait_signature() {
    // Dog overrides Animal.grow with a mutating body; Cat inherits the
    // non-mutating default. The trait signature must widen to `&mut self`
    // for BOTH, or the impls would not agree — so a Cat-typed call borrows
    // mutably too.
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "    def grow(self) -> None:\n",
        "        pass\n",
        "\n",
        "class Dog(Animal):\n",
        "    def grow(self) -> None:\n",
        "        self.name = self.name + \"!\"\n",
        "\n",
        "class Cat(Animal):\n",
        "    pass\n",
    );
    let out = compile(src, "mut_override.py");
    // The trait default itself must be &mut self (widened by Dog's override).
    let trait_decl = out.find("trait AnimalTrait");
    let impl_animal = out.find("impl AnimalTrait for Animal");
    assert!(trait_decl.is_some() && impl_animal.is_some(), "{}", out);
    assert!(
        out[trait_decl.unwrap()..impl_animal.unwrap()].contains("fn grow (& mut self"),
        "trait signature must widen to &mut self: {}",
        out
    );
}

#[test]
fn grandchild_override_super_targets_the_definer_base() {
    // Dog.describe overrides Animal.describe and calls super() inside it;
    // Puppy inherits WITHOUT overriding. The re-emitted Dog body inside
    // `impl AnimalTrait for Puppy` must super() to ANIMAL's describe (the
    // definer), not Dog's — via the trampoline, with the derived Self, so
    // nested dispatch inside Animal's body still resolves against Puppy.
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "    def describe(self) -> str:\n",
        "        return self.name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name: str):\n",
        "        super().__init__(name)\n",
        "\n",
        "    def describe(self) -> str:\n",
        "        return super().describe() + \" (dog)\"\n",
        "\n",
        "class Puppy(Dog):\n",
        "    pass\n",
    );
    let out = compile(src, "deep_super.py");
    // The super() inside the re-emitted Dog body resolves against Dog's base
    // (Animal — the definer), so it dispatches through AnimalTrait's
    // trampoline.
    assert!(
        out.contains("< Self as AnimalTrait > :: __rython_super_describe (self ,)"),
        "re-emitted Dog override must super() to Animal's trampoline: {}",
        out
    );
    assert!(
        out.contains("fn __rython_super_describe (& self ,) -> Result < String , PyException >"),
        "AnimalTrait must carry the describe trampoline: {}",
        out
    );
}

#[test]
fn sibling_mutation_widens_through_a_middle_redefinition() {
    // A defines m; B (middle) redefines m WITHOUT mutating; D (a sibling of
    // B) mutates m. The widening is recorded under the TOPMOST definer (A),
    // so a call through B-derived C must still resolve the widened
    // signature — the nearest-definer lookup used to miss and emit a
    // non-mut binding that the trait impl then rejects.
    let src = concat!(
        "class A:\n",
        "    def m(self) -> int:\n",
        "        return 1\n",
        "\n",
        "class B(A):\n",
        "    def m(self) -> int:\n",
        "        return 2\n",
        "\n",
        "class C(B):\n",
        "    pass\n",
        "\n",
        "class D(A):\n",
        "    def __init__(self, x: int):\n",
        "        self.x = x\n",
        "\n",
        "    def m(self) -> int:\n",
        "        self.x = 1\n",
        "        return 3\n",
        "\n",
        "c = C()\n",
        "y = c.m()\n",
    );
    let out = compile(src, "sibling_widen.py");
    assert!(
        out.contains("let mut c"),
        "call site must borrow c mutably (trait widened by sibling D): {}",
        out
    );
}

#[test]
fn middle_class_reassigning_base_field_emits_no_accessor() {
    // Three-level hierarchy where the middle class re-assigns a field its
    // own base owns (Dog sets self.name without super().__init__): the
    // field physically lives in Animal, so `impl DogTrait for Puppy` must
    // not emit a `name` accessor — its own trait declares only `breed` +
    // `base`, and an undeclared trait method (or one reaching a
    // non-existent Dog.name field) would not compile.
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name: str, breed: str):\n",
        "        self.name = name\n",
        "        self.breed = breed\n",
        "\n",
        "class Puppy(Dog):\n",
        "    pass\n",
    );
    let out = compile(src, "reassigned_field.py");
    // The correct layout: Animal's own field is reached from Puppy through
    // TWO embedded base levels (Puppy -> Dog -> Animal).
    assert!(
        out.contains("__rython_base . __rython_base . name"),
        "name must live two levels down in Animal: {}",
        out
    );
    // Dog's own fields are only breed (+base). Slice the Dog-impl-for-Puppy
    // block and require no `name` accessor there.
    let start = out
        .find("impl DogTrait for Puppy")
        .unwrap_or_else(|| panic!("missing Dog impl for Puppy: {}", out));
    let rest = &out[start..];
    let end = rest
        .find("\nimpl ")
        .map(|i| i + start)
        .unwrap_or(out.len());
    let dog_impl = &out[start..end];
    assert!(
        !dog_impl.contains("fn name"),
        "Dog's trait impl for Puppy must not declare a `name` accessor: {}",
        dog_impl
    );
    assert!(
        dog_impl.contains("fn breed"),
        "Dog's own field accessor must still be emitted: {}",
        dog_impl
    );
}

#[test]
fn relative_import_above_crate_root_is_a_clean_error() {
    // A relative import with more leading dots than the module's package
    // depth must fail loudly with the dedicated message, not panic on an
    // index underflow in the resolved-module-path computation. The import
    // must be USED as a class BEFORE the import statement renders:
    // construction of an imported name forces the module-path resolution
    // (call.rs), which runs while rendering the first statement — before
    // the import statement's own error check fires.
    let err = compile_err(
        "x = Thing()\nfrom ....nope import Thing\n",
        "deep_relative.py",
    );
    assert!(
        err.contains("relative import goes above the crate root"),
        "expected the clean above-crate-root error, got: {}",
        err
    );
}

#[test]
fn own_override_super_targets_the_direct_base() {
    // Puppy's OWN describe calls super().describe(): it resolves Dog's
    // override (one level up), so the trampoline is DogTrait's — called
    // with the derived Self (Puppy at the call site inside Puppy's body).
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "    def describe(self) -> str:\n",
        "        return self.name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def describe(self) -> str:\n",
        "        return super().describe() + \" (dog)\"\n",
        "\n",
        "class Puppy(Dog):\n",
        "    def describe(self) -> str:\n",
        "        return \"puppy \" + super().describe()\n",
    );
    let out = compile(src, "own_super.py");
    assert!(
        out.contains("< Self as DogTrait > :: __rython_super_describe (self ,)"),
        "Puppy's super() must hit Dog's trampoline: {}",
        out
    );
    // Dog's own body ALSO supers to Animal, and its trampoline is re-emitted
    // nowhere: DogTrait::__rython_super_describe runs Animal's body with the
    // derived Self.
    assert!(
        out.contains("< Self as AnimalTrait > :: __rython_super_describe (self ,)"),
        "Dog's super() must hit Animal's trampoline: {}",
        out
    );
}

// ---- Trait-based inheritance ----










#[test]
fn str_getters_clone_the_field_out_of_the_shared_receiver() {
    // `def name(self) -> str: return self.name` reads a String field
    // through &self: the return clones it — semantically exact, since
    // Python strings are immutable.
    let src = concat!(
        "class Tag:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "    def get_name(self) -> str:\n",
        "        return self.name\n",
    );
    let out = compile(src, "getter.py");
    assert!(
        out.contains("Ok ((self . name) . clone ())"),
        "generated: {}",
        out
    );
}

#[test]
fn class_method_named_new_is_dropped_for_the_constructor() {
    // A method named `new` collides with the synthesized constructor. It
    // is dropped (the constructor occupies the name) and the -W channel
    // reports the lossy divergence.
    let (out, warnings) = compile_with_warnings(
        "class C:\n    def new(self) -> int:\n        return 1\n",
        "newclash.py",
    );
    assert!(
        out.contains("pub fn new () -> Result < Self , PyException >"),
        "the synthesized constructor must be emitted: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("method `new") && w.contains("dropped")),
        "the dropped `new` method must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn read_only_methods_with_mutator_names_do_not_force_mut() {
    // A user method shadowing a builtin mutator name (`pop`) that only
    // reads must not force a mutable receiver binding — class resolution
    // is authoritative over the syntactic method-name list.
    let src = concat!(
        "class Box:\n",
        "    def __init__(self, v: int):\n",
        "        self.v = v\n",
        "\n",
        "    def pop(self) -> int:\n",
        "        return self.v\n",
        "\n",
        "def run() -> int:\n",
        "    b = Box(3)\n",
        "    return b.pop()\n",
    );
    let out = compile(src, "romut.py");
    assert!(out.contains("fn pop (& self ,"), "generated: {}", out);
    assert!(
        out.contains("let b ;") && !out.contains("let mut b ;"),
        "read-only pop must not force `mut`: {}",
        out
    );
}

#[test]
fn mutations_inside_keyword_arguments_are_detected() {
    // `use_it(n=c.bump(2))` mutates `c` through a keyword-argument value;
    // the binding must be mutable.
    let src = concat!(
        "class Counter:\n",
        "    def __init__(self, start: int):\n",
        "        self.count = start\n",
        "\n",
        "    def bump(self, amount: int) -> int:\n",
        "        self.count += amount\n",
        "        return self.count\n",
        "\n",
        "def use_it(n: int) -> int:\n",
        "    return n\n",
        "\n",
        "def run() -> int:\n",
        "    c = Counter(1)\n",
        "    return use_it(n=c.bump(2))\n",
    );
    let out = compile(src, "kwmut.py");
    assert!(
        out.contains("let mut c ;"),
        "keyword-nested mutation must mark `c` mutable: {}",
        out
    );
}

#[test]
fn split_keyword_arguments_map_or_lower_positionally() {
    // maxsplit by keyword maps to the right runtime variant...
    let out = compile(
        "def f(s: str):\n    return s.split(\",\", maxsplit=1)\n",
        "kwsplit.py",
    );
    assert!(
        out.contains("py_split_maxsplit (& (\",\") , 1) ?"),
        "generated: {}",
        out
    );
    // ...including whitespace mode with a keyword-only maxsplit.
    let out = compile(
        "def f(s: str):\n    return s.rsplit(maxsplit=2)\n",
        "kwrsplit.py",
    );
    assert!(
        out.contains("py_rsplit_whitespace_maxsplit (2)"),
        "generated: {}",
        out
    );
    // Foreign keywords on str.split drop the keyword NAME and lower the
    // value positionally (the dynamic-dispatch divergence), so the call
    // still converts.
    let out = compile(
        "def f(s: str):\n    return s.split(\",\", bogus=1)\n",
        "kwbad.py",
    );
    assert!(
        out.contains("s . split (\",\" , 1)"),
        "the bogus keyword must lower its value positionally: {}",
        out
    );
    // The same for a positional-only builtin: ljust's fillchar keyword
    // lowers positionally instead of erroring on the unknown signature.
    let out = compile(
        "def f(s: str):\n    return s.ljust(5, fillchar=\".\")\n",
        "kwljust.py",
    );
    assert!(
        out.contains("s . ljust (5 , \".\")"),
        "the fillchar keyword must lower positionally: {}",
        out
    );
}

// ---- str.format ----

#[test]
fn str_format_lowers_to_format_macro() {
    let out = compile(
        "def f(a: int, b: str) -> str:\n    return \"{} and {}\".format(a, b)\n",
        "fmt1.py",
    );
    assert!(out.contains("format !"), "generated: {}", out);
    assert!(out.contains("__rython_fmt0"), "generated: {}", out);

    // Positional reuse, keywords, and specs translate.
    let out = compile(
        "def f(x: float) -> str:\n    return \"{0} {0} {v:.2f}\".format(x, v=x)\n",
        "fmt2.py",
    );
    assert!(out.contains("__rython_fmt_v"), "generated: {}", out);
}

#[test]
fn str_format_errors_are_loud_or_lower_to_variants() {
    // Mixing auto and manual numbering is Python's ValueError.
    let err = compile_err(
        "def f(a: int, b: int) -> str:\n    return \"{} {1}\".format(a, b)\n",
        "fmtmix.py",
    );
    assert!(err.contains("automatic field numbering"), "error: {}", err);

    // A template name with no matching keyword.
    let err = compile_err(
        "def f() -> str:\n    return \"{missing}\".format(present=1)\n",
        "fmtname.py",
    );
    assert!(err.contains("missing"), "error: {}", err);

    // The thousands separator now lowers through py_grouped_int instead of
    // being rejected.
    let out = compile(
        "def f(x: int) -> str:\n    return \"{:,}\".format(x)\n",
        "fmtgroup.py",
    );
    assert!(
        out.contains("py_grouped_int"),
        "the thousands separator must lower through py_grouped_int: {}",
        out
    );

    // Non-literal templates can't be checked at conversion time: the
    // dynamic-format divergence — the call is dropped and -W reports it.
    let (out, warnings) = compile_with_warnings(
        "def f(t: str, x: int) -> str:\n    return t.format(x)\n",
        "fmtdyn.py",
    );
    assert!(
        out.contains("stdpython :: PyValue :: None_"),
        "the dynamic format must drop to a no-op: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("non-literal template") && w.contains("dropped")),
        "the dynamic-format divergence must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn fstring_specs_translate_or_error_loudly() {
    let out = compile(
        "def f(n: int) -> str:\n    return f\"{n:05d}|{n:>4}\"\n",
        "fspec.py",
    );
    assert!(out.contains("{:05}"), "generated: {}", out);
    assert!(out.contains("{:>4}"), "generated: {}", out);

    // The old behavior silently fell back to {} for unsupported specs;
    // now they fail at conversion time.
    let err = compile_err(
        "def f(x: float) -> str:\n    return f\"{x:e}\"\n",
        "fspecbad.py",
    );
    assert!(err.contains("presentation type"), "error: {}", err);
}

#[test]
fn repr_conversion_keeps_its_format_spec() {
    // "{0!r:>10}" pads the repr — the spec must not be dropped.
    let out = compile(
        "def f(n: int) -> str:\n    return \"{0!r:>10}\".format(n)\n",
        "reprspec.py",
    );
    assert!(out.contains(":>10}"), "generated: {}", out);
    assert!(out.contains("repr ("), "generated: {}", out);

    let out = compile(
        "def f(n: int) -> str:\n    return f\"{n!r:>10}\"\n",
        "freprspec.py",
    );
    assert!(out.contains(":>10}"), "generated: {}", out);
    assert!(out.contains("repr ("), "generated: {}", out);

    // Numeric presentation types on a repr are Python errors; loud here.
    let err = compile_err(
        "def f(n: int) -> str:\n    return \"{0!r:.2f}\".format(n)\n",
        "reprbad.py",
    );
    assert!(err.contains("cannot combine"), "error: {}", err);
}

#[test]
fn bare_precision_without_type_errors_loudly() {
    // Python's "{:.3}" on a float is GENERAL format (significant figures,
    // possibly scientific); Rust's is fixed decimals. Unknowable operand
    // type means loud rejection, pointing at .Ns / .Nf.
    let err = compile_err(
        "def f(x: float) -> str:\n    return \"{:.3}\".format(x)\n",
        "barep.py",
    );
    assert!(
        err.contains("presentation type is ambiguous"),
        "error: {}",
        err
    );
    let err = compile_err(
        "def f(x: float) -> str:\n    return f\"{x:.3}\"\n",
        "barepf.py",
    );
    assert!(
        err.contains("presentation type is ambiguous"),
        "error: {}",
        err
    );
}

// ---- Module-level globals and entry points ----

#[test]
fn module_constants_lower_to_statics() {
    let out = compile(
        concat!(
            "PI = 3.14159\n",
            "GREETING = \"hello\"\n",
            "DEBUG = True\n",
            "OFFSET = -3\n",
            "\n",
            "def area(r: float) -> float:\n",
            "    return PI * r * r\n",
        ),
        "consts.py",
    );
    assert!(
        out.contains("pub static PI : f64 = 3.14159"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("pub static GREETING : & 'static str = \"hello\""),
        "generated: {}",
        out
    );
    assert!(
        out.contains("pub static DEBUG : bool = true"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("pub static OFFSET : i64 = - 3"),
        "generated: {}",
        out
    );

    // A reassigned module name is NOT a constant; it keeps the old
    // module-init lowering.
    let out = compile("X = 1\nX = 2\n", "reassigned.py");
    assert!(!out.contains("pub static X"), "generated: {}", out);
}

#[test]
fn value_returning_main_gets_a_wrapper_entry_point() {
    // `def main() -> int` cannot be the Rust entry point (Result<i64, _>
    // does not implement Termination); the wrapper discards the value like
    // Python's `if __name__: main()` does.
    let out = compile(
        concat!(
            "def main() -> int:\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
        "intmain.py",
    );
    assert!(out.contains("fn python_main ()"), "generated: {}", out);
    assert!(
        out.contains("fn main () {"),
        "wrapper entry point expected: {}",
        out
    );
}

#[test]
fn integral_float_literals_keep_their_float_type() {
    // 2.0 must stay a float literal: Rust's Display drops the ".0" and the
    // re-parse would silently produce an integer (2.0 / 4 is 0.5 in
    // Python, but 2 / 4 as integers is 0).
    let out = compile("def f() -> float:\n    y = 2.0\n    return y\n", "flit.py");
    assert!(out.contains("y = 2.0"), "generated: {}", out);
    assert!(!out.contains("y = 2 ;"), "generated: {}", out);
}

#[test]
fn conditionally_reassigned_module_names_are_not_constants() {
    // DEBUG = False overwritten inside a module-level `if` must NOT freeze
    // as a static: the nested store would land on a shadowing local inside
    // __module_init__ while functions read the stale static.
    let out = compile(
        "DEBUG = False\nif 1 > 0:\n    DEBUG = True\n",
        "condglobal.py",
    );
    assert!(!out.contains("pub static DEBUG"), "generated: {}", out);

    // A for-loop target at module level is rebound each iteration.
    let out = compile("I = 0\nfor I in [1, 2]:\n    pass\n", "forglobal.py");
    assert!(!out.contains("pub static I"), "generated: {}", out);

    // Reassignment inside a module-level try body.
    let out = compile(
        "MODE = \"a\"\ntry:\n    MODE = \"b\"\nexcept ValueError:\n    pass\n",
        "tryglobal.py",
    );
    assert!(!out.contains("pub static MODE"), "generated: {}", out);
}

#[test]
fn sibling_imported_module_values_promote_to_statics() {
    // charset_normalizer/constant.py: `TOO_BIG_SEQUENCE = int(10e6)` (a
    // NON-const value, so it needs the LazyLock promotion, not the
    // literal-static path) is imported by sibling modules. The DEFINING
    // module must promote it to a `pub static` LazyLock (a module-init
    // local is invisible to other modules — E0432), and the IMPORTING
    // module's reads must deref-clone (`(*TOO_BIG_SEQUENCE).clone()`).
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        vec!["pkg".to_string(), "constant".to_string()],
        std::rc::Rc::new(
            parse("TOO_BIG_SEQUENCE = int(10e6)\n", "constant.py").unwrap(),
        ),
    );
    defs.insert(
        vec!["pkg".to_string(), "utils".to_string()],
        std::rc::Rc::new(
            parse(
                "from .constant import TOO_BIG_SEQUENCE\n\ndef f() -> int:\n    return TOO_BIG_SEQUENCE\n",
                "utils.py",
            )
            .unwrap(),
        ),
    );
    let mut options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        module_path: vec!["pkg".to_string()],
        ..Default::default()
    };
    // Convert the DEFINING module first: the promotion pass records the
    // promoted set into the shared cache that the importing module consults.
    options.this_module_path = vec!["pkg".to_string(), "constant".to_string()];
    let constant_out = compile_with_options(
        "TOO_BIG_SEQUENCE = int(10e6)\n",
        "constant.py",
        options.clone(),
    )
    .expect("constant module converts");
    assert!(
        constant_out.contains("pub static TOO_BIG_SEQUENCE"),
        "sibling-imported value must be a static in the defining module: {}",
        constant_out
    );

    options.this_module_path = vec!["pkg".to_string(), "utils".to_string()];
    let utils_out = compile_with_options(
        "from .constant import TOO_BIG_SEQUENCE\n\ndef f() -> int:\n    return TOO_BIG_SEQUENCE\n",
        "utils.py",
        options,
    )
    .expect("utils module converts");
    assert!(
        utils_out.contains("use crate :: pkg :: constant :: TOO_BIG_SEQUENCE"),
        "import must resolve to the static: {}",
        utils_out
    );
    assert!(
        utils_out.contains("(* TOO_BIG_SEQUENCE) . clone ()"),
        "read of the imported static must deref-clone: {}",
        utils_out
    );
}

#[test]
fn bitwise_module_constant_binop_lowers_to_plain_static() {
    // `_THAI = 1 << 6` is a constant expression: it lowers to a plain
    // `pub static` (importable directly, no LazyLock needed) — the
    // charset_normalizer flag constants.
    let out = compile("_THAI = 1 << 6\n", "flags.py");
    assert!(
        out.contains("pub static _THAI : i64 = (1) << (6)"),
        "generated: {}",
        out
    );
}


// ---------------------------------------------------------------------------
// no_std profile: OS-facing constructs fail at conversion time
// ---------------------------------------------------------------------------

fn compile_nostd(src: &str, name: &str) -> Result<String, String> {
    let module = parse(src, name).unwrap_or_else(|e| panic!("parse failed: {}", e));
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let options = PythonOptions {
        no_std: true,
        ..Default::default()
    };
    module
        .to_rust(
            CodeGenContext::Module(name.replace(".py", "")),
            options,
            symbols,
        )
        .map(|tokens| tokens.to_string())
        .map_err(|e| python_ast::format_error_chain(e.as_ref()))
}

#[test]
fn nostd_modules_carry_an_alloc_prelude() {
    // Under #![no_std] the prelude has no String/Vec/format!; every module
    // brings the alloc surface generated code leans on into scope itself.
    let out = compile_nostd("def f(n: int) -> str:\n    return f\"n={n}\"\n", "np.py")
        .expect("OS-free module must convert");
    assert!(out.contains("extern crate alloc"), "generated: {}", out);
    assert!(out.contains("use alloc ::"), "generated: {}", out);

    // The std profile stays exactly as before: no alloc plumbing.
    let std_out = compile("def f(n: int) -> str:\n    return f\"n={n}\"\n", "sp.py");
    assert!(
        !std_out.contains("extern crate alloc"),
        "generated: {}",
        std_out
    );
}

#[test]
fn nostd_io_builtins_error_loudly() {
    for src in ["print(\"hi\")\n", "x = input()\n", "f = open(\"a.txt\")\n"] {
        let err = compile_nostd(src, "io.py").expect_err("I/O builtin must fail");
        assert!(err.contains("no_std profile"), "{:?}: {}", src, err);
    }

    // A user definition shadows the builtin as usual and stays convertible.
    let out = compile_nostd(
        "def print(s: str) -> str:\n    return s\n\ndef f() -> str:\n    return print(\"x\")\n",
        "shadow.py",
    )
    .expect("shadowed print is the user's own function");
    assert!(out.contains("fn print"), "generated: {}", out);
}

#[test]
fn nostd_std_tier_imports_error_loudly() {
    for src in [
        "import os\n",
        "import sys\n",
        "from datetime import datetime\n",
        "import math\n",
        "from os.path import join\n",
    ] {
        let err = compile_nostd(src, "imp.py").expect_err("std-tier import must fail");
        assert!(err.contains("std tier"), "{:?}: {}", src, err);
    }

    // alloc-tier runtime modules stay importable.
    for src in [
        "import json\n",
        "import collections\n",
        "import itertools\n",
    ] {
        compile_nostd(src, "ok.py")
            .unwrap_or_else(|e| panic!("alloc-tier import must convert: {:?}: {}", src, e));
    }
}

#[test]
fn nostd_main_blocks_error_loudly() {
    let err = compile_nostd(
        "def main() -> int:\n    return 0\n\nif __name__ == \"__main__\":\n    main()\n",
        "entry.py",
    )
    .expect_err("__main__ needs a process entry point");
    assert!(err.contains("no_std profile"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// Builtin lowering: min/max/sorted/enumerate/pow/len/repr/reversed
// ---------------------------------------------------------------------------

#[test]
fn min_max_lower_to_variant_functions_with_exception_propagation() {
    // Single-iterable form raises on empty, so it propagates with `?`.
    let out = compile(
        "def f(xs: list[int]) -> int:\n    return min(xs)\n",
        "m1.py",
    );
    assert!(out.contains("min (& (xs)) ?"), "generated: {}", out);

    // Two and three scalar arguments fold pairwise.
    let out = compile(
        "def f(a: int, b: int) -> int:\n    return max(a, b)\n",
        "m2.py",
    );
    assert!(out.contains("max2 (a , b)"), "generated: {}", out);
    let out = compile(
        "def f(a: int, b: int, c: int) -> int:\n    return min(a, b, c)\n",
        "m3.py",
    );
    assert!(
        out.contains("min2 (min2 (a , b) , c)"),
        "generated: {}",
        out
    );

    // default= never raises; key= does.
    let out = compile(
        "def f(xs: list[int]) -> int:\n    return min(xs, default=7)\n",
        "m4.py",
    );
    assert!(
        out.contains("min_default (& (xs) , 7)"),
        "generated: {}",
        out
    );
    let out = compile(
        "def f(xs: list[int]) -> int:\n    return max(xs, key=lambda x: -x)\n",
        "m5.py",
    );
    assert!(out.contains("max_key (& (xs) ,"), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);

    // Unknown keywords stay loud.
    let err = compile_err("x = min([1], foo=2)\n", "m6.py");
    assert!(err.contains("unexpected"), "error: {}", err);
}

#[test]
fn sorted_lowers_by_keyword_combination() {
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return sorted(xs)\n",
        "s1.py",
    );
    assert!(out.contains("sorted (& (xs))"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return sorted(xs, reverse=True)\n",
        "s2.py",
    );
    assert!(
        out.contains("sorted_reverse (& (xs) , true)"),
        "generated: {}",
        out
    );
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return sorted(xs, key=lambda x: -x)\n",
        "s3.py",
    );
    assert!(out.contains("sorted_key (& (xs) ,"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return sorted(xs, key=lambda x: -x, reverse=True)\n",
        "s4.py",
    );
    assert!(
        out.contains("sorted_key_reverse (& (xs) ,"),
        "generated: {}",
        out
    );
}

#[test]
fn enumerate_start_and_pow_arities_lower_to_their_variants() {
    let out = compile(
        "for i, x in enumerate([10, 20], start=5):\n    pass\n",
        "e1.py",
    );
    assert!(out.contains("enumerate_start ("), "generated: {}", out);
    let out = compile("for i, x in enumerate([10]):\n    pass\n", "e2.py");
    assert!(out.contains("enumerate ("), "generated: {}", out);
    assert!(!out.contains("enumerate_start"), "generated: {}", out);

    let out = compile("y = pow(2, 5)\n", "p1.py");
    assert!(out.contains("pow (2 , 5)"), "generated: {}", out);
    let out = compile("y = pow(2, 5, 7)\n", "p2.py");
    assert!(out.contains("pow_mod (2 , 5 , 7) ?"), "generated: {}", out);
}

#[test]
fn by_reference_builtins_borrow_their_argument() {
    // len/repr/reversed take references at the runtime layer; Python's
    // calls never consume the value.
    let out = compile(
        "def f(xs: list[int]) -> int:\n    return len(xs)\n",
        "b1.py",
    );
    assert!(out.contains("len (& (xs))"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> str:\n    return repr(xs)\n",
        "b2.py",
    );
    assert!(out.contains("repr (& (xs))"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return reversed(xs)\n",
        "b3.py",
    );
    assert!(out.contains("reversed (& (xs))"), "generated: {}", out);

    // A user-defined function of the same name shadows the builtin shape.
    let out = compile(
        "def len(x: int) -> int:\n    return x\n\ndef g(v: int) -> int:\n    return len(v)\n",
        "b4.py",
    );
    assert!(out.contains("len (v)"), "generated: {}", out);
}

// ---------------------------------------------------------------------------
// datetime constructors, strptime, and runtime-module imports
// ---------------------------------------------------------------------------

#[test]
fn datetime_constructors_map_keywords_onto_new() {
    let out = compile(
        "from datetime import timedelta\ntd = timedelta(days=1, hours=2)\n",
        "td.py",
    );
    assert!(
        out.contains("timedelta :: new (Some (1) , None , None , None , None , Some (2) , None)"),
        "generated: {}",
        out
    );
    let out = compile("from datetime import date\nd = date(2024, 3, 1)\n", "d.py");
    assert!(
        out.contains("date :: new (2024 , 3 , 1) ?"),
        "generated: {}",
        out
    );
    let out = compile(
        "from datetime import datetime\ndt = datetime(2024, 3, 1, hour=10)\n",
        "dt.py",
    );
    assert!(
        out.contains("datetime :: new (2024 , 3 , 1 , Some (10) , None , None , None) ?"),
        "generated: {}",
        out
    );

    // Unknown keywords and missing required arguments stay loud.
    let err = compile_err(
        "from datetime import timedelta\ntd = timedelta(fortnights=1)\n",
        "tde.py",
    );
    assert!(err.contains("unexpected keyword"), "error: {}", err);
    let err = compile_err("from datetime import date\nd = date(2024)\n", "de.py");
    assert!(err.contains("missing required argument"), "error: {}", err);
}

#[test]
fn strptime_and_module_attribute_calls_lower_to_paths() {
    let out = compile(
        "from datetime import datetime\ndt = datetime.strptime(\"x\", \"%Y\")\n",
        "sp.py",
    );
    assert!(
        out.contains("datetime :: strptime (\"x\" , \"%Y\") ?"),
        "generated: {}",
        out
    );
    let out = compile("import time\nt = time.monotonic()\n", "tm.py");
    assert!(out.contains("time :: monotonic ()"), "generated: {}", out);
}

#[test]
fn runtime_module_imports_lower_to_nothing_and_aliases_resolve() {
    // The modules are already in scope via `use stdpython::*`; a bare
    // `use math;` would not even resolve.
    let out = compile("import math\nimport random\n", "imp.py");
    assert!(!out.contains("use math"), "generated: {}", out);
    assert!(!out.contains("use random"), "generated: {}", out);

    // An aliased runtime module resolves through the alias (`import time
    // as t` → `t::monotonic()`): the alias is a module intercept, not a
    // user variable.
    let out = compile("import time as t\nx = t.monotonic()\n", "alias.py");
    assert!(
        out.contains("t :: monotonic ()"),
        "the aliased module call must resolve: {}",
        out
    );
}

// ---------------------------------------------------------------------------
// itertools lowering: keyword variants and by-reference iterables
// ---------------------------------------------------------------------------

#[test]
fn itertools_keyword_spellings_lower_to_variants() {
    let base = "from itertools import accumulate, product, zip_longest, groupby\n";
    let out = compile(&format!("{}a = accumulate([1, 2])\n", base), "i1.py");
    assert!(
        out.contains("accumulate_sum (& (vec ! [1 , 2]))"),
        "generated: {}",
        out
    );
    let out = compile(
        &format!("{}a = accumulate([1, 2], initial=10)\n", base),
        "i2.py",
    );
    assert!(
        out.contains("accumulate_sum_initial ("),
        "generated: {}",
        out
    );
    let out = compile(
        &format!("{}a = accumulate([1, 2], lambda x, y: x * y)\n", base),
        "i3.py",
    );
    assert!(out.contains("accumulate_func ("), "generated: {}", out);

    let out = compile(&format!("{}p = product([1], [2])\n", base), "i4.py");
    assert!(out.contains("product2 ("), "generated: {}", out);
    let out = compile(&format!("{}p = product([1], repeat=2)\n", base), "i5.py");
    assert!(out.contains("product_repeat2 ("), "generated: {}", out);
    // repeat must be a literal arity — tuple width is a compile-time shape.
    let err = compile_err(&format!("{}p = product([1], repeat=5)\n", base), "i6.py");
    assert!(err.contains("literal 2 or 3"), "error: {}", err);

    let out = compile(
        &format!("{}z = zip_longest([1], [2], fillvalue=0)\n", base),
        "i7.py",
    );
    assert!(out.contains("zip_longest_fill ("), "generated: {}", out);
    let out = compile(
        &format!("{}g = groupby([1], key=lambda x: x)\n", base),
        "i8.py",
    );
    assert!(out.contains("groupby_key ("), "generated: {}", out);

    // Unknown keywords stay loud.
    let err = compile_err(&format!("{}g = groupby([1], foo=1)\n", base), "i9.py");
    assert!(err.contains("unexpected"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// functools/heapq/copy/textwrap lowering, and mutating methods on
// subscripted receivers
// ---------------------------------------------------------------------------

#[test]
fn pure_module_calls_lower_with_borrows_and_arity_variants() {
    let out = compile(
        "from functools import reduce\nr = reduce(lambda a, b: a + b, [1, 2])\n",
        "f1.py",
    );
    assert!(out.contains("reduce ("), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);
    let out = compile(
        "from functools import reduce\nr = reduce(lambda a, b: a + b, [1, 2], 10)\n",
        "f2.py",
    );
    assert!(out.contains("reduce_initial ("), "generated: {}", out);

    // heapq mutates its first argument: &mut lowering and a mut binding.
    let out = compile(
        "from heapq import heappush, heappop\nh = [3, 1]\nheappush(h, 2)\nx = heappop(h)\n",
        "h1.py",
    );
    assert!(
        out.contains("heappush (& mut (h) , 2)"),
        "generated: {}",
        out
    );
    assert!(out.contains("heappop (& mut (h)) ?"), "generated: {}", out);
    assert!(
        out.contains("let mut h"),
        "heap binding must be mut: {}",
        out
    );

    // Module-attribute spelling lowers to the same shapes AND marks the
    // heap binding mutable (Devin review on #53: only the bare-function
    // spelling used to).
    let out = compile("import heapq\nh = [2, 1]\nheapq.heapify(h)\n", "h2.py");
    assert!(
        out.contains("heapq :: heapify (& mut (h))"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("let mut h"),
        "heap binding must be mut: {}",
        out
    );

    let out = compile("from copy import deepcopy\nc = deepcopy([1])\n", "c1.py");
    assert!(out.contains("deepcopy (& ("), "generated: {}", out);
    let out = compile(
        "from textwrap import indent\ns = indent(\"a\", \"> \")\n",
        "t1.py",
    );
    assert!(
        out.contains("indent (& (\"a\") , & (\"> \"))"),
        "generated: {}",
        out
    );
}

#[test]
fn mutating_methods_on_subscripted_receivers_use_the_place_lowering() {
    // xs[0].append(v) must mutate the real element: the Load lowering
    // (py_index) yields a clone and the write would silently vanish.
    let out = compile("xs = [[1], [2]]\nxs[0].append(9)\n", "sub1.py");
    assert!(
        out.contains("py_index_mut (0) ?) . push (9)"),
        "generated: {}",
        out
    );
    // Read-only methods keep the Load lowering.
    let out = compile("xs = [[1]]\nn = xs[0].count(1)\n", "sub2.py");
    assert!(!out.contains("py_index_mut"), "generated: {}", out);

    // The heapq mutators' heap argument is a place too: heappush(rows[i], v)
    // through the Load path would push into a clone.
    let out = compile(
        "from heapq import heappush\nrows = [[1], [2]]\nheappush(rows[0], 5)\n",
        "sub3.py",
    );
    assert!(
        out.contains("heappush ((rows) . py_index_mut (0) ? , 5)"),
        "generated: {}",
        out
    );
}

// ---------------------------------------------------------------------------
// re module lowering
// ---------------------------------------------------------------------------

#[test]
fn re_calls_lower_to_borrowing_fallible_paths() {
    let out = compile("import re\nm = re.search(r\"\\d\", \"a1\")\n", "r1.py");
    assert!(
        out.contains("re :: search (& (\"\\\\d\") , & (\"a1\") , \"\") ?"),
        "generated: {}",
        out
    );
    // `match` is a Rust keyword: the runtime function is r#match.
    let out = compile("import re\nm = re.match(r\"\\d\", \"1\")\n", "r2.py");
    assert!(out.contains("re :: r#match ("), "generated: {}", out);
    let out = compile("import re\ns = re.sub(r\"a\", \"b\", \"aa\")\n", "r3.py");
    assert!(out.contains("re :: sub ("), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);
    // m.group() lowers to group(0).
    let out = compile(
        "import re\nm = re.search(r\"a\", \"a\")\ng = m.group()\n",
        "r4.py",
    );
    assert!(out.contains(". group (0)"), "generated: {}", out);
    // from-import spelling, including the keyword-name function.
    let out = compile(
        "from re import findall, match\nxs = findall(r\"a\", \"aa\")\nm = match(r\"a\", \"ab\")\n",
        "r5.py",
    );
    assert!(out.contains("findall (& ("), "generated: {}", out);
    assert!(out.contains("r#match (& ("), "generated: {}", out);
    // Flags lower to inline flag letters; unknown flags are loud.
    let out = compile(
        "import re\nxs = re.findall(r\"a\", \"A\", re.IGNORECASE)\n",
        "r6.py",
    );
    assert!(out.contains("\"i\") ?"), "generated: {}", out);
    let out = compile(
        "import re\nxs = re.findall(r\"a\", \"A\", flags=re.IGNORECASE | re.MULTILINE)\n",
        "r7.py",
    );
    assert!(out.contains("\"im\") ?"), "generated: {}", out);
    let out = compile(
        "import re\ns = re.sub(r\"a\", \"b\", \"aa\", count=1)\n",
        "r8.py",
    );
    assert!(out.contains(", 1 , \"\") ?"), "generated: {}", out);
    let err = compile_err(
        "import re\nxs = re.findall(r\"a\", \"A\", re.VERBOSE)\n",
        "r9.py",
    );
    assert!(err.contains("unsupported re flag"), "error: {}", err);
    // split's THIRD positional is maxsplit (not flags, unlike the rest).
    let out = compile("import re\nxs = re.split(r\"a\", \"b\", 1)\n", "r10.py");
    assert!(
        out.contains("re :: split (& (\"a\") , & (\"b\") , 1 , \"\") ?"),
        "generated: {}",
        out
    );
    let out = compile(
        "import re\nxs = re.split(r\"a\", \"b\", maxsplit=2, flags=re.IGNORECASE)\n",
        "r11.py",
    );
    assert!(out.contains(", 2 , \"i\") ?"), "generated: {}", out);
    // Surplus positionals are loud, not silently dropped.
    let err = compile_err(
        "import re\nm = re.search(r\"a\", \"b\", re.IGNORECASE, 5)\n",
        "r12.py",
    );
    assert!(err.contains("at most 3"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// map/filter/list lowering
// ---------------------------------------------------------------------------

#[test]
fn map_filter_dispatch_on_the_function_arguments_shape() {
    // Lambdas are plain closures.
    let out = compile("ys = list(map(lambda x: x * 2, [1, 2]))\n", "mf1.py");
    assert!(out.contains("list (map (| x |"), "generated: {}", out);
    assert!(!out.contains("map_fallible"), "generated: {}", out);

    // User-defined functions return Result: the fallible variant + `?`.
    let out = compile(
        "def double(n: int) -> int:\n    return n * 2\n\nys = list(map(double, [1, 2]))\n",
        "mf2.py",
    );
    assert!(out.contains("map_fallible (double ,"), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);

    let out = compile("ys = filter(lambda x: x > 1, [1, 2, 3])\n", "mf3.py");
    assert!(out.contains("filter (| x |"), "generated: {}", out);
    // filter(None, xs) keeps truthy elements.
    let out = compile("ys = filter(None, [0, 1, 2])\n", "mf4.py");
    assert!(out.contains("filter_truthy ("), "generated: {}", out);

    // list() with no argument has no inferable type: loud.
    let err = compile_err("ys = list()\n", "mf5.py");
    assert!(err.contains("iterable argument"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// hashlib lowering and str.encode()
// ---------------------------------------------------------------------------

#[test]
fn hashlib_and_encode_lower_correctly() {
    let out = compile(
        "import hashlib\nh = hashlib.sha256(\"x\".encode())\n",
        "hl1.py",
    );
    assert!(
        out.contains("hashlib :: sha256 (& ((\"x\") . as_bytes () . to_vec ()))"),
        "generated: {}",
        out
    );
    // Zero-arg constructors map to the _new variants for the update idiom.
    let out = compile("from hashlib import sha256\nh = sha256()\n", "hl2.py");
    assert!(out.contains("sha256_new ()"), "generated: {}", out);
    // The latin-1 codec is supported now (codec::encode_latin1).
    let out = compile("s = \"x\".encode(\"latin-1\")\n", "hl3.py");
    assert!(
        out.contains("encode_latin1"),
        "latin-1 must lower through the codec: {}",
        out
    );
    // Encodings outside the supported set stay loud.
    let err = compile_err("s = \"x\".encode(\"utf-16\")\n", "hl4.py");
    assert!(err.contains("utf-8"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// textwrap.wrap/fill lowering
// ---------------------------------------------------------------------------

#[test]
fn wrap_and_fill_lower_with_width_defaults() {
    let out = compile("from textwrap import wrap\nxs = wrap(\"a b\")\n", "w1.py");
    assert!(
        out.contains("wrap (& (\"a b\") , 70) ?"),
        "generated: {}",
        out
    );
    let out = compile(
        "from textwrap import fill\ns = fill(\"a b\", width=9)\n",
        "w2.py",
    );
    assert!(
        out.contains("fill (& (\"a b\") , 9) ?"),
        "generated: {}",
        out
    );
    let out = compile(
        "import textwrap\nxs = textwrap.wrap(\"a b\", 12)\n",
        "w3.py",
    );
    assert!(
        out.contains("textwrap :: wrap (& (\"a b\") , 12) ?"),
        "generated: {}",
        out
    );
    // Unsupported options stay loud.
    let err = compile_err(
        "from textwrap import wrap\nxs = wrap(\"a\", initial_indent=\"> \")\n",
        "w4.py",
    );
    assert!(err.contains("unexpected keyword"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// isinstance (static constant) and hash lowering
// ---------------------------------------------------------------------------

#[test]
fn isinstance_lowers_to_a_static_constant() {
    // Annotated parameters decide at conversion time.
    let out = compile(
        "def f(n: int) -> bool:\n    return isinstance(n, int)\n",
        "is1.py",
    );
    assert!(
        out.contains("return Ok (true)") || out.contains("true"),
        "generated: {}",
        out
    );
    let out = compile(
        "def f(n: int) -> bool:\n    return isinstance(n, str)\n",
        "is2.py",
    );
    assert!(out.contains("false"), "generated: {}", out);
    // Literal-assigned locals count; bool is a subclass of int.
    let out = compile(
        "def f() -> bool:\n    x = 1.5\n    return isinstance(x, float)\n",
        "is3.py",
    );
    assert!(out.contains("true"), "generated: {}", out);
    let out = compile(
        "def f(b: bool) -> bool:\n    return isinstance(b, int)\n",
        "is4.py",
    );
    assert!(out.contains("true"), "generated: {}", out);
    let out = compile(
        "def f(n: int) -> bool:\n    return isinstance(n, bool)\n",
        "is5.py",
    );
    assert!(out.contains("false"), "generated: {}", out);

    // An unannotated parameter is a generic type variable: no known type
    // satisfies the check, so isinstance lowers to the static constant
    // false (the class-as-value divergence) instead of failing.
    let out = compile("def f(v):\n    return isinstance(v, int)\n", "is6.py");
    assert!(out.contains("false"), "generated: {}", out);
    assert!(!out.contains("true"), "generated: {}", out);
}

#[test]
fn hash_lowers_by_reference() {
    let out = compile("h = hash(\"a\")\n", "hs1.py");
    assert!(out.contains("hash (& (\"a\"))"), "generated: {}", out);
}

// ---------------------------------------------------------------------------
// csv lowering
// ---------------------------------------------------------------------------

#[test]
fn csv_reader_lowers_by_reference() {
    let out = compile("import csv\nrows = csv.reader([\"a,b\"])\n", "cv1.py");
    assert!(out.contains("csv :: reader (& ("), "generated: {}", out);
    let out = compile(
        "from csv import reader\nrows = reader([\"a,b\"])\n",
        "cv2.py",
    );
    assert!(out.contains("reader (& ("), "generated: {}", out);
}

// ---- print and list.sort ----

#[test]
fn print_multi_arg_renders_through_py_display() {
    // Multi-argument print pre-renders each argument with py_display
    // (Python str semantics) and joins with the default sep/end.
    let out = compile("def f(x: int, s: str):\n    print(x, s)\n", "pr1.py");
    assert!(
        out.contains("print_parts (& [py_display (& (x)) , py_display (& (s))] , \" \" , \"\\n\")"),
        "generated: {}",
        out
    );
}

#[test]
fn print_sep_end_flush_keywords_map() {
    let out = compile(
        "def f(a: int, b: int):\n    print(a, b, sep='-', end='!')\n",
        "pr2.py",
    );
    assert!(
        out.contains("print_parts (& [py_display (& (a)) , py_display (& (b))] , \"-\" , \"!\")"),
        "generated: {}",
        out
    );

    // flush= routes to the flushing variant; sep=None means default.
    let out = compile(
        "def f(a: int):\n    print(a, sep=None, flush=True)\n",
        "pr3.py",
    );
    assert!(
        out.contains("print_parts_flush (& [py_display (& (a))] , \" \" , \"\\n\" , true)"),
        "generated: {}",
        out
    );
}

#[test]
fn print_zero_and_single_arg_shapes() {
    let out = compile("def f():\n    print()\n", "pr4.py");
    assert!(out.contains("println ! ()"), "generated: {}", out);

    // print(end="") with no arguments still needs a typed empty slice.
    let out = compile("def f():\n    print(end='')\n", "pr5.py");
    assert!(
        out.contains("print_parts (& [] as & [& str] , \" \" , \"\")"),
        "generated: {}",
        out
    );

    let out = compile("def f(x: int):\n    print(x)\n", "pr6.py");
    assert!(out.contains("print (& (x))"), "generated: {}", out);
}

#[test]
fn print_file_keyword_is_a_loud_error() {
    let err = compile_err(
        "import sys\n\ndef f():\n    print('x', file=sys.stderr)\n",
        "pr7.py",
    );
    assert!(err.contains("file"), "error: {}", err);
}

#[test]
fn list_sort_maps_keyword_shapes_in_place() {
    let out = compile("def f(xs: list[int]):\n    xs.sort()\n", "srt1.py");
    assert!(out.contains("(xs) . py_sort ()"), "generated: {}", out);

    let out = compile(
        "def f(xs: list[int]):\n    xs.sort(reverse=True)\n",
        "srt2.py",
    );
    assert!(
        out.contains("(xs) . py_sort_reverse (true)"),
        "generated: {}",
        out
    );

    let out = compile(
        "def f(xs: list[str]):\n    xs.sort(key=lambda w: len(w))\n",
        "srt3.py",
    );
    assert!(out.contains("py_sort_key"), "generated: {}", out);

    let out = compile(
        "def f(xs: list[str]):\n    xs.sort(key=lambda w: len(w), reverse=True)\n",
        "srt4.py",
    );
    assert!(out.contains("py_sort_key_reverse"), "generated: {}", out);
}

#[test]
fn list_sort_on_subscript_uses_place_lowering() {
    // grid[0].sort() must mutate the real element, not a py_index clone.
    let out = compile(
        "def f(grid: list[list[int]]):\n    grid[0].sort()\n",
        "srt5.py",
    );
    assert!(out.contains("py_index_mut"), "generated: {}", out);
    assert!(out.contains("py_sort"), "generated: {}", out);
}

#[test]
fn list_sort_positional_arg_is_a_loud_error() {
    // Python: TypeError: sort() takes no positional arguments.
    let err = compile_err("def f(xs: list[int]):\n    xs.sort(True)\n", "srt6.py");
    assert!(err.contains("no positional arguments"), "error: {}", err);
}

// ---- re named groups and findall tuple shapes ----

#[test]
fn findall_picks_variant_from_literal_group_count() {
    let src = "import re\n\ndef f(s: str):\n    return re.findall(r\"(\\w+)=(\\d+)\", s)\n";
    let out = compile(src, "fa2.py");
    assert!(out.contains("findall2"), "generated: {}", out);

    let src = "import re\n\ndef f(s: str):\n    return re.findall(r\"(\\d+)-(\\d+)-(\\d+)\", s)\n";
    let out = compile(src, "fa3.py");
    assert!(out.contains("findall3"), "generated: {}", out);

    // 0 or 1 group keeps the string-shaped findall.
    let src = "import re\n\ndef f(s: str):\n    return re.findall(r\"\\d+\", s)\n";
    let out = compile(src, "fa1.py");
    assert!(out.contains("findall ("), "generated: {}", out);
    assert!(!out.contains("findall2"), "generated: {}", out);

    // A non-literal pattern can't be counted at conversion time; the
    // string shape (with its loud runtime error for 2+ groups) stays.
    let src = "import re\n\ndef f(p: str, s: str):\n    return re.findall(p, s)\n";
    let out = compile(src, "fa_dyn.py");
    assert!(out.contains("findall ("), "generated: {}", out);
}

#[test]
fn findall_bad_or_wide_literal_patterns_error_at_conversion() {
    let err = compile_err(
        "import re\n\ndef f(s: str):\n    return re.findall(r\"(a)(b)(c)(d)\", s)\n",
        "fa4.py",
    );
    assert!(err.contains("4 capture groups"), "error: {}", err);

    // An invalid literal pattern surfaces at conversion time, not runtime.
    let err = compile_err(
        "import re\n\ndef f(s: str):\n    return re.findall(r\"(unclosed\", s)\n",
        "fa_bad.py",
    );
    assert!(err.contains("cannot compile pattern"), "error: {}", err);
}

#[test]
fn match_group_string_routes_to_group_name() {
    let src = concat!(
        "import re\n",
        "\n",
        "def f(s: str):\n",
        "    m = re.search(r\"(?P<word>\\w+)\", s)\n",
        "    return m.group(\"word\")\n",
    );
    let out = compile(src, "gn1.py");
    assert!(out.contains("group_name (\"word\")"), "generated: {}", out);

    // Numeric group access is untouched.
    let src = concat!(
        "import re\n",
        "\n",
        "def f(s: str):\n",
        "    m = re.search(r\"(\\w+)\", s)\n",
        "    return m.group(1)\n",
    );
    let out = compile(src, "gn2.py");
    assert!(out.contains("group (1)"), "generated: {}", out);
    assert!(!out.contains("group_name"), "generated: {}", out);
}

// ---- replace() with datetime-family keywords ----

#[test]
fn replace_keywords_lower_through_py_replace() {
    let src = concat!(
        "from datetime import datetime\n",
        "\n",
        "def f(d: datetime):\n",
        "    return d.replace(hour=14)\n",
    );
    let out = compile(src, "rep1.py");
    assert!(out.contains("py_replace"), "generated: {}", out);
    assert!(out.contains("hour : Some (14)"), "generated: {}", out);
    assert!(
        out.contains(".. ReplaceArgs :: default ()"),
        "generated: {}",
        out
    );

    // Positional year plus keyword day both map into slots.
    let src = concat!(
        "from datetime import datetime\n",
        "\n",
        "def f(d: datetime):\n",
        "    return d.replace(2023, day=28)\n",
    );
    let out = compile(src, "rep2.py");
    assert!(out.contains("year : Some (2023)"), "generated: {}", out);
    assert!(out.contains("day : Some (28)"), "generated: {}", out);
}

#[test]
fn replace_bad_keywords_drop_the_call_or_stay_loud() {
    // An unknown replace() keyword on an external (unmodeled) object: the
    // external-object divergence — the call is dropped and -W reports it.
    let (out, warnings) = compile_with_warnings(
        "from datetime import datetime\n\ndef f(d: datetime):\n    return d.replace(bogus=1)\n",
        "rep3.py",
    );
    assert!(
        out.contains("stdpython :: PyValue :: None_"),
        "the unknown-keyword replace must drop to a no-op: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("replace(bogus)") && w.contains("is dropped")),
        "the dropped replace must be reported through -W: {:?}",
        warnings
    );

    // A POSITIONAL+keyword collision on replace() stays loud with Python's
    // message.
    let err = compile_err(
        "from datetime import datetime\n\ndef f(d: datetime):\n    return d.replace(2023, year=1)\n",
        "rep4.py",
    );
    assert!(
        err.contains("multiple values for argument 'year'"),
        "error: {}",
        err
    );
}

#[test]
fn str_replace_positional_stays_a_plain_method_call() {
    let out = compile(
        "def f(s: str):\n    return s.replace(\"a\", \"o\")\n",
        "rep5.py",
    );
    assert!(
        out.contains("replace (\"a\" , \"o\")"),
        "generated: {}",
        out
    );
    assert!(!out.contains("py_replace"), "generated: {}", out);
}

// ---- functools.partial over statically-known functions ----

#[test]
fn partial_lowers_to_a_move_closure_with_remaining_params() {
    let src = concat!(
        "from functools import partial\n",
        "\n",
        "def add(a: int, b: int) -> int:\n",
        "    return a + b\n",
        "\n",
        "def f() -> int:\n",
        "    add5 = partial(add, 5)\n",
        "    return add5(3)\n",
    );
    let out = compile(src, "part1.py");
    // The closure binds 5 and keeps the remaining parameter's Python name.
    assert!(out.contains("move | b | add (5 , b)"), "generated: {}", out);
    // Calls through the bound name propagate the function's Result.
    assert!(out.contains("add5 (3) ?"), "generated: {}", out);
    // The import emits no `use` — partial has no runtime symbol.
    assert!(
        !out.contains("use stdpython :: functools :: partial"),
        "generated: {}",
        out
    );

    // Binding ALL parameters leaves a zero-argument closure.
    let src = concat!(
        "from functools import partial\n",
        "\n",
        "def add(a: int, b: int) -> int:\n",
        "    return a + b\n",
        "\n",
        "def f() -> int:\n",
        "    g = partial(add, 2, 3)\n",
        "    return g()\n",
    );
    let out = compile(src, "part2.py");
    assert!(out.contains("move | | add (2 , 3 ,)"), "generated: {}", out);

    // The functools.partial attribute spelling works too.
    let src = concat!(
        "import functools\n",
        "\n",
        "def add(a: int, b: int) -> int:\n",
        "    return a + b\n",
        "\n",
        "def f() -> int:\n",
        "    add5 = functools.partial(add, 5)\n",
        "    return add5(1)\n",
    );
    let out = compile(src, "part3.py");
    assert!(out.contains("move | b | add (5 , b)"), "generated: {}", out);
}

#[test]
fn partial_over_nonlocal_functions_drops_and_overbinding_stays_loud() {
    // partial over a non-local function has no statically-known signature:
    // the callable-as-value divergence (issue #122) — the partial is
    // dropped and -W reports it.
    let (out, warnings) = compile_with_warnings(
        "from functools import partial\n\ndef f():\n    g = partial(unknown_fn, 1)\n",
        "part4.py",
    );
    assert!(
        out.contains("g = stdpython :: PyValue :: None_"),
        "the dropped partial must lower as the boxed None: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("functools.partial over a non-local function") && w.contains("dropped")),
        "the callable-as-value divergence must be reported through -W: {:?}",
        warnings
    );

    // partial with a keyword binding over a LOCAL function still converts
    // (the closure binds the named parameter).
    let out = compile(
        concat!(
            "from functools import partial\n",
            "\n",
            "def add(a: int, b: int) -> int:\n",
            "    return a + b\n",
            "\n",
            "def f():\n",
            "    g = partial(add, b=1)\n",
        ),
        "part5.py",
    );
    assert!(out.contains("fn f"), "generated: {}", out);

    // Overbinding still fails loudly.
    let err = compile_err(
        concat!(
            "from functools import partial\n",
            "\n",
            "def add(a: int, b: int) -> int:\n",
            "    return a + b\n",
            "\n",
            "def f():\n",
            "    g = partial(add, 1, 2, 3)\n",
        ),
        "part6.py",
    );
    assert!(
        err.contains("takes 2 argument(s), but 3 were bound"),
        "error: {}",
        err
    );
}

// ---- file objects, io.StringIO, csv.writer ----

#[test]
fn open_arity_splits_onto_the_option_mode() {
    let out = compile(
        "def f():\n    g = open(\"x.txt\")\n    return g.read()\n",
        "op1.py",
    );
    assert!(
        out.contains("open (& (\"x.txt\") , None :: < & str >) ?"),
        "generated: {}",
        out
    );
    assert!(out.contains(". read () ?"), "generated: {}", out);

    let out = compile(
        "def f():\n    g = open(\"x.txt\", \"w\")\n    g.write(\"hi\")\n",
        "op2.py",
    );
    assert!(
        out.contains("open (& (\"x.txt\") , Some (\"w\")) ?"),
        "generated: {}",
        out
    );
    assert!(out.contains(". write (& (\"hi\")) ?"), "generated: {}", out);
    // The file binding is mutable: write takes &mut self.
    assert!(out.contains("let mut g"), "generated: {}", out);
}

#[test]
fn stringio_and_csv_writer_lower_with_mut_borrows() {
    let src = concat!(
        "import csv\n",
        "import io\n",
        "\n",
        "def f() -> str:\n",
        "    buf = io.StringIO()\n",
        "    w = csv.writer(buf)\n",
        "    w.writerow([\"a\", \"b\"])\n",
        "    w.writerow([])\n",
        "    return buf.getvalue()\n",
    );
    let out = compile(src, "csw1.py");
    assert!(out.contains("io :: StringIO ()"), "generated: {}", out);
    assert!(
        out.contains("csv :: writer (& mut (buf))"),
        "generated: {}",
        out
    );
    assert!(out.contains("let mut buf"), "generated: {}", out);
    assert!(out.contains("let mut w"), "generated: {}", out);
    assert!(
        out.contains(". writerow (& (vec ! [\"a\" . to_string () , \"b\" . to_string ()])) ?")
            || out.contains(". writerow ("),
        "generated: {}",
        out
    );
    // The empty record gets a typed slice.
    assert!(
        out.contains("writerow (& [] as & [& str]) ?"),
        "generated: {}",
        out
    );
    assert!(out.contains(". getvalue () ?"), "generated: {}", out);

    // The seeded StringIO variant.
    let out = compile(
        "import io\n\ndef f() -> str:\n    b = io.StringIO(\"seed\")\n    return b.read()\n",
        "csw2.py",
    );
    assert!(
        out.contains("io :: StringIO_seeded (& (\"seed\"))"),
        "generated: {}",
        out
    );
}

// ---- functools.lru_cache / cache decorators ----

#[test]
fn lru_cache_wraps_the_body_with_a_static_cache() {
    let src = concat!(
        "from functools import lru_cache\n",
        "\n",
        "@lru_cache\n",
        "def fib(n: int) -> int:\n",
        "    if n < 2:\n",
        "        return n\n",
        "    return fib(n - 1) + fib(n - 2)\n",
    );
    let out = compile(src, "lru1.py");
    // Python's bare @lru_cache default is maxsize=128.
    assert!(
        out.contains("PyLruCache :: new (Some (128"),
        "generated: {}",
        out
    );
    assert!(out.contains("__lru_uncached"), "generated: {}", out);
    assert!(out.contains("static __LRU_CACHE"), "generated: {}", out);

    // maxsize=None and functools.cache are unbounded.
    let src = concat!(
        "from functools import lru_cache\n",
        "\n",
        "@lru_cache(maxsize=None)\n",
        "def f(n: int) -> int:\n",
        "    return n\n",
    );
    let out = compile(src, "lru2.py");
    assert!(
        out.contains("PyLruCache :: new (None)"),
        "generated: {}",
        out
    );

    let src = concat!(
        "import functools\n",
        "\n",
        "@functools.cache\n",
        "def f(s: str) -> str:\n",
        "    return s\n",
    );
    let out = compile(src, "lru3.py");
    assert!(
        out.contains("PyLruCache :: new (None)"),
        "generated: {}",
        out
    );
    // str parameters key as concrete String.
    assert!(out.contains("(String ,)"), "generated: {}", out);
}

#[test]
fn unknown_decorators_and_unhashable_keys_are_loud() {
    // Silently ignoring a decorator converts the program into a
    // different one; refuse.
    let err = compile_err("@mystery\ndef f(n: int) -> int:\n    return n\n", "lru4.py");
    assert!(err.contains("not supported yet"), "error: {}", err);
    assert!(err.contains("refuses to silently ignore"), "error: {}", err);

    // Floats ARE supported as cache keys, with Python's semantics
    // (-0.0 == 0.0, NaN never hits) via the PyFloatKey wrapper.
    let out = compile(
        concat!(
            "from functools import lru_cache\n",
            "\n",
            "@lru_cache\n",
            "def f(x: float) -> float:\n",
            "    return x\n",
        ),
        "lru5.py",
    );
    assert!(out.contains("PyFloatKey"), "float keys must wrap: {}", out);
}

// ---- argparse: conversion-time parsers ----

#[test]
fn argparse_parser_statements_become_a_typed_struct() {
    let src = concat!(
        "import argparse\n",
        "\n",
        "def main() -> None:\n",
        "    p = argparse.ArgumentParser(prog=\"tool\", description=\"Demo\")\n",
        "    p.add_argument(\"name\")\n",
        "    p.add_argument(\"count\", type=int)\n",
        "    p.add_argument(\"--verbose\", action=\"store_true\")\n",
        "    p.add_argument(\"--scale\", type=float, default=1.0)\n",
        "    args = p.parse_args()\n",
        "    print(args.name, args.count, args.scale)\n",
    );
    let out = compile(src, "ap1.py");
    // The parser-building statements vanish; a typed namespace struct
    // and one run_parser call take their place.
    assert!(out.contains("struct __ArgparseArgs"), "generated: {}", out);
    assert!(out.contains("argparse :: run_parser"), "generated: {}", out);
    assert!(out.contains("name : String"), "generated: {}", out);
    assert!(out.contains("count : i64"), "generated: {}", out);
    assert!(out.contains("verbose : bool"), "generated: {}", out);
    assert!(out.contains("scale : f64"), "generated: {}", out);
    assert!(!out.contains("ArgumentParser"), "generated: {}", out);
    assert!(!out.contains("add_argument"), "generated: {}", out);
    // The parser variable is gone entirely (not even a hoisted let).
    assert!(!out.contains("let p"), "generated: {}", out);
}

#[test]
fn argparse_dynamic_or_unsupported_specs_are_loud() {
    // A value-taking option without default= would be None in Python,
    // which the typed field cannot hold.
    let err = compile_err(
        concat!(
            "import argparse\n",
            "\n",
            "def main() -> None:\n",
            "    p = argparse.ArgumentParser()\n",
            "    p.add_argument(\"--scale\", type=float)\n",
            "    args = p.parse_args()\n",
        ),
        "ap2.py",
    );
    assert!(err.contains("needs default="), "error: {}", err);

    // Dynamic names cannot shape a struct at conversion time.
    let err = compile_err(
        concat!(
            "import argparse\n",
            "\n",
            "def main(n: str) -> None:\n",
            "    p = argparse.ArgumentParser()\n",
            "    p.add_argument(n)\n",
            "    args = p.parse_args()\n",
        ),
        "ap3.py",
    );
    assert!(err.contains("string literal"), "error: {}", err);

    // Unsupported add_argument keywords refuse loudly.
    let err = compile_err(
        concat!(
            "import argparse\n",
            "\n",
            "def main() -> None:\n",
            "    p = argparse.ArgumentParser()\n",
            "    p.add_argument(\"xs\", nargs=\"+\")\n",
            "    args = p.parse_args()\n",
        ),
        "ap4.py",
    );
    assert!(
        err.contains("'nargs' is not supported yet"),
        "error: {}",
        err
    );
}

// ---- chained comparisons and loop control through try ----

#[test]
fn chained_comparison_evaluates_each_operand_once() {
    // `a < f() < b` must NOT expand to `a < f() && f() < b`: Python
    // evaluates the middle operand exactly once, so a side-effecting or
    // non-deterministic operand would otherwise diverge.
    let out = compile(
        "def f(n: int) -> int:\n    return n\n\ndef g() -> bool:\n    return 1 < f(5) < 10\n",
        "chain1.py",
    );
    assert_eq!(
        out.matches("f (5)").count(),
        1,
        "middle operand must be evaluated once: {}",
        out
    );
    assert!(out.contains("__rython_cmp"), "generated: {}", out);

    // The later operand stays inside the `&&` so a false prefix leaves
    // it unevaluated, as Python short-circuits.
    let out = compile(
        "def f(n: int) -> int:\n    return n\n\ndef g() -> bool:\n    return 1 < f(2) < f(3)\n",
        "chain2.py",
    );
    assert!(
        out.contains("&& {"),
        "later operand must stay guarded: {}",
        out
    );

    // A plain (unchained) comparison keeps the simple lowering.
    let out = compile(
        "def g(a: int, b: int) -> bool:\n    return a < b\n",
        "chain3.py",
    );
    assert!(!out.contains("__rython_cmp"), "generated: {}", out);
    // Comparisons lower through the PyLt trait (borrowed operands).
    assert!(out.contains("(a) . py_lt (& (b))"), "generated: {}", out);
}

#[test]
fn break_and_continue_thread_out_of_a_try_body() {
    // A break inside a try body targets the enclosing loop, which lies
    // outside the body's closure — it must be signalled out and replayed
    // after the finally clause, not emitted as a `break` in the closure.
    let src = concat!(
        "def f() -> None:\n",
        "    for i in range(3):\n",
        "        try:\n",
        "            if i == 1:\n",
        "                break\n",
        "        finally:\n",
        "            cleanup()\n",
    );
    let out = compile(src, "tryflow1.py");
    assert!(
        out.contains("return Ok (PyFlow :: Break)"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("Ok (PyFlow :: Break) => { cleanup () ; break ; }"),
        "the finally must run before the break resumes: {}",
        out
    );

    // A break belonging to a loop INSIDE the try body stays a plain break.
    let src = concat!(
        "def f() -> None:\n",
        "    try:\n",
        "        for i in range(3):\n",
        "            break\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "tryflow2.py");
    assert!(!out.contains("PyFlow :: Break"), "generated: {}", out);
}

#[test]
fn loop_control_in_a_finally_guarded_handler_is_loud() {
    // The handler body is closure-wrapped when a finally clause exists,
    // so a break there has no signal path out; refuse at conversion time
    // rather than emit Rust that cannot compile.
    let src = concat!(
        "def f() -> None:\n",
        "    for i in range(3):\n",
        "        try:\n",
        "            risky()\n",
        "        except ValueError:\n",
        "            break\n",
        "        finally:\n",
        "            cleanup()\n",
    );
    let err = compile_err(src, "tryflow3.py");
    assert!(err.contains("except handler"), "error: {}", err);
    assert!(err.contains("finally"), "error: {}", err);
}

// ---- f-strings, true division, `not`, `or None` ----

#[test]
fn f_strings_render_through_py_display_not_rust_display() {
    // Rust's Display prints `1` for 1.0 and `true` for True; Python's
    // str() prints `1.0` and `True`.
    let out = compile("def f(x: float):\n    return f\"v={x}\"\n", "fs1.py");
    assert!(out.contains("py_display (& (x))"), "generated: {}", out);

    // A format spec still uses Rust's translated formatting.
    let out = compile("def f(x: float):\n    return f\"v={x:.2f}\"\n", "fs2.py");
    assert!(out.contains("{:.2}"), "generated: {}", out);
    assert!(!out.contains("py_display"), "generated: {}", out);

    // !r renders the repr STRING, so the spec pads the repr like Python;
    // Rust's `{:?}` would print its own Debug form instead.
    let out = compile("def f(s: str):\n    return f\"{s!r}\"\n", "fs3.py");
    assert!(out.contains("repr (& (s))"), "generated: {}", out);
    assert!(!out.contains("{:?}"), "generated: {}", out);
}

#[test]
fn augmented_division_is_true_division() {
    // Python's `/=` yields a float; Rust's `/=` on an integer truncates,
    // so the aug-assign lowers to a py_div rebinding (true division).
    let out = compile("def f(y: float):\n    y /= 2\n    return y\n", "td1.py");
    assert!(out.contains("y = py_div (y , 2)"), "generated: {}", out);
    assert!(!out.contains("y /= 2"), "generated: {}", out);
}

#[test]
fn not_is_a_truthiness_test_not_bitwise_complement() {
    // `not 5` is False; `!5i64` is -6.
    let out = compile("def f(n: int):\n    return not n\n", "not1.py");
    assert!(out.contains("! (n) . is_truthy ()"), "generated: {}", out);

    // `~n` stays a bitwise complement.
    let out = compile("def f(n: int):\n    return ~n\n", "not2.py");
    assert!(out.contains("! n"), "generated: {}", out);
    assert!(!out.contains("is_truthy"), "generated: {}", out);
}

#[test]
fn or_none_yields_none_instead_of_dropping_it() {
    // `count or None` must be None when count is falsy — the None was
    // previously dropped, silently returning the falsy value.
    let out = compile("def f(count: int):\n    return count or None\n", "orn.py");
    assert!(out.contains("is_truthy ()"), "generated: {}", out);
    assert!(out.contains("Some (__rython_or)"), "generated: {}", out);
    assert!(out.contains("None"), "generated: {}", out);
}

// ---- Devin review on PR #103 (F1/F6/F7/F8/F10) ----

#[test]
fn module_level_empty_list_pinned_by_later_use() {
    // F1: `xs = []` at module level used to error because type info was
    // only computed for function bodies; the `xs.append(1)` use must pin
    // the element type the same way it does inside a function.
    let out = compile("xs = []\nxs.append(1)\n", "mempty.py");
    assert!(out.contains("Vec :: < i64 > :: new ()"), "generated: {}", out);
}

#[test]
fn module_level_empty_dict_pinned_by_later_store() {
    let out = compile("d = {}\nd[\"k\"] = 1\n", "memptyd.py");
    assert!(
        out.contains("PyDict :: < String , i64 > :: from ([])"),
        "generated: {}",
        out
    );
}

#[test]
fn annotated_empty_list_honors_annotation() {
    // F8: the empty-container error suggests `xs: list[float] = []`, so the
    // annotation must actually pin the type (it used to be discarded).
    let out = compile("xs: list[float] = []\nxs.append(1.0)\n", "ann_empty.py");
    assert!(out.contains("Vec :: < f64 > :: new ()"), "generated: {}", out);
}

#[test]
fn annotated_empty_dict_honors_annotation() {
    let out = compile("d: dict[str, int] = {}\nd[\"k\"] = 1\n", "ann_emptyd.py");
    assert!(
        out.contains("PyDict :: < String , i64 > :: from ([])"),
        "generated: {}",
        out
    );
}

#[test]
fn post_try_lambda_gets_no_dummy_init() {
    // F6: a variable assigned after a try statement was given a
    // `Default::default()` initializer because the try-closure boundary
    // scanned the whole function's assigned names; a lambda-typed variable
    // (which cannot be Default) then failed to build.
    let out = compile(
        "try:\n    x = 1\nexcept Exception:\n    pass\ng = lambda: 2\n",
        "posttry.py",
    );
    // `x` (assigned inside the try body) keeps its dummy init; the
    // post-try lambda `g` must NOT get one (it cannot be Default).
    assert!(out.contains("x = Default :: default"), "generated: {}", out);
    assert!(!out.contains("g = Default :: default"), "generated: {}", out);
}

#[test]
fn try_body_assigned_name_still_gets_dummy_init() {
    // The flip side of F6: a name first assigned inside the try body IS
    // captured possibly-uninitialized by the try closure (E0381), so it
    // still needs the Default initializer (issue #78 regression guard).
    let out = compile(
        "try:\n    x = 1\nexcept Exception:\n    pass\nprint(x)\n",
        "trybody.py",
    );
    assert!(out.contains("Default :: default"), "generated: {}", out);
}

#[test]
fn starred_list_lowers_through_list_building() {
    // F7: `[*xs, 1]` used to be rejected as "(list, int) incompatible
    // element types" (or a starred-unpacking error). Starred elements now
    // lower through a list-building block that extends with the spread.
    let out = compile("xs = [1, 2]\ny = [*xs, 3]\n", "starred.py");
    assert!(
        out.contains("extend") && out.contains("__rython_list"),
        "the starred list must build by extending: {}",
        out
    );
    assert!(!out.contains("incompatible element types"), "got: {}", out);
}

#[test]
fn aliased_import_resolves_through_module_intercept() {
    // F10: `import numpy as np` bound `np` as an Alias symbol, which
    // module_name_shadowed treated as a user variable, so `np.zeros` never
    // reached the numpy lowering.
    let out = compile("import numpy as np\nx = np.zeros((2, 2))\n", "aliasnp.py");
    assert!(out.contains("numpy :: zeros"), "generated: {}", out);
}

#[test]
fn isinstance_folds_through_the_inheritance_tree() {
    // python3: isinstance(d, Animal) is True for d: Dog — the fold walks
    // the class tree instead of requiring an exact class match, and the
    // constant condition prunes the dead branch (no `if true` residue).
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name: str):\n",
        "        super().__init__(name)\n",
        "\n",
        "class Robot:\n",
        "    def __init__(self, tag: int):\n",
        "        self.tag = tag\n",
        "\n",
        "def check(d: Dog) -> None:\n",
        "    if isinstance(d, Animal):\n",
        "        print(\"animal\")\n",
        "    if isinstance(d, Robot):\n",
        "        print(\"robot\")\n",
    );
    let out = compile(src, "inhtree.py");
    let check_part = out.split("fn check").nth(1).expect("check fn");
    assert!(
        check_part.contains("\"animal\""),
        "the Animal branch must survive (Dog extends Animal): {}",
        out
    );
    assert!(
        !check_part.contains("\"robot\""),
        "the Robot branch is dead and must be pruned: {}",
        out
    );
    assert!(
        !check_part.contains("if (true)") && !check_part.contains("if (false)"),
        "constant isinstance conditions must not remain as if-tests: {}",
        out
    );
}

#[test]
fn constructor_locals_carry_their_class_for_isinstance() {
    // python3: a constructor-assigned local knows its class — isinstance
    // folds true for it (previously the local was untyped and the check
    // silently folded false).
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "def go() -> None:\n",
        "    a = Animal(\"blob\")\n",
        "    if isinstance(a, Animal):\n",
        "        print(\"yes\")\n",
    );
    let out = compile(src, "ctorlocal.py");
    let go_part = out.split("fn go").nth(1).expect("go fn");
    assert!(
        go_part.contains("\"yes\""),
        "the constructor local's class must fold the check true: {}",
        out
    );
}

#[test]
fn isinstance_dispatch_specializes_by_input_type() {
    // The isinstance-dispatch idiom monomorphizes: one Rust function per
    // tested type (class variants per CONCRETE class in the tested
    // subtree) plus a generic residual, and call sites bind the variant
    // matching the argument's static type.
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name: str):\n",
        "        super().__init__(name)\n",
        "\n",
        "def describe(x):\n",
        "    if isinstance(x, int):\n",
        "        return \"int\"\n",
        "    if isinstance(x, Animal):\n",
        "        return \"animal\"\n",
        "    return \"other\"\n",
        "\n",
        "def main() -> None:\n",
        "    print(describe(5))\n",
        "    print(describe(Dog(\"rex\")))\n",
        "    print(describe(2.5))\n",
    );
    let out = compile(src, "specialize.py");
    for variant in [
        "fn describe_int",
        "fn describe_animal",
        "fn describe_dog",
        "fn describe_any",
    ] {
        assert!(out.contains(variant), "missing {variant}: {}", out);
    }
    assert!(
        out.contains("describe_int (5)"),
        "int literal must dispatch to the int variant: {}",
        out
    );
    assert!(
        out.contains("describe_dog ("),
        "a Dog argument must dispatch to Dog's own variant: {}",
        out
    );
    assert!(
        out.contains("describe_any (2.5)"),
        "an untested type must dispatch to the residual: {}",
        out
    );
}

#[test]
fn router_threads_extra_parameters_and_diverging_returns() {
    // Two generalizations of the dynamic router: a NON-tested parameter
    // passes through the router positionally (no enum needed for it),
    // and morphs with DIVERGING return types still get a router — it
    // returns an output enum (`FlipOut`) with `From<T>` per member, and
    // `From<FlipOut> for PyValue` when every member boxes, so a boxed
    // call site consumes the result as Python's union value.
    let src = concat!(
        "def pick(flag: bool) -> str | int:\n",
        "    if flag:\n",
        "        return \"fox\"\n",
        "    return 42\n",
        "\n",
        "def tag(x, prefix: str):\n",
        "    if isinstance(x, str):\n",
        "        return prefix + \": \" + x\n",
        "    if isinstance(x, int):\n",
        "        return prefix + \" #\" + str(x)\n",
        "    return prefix + \"?\"\n",
        "\n",
        "def flip(x):\n",
        "    if isinstance(x, str):\n",
        "        return len(x)\n",
        "    if isinstance(x, int):\n",
        "        return str(x)\n",
        "    return 0\n",
        "\n",
        "def main() -> None:\n",
        "    print(tag(pick(True), \"dyn\"))\n",
        "    print(flip(pick(True)))\n",
    );
    let out = compile(src, "routergen.py");
    for entry in [
        // The untested `prefix` parameter passes through positionally.
        "pub fn tag (x : impl Into < TagArg > , prefix : impl Into < String > ,)",
        // The output enum, its From impls, and the PyValue landing.
        "enum FlipOut",
        "impl From < i64 > for FlipOut",
        "impl From < String > for FlipOut",
        "impl From < FlipOut > for stdpython :: PyValue",
        // Router arms wrap diverging morph results into the enum.
        "Ok (FlipOut :: from (",
    ] {
        assert!(out.contains(entry), "missing {entry}: {}", out);
    }
    // A boxed call site consumes the enum result as the boxed union.
    assert!(
        out.contains("stdpython :: PyValue :: from ((flip ("),
        "a boxed call site must box the output-enum result: {}",
        out
    );
}

#[test]
fn isinstance_dispatch_specializes_over_multiple_axes() {
    // SEVERAL isinstance-tested parameters: the morphs are the cartesian
    // product over each axis of (its variants + Any), named
    // `pair_str_int` / `pair_str_any` / `pair_any_any` / ..., static
    // call sites dispatch each argument independently, and the router
    // takes one NUMBERED argument enum per tested parameter and
    // tuple-matches them.
    let src = concat!(
        "def pick(flag: bool) -> str | int:\n",
        "    if flag:\n",
        "        return \"fox\"\n",
        "    return 42\n",
        "\n",
        "def pair(a, b):\n",
        "    if isinstance(a, str):\n",
        "        if isinstance(b, int):\n",
        "            return a + \" x\" + str(b)\n",
        "        return a + \" ?\"\n",
        "    if isinstance(a, int):\n",
        "        if isinstance(b, int):\n",
        "            return str(a * b)\n",
        "        return str(a)\n",
        "    return \"neither\"\n",
        "\n",
        "def main() -> None:\n",
        "    print(pair(\"fox\", 3))\n",
        "    print(pair(2.5, 1))\n",
        "    print(pair(pick(True), 3))\n",
    );
    let out = compile(src, "multiaxis.py");
    for entry in [
        // Cross-product morphs (bool auto-added per int-tested axis).
        "fn pair_str_int",
        "fn pair_str_any",
        "fn pair_int_int",
        "fn pair_bool_int",
        "fn pair_any_int",
        "fn pair_any_any",
        // One numbered argument enum per axis; the router tuple-matches.
        "enum PairArg1",
        "enum PairArg2",
        "pub fn pair (a : impl Into < PairArg1 > , b : impl Into < PairArg2 > ,)",
        "(PairArg1 :: Str (v1) , PairArg2 :: Int (v2)) => pair_str_int (v1 , v2)",
        "(PairArg1 :: Other (v1) , PairArg2 :: Other (v2)) => pair_any_any (v1 , v2)",
    ] {
        assert!(out.contains(entry), "missing {entry}: {}", out);
    }
    // Static sites dispatch each argument independently...
    assert!(
        out.contains("pair_str_int (\"fox\" , 3)"),
        "static cross dispatch must bind both axes: {}",
        out
    );
    assert!(
        out.contains("pair_any_int ("),
        "an untested type on one axis takes that axis's residual: {}",
        out
    );
    // ...and a boxed axis routes the whole call through the router, the
    // static axis passing as a plain value via From<T>.
    assert!(
        out.contains("pair ((pick") || out.contains("pair (pick"),
        "a boxed axis must dispatch through the router: {}",
        out
    );
}

#[test]
fn isinstance_dispatch_emits_a_dynamic_router() {
    // A single-parameter specialized function whose morphs share a return
    // type also gets a RUNTIME router under the original name: a closed
    // argument enum (one variant per morph + `Other(PyValue)`), `From<T>`
    // per morph so callers pass plain values through `impl Into`, and
    // `From<PyValue>` routing a boxed value in Python's first-true-test
    // order. A call site whose argument is boxed (a `str | int` return)
    // dispatches through the router instead of failing.
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name: str):\n",
        "        super().__init__(name)\n",
        "\n",
        "def label(x):\n",
        "    if isinstance(x, str):\n",
        "        return \"word\"\n",
        "    if isinstance(x, int):\n",
        "        return \"count\"\n",
        "    if isinstance(x, Animal):\n",
        "        return \"pet\"\n",
        "    return \"mystery\"\n",
        "\n",
        "def pick(flag: bool) -> str | int:\n",
        "    if flag:\n",
        "        return \"fox\"\n",
        "    return 42\n",
        "\n",
        "def main() -> None:\n",
        "    print(label(pick(True)))\n",
        "    print(label(True))\n",
    );
    let out = compile(src, "router.py");
    for entry in [
        "enum LabelArg",
        "Other (stdpython :: PyValue)",
        "impl From < String > for LabelArg",
        "impl From < & str > for LabelArg",
        "impl From < i64 > for LabelArg",
        "impl From < bool > for LabelArg",
        "impl From < Animal > for LabelArg",
        "impl From < Dog > for LabelArg",
        "impl From < stdpython :: PyValue > for LabelArg",
        "pub fn label (x : impl Into < LabelArg > ,)",
        "fn from_py_value",
    ] {
        assert!(out.contains(entry), "missing {entry}: {}", out);
    }
    // bool ⊂ int in Python: an int-tested axis carries a bool MORPH of
    // its own (`label_bool(x: bool)`, body folded through the int arm
    // with x kept bool so str(x) renders True/False), a statically-bool
    // call site dispatches to it, and a boxed bool routes to it.
    assert!(
        out.contains("fn label_bool (_x : bool)"),
        "an int-tested axis must carry a bool morph: {}",
        out
    );
    assert!(
        out.contains("label_bool (true)"),
        "a statically-bool argument must dispatch to the bool morph: {}",
        out
    );
    assert!(
        out.contains("stdpython :: PyValue :: Bool (v) => LabelArg :: Bool (v)"),
        "a boxed bool must route to the bool morph: {}",
        out
    );
    // The boxed call site goes through the router (the original name), not
    // a compile-time variant and not a loud error.
    assert!(
        out.contains("label ((pick") || out.contains("label (pick"),
        "a boxed argument must dispatch through the router: {}",
        out
    );
}

#[test]
fn classes_emit_the_type_level_inheritance_tree() {
    // Every class carries `impl PyInherits<Ancestor> for Class` for its
    // full base chain (reflexive included) — the generic inheritance tree
    // generic Rust code can bound on.
    let src = concat!(
        "class Animal:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name: str):\n",
        "        super().__init__(name)\n",
    );
    let out = compile(src, "pyinherits.py");
    for entry in [
        "impl PyInherits < Animal > for Animal",
        "impl PyInherits < Dog > for Dog",
        "impl PyInherits < Animal > for Dog",
    ] {
        assert!(out.contains(entry), "missing {entry}: {}", out);
    }
}

#[test]
fn literal_seeded_local_concretizes_the_loop_element() {
    // Inference: `best = ""` then `best = w` inside `for w in words` — the
    // local keeps ONE type, so the seed's concrete type (String) forces
    // the element instead of emitting a generic that cannot unify with
    // the seed (`longest<A, B>` with `best: String` but `best = w: B`).
    let out = compile(
        concat!(
            "def longest(words):\n",
            "    best = \"\"\n",
            "    for w in words:\n",
            "        if len(w) > len(best):\n",
            "            best = w\n",
            "    return best\n",
        ),
        "seed.py",
    );
    assert!(
        out.contains("IntoIterator < Item = String >"),
        "the element must be concretized to String: {}",
        out
    );
    assert!(
        out.contains("Result < String"),
        "the return type must be the concrete String: {}",
        out
    );
}

#[test]
fn literal_seeded_accumulator_forces_the_element_type() {
    // Inference: `s = 0; s = s + x` inside `for x in items` — the
    // accumulator keeps its i64 seed type, so the element is forced to
    // i64 (a float call site is a LOUD type error, matching the
    // one-type-per-variable model).
    let out = compile(
        concat!(
            "def total(items):\n",
            "    s = 0\n",
            "    for x in items:\n",
            "        s = s + x\n",
            "    return s\n",
        ),
        "acc.py",
    );
    assert!(
        out.contains("IntoIterator < Item = i64 >"),
        "the element must be forced to the accumulator's seed type: {}",
        out
    );
}

#[test]
fn returned_parameters_unify_into_one_type_variable() {
    // Inference: a function returning several bare parameters (`clamp`
    // returns value, low, or high) gives them ONE unified type variable —
    // `clamp<T>(value: T, low: T, high: T) -> Result<T, ...>` — instead
    // of boxing the mixed return to PyValue.
    let out = compile(
        concat!(
            "def clamp(value, low, high):\n",
            "    if value < low:\n",
            "        return low\n",
            "    if value > high:\n",
            "        return high\n",
            "    return value\n",
        ),
        "clampuni.py",
    );
    assert!(
        out.contains("clamp < T >"),
        "returned params must share one type variable: {}",
        out
    );
    assert!(
        !out.contains("PyValue"),
        "the unified return must not box to PyValue: {}",
        out
    );
}

#[test]
fn chained_operator_expressions_get_intermediate_output_bounds() {
    // Inference: `a + (b - a) * t` composes operator Outputs, so the
    // signature must carry the intermediate bounds
    // (`<B as PySub<A>>::Output: PyMul<C>`) or the return type is not
    // well-formed and rustc rejects the definition.
    let out = compile(
        "def lerp(a, b, t):\n    return a + (b - a) * t\n",
        "lerp.py",
    );
    assert!(
        out.contains(":: Output : PyMul < C >"),
        "the intermediate operator Output must be bounded: {}",
        out
    );
}

#[test]
fn np_set_backend_lowers_to_a_raisable_runtime_error() {
    // np.set_backend's runtime helper errors with a plain String (unknown
    // backend name), which `?` alone cannot convert in the generated
    // Result<_, PyException> functions — the lowering must map it into a
    // raised RuntimeError, and take the argument by reference so &str
    // literals and String locals both coerce.
    let out = compile(
        "import numpy as np\ndef pick() -> None:\n    np.set_backend(\"cuda\")\n",
        "setbackend.py",
    );
    assert!(
        out.contains("set_backend_by_name"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("RuntimeError"),
        "the String error must surface as a raised RuntimeError: {}",
        out
    );
    assert!(
        !out.contains("set_backend_by_name (\"cuda\") ?"),
        "bare `?` on the String error can never compile: {}",
        out
    );
}

#[test]
fn aliased_import_reassignment_still_shadows() {
    // F10 guard: a later `np = ...` replaces the alias in the symbol
    // table, so user code still wins over the module intercept.
    let out = compile("import numpy as np\nnp = 5\nx = np\n", "shadownp.py");
    assert!(!out.contains("numpy :: zeros"), "generated: {}", out);
}

#[test]
fn keyword_call_as_statement_keeps_propagation() {
    // F9: `f(a=1)` on its own line emitted `{...}?` — a bare block with `?`
    // is not a valid statement (the block's tail value mismatches `()`),
    // so the generated Rust failed to build. The block must be
    // parenthesized: `({...})?`, valid in both statement and expression
    // position.
    let out = compile(
        "def f(a: int) -> int:\n    return a\nf(a=1)\n",
        "kwstmt.py",
    );
    assert!(
        out.contains("({ let __rython_arg_0 = 1 ; f (__rython_arg_0) }) ?"),
        "generated: {}",
        out
    );
    // The assignment form must keep the same parenthesized shape.
    let out = compile(
        "def f(a: int) -> int:\n    return a\ny = f(a=1)\n",
        "kwexpr.py",
    );
    assert!(
        out.contains("y = ({ let __rython_arg_0 = 1 ; f (__rython_arg_0) }) ?"),
        "generated: {}",
        out
    );
}

#[test]
fn defaulted_call_as_statement_keeps_propagation() {
    // F9 guard: omitted defaulted parameters go through the same mapping
    // path, so `f(1)` (with a defaulted `b`) must also build as a statement.
    let out = compile(
        "def f(a: int, b: int = 10) -> int:\n    return a + b\nf(1)\n",
        "defstmt.py",
    );
    assert!(out.contains("? ;"), "generated: {}", out);
    assert!(!out.contains("} ? ;"), "bare block with `?` leaked: {}", out);
}


// ---- issue #79 cheap guard: conversion-time aliasing detection ----

#[test]
fn aliased_container_mutated_through_second_name_is_a_loud_error() {
    // Issue #79: CPython binds a reference, so b.append shows through a;
    // rython copies containers by value, so it would not. The cheap guard
    // rejects the shape at conversion time instead of silently diverging.
    let err = compile_err(
        "def f():\n    a = [1, 2]\n    b = a\n    b.append(3)\n",
        "alias1.py",
    );
    assert!(err.contains("`b = a`"), "error: {}", err);
    assert!(err.contains("line 3"), "error: {}", err);
}

#[test]
fn aliased_container_mutated_through_first_name_is_a_loud_error() {
    let err = compile_err(
        "def f():\n    a = [1, 2]\n    b = a\n    a[0] = 9\n",
        "alias2.py",
    );
    assert!(err.contains("`b = a`"), "error: {}", err);
}

#[test]
fn aliased_container_rebound_is_not_a_mutation() {
    // Rebinding b to a NEW value never touches the shared object, so the
    // alias is unobservable and must keep converting.
    let out = compile(
        "def f() -> int:\n    a = [1, 2]\n    b = a\n    b = [9]\n    return a[0]\n",
        "alias3.py",
    );
    assert!(out.contains("fn f"), "generated: {}", out);
}

#[test]
fn unmutated_alias_is_not_an_error() {
    // No mutation anywhere: the copy is unobservable.
    let out = compile(
        "def f() -> int:\n    a = [1, 2]\n    b = a\n    return len(b)\n",
        "alias4.py",
    );
    assert!(out.contains("fn f"), "generated: {}", out);
}

#[test]
fn container_passed_to_mutating_function_and_reused_is_a_loud_error() {
    // Shape 2: mutate() appends to its parameter; CPython's caller sees
    // it, rython's clone does not.
    let err = compile_err(
        concat!(
            "def mutate(xs: list[int]) -> None:\n",
            "    xs.append(99)\n",
            "\n",
            "def f():\n",
            "    a = [1, 2]\n",
            "    mutate(a)\n",
            "    print(a)\n",
        ),
        "alias5.py",
    );
    assert!(err.contains("`a`"), "error: {}", err);
    assert!(err.contains("mutates it"), "error: {}", err);
}

#[test]
fn container_passed_to_mutating_function_but_never_reused_is_fine() {
    // The name is used only at the call: rython moves it in, the mutation
    // happens inside, and nothing observes the caller-side copy.
    let out = compile(
        concat!(
            "def mutate(xs: list[int]) -> int:\n",
            "    xs.append(99)\n",
            "    return len(xs)\n",
            "\n",
            "def f() -> int:\n",
            "    a = [1, 2]\n",
            "    return mutate(a)\n",
        ),
        "alias6.py",
    );
    assert!(out.contains("fn f"), "generated: {}", out);
}

#[test]
fn scalar_alias_is_not_a_container_alias() {
    // ints are Copy: b = a shares nothing mutable.
    let out = compile(
        "def f() -> int:\n    a = 5\n    b = a\n    b += 1\n    return a\n",
        "alias7.py",
    );
    assert!(out.contains("fn f"), "generated: {}", out);
}

#[test]
fn alias_inside_loop_body_is_detected() {
    let err = compile_err(
        "def f():\n    a = [1]\n    for i in [1, 2]:\n        b = a\n        b.append(i)\n",
        "alias8.py",
    );
    assert!(err.contains("`b = a`"), "error: {}", err);
}


// ---- async/await: runtime feature gating, asyncio lowering ----

/// Compile with a custom options object.
fn compile_with_options(
    src: &str,
    name: &str,
    options: PythonOptions,
) -> Result<String, String> {
    let module = parse(src, name).map_err(|e| format!("{e}"))?;
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    module
        .to_rust(
            CodeGenContext::Module(name.replace(".py", "")),
            options,
            symbols,
        )
        .map(|t| t.to_string())
        .map_err(|e| format!("{}", e))
}

#[test]
fn async_binary_entry_is_feature_gated_and_imports_runtime() {
    // A BINARY conversion (async_runtime_dep) emits the runtime import and
    // the entry attribute behind the async-tokio feature, plus a
    // compile_error that names the feature when it is off.
    let src = concat!(
        "import asyncio\n",
        "async def main() -> None:\n",
        "    await asyncio.sleep(0)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    asyncio.run(main())\n",
    );
    let out = compile_with_options(
        src,
        "asyncbin.py",
        PythonOptions {
            async_runtime_dep: true,
            ..Default::default()
        },
    )
    .expect("async binary converts");
    assert!(
        out.contains("cfg_attr (feature = \"async-tokio\" , tokio :: main)"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("cfg (feature = \"async-tokio\")") && out.contains("use tokio ;"),
        "runtime import must be feature-gated: {}",
        out
    );
    assert!(
        out.contains("compile_error !"),
        "feature-off build must name the fix: {}",
        out
    );
}

#[test]
fn async_library_has_no_runtime_import_or_entry_attribute() {
    // A LIBRARY conversion (async_runtime_dep unset) transpiles async
    // functions to plain async fns with no tokio import, no entry
    // attribute, and no compile_error: the consumer brings the executor.
    let src = "async def compute(x: int) -> int:\n    return x * 2\n";
    let out = compile_with_options(src, "asynclib.py", PythonOptions::default())
        .expect("async library converts");
    assert!(out.contains("pub async fn compute"), "generated: {}", out);
    assert!(!out.contains("tokio"), "no runtime import for a lib: {}", out);
    assert!(!out.contains("compile_error"), "no feature error for a lib: {}", out);
    assert!(!out.contains("cfg_attr"), "no entry attribute for a lib: {}", out);
}

#[test]
fn asyncio_run_lowers_to_awaited_coroutine() {
    // asyncio.run(coro) drives the coroutine on the current runtime: the
    // argument's trailing `?` (calls to user async functions propagate) is
    // moved after the `.await`, so the Result unwraps, not the future.
    let out = compile(
        concat!(
            "import asyncio\n",
            "async def helper() -> int:\n",
            "    return 1\n",
            "async def main() -> None:\n",
            "    asyncio.run(helper())\n",
        ),
        "asyncio_run.py",
    );
    assert!(
        out.contains("asyncio :: run (helper () ,) . await ?"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains("run (helper () ?)"),
        "the `?` must apply to the awaited Result, not the future: {}",
        out
    );
}

#[test]
fn await_asyncio_sleep_awaits_once() {
    // `await asyncio.sleep(1)` coerces the int argument to a float and the
    // enclosing Await node adds exactly ONE `.await`.
    let out = compile(
        "import asyncio\nasync def f() -> None:\n    await asyncio.sleep(1)\n",
        "asyncio_sleep.py",
    );
    assert!(
        out.contains("asyncio :: sleep ((1) as f64) . await"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains(". await . await"),
        "exactly one await: {}",
        out
    );
}

// ---- issue #109 M1: parameter type inference (trait-bound generics) ----

#[test]
fn unannotated_add_infers_pyadd_bounds() {
    // The issue's acceptance example: `def add(a, b): return a + b` becomes
    // one generic function (`fn add<A, B>(a: A, b: B) -> Result<<A as
    // PyAdd<B>>::Output, ...> where A: PyAdd<B>`), NOT the dead
    // `impl Into<PyObject>` fallback.
    let out = compile(
        "def add(a, b):\n    return a + b\n",
        "inf_add.py",
    );
    assert!(out.contains("fn add < A , B >"), "generated: {}", out);
    assert!(out.contains("where A : PyAdd < B >"), "generated: {}", out);
    assert!(
        out.contains("< A as PyAdd < B >> :: Output"),
        "generated: {}",
        out
    );
    assert!(!out.contains("Into < PyObject >"), "no dead fallback: {}", out);
}

#[test]
fn unannotated_multi_param_add_uses_letter_variables() {
    let out = compile(
        "def add(a, b):\n    return a + b\n",
        "inf_add2.py",
    );
    assert!(out.contains("fn add < A , B >"), "generated: {}", out);
    assert!(out.contains("where A : PyAdd < B >"), "generated: {}", out);
    assert!(
        out.contains("< A as PyAdd < B >> :: Output"),
        "generated: {}",
        out
    );
}

#[test]
fn int_conversion_yields_a_bound_not_a_concrete_type() {
    // The issue's minimal-constraint rule: `int(p)` bounds on PyInt, never
    // forces `p: i64`.
    let out = compile("def to_int(x):\n    return int(x)\n", "inf_int.py");
    assert!(out.contains("pub fn to_int < T >"), "generated: {}", out);
    assert!(out.contains("where T : PyInt"), "generated: {}", out);
    assert!(out.contains("-> Result < i64 , PyException >"), "generated: {}", out);
}

#[test]
fn literal_comparison_is_a_bound_not_a_numeric_type() {
    // `n > 0` bounds on PyGt<T> + PyFromInt — it must NOT force `n: i64`
    // (CPython accepts any comparable instantiation, and Rust std has no
    // int/float cross-PartialOrd, so the literal converts to the
    // parameter's own type). A comparison RETURN is the bound's Output
    // associated type (the codegen emits `n.py_gt(...)`), which the
    // `T: PyGt<T>` bound leaves unconstrained but returnable.
    let out = compile("def positive(n):\n    return n > 0\n", "inf_cmp.py");
    assert!(out.contains("pub fn positive < T >"), "generated: {}", out);
    assert!(out.contains("T : PyGt < T >"), "generated: {}", out);
    assert!(out.contains("T : PyFromInt"), "generated: {}", out);
    assert!(
        out.contains("< T as PyGt < T >> :: Output"),
        "generated: {}",
        out
    );
}

#[test]
fn truthiness_lens_and_display_infer_bounds() {
    let out = compile(
        concat!(
            "def f(x, ys, z):\n",
            "    if x:\n",
            "        n = len(ys)\n",
            "        print(z)\n",
            "        return n\n",
            "    return 0\n",
        ),
        "inf_multi.py",
    );
    assert!(out.contains("where A : Truthy , B : Len , C : PyDisplay"), "generated: {}", out);
}

#[test]
fn unannotated_method_parameter_lowers_as_boxed_pyvalue() {
    // M1 infers free functions only; a method's unannotated parameter has
    // no inference collector, so it lowers as the boxed PyValue with a -W
    // warning (the unannotated-method-parameter divergence).
    let (out, warnings) = compile_with_warnings(
        "class C:\n    def m(self, x):\n        return x\n",
        "inf_method.py",
    );
    assert!(
        out.contains("pub fn m (& self , x : stdpython :: PyValue)"),
        "the unannotated method parameter must lower as PyValue: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unannotated parameter(s) `x`") && w.contains("boxed PyValue")),
        "the boxed-parameter divergence must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn callable_parameter_call_drops_as_a_noop() {
    // A callable parameter called as a function: the callable-as-value
    // divergence (issue #122) — the call site drops to the boxed None and
    // -W reports it.
    let (out, warnings) = compile_with_warnings("def f(cb):\n    return cb(1)\n", "inf_callable.py");
    assert!(
        out.contains("stdpython :: PyValue :: None_"),
        "the call through the callable must drop to a no-op: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("call through callable value `cb`") && w.contains("dropped")),
        "the callable-as-value divergence must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn unknown_method_on_unannotated_parameter_warns_but_converts() {
    // M2's method table covers the stdlib traits; an unknown method bounds
    // the parameter on the duck-unknown trait, and the definitionally
    // unsatisfiable bound becomes a -W warning (M5) plus a #[deprecated]
    // note — never a rustc surprise.
    let (out, warnings) = compile_with_warnings(
        "def frob(s):\n    return s.upar()\n",
        "inf_attr.py",
    );
    assert!(
        out.contains("s . upar ()"),
        "the unknown-method call must still be emitted: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("`s`") && w.contains("satisfied by no known rython type")),
        "the M5 warning must be reported through -W: {:?}",
        warnings
    );
    assert!(
        out.contains("deprecated") && out.contains("PyDuckUnknown"),
        "the #[deprecated] note must carry the warning: {}",
        out
    );
}

#[test]
fn annotated_callee_parameter_identity_forces_the_argument() {
    // M4 FlowsTo: `g(v)` with an ANNOTATED callee parameter forces `v` to
    // the concrete type — no type variable, no bounds, the concrete type's
    // impls are checked at the call site.
    let out = compile(
        "def g(x: int) -> int:\n    return x\ndef f(v):\n    return g(v)\n",
        "inf_flow.py",
    );
    assert!(out.contains("pub fn f (v : i64)"), "generated: {}", out);
    assert!(!out.contains("pub fn f < T >"), "generated: {}", out);
    assert!(out.contains("return Ok (g (v) ?)"), "generated: {}", out);
}

#[test]
fn unannotated_callee_return_flows_to_the_callers_return() {
    // M4 FlowsTo: `caller(v): return helper(v)` where `helper(x): return
    // x * 2` — the callee's return type (in terms of its parameter) flows
    // to the caller's return.
    let out = compile(
        concat!(
            "def helper(x):\n",
            "    return x * 2\n",
            "def caller(v):\n",
            "    return helper(v)\n",
        ),
        "inf_flow2.py",
    );
    assert!(
        out.contains("pub fn caller < T > (v : T) -> Result < < T as PyMul < i64 >> :: Output"),
        "generated: {}",
        out
    );
    assert!(out.contains("T : PyMul < i64 >"), "generated: {}", out);
}

#[test]
fn unsatisfiable_call_site_warns_at_module_level() {
    // M5 call-site satisfiability: `add("a", 1)` — a String argument cannot
    // satisfy `a`'s inferred `PyAdd` bound (stdpython only adds strings
    // with strings; Python would raise TypeError at runtime). The call
    // still lowers; the -W channel reports the unsatisfiable argument.
    let (out, warnings) = compile_with_warnings(
        "def add(a, b):\n    return a + b\nprint(add(\"a\", 1))\n",
        "inf_m5_mod.py",
    );
    assert!(
        out.contains("add ((\"a\") . to_string () , 1)"),
        "the unsatisfiable call must still lower: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("cannot satisfy") && w.contains("PyAdd") && w.contains("str")),
        "the M5 warning must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn unsatisfiable_call_site_warns_inside_a_function() {
    // The same check fires for calls inside annotated/paramless functions,
    // which have no inference collector of their own: the call lowers and
    // the -W channel carries the warning.
    let (out, warnings) = compile_with_warnings(
        "def add(a, b):\n    return a + b\ndef wrapper(x):\n    return add(x, \"boom\")\nprint(wrapper(1))\n",
        "inf_m5_fn.py",
    );
    assert!(
        out.contains("add (x , (\"boom\") . to_string ())"),
        "the unsatisfiable call must still lower: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("cannot satisfy") && w.contains("`x`")),
        "the M5 warning must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn call_site_check_warns_where_a_string_meets_a_numeric_bound() {
    // `is_big("hello")`: a str cannot satisfy the numeric comparison
    // bounds (PyFromInt) — Python raises TypeError for str > int too. The
    // call lowers; the -W channel reports the unsatisfiable argument.
    let (out, warnings) = compile_with_warnings(
        "def is_big(n):\n    return n > 0\nprint(is_big(\"hello\"))\n",
        "inf_m5_str.py",
    );
    assert!(
        out.contains("is_big ((\"hello\") . to_string ())"),
        "the unsatisfiable call must still lower: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("cannot satisfy") && w.contains("PyFromInt")),
        "the M5 warning must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn satisfiable_call_sites_still_convert() {
    // Every accepted call site from M1/M4 keeps converting: numeric
    // promotion, string concatenation, param-to-param flow (checked at
    // the outer call), and comparisons with literal conversion.
    let out = compile(
        concat!(
            "def add(a, b):\n",
            "    return a + b\n",
            "def positive(n):\n",
            "    return n > 0\n",
            "def caller(v):\n",
            "    return add(v, 1)\n",
            "print(add(1, 2))\n",
            "print(add(1.5, 2.5))\n",
            "print(add(\"ab\", \"cd\"))\n",
            "print(caller(7))\n",
            "print(positive(3))\n",
            "print(positive(-1))\n",
        ),
        "inf_m5_ok.py",
    );
    assert!(out.contains("PyAdd"), "generated: {}", out);
    assert!(!out.contains("cannot satisfy"), "generated: {}", out);
}

#[test]
fn iterating_a_parameter_infers_into_iterator_bounds() {
    // M2 iteration: `for x in p` bounds the parameter as IntoIterator and
    // threads the element type into the loop variable, whose own uses get
    // bounds.
    let out = compile(
        "def f(p):\n    for x in p:\n        print(x)\n",
        "iter1.py",
    );
    assert!(
        out.contains("where A : IntoIterator < Item = B >"),
        "generated: {}",
        out
    );
    assert!(out.contains("B : PyDisplay"), "generated: {}", out);
}

#[test]
fn loop_element_as_method_receiver_infers_its_own_bounds() {
    // `for w in words: result.append(w.upper())` — the element's method
    // use bounds the ELEMENT (`B: PyStrOps`), not just the iterable.
    let out = compile(
        concat!(
            "def shout_all(words):\n",
            "    result: list[str] = []\n",
            "    for w in words:\n",
            "        result.append(w.upper())\n",
            "    return result\n",
        ),
        "iter2.py",
    );
    assert!(
        out.contains("A : IntoIterator < Item = B >"),
        "generated: {}",
        out
    );
    assert!(out.contains("B : PyStrOps"), "generated: {}", out);
    assert!(out.contains("-> Result < Vec < String > , PyException >"), "generated: {}", out);
}

#[test]
fn iteration_bounds_flow_through_a_callee() {
    // `caller(v): return shout_all(v)` adopts the callee's Iterate bound
    // (a fresh element) AND the element's own requirements.
    let out = compile(
        concat!(
            "def shout_all(words):\n",
            "    result: list[str] = []\n",
            "    for w in words:\n",
            "        result.append(w.upper())\n",
            "    return result\n",
            "def caller(v):\n",
            "    return shout_all(v)\n",
        ),
        "iter3.py",
    );
    assert!(
        out.contains("pub fn caller < A , B > (v : A) -> Result < Vec < String > , PyException >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("where A : IntoIterator < Item = B > , B : PyStrOps"),
        "caller must adopt the element bounds: {}",
        out
    );
}

#[test]
fn loop_element_return_with_fall_through_is_a_loud_error() {
    // `for x in p: return x` can fall through (empty p → Python None):
    // the inferred generic return cannot coexist with a unit fall-through.
    let err = compile_err(
        "def first(p):\n    for x in p:\n        return x\n",
        "iter4.py",
    );
    assert!(err.contains("fall through"), "error: {}", err);
}

#[test]
fn tuple_loop_target_iterates_a_tuple_item() {
    // Tuple loop targets are supported for any arity (IterateTuple
    // generalised): the iterable's element is a tuple of fresh type
    // variables and the loop destructures it.
    let out = compile(
        "def f(p):\n    for a, b in p:\n        print(a)\n",
        "iter5.py",
    );
    // The iterable's element is a fresh TUPLE of the two target variables
    // (the two variables' order is HashMap-nondeterministic, so only the
    // tuple shape is pinned).
    assert!(
        out.contains("A : IntoIterator < Item = (") && out.contains("B") && out.contains("C"),
        "the element must be a tuple of fresh variables: {}",
        out
    );
    assert!(out.contains("for (a , b) in p"), "generated: {}", out);
}

#[test]
fn iterating_a_non_iterable_argument_warns_but_converts() {
    // M5 call-site satisfiability: `f(5)` cannot satisfy `p`'s
    // IntoIterator bound. The call lowers; the -W channel reports the
    // unsatisfiable argument.
    let (out, warnings) = compile_with_warnings(
        "def f(p):\n    for x in p:\n        print(x)\nprint(f(5))\n",
        "iter6.py",
    );
    assert!(out.contains("f (5)"), "the call must still lower: {}", out);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("IntoIterator") && w.contains("cannot satisfy")),
        "the M5 warning must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn self_recursive_receiver_gets_a_pyadd_self_bound() {
    // `fib(n-1) + fib(n-2)`: the receiver of `+` is the function's OWN
    // return (the fixpoint — the returned parameter's type), so the body
    // needs `T: PyAdd<Self>`, collected from the self-recursive call on
    // the operator's left (M4).
    let out = compile(
        concat!(
            "def fib(n):\n",
            "    if n <= 1:\n",
            "        return n\n",
            "    return fib(n - 1) + fib(n - 2)\n",
        ),
        "inf_fib.py",
    );
    assert!(out.contains("T : PyAdd < T , Output = T >"), "generated: {}", out);
    assert!(out.contains("T : PySub < i64 , Output = T >"), "generated: {}", out);
    assert!(out.contains("-> Result < T , PyException >"), "generated: {}", out);
}

#[test]
fn definitionally_unsatisfiable_bounds_warn_but_convert() {
    // M5: `p.upper()` + `p.pop()` — no known type satisfies
    // PyStrOps + PyPop. A well-formed Python definition: it converts, with
    // the warning baked as a #[deprecated] note (the -W channel reports
    // it; -W deny promotes it to an error).
    let out = compile(
        "def bad(p):\n    p.upper()\n    p.pop()\n",
        "inf_unsat.py",
    );
    assert!(
        out.contains("satisfied by no known rython type"),
        "deprecated note must carry the warning: {}",
        out
    );
    assert!(out.contains("PyStrOps"), "generated: {}", out);
    assert!(out.contains("PyPop"), "generated: {}", out);
}

#[test]
fn satisfiable_bound_sets_do_not_warn() {
    // No #[deprecated] note when some known type satisfies the bounds
    // (PyStrOps alone, Has* duck traits, IntoIterator, ...).
    for src in [
        "def f(s):\n    return s.upper()\n",
        "class Dog:\n    def speak(self) -> str:\n        return \"woof\"\ndef hear(a):\n    return a.speak()\n",
        "def f(xs):\n    for x in xs:\n        print(x)\n",
        "def add(a, b):\n    return a + b\n",
    ] {
        let out = compile(src, "inf_ok_warn.py");
        assert!(
            !out.contains("satisfied by no known rython type"),
            "spurious definition warning for: {}\ngenerated: {}",
            src,
            out
        );
    }
}

#[test]
fn join_on_a_string_literal_infers_a_string_return() {
    // Issue #116: `",".join(parts)` on a literal receiver — the return is
    // an owned String, and the argument must be an iterable of AsRef<str>.
    let out = compile(
        "def join_all(parts):\n    return \",\".join(parts)\n",
        "inf_join.py",
    );
    assert!(
        out.contains("-> Result < String , PyException >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("A : IntoIterator < Item = B >"),
        "generated: {}",
        out
    );
    assert!(out.contains("B : AsRef < str >"), "generated: {}", out);
}

#[test]
fn genexpr_over_a_parameter_infers_iteration_bounds() {
    // Issue #116: `".".join(str(v) for v in version)` — the pip pattern.
    // The generator's iterable bounds IntoIterator; the element's uses
    // (str(v)) bound the element.
    let out = compile(
        "def version_str(version):\n    return \".\".join(str(v) for v in version)\n",
        "inf_genexpr.py",
    );
    assert!(out.contains("A : IntoIterator < Item = B >"), "generated: {}", out);
    assert!(out.contains("B : PyToString"), "generated: {}", out);
    assert!(
        out.contains("-> Result < String , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn list_comprehension_over_a_parameter_infers_a_vec_return() {
    // `[w.upper() for w in words]` — Vec<String> return, element bounds
    // from the comprehension body (issue #116).
    let out = compile(
        "def upper_all(words):\n    return [w.upper() for w in words]\n",
        "inf_listcomp.py",
    );
    assert!(
        out.contains("-> Result < Vec < String > , PyException >"),
        "generated: {}",
        out
    );
    assert!(out.contains("B : PyStrOps"), "generated: {}", out);
}

#[test]
fn string_literal_local_rebound_by_aug_assign_is_owned() {
    // Issue #110: `out = ""; out += "x"` — the literal assignment is owned
    // (`"".to_string()`), the binding is String, and the return is String.
    let out = compile(
        concat!(
            "def accumulate():\n",
            "    out = \"\"\n",
            "    out += \"x\"\n",
            "    return out\n",
        ),
        "str_aug.py",
    );
    assert!(out.contains("out = (\"\") . to_string ()"), "generated: {}", out);
    assert!(
        out.contains("-> Result < String , PyException >"),
        "generated: {}",
        out
    );
    assert!(!out.contains("-> Result < & 'static str"), "generated: {}", out);
}

#[test]
fn plain_string_literal_local_stays_unowned() {
    // A string-literal local that is never rebound keeps its old lowering
    // (&'static str) — no to_string noise.
    let out = compile(
        "def plain():\n    s = \"hi\"\n    return s\n",
        "str_plain.py",
    );
    assert!(!out.contains("to_string"), "generated: {}", out);
    assert!(
        out.contains("-> Result < & 'static str , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn global_declaration_with_read_converts() {
    // Issue #115: `global x` is accepted; READS of the name resolve to
    // the module static.
    let out = compile(
        concat!(
            "DEFAULT_SESSION = \"initial\"\n",
            "def show():\n",
            "    global DEFAULT_SESSION\n",
            "    print(DEFAULT_SESSION)\n",
        ),
        "global_read.py",
    );
    assert!(out.contains("pub static DEFAULT_SESSION"), "generated: {}", out);
    assert!(!out.contains("not supported"), "generated: {}", out);
}

#[test]
fn global_string_write_lowers_to_a_lazylock_static() {
    // Issue #115 completion: a STRING-initialized global written through
    // `global` becomes `static name: LazyLock<Mutex<String>>` (String
    // construction is not const); reads/writes deref the LazyLock
    // (`&*name`), literal stores own themselves, and the old write-drop
    // warning is gone.
    let (out, warnings) = compile_with_warnings(
        concat!(
            "DEFAULT_SESSION = \"initial\"\n",
            "def set_it():\n",
            "    global DEFAULT_SESSION\n",
            "    DEFAULT_SESSION = \"new\"\n",
            "def show() -> str:\n",
            "    return DEFAULT_SESSION\n",
        ),
        "global_write.py",
    );
    assert!(
        out.contains(
            "pub static DEFAULT_SESSION : std :: sync :: LazyLock < std :: sync :: Mutex < String >>"
        ),
        "generated: {}",
        out
    );
    assert!(
        out.contains("py_global_write (& * DEFAULT_SESSION , (\"new\") . to_string ())"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("py_global_read (& * DEFAULT_SESSION)"),
        "generated: {}",
        out
    );
    assert!(
        warnings.iter().all(|w| !w.contains("writes to module-level name")),
        "the supported write must not warn: {:?}",
        warnings
    );
}

#[test]
fn global_computed_initializer_lowers_to_a_lazylock_static() {
    // Issue #115 completion: a COMPUTED single-store initializer becomes
    // `static name: LazyLock<Mutex<T>>` with the inferred type, a
    // panic-on-Err closure for a fallible initializer, and a touch in
    // __module_init__ so its side effects still run at import time.
    let out = compile(
        concat!(
            "def compute() -> int:\n",
            "    return 2\n",
            "limit = compute()\n",
            "def raise_limit():\n",
            "    global limit\n",
            "    limit = limit + 10\n",
        ),
        "global_computed.py",
    );
    assert!(
        out.contains("pub static limit : std :: sync :: LazyLock < std :: sync :: Mutex < i64 >>"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("initialization failed"),
        "fallible init must panic on Err: {}",
        out
    );
    assert!(
        out.contains("let _ = & * limit"),
        "__module_init__ must touch the static: {}",
        out
    );
    assert!(
        out.contains("py_global_write (& * limit"),
        "generated: {}",
        out
    );
}

#[test]
fn del_index_bounds_on_pypop() {
    // Issue #112: `del xs[i]` on an unannotated parameter bounds
    // `T: PyPop<i64>` (list) / `T: PyPop<String>` (string-keyed dict).
    let out = compile(
        "def drop(xs):\n    del xs[1]\n    return xs\n",
        "del_list.py",
    );
    assert!(out.contains("T : PyPop < i64 >"), "generated: {}", out);
    let out2 = compile(
        "def drop(d):\n    del d[\"b\"]\n    return d\n",
        "del_dict.py",
    );
    assert!(out2.contains("T : PyPop < String >"), "generated: {}", out2);
}

#[test]
fn del_name_unused_afterwards_is_a_noop() {
    // Issue #112: `del name` lowers to a no-op when the name is never
    // referenced afterwards — behaviorally identical to Python.
    let out = compile(
        "from logging import NullHandler\nlog = NullHandler\ndel NullHandler\n",
        "del_noop.py",
    );
    assert!(!out.contains("not supported"), "generated: {}", out);
    assert!(!out.contains("unbinding"), "generated: {}", out);
}

#[test]
fn use_after_del_is_a_loud_error() {
    // `del x` then a use of `x` would still see the value where Python
    // raises NameError — loud error.
    let err = compile_err(
        "import sys\ndef f():\n    del sys\n    return sys\n",
        "del_use.py",
    );
    assert!(err.contains("del sys"), "error: {}", err);
    assert!(err.contains("issue #112"), "error: {}", err);
}

#[test]
fn del_then_reassign_is_allowed() {
    // Python's `del x; x = 1` rebinds — no error.
    let out = compile(
        "x = 0\ndef f():\n    x = 1\n    del x\n    x = 2\n    return x\n",
        "del_rebind.py",
    );
    assert!(!out.contains("unbinding"), "generated: {}", out);
}

#[test]
fn warnings_calls_render_through_their_signature() {
    // Issue #111: warnings functions accept keyword arguments and omitted
    // trailing parameters — slots fill with Some(...)/None.
    let out = compile(
        concat!(
            "import warnings\n",
            "def check(x):\n",
            "    warnings.warn(\"hi\")\n",
            "    warnings.simplefilter(\"ignore\", append=True)\n",
            "    return x\n",
        ),
        "warnings.py",
    );
    assert!(
        out.contains("warnings :: warn (Some (\"hi\") , None , None , None)"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("warnings :: simplefilter (Some (\"ignore\") , None , None , None , Some (true))"),
        "generated: {}",
        out
    );
}

#[test]
fn warnings_unknown_keyword_is_a_loud_error() {
    let err = compile_err(
        "import warnings\ndef f():\n    warnings.warn(bogus_keyword=1)\n",
        "warnings_bad.py",
    );
    assert!(err.contains("unexpected keyword"), "error: {}", err);
}

#[test]
fn class_with_a_foreign_base_lowers_as_a_plain_struct() {
    // A dotted base (`class ShutdownQueue(queue.Queue)`) used to crash the
    // parser (bases extracted as Vec<Name>). The foreign/imported base is
    // now tolerated as metadata: the class lowers as a plain struct with
    // no embedded base (the foreign-base divergence).
    let out = compile(
        concat!(
            "class ShutdownQueue(queue.Queue):\n",
            "    pass\n",
        ),
        "foreign_base.py",
    );
    assert!(
        out.contains("pub struct ShutdownQueue"),
        "the class must lower as a plain struct: {}",
        out
    );
    assert!(
        !out.contains("__rython_base"),
        "no embedded base may be emitted for a foreign base: {}",
        out
    );
}

#[test]
fn classmethod_and_staticmethod_lower_as_associated_functions() {
    // Issue #117: @classmethod drops the class reference parameter and
    // emits an associated fn; @staticmethod likewise. Calls route
    // `Class::method(...)`.
    let out = compile(
        concat!(
            "class Finder:\n",
            "    @classmethod\n",
            "    def find_spec(cls, fullname):\n",
            "        return fullname\n",
            "    @staticmethod\n",
            "    def hint():\n",
            "        return \"hint\"\n",
            "print(Finder.find_spec(\"pip\"))\n",
            "print(Finder.hint())\n",
        ),
        "classmethod.py",
    );
    assert!(out.contains("Finder :: find_spec"), "generated: {}", out);
    assert!(out.contains("Finder :: hint"), "generated: {}", out);
    // The class reference is dropped: no receiver, no cls parameter.
    assert!(
        !out.contains("fn find_spec (cls"),
        "generated: {}",
        out
    );
}

#[test]
fn type_subscript_annotation_is_tolerated() {
    // `category: type[Warning]` is a CLASS annotation — tolerated as an
    // opaque Option<()> so the definition compiles (class-as-value calls
    // are the documented divergence).
    let out = compile(
        concat!(
            "import warnings\n",
            "def disable_warnings(category: type[Warning] = None) -> None:\n",
            "    warnings.simplefilter(\"ignore\", category)\n",
        ),
        "type_ann.py",
    );
    assert!(
        !out.contains("unsupported annotation"),
        "generated: {}",
        out
    );
}

#[test]
fn same_rust_type_union_annotations_are_accepted() {
    // `bytes | bytearray` — a PEP 604 union whose members both map to
    // Vec<u8> — lowers to that type (charset_normalizer's from_bytes).
    let out = compile(
        "def from_bytes(sequences: bytes | bytearray):\n    return len(sequences)\n",
        "union_ann.py",
    );
    assert!(
        !out.contains("unsupported annotation"),
        "generated: {}",
        out
    );
}

#[test]
fn mutually_recursive_returns_are_a_loud_error() {
    // M4: mutual recursion without return annotations cannot be resolved
    // to a single return type — loud error naming the cycle.
    let err = compile_err(
        concat!(
            "def a(x):\n",
            "    return b(x)\n",
            "def b(y):\n",
            "    return a(y)\n",
        ),
        "inf_mutrec.py",
    );
    assert!(err.contains("mutually recursive"), "error: {}", err);
}

#[test]
fn no_impl_into_pyobject_anywhere_for_unannotated_params() {
    // The fallback is deleted: every unannotated parameter either infers or
    // errors loudly.
    let out = compile("def f(x):\n    return x\n", "inf_id.py");
    assert!(out.contains("pub fn f < T > (x : T)"), "generated: {}", out);
    assert!(!out.contains("Into < PyObject >"), "no fallback: {}", out);
}

// ---- issue #109 M2: stdlib method table (duck-typed method bounds) ----

#[test]
fn str_methods_infer_pystrops_bounds() {
    let out = compile(
        "def shout(s):\n    return s.upper()\n",
        "m2_upper.py",
    );
    assert!(out.contains("pub fn shout < T >"), "generated: {}", out);
    assert!(out.contains("where T : PyStrOps"), "generated: {}", out);
    assert!(out.contains("-> Result < String , PyException >"), "generated: {}", out);

    let out = compile(
        "def parts(s):\n    return s.split(\" \")\n",
        "m2_split.py",
    );
    assert!(out.contains("T : PyStrOps"), "generated: {}", out);
    assert!(
        out.contains("-> Result < Vec < String > , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn pop_infers_pypop_with_index_type() {
    let out = compile(
        "def last(xs):\n    return xs.pop()\n",
        "m2_pop.py",
    );
    assert!(out.contains("where T : PyPop < i64 >"), "generated: {}", out);
    assert!(
        out.contains("< T as PyPop < i64 >> :: Output"),
        "generated: {}",
        out
    );
}

#[test]
fn mixed_method_bounds_on_one_parameter() {
    // A parameter used through several stdlib methods accumulates exactly
    // those bounds (minimal constraint).
    let out = compile(
        "def stats(s):\n    n = s.count(\"a\")\n    return s.upper() + str(n)\n",
        "m2_mixed.py",
    );
    assert!(out.contains("where T : PyStrOps"), "generated: {}", out);
    // s is read twice (count + upper), so the reuse-clone rule adds Clone.
    assert!(out.contains("T : Clone"), "generated: {}", out);
    assert!(
        out.contains("-> Result < String , PyException >"),
        "generated: {}",
        out
    );
}

// ---- issue #109 M3: user-class duck typing (generated Has* traits) ----

#[test]
fn two_class_method_generates_has_trait() {
    // The issue's hear() example: a method on multiple classes bounds the
    // parameter on a generated HasSpeak trait, with one impl per class.
    let out = compile(
        concat!(
            "class Dog:\n",
            "    def speak(self) -> str:\n",
            "        return \"woof\"\n",
            "\n",
            "class Cat:\n",
            "    def speak(self) -> str:\n",
            "        return \"meow\"\n",
            "\n",
            "def hear(animal):\n",
            "    return animal.speak()\n",
        ),
        "m3_hear.py",
    );
    assert!(out.contains("pub trait HasSpeak"), "generated: {}", out);
    assert!(out.contains("impl HasSpeak for Dog"), "generated: {}", out);
    assert!(out.contains("impl HasSpeak for Cat"), "generated: {}", out);
    assert!(out.contains("fn speak (& self) -> Result < String , PyException >"), "generated: {}", out);
    assert!(out.contains("pub fn hear < T >"), "generated: {}", out);
    assert!(out.contains("where T : HasSpeak"), "generated: {}", out);
    assert!(out.contains("animal . speak () ?"), "call threads `?`: {}", out);
}

#[test]
fn single_class_method_still_generates_has_trait() {
    // The single-implementor case still bounds on the trait, never on the
    // concrete class.
    let out = compile(
        concat!(
            "class Dog:\n",
            "    def bark(self) -> str:\n",
            "        return \"woof\"\n",
            "\n",
            "def hear(d):\n",
            "    return d.bark()\n",
        ),
        "m3_single.py",
    );
    assert!(out.contains("pub trait HasBark"), "generated: {}", out);
    assert!(out.contains("impl HasBark for Dog"), "generated: {}", out);
    assert!(out.contains("where T : HasBark"), "generated: {}", out);
    assert!(!out.contains("where T : Dog"), "never the concrete class: {}", out);
}

#[test]
fn conflicting_duck_signatures_are_a_loud_error() {
    // Two classes define `m` with different parameter types: one trait
    // cannot bound both, so the conversion fails loudly.
    let err = compile_err(
        concat!(
            "class A:\n",
            "    def m(self, x: int) -> int:\n",
            "        return x\n",
            "\n",
            "class B:\n",
            "    def m(self, x: str) -> int:\n",
            "        return 1\n",
            "\n",
            "def f(v):\n",
            "    return v.m(1)\n",
        ),
        "m3_conflict.py",
    );
    assert!(err.contains("conflicting"), "error: {}", err);
    assert!(err.contains("`m`"), "error: {}", err);
}

#[test]
fn duck_trait_generated_once_per_module() {
    // Two functions bound on the same method share ONE trait definition.
    let out = compile(
        concat!(
            "class Dog:\n",
            "    def speak(self) -> str:\n",
            "        return \"woof\"\n",
            "\n",
            "def hear1(animal):\n",
            "    return animal.speak()\n",
            "\n",
            "def hear2(animal):\n",
            "    return animal.speak()\n",
        ),
        "m3_once.py",
    );
    assert_eq!(out.matches("pub trait HasSpeak").count(), 1, "generated: {}", out);
    assert_eq!(out.matches("impl HasSpeak for Dog").count(), 1, "generated: {}", out);
}

#[test]
fn pyvalue_bool_str_none_narrows_via_isinstance() {
    // Issue #121: `bool | str | None` (requests' `verify`) lowers to the
    // boxed PyValue; `is False` tests the Bool member, isinstance narrows
    // the str branch, and the dict[str, Any] stores wrap.
    let out = compile(
        "from typing import Any\n\
         def f(verify: bool | str | None) -> dict[str, Any]:\n\
         \x20   pool_kwargs: dict[str, Any] = {}\n\
         \x20   cert_reqs = \"CERT_REQUIRED\"\n\
         \x20   if verify is False:\n\
         \x20       cert_reqs = \"CERT_NONE\"\n\
         \x20   elif isinstance(verify, str):\n\
         \x20       pool_kwargs[\"ca_certs\"] = verify\n\
         \x20   pool_kwargs[\"cert_reqs\"] = cert_reqs\n\
         \x20   return pool_kwargs\n",
        "verify.py",
    );
    assert!(out.contains("PyValue"), "bool | str | None must lower to PyValue: {}", out);
    assert!(out.contains("is_bool ()") || out.contains("is_bool()"), "is False must test is_bool: {}", out);
    assert!(out.contains("is_str ()") || out.contains("is_str()"), "isinstance(str) must dispatch: {}", out);
    assert!(out.contains("as_str () . unwrap ()") || out.contains("as_str().unwrap()"), "str branch must narrow: {}", out);
    assert!(out.contains("PyValue :: from") || out.contains("PyValue::from"), "stores must box: {}", out);
}

#[test]
fn pyvalue_tuple_union_len_and_index() {
    // Issue #121: `tuple[str, str] | str | None` (requests' `client_cert`):
    // the compound `isinstance(x, tuple) and len(x) == 2` narrows the
    // body, and tuple indexing reads the elements.
    let out = compile(
        "from typing import Any\n\
         def f(client_cert: tuple[str, str] | str | None) -> dict[str, Any]:\n\
         \x20   pool_kwargs: dict[str, Any] = {}\n\
         \x20   if client_cert is not None:\n\
         \x20       if isinstance(client_cert, tuple) and len(client_cert) == 2:\n\
         \x20           pool_kwargs[\"cert_file\"] = client_cert[0]\n\
         \x20           pool_kwargs[\"key_file\"] = client_cert[1]\n\
         \x20       else:\n\
         \x20           pool_kwargs[\"cert_file\"] = client_cert\n\
         \x20   return pool_kwargs\n",
        "cert.py",
    );
    assert!(out.contains("is_tuple ()") || out.contains("is_tuple()"), "isinstance(tuple) must dispatch: {}", out);
    assert!(out.contains("as_tuple () . unwrap ()") || out.contains("as_tuple().unwrap()"), "tuple branch must narrow: {}", out);
    assert!(out.contains("py_index"), "tuple indexing must use py_index: {}", out);
}

#[test]
fn object_param_is_boxed_and_class_isinstance_is_static() {
    // `other: object` lowers to the boxed PyValue; isinstance against a
    // NON-exception class is statically false (rython cannot hold class
    // instances in a PyValue), so `__eq__` guards convert.
    let out = compile(
        "class CharsetMatch:\n\
         \x20   def __eq__(self, other: object) -> bool:\n\
         \x20       if not isinstance(other, CharsetMatch):\n\
         \x20           return False\n\
         \x20       return True\n",
        "eq.py",
    );
    assert!(out.contains("PyValue"), "object must lower to PyValue: {}", out);
    assert!(out.contains("false"), "isinstance(class) on an object must be false: {}", out);
}

#[test]
fn kwargs_param_packs_extra_keywords() {
    // Issue #120: a **kwargs parameter lowers to a boxed PyDict<String,
    // PyValue>; call sites pack extra keywords (and spread dicts) into it.
    let out = compile(
        "from typing import Any\n\
         def make_pool(num_pools: int = 10, **connection_pool_kw: Any):\n\
         \x20   if \"retries\" in connection_pool_kw:\n\
         \x20       retries = connection_pool_kw[\"retries\"]\n\
         \x20   return connection_pool_kw\n\
         p = make_pool(retries=3, timeout=1.5)\n",
        "kw.py",
    );
    assert!(
        out.contains("PyDict < String , stdpython :: PyValue >")
            || out.contains("PyDict<String, stdpython::PyValue>"),
        "**kwargs must lower to the boxed dict: {}",
        out
    );
    assert!(out.contains("PyValue :: from") || out.contains("PyValue::from"), "keyword values must box: {}", out);
}

#[test]
fn generator_builds_and_returns_list() {
    // Issue #122-family: a `yield` body lowers to a collector Vec and a
    // final return — `for chunk in cut(...)` callers iterate the list.
    let out = compile(
        "from typing import Generator\n\
         def cut(decoded: str, n: int) -> Generator[str, None, None]:\n\
         \x20   for i in range(0, 100, n):\n\
         \x20       chunk = decoded[i : i + n]\n\
         \x20       if not chunk:\n\
         \x20           break\n\
         \x20       yield chunk\n",
        "gen.py",
    );
    assert!(out.contains("__rython_gen"), "generator must build a collector Vec: {}", out);
    assert!(out.contains("push"), "yield must push into the collector: {}", out);
    assert!(out.contains("Vec < String >") || out.contains("Vec<String>"), "element type must come from Generator[T, ...]: {}", out);
    // Inside the function's Result, like every return (a bare
    // `return __rython_gen` was an E0308 against the Result signature).
    assert!(
        out.contains("return Ok (__rython_gen)"),
        "generator must return the collector in the Result: {}",
        out
    );
}

#[test]
fn dict_any_literal_wraps_mixed_values() {
    // Issue #121: a dict literal stored into a `dict[str, Any]` name wraps
    // each value in PyValue::from (mixed str/int values).
    let out = compile(
        "from typing import Any\n\
         def f() -> dict[str, Any]:\n\
         \x20   host_params: dict[str, Any] = {}\n\
         \x20   host_params = {\"scheme\": \"https\", \"port\": 443}\n\
         \x20   return host_params\n",
        "host.py",
    );
    assert!(out.contains("PyDict < String , stdpython :: PyValue >") || out.contains("PyDict<String, stdpython::PyValue>"), "dict[str, Any] must lower to the boxed dict: {}", out);
    assert!(
        out.matches("PyValue :: from").count() >= 2 || out.matches("PyValue::from").count() >= 2,
        "mixed values must wrap: {}",
        out
    );
}

#[test]
fn classmethod_cls_reference_resolves_to_class() {
    // urllib3's Retry.from_int: `@classmethod def from_int(cls, ...)` —
    // the class parameter is dropped from the signature, but the body
    // references it (`cls.DEFAULT`, `cls(...)`). `cls` must resolve to the
    // enclosing class: constant reads render `Retry::DEFAULT` and calls
    // render `Retry::new(...)`.
    let out = compile(
        "class Retry:\n    DEFAULT = 3\n    @classmethod\n    def from_int(cls, retries: int) -> int:\n        if retries is None:\n            retries = cls.DEFAULT\n        return cls(retries)\n",
        "retry.py",
    );
    assert!(
        out.contains("Retry :: DEFAULT"),
        "cls.DEFAULT must render the class constant: {}",
        out
    );
    assert!(
        out.contains("Retry :: new"),
        "cls(...) must render the class constructor: {}",
        out
    );
    assert!(
        !out.contains("cls . DEFAULT"),
        "cls must not leak as a bare receiver: {}",
        out
    );
}

#[test]
fn class_body_computed_constant_promotes_to_lazylock() {
    // urllib3's Retry: `DEFAULT_ALLOWED_METHODS = frozenset(["HEAD",
    // "GET", ...])` — a class-body COMPUTED constant (not a literal, so no
    // `pub const`). It is emitted as a class-level LazyLock static, and a
    // dropped-default reference inside the class (`__init__`'s
    // `allowed_methods=DEFAULT_ALLOWED_METHODS`, inlined at a
    // `Retry::new(...)` call site) deref-clones it.
    let out = compile(
        "class Retry:\n    DEFAULT_ALLOWED_METHODS = frozenset([\"HEAD\", \"GET\"])\n    def __init__(self, allowed_methods=None) -> None:\n        self.allowed = allowed_methods\n",
        "retryconst.py",
    );
    assert!(
        out.contains("pub static Retry_DEFAULT_ALLOWED_METHODS"),
        "class-body computed constant must be a module-level class-mangled LazyLock static (issue #137): {}",
        out
    );
}

#[test]
fn imported_class_constant_default_resolves_through_import() {
    // requests' adapters.py: `from urllib3.util.retry import Retry` then
    // `Retry(total=0, connect=None, ...)` — the call inlines Retry.__init__'s
    // dropped defaults, which reference Retry's CLASS-BODY constants
    // (`allowed_methods=DEFAULT_ALLOWED_METHODS`). The caller only imported
    // the class, so the default must render through the imported LOCAL name
    // (`Retry::DEFAULT_ALLOWED_METHODS`), not a bare undefined identifier.
    // Two-module conversion: retry.py defines the class, adapters.py imports
    // and constructs it.
    let retry_src =
        "class Retry:\n    DEFAULT_ALLOWED_METHODS = frozenset([\"HEAD\", \"GET\"])\n    def __init__(self, allowed_methods=DEFAULT_ALLOWED_METHODS, backoff_max=120) -> None:\n        self.allowed = allowed_methods\n        self.backoff = backoff_max\n";
    let retry_mod = parse(retry_src, "retry.py").unwrap();
    let adapters_src =
        "from retry import Retry\nclass HTTPAdapter:\n    def __init__(self) -> None:\n        self.retries = Retry()\n";
    let adapters_mod = parse(adapters_src, "adapters.py").unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["retry".to_string()], std::rc::Rc::new(retry_mod));
    defs.insert(vec!["adapters".to_string()], std::rc::Rc::new(adapters_mod));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(adapters_src, "adapters.py", options).expect("converts");
    assert!(
        out.contains("Retry :: DEFAULT_ALLOWED_METHODS") || out.contains("Retry::DEFAULT_ALLOWED_METHODS"),
        "dropped default must render through the imported class name: {}",
        out
    );
    assert!(
        !out.contains("Some (DEFAULT_ALLOWED_METHODS)") && !out.contains("Some(DEFAULT_ALLOWED_METHODS)"),
        "bare DEFAULT_ALLOWED_METHODS must not leak: {}",
        out
    );
}

#[test]
fn itertools_takewhile_swaps_predicate_and_iterable() {
    // urllib3's retry.get_backoff_time: `takewhile(lambda x: ..., reversed(
    // self.history))` — Python (predicate, iterable) maps to the runtime
    // (iterable, predicate).
    let out = compile(
        "from itertools import takewhile\n\ndef f(items: list[int]) -> int:\n    return len(list(takewhile(lambda x: x > 0, reversed(items))))\n",
        "tw.py",
    );
    assert!(
        out.contains("takewhile (reversed") || out.contains("takewhile(reversed"),
        "iterable must be the first runtime arg: {}",
        out
    );
    assert!(
        !out.contains("takewhile :: new"),
        "takewhile must not lower as a module-path construction: {}",
        out
    );
}

#[test]
fn iter_builtin_boxes_its_argument() {
    // urllib3's request.py body_to_chunks: `chunks = iter(body)` — the
    // iterator factory lowers to the boxed argument (values are already
    // iterable at their natural position).
    let out = compile(
        "def f(body: bytes) -> None:\n    chunks = iter(body)\n    for chunk in chunks:\n        pass\n",
        "iterb.py",
    );
    assert!(
        out.contains("PyValue :: from") || out.contains("PyValue::from"),
        "iter(x) must box its argument: {}",
        out
    );
    assert!(
        !out.contains("iter (body)"),
        "bare iter(body) must not leak: {}",
        out
    );
}

#[test]
fn tuple_builtin_boxes_its_argument() {
    // urllib3's poolmanager.py: `context["socket_options"] = tuple(socket_opts)`
    // — the tuple constructor lowers to the boxed argument (values are already
    // boxed; tuple() on a boxed iterable is identity in rython's model).
    let out = compile(
        "def f(socket_opts: list[bytes]) -> list[bytes]:\n    return tuple(socket_opts)\n",
        "tupleb.py",
    );
    assert!(
        out.contains("PyValue :: from") || out.contains("PyValue::from"),
        "tuple(x) must box its argument: {}",
        out
    );
    assert!(
        !out.contains("tuple (socket_opts)"),
        "bare tuple(socket_opts) must not leak: {}",
        out
    );
}

#[test]
fn except_binding_attribute_read_boxes_to_none() {
    // urllib3's response.py _error_catcher: `except IncompleteRead as e:` then
    // `e.expected` / `e.partial` — the exception object has no static fields
    // (rython models exceptions as name + message), so the attribute read
    // lowers to the boxed None with a warning (dynamic-attribute divergence).
    // The bare name read (`raise ProtocolError(arg, e) from e`) still renders
    // the bound PyException.
    let out = compile(
        "def f() -> None:\n    try:\n        g()\n    except IncompleteRead as e:\n        if e.expected is not None:\n            h(e)\n",
        "excattr.py",
    );
    assert!(
        out.contains("stdpython :: PyValue :: None_") || out.contains("stdpython::PyValue::None_"),
        "e.expected must lower to the boxed None: {}",
        out
    );
    assert!(
        !out.contains("e . expected") && !out.contains("e.expected"),
        "e.expected must not emit a field read: {}",
        out
    );
    assert!(
        out.contains("let mut e = __rython_exc . clone ()") || out.contains("let mut e = __rython_exc.clone()"),
        "the except binding must still bind the PyException: {}",
        out
    );
    let (_out, warnings) = compile_with_warnings(
        "def f() -> None:\n    try:\n        g()\n    except IncompleteRead as e:\n        if e.expected is not None:\n            h(e)\n",
        "excattr.py",
    );
    assert!(
        warnings.iter().any(|w| w.contains("e.expected") && w.contains("dynamic-attribute divergence")),
        "must warn about the dropped attribute read: {:?}",
        warnings
    );
}

#[test]
fn next_builtin_returns_first_element_or_stopiteration() {
    // requests' sessions.py: `r._next = next(self.resolve_redirects(
    // ..., yield_requests=True))` inside try/except StopIteration —
    // rython's eager generator model collects the body into a Vec, so
    // next(vec) is the FIRST element, raising StopIteration when empty.
    let out = compile(
        "def f(gen: list[bytes]) -> bytes:\n    try:\n        return next(gen)\n    except StopIteration:\n        return b''\n",
        "nextb.py",
    );
    assert!(
        out.contains("StopIteration"),
        "next() on an empty generator must raise StopIteration: {}",
        out
    );
    assert!(
        !out.contains("next (gen)"),
        "bare next(gen) must not leak: {}",
        out
    );
}

#[test]
fn external_import_value_read_boxes_to_none() {
    // `from logging import DEBUG` — the name is read as a VALUE in a
    // function body. The import is external (logging is unmodeled), so
    // the read lowers to the boxed None with a warning (external-module
    // divergence), not a bare unresolved identifier. Needs a
    // multi-module conversion (an empty module_defs assumes any import
    // may be a sibling). Previously exercised `from ssl import
    // CERT_REQUIRED`; ssl is now a modeled runtime module (ssl-rustls).
    let src = "from logging import DEBUG\ndef f() -> object:\n    return DEBUG\n";
    let m = parse(src, "logread.py").unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["logread".to_string()], std::rc::Rc::new(m));
    // A second module makes this a multi-module conversion, so the
    // external-import analysis is active (a lone module assumes any
    // import may be a sibling).
    let other = parse("x = 1\n", "other.py").unwrap();
    defs.insert(vec!["other".to_string()], std::rc::Rc::new(other));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(src, "logread.py", options).expect("converts");
    assert!(
        out.contains("stdpython :: PyValue :: None_") || out.contains("stdpython::PyValue::None_"),
        "external-import value read must box to None: {}",
        out
    );
    assert!(
        !out.contains("DEBUG"),
        "bare DEBUG must not leak: {}",
        out
    );
}

#[test]
fn module_level_try_import_error_flattens_imports_to_module_scope() {
    // requests' adapters.py: `try: from urllib3.contrib.socks import
    // SOCKSProxyManager except ImportError: ...` at module level — the
    // ImportError fallback is dropped (static imports), so the import's
    // `use` must land at MODULE scope where call sites outside the
    // wrapper can see it, not inside the lowered try closure.
    let out = compile(
        "try:\n    from socks import SOCKSProxyManager\nexcept ImportError:\n    pass\n\ndef f() -> object:\n    return SOCKSProxyManager\n",
        "tryimp.py",
    );
    assert!(
        out.contains("use crate :: socks :: SOCKSProxyManager") || out.contains("use crate::socks::SOCKSProxyManager"),
        "the import must lower to a module-scope use: {}",
        out
    );
    assert!(
        !out.contains("__rython_try_result"),
        "the dead try wrapper must not lower: {}",
        out
    );
}

#[test]
fn static_initializer_reads_promote_transitively() {
    // urllib3's url.py: `_IPV6_ADDRZ_PAT = r"\[" + _IPV6_PAT + ...` then
    // `_IPV6_ADDRZ_RE = re.compile("^" + _IPV6_ADDRZ_PAT + "$")` — the RE
    // is promoted (functions use it), so its closure references
    // _IPV6_ADDRZ_PAT; that name is only read by OTHER module-level
    // initializers, never a function, yet it must ALSO become a static
    // (a static closure cannot see a module-init local — E0425).
    let out = compile(
        "import re\n_IPV6_PAT = \"x\"\n_IPV6_ADDRZ_PAT = \"[\" + _IPV6_PAT + \"]\"\n_IPV6_ADDRZ_RE = re.compile(\"^\" + _IPV6_ADDRZ_PAT + \"$\")\ndef use(re_: object) -> object:\n    return _IPV6_ADDRZ_RE\n",
        "urltest.py",
    );
    assert!(
        out.contains("pub static _IPV6_ADDRZ_RE") && out.contains("pub static _IPV6_ADDRZ_PAT"),
        "transitive promotion must static both names: {}",
        out
    );
    assert!(
        out.contains("pub static _IPV6_PAT"),
        "the chain's root must promote too: {}",
        out
    );
}

#[test]
fn aliased_external_import_field_call_boxes_field() {
    // urllib3's response.py: `try: import brotlicffi as brotli except
    // ImportError: ...` then `self._obj = brotli.Decompressor()` — the
    // alias resolves to an EXTERNAL module, so the field's value is a
    // foreign object — a boxed PyValue (external-object divergence).
    let out = compile(
        "try:\n    import brotlicffi as brotli\nexcept ImportError:\n    brotli = None\n\nclass ContentDecoder:\n    pass\n\nif brotli is not None:\n    class BrotliDecoder(ContentDecoder):\n        def __init__(self) -> None:\n            self._obj = brotli.Decompressor()\n",
        "brdec.py",
    );
    assert!(
        out.contains("_obj : stdpython :: PyValue") || out.contains("_obj: stdpython::PyValue"),
        "external-import call must box the field: {}",
        out
    );
}

#[test]
fn id_builtin_lowers_to_address_cast() {
    // urllib3's connection.py: `f"<{self} at {id(self):#x}>"` — id(x)
    // lowers to the value's address cast to i64 (identity divergence:
    // the exact number is not CPython's, but the repr shape is).
    let out = compile(
        "class C:\n    def __repr__(self) -> str:\n        return f'<C at {id(self):#x}>'\n",
        "idtest.py",
    );
    assert!(
        out.contains("as * const _ as i64") || out.contains("as *const _ as i64"),
        "id(x) must cast the value's address to i64: {}",
        out
    );
    assert!(
        !out.contains("id (self)"),
        "bare id(self) must not leak: {}",
        out
    );
}

#[test]
fn tuple_import_error_handler_drops_with_import_body() {
    // urllib3's connection.py: `try: import ssl except (ImportError,
    // AttributeError): ssl = None` — ssl resolves (the rustls-backed
    // runtime module), so the tuple handler is the dead fallback of an
    // import that statically succeeds: the try body splices in place,
    // the handler's `ssl = None` never emits, and reads are real runtime
    // paths.
    let src = "try:\n    import ssl\nexcept (ImportError, AttributeError):\n    ssl = None\n\ndef f() -> object:\n    return ssl.CERT_NONE\n";
    let m = parse(src, "ssltry.py").unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["ssltry".to_string()], std::rc::Rc::new(m));
    let other = parse("x = 1\n", "other.py").unwrap();
    defs.insert(vec!["other".to_string()], std::rc::Rc::new(other));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(src, "ssltry.py", options).expect("converts");
    assert!(
        !out.contains("ssl = None"),
        "the dropped fallback's ssl = None must not leak: {}",
        out
    );
    assert!(
        out.contains("ssl :: CERT_NONE"),
        "ssl.CERT_NONE must read the runtime module's constant: {}",
        out
    );
}

#[test]
fn property_getter_setter_pair_renames_setter_and_routes_access() {
    // urllib3's response.py: `@property def url` + `@url.setter def url` —
    // two same-named methods, which Rust forbids (E0428). The setter
    // lowers as `{name}_set`; a read `obj.url` routes to the getter call
    // `obj.url()?`; a store `obj.url = v` routes to the setter call
    // `obj.url_set(v)?`.
    let out = compile(
        "class C:\n    def __init__(self) -> None:\n        self._url = ''\n\n    @property\n    def url(self) -> str:\n        return self._url\n\n    @url.setter\n    def url(self, v: str) -> None:\n        self._url = v\n\ndef get(c: C) -> str:\n    return c.url\n\ndef set(c: C, v: str) -> None:\n    c.url = v\n",
        "propair.py",
    );
    assert!(
        out.contains("fn url_set") && !out.contains("fn url(&mut self, v"),
        "setter must lower as url_set: {}",
        out
    );
    assert!(
        out.contains("c.url()?") || out.contains("c . url () ?"),
        "property read must route to the getter call: {}",
        out
    );
    assert!(
        out.contains("c.url_set(v)?") || out.contains("c . url_set (v) ?"),
        "property store must route to the setter call: {}",
        out
    );
}

#[test]
fn definite_try_except_module_value_promotes_to_static() {
    // requests' compat.py: `try: is_urllib3_1 = int(...) == 1 except
    // (TypeError, AttributeError): is_urllib3_1 = True` — the SAME name is
    // stored once in the try body and once in the handler, so the value is
    // definitely set. It must promote to a LazyLock static (functions read
    // it — E0425 otherwise), with the handler value as the fallback.
    let out = compile(
        "try:\n    is_urllib3_1 = int('2'.split('.')[0]) == 1\nexcept (TypeError, AttributeError):\n    is_urllib3_1 = True\n\ndef f() -> bool:\n    return not is_urllib3_1\n",
        "trydef.py",
    );
    assert!(
        out.contains("pub static is_urllib3_1"),
        "definite try/except value must promote to a static: {}",
        out
    );
    assert!(
        out.contains("(* is_urllib3_1) . clone ()") || out.contains("(*is_urllib3_1).clone()"),
        "function reads must deref-clone the static: {}",
        out
    );
}

#[test]
fn trait_imports_dedupe_across_class_aliases() {
    // urllib3/__init__.py: `from .connectionpool import HTTPConnectionPool,
    // HTTPSConnectionPool` — both classes share the ancestor trait
    // ConnectionPoolTrait; the bring-along must emit it ONCE (E0252
    // duplicate import otherwise).
    let a = parse(
        "class ConnectionPool:\n    pass\nclass HTTPConnectionPool(ConnectionPool):\n    pass\nclass HTTPSConnectionPool(ConnectionPool):\n    pass\n",
        "connectionpool.py",
    )
    .unwrap();
    let b = parse(
        "from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool\n",
        "pkg.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["connectionpool".to_string()], std::rc::Rc::new(a));
    defs.insert(vec!["pkg".to_string()], std::rc::Rc::new(b));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(
        "from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool\n",
        "pkg.py",
        options,
    )
    .expect("converts");
    // Count the EXACT ancestor-trait import (the subclass traits
    // HTTPConnectionPoolTrait/HTTPSConnectionPoolTrait contain the token
    // as a substring, so match the `:: Name` form).
    let needle = ":: ConnectionPoolTrait";
    let count = out.split(needle).count() - 1;
    assert!(
        count == 1,
        "ancestor trait must be imported exactly once: {} ({} mentions)",
        out,
        count
    );
}

#[test]
fn imported_class_method_keyword_values_resolve_in_caller_scope() {
    // requests' sessions.py: `p.prepare(headers=merge_setting(request.
    // headers, self.headers, dict_class=CaseInsensitiveDict))` where p is a
    // models.py PreparedRequest. The keyword VALUE (a call into the CALLER's
    // module) must resolve its callee in the caller's scope — passing the
    // defining module's symbols made merge_setting unknown, so the keyword
    // lowered positionally and the class arg rendered raw (E0423).
    let a = parse(
        "class OrderedDict:\n    pass\n\nclass PreparedRequest:\n    def prepare(self, method, url, headers, params, auth, cookies, hooks, json):\n        return headers\n",
        "models.py",
    )
    .unwrap();
    let b = parse(
        concat!(
            "from .models import PreparedRequest, OrderedDict\n",
            "\n",
            "def merge_setting(request_setting, session_setting, dict_class: type = OrderedDict):\n",
            "    return dict_class\n",
            "\n",
            "class Session:\n",
            "    def __init__(self):\n",
            "        self.headers = None\n",
            "\n",
            "    def prepare_request(self, request):\n",
            "        p = PreparedRequest()\n",
            "        return p.prepare(\n",
            "            method=\"GET\",\n",
            "            url=\"u\",\n",
            "            headers=merge_setting(request.headers, self.headers, dict_class=OrderedDict),\n",
            "            params=None,\n",
            "            auth=None,\n",
            "            cookies=None,\n",
            "            hooks=None,\n",
            "            json=None,\n",
            "        )\n",
        ),
        "sessions.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["models".to_string()], std::rc::Rc::new(a));
    defs.insert(vec!["sessions".to_string()], std::rc::Rc::new(b));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(
        concat!(
            "from .models import PreparedRequest, OrderedDict\n",
            "\n",
            "def merge_setting(request_setting, session_setting, dict_class: type = OrderedDict):\n",
            "    return dict_class\n",
            "\n",
            "class Session:\n",
            "    def __init__(self):\n",
            "        self.headers = None\n",
            "\n",
            "    def prepare_request(self, request):\n",
            "        p = PreparedRequest()\n",
            "        return p.prepare(\n",
            "            method=\"GET\",\n",
            "            url=\"u\",\n",
            "            headers=merge_setting(request.headers, self.headers, dict_class=OrderedDict),\n",
            "            params=None,\n",
            "            auth=None,\n",
            "            cookies=None,\n",
            "            hooks=None,\n",
            "            json=None,\n",
            "        )\n",
        ),
        "sessions.py",
        options,
    )
    .expect("converts");
    // The merge_setting call's third argument (dict_class=OrderedDict) must
    // lower to the boxed None — the class-as-value divergence — not a raw
    // class-name token (E0423 `expected value, found struct`). The temp
    // prelude binds it as __rython_arg_2 before the call.
    let call_idx = out.find("merge_setting (").expect("call present");
    let prelude_start = out[..call_idx].rfind("let __rython_arg_").map(|i| i).unwrap_or(0);
    let block = &out[prelude_start..call_idx + 200];
    assert!(
        block.contains("stdpython :: PyValue :: None_") || block.contains("PyValue::None_"),
        "class-value arg for dict_class must box to None, got: {}",
        block
    );
    assert!(
        !block.contains("OrderedDict"),
        "class name must not render raw as an argument: {}",
        block
    );
}

#[test]
fn sibling_import_of_reexported_and_submodule_names_is_kept() {
    // urllib3's connection.py: `from .util import SKIP_HEADER,
    // SKIPPABLE_HEADERS, connection, ssl_` — SKIP_HEADER/SKIPPABLE_HEADERS
    // are RE-EXPORTED by util/__init__.py (`from .request import ...`),
    // and connection/ssl_ are SUBMODULES (util/connection.rs,
    // util/ssl_.rs). The f103e59 "drop ungenerated sibling imports" check
    // must NOT drop them (E0425/E0433 otherwise).
    let a = parse(
        concat!(
            "def make_headers():
",
            "    return {}
",
            "SKIP_HEADER = \"@@@SKIP_HEADER@@@\"
",
            "SKIPPABLE_HEADERS = [SKIP_HEADER]
",
        ),
        "request.py",
    )
    .unwrap();
    let b = parse(
        concat!(
            "from .request import SKIP_HEADER, SKIPPABLE_HEADERS, make_headers
",
        ),
        "util/__init__.py",
    )
    .unwrap();
    // The SUBMODULES the `from .util import connection, ssl_` names
    // resolve to (urllib3's util/connection.py, util/ssl_.py).
    let conn_sub = parse("def create_connection(addr):\n    return addr\n", "util/connection.py").unwrap();
    let ssl_sub = parse("ALPN_PROTOCOLS = [\"h2\"]\n", "util/ssl_.py").unwrap();
    let c = parse(
        concat!(
            "from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection, ssl_
",
            "
",
            "def f(headers):
",
            "    return SKIP_HEADER
",
        ),
        "connection.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["util".to_string(), "request".to_string()], std::rc::Rc::new(a));
    defs.insert(vec!["util".to_string()], std::rc::Rc::new(b));
    defs.insert(
        vec!["util".to_string(), "connection".to_string()],
        std::rc::Rc::new(conn_sub),
    );
    defs.insert(vec!["util".to_string(), "ssl_".to_string()], std::rc::Rc::new(ssl_sub));
    defs.insert(vec!["connection".to_string()], std::rc::Rc::new(c));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(
        concat!(
            "from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection, ssl_\n",
            "\n",
            "def f(headers):\n",
            "    return SKIP_HEADER\n",
        ),
        "connection.py",
        options,
    )
    .expect("converts");
    // The SKIP_HEADER re-export import must be KEPT (its use chain
    // resolves), and the submodule imports too.
    assert!(
        out.contains(":: util :: SKIP_HEADER"),
        "re-exported SKIP_HEADER import must be kept: {}",
        out
    );
    assert!(
        out.contains(":: util :: connection") && out.contains(":: util :: ssl_"),
        "submodule imports must be kept: {}",
        out
    );
    assert!(
        out.contains("SKIP_HEADER"),
        "SKIP_HEADER must resolve at the use site: {}",
        out
    );
}


#[test]
fn import_reexport_of_stdpython_module_is_kept() {
    // requests' models.py: `from .compat import json as complexjson` where
    // compat.py does `import json` (stdlib). The f103e59 sibling-import
    // drop must not drop the re-exported stdpython-module name (E0425
    // `cannot find value complexjson` otherwise).
    let a = parse("import json\n", "compat.py").unwrap();
    let b = parse(
        "from .compat import json as complexjson\n\ndef f():\n    return complexjson.loads(\"{}\")\n",
        "models.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["compat".to_string()], std::rc::Rc::new(a));
    defs.insert(vec!["models".to_string()], std::rc::Rc::new(b));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(
        "from .compat import json as complexjson\n\ndef f():\n    return complexjson.loads(\"{}\")\n",
        "models.py",
        options,
    )
    .expect("converts");
    // The import must be kept and the alias must resolve.
    assert!(
        out.contains("complexjson"),
        "aliased stdpython re-export import must be kept: {}",
        out
    );
}

#[test]
fn module_path_member_read_missing_item_boxes_to_none() {
    // urllib3's pyopenssl.py: `util.ssl_.PROTOCOL_TLS` — a module-path
    // member read where the generated ssl_ module has no PROTOCOL_TLS
    // item (it is an external ssl constant). The read must box to None
    // (E0425 otherwise).
    let a = parse("def ssl_wrap_socket():\n    return 1\n", "ssl_.py").unwrap();
    let b = parse(
        concat!(
            "from . import util\n",
            "\n",
            "_versions = {\n",
            "    util.ssl_.PROTOCOL_TLS: 1,\n",
            "}\n",
        ),
        "pyopenssl.py",
    )
    .unwrap();
    let util_init = parse("from . import ssl_\n", "util/__init__.py").unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["util".to_string(), "ssl_".to_string()], std::rc::Rc::new(a));
    defs.insert(vec!["util".to_string()], std::rc::Rc::new(util_init));
    defs.insert(vec!["pyopenssl".to_string()], std::rc::Rc::new(b));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(
        "from . import util\n\n_versions = {\n    util.ssl_.PROTOCOL_TLS: 1,\n}\n",
        "pyopenssl.py",
        options,
    )
    .expect("converts");
    // The missing member must not render as a raw `::PROTOCOL_TLS` path.
    assert!(
        !out.contains("PROTOCOL_TLS"),
        "missing module member must not render as a raw path: {}",
        out
    );
}


#[test]
fn stdpython_module_reexport_via_sibling_aliases_to_runtime() {
    // requests' models.py: `from .compat import json as complexjson` where
    // compat.py does `import json` (stdlib) inside a try/except. The
    // generated compat.rs has no `json` item, so the import must route to
    // the runtime module (`use <stdpython>::json as complexjson;`) —
    // otherwise `complexjson.dumps(...)` is E0425.
    let a = parse(
        "import json\n\ndef to_native_string(s, encoding):\n    return s\n",
        "compat.py",
    )
    .unwrap();
    let b = parse(
        "from .compat import json as complexjson\n\ndef f():\n    return complexjson.dumps({})\n",
        "models.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["compat".to_string()], std::rc::Rc::new(a));
    defs.insert(vec!["models".to_string()], std::rc::Rc::new(b));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(
        "from .compat import json as complexjson\n\ndef f():\n    return complexjson.dumps({})\n",
        "models.py",
        options,
    )
    .expect("converts");
    assert!(
        out.contains("as complexjson"),
        "aliased stdpython-module reexport must route to the runtime module: {}",
        out
    );
}

// ---------------------------------------------------------------------------
// threading / socket / urllib.request / alloc-tier io
// ---------------------------------------------------------------------------

#[test]
fn threading_thread_lowers_target_and_args_statically() {
    // threading.Thread(target=f, args=(...)) resolves the callable at
    // conversion time (the functools.partial model): the body closure
    // calls the target through the normal lowering, and Name-arguments
    // are cloned so the caller's bindings stay usable after start().
    let out = compile(
        concat!(
            "import threading\n",
            "\n",
            "def worker(name: str, n: int) -> None:\n",
            "    print(name, n)\n",
            "\n",
            "def run() -> None:\n",
            "    label = \"x\"\n",
            "    t = threading.Thread(target=worker, args=(label, 2))\n",
            "    t.start()\n",
            "    t.join()\n",
            "    print(label)\n",
        ),
        "threads.py",
    );
    assert!(
        out.contains("threading :: Thread :: new"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("let label = (label) . clone ()"),
        "args must be cloned into the closure: {}",
        out
    );
    assert!(
        out.contains("report_thread_exception"),
        "thread bodies report unhandled exceptions: {}",
        out
    );
}

#[test]
fn threading_thread_unsupported_shapes_error_loudly() {
    // A lambda target is the callable-as-value divergence: loud.
    let err = compile_err(
        "import threading\nt = threading.Thread(target=lambda: 1)\n",
        "tl.py",
    );
    assert!(err.contains("target"), "error: {}", err);
    // Unknown keywords never silently drop.
    let err = compile_err(
        concat!(
            "import threading\n",
            "def w() -> None:\n",
            "    pass\n",
            "t = threading.Thread(target=w, kwargs={\"a\": 1})\n",
        ),
        "tk.py",
    );
    assert!(err.contains("not supported"), "error: {}", err);
}

#[test]
fn with_lock_lowers_to_the_raii_guard() {
    // `with lock:` must acquire and release (Python's __enter__/__exit__),
    // not silently bind-and-drop: the guard acquires now and releases on
    // Drop, exception-safe through `?` unwinding.
    let out = compile(
        concat!(
            "import threading\n",
            "\n",
            "def run() -> None:\n",
            "    lock = threading.Lock()\n",
            "    with lock:\n",
            "        print(\"held\")\n",
        ),
        "wl.py",
    );
    assert!(
        out.contains("py_guard () ?"),
        "with-lock must lower to the RAII guard: {}",
        out
    );

    // The `as` form binds __enter__'s True in Python — no honest lowering.
    let err = compile_err(
        concat!(
            "import threading\n",
            "def run() -> None:\n",
            "    lock = threading.Lock()\n",
            "    with lock as got:\n",
            "        print(got)\n",
        ),
        "wlas.py",
    );
    assert!(err.contains("not supported"), "error: {}", err);
}

#[test]
fn socket_calls_thread_the_result_question_mark() {
    // socket.socket() and the socket-object methods return Result
    // (network errors are catchable OSError kinds), so every call site
    // threads `?`.
    let out = compile(
        concat!(
            "import socket\n",
            "\n",
            "def run() -> None:\n",
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n",
            "    s.connect((\"127.0.0.1\", 80))\n",
            "    s.sendall(\"hi\".encode(\"utf-8\"))\n",
            "    data = s.recv(64)\n",
            "    s.close()\n",
        ),
        "sock.py",
    );
    assert!(
        out.contains("socket :: socket (socket :: AF_INET , socket :: SOCK_STREAM) ?"),
        "generated: {}",
        out
    );
    assert!(out.contains(". connect ("), "generated: {}", out);
    assert!(
        out.contains(". recv (64) ?"),
        "recv must thread ?: {}",
        out
    );
    // sendall passes the payload by reference (the runtime takes
    // AsRef<[u8]>), so a named buffer survives its send.
    assert!(
        out.contains(". sendall (& ("),
        "generated: {}",
        out
    );
}

#[test]
fn urllib_request_urlopen_lowers_with_question_mark() {
    let out = compile(
        concat!(
            "import urllib.request\n",
            "\n",
            "def run() -> None:\n",
            "    resp = urllib.request.urlopen(\"http://example.com/\")\n",
            "    print(resp.status)\n",
            "    data = resp.read()\n",
            "    print(resp.getcode())\n",
        ),
        "fetch.py",
    );
    assert!(
        out.contains("urllib :: request :: urlopen (\"http://example.com/\") ?"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("resp . status"),
        "status is a field read: {}",
        out
    );
    assert!(
        out.contains(". read () ?"),
        "read threads ?: {}",
        out
    );
}

#[test]
fn nostd_threading_socket_urllib_error_loudly_but_io_converts() {
    // The OS-backed modules have no no_std lowering: loud at conversion.
    for src in [
        "import threading\n",
        "import socket\n",
        "import urllib.request\n",
    ] {
        let err = compile_nostd(src, "imp.py").expect_err("std-tier import must fail");
        assert!(err.contains("std tier"), "{:?}: {}", src, err);
    }
    // io's in-memory buffers are pure alloc: the no_std profile keeps
    // them (this IS no_std file I/O).
    let out = compile_nostd(
        concat!(
            "import io\n",
            "\n",
            "def run() -> str:\n",
            "    buf = io.StringIO()\n",
            "    buf.write(\"hello\")\n",
            "    return buf.getvalue()\n",
        ),
        "nostd_io.py",
    )
    .expect("io.StringIO is alloc-tier");
    assert!(out.contains("io :: StringIO"), "generated: {}", out);
}

#[test]
fn bytesio_lowers_to_the_runtime_buffer() {
    // io.BytesIO is a real binary buffer now (arity-split like StringIO),
    // not a boxed PyValue drop.
    let out = compile(
        concat!(
            "import io\n",
            "\n",
            "def run() -> None:\n",
            "    b = io.BytesIO(b\"seed\")\n",
            "    b.write(b\"!\")\n",
            "    data = b.getvalue()\n",
            "    e = io.BytesIO()\n",
        ),
        "bio.py",
    );
    assert!(
        out.contains("io :: BytesIO_seeded"),
        "generated: {}",
        out
    );
    assert!(out.contains("io :: BytesIO ()"), "generated: {}", out);
    assert!(
        !out.contains("PyValue :: None_"),
        "BytesIO must not box away: {}",
        out
    );
}

#[test]
fn with_lock_on_an_annotated_parameter_lowers_to_the_guard() {
    // Devin review on PR #144: a lock received as a FUNCTION PARAMETER —
    // the exact pass-a-lock-to-a-worker pattern — must also lower
    // `with lock:` to the RAII guard, not the silent bind-and-drop.
    // Both the dotted and the from-import annotation spellings classify.
    let out = compile(
        concat!(
            "import threading\n",
            "\n",
            "def crit(lock: threading.Lock, n: int) -> None:\n",
            "    with lock:\n",
            "        print(n)\n",
        ),
        "wlp.py",
    );
    assert!(
        out.contains("py_guard () ?"),
        "parameter lock must lower to the RAII guard: {}",
        out
    );

    let out = compile(
        concat!(
            "from threading import Semaphore\n",
            "\n",
            "def crit(sem: Semaphore) -> None:\n",
            "    with sem:\n",
            "        print(\"held\")\n",
        ),
        "wsp.py",
    );
    assert!(
        out.contains("py_guard () ?"),
        "from-import annotated semaphore must lower to the RAII guard: {}",
        out
    );
}

#[test]
fn threading_thread_daemon_flag_lowers_from_bool_constants() {
    // Devin review round 3 on PR #144: the parser represents True/False
    // as bool CONSTANTS (not Names), so daemon=True was wrongly rejected
    // by a Name-only match. Both values must lower into Thread::new's
    // daemon argument.
    let out = compile(
        concat!(
            "import threading\n",
            "\n",
            "def w() -> None:\n",
            "    pass\n",
            "\n",
            "def run() -> None:\n",
            "    t = threading.Thread(target=w, daemon=True)\n",
            "    t.start()\n",
            "    u = threading.Thread(target=w, daemon=False)\n",
            "    u.start()\n",
            "    u.join()\n",
        ),
        "daemon.py",
    );
    assert!(
        out.contains("Thread :: new (\"w\" , true ,"),
        "daemon=True must reach Thread::new: {}",
        out
    );
    assert!(
        out.contains("Thread :: new (\"w\" , false ,"),
        "daemon=False must reach Thread::new: {}",
        out
    );
}

// ---- Module attribute protocol (PEP 562) — issue #119 ----

#[test]
fn module_getattr_is_a_loud_error() {
    // A module-level `__getattr__` (dateutil's lazy submodule loading) is
    // the PEP 562 dynamic-attribute fallback: rython resolves module
    // attributes statically, so the definition is a loud error naming the
    // dunder — not an inference error, and never a silently dead function.
    let err = compile_err(
        concat!(
            "def __getattr__(name):\n",
            "    raise AttributeError(name)\n",
        ),
        "modgetattr.py",
    );
    assert!(err.contains("module attribute protocol"), "err: {}", err);
    assert!(err.contains("__getattr__"), "err: {}", err);
    assert!(err.contains("issue #119"), "err: {}", err);
}

#[test]
fn module_dir_is_a_loud_error() {
    let err = compile_err("def __dir__():\n    return []\n", "moddir.py");
    assert!(err.contains("module attribute protocol"), "err: {}", err);
    assert!(err.contains("__dir__"), "err: {}", err);
}

// ---- Mutable module globals (`global` writes) — issue #115 ----

#[test]
fn global_write_lowers_to_a_mutable_static() {
    // A module scalar written through `global` becomes `static name:
    // Mutex<T>`: writes go through py_global_write, reads through
    // py_global_read, and the write-drop warning is gone.
    let (out, warnings) = compile_with_warnings(
        concat!(
            "count = 0\n",
            "def bump():\n",
            "    global count\n",
            "    count += 1\n",
            "def peek() -> int:\n",
            "    return count\n",
        ),
        "global_mut.py",
    );
    assert!(
        out.contains("pub static count : std :: sync :: Mutex < i64 >"),
        "generated: {}",
        out
    );
    assert!(out.contains("py_global_write"), "generated: {}", out);
    assert!(out.contains("py_global_read (& count)"), "generated: {}", out);
    assert!(
        warnings.iter().all(|w| !w.contains("writes to module-level name")),
        "the supported write must not warn: {:?}",
        warnings
    );
}

#[test]
fn global_none_singleton_boxes_to_a_pyvalue_static() {
    // The None-initialized singleton pattern (boto3's DEFAULT_SESSION):
    // the static boxes to Mutex<PyValue>; scalar stores wrap in
    // PyValue::from.
    let out = compile(
        concat!(
            "DEFAULT = None\n",
            "def setup(v: int):\n",
            "    global DEFAULT\n",
            "    DEFAULT = v\n",
            "def is_set() -> bool:\n",
            "    return DEFAULT is not None\n",
        ),
        "global_none.py",
    );
    assert!(
        out.contains("pub static DEFAULT : std :: sync :: Mutex < stdpython :: PyValue >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("PyValue :: from (v)"),
        "scalar stores must box: {}",
        out
    );
}

#[test]
fn global_shadowed_by_a_plain_local_disqualifies_the_static() {
    // A function that binds the name WITHOUT `global` has a plain local;
    // the name must not become a mutable static (the local read would be
    // misread as the module global) — the write keeps the documented
    // drop-with-warning divergence.
    let (out, warnings) = compile_with_warnings(
        concat!(
            "flag = False\n",
            "def set_local():\n",
            "    flag = True\n",
            "def set_global():\n",
            "    global flag\n",
            "    flag = True\n",
        ),
        "global_shadow.py",
    );
    assert!(!out.contains("py_global_write"), "generated: {}", out);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("writes to module-level name `flag`")),
        "the unsupported write must still warn: {:?}",
        warnings
    );
}

#[test]
fn deepcopy_memo_kwarg_is_dropped_with_a_warning() {
    // Issue #154: copy.deepcopy(x, memo=...) — boto3's dynamodb transform
    // passes a forgetful memo dict. rython's value semantics already copy
    // everything fresh, so the kwarg drops with a -W note instead of the
    // unexpected-keyword error.
    let (out, warnings) = compile_with_warnings(
        concat!(
            "import copy\n",
            "def f(params: dict[str, int]) -> dict[str, int]:\n",
            "    return copy.deepcopy(params, memo={})\n",
        ),
        "dc_memo.py",
    );
    assert!(out.contains("copy :: deepcopy"), "generated: {}", out);
    assert!(
        warnings.iter().any(|w| w.contains("deepcopy(memo=...)")),
        "the dropped memo must be reported through -W: {:?}",
        warnings
    );
}

#[test]
fn iter_sentinel_outside_a_for_loop_is_loud() {
    // Issue #155: the two-argument iter() lowers only in for-loop
    // iterable position; a bare value would need an iterator object.
    let err = compile_err(
        concat!(
            "def f() -> str:\n",
            "    return \"\"\n",
            "it = iter(f, \"\")\n",
        ),
        "itersent_bare.py",
    );
    assert!(err.contains("for-loop iterable"), "err: {err}");
    assert!(err.contains("issue #155"), "err: {err}");
}

// ---- Variadic parameters (issue #120) ----

#[test]
fn varargs_lower_to_a_boxed_vector() {
    // Issue #120: `*args` is Vec<PyValue>; extras box at the call site,
    // an empty call still passes the vector, and `f(*args)` forwards it.
    let out = compile(
        concat!(
            "def tag(*args) -> int:\n",
            "    return len(args)\n",
            "def fwd(*args) -> int:\n",
            "    return tag(*args)\n",
            "def run() -> int:\n",
            "    return tag(1, \"x\") + tag() + fwd(True)\n",
        ),
        "varargs.py",
    );
    assert!(
        out.contains("args : Vec < stdpython :: PyValue >"),
        "generated: {}",
        out
    );
    assert!(out.contains("PyValue :: from"), "generated: {}", out);
    assert!(out.contains("tag (vec ! [])"), "generated: {}", out);
    assert!(out.contains("__rython_varargs"), "generated: {}", out);
}

#[test]
fn keyword_to_pure_varargs_callee_is_a_python_type_error() {
    // Python raises TypeError for `f(x=1)` when f takes only *args; the
    // conversion reports the same shape loudly.
    let err = compile_err(
        concat!(
            "def f(*args) -> int:\n",
            "    return len(args)\n",
            "f(x=1)\n",
        ),
        "vakw.py",
    );
    assert!(err.contains("unexpected keyword argument"), "err: {err}");
}

// ---- Issue #133: return-unification boxing agrees with the signature ----

#[test]
fn generic_mixed_returns_box_and_wrap() {
    // Inference: `return val == "yes"` on one path and `return val` on the
    // other (botocore's ensure_boolean shape) unify to the boxed PyValue —
    // the signature AND the body must agree: every return site wraps in
    // PyValue::from, and the where clause carries the From bounds rustc
    // will not invent (PyValue: From<T> for the bare parameter return,
    // PyValue: From<<T as PyEq<&'static str>>::Output> for the comparison
    // return).
    let out = compile(
        concat!(
            "def flagify(val):\n",
            "    if val:\n",
            "        return val == \"yes\"\n",
            "    return val\n",
        ),
        "flagify.py",
    );
    assert!(
        out.contains("Result < stdpython :: PyValue , PyException >"),
        "mixed returns must box the signature: {}",
        out
    );
    assert!(
        out.contains("PyValue :: from (val)"),
        "the bare-parameter return must wrap: {}",
        out
    );
    assert!(
        out.contains("stdpython :: PyValue : From < T >"),
        "the bare type-variable return needs a From bound: {}",
        out
    );
    assert!(
        out.contains(":: Output >"),
        "the comparison return needs a From bound on the operator Output: {}",
        out
    );
    // The str-literal comparison bound must use the runtime's
    // `impl<'a> PyEq<&'a str>` shape (a `String` here would name a trait
    // instantiation the emitted `py_eq(&("yes"))` call never exercises).
    assert!(
        out.contains("PyEq < & 'static str >"),
        "the str-literal comparison must bound PyEq<&'static str>: {}",
        out
    );
}

#[test]
fn annotated_params_mixed_literal_returns_box() {
    // The NON-generic path (annotated parameter, unannotated return):
    // `return 1` / `return None` has no single concrete type — the
    // signature boxes to PyValue and the returns wrap (previously the
    // signature said Result<(), _> while the body returned Ok(1)/Ok(None)).
    let out = compile(
        concat!(
            "def pick(flag: bool):\n",
            "    if flag:\n",
            "        return 1\n",
            "    return None\n",
        ),
        "pick.py",
    );
    assert!(
        out.contains("Result < stdpython :: PyValue , PyException >"),
        "mixed literal/None returns must box the signature: {}",
        out
    );
    assert!(
        out.contains("PyValue :: from (1)"),
        "the literal return must wrap: {}",
        out
    );
    assert!(
        out.contains("PyValue :: None_"),
        "the None return must box: {}",
        out
    );
}

#[test]
fn partial_literal_return_boxes_with_none_tail() {
    // A value return on one path and a FALL-THROUGH on the other returns
    // `1 | None` in Python: the signature boxes and the implicit tail is
    // the boxed None.
    let out = compile(
        concat!(
            "def partial(flag: bool):\n",
            "    if flag:\n",
            "        return 1\n",
        ),
        "partial.py",
    );
    assert!(
        out.contains("Result < stdpython :: PyValue , PyException >"),
        "a partial literal return must box the signature: {}",
        out
    );
    assert!(
        out.contains("Ok (PyValue :: None_)"),
        "the fall-through tail must be the boxed None: {}",
        out
    );
}

#[test]
fn same_kind_literal_returns_stay_concrete() {
    // Consistent literal returns keep their concrete type — boxing is only
    // for mixes the concrete system cannot express.
    let out = compile(
        concat!(
            "def same(flag: bool):\n",
            "    if flag:\n",
            "        return 1\n",
            "    return 2\n",
        ),
        "same.py",
    );
    assert!(
        out.contains("Result < i64 , PyException >"),
        "consistent literal returns must stay concrete: {}",
        out
    );
    assert!(
        !out.contains("PyValue"),
        "no boxing for a consistent return type: {}",
        out
    );
}

#[test]
fn mixed_element_list_returns_get_vec_pyvalue_signature() {
    // A returned list literal whose elements mix boxable kinds renders
    // element-boxed (`vec![PyValue::from(1), PyValue::from("a")]` — issue
    // #130); the SIGNATURE must agree (previously it said Result<(), _>).
    let out = compile(
        concat!(
            "def mixed_list(flag: bool):\n",
            "    if flag:\n",
            "        return [1, \"a\"]\n",
            "    return [2, \"b\"]\n",
        ),
        "mixedlist.py",
    );
    assert!(
        out.contains("Result < Vec < stdpython :: PyValue >"),
        "element-boxed list returns must box the signature's element: {}",
        out
    );
}

// ---- Issue #161: boxed fallback for isinstance dispatch ----

#[test]
fn unknown_typed_argument_dispatches_through_the_router() {
    // An isinstance-dispatched call whose argument has NO statically-known
    // type (`path` reassigned through an untyped call — botocore
    // configloader's `path = os.path.expandvars(path)`) routes through
    // the dynamic router at runtime instead of failing loudly. The
    // reassigned-from-a-call parameter value-pins to the boxed PyValue:
    // `impl Into<stdpython::PyValue>` with a boxing prologue, stores
    // wrapped in PyValue::from — call sites keep passing plain values.
    let out = compile(
        concat!(
            "def _unicode_path(path):\n",
            "    if isinstance(path, str):\n",
            "        return path\n",
            "    return path.decode(\"utf-8\", \"replace\")\n",
            "\n",
            "def norm(p):\n",
            "    return p\n",
            "\n",
            "def load(path):\n",
            "    path = norm(path)\n",
            "    return _unicode_path(path)\n",
        ),
        "router161.py",
    );
    assert!(
        out.contains("load (path : impl Into < stdpython :: PyValue >)"),
        "the value-pinned parameter must take impl Into<PyValue>: {}",
        out
    );
    assert!(
        out.contains("let mut path : stdpython :: PyValue = path . into () ;"),
        "the prologue must box the parameter: {}",
        out
    );
    // The dispatch site calls the ROUTER (the original name), not a
    // compile-time morph and not a loud error.
    assert!(
        out.contains("_unicode_path (path)"),
        "the unknown-typed call must go through the router: {}",
        out
    );
    // The parameter takes no dead type variable: its type is concrete.
    assert!(
        !out.contains("load < T >"),
        "a value-pinned parameter must not leave a generic leftover: {}",
        out
    );
}

#[test]
fn decode_residual_morph_derives_and_bounds_pydecode() {
    // The RESIDUAL morph of a str-tested dispatcher whose fall-through
    // decodes (`return path.decode(enc, 'replace')`) types as String —
    // only bytes has decode in Python 3 — so the dynamic router derives,
    // and the morph's parameter carries the PyDecode bound the boxed
    // Other arm (PyValue) and static bytes callers both satisfy.
    let out = compile(
        concat!(
            "def _unicode_path(path):\n",
            "    if isinstance(path, str):\n",
            "        return path\n",
            "    return path.decode(\"utf-8\", \"replace\")\n",
        ),
        "residual161.py",
    );
    assert!(
        out.contains("T : PyDecode"),
        "the residual morph must bound PyDecode: {}",
        out
    );
    assert!(
        out.contains("py_decode (\"utf-8\" , \"replace\")"),
        "two-positional decode must lower through the trait: {}",
        out
    );
    // The router exists (the residual's return derived as String).
    assert!(
        out.contains("pub fn _unicode_path (path : impl Into < UnicodePathArg > ,)"),
        "the router must derive for the decode residual: {}",
        out
    );
}

#[test]
fn two_arg_decode_on_bytes_receiver_lowers_through_pydecode() {
    // bytes.decode(enc, errors) with BOTH positional arguments (botocore
    // configloader passes the errors mode positionally): the PyDecode
    // lowering ('replace' follows CPython for utf-8).
    let out = compile(
        "def f(b: bytes) -> str:\n    return b.decode(\"utf-8\", \"replace\")\n",
        "decode2arg.py",
    );
    assert!(
        out.contains("py_decode (\"utf-8\" , \"replace\")"),
        "two-positional decode must lower through the trait: {}",
        out
    );
}

// ---- Issue #133 (completion): sum() on generic arguments ----

#[test]
fn generic_sum_return_projects_the_output() {
    // `return sum(p)` on an unannotated parameter: the bound is the
    // associated-Output PySum and the return type is its projection, so
    // one generic function serves int AND float lists.
    let out = compile("def total(xs):\n    return sum(xs)\n", "sumret.py");
    assert!(
        out.contains("T : PySum"),
        "sum on a parameter must bound PySum: {}",
        out
    );
    assert!(
        out.contains("Result < < T as PySum > :: Output , PyException >"),
        "the return must project the Output: {}",
        out
    );
}

#[test]
fn sum_into_a_typed_slot_pins_the_output() {
    // The issue's calc: `chunks = [sum(xs)]` where chunks was seeded
    // Vec<i64> by `chunks.append(len(xs))` — the slot's element type
    // pins the trait's Output (`T: PySum<Output = i64>`), which rustc
    // will not infer across the generic boundary; the plain PySum bound
    // is subsumed.
    let out = compile(
        concat!(
            "def calc(xs):\n",
            "    chunks = []\n",
            "    chunks.append(len(xs))\n",
            "    chunks = [sum(xs)]\n",
            "    return chunks\n",
        ),
        "sumpin.py",
    );
    assert!(
        out.contains("T : PySum < Output = i64 >"),
        "the typed slot must pin the Output: {}",
        out
    );
    assert!(
        !out.contains("T : PySum ,"),
        "the plain bound is subsumed by the pinned one: {}",
        out
    );
}

// ---- Issue #137: build-sweep round (top-5 packages) ----

#[test]
fn with_block_returns_feed_m4_callee_inference() {
    // requests' api.py: `get` calls `request`, whose only return sits
    // inside a `with` block — the M4 callee-return collector previously
    // missed it ("no return statements") and the whole package failed to
    // convert.
    let out = compile(
        concat!(
            "def request(method, url):\n",
            "    with open(url) as session:\n",
            "        return method\n",
            "\n",
            "def get(url):\n",
            "    return request(\"GET\", url)\n",
        ),
        "withret.py",
    );
    assert!(out.contains("fn get"), "conversion must succeed: {}", out);
}

#[test]
fn attribute_of_call_field_boxes_to_pyvalue() {
    // requests' cookies.MockRequest: `self.type = urlparse(...).scheme` —
    // a dynamic member of a foreign object — types the field as the
    // boxed PyValue instead of failing the class.
    let out = compile(
        concat!(
            "class MockRequest:\n",
            "    def __init__(self, url: str):\n",
            "        self.kind = open(url).scheme\n",
        ),
        "mockreq.py",
    );
    assert!(
        out.contains("kind : stdpython :: PyValue"),
        "the field must box: {}",
        out
    );
}

#[test]
fn callee_element_operands_map_to_the_fresh_iterate_element() {
    // requests' cookiejar_from_dict/merge_cookies: the callee subscripts
    // its parameter by its own LOOP ELEMENT (`cookie_dict[name]` under
    // `for name in cookie_dict`); propagating that requirement into a
    // caller previously failed with "parameter `name` used as an operand
    // but has no type".
    let out = compile(
        concat!(
            "def from_dict(cookie_dict):\n",
            "    total = 0\n",
            "    for name in cookie_dict:\n",
            "        total = total + cookie_dict[name]\n",
            "    return total\n",
            "\n",
            "def merge(cookies):\n",
            "    return from_dict(cookies)\n",
        ),
        "eltmap.py",
    );
    assert!(out.contains("fn merge"), "conversion must succeed: {}", out);
}

#[test]
fn class_computed_constants_are_module_level_statics() {
    // urllib3's RequestMethods._encode_url_methods: associated statics
    // are not legal Rust — the LazyLock lives at module level under the
    // class-mangled name, and `self.X` reads deref-clone it.
    let out = compile(
        concat!(
            "class RequestMethods:\n",
            "    _encode_url_methods = frozenset([\"DELETE\", \"GET\"])\n",
            "    def uses_url(self, method: str) -> bool:\n",
            "        return method in self._encode_url_methods\n",
        ),
        "clsconst.py",
    );
    assert!(
        out.contains("pub static RequestMethods__encode_url_methods"),
        "the constant must be a module-level class-mangled static: {}",
        out
    );
    assert!(
        out.contains("RequestMethods :: _encode_url_methods ()"),
        "the self-read must call the associated accessor: {}",
        out
    );
    // Nothing static remains inside the impl block.
    assert!(
        !out.contains("impl RequestMethods { pub static"),
        "no associated statics: {}",
        out
    );
}

#[test]
fn stdlib_exception_aliases_canonicalize_on_raise_and_except() {
    // urllib3's pyopenssl: `from socket import timeout` then
    // `raise timeout(...)`; response.py catches it as an aliased import.
    // CPython aliases socket.timeout to TimeoutError, so both sides
    // lower to the canonical builtin and the hierarchy walk matches.
    let out = compile(
        concat!(
            "from socket import timeout as SocketTimeout\n",
            "\n",
            "def read(n: int) -> int:\n",
            "    try:\n",
            "        if n > 0:\n",
            "            raise SocketTimeout(\"The read operation timed out\")\n",
            "    except SocketTimeout:\n",
            "        return -1\n",
            "    return n\n",
        ),
        "socktimeout.py",
    );
    assert!(
        out.contains("PyException :: new (\"TimeoutError\""),
        "the raise must carry the canonical builtin: {}",
        out
    );
    assert!(
        out.contains("matches (\"TimeoutError\")"),
        "the handler must match the canonical builtin: {}",
        out
    );
}

#[test]
fn urllib_calls_without_runtime_items_drop_boxed() {
    // urllib3's `urlencode(fields)` under `from urllib.parse import
    // urlencode`: no runtime item exists — the call drops to the boxed
    // None with the divergence warning instead of rendering a
    // `urlencode::new(...)` class construction.
    let out = compile(
        concat!(
            "from urllib.parse import urlencode\n",
            "\n",
            "def q(fields: str) -> None:\n",
            "    x = urlencode(fields)\n",
        ),
        "urlenc.py",
    );
    assert!(
        out.contains("PyValue :: None_"),
        "the call must drop boxed: {}",
        out
    );
    assert!(
        !out.contains("urlencode :: new"),
        "no class construction for a dropped import: {}",
        out
    );
}

// ---- ssl module wiring: rustls-backed runtime surface (issue #137) ----

#[test]
fn ssl_imports_resolve_to_the_runtime_module() {
    // `import ssl` / `from ssl import ...` resolve under stdpython's
    // rustls-backed ssl module (the ssl-rustls feature, on by default).
    // From-imports are `pub use` so sibling re-export chains resolve
    // (E0603 otherwise), and TLSVersion attribute chains are paths.
    let out = compile(
        "import ssl\n\
         from ssl import CERT_REQUIRED, SSLContext, TLSVersion\n\
         \n\
         def make() -> int:\n\
         \x20   ctx = SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n\
         \x20   ctx.minimum_version = TLSVersion.TLSv1_2\n\
         \x20   return CERT_REQUIRED\n",
        "sslmod.py",
    );
    assert!(
        out.contains("pub use stdpython :: ssl :: SSLContext"),
        "from-ssl imports must be pub-use of the runtime module: {}",
        out
    );
    assert!(
        out.contains("ssl :: PROTOCOL_TLS_CLIENT"),
        "qualified ssl constants must render as runtime paths: {}",
        out
    );
    assert!(
        out.contains("TLSVersion :: TLSv1_2"),
        "TLSVersion members must render as paths, not field reads: {}",
        out
    );
}

#[test]
fn resolved_import_try_splices_body_and_drops_dead_handler() {
    // The dual of the failed-import fold: a try/except-ImportError whose
    // imports ALL resolve statically always takes the try path — the
    // body splices in place and the handler (urllib3 connection.py's
    // fallback BaseSSLError class) never emits. The alias assign
    // `BaseSSLError = ssl.SSLError` registers an exception alias, so
    // except sites canonicalize to the runtime's SSLError tag.
    let out = compile(
        "try:\n\
         \x20   import ssl\n\
         \n\
         \x20   BaseSSLError = ssl.SSLError\n\
         except (ImportError, AttributeError):\n\
         \x20   ssl = None\n\
         \n\
         \x20   class BaseSSLError(Exception):\n\
         \x20       pass\n\
         \n\
         def f() -> int:\n\
         \x20   try:\n\
         \x20       return 1\n\
         \x20   except BaseSSLError:\n\
         \x20       return 2\n",
        "sslfold.py",
    );
    assert!(
        !out.contains("struct BaseSSLError"),
        "the dead handler's fallback class must not emit: {}",
        out
    );
    assert!(
        out.contains("matches (\"SSLError\")"),
        "except BaseSSLError must canonicalize to the SSLError tag: {}",
        out
    );
}

#[test]
fn exception_union_parameter_boxes_to_pyvalue() {
    // `err: BaseSSLError | OSError | SocketTimeout` (urllib3's
    // _raise_timeout): exception members — by naming convention or via
    // the imported-alias table — box the parameter as PyValue instead of
    // rendering the union literally (invalid Rust).
    let out = compile(
        "from socket import timeout as SocketTimeout\n\
         \n\
         def f(err: BaseSSLError | OSError | SocketTimeout) -> None:\n\
         \x20   pass\n",
        "excunion.py",
    );
    assert!(
        out.contains("err : stdpython :: PyValue"),
        "an all-exception union parameter must box: {}",
        out
    );
    assert!(
        !out.contains("| (SocketTimeout)"),
        "the union must not render literally: {}",
        out
    );
}

#[test]
fn getattr_on_stdlib_module_folds_statically() {
    // The version-probing idiom (urllib3's ssl_.py): getattr over a
    // stdlib module with a literal name resolves at conversion time —
    // the runtime item when present (promoted to a pub use so functions
    // see it), else the literal default.
    let out = compile(
        "import ssl\n\
         \n\
         VERIFY_X509_PARTIAL_CHAIN = getattr(ssl, \"VERIFY_X509_PARTIAL_CHAIN\", 0x80000)\n\
         MISSING = getattr(ssl, \"NOT_A_REAL_CONSTANT\", 42)\n\
         \n\
         def f() -> int:\n\
         \x20   return VERIFY_X509_PARTIAL_CHAIN\n",
        "sslgetattr.py",
    );
    assert!(
        out.contains("pub use stdpython :: ssl :: VERIFY_X509_PARTIAL_CHAIN"),
        "a present item must alias the runtime constant: {}",
        out
    );
    assert!(
        out.contains("MISSING = 42"),
        "a missing item must fold to the default: {}",
        out
    );
}

#[test]
fn resolved_module_import_gates_fold_statically() {
    // `if not ssl:` / `if ssl is None:` over a RESOLVED module import
    // (urllib3 connection.py's DummyConnection fallback): the module is
    // always truthy and never None, so the gates fold — a module object
    // as a runtime value has no lowering (E0423 otherwise).
    let out = compile(
        "import ssl\n\
         \n\
         if not ssl:\n\
         \x20   CHOSEN = 1\n\
         else:\n\
         \x20   CHOSEN = 2\n\
         \n\
         def f() -> int:\n\
         \x20   if ssl is None:\n\
         \x20       return 0\n\
         \x20   return 1\n",
        "sslgate.py",
    );
    assert!(
        !out.contains("CHOSEN = 1"),
        "the not-ssl branch is dead when the import resolves: {}",
        out
    );
    assert!(
        out.contains("CHOSEN = 2"),
        "the else branch is the live one: {}",
        out
    );
    assert!(
        !out.contains("return Ok (0"),
        "`ssl is None` folds false inside functions too: {}",
        out
    );
}

#[test]
fn module_constant_method_call_is_a_value_call_not_a_path() {
    // `ssl.OPENSSL_VERSION.startswith(...)` (urllib3's __init__): the
    // SCREAMING_SNAKE segment ends the module path — the method call is
    // on the constant's VALUE (`ssl::OPENSSL_VERSION.startswith(...)`),
    // not a `::startswith` path item.
    let out = compile(
        "import ssl\n\
         \n\
         def f() -> bool:\n\
         \x20   return ssl.OPENSSL_VERSION.startswith(\"OpenSSL \")\n",
        "sslver.py",
    );
    assert!(
        !out.contains("OPENSSL_VERSION :: startswith"),
        "method on a module constant must not render as a path: {}",
        out
    );
    assert!(
        out.contains("startswith"),
        "the method call itself must survive: {}",
        out
    );
}

// ---- issue #137 round 16: urllib3 residual clusters ----

#[test]
fn exception_class_docstring_is_a_doc_attribute() {
    // `class HTTPError(Exception): """Base exception."""` — the marker
    // struct's docstring must be a real #[doc] ATTRIBUTE; interpolating
    // a String into quote! yields a string-literal token (`""` in item
    // position for the doc-less case — a parse error in the generated
    // crate, which hid every later error in the file).
    let out = compile(
        "class HTTPError(Exception):\n\
         \x20   \"\"\"Base exception.\"\"\"\n\
         \n\
         class Bare(Exception):\n\
         \x20   pass\n",
        "excdoc.py",
    );
    assert!(
        out.contains("# [doc = \"Base exception.\"]"),
        "docstring must render as a doc attribute: {}",
        out
    );
    assert!(
        !out.contains("\"\" #"),
        "no stray empty-string tokens in item position: {}",
        out
    );
}

#[test]
fn underscore_is_a_real_readable_variable() {
    // Python's `_` is an ordinary name — bound by `except ... as _` and
    // READ afterwards (urllib3's util/connection), and stored by tuple
    // destructures (`(scheme, _, host) = ...`). Rust's `_` is a wildcard
    // (unreadable, illegal after `let mut`), so it maps to a real
    // underscore-prefixed identifier; throwaway loop indices keep the
    // wildcard through the unused-index path.
    let out = compile(
        "def f(url: str) -> str:\n\
         \x20   err = ''\n\
         \x20   try:\n\
         \x20       return url\n\
         \x20   except OSError as _:\n\
         \x20       err = str(_)\n\
         \x20   return err\n",
        "underscore.py",
    );
    assert!(
        out.contains("__rython_underscore"),
        "`as _` must bind a real identifier: {}",
        out
    );
    assert!(
        !out.contains("let mut _ ="),
        "no illegal wildcard binding: {}",
        out
    );
}

#[test]
fn qualified_collections_class_constructs_via_new() {
    // `collections.deque()` (urllib3's response.py): the runtime item is
    // a struct — the qualified call goes through `::new` exactly like
    // the from-import spelling.
    let out = compile(
        "import collections\n\
         \n\
         def f() -> None:\n\
         \x20   buf = collections.deque()\n\
         \x20   buf.append(1)\n",
        "cdeque.py",
    );
    assert!(
        out.contains("collections :: deque :: new ()"),
        "qualified deque() must construct via ::new: {}",
        out
    );
}

#[test]
fn from_imported_socket_function_calls_directly() {
    // `from socket import getdefaulttimeout` then `getdefaulttimeout()`
    // (urllib3's util/timeout): a stdlib FUNCTION import — a direct
    // call, never a `::new` class construction.
    let out = compile(
        "from socket import getdefaulttimeout\n\
         \n\
         def f() -> object:\n\
         \x20   return getdefaulttimeout()\n",
        "gdt.py",
    );
    assert!(
        out.contains("getdefaulttimeout ()"),
        "the function must call directly: {}",
        out
    );
    assert!(
        !out.contains("getdefaulttimeout :: new"),
        "no class construction for a stdlib function: {}",
        out
    );
}

#[test]
fn generator_annotation_types_the_stub_signature() {
    // An abstract generator STUB (`-> typing.Generator[bytes]` with no
    // yields — urllib3's BaseHTTPResponse.stream): the annotation still
    // decides the signature, so overriding generators' Vec returns
    // agree with the trait declaration (E0053 otherwise). Both the
    // typing-qualified and single-parameter spellings resolve.
    let out = compile(
        "import typing\n\
         \n\
         class Base:\n\
         \x20   def stream(self, amt: int) -> typing.Generator[bytes]:\n\
         \x20       raise NotImplementedError()\n\
         \n\
         class Impl(Base):\n\
         \x20   def stream(self, amt: int) -> typing.Generator[bytes]:\n\
         \x20       yield b'x'\n",
        "genstub.py",
    );
    assert!(
        out.contains("Result < Vec < Vec < u8 > >"),
        "the stub's signature must carry the annotated element: {}",
        out
    );
    assert!(
        !out.contains("Vec < _ >"),
        "no inference placeholder in item signatures: {}",
        out
    );
}

#[test]
fn module_self_assign_is_a_noop() {
    // `__version__ = __version__` (urllib3's __init__, a typing/
    // re-export idiom): a no-op — it must not demote the name to a
    // module-init local (E0530 against the imported static) nor count
    // as a second store.
    let out = compile(
        "CONST = 5\n\
         CONST = CONST\n\
         \n\
         def f() -> int:\n\
         \x20   return CONST\n",
        "selfassign.py",
    );
    assert!(
        out.contains("pub static CONST"),
        "the single real store must still promote: {}",
        out
    );
}

// ---- issue #137 round 17: shadowed external aliases and generators ----

#[test]
fn shadowed_external_base_is_metadata_not_self() {
    // `from http.client import HTTPConnection as _HTTPConnection` then
    // `class HTTPConnection(_HTTPConnection)` — urllib3's connection.py:
    // the alias's canonical name is shadowed by the class itself, so
    // following it made the class its own base (a self-supertrait cycle,
    // E0391, and an infinitely-sized embedded struct, E0072). The base
    // is the shadowed EXTERNAL class — metadata.
    let out = compile(
        "from socks import HTTPConnection as _HTTPConnection\n\
         \n\
         class HTTPConnection(_HTTPConnection):\n\
         \x20   def __init__(self, host: str) -> None:\n\
         \x20       self.host = host\n",
        "shadowbase.py",
    );
    assert!(
        !out.contains("HTTPConnectionTrait : HTTPConnectionTrait"),
        "no self-supertrait: {}",
        out
    );
    assert!(
        !out.contains("__rython_base : HTTPConnection"),
        "no self-embedded base struct: {}",
        out
    );
}

#[test]
fn shadowed_external_alias_annotation_boxes() {
    // `self._fp: _HttplibHTTPResponse | None` (urllib3's response.py)
    // where the alias's canonical name is shadowed by the local class:
    // the annotation means the external class — a boxed value — never
    // Option<LocalClass> (which made the field self-recursive, E0072).
    let src = "from http_client import HTTPResponse as _HttplibHTTPResponse\n\
               \n\
               class HTTPResponse:\n\
               \x20   def __init__(self) -> None:\n\
               \x20       self._fp: _HttplibHTTPResponse | None = None\n";
    let m = parse(src, "shadowfield.py").unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["shadowfield".to_string()], std::rc::Rc::new(m));
    let other = parse("x = 1\n", "other.py").unwrap();
    defs.insert(vec!["other".to_string()], std::rc::Rc::new(other));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        this_module_path: vec!["shadowfield".to_string()],
        ..Default::default()
    };
    let out = compile_with_options(src, "shadowfield.py", options).expect("converts");
    assert!(
        out.contains("_fp : stdpython :: PyValue"),
        "the field must box to PyValue: {}",
        out
    );
    assert!(
        !out.contains("Option < HTTPResponse >"),
        "no self-recursive Option field: {}",
        out
    );
}

#[test]
fn unannotated_generator_boxes_and_returns_ok() {
    // A generator whose yield type the inference cannot resolve boxes to
    // PyValue — the `_` placeholder is illegal in item signatures
    // (E0121) — and the collector returns inside the function's Result
    // (`return __rython_gen` bare was E0308).
    let out = compile(
        "def gen(n: int):\n\
         \x20   for i in range(n):\n\
         \x20       yield i\n",
        "boxgen.py",
    );
    assert!(
        out.contains("Vec < stdpython :: PyValue >"),
        "unresolved yield element must box: {}",
        out
    );
    assert!(
        out.contains("return Ok (__rython_gen)"),
        "the collector returns in the Result: {}",
        out
    );
    assert!(
        out.contains("push (stdpython :: PyValue :: from"),
        "yields box into the collector: {}",
        out
    );
}

#[test]
fn bare_yield_contextmanager_pushes_boxed_none() {
    // `@contextmanager`-style bare `yield` (urllib3's _error_catcher):
    // yields None — the boxed None in a PyValue collector, not a
    // no-op expression in a Vec<_> signature.
    let out = compile(
        "def catcher():\n\
         \x20   try:\n\
         \x20       yield\n\
         \x20   except OSError:\n\
         \x20       pass\n",
        "bareyield.py",
    );
    assert!(
        out.contains("push (stdpython :: PyValue :: None_)"),
        "bare yield must push the boxed None: {}",
        out
    );
    assert!(
        !out.contains("Vec < _ >"),
        "no inference placeholder in the signature: {}",
        out
    );
}

// ---- issue #137 round 18: cross-module ancestor-trait impls ----

/// Two-module fixture: module `animals` defines a hierarchy; the module
/// under test subclasses the imported Dog.
fn cross_module_subclass_options() -> PythonOptions {
    let a = parse(
        concat!(
            "class Animal:\n",
            "    def __init__(self, name: str):\n",
            "        self.name = name\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return self.name\n",
            "\n",
            "class Dog(Animal):\n",
            "    def __init__(self, name: str):\n",
            "        self.tricks = 0\n",
            "\n",
            "    def grow(self) -> None:\n",
            "        pass\n",
        ),
        "animals.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["animals".to_string()], std::rc::Rc::new(a));
    PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    }
}

#[test]
fn cross_module_subclass_implements_imported_ancestor_traits() {
    // `class Puppy(Dog)` with Dog imported from another module (urllib3's
    // SOCKSConnection(HTTPConnection) shape): Puppy must implement the
    // imported ancestors' traits — named by their crate paths (this
    // module need not import them) — with accessor types resolved in the
    // DEFINING module's scope.
    let src = "from animals import Dog\n\nclass Puppy(Dog):\n    def fetch(self) -> int:\n        return 1\n";
    let out = compile_with_options(src, "puppy.py", cross_module_subclass_options())
        .expect("converts");
    assert!(
        out.contains("impl crate :: animals :: DogTrait for Puppy"),
        "the imported base's trait must be implemented by crate path: {}",
        out
    );
    assert!(
        out.contains("impl crate :: animals :: AnimalTrait for Puppy"),
        "the whole imported chain implements, root included: {}",
        out
    );
}

#[test]
fn covariant_cross_module_override_is_dropped_with_warning() {
    // An override whose signature disagrees with the imported base's
    // trait declaration (`grow() -> int` over `-> None` — the SOCKS
    // `_new_conn` shape): dropped with the divergence warning, never a
    // mismatched impl (E0053).
    let src = "from animals import Dog\n\nclass Puppy(Dog):\n    def grow(self) -> int:\n        return 1\n";
    let module = parse(src, "puppy2.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let options = cross_module_subclass_options();
    let warnings = options.definition_warnings.clone();
    let out = module
        .to_rust(
            CodeGenContext::Module("puppy2".to_string()),
            options,
            symbols,
        )
        .expect("converts")
        .to_string();
    assert!(
        !out.contains("fn grow (& self) -> Result < i64"),
        "the disagreeing override must not land in the ancestor impl: {}",
        out
    );
    assert!(
        warnings
            .borrow()
            .iter()
            .any(|w| w.contains("covariant-override divergence")),
        "the drop must be loud: {:?}",
        warnings.borrow()
    );
}

/// Absolute imports of same-crate modules must resolve for src-layout
/// sdists, whose `module_defs` keys are RELATIVE to the package root
/// (pip, boto3): `from pkg.reqmod import with_cleanup` resolves to
/// ["pkg", "reqmod"] while the key is ["reqmod"]. Without the
/// two-form lookup, the local-wrapper decorator bypass missed and the
/// conversion failed with "decorator `with_cleanup` is not supported
/// yet" (pip's `_internal/commands/download.py`).
#[test]
fn absolute_import_of_src_layout_sibling_decorator_resolves() {
    let reqmod = parse(
        "def with_cleanup(func):\n    return func\n",
        "reqmod.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        vec!["reqmod".to_string()],
        std::rc::Rc::new(reqmod),
    );
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        // The stripped-prefix lookup applies only to the package's OWN
        // root-qualified name.
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let usemod = parse(
        "from pkg.reqmod import with_cleanup\n\
         \n\
         class DownloadCommand:\n\
         \x20   @with_cleanup\n\
         \x20   def run(self, options, args):\n\
         \x20       return 0\n",
        "usemod.py",
    )
    .unwrap();
    let symbols = usemod.clone().find_symbols(SymbolTableScopes::new());
    let out = usemod
        .to_rust(
            CodeGenContext::Module("usemod".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        !out.contains("is not supported yet"),
        "the local-wrapper decorator must lower directly: {}",
        out
    );
    assert!(out.contains("fn run"), "method must be emitted: {}", out);
}

/// The generated `use` for an absolute sibling import must match the
/// crate's mod tree (relative keys): `crate::session::make`, not
/// `crate::pkg::session::make` (which would fail E0432).
#[test]
fn absolute_import_of_src_layout_sibling_emits_relative_use() {
    let session = parse(
        "def make() -> int:\n    return 42\n",
        "session.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        vec!["session".to_string()],
        std::rc::Rc::new(session),
    );
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let caller = parse(
        "from pkg.session import make\n\
         \n\
         def answer() -> int:\n\
         \x20   return make()\n",
        "caller.py",
    )
    .unwrap();
    let symbols = caller.clone().find_symbols(SymbolTableScopes::new());
    let out = caller
        .to_rust(
            CodeGenContext::Module("caller".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        out.contains("session :: make")
            && !out.contains("pkg :: session"),
        "use must reference the crate-relative module: {}",
        out
    );

    assert!(
        out.contains("make ("),
        "the cross-module call must lower: {}",
        out
    );
}

/// A plain `import pkg.connection` inside the pkg conversion binds only
/// the ROOT name in Python — never the leaf. Emitting `use
/// crate::connection;` would bind a name Python doesn't and collide with
/// a sibling's own `connection` submodule (urllib3's
/// `contrib/emscripten`, E0255); `use crate::pkg::connection;` names a
/// module the crate doesn't contain (E0432). Unaliased emits nothing;
/// aliased binds the crate-relative path.
#[test]
fn plain_import_of_root_qualified_sibling_binds_root_only() {
    let connection = parse("A = 1\n", "connection.py").unwrap();
    let other = parse("B = 2\n", "other.py").unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["connection".to_string()], std::rc::Rc::new(connection));
    defs.insert(vec!["other".to_string()], std::rc::Rc::new(other));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let user = parse(
        "import pkg.connection\nimport pkg.connection as conn\n",
        "user.py",
    )
    .unwrap();
    let symbols = user.clone().find_symbols(SymbolTableScopes::new());
    let out = user
        .to_rust(
            CodeGenContext::Module("user".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        !out.contains("crate :: pkg"),
        "the crate has no `pkg` module: {}",
        out
    );
    assert!(
        !out.contains("use crate :: connection ;"),
        "the unaliased form must not bind the leaf: {}",
        out
    );
    assert!(
        out.contains("use crate :: connection as conn"),
        "the aliased form binds the crate-relative path: {}",
        out
    );
}

/// The stripped-prefix lookup covers ONLY the package's own
/// root-qualified name: `import h2.connection` must not resolve to a
/// same-named crate module (urllib3's connection.py) — it is an external
/// module, dropped with the divergence warning.
#[test]
fn plain_import_of_external_root_is_not_aliased_onto_crate_modules() {
    let connection = parse("A = 1\n", "connection.py").unwrap();
    let other = parse("B = 2\n", "other.py").unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["connection".to_string()], std::rc::Rc::new(connection));
    defs.insert(vec!["other".to_string()], std::rc::Rc::new(other));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let warnings = options.definition_warnings.clone();
    let user = parse("import h2.connection\n", "user.py").unwrap();
    let symbols = user.clone().find_symbols(SymbolTableScopes::new());
    let out = user
        .to_rust(
            CodeGenContext::Module("user".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        !out.contains("use crate"),
        "an external module must not resolve into the crate: {}",
        out
    );
    assert!(
        warnings
            .borrow()
            .iter()
            .any(|w| w.contains("external-module divergence")),
        "the drop must be loud: {:?}",
        warnings.borrow()
    );
}
/// Issue #163: an annotated empty dict keeps its element types when the
/// subscript stores are inside a loop. The loop-variable store
/// (`d[i] = i`, with `i` untyped) previously produced an unknown-key /
/// unknown-value pinning suggestion whose PyValue value absorbed the
/// annotated `dict[int, int]` in `unify`, downgrading the literal to
/// `PyDict<String, PyValue>` (and the i64-key store then failed to
/// compile). An existing container type must win over unknown
/// suggestions.
#[test]
fn annotated_empty_dict_keeps_types_inside_loop() {
    let out = compile(
        "def main() -> int:\n\
         \x20   d: dict[int, int] = {}\n\
         \x20   for i in range(2):\n\
         \x20       d[i] = i\n\
         \x20   print(d[0])\n\
         \x20   return 0\n",
        "dictloop.py",
    );
    assert!(
        out.contains("PyDict :: < i64 , i64 > :: from ([])"),
        "the annotated element types must survive the loop: {}",
        out
    );
}
