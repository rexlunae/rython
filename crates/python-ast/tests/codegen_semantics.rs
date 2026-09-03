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
fn option_field_aug_assign_unwraps_inner() {
    // A `-=` on an `int | None` field (urllib3's `self.chunk_left -=
    // ...`): the target is Option, so the read-modify-write must operate
    // on the INNER value through the runtime py_sub, not Rust's `-=` on
    // the Option (which has no such operator). A None target is CPython's
    // TypeError — a loud §12.2 panic with the message (the `is not None`
    // guard in real code prevents it).
    let out = compile(
        "class Counter:\n\
         \x20   def __init__(self):\n\
         \x20       self.count: int | None = None\n\
         \x20   def dec(self, n: i64):\n\
         \x20       if self.count is not None:\n\
         \x20           self.count -= n\n",
        "optdecr.py",
    );
    assert!(
        out.contains("py_sub (& __rython_w)") || out.contains("py_sub(&__rython_w)"),
        "Option-target -= must unwrap and py_sub the inner value: {}",
        out
    );
    assert!(
        out.contains("unsupported operand type(s) for -=: 'NoneType' and 'i64'"),
        "the None-target panic must carry CPython's message: {}",
        out
    );
}

#[test]
fn option_field_aug_assign_pyvalue_target_uses_runtime() {
    // A `-=` on a boxed PyValue field routes through the runtime py_sub
    // (the boxed int arithmetic) — the read-modify-write on the box.
    let out = compile(
        "class Counter:\n\
         \x20   def __init__(self):\n\
         \x20       self.count: object = 5\n\
         \x20   def dec(self, n):\n\
         \x20       self.count -= n\n",
        "pyvaldecr.py",
    );
    assert!(
        out.contains("py_sub (& (") || out.contains("py_sub(&("),
        "PyValue-target -= must route through py_sub: {}",
        out
    );
}

#[test]
fn option_field_bitor_aug_assign_unwraps_inner() {
    // A `|=` on an `int | None` local (urllib3's `options |= ...` after
    // `options = 0` inside the None guard): the Option unwrap is the INNER
    // value OR'd with the RHS.
    let out = compile(
        "def orit(x: int | None) -> int | None:\n\
         \x20   if x is None:\n\
         \x20       x = 0\n\
         \x20       x |= 2\n\
         \x20   return x\n",
        "optor.py",
    );
    assert!(
        out.contains("__rython_v | (2)") || out.contains("__rython_v |(2)"),
        "Option-target |= must OR the inner value: {}",
        out
    );
    assert!(
        out.contains("unsupported operand type(s) for |=: 'NoneType' and 'i64'"),
        "the None-target panic must carry CPython's message: {}",
        out
    );
}

#[test]
fn sub_with_option_rhs_unwraps_loudly() {
    // `x - y` where y is `int | None` (urllib3's `self.chunk_left - amt`
    // and `time.monotonic() - self._start_connect`): the runtime Option
    // blanket unwraps an Option LHS, but a None RHS needs a bound the
    // blanket cannot provide — unwrap the RHS with the loud TypeError
    // panic instead (guarded code never hits it).
    let out = compile(
        "def diff(x: float, y: float | None) -> float:\n\
         \x20   return x - y\n",
        "optdiff.py",
    );
    assert!(
        out.contains("match (y) . clone ()") || out.contains("match (y).clone()"),
        "Option-typed RHS of - must be unwrapped: {}",
        out
    );
    assert!(
        out.contains("unsupported operand type(s) for -: 'float' and 'NoneType'"),
        "the RHS-None panic must name the LHS type: {}",
        out
    );
}

#[test]
fn option_field_store_wraps_in_some() {
    // A plain value stored into an `int | None` field (urllib3's
    // `self._start_connect = time.monotonic()` and `self.chunk_left =
    // self.chunk_left - amt`) wraps in Some — Python's `int | None` slot
    // absorbs a plain int. None stores keep plain None; an already-Option
    // value stores through unchanged.
    let out = compile(
        "class Timer:\n\
         \x20   def __init__(self):\n\
         \x20       self._start: float | None = None\n\
         \x20   def start(self):\n\
         \x20       self._start = 1.0\n\
         \x20   def reset(self):\n\
         \x20       self._start = None\n",
        "optstore.py",
    );
    assert!(
        out.contains("self . _start = Some (1.0)") || out.contains("self._start = Some(1.0)"),
        "a plain value into an Option field must wrap in Some: {}",
        out
    );
    assert!(
        out.contains("self . _start = None") || out.contains("self._start = None"),
        "a None store into an Option field stays plain None: {}",
        out
    );
}

#[test]
fn option_field_store_of_option_value_passes_through() {
    // `self.a = self.b` where BOTH fields are Option must NOT double-wrap
    // (Some(Some(..))): an already-Option value stores through unchanged.
    let out = compile(
        "class Pair:\n\
         \x20   def __init__(self):\n\
         \x20       self.a: int | None = None\n\
         \x20       self.b: int | None = None\n\
         \x20   def copy(self):\n\
         \x20       self.a = self.b\n",
        "optpassthru.py",
    );
    assert!(
        !out.contains("Some (Some") && !out.contains("Some(Some"),
        "an Option value into an Option field must not double-wrap: {}",
        out
    );
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
    // Issue #130: primitive mixes ([1, "a"]) BOX to Vec<PyValue>. Since
    // #180 (PyValue::Dict) a dict literal is boxable too, so [1, {'a':
    // 2}] now boxes instead of erroring.
    let out = compile("[1, {'a': 2}]", "boxlist.py");
    assert!(
        out.contains("PyValue :: from"),
        "a dict element must box like any other boxable value: {}",
        out
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
        out.contains("__rython_exc . matches_builtin (BuiltinException :: ValueError)")
            || out.contains("__rython_exc . matches (\"ValueError\")"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("matches_builtin (BuiltinException :: TypeError) || __rython_exc . matches_builtin (BuiltinException :: KeyError)")
            || out.contains("matches (\"TypeError\") || __rython_exc . matches (\"KeyError\")"),
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
            out.contains("__rython_exc . matches_builtin (BuiltinException :: ZeroDivisionError)")
                || out.contains("__rython_exc . matches (\"ZeroDivisionError\")"),
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
fn boxed_union_param_stores_plain_pyvalue() {
    // Round 40: a `int | str | None` parameter resolves to the BOXED
    // PyValue (the box already contains None — issue #121's
    // boxable-union rule), so it is NOT an Option slot: plain stores go
    // through PyValue::from, never Some(...) (urllib3's `cert_reqs =
    // resolve_cert_reqs(None)` was Some-wrapping and failing to build).
    // The syntactic optional-ness (`is_optional_annotation`) must not
    // override the resolved type.
    let out = compile(
        "def f(cert_reqs: int | str | None = None) -> int:\n\
         \x20   if cert_reqs is None:\n\
         \x20       cert_reqs = 5\n\
         \x20   return 0\n",
        "boxedparam.py",
    );
    assert!(
        out.contains("cert_reqs = PyValue :: from (5)")
            || out.contains("cert_reqs = PyValue::from(5)"),
        "boxed-union param stores must go through PyValue::from: {}",
        out
    );
    assert!(
        !out.contains("cert_reqs = Some"),
        "boxed-union param stores must not Some-wrap: {}",
        out
    );
}

#[test]
fn plain_option_param_still_narrows() {
    // Round 40 guard: the optional_names exclusion must apply ONLY to
    // boxed-union annotations — a genuine `int | None` parameter stays an
    // Option slot and still unwraps under `is not None` narrowing.
    let out = compile(
        "def g(x: int | None) -> int:\n\
         \x20   if x is not None:\n\
         \x20       return x + 1\n\
         \x20   return 0\n",
        "plainopt.py",
    );
    assert!(
        out.contains("x : Option < i64 >") || out.contains("x: Option<i64>"),
        "plain optional param must stay Option-typed: {}",
        out
    );
    assert!(
        out.contains("unwrap ()") || out.contains("unwrap()"),
        "narrowed read must unwrap: {}",
        out
    );
}

#[test]
fn class_member_union_param_is_not_optional() {
    // Round 42: `Retry | bool | int | None` (urllib3's retries) resolves
    // to the boxed PyValue — the class member makes the union boxable,
    // and the box absorbs None — so stores must NOT Some-wrap (they go
    // through PyValue::from; a class-instance member has no boxed repr
    // and stays loudly unboxable). The symbol-aware alias resolver sees
    // this even though the class member is not a builtin scalar.
    let out = compile(
        "class Retry:\n\
         \x20   pass\n\
         \n\
         def f(retries: Retry | bool | int | None = None) -> int:\n\
         \x20   if retries is None:\n\
         \x20       retries = 5\n\
         \x20   return 0\n",
        "classmember.py",
    );
    assert!(
        out.contains("retries = PyValue :: from (5)") || out.contains("retries = PyValue::from(5)"),
        "class-member union param stores must go through PyValue::from: {}",
        out
    );
    assert!(
        !out.contains("retries = Some"),
        "class-member union param stores must not Some-wrap: {}",
        out
    );
    // A genuine `Retry | None` is Option<Retry> — the narrow class-only
    // optional keeps its Option slot.
    let out2 = compile(
        "class Retry:\n\
         \x20   pass\n\
         \n\
         def g(x: Retry | None = None) -> int:\n\
         \x20   if x is not None:\n\
         \x20       return 1\n\
         \x20   return 0\n",
        "classonly.py",
    );
    assert!(
        out2.contains("x : Option < Retry >") || out2.contains("x: Option<Retry>"),
        "a class-only optional stays Option-typed: {}",
        out2
    );
}

#[test]
fn typing_imports_lower_to_nothing() {
    let out = compile("from typing import Optional\nx = 1\n", "typing.py");
    assert!(!out.contains("typing"), "generated: {}", out);
}

#[test]
fn typing_any_annotation_maps_to_boxed_pyvalue() {
    // Round 44: `dict[str, typing.Any]` — a return annotation whose value
    // type is `typing.Any` (urllib3's `_merge_pool_kwargs`): the `Any`
    // maps to the boxed PyValue, so the method's signature is
    // `Result<PyDict<String, PyValue>>` instead of collapsing to unit
    // while the body still emits `Ok(dict)` (which cannot compile).
    let out = compile(
        "import typing\n\
         def merge(override: dict[str, typing.Any] | None) -> dict[str, typing.Any]:\n\
         \x20   base: dict[str, typing.Any] = {}\n\
         \x20   if override:\n\
         \x20       for k, v in override.items():\n\
         \x20           base[k] = v\n\
         \x20   return base\n",
        "typany.py",
    );
    assert!(
        out.contains("Result < PyDict < String , stdpython :: PyValue > , PyException >")
            || out.contains("Result<PyDict<String, stdpython::PyValue>, PyException>"),
        "a dict[str, typing.Any] return must type the boxed value dict: {}",
        out
    );
    assert!(
        !out.contains("-> Result < () , PyException >"),
        "the return must not collapse to unit: {}",
        out
    );
}

#[test]
fn local_assigned_from_option_param_some_wraps_stores() {
    // Round 45: a local assigned from an OPTION-typed parameter
    // (`release_this_conn = release_conn` where the param is `bool |
    // None` — urllib3's urlopen) is itself an Option binding: later
    // plain stores (`= False`) wrap in Some, so the binding stays
    // Option<bool> and the generated crate typechecks. The param's
    // `T | None` annotation resolves through local_types (py_type now
    // parses the union) and infer_type (the Assign Name-value arm now
    // consults it).
    let out = compile(
        "def f(release_conn: bool | None = None) -> bool | None:\n\
         \x20   release_this_conn = release_conn\n\
         \x20   release_this_conn = False\n\
         \x20   return release_this_conn\n",
        "optlocal.py",
    );
    assert!(
        out.contains("release_this_conn = Some (false)")
            || out.contains("release_this_conn = Some(false)"),
        "a plain store into an Option-assigned local must Some-wrap: {}",
        out
    );
    assert!(
        out.contains("release_this_conn = release_conn"),
        "the Option param value must pass through: {}",
        out
    );
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
fn typing_optional_namedtuple_fields_are_option_slots() {
    // Round 47: `typing.NamedTuple("Url", [("scheme",
    // typing.Optional[str]), ...])` — urllib3's Url — types each field as
    // `Option<String>`/`Option<i64>`, NOT the boxed PyValue (the
    // alias-aware resolver previously lumped `typing.Optional[T]` into
    // the boxed-union tolerance). The synthesized `__init__` stores pass
    // the Option params through unwrapped.
    let out = compile(
        "from typing import NamedTuple\n\
         class Url(NamedTuple):\n\
         \x20   scheme: typing.Optional[str]\n\
         \x20   port: typing.Optional[int]\n",
        "typopt.py",
    );
    assert!(
        out.contains("pub scheme : Option < String >")
            || out.contains("pub scheme: Option<String>"),
        "a typing.Optional[str] NamedTuple field must be Option<String>: {}",
        out
    );
    assert!(
        out.contains("pub port : Option < i64 >") || out.contains("pub port: Option<i64>"),
        "a typing.Optional[int] NamedTuple field must be Option<i64>: {}",
        out
    );
}

#[test]
fn tuple_all_none_store_marks_names_optional() {
    // Round 47: `auth, host, port = None, None, None` (urllib3's
    // parse_url) marks each name an Option binding — the tuple-target
    // analysis previously only tracked single-name None stores, so the
    // names stayed PyObject and a later Option-returning store
    // double-wrapped when the name was passed to an Option slot. The
    // tuple-None store is the missing half (round 47).
    let out = compile(
        "def take(x: int) -> int | None:\n\
         \x20   return x if x else None\n\
         \ndef f(flag: bool) -> None:\n\
         \x20   a, b, c = None, None, None\n\
         \x20   if flag:\n\
         \x20       a = take(1)\n",
        "tupnone.py",
    );
    assert!(
        out.contains("a = take (1) ?") || out.contains("a = take(1)?"),
        "the Option-assigned local must pass its later store through: {}",
        out
    );
}

#[test]
fn local_from_dict_returning_self_method_owns_string_keys() {
    // Round 46: `request_context = self._merge_pool_kwargs(pool_kwargs)`
    // (urllib3's PoolManager) — the local's type comes from the callee's
    // `-> dict[str, typing.Any]` return annotation (the class-aware
    // seeding only types DICT-returning self-method calls; the broad
    // round-44 version cascaded on conn-style locals). A Dict-typed
    // local makes the subscript STORES own their string keys and box
    // their Option values (py_set_index takes String / the value type
    // is PyValue).
    let out = compile(
        "class P:\n\
         \x20   def _merge(self, override: dict[str, object] | None) -> dict[str, object]:\n\
         \x20       return {}\n\
         \x20   def go(self, scheme: str | None) -> None:\n\
         \x20       ctx = self._merge(None)\n\
         \x20       ctx[\"scheme\"] = scheme or \"http\"\n\
         \x20       ctx[\"port\"] = None\n",
        "selfdict.py",
    );
    assert!(
        out.contains("(ctx) . py_set_index ((\"scheme\") . to_string ()")
            || out.contains("(ctx).py_set_index((\"scheme\").to_string()"),
        "a Dict-typed local must own its string keys at the store: {}",
        out
    );
    assert!(
        out.contains("ctx = { (self) . _merge (None) ? }")
            || out.contains("ctx = { (self)._merge(None)? }")
            || out.contains("ctx = (self . _merge (None)) ?"),
        "the local must be assigned from the self-method call: {}",
        out
    );
}

#[test]
fn string_literal_store_into_string_typed_name_owns_itself() {
    // Round 46: `method = "GET"` where the parameter is `str` and the
    // prologue bound `let mut method: String = method.into()` (urllib3's
    // urlopen): the literal is a &'static str and the binding is owned,
    // so the store owns it. A literal-only local (StrRef — `&'static
    // str`) keeps the bare store.
    let out = compile(
        "def f(method: str, flag: bool) -> str:\n\
         \x20   method = method.upper()\n\
         \x20   if flag:\n\
         \x20       method = \"GET\"\n\
         \x20   return method\n\
         \ndef g() -> str:\n\
         \x20   label = \"fine\"\n\
         \x20   return label\n",
        "strname.py",
    );
    assert!(
        out.contains("method = (\"GET\") . to_string ()")
            || out.contains("method = (\"GET\").to_string()"),
        "a str literal into a String-typed name must own: {}",
        out
    );
}

#[test]
fn str_literal_append_insert_into_string_vec_owns_itself() {
    // Round 46: `lines.append("\r\n")` and `output.insert(0, "")` on
    // Vec<String> locals (urllib3's render_headers and
    // _remove_path_dot_segments): the &'static str literal owns at the
    // push/insert site, mirroring the String-name store rule.
    let out = compile(
        "def f(seed: str) -> list[str]:\n\
         \x20   lines = []\n\
         \x20   lines.append(seed)\n\
         \x20   lines.append(\"\\r\\n\")\n\
         \x20   return lines\n\
         \ndef g(seed2: str) -> list[str]:\n\
         \x20   out = []\n\
         \x20   out.append(seed2)\n\
         \x20   out.insert(0, \"\")\n\
         \x20   return out\n",
        "strvec.py",
    );
    assert!(
        out.contains("push ((\"\\r\\n\") . to_string ())")
            || out.contains("push((\"\\r\\n\").to_string())"),
        "a str literal appended to a Vec<String> must own: {}",
        out
    );
    assert!(
        out.contains("py_insert (0 , (\"\") . to_string ())")
            || out.contains("py_insert(0, (\"\").to_string())"),
        "a str literal inserted into a Vec<String> must own: {}",
        out
    );
}

#[test]
fn tuple_destructure_string_literal_owns_into_string_slot() {
    // Round 46: `(body, content_type) = (urlencode(fields),
    // "application/x-www-form-urlencoded")` — urllib3's request() — the
    // content_type slot is String-typed (from the `(Vec<u8>, String)`
    // return of encode_multipart_formdata), so the literal owns at the
    // destructure.
    let out = compile(
        "def enc() -> tuple[bytes, str]:\n\
         \x20   return b\"\", \"x\"\n\
         \ndef f() -> str:\n\
         \x20   body, content_type = enc()\n\
         \x20   body, content_type = (None, \"application/x-www-form-urlencoded\")\n\
         \x20   return content_type\n",
        "tupslot.py",
    );
    assert!(
        out.contains("(\"application/x-www-form-urlencoded\") . to_string ()")
            || out.contains("(\"application/x-www-form-urlencoded\").to_string()"),
        "a str literal into a String-typed tuple slot must own: {}",
        out
    );
}

#[test]
fn literal_builtin_except_clauses_lower_to_discriminant_matches() {
    // Round 52: `except ValueError:` — the class name is a source
    // literal and the runtime knows its variant and ancestor slice
    // statically, so the handler lowers to a discriminant comparison
    // (no string walk per clause). User classes and builtin ALIASES
    // (EnvironmentError — a variant of OSError) keep the string path.
    let out = compile(
        concat!(
            "def f(x: int) -> int:\n",
            "    try:\n",
            "        return 10 // x\n",
            "    except ValueError:\n",
            "        return -1\n",
            "    except EnvironmentError:\n",
            "        return -2\n",
        ),
        "exceptfast.py",
    );
    assert!(
        out.contains("matches_builtin (BuiltinException :: ValueError)"),
        "a literal builtin clause must lower to the discriminant match: {}",
        out
    );
    assert!(
        out.contains("__rython_exc . matches (\"EnvironmentError\")"),
        "a builtin ALIAS must keep the string path (no EnvironmentError variant): {}",
        out
    );
}

#[test]
fn reassigned_unannotated_param_boxes_with_warning() {
    // Round 53: `hooks = hooks or {}` / `hook_data = _hook_data` —
    // requests' dispatch_hook (the ~20-round unblock): an unannotated
    // parameter reassigned inside the function cannot keep one inferred
    // generic type, so it lowers as the boxed PyValue (the honest
    // dynamic-shape fallback) with a definition warning, instead of
    // failing the whole module. charset_normalizer converts again;
    // requests progresses past hooks.py.
    let (out, warnings) = compile_with_warnings(
        "def dispatch_hook(key, hooks, hook_data):\n    hooks = hooks or {}\n    hooks = hooks.get(key)\n    return hook_data\n",
        "reassigned.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("hooks:stdpython::PyValue")
            || flat.contains("hooks:PyValue"),
        "the reassigned param must box to PyValue: {}",
        out
    );
    assert!(
        warnings.iter().any(|w| w.contains("reassigned") && w.contains("boxed PyValue")),
        "the boxing must carry a definition warning: {:?}",
        warnings
    );
}

#[test]
fn boxed_isinstance_dispatch_without_router_drops_loudly() {
    // Round 54: an isinstance-dispatched call whose axis argument is a
    // boxed/unknown value and whose dynamic router could not be planned
    // (an unannotated non-axis parameter) drops loudly with a warning
    // naming the rewrite, instead of failing the whole module (requests'
    // `_validate_header_part(header, name, 0)` — the last requests
    // blocker; the package now converts).
    let (out, warnings) = compile_with_warnings(
        concat!(
            "def _v(header, header_part, idx):\n",
            "    if isinstance(header_part, str):\n",
            "        return 1\n",
            "    elif isinstance(header_part, bytes):\n",
            "        return 2\n",
            "    return 0\n",
            "def check(header):\n",
            "    name, value = header\n",
            "    _v(header, name, 0)\n",
        ),
        "dispatchboxed.py",
    );
    assert!(
        out.contains("stdpython :: PyValue :: None_") || out.contains("PyValue :: None_"),
        "the undispatchable call must drop: {}",
        out
    );
    assert!(
        warnings.iter().any(|w| w.contains("is dropped") && w.contains("dynamic router")),
        "the drop must warn with the rewrite: {:?}",
        warnings
    );
}

#[test]
fn version_gate_bare_form_splices_module_defs() {
    // Round 51: `if sys.version_info >= (3, 11):` at MODULE level — the
    // bare (non-subscripted) form — was never statically evaluated (the
    // gate arm passed the receiver Name to is_sys_version_info), so the
    // guarded `def` stayed nested inside __module_init__ (invalid Rust).
    // The bare form now evaluates against rython's target (3.11.0) and
    // the taken branch's statements splice into the module body BEFORE
    // every pass — a version-gated def is a module item (certifi's
    // core.py: 12 errors -> 0).
    let out = compile(
        concat!(
            "import sys\n",
            "if sys.version_info >= (3, 11):\n",
            "    def where() -> str:\n",
            "        return \"x\"\n",
            "else:\n",
            "    def where() -> str:\n",
            "        return \"old\"\n",
        ),
        "vergate.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !flat.contains("(sys::version_info)"),
        "the version gate must be decided at conversion time, not emitted: {}",
        out
    );
    assert!(
        flat.contains("pubfnr#where()->Result<String,PyException>"),
        "the taken branch's def must be a module item: {}",
        out
    );
    assert!(
        !flat.contains("\"old\""),
        "the dead branch must be dropped: {}",
        out
    );
}

#[test]
fn urllib_parse_functions_lower_as_plain_calls() {
    // Round 55: `from urllib.parse import urlparse` — the item is a
    // FUNCTION in the runtime, so the call must lower as
    // `stdpython::urllib::parse::urlparse(&(...))?`, NOT a class
    // construction (`urlparse::new(...)` — E0433: a function used as a
    // module path). The stdpython_class registry separates class items
    // (OrderedDict, ...) from function items.
    let out = compile(
        concat!(
            "from urllib.parse import urlparse, urljoin, quote, unquote, urldefrag\n",
            "def f(url: str, base: str) -> str:\n",
            "    p = urlparse(url)\n",
            "    j = urljoin(base, url)\n",
            "    q = quote(url)\n",
            "    u = unquote(url)\n",
            "    d = urldefrag(url)\n",
            "    return p.scheme + j + q + u + d[0]\n",
        ),
        "urlparse.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !flat.contains("urlparse::new") && !flat.contains("urljoin::new"),
        "the parse functions must NOT lower as class constructions: {}",
        out
    );
    for needle in [
        "stdpython::urllib::parse::urlparse(&(url))?",
        "stdpython::urllib::parse::urljoin(&(base),&(url))?",
        "stdpython::urllib::parse::quote(&(url),None)?",
        "stdpython::urllib::parse::unquote(&(url))?",
        "stdpython::urllib::parse::urldefrag(&(url))?",
    ] {
        assert!(
            flat.contains(needle),
            "missing direct-call lowering `{}` in: {}",
            needle,
            out
        );
    }
    assert!(
        flat.contains("p.scheme"),
        "the ParseResult field read must survive: {}",
        out
    );
}

#[test]
fn urllib_urlencode_lowers_with_doseq_and_unquote_plus_exists() {
    // Round 55: urlencode takes the query (a dict-like PyValue) plus the
    // doseq keyword; quote_plus/unquote_plus are the +-flavored pair. The
    // calls render with `?` like every other fallible runtime function.
    let out = compile(
        concat!(
            "from urllib.parse import urlencode, quote_plus, unquote_plus\n",
            "def f(q) -> str:\n",
            "    a = urlencode(q)\n",
            "    b = urlencode(q, doseq=True)\n",
            "    c = quote_plus(\"a b\")\n",
            "    d = unquote_plus(\"a+b\")\n",
            "    return a + b + c + d\n",
        ),
        "urlencode.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("stdpython::urllib::parse::urlencode(&(q),false)?"),
        "urlencode without doseq must pass false: {}",
        out
    );
    assert!(
        flat.contains("stdpython::urllib::parse::urlencode(&(q),true)?"),
        "urlencode(doseq=True) must pass true: {}",
        out
    );
    assert!(
        flat.contains("stdpython::urllib::parse::quote_plus(&(")
            && flat.contains("))?"),
        "quote_plus must lower as a plain call with `?`: {}",
        out
    );
    assert!(
        flat.contains("stdpython::urllib::parse::unquote_plus(&(")
            && flat.contains("))?"),
        "unquote_plus must lower as a plain call with `?`: {}",
        out
    );}

#[test]
fn urllib_urlunparse_lowers_the_six_part_sequence() {
    // Round 55: `urlunparse([scheme, netloc, path, None, query,
    // fragment])` — requests' prepare_url — takes a 6-element sequence
    // with str-or-None members; the literal sequence renders as a boxed
    // tuple the runtime extracts. A None member boxes to PyValue::None_.
    let out = compile(
        concat!(
            "from urllib.parse import urlunparse\n",
            "def f(scheme: str, netloc: str, path: str, query: str, fragment: str) -> str:\n",
            "    return urlunparse([scheme, netloc, path, None, query, fragment])\n",
        ),
        "urlunparse.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("stdpython::urllib::parse::urlunparse(&(PyValue::from(vec![")
            && flat.contains("stdpython::PyValue::None_"),
        "the 6-part literal sequence must render as a boxed tuple with None: {}",
        out
    );
    assert!(
        !flat.contains("urlunparse::new"),
        "urlunparse must not lower as a class construction: {}",
        out
    );
}

#[test]
fn stdpython_class_registry_keeps_classes_as_constructions() {
    // Round 55: the class-aware stdpython_class must NOT regress real
    // classes — `from collections import OrderedDict; OrderedDict()`
    // still lowers to `OrderedDict::new(...)`.
    let out = compile(
        "from collections import OrderedDict\n\ndef f():\n    return OrderedDict()\n",
        "ordered.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("OrderedDict::new()"),
        "OrderedDict must still lower as a class construction: {}",
        out
    );
}

#[test]
fn re_alias_dispatch_uses_the_bound_name() {
    // Round 55: `from re import compile as re_compile; re_compile(pat,
    // flags)` — the dispatch arms match the CANONICAL name ("compile"),
    // but the generated call must use the BOUND name (only `re_compile`
    // is in scope via the alias re-export; rendering `compile` would be
    // E0425). charset_normalizer's constant.py.
    let out = compile(
        concat!(
            "from re import compile as re_compile\n",
            "from re import IGNORECASE\n",
            "def f():\n",
            "    return re_compile(\"a\", IGNORECASE)\n",
        ),
        "realiased.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("re_compile(&(\"a\"),\"i\")?")
            || flat.contains("re_compile(&(\"a\"),\"i\")"),
        "the aliased re call must render through the BOUND name with flags: {}",
        out
    );
    assert!(
        !flat.contains("stdpython::re::compile(&"),
        "the canonical runtime path is not what the call must render (the alias is in scope): {}",
        out
    );
}

#[test]
fn json_dumps_from_import_converts_the_boxed_value() {
    // Round 55: `from json import dumps; dumps(self.__dict__, ...)` —
    // charset_normalizer's models.py. The runtime's `dumps` takes
    // `&JSONValue`; the from-import call passes a PyValue/PyDict, so the
    // call routes through `dumps_pyvalue` (pyvalue_to_json conversion).
    let out = compile(
        concat!(
            "from json import dumps\n",
            "def f(data) -> str:\n",
            "    return dumps(data, ensure_ascii=True, indent=4)\n",
        ),
        "dumps.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("stdpython::json::dumps_pyvalue(data,Some(4))?"),
        "the from-import dumps must route through dumps_pyvalue: {}",
        out
    );
    assert!(
        !flat.contains("dumps::new"),
        "dumps must not lower as a class construction: {}",
        out
    );
}

// The external-chain typed-return panic is pinned in the MULTI-module
// e2e suite (crates/rypip/tests/convert_tests.rs): a single-module
// conversion treats every import as a potential sibling, so the
// external-module drop only fires under rypip's crate-wide resolution
// (certifi's contents() — the round-51 milestone).

#[test]
fn boxed_global_read_in_typed_return_panics_loudly() {
    // Round 51: reading a BOXED mutable global (the None-initialized
    // `_CACERT_PATH` — certifi's where()) from a `-> str` function: the
    // global's value is the boxed PyValue, which the typed return cannot
    // express — the exact point of divergence is a loud panic.
    let out = compile(
        concat!(
            "_CACERT_PATH = None\n",
            "def where() -> str:\n",
            "    global _CACERT_PATH\n",
            "    _CACERT_PATH = \"x\"\n",
            "    return _CACERT_PATH\n",
        ),
        "boxedglobal.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("panic!") && flat.contains("boxed-global"),
        "the typed return of a boxed global read must panic loudly: {}",
        out
    );
}

#[test]
fn one_type_authority_annotation_rendering_pins() {
    // Round 49 (issue #137's systemic review of rounds 38–47): the three
    // annotation resolvers collapsed into one TypeInfo authority, with the
    // generated structs as the drift arbiter. Pins the fixed answers:
    // `set[T]`/`frozenset[T]` are HashSet (set literals generate HashSet —
    // urllib3's PoolKey fields are Option<HashSet<(String, String)>>),
    // `socket.socket`/`threading.Event` are runtime handles (wait.py /
    // ssltransport.py compile that way), `type[X]` is the opaque
    // Option<()>, and the typing-module spellings map like the bare
    // containers.
    let out = compile(
        concat!(
            "from typing import Tuple, Optional\n",
            "class PoolKey:\n",
            "    def __init__(self, key_headers: frozenset[tuple[str, str]] | None,\n",
            "                 ready: threading.Event, sock: socket.socket,\n",
            "                 tp: type[BaseException] | None, pair: Tuple[int, str],\n",
            "                 maybe: Optional[str]) -> None:\n",
            "        self.key_headers = key_headers\n",
            "        self.ready = ready\n",
            "        self.sock = sock\n",
            "        self.tp = tp\n",
            "        self.pair = pair\n",
            "        self.maybe = maybe\n",
        ),
        "typeauth.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("Option<std::collections::HashSet<(String,String)>>"),
        "frozenset[tuple[str, str]] | None must be Option<HashSet<...>> (PoolKey-verified): {}",
        out
    );
    assert!(
        flat.contains("threading::Event"),
        "threading.Event must render the runtime handle: {}",
        out
    );
    assert!(
        flat.contains("socket::Socket"),
        "socket.socket must render socket::Socket: {}",
        out
    );
    assert!(
        flat.contains("Option<Option<()>>"),
        "type[BaseException] | None must be Option<Option<()>> (the class marker): {}",
        out
    );
    assert!(
        flat.contains("(i64,String)"),
        "Tuple[int, str] must map like the bare tuple: {}",
        out
    );
    assert!(
        flat.contains("Option<String>"),
        "Optional[str] must map like the bare Optional: {}",
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
    assert!(out.contains("move | | add (2 , 3)"), "generated: {}", out);

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
fn numpy_dtype_keyword_renders_an_enum_variant_not_a_string() {
    // The dtype variant is an IDENT: interpolating the `&str` rendered
    // `numpy :: Dtype :: "Int64"`, which is not valid Rust, so every
    // `dtype=` on zeros/ones/empty failed in rustc (issue #193).
    for (spelling, variant) in [
        ("np.float64", "Float64"),
        ("np.float32", "Float32"),
        ("np.int64", "Int64"),
        ("np.int32", "Int32"),
        ("np.bool_", "Bool"),
        ("\"int64\"", "Int64"),
        ("\"float32\"", "Float32"),
    ] {
        let out = compile(
            &format!("import numpy as np\nx = np.zeros(3, dtype={spelling})\n"),
            "npdtype.py",
        );
        assert!(
            out.contains(&format!("numpy :: Dtype :: {variant}")),
            "dtype={spelling} generated: {out}"
        );
        assert!(
            !out.contains(&format!("Dtype :: \"{variant}\"")),
            "dtype={spelling} rendered a string literal: {out}"
        );
    }
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
fn numpy_attributes_use_the_runtime_accessors() {
    // `a.shape` is a Python TUPLE and `a.T` a transpose; both used to fall
    // through to a plain field read, which printed a list and failed in
    // rustc respectively (issues #197, #204).
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    a = np.array([1.0, 2.0])\n    \
         s = a.shape\n    t = a.T\n",
        "npattr.py",
    );
    assert!(out.contains("shape_tuple ()"), "generated: {}", out);
    assert!(out.contains("transpose ()"), "generated: {}", out);
    assert!(
        !out.contains("a . T"),
        "a.T must not be a field read: {}",
        out
    );
}

#[test]
fn numpy_astype_maps_its_dtype_argument() {
    // `np.int64` is a CAST call elsewhere; as astype's argument it names a
    // dtype (issue #204).
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    a = np.array([1.0, 2.0])\n    \
         b = a.astype(np.int64)\n",
        "npastype.py",
    );
    assert!(
        out.contains("astype (numpy :: Dtype :: Int64)"),
        "generated: {}",
        out
    );
}

#[test]
fn numpy_submodules_other_than_linalg_are_refused() {
    // `np.random.rand(3)` used to lower to a bare `np :: random :: rand`
    // path and fail in rustc (issue #204).
    let err = compile_err("import numpy as np\nx = np.random.rand(3)\n", "nprandom.py");
    assert!(err.contains("np.random"), "must name the submodule: {err}");
    assert!(
        err.contains("np.linalg"),
        "must name what IS modeled: {err}"
    );
    // linalg still lowers.
    let out = compile(
        "import numpy as np\nx = np.linalg.det(np.eye(2))\n",
        "nplinalg.py",
    );
    assert!(out.contains("numpy :: linalg :: det"), "generated: {}", out);
}

#[test]
fn numpy_operator_and_list_arguments_do_not_move_their_operands() {
    // `py_div`/`py_mod`/... take their operands BY VALUE, so a variable
    // used again afterwards was a borrow-checker error in the generated
    // crate; the same for a list literal's elements (issue #201).
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    a = np.array([1.0])\n    \
         b = np.array([2.0])\n    c = b / a\n    d = a * 2.0\n",
        "npops.py",
    );
    assert!(out.contains("py_div ((b) . clone ()"), "generated: {}", out);
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    p = np.zeros(2)\n    \
         q = np.ones(2)\n    r = np.concatenate([p, q], axis=0)\n",
        "npconcat.py",
    );
    assert!(
        out.contains("vec ! [(p) . clone () , (q) . clone ()]"),
        "list elements must clone individually: {}",
        out
    );
}

#[test]
fn numpy_fallible_calls_propagate_with_question_mark() {
    // A broadcast mismatch, a singular matrix and an empty reduction are
    // CATCHABLE exceptions now, so their call sites propagate (issue #205).
    for (src, needle) in [
        ("x = np.add(np.zeros(2), np.zeros(3))", "numpy :: add"),
        ("x = np.max(np.zeros(0))", "numpy :: max"),
        ("x = np.linalg.inv(np.eye(2))", "numpy :: linalg :: inv"),
    ] {
        let out = compile(
            &format!("import numpy as np\ndef f() -> None:\n    {src}\n"),
            "npfallible.py",
        );
        assert!(out.contains(needle), "generated: {out}");
        assert!(
            out.contains(") ?"),
            "the fallible call must propagate: {out}"
        );
    }
    // The borrow-set numpy calls pass BORROWED temporaries (`&(numpy ::
    // zeros (...))`) — issue #220's zero-copy args — instead of cloning.
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    x = np.add(np.zeros(2), np.zeros(3))\n",
        "npborrow.py",
    );
    assert!(
        out.contains("numpy :: add (& (numpy :: zeros"),
        "the array args must be borrowed: {out}"
    );

    // An infallible one does NOT grow a stray `?`.
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    x = np.sum(np.zeros(2))\n",
        "npsum.py",
    );
    assert!(out.contains("numpy :: sum"), "generated: {}", out);
}

#[test]
fn ndarray_returning_functions_type_their_callers_local() {
    // A local assigned from a `-> np.ndarray` function was boxed into
    // PyValue (which has no From<NdArray>), so every use failed in rustc
    // (issue #203).
    let out = compile(
        "import numpy as np\ndef build(n: int) -> np.ndarray:\n    \
         return np.zeros(n)\ndef f() -> None:\n    v = build(5)\n    \
         print(np.sum(v))\n",
        "ndret.py",
    );
    assert!(
        !out.contains("PyValue :: from (build"),
        "an ndarray result must not box: {}",
        out
    );
}

#[test]
fn np_dot_on_provable_vectors_returns_a_scalar() {
    // numpy's inner product is a SCALAR; rython routes the provably-1-D
    // case to vdot, which returns f64 (issue #206).
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    a = np.linspace(0.0, 1.0, 4)\n    \
         b = np.ones(4)\n    s = np.dot(a, b)\n",
        "npdot.py",
    );
    assert!(out.contains("numpy :: vdot"), "generated: {}", out);
    // A matrix operand keeps the array-returning `dot`.
    let out = compile(
        "import numpy as np\ndef f() -> None:\n    m = np.eye(2)\n    \
         s = np.dot(m, m)\n",
        "npdotm.py",
    );
    assert!(out.contains("numpy :: dot"), "generated: {}", out);
}

#[test]
fn np_std_var_reject_the_axis_positional() {
    // numpy's second positional parameter of std/var is `axis`, not
    // `ddof`: `np.std(a, 1)` reduces per row there. Binding it to ddof
    // made the same call silently mean something else (issue #196).
    for fname in ["std", "var"] {
        let err = compile_err(
            &format!("import numpy as np\nx = np.{fname}(np.zeros(4), 1)\n"),
            "npstd.py",
        );
        assert!(
            err.contains("second positional parameter is `axis`"),
            "np.{fname}(a, 1) must be refused by name: {err}"
        );
    }
    // The keyword form is the supported spelling and still lowers.
    let out = compile(
        "import numpy as np\nx = np.std(np.zeros(4), ddof=1)\n",
        "npstdkw.py",
    );
    assert!(out.contains("numpy :: std"), "generated: {}", out);
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
fn loop_element_return_with_fall_through_returns_an_option() {
    // `for x in p: return x` can fall through (empty p → Python None).
    // Round 85 (the return-type directive): a function that can return
    // EXACTLY two types — the element B and None — returns Option<B> (the
    // caller decides what to do with the None). Verified against python3:
    // first([1,2,3]) → 1, first([]) → None. The old boxed-PyValue
    // unification (issue #122 step 3) is replaced by the Option.
    let out = compile(
        "def first(p):\n    for x in p:\n        return x\n",
        "iter4.py",
    );
    assert!(
        out.contains("-> Result < Option < B > , PyException >"),
        "the element | None return must be Option<B>: {}",
        out
    );
    assert!(
        out.contains("return Ok (Some (x))"),
        "the element return must Some-wrap, not box: {}",
        out
    );
    assert!(
        out.contains("Ok (None)"),
        "the fall-through must be the Option's None member: {}",
        out
    );
    assert!(
        !out.contains("stdpython :: PyValue : From < B >"),
        "the boxed-conversion bound must be gone: {}",
        out
    );
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
fn class_instance_global_wraps_fallible_construction_in_a_panicking_match() {
    // Issue #229: `REC = Klass(7)` promoted to `LazyLock<Klass>`. The
    // construction renders as a brace block (`{ Klass::new(7)? }` — the
    // argument-mapping prelude form), and the promoted-static path's
    // trailing-`?` strip only looked at the OUTER stream's last token, so
    // the `?` survived inside a closure that returns `Klass` (E0277). The
    // strip now descends into a sole brace block, and the closure
    // panics on Err like every other fallible initializer (§12.2
    // import-time divergence). The same repro's list field
    // (`self.items = ["kept"]`) infers `Vec<String>`; the store side
    // owns its string-literal elements (the literal renders Vec<&str>).
    let out = compile(
        concat!(
            "class Klass:\n",
            "    def __init__(self, n: int):\n",
            "        self.count = n\n",
            "        self.items = [\"kept\"]\n",
            "\n",
            "REC = Klass(7)\n",
            "\n",
            "def show() -> int:\n",
            "    return REC.count\n",
        ),
        "classglobal.py",
    );
    assert!(
        out.contains("pub static REC : std :: sync :: LazyLock < Klass >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("match { Klass :: new (7) }"),
        "the fallible construction must unwrap inside a match: {}",
        out
    );
    assert!(
        out.contains("initialization failed"),
        "the closure must panic on Err: {}",
        out
    );
    // No `?` may fire before the panicking match inside REC's closure.
    let static_start = out.find("pub static REC").expect("REC static emitted");
    let closure = &out[static_start..];
    let match_at = closure.find("match").expect("panicking match emitted");
    assert!(
        !closure[..match_at].contains('?'),
        "a `?` cannot precede the panicking match in the LazyLock closure: {}",
        out
    );
    assert!(
        out.contains("vec ! [(\"kept\") . to_string ()]"),
        "the list field's store must own its string elements: {}",
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
    // The merge_setting call's third argument (dict_class=OrderedDict)
    // must lower to the class's NAME STRING — the class object's runtime
    // value under the class-as-value model (round 33) — not a raw
    // class-name token (E0423) and not the pre-round-33 boxed None. The
    // temp prelude binds it as __rython_arg_2 before the call.
    let call_idx = out.find("merge_setting (").expect("call present");
    let prelude_start = out[..call_idx].rfind("let __rython_arg_").map(|i| i).unwrap_or(0);
    let block = &out[prelude_start..call_idx + 200];
    assert!(
        block.contains("\"OrderedDict\" . to_string ()")
            || block.contains("\"OrderedDict\".to_string()"),
        "class-value arg for dict_class must lower to its name string, got: {}",
        block
    );
    assert!(
        !block.contains("PyValue::None_") && !block.contains("PyValue :: None_"),
        "the class arg must not box to None: {}",
        block
    );
}

#[test]
fn class_value_lists_box_and_extend_across_fallthrough() {
    // The round-33 class-as-value model, end to end (botocore's
    // retryhandler.py shapes, verified against python3): a class NAME in
    // value position lowers to its name string; a list of classes is
    // Vec<String>; `exceptions.extend([...])` into a boxed-element Vec
    // element-converts; and the elif fall-through makes the function's
    // return the boxed PyValue (None on the fall-through).
    let out = compile(
        concat!(
            "class ChecksumError(Exception):\n",
            "    pass\n",
            "\n",
            "class ConnectionError(Exception):\n",
            "    pass\n",
            "\n",
            "def extract(kind: str):\n",
            "    if kind == \"a\":\n",
            "        return [ChecksumError]\n",
            "    elif kind == \"b\":\n",
            "        exceptions = []\n",
            "        exceptions.extend([ConnectionError, ChecksumError])\n",
            "        return exceptions\n",
        ),
        "retryhandler.py",
    );
    assert!(
        out.contains("-> Result < stdpython :: PyValue , PyException >"),
        "fall-through must box the return: {}",
        out
    );
    assert!(
        out.contains("\"ChecksumError\" . to_string ()")
            || out.contains("\"ChecksumError\".to_string()"),
        "class names must lower to strings: {}",
        out
    );
    assert!(
        out.contains("map (stdpython :: PyValue :: from)")
            || out.contains(".map(stdpython::PyValue::from)"),
        "the extend into the boxed Vec must element-convert: {}",
        out
    );
    assert!(
        out.contains("PyValue :: None_"),
        "the fall-through must be the boxed None: {}",
        out
    );
}

#[test]
fn dynamic_except_on_a_boxed_field_uses_matches_value() {
    // botocore's `except self._retryable_exceptions as e:` — the
    // exception list is a RUNTIME boxed value (a tuple of class-name
    // strings, or None), so the handler lowers to the lazy
    // matches_value if-chain instead of a static matches guard, and the
    // None-defaulted constructor parameter carries the boxed value
    // (round 33).
    let out = compile(
        concat!(
            "class ChecksumError(Exception):\n",
            "    pass\n",
            "\n",
            "class Decorator:\n",
            "    def __init__(self, retryable_exceptions=None):\n",
            "        self._retryable_exceptions = retryable_exceptions\n",
            "\n",
            "    def check(self, value: int):\n",
            "        try:\n",
            "            raise ChecksumError(\"boom\")\n",
            "        except self._retryable_exceptions as e:\n",
            "            return True\n",
        ),
        "retryhandler.py",
    );
    assert!(
        out.contains("matches_value (& (self . _retryable_exceptions)) ?")
            || out.contains("matches_value(&(self._retryable_exceptions))?"),
        "the dynamic handler must guard with matches_value: {}",
        out
    );
    assert!(
        out.contains("retryable_exceptions : stdpython :: PyValue")
            || out.contains("retryable_exceptions: stdpython::PyValue"),
        "the value-used None-defaulted param must carry the boxed value: {}",
        out
    );
    assert!(
        !out.contains("Option < () >"),
        "the value-used param must not stay Option<()>: {}",
        out
    );
}

#[test]
fn none_defaulted_value_used_parameter_boxes_at_the_call_site() {
    // A free function whose None-defaulted unannotated parameter is used
    // as a value (`y = x; return y` — round 33) types the parameter as
    // the boxed PyValue (impl Into, boxed by the prologue), and call
    // sites coerce plain arguments (including an omitted default → the
    // boxed None).
    let out = compile(
        concat!(
            "def store(x=None):\n",
            "    y = x\n",
            "    return y\n",
            "\n",
            "print(store(5))\n",
            "print(store())\n",
        ),
        "optval.py",
    );
    assert!(
        out.contains("x : impl Into < stdpython :: PyValue >")
            || out.contains("x: impl Into<stdpython::PyValue>"),
        "the value-used None-defaulted free-function param must box: {}",
        out
    );
    assert!(
        out.contains("let x : stdpython :: PyValue = x . into () ;")
            || out.contains("let x: stdpython::PyValue = x.into();"),
        "the prologue must box the parameter: {}",
        out
    );
    // The direct-call path relies on the impl Into bound (a plain `5`
    // coerces through Into), while the omitted default boxes explicitly.
    assert!(
        out.contains("store (5)"),
        "call-site arguments must pass through the Into bound: {}",
        out
    );
    assert!(
        out.contains("store (stdpython :: PyValue :: None_)")
            || out.contains("store(stdpython::PyValue::None_)"),
        "the omitted default must box to the boxed None: {}",
        out
    );
}

#[test]
fn tuple_call_return_types_as_boxed() {
    // `return tuple(retryable_exceptions)` (botocore's retryhandler)
    // always yields a tuple — the boxed value — so the function's
    // return type resolves to Result<PyValue, _> and the body's
    // PyValue::from(...) matches the signature.
    let out = compile(
        concat!(
            "def collect(kind: str):\n",
            "    retryable = []\n",
            "    retryable.extend([\"a\", \"b\"])\n",
            "    return tuple(retryable)\n",
        ),
        "retryhandler.py",
    );
    assert!(
        out.contains("-> Result < stdpython :: PyValue , PyException >"),
        "tuple() return must type boxed: {}",
        out
    );
    assert!(
        out.contains("PyValue :: from (retryable)")
            || out.contains("PyValue::from(retryable)"),
        "the return body must box the tuple: {}",
        out
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
fn global_class_instance_lowers_to_a_typed_static() {
    // Issue #189: the lazy-singleton shape (botocore's history.py) — a
    // None-initialized global whose `global`-writing function stores
    // exactly one LOCAL class construction — lowers to
    // `Mutex<Option<Class>>`: None / `Some(instance)` stores, the `is
    // None` compare reading the Option, value reads unwrapping with a
    // loud panic while None, and the getter's return type is the class.
    let out = compile(
        concat!(
            "class HistoryRecorder:\n",
            "    def __init__(self) -> None:\n",
            "        self.events = []\n",
            "RECORDER = None\n",
            "def get_recorder() -> HistoryRecorder:\n",
            "    global RECORDER\n",
            "    if RECORDER is None:\n",
            "        RECORDER = HistoryRecorder()\n",
            "    return RECORDER\n",
        ),
        "global_class_store.py",
    );
    assert!(
        out.contains("pub static RECORDER : std :: sync :: Mutex < Option < HistoryRecorder >> = std :: sync :: Mutex :: new (None)"),
        "the static must be typed Option<HistoryRecorder>: {}",
        out
    );
    assert!(
        out.contains("py_global_write (& RECORDER , Some ({ HistoryRecorder :: new () ? }))"),
        "the class store must wrap in Some: {}",
        out
    );
    assert!(
        out.contains("(stdpython :: py_global_read (& RECORDER)) . py_is_none ()"),
        "the None check must read the Option: {}",
        out
    );
    assert!(
        out.contains("stdpython :: py_global_read (& RECORDER) . expect ("),
        "the value read must unwrap the instance: {}",
        out
    );
    assert!(
        out.contains("fn get_recorder () -> Result < HistoryRecorder , PyException >"),
        "the getter returns the instance: {}",
        out
    );
}

#[test]
fn global_class_instance_stays_loud_for_unsupported_stores() {
    // Outside the recognized pattern the store is still a loud conversion
    // error: a container literal into a None-initialized Boxed global, and
    // a class instance into a global the detection disqualified (two
    // different classes). Correct-or-loud, never silently None.
    let err = compile_err(
        concat!(
            "class A:\n",
            "    pass\n",
            "class B:\n",
            "    pass\n",
            "X = None\n",
            "def set_x(flag: bool) -> None:\n",
            "    global X\n",
            "    if flag:\n",
            "        X = A()\n",
            "    else:\n",
            "        X = B()\n",
            "    return None\n",
        ),
        "global_class_loud.py",
    );
    assert!(
        err.contains("no boxed representation") && err.contains("issue #189"),
        "two different classes disqualify the pattern: {}",
        err
    );
}

#[test]
fn global_class_instance_store_warns_at_module_scope() {
    // At MODULE scope the same store degrades to a -W drop (None is
    // stored) so the module still converts — the §12 boxed-global
    // divergence carried by the warning channel. This is the issue #137
    // emscripten pattern (urllib3's `_fetcher = _StreamingFetcher()`
    // init branch): a None-initialized module value, reassigned inside
    // module-level control flow, read by function bodies.
    let (out, warnings) = compile_with_warnings(
        concat!(
            "class Fetcher:\n",
            "    pass\n",
            "def worker_available() -> bool:\n",
            "    return False\n",
            "_fetcher = None\n",
            "def streaming_ready() -> bool:\n",
            "    return _fetcher is not None\n",
            "if worker_available():\n",
            "    _fetcher = Fetcher()\n",
            "else:\n",
            "    _fetcher = None\n",
        ),
        "global_module_store.py",
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("no boxed representation")),
        "the module-scope store must warn: {:?}",
        warnings
    );
    assert!(
        out.contains("PyValue :: None_"),
        "the drop stores None: {}",
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
fn annotated_params_mixed_literal_returns_an_option() {
    // The NON-generic path (annotated parameter, unannotated return):
    // `return 1` / `return None` is exactly `i64 | None` — round 85 (the
    // return-type directive) says `Option<i64>`: the literal Some-wraps
    // and None stays the empty member (previously the signature boxed to
    // PyValue).
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
        out.contains("Result < Option < i64 > , PyException >"),
        "mixed literal/None returns must be Option<i64>: {}",
        out
    );
    assert!(
        out.contains("Some (1)"),
        "the literal return must Some-wrap: {}",
        out
    );
    assert!(
        out.contains("return Ok (None)"),
        "the None return must be the Option's empty member: {}",
        out
    );
}

#[test]
fn partial_literal_return_becomes_an_option_with_none_tail() {
    // A value return on one path and a FALL-THROUGH on the other returns
    // `1 | None` in Python. Round 85 (the return-type directive): exactly
    // two types — i64 and None — returns Option<i64>; the implicit tail
    // is the Option's None member. The old boxed-PyValue signature is
    // replaced by the Option.
    let out = compile(
        concat!(
            "def partial(flag: bool):\n",
            "    if flag:\n",
            "        return 1\n",
        ),
        "partial.py",
    );
    assert!(
        out.contains("Result < Option < i64 > , PyException >"),
        "a partial literal return must be Option<i64>: {}",
        out
    );
    assert!(
        out.contains("return Ok (Some (1))"),
        "the literal return must Some-wrap: {}",
        out
    );
    assert!(
        out.contains("Ok (None)"),
        "the fall-through tail must be the Option's None member: {}",
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
        out.contains("matches_builtin (BuiltinException :: TimeoutError)")
            || out.contains("matches (\"TimeoutError\")"),
        "the handler must match the canonical builtin: {}",
        out
    );
}

#[test]
fn urllib_calls_without_runtime_items_drop_boxed() {
    // Round 55: `from urllib.parse import urlencode` now resolves to a
    // REAL runtime item, so `urlencode(fields)` lowers as a plain call
    // with `?` — never a `urlencode::new(...)` class construction (that
    // misfire was the round-55 regression; the stdpython_class registry
    // separates class items from function items).
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
        out.contains("stdpython :: urllib :: parse :: urlencode"),
        "the call must resolve to the runtime function: {}",
        out
    );
    assert!(
        !out.contains("urlencode :: new"),
        "no class construction for a runtime function: {}",
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

#[test]
fn ssl_version_constant_attribute_reads_deref_the_lazy_static() {
    // The ssl version constants are LazyLock statics in both backends
    // (the openssl backend's real version is only knowable at runtime),
    // so a module-attribute read (`ssl.OPENSSL_VERSION`) emits a deref,
    // exactly like sys::executable: converted code sees the plain &str.
    let out = compile(
        "import ssl\n\
         \n\
         def f() -> str:\n\
         \x20   return ssl.OPENSSL_VERSION\n",
        "sslver.py",
    );
    assert!(
        out.contains("(* ssl :: OPENSSL_VERSION)"),
        "the LazyLock read must deref to the plain value: {}",
        out
    );
}

#[test]
fn ssl_version_constant_from_import_reads_deref_clone() {
    // `from ssl import OPENSSL_VERSION` (urllib3's util/ssl_.py) brings
    // the LazyLock static into scope; a plain NAME read of it must also
    // deref-clone so the value is a &str, not the static.
    let out = compile(
        "from ssl import OPENSSL_VERSION\n\
         \n\
         def f() -> bool:\n\
         \x20   return OPENSSL_VERSION.startswith(\"OpenSSL \")\n",
        "sslver.py",
    );
    assert!(
        out.contains("(* OPENSSL_VERSION) . clone ()"),
        "the imported LazyLock read must deref-clone to the plain value: {}",
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
    // A generator's yield element type: the For-target seeding (round 99)
    // types the loop element from the iterable (`for i in range(n)` seeds
    // i as int), so the collector is Vec<i64> — precise, not the boxed
    // PyValue the unseeded analysis used to emit. The collector returns
    // inside the function's Result (`return __rython_gen` bare was E0308).
    let out = compile(
        "def gen(n: int):\n\
         \x20   for i in range(n):\n\
         \x20       yield i\n",
        "boxgen.py",
    );
    assert!(
        out.contains("Vec < i64 >") || out.contains("Vec<i64>"),
        "the seeded yield element must type the collector: {}",
        out
    );
    assert!(
        out.contains("return Ok (__rython_gen)"),
        "the collector returns in the Result: {}",
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

/// Issue #137 round 19 (urllib3's emscripten fetch): a module value
/// initialized None at top level, reassigned only inside module-level
/// control flow, and read by functions promotes to a BOXED mutable
/// static — reads render py_global_read everywhere instead of E0425
/// against an init-local. The class-instance branch's store has no boxed
/// representation: None is stored and the -W channel carries the
/// divergence.
#[test]
fn none_initialized_module_global_rebound_in_module_if_is_boxed_static() {
    let src = "class Thing:\n    def __init__(self):\n        self.x = 1\n\n\
               def cond() -> bool:\n    return False\n\n\
               _holder = None\n\
               if cond():\n    _holder = Thing()\nelse:\n    _holder = None\n\n\
               def has_holder() -> bool:\n    if _holder:\n        return True\n    return False\n";
    let module = parse(src, "holder.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let options = PythonOptions::default();
    let warnings = options.definition_warnings.clone();
    let out = module
        .to_rust(
            CodeGenContext::Module("holder".to_string()),
            options,
            symbols,
        )
        .expect("converts")
        .to_string();
    assert!(
        out.contains("pub static _holder : std :: sync :: Mutex < stdpython :: PyValue >"),
        "the None-initialized, module-if-rebound global must become a boxed static: {}",
        out
    );
    assert!(
        out.contains("py_global_read (& _holder)"),
        "function reads must go through the static: {}",
        out
    );
    assert!(
        warnings
            .borrow()
            .iter()
            .any(|w| w.contains("no boxed representation")),
        "the class-instance store must be loud: {:?}",
        warnings.borrow()
    );
}

/// `-> T` where `T = typing.TypeVar("T")` lowers to the boxed PyValue in
/// return position, matching the parameter-position lowering (urllib3's
/// http2 `_LockedObject.__enter__`) — a bare `T` names nothing in Rust.
#[test]
fn typevar_return_annotation_lowers_to_pyvalue() {
    let src = "import typing\n\nT = typing.TypeVar(\"T\")\n\n\
               class Box:\n    def __init__(self, obj: T):\n        self._obj = obj\n\n\
               \x20   def get(self) -> T:\n        return self._obj\n";
    let module = parse(src, "boxmod.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let out = module
        .to_rust(
            CodeGenContext::Module("boxmod".to_string()),
            PythonOptions::default(),
            symbols,
        )
        .expect("converts")
        .to_string();
    assert!(
        !out.contains("Result < T ,"),
        "the raw TypeVar must not leak into a signature: {}",
        out
    );
    assert!(
        out.contains("fn get (& self ,) -> Result < stdpython :: PyValue"),
        "the TypeVar return must box: {}",
        out
    );
}

/// A TYPE_CHECKING import mixing a GENERATED name with a stub-only one
/// (urllib3's `from .ssl_ import _TYPE_PEER_CERT_RET,
/// _TYPE_PEER_CERT_RET_DICT`, where the DICT is a TYPE_CHECKING-only
/// TypedDict) emits the `use` for the generated name alone — previously
/// the all-or-nothing check dropped both, leaving annotations unresolved.
#[test]
fn type_checking_import_filters_per_name() {
    let sib = parse(
        "import typing\n\n_RET = typing.Union[bytes, None]\n\n\
         if typing.TYPE_CHECKING:\n    class _RET_DICT:\n        pass\n",
        "sib.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["sib".to_string()], std::rc::Rc::new(sib));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let user = parse(
        "import typing\n\nif typing.TYPE_CHECKING:\n    from sib import _RET, _RET_DICT\n\n\
         def f() -> bool:\n    return True\n",
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
        .expect("converts")
        .to_string();
    assert!(
        out.contains("_RET"),
        "the generated alias keeps its use: {}",
        out
    );
    assert!(
        !out.contains("_RET_DICT"),
        "the TYPE_CHECKING-only stub must not emit a use (E0432): {}",
        out
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

/// Issue #137 round 20 (urllib3's RequestMethods): a plain-struct class
/// subclassed ONLY cross-module emits an ACCESSOR-ONLY companion trait —
/// the subclass modules' ancestor impls and supertrait bounds name
/// `{Name}Trait`, so it must exist — while its methods stay INHERENT
/// (trait-default methods would re-route the subclasses' inherited-call
/// resolution).
#[test]
fn cross_module_only_base_emits_accessor_only_trait() {
    let mixin_src = "class Mixin:\n    def __init__(self):\n        self.headers = \"\"\n\n\
                     \x20   def ask(self) -> str:\n        return self.headers\n";
    // Convert the DEFINING module with the subclassing module visible in
    // module_defs and this module's own path set.
    let user_mod = parse(
        "from mixmod import Mixin\n\nclass Manager(Mixin):\n    def own(self) -> int:\n        return 1\n",
        "user.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["user".to_string()], std::rc::Rc::new(user_mod));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        this_module_path: vec!["mixmod".to_string()],
        ..Default::default()
    };
    let module = parse(mixin_src, "mixmod.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let out = module
        .to_rust(
            CodeGenContext::Module("mixmod".to_string()),
            options,
            symbols,
        )
        .expect("converts")
        .to_string();
    assert!(
        out.contains("pub trait MixinTrait"),
        "the cross-module-only base must emit its trait: {}",
        out
    );
    assert!(
        out.contains("fn headers (& self)"),
        "the trait carries the field accessors: {}",
        out
    );
    let trait_block = out
        .split("pub trait MixinTrait")
        .nth(1)
        .and_then(|rest| rest.split("impl MixinTrait for Mixin").next())
        .unwrap_or("");
    assert!(
        !trait_block.contains("fn ask"),
        "methods must NOT become trait defaults (accessor-only): {}",
        out
    );
    assert!(
        out.contains("fn ask (& self ,)"),
        "the method stays inherent on the struct: {}",
        out
    );
}

/// The supertrait bound on a subclass of a CROSS-MODULE base names the
/// base's trait by its crate path — the subclass module imports the
/// STRUCT, not the trait (`PoolManagerTrait:
/// crate::_request_methods::RequestMethodsTrait`, urllib3).
#[test]
fn cross_module_base_supertrait_is_named_by_crate_path() {
    let src = "from animals import Animal\n\nclass Keeper(Animal):\n    def feed(self) -> int:\n        return 1\n";
    let out = compile_with_options(src, "keeper.py", cross_module_subclass_options())
        .expect("converts");
    assert!(
        out.contains("pub trait KeeperTrait : crate :: animals :: AnimalTrait"),
        "the cross-module supertrait must be crate-path-qualified: {}",
        out
    );
}

/// Issue #180: a dict literal whose value types mix (a string and a
/// NESTED DICT) must widen to the boxed PyValue rather than erroring:
/// `{'ProviderType': 'sso', 'Credentials': {...}}` (botocore's
/// credentials.py) lowers to PyDict<String, PyValue> with the nested
/// dict boxed via PyValue::from(PyDict...), and the returned dict keeps
/// its type so `c['Credentials']['AccessKeyId']` compiles.
#[test]
fn mixed_dict_literal_with_nested_dict_boxes_values() {
    let out = compile(
        "def make_credentials(account_id: str):\n\
         \x20   credentials = {\n\
         \x20       'ProviderType': 'sso',\n\
         \x20       'Credentials': {\n\
         \x20           'AccessKeyId': 'AK',\n\
         \x20           'AccountId': account_id,\n\
         \x20       },\n\
         \x20   }\n\
         \x20   return credentials\n\
         \n\
         def main() -> int:\n\
         \x20   c = make_credentials('123')\n\
         \x20   print(c['Credentials']['AccessKeyId'])\n\
         \x20   return 0\n",
        "mixeddict.py",
    );
    // The nested dict VALUE boxes via PyValue::from, and the function's
    // inferred return type proves the dict widened to PyDict<String,
    // PyValue> (the literal itself renders as inferred PyDict::from([...])).
    assert!(
        out.contains("PyValue :: from")
            && out.contains("PyDict < String , stdpython :: PyValue >"),
        "mixed values must box into a PyDict<String, PyValue>: {}",
        out
    );
    assert!(
        out.contains("Credentials"),
        "the nested-dict key must survive: {}",
        out
    );
}

/// Issue #137 round 21: a bytes literal is an OWNED value — the typed
/// paths declare it Vec<u8>, so the rendering agrees (`b"".to_vec()`),
/// and `return Ok(b"")` against a Result<Vec<u8>> signature compiles
/// (urllib3's emscripten response).
#[test]
fn bytes_literals_render_owned() {
    let out = compile(
        "def empty() -> bytes:\n    return b\"\"\n",
        "ownedbytes.py",
    );
    assert!(
        out.contains("b\"\" . to_vec ()"),
        "the bytes literal must render owned: {}",
        out
    );
}

/// Issue #137 round 21: a class defining `__len__` participates in the
/// len() protocol — `len(x)` lowers to `stdpython::len(&x)` bound on
/// `Len`, so the impl must exist (urllib3's BytesQueueBuffer).
#[test]
fn class_with_dunder_len_implements_len() {
    let out = compile(
        "class Buf:\n    def __init__(self):\n        self._size = 0\n\n\
         \x20   def __len__(self) -> int:\n        return self._size\n",
        "lenbuf.py",
    );
    assert!(
        out.contains("impl stdpython :: Len for Buf"),
        "__len__ must produce the Len impl: {}",
        out
    );
}

#[test]
fn functools_partial_keyword_bindings_emit_in_callee_order() {
    // Issue #189-family (botocore's retryhandler): keyword bindings may
    // bind ANY subset of the callee's parameters in any order — the
    // closure's call emits arguments in the CALLEE'S DECLARED ORDER, so
    // `partial(delay_exponential, base=base, growth_factor=growth_factor)`
    // (keyword-bound parameters BEFORE the unbound one) lowers instead of
    // demanding a parameter reorder.
    let out = compile(
        concat!(
            "import functools\n",
            "\n",
            "def delay_exponential(base: int, growth_factor: int, attempts: int) -> int:\n",
            "    return base * growth_factor ** (attempts - 1)\n",
            "\n",
            "def create_delay(base: int, growth_factor: int):\n",
            "    return functools.partial(\n",
            "        delay_exponential, base=base, growth_factor=growth_factor\n",
            "    )\n",
        ),
        "partial_kw.py",
    );
    assert!(
        out.contains("move | attempts | delay_exponential (base , growth_factor , attempts)"),
        "the closure emits the callee's declared order: {}",
        out
    );
}

#[test]
fn functools_partial_keyword_call_through_the_bound_name_is_loud() {
    // A keyword call through a partial-bound name (`unit(x=-4)`) has no
    // named closure parameters to map onto — the keyword would be silently
    // dropped and the call mis-arity'd — so it is a loud conversion error
    // (the callable-as-value divergence, issue #122).
    let err = compile_err(
        concat!(
            "import functools\n",
            "\n",
            "def clamp(lo: int, hi: int, x: int) -> int:\n",
            "    return lo + hi + x\n",
            "\n",
            "unit = functools.partial(clamp, lo=0, hi=100)\n",
            "\n",
            "def f() -> int:\n",
            "    return unit(x=-4)\n",
        ),
        "partial_kw_call.py",
    );
    assert!(
        err.contains("keyword call through a functools.partial-bound name")
            && err.contains("issue #122"),
        "the keyword call must be loud: {}",
        err
    );
}

#[test]
fn boxed_self_field_method_drop_warns_with_a_readable_spelling() {
    // Issue #209: `self.items.append(x)` on an UNTYPED list field (the
    // empty literal types the field as the boxed PyValue) cannot lower —
    // the boxed value's methods are unmodeled — so the call drops through
    // the -W channel with a READABLE spelling of the dropped call, not an
    // AST Debug dump. (Annotating the field — `self.items: list[str] = []`
    // — lowers append to Vec::push; that is the rewrite the message
    // names.)
    let (out, warnings) = compile_with_warnings(
        concat!(
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items = []\n",
            "\n",
            "    def add(self, x: str) -> None:\n",
            "        self.items.append(x)\n",
        ),
        "bag.py",
    );
    assert!(
        warnings.iter().any(|w| w
            .contains("`self.items.append(...)` is dropped: the receiver is a boxed \
                       PyValue (dynamic-method divergence)")),
        "the drop must warn with a readable source spelling: {:?}",
        warnings
    );
    // The dropped call is a no-op in the generated body.
    assert!(
        out.contains("stdpython :: PyValue :: None_ ; Ok (())"),
        "the dropped call lowers to the boxed None: {}",
        out
    );
    // The pinned shape lowers for real: append becomes Vec::push.
    let (pinned, pinned_warnings) = compile_with_warnings(
        concat!(
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items: list[str] = []\n",
            "\n",
            "    def add(self, x: str) -> None:\n",
            "        self.items.append(x)\n",
        ),
        "bag_pinned.py",
    );
    assert!(
        pinned.contains("(self . items) . push (x)"),
        "the annotated field lowers append to push: {}",
        pinned
    );
    assert!(
        pinned_warnings.is_empty(),
        "the pinned shape must not warn: {:?}",
        pinned_warnings
    );
}


/// Issue #137 round 22: a method call whose NAME receiver is bound to a
/// call into an EXTERNAL module (`conn = zlib.compressobj()` — the value
/// lowered to the boxed None) drops through the -W channel like the
/// field-chain case; a receiver that is merely unknown (a socket, a
/// generic parameter) keeps its calls.
#[test]
fn method_call_on_external_bound_name_drops_loudly() {
    let src = "import zlib\n\ndef f() -> None:\n    conn = zlib.compressobj()\n    conn.compress(b\"x\")\n";
    let module = parse(src, "extname.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    // Two module_defs entries make the sibling check authoritative, so
    // zlib resolves EXTERNAL (a single-module conversion would assume it
    // is a crate sibling).
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        vec!["extname".to_string()],
        std::rc::Rc::new(module.clone()),
    );
    defs.insert(
        vec!["other".to_string()],
        std::rc::Rc::new(parse("A = 1\n", "other.py").unwrap()),
    );
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let warnings = options.definition_warnings.clone();
    let out = module
        .to_rust(
            CodeGenContext::Module("extname".to_string()),
            options,
            symbols,
        )
        .expect("converts")
        .to_string();
    assert!(
        !out.contains(". compress ("),
        "the dynamic method must not be emitted: {}",
        out
    );
    assert!(
        warnings
            .borrow()
            .iter()
            .any(|w| w.contains("dynamic-method divergence") && w.contains("compress")),
        "the drop must be loud: {:?}",
        warnings.borrow()
    );
}

/// Issue #137 round 23: an attribute FIRST assigned outside `__init__`
/// is a real Python attribute — attributes are created on assignment,
/// wherever that assignment lives — so it must become a struct field
/// (urllib3's `self.sock = ...` in connect()).
#[test]
fn attribute_assigned_only_in_a_method_becomes_a_field() {
    let out = compile(
        "class Conn:\n    def __init__(self):\n        self.opened = False\n\n\
         \x20   def connect(self) -> None:\n        self.tries = 3\n",
        "methodattr.py",
    );
    assert!(
        out.contains("pub tries : i64"),
        "the method-assigned attribute must be a field: {}",
        out
    );
}

/// The whole-class JOIN: `self.x = None` in one method and a typed store
/// in another describe ONE attribute that is a value OR None — exactly
/// Rust's `Option<T>`. Typing it `T` breaks the None store; boxing it
/// throws the type away.
#[test]
fn none_and_typed_stores_join_to_option() {
    let out = compile(
        "class Conn:\n    def __init__(self):\n        self.opened = False\n\n\
         \x20   def connect(self) -> None:\n        self.count = 5\n\n\
         \x20   def close(self) -> None:\n        self.count = None\n",
        "joinattr.py",
    );
    assert!(
        out.contains("pub count : Option < i64 >"),
        "None plus a typed store must join to Option<T>: {}",
        out
    );
}

/// A declared annotation is a FIRST PREFERENCE that observed stores
/// override: when every store agrees on a concrete type, what the class
/// actually stores is the better evidence. An inconclusive join keeps
/// the annotation.
#[test]
fn confident_stores_override_a_class_annotation() {
    let out = compile(
        "class Box:\n    payload: str\n\n\
         \x20   def __init__(self):\n        self.ready = False\n\n\
         \x20   def fill(self) -> None:\n        self.payload = 7\n",
        "annoverride.py",
    );
    assert!(
        out.contains("pub payload : i64"),
        "the agreeing store must override the annotation: {}",
        out
    );
}

/// An attribute a class READS but never assigns, where the base is
/// EXTERNAL (unmodeled), belongs to that base. It lowers to a BOXED
/// field so the reads compile, and the degradation is LOUD — nothing
/// populates it.
#[test]
fn unassigned_read_with_an_external_base_boxes_loudly() {
    let (out, warnings) = compile_with_warnings(
        "from http.client import HTTPConnection as _HTTPConnection\n\n\
         class Conn(_HTTPConnection):\n    def __init__(self):\n        self.opened = False\n\n\
         \x20   def describe(self) -> bool:\n        return self.sock\n",
        "extbase.py",
    );
    assert!(
        out.contains("pub sock : stdpython :: PyValue"),
        "the external base's attribute must box: {}",
        out
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("external-base divergence") && w.contains("sock")),
        "the degradation must be loud: {:?}",
        warnings
    );
}

/// The boxed synthesis is gated on an UNMODELED base: with no base, a
/// read of a never-assigned attribute is a genuine Python AttributeError
/// and must NOT be papered over with a silently empty field.
#[test]
fn unassigned_read_without_a_base_is_not_synthesized() {
    let out = compile(
        "class Plain:\n    def __init__(self):\n        self.opened = False\n\n\
         \x20   def describe(self) -> bool:\n        return self.opened\n",
        "nobase.py",
    );
    assert!(
        !out.contains("pub missing"),
        "no base means no synthesis: {}",
        out
    );
    assert!(
        out.contains("pub opened : bool"),
        "the real field still lands: {}",
        out
    );
}

// ---------------------------------------------------------------------
// Issue #181: functools.singledispatch
//
// The family is fused into ONE `isinstance`-dispatching function, which
// the monomorphizing specialization pass then lowers into a morph per
// registered type plus the `_any` residual. Inside each morph the
// dispatch parameter is a CONCRETE type, so specialization bodies get
// real `str`/`int` methods instead of method calls on a boxed value.
// ---------------------------------------------------------------------

const SINGLEDISPATCH: &str = concat!(
    "import functools\n",
    "\n",
    "@functools.singledispatch\n",
    "def describe(value):\n",
    "    return \"other\"\n",
    "\n",
    "@describe.register(int)\n",
    "def _(n):\n",
    "    return \"int\"\n",
    "\n",
    "@describe.register(str)\n",
    "def _(text):\n",
    "    return \"str \" + text\n",
);

#[test]
fn singledispatch_registers_become_morphs() {
    let out = compile(SINGLEDISPATCH, "sd.py");
    for expected in [
        "fn describe_int",
        "fn describe_str",
        "fn describe_any",
    ] {
        assert!(out.contains(expected), "missing {expected}: {}", out);
    }
    // The `_`-named register definitions never emit functions of their
    // own: they exist only as arms of the fused generic.
    assert!(
        !out.contains("pub fn _ ("),
        "register definitions must be absorbed: {}",
        out
    );
}

#[test]
fn singledispatch_specialization_gets_a_concrete_parameter() {
    let out = compile(SINGLEDISPATCH, "sd2.py");
    // The str morph binds a real String, which is the whole point: the
    // specialization body can call str methods on it.
    assert!(
        out.contains("fn describe_str (value : impl Into < String >)"),
        "the str morph must take a concrete String: {}",
        out
    );
    assert!(
        out.contains("fn describe_int (value : i64)"),
        "the int morph must take a concrete i64: {}",
        out
    );
}

#[test]
fn singledispatch_binds_the_specializations_own_parameter_name() {
    let out = compile(SINGLEDISPATCH, "sd3.py");
    // `def _(text)` reads `text`; the fused body binds it to the
    // generic's parameter so the specialization body is unchanged.
    assert!(
        out.contains("text = value"),
        "the specialization's parameter name must be bound: {}",
        out
    );
}

#[test]
fn singledispatch_call_sites_dispatch_statically() {
    let out = compile(
        &format!("{}\ndef main():\n    print(describe(1))\n    print(describe(\"x\"))\n", SINGLEDISPATCH),
        "sd4.py",
    );
    assert!(out.contains("describe_int (1)"), "generated: {}", out);
    assert!(out.contains("describe_str (\"x\")"), "generated: {}", out);
}

#[test]
fn singledispatch_shares_the_generics_parameter_name() {
    // A specialization whose parameter is already the generic's name
    // needs no binding statement.
    let out = compile(
        concat!(
            "from functools import singledispatch\n",
            "\n",
            "@singledispatch\n",
            "def describe(value):\n",
            "    return \"other\"\n",
            "\n",
            "@describe.register(str)\n",
            "def _(value):\n",
            "    return value\n",
        ),
        "sd5.py",
    );
    assert!(out.contains("fn describe_str"), "generated: {}", out);
    assert!(
        !out.contains("value = value"),
        "an identity binding must not be emitted: {}",
        out
    );
}

#[test]
fn singledispatch_register_without_a_generic_is_loud() {
    let err = compile_err(
        concat!(
            "def describe(value):\n",
            "    return \"other\"\n",
            "\n",
            "@describe.register(str)\n",
            "def _(text):\n",
            "    return text\n",
        ),
        "sdbad.py",
    );
    assert!(
        err.contains("is not a `@functools.singledispatch` definition"),
        "unexpected error: {}",
        err
    );
}

#[test]
fn singledispatch_unreadable_register_form_is_loud() {
    // The annotation-typed form (`@describe.register` with the dispatch
    // type on the parameter) is not read; refusing beats dropping it.
    let err = compile_err(
        concat!(
            "import functools\n",
            "\n",
            "@functools.singledispatch\n",
            "def describe(value):\n",
            "    return \"other\"\n",
            "\n",
            "@describe.register\n",
            "def _(text: str):\n",
            "    return text\n",
        ),
        "sdann.py",
    );
    assert!(
        err.contains("is not in a form rython can read"),
        "unexpected error: {}",
        err
    );
}

#[test]
fn singledispatch_arity_mismatch_is_loud() {
    let err = compile_err(
        concat!(
            "import functools\n",
            "\n",
            "@functools.singledispatch\n",
            "def describe(value, extra):\n",
            "    return \"other\"\n",
            "\n",
            "@describe.register(str)\n",
            "def _(text):\n",
            "    return text\n",
        ),
        "sdarity.py",
    );
    assert!(
        err.contains("parameter(s) but the generic"),
        "unexpected error: {}",
        err
    );
}

// ---------------------------------------------------------------------
// Issue #222: return-type inference for parameters and computed values.
//
// A signature that collapses to `-> Result<(), PyException>` while the
// body still emits `Ok(<value>)` is code rustc rejects, so these cases
// were not cosmetic gaps. The fixes are additive: an annotated parameter
// is typed from its annotation, and anything else falls back to the type
// every `return` agrees on — never overriding a type an earlier rule
// already derived.
// ---------------------------------------------------------------------

#[test]
fn returning_an_annotated_parameter_types_the_signature() {
    let out = compile("def g(x: int):\n    return x\n", "retparam.py");
    assert!(
        out.contains("fn g (x : i64) -> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_str_parameter_types_it_as_the_owned_string() {
    // A `str` parameter arrives as `impl Into<String>` and the prologue
    // converts it, so the returned value is an owned String.
    let out = compile("def g(s: str):\n    return s\n", "retstr.py");
    assert!(
        out.contains("-> Result < String , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_an_arithmetic_expression_types_the_signature() {
    let out = compile("def h(x: int):\n    return x + 1\n", "retbin.py");
    assert!(
        out.contains("-> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
    // int * float widens to float, like Python.
    let out = compile("def h(x: int, y: float):\n    return x * y\n", "retbin2.py");
    assert!(
        out.contains("-> Result < f64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_builtin_call_types_the_signature() {
    let out = compile("def j(xs: list[int]):\n    return len(xs)\n", "retlen.py");
    assert!(
        out.contains("-> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
    let out = compile("def n(x: int):\n    return str(x)\n", "retstr2.py");
    assert!(
        out.contains("-> Result < String , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_container_of_parameters_types_the_signature() {
    let out = compile("def o(x: int):\n    return [x, x]\n", "retvec.py");
    assert!(
        out.contains("-> Result < Vec < i64 > , PyException >"),
        "generated: {}",
        out
    );
    let out = compile("def p(a: int, b: int):\n    return (a, b)\n", "rettup.py");
    assert!(
        out.contains("-> Result < (i64 , i64) , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn a_none_returning_body_still_lowers_to_unit() {
    // The fallback must not claim a type for a body that genuinely
    // returns Python's None — `()` is the correct lowering there.
    let out = compile("def f(x: int):\n    return None\n", "retnone.py");
    assert!(
        out.contains("-> Result < () , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn disagreeing_returns_still_box_rather_than_picking_a_winner() {
    // Two returns of different types keep the existing literal-boxing
    // behavior; the fallback refuses rather than choosing one.
    let out = compile(
        "def f(x: int):\n    if x:\n        return 1\n    return \"s\"\n",
        "retmix.py",
    );
    assert!(
        out.contains("-> Result < stdpython :: PyValue , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn an_untypeable_return_still_lowers_to_unit() {
    // A METHOD call has no inferred type — there is no Python-level
    // method return table — so the inferrer has no answer and refuses
    // rather than guessing at one.
    //
    // (This case used `sorted(xs)` until the iterator builtins learned to
    // carry their element type, then `s.splitlines()` until the str-method
    // table typed it; the assertion is about the refusal, so it moves to
    // an expression that is still genuinely untypeable.)
    let out = compile("def m(s: str):\n    return s.partition(\",\")\n", "retunk.py");
    assert!(
        out.contains("-> Result < () , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn a_chained_external_call_field_boxes_instead_of_failing() {
    // urllib3's ZstdDecoder: `self._obj = zstd.ZstdDecompressor().
    // decompressobj()` — a method call on a CALL RESULT whose chain root
    // is an external module. The direct `mod.fn()` field shape had an
    // external-boxing rule; the chained twin fell through to the loud
    // "cannot infer a type" error and aborted the WHOLE urllib3
    // conversion at response.py. The chain now boxes the same way, and
    // the try/except fallback (`zstd = None` shadowing the aliased
    // import) counts as external too.
    let out = compile(
        concat!(
            "try:\n",
            "    import zstandard as zstd\n",
            "except (AttributeError, ImportError, ValueError):\n",
            "    zstd = None\n",
            "\n",
            "class ZstdDecoder:\n",
            "    def __init__(self) -> None:\n",
            "        self._obj = zstd.ZstdDecompressor().decompressobj()\n",
        ),
        "extchainfield.py",
    );
    assert!(
        out.contains("pub _obj : stdpython :: PyValue"),
        "the chained external call must box the field: {}",
        out
    );
}

// ---------------------------------------------------------------------
// Issue #222, iterator builtins: sorted/filter/map/list carry their
// argument's element type, so a function returning one gets a real
// signature instead of collapsing to unit.
//
// Each rule mirrors the emitted lowering, not just Python semantics:
// `sorted` renders `stdpython::sorted(&[T]) -> Vec<T>`, `filter` renders
// `filter_fallible(f, Vec<T>) -> Result<Vec<T>, _>`, and `map` renders
// `map_fallible(f, Vec<T>) -> Result<Vec<U>, _>` where U is the
// callable's return type.
// ---------------------------------------------------------------------

const ITER_HELPERS: &str = concat!(
    "def double(n: int) -> int:\n",
    "    return n * 2\n",
    "\n",
    "def keep(n: int) -> bool:\n",
    "    return n > 0\n",
    "\n",
);

#[test]
fn sorted_preserves_the_element_type() {
    let out = compile(
        &format!("{}def a(xs: list[int]):\n    return sorted(xs)\n", ITER_HELPERS),
        "itersorted.py",
    );
    assert!(
        out.contains("fn a (xs : Vec < i64 >) -> Result < Vec < i64 > , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn filter_preserves_the_element_type() {
    let out = compile(
        &format!("{}def c(xs: list[int]):\n    return list(filter(keep, xs))\n", ITER_HELPERS),
        "iterfilter.py",
    );
    assert!(
        out.contains("-> Result < Vec < i64 > , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn map_takes_its_element_type_from_the_callable() {
    // The element type is the CALLABLE's return type, not the iterable's.
    let out = compile(
        &format!(
            "{}def to_text(n: int) -> str:\n    return str(n)\n\n\
             def b(xs: list[int]):\n    return list(map(to_text, xs))\n",
            ITER_HELPERS
        ),
        "itermap.py",
    );
    assert!(
        out.contains("-> Result < Vec < String > , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn map_over_an_unresolvable_callable_stays_untyped() {
    // A bound method (`str.strip`) is not a name this can resolve, so the
    // element type is refused rather than guessed at.
    let out = compile(
        "def d(xs: list[str]):\n    return list(map(str.strip, xs))\n",
        "itermapunk.py",
    );
    assert!(
        out.contains("-> Result < () , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn sorted_over_an_untyped_iterable_stays_untyped() {
    // No element type to carry: refused, not guessed.
    let out = compile(
        "def e(xs):\n    ys = sorted(xs)\n    return ys\n",
        "itersortedunk.py",
    );
    assert!(
        !out.contains("-> Result < Vec < i64 >"),
        "must not invent an element type: {}",
        out
    );
}

// ---------------------------------------------------------------------
// Issue #222, self-method and module-path call returns: an unannotated
// method returning a call to another method of its own class
// (`return self._retries()`), and a function returning a crate-module
// function by path (`return helper.parse(s)`), both used to collapse to
// `-> Result<(), PyException>` while the body emitted `Ok(call?)` —
// rustc rejects that shape. The two rules sit at the END of the
// resolution chain: they can only replace a unit signature, all returns
// must agree, and unresolvable callees refuse rather than guess.
// ---------------------------------------------------------------------

#[test]
fn returning_a_self_method_call_types_the_signature() {
    // The callee is itself unannotated, so its own all-returns
    // unification (`return 3` → int) types the caller — one level deep.
    let out = compile(
        concat!(
            "class Retry:\n",
            "    def _retries(self):\n",
            "        return 3\n",
            "\n",
            "    def total(self):\n",
            "        return self._retries()\n",
        ),
        "retselfcall.py",
    );
    assert!(
        out.contains("fn total (& self ,) -> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_self_method_call_with_annotation_types_the_signature() {
    // An annotated callee resolves through its annotation, alias-aware.
    let out = compile(
        concat!(
            "class Retry:\n",
            "    def _retries(self) -> int:\n",
            "        return 3\n",
            "\n",
            "    def total(self):\n",
            "        return self._retries()\n",
        ),
        "retselfann.py",
    );
    assert!(
        out.contains("fn total (& self ,) -> Result < i64 , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn a_disagreeing_self_method_return_stays_unit() {
    // Two returns naming different methods with different types refuse —
    // the signature keeps its unit fallback rather than picking a winner.
    let out = compile(
        concat!(
            "class C:\n",
            "    def a(self) -> int:\n",
            "        return 1\n",
            "\n",
            "    def b(self) -> str:\n",
            "        return \"s\"\n",
            "\n",
            "    def pick(self, flag: int):\n",
            "        if flag:\n",
            "            return self.a()\n",
            "        return self.b()\n",
        ),
        "retselfmix.py",
    );
    assert!(
        out.contains("fn pick (& self , flag : i64) -> Result < () , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_none_annotated_self_method_stays_unit() {
    // A `-> None` callee's value is Python None — the unit signature is
    // already the correct lowering, so the rule declines.
    let out = compile(
        concat!(
            "class C:\n",
            "    def reset(self) -> None:\n",
            "        pass\n",
            "\n",
            "    def run(self):\n",
            "        return self.reset()\n",
        ),
        "retselfnone.py",
    );
    assert!(
        out.contains("fn run (& self ,) -> Result < () , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_module_path_call_types_the_signature() {
    // `from . import helper` then `return helper.parse(s)` — the callee's
    // return annotation, resolved in its DEFINING module.
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        vec!["mainmod".to_string(), "helper".to_string()],
        std::rc::Rc::new(
            parse("def parse(s: str) -> str:\n    return s\n", "helper.py").unwrap(),
        ),
    );
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        module_path: vec!["mainmod".to_string()],
        this_module_path: vec!["mainmod".to_string()],
        ..Default::default()
    };
    let out = compile_with_options(
        "from . import helper\n\ndef f(s: str):\n    return helper.parse(s)\n",
        "retmodcall.py",
        options,
    )
    .expect("module converts");
    assert!(
        out.contains("fn f (s : impl Into < String >) -> Result < String , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_self_field_types_the_signature() {
    // The deferred self-field half: the field's inferred type comes from
    // the same infer_fields table the struct uses.
    let out = compile(
        concat!(
            "class Conn:\n",
            "    def __init__(self, scheme: str):\n",
            "        self.scheme = scheme\n",
            "\n",
            "    def direct(self):\n",
            "        return self.scheme\n",
        ),
        "retselffield.py",
    );
    assert!(
        out.contains("fn direct (& self ,) -> Result < String , PyException >"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_a_local_assigned_a_self_field_types_the_signature() {
    // One-step indirection (`box = self.scheme; return box`) — the local's
    // single self-field assignment types the return, and the STORE clones
    // the immutable field out of the shared receiver (E0507 otherwise).
    let out = compile(
        concat!(
            "class Conn:\n",
            "    def __init__(self, scheme: str):\n",
            "        self.scheme = scheme\n",
            "\n",
            "    def give(self):\n",
            "        box = self.scheme\n",
            "        return box\n",
        ),
        "retselflocal.py",
    );
    assert!(
        out.contains("fn give (& self ,) -> Result < String , PyException >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("= (self . scheme) . clone ()"),
        "the local store must clone the immutable field out of &self: {}",
        out
    );
}

#[test]
fn returning_a_self_field_directly_clones_out_of_self() {
    // The MOVE side: a non-Copy field read moved into `Ok(..)` would
    // leave `&self` — the return clones it (Python objects are
    // references; the clone reproduces the caller's value).
    let out = compile(
        concat!(
            "class Conn:\n",
            "    def __init__(self, scheme: str):\n",
            "        self.scheme = scheme\n",
            "\n",
            "    def direct(self):\n",
            "        return self.scheme\n",
        ),
        "retselffieldclone.py",
    );
    assert!(
        out.contains("return Ok ((self . scheme) . clone ())"),
        "generated: {}",
        out
    );
}

#[test]
fn returning_an_unresolvable_module_path_call_stays_unit() {
    // The receiver is not a crate module (json is stdpython's): the rule
    // refuses rather than guessing at the runtime's return type.
    let out = compile(
        "import json\n\ndef f(s: str):\n    return json.loads(s)\n",
        "retmodunk.py",
    );
    assert!(
        out.contains("-> Result < () , PyException >"),
        "generated: {}",
        out
    );
}

// ---------------------------------------------------------------------
// Issue #223: a bare reference to a monomorphized function.
//
// Only the morphs (`f_str`, `f_any`, ...) and, when one can be planned,
// the dynamic router carry the original name. With no router the bare
// name resolved to nothing and the generated crate failed E0425 with
// nothing pointing at the construct — silently different in the one way
// that matters, an unexplained build break. It is a conversion error now.
// ---------------------------------------------------------------------

#[test]
fn a_morph_reference_with_a_router_still_lowers() {
    // The router exists here (both morphs return String), so the bare
    // name resolves to it and passing the function as a value is fine.
    let out = compile(
        concat!(
            "def f(x):\n",
            "    if isinstance(x, str):\n",
            "        return x\n",
            "    return \"other\"\n",
            "\n",
            "def g(xs):\n",
            "    return list(map(f, xs))\n",
        ),
        "morphok.py",
    );
    assert!(out.contains("map_fallible (f ,"), "generated: {}", out);
}

#[test]
fn a_morph_reference_without_a_router_is_loud() {
    // The residual morph returns a value built from its untyped
    // parameter, so no morph return type derives and no router can be
    // planned — the reference has nothing to resolve to.
    let err = compile_err(
        concat!(
            "import functools\n",
            "\n",
            "@functools.singledispatch\n",
            "def pick(value):\n",
            "    return sorted(value)\n",
            "\n",
            "@pick.register(int)\n",
            "def _(n):\n",
            "    return [n]\n",
            "\n",
            "def use(xs):\n",
            "    return list(map(pick, xs))\n",
        ),
        "morphbad.py",
    );
    assert!(
        err.contains("is used as a value")
            && err.contains("no dynamic router could be planned"),
        "unexpected error: {}",
        err
    );
}

// ---------------------------------------------------------------------
// Issue #137 round 25: the external-base READ synthesis and the accessor
// rewrite have to agree on the field set.
//
// Round 23 gave `infer_fields` a synthesis for attributes a class reads
// but never assigns when its base is external and unmodeled (urllib3's
// HTTPConnection reading `self.port` off http.client.HTTPConnection).
// Round 24 taught `owns_field` about stores in every method — but not
// about that synthesis, so the field was on the struct and in the trait
// while `field_owner_depth` said no class owned it. The rewrite routing
// `self.x` through `self.x()` inside a generic trait default then never
// fired, and the body read the accessor METHOD as a value (E0615).
// ---------------------------------------------------------------------

#[test]
fn a_read_synthesized_field_routes_through_its_accessor() {
    let (out, warnings) = compile_with_warnings(
        concat!(
            // The base must be a bare NAME, and the class must have an
            // __init__ (infer_fields yields nothing without one) — the
            // shape urllib3 actually uses.
            "from http.client import HTTPConnection as _HTTPConnection\n",
            "\n",
            "class Conn(_HTTPConnection):\n",
            "    def __init__(self, label: str):\n",
            "        self.label = label\n",
            "\n",
            "    def describe(self) -> str:\n",
            "        return repr(self.port)\n",
        ),
        "readsynth.py",
    );
    // The synthesis itself must still be loud.
    assert!(
        warnings.iter().any(|w| w.contains("external-base divergence")
            && w.contains("port")),
        "the synthesis must stay loud: {:?}",
        warnings
    );
    // The synthesized field is on the struct and its accessor is on the
    // trait, so the generic default body must call the accessor.
    assert!(
        out.contains("pub port : stdpython :: PyValue"),
        "the synthesis must put the field on the struct: {}",
        out
    );
    assert!(
        out.contains("fn port (& self) -> stdpython :: PyValue ;"),
        "the trait must declare the accessor: {}",
        out
    );
    assert!(
        out.contains("repr (& (self . port ()))"),
        "the trait default must read through the accessor, not the field: {}",
        out
    );
}

// ---------------------------------------------------------------------
// Issue #137 round 26: dropping a read needs POSITIVE evidence of boxing.
//
// `receiver_is_pyvalue` accepts `TypeInfo::PyObject` as well as
// `PyValue` — and `PyObject` is the inferrer's "no answer". Round 24
// widened a drop on the back of that helper and discarded a live value:
// a module global bound to `Klass()` printed `None` where CPython
// printed `['kept']`. The signal is now `PyValue`/`PyValueMember` only,
// which a concrete class can never be.
// ---------------------------------------------------------------------

const ROUND26: &str = concat!(
    "from typing import Any\n",
    "\n",
    "class Klass:\n",
    "    def __init__(self, n: int):\n",
    "        self.count = n\n",
    "\n",
    "REC = Klass(7)\n",
    "\n",
    "def boxed_read(v: Any):\n",
    "    return v.whatever\n",
    "\n",
    "def boxed_protocol(v: Any):\n",
    "    return v.lower()\n",
    "\n",
    "def concrete_read():\n",
    "    return REC.count\n",
);

#[test]
fn a_positively_boxed_name_read_drops_loudly() {
    let (out, warnings) = compile_with_warnings(ROUND26, "r26a.py");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("dynamic-attribute divergence") && w.contains("whatever")),
        "the drop must be loud: {:?}",
        warnings
    );
    assert!(
        out.contains("fn boxed_read (v : stdpython :: PyValue)"),
        "generated: {}",
        out
    );
}

#[test]
fn a_concrete_class_global_read_is_never_dropped() {
    // Round 24's exact counterexample. `REC = Klass()` infers
    // TypeInfo::Class, so the positive signal cannot match it and the
    // live value survives.
    let out = compile(ROUND26, "r26b.py");
    assert!(
        out.contains("(* REC) . clone () . count"),
        "the concrete read must survive: {}",
        out
    );
}

#[test]
fn a_protocol_method_on_a_boxed_name_is_not_dropped() {
    // `v.lower()` on a boxed value is real code the runtime forwards —
    // dropping its callee would emit `PyValue::None_(...)` (E0618).
    // Round 64: the boxed str method now dispatches on the runtime
    // member (py_boxed_lower) instead of the plain PyStrOps form.
    let out = compile(ROUND26, "r26c.py");
    assert!(
        out.contains("py_boxed_lower"),
        "a protocol method must survive as the boxed dispatch: {}",
        out
    );
    assert!(
        !out.contains("PyValue :: None_ (") && !out.contains("PyValue::None_("),
        "the callee must not be dropped: {}",
        out
    );
}

// ---------------------------------------------------------------------
// `type(self).__name__` and `type(x).__name__` (the #137 sweep's
// class-name repr family): the whole expression IS the class name. The
// self receiver resolves statically; a concrete non-self receiver routes
// through the boxed value's runtime type name (CPython's spelling); an
// inferred GENERIC receiver drops loudly (no `PyValue: From<T>` bound
// can be added here).
// ---------------------------------------------------------------------

#[test]
fn type_self_dunder_name_is_the_class_name() {
    let out = compile(
        concat!(
            "class Pool:\n",
            "    def __init__(self, host: str):\n",
            "        self.host = host\n",
            "\n",
            "    def typename(self) -> str:\n",
            "        return type(self).__name__\n",
        ),
        "typename.py",
    );
    assert!(
        out.contains("stringify ! (Pool) . to_string ()"),
        "type(self).__name__ must be the class name string: {}",
        out
    );
}

#[test]
fn type_concrete_arg_dunder_name_uses_the_runtime_type_name() {
    let out = compile(
        "def name_of(x: int) -> str:\n    return type(x).__name__\n",
        "typenamearg.py",
    );
    assert!(
        out.contains("py_value_type_name"),
        "type(x).__name__ on a concrete receiver uses the runtime type name: {}",
        out
    );
}

#[test]
fn type_generic_arg_dunder_name_drops_loudly() {
    let (_, warnings) = compile_with_warnings(
        "def name_of(x):\n    return type(x).__name__\n",
        "typenamegen.py",
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("type(x).__name__ on an inferred generic parameter is dropped")),
        "the generic receiver must warn: {:?}",
        warnings
    );
}

// ---------------------------------------------------------------------
// The #137 boxed-field-arg cluster: a PyValue-typed self-field read in a
// MOVE position (a call argument, a boxed return) clones out of the
// shared receiver — the wrap/binding would move out of `&self` (E0507).
// PyValue's clone is the Arc-sharing reference copy, so it reproduces
// Python's semantics; mutable containers are NOT cloned, keeping their
// E0507 loud (issue #79's discipline).
// ---------------------------------------------------------------------

#[test]
fn a_pyvalue_self_field_argument_clones_out_of_self() {
    let out = compile(
        concat!(
            "from typing import Any\n",
            "\n",
            "class Resp:\n",
            "    def __init__(self) -> None:\n",
            "        self._fp: Any = b\"data\"\n",
            "\n",
            "    def check(self) -> bool:\n",
            "        return is_open(self._fp)\n",
            "\n",
            "def is_open(x: Any) -> bool:\n",
            "    return x is not None\n",
        ),
        "pyvaluearg.py",
    );
    assert!(
        out.contains("(self . _fp) . clone ()"),
        "the boxed field argument must clone out of &self: {}",
        out
    );
}

#[test]
fn a_pyvalue_self_field_return_clones_out_of_self() {
    // A function whose resolved return is PyValue wraps its returns; a
    // PyValue self-field read is already boxed and the wrap would move —
    // it clones instead (issue #137).
    let out = compile(
        concat!(
            "from typing import Any\n",
            "\n",
            "class Resp:\n",
            "    def __init__(self) -> None:\n",
            "        self._fp: Any = b\"data\"\n",
            "\n",
            "    def raw(self):\n",
            "        return self._fp\n",
        ),
        "pyvalueret.py",
    );
    assert!(
        out.contains("return Ok ((self . _fp) . clone ())"),
        "the boxed field return must clone out of &self: {}",
        out
    );
}

// ---------------------------------------------------------------------
// Python's `and`/`or` return OPERANDS, not booleans. The fold below
// reproduces that when the operands' types unify — the Option/String
// mix (`ca_certs and expanduser(ca_certs)` — urllib3) returns the
// operand with the Option wrapping; anything else keeps the `&&`/`||`
// approximation, which is loud in rustc (§12.1) — never a silent
// operand-vs-bool swap. Also pins the Option<String> literal argument
// ownership (`pick("x")` for a `str | None` parameter).
// ---------------------------------------------------------------------

#[test]
fn and_over_option_and_value_returns_the_operand() {
    let out = compile(
        "def pick(ca: str | None, x: str) -> str | None:\n    return ca and x\n",
        "andopt.py",
    );
    assert!(
        out.contains("if (__rython_and) . is_truthy () { Some (x) } else { __rython_and }"),
        "the truthy arm must wrap the value, the falsy arm the Option: {}",
        out
    );
}

#[test]
fn or_over_value_and_option_returns_the_operand() {
    let out = compile(
        "def pick_or(ca: str, x: str | None) -> str | None:\n    return ca or x\n",
        "oropt.py",
    );
    assert!(
        out.contains("if (__rython_or) . is_truthy () { Some (__rython_or) } else { x }"),
        "the truthy arm must wrap the value, the falsy arm the Option: {}",
        out
    );
}

#[test]
fn ununifiable_operands_keep_the_loud_boolean_approximation() {
    // `bool and String` has no static result type — the approximation
    // stays, and rustc reports the mismatch loudly rather than the
    // codegen silently returning a bool.
    let out = compile(
        "def f(flag: bool, s: str) -> str:\n    return flag and s\n",
        "andbool.py",
    );
    assert!(
        out.contains("(flag) && (s)"),
        "ununifiable operands keep the && approximation: {}",
        out
    );
}

#[test]
fn option_and_untyped_call_uses_the_option_arm() {
    // Round 43: `ca and expanduser(ca)` (urllib3's ca_certs) — the first
    // operand is Option<String>, the second is a CALL whose return type
    // infers PyObject (method/module-call returns are unresolved) but
    // renders the inner String. The fold must still use the
    // operand-returning Option arm (Some-wrap the truthy arm), not the
    // `&&` boolean approximation (which rustc rejects — Option has no
    // bool operator).
    let out = compile(
        "def pick(ca: str | None, fn) -> str | None:\n    return ca and fn(ca)\n",
        "andcall.py",
    );
    assert!(
        out.contains("__rython_and") && out.contains("is_truthy () { Some (")
            || out.contains("__rython_and") && out.contains("is_truthy() { Some("),
        "Option and an untyped call must use the operand-returning fold: {}",
        out
    );
    assert!(
        !out.contains("(ca) && (fn"),
        "must not fall back to the && approximation: {}",
        out
    );
}

#[test]
fn option_or_string_literal_owns_the_literal() {
    // Round 43: `scheme or "http"` (urllib3) — the falsy arm Some-wraps
    // the string literal and OWNS it (`Some(("http").to_string())`), so
    // the Option<String> slot typechecks (a raw &str literal would make
    // Option<&str> vs Option<String>).
    let out = compile(
        "def pick(scheme: str | None) -> str | None:\n    return scheme or \"http\"\n",
        "orlit.py",
    );
    assert!(
        out.contains("Some ((\"http\") . to_string ())") || out.contains("Some((\"http\").to_string())"),
        "the literal arm must own the string: {}",
        out
    );
}

#[test]
fn self_field_option_or_concrete_unwraps_to_plain() {
    // Round 48: `self.path or "/"` where the field is `str | None`
    // (urllib3's Url) — the fold's Option arm UNWRAPS the Some to the
    // inner value and defaults to the concrete operand (Python's result
    // is never None). A NAME-typed Option operand (`scheme or "http"`)
    // keeps the round-43 Option-producing fold; only the self-FIELD
    // case (whose Option-ness infer_type cannot see) unwraps.
    let out = compile(
        "class U:\n\
         \x20   def __init__(self) -> None:\n\
         \x20       self.path: str | None = None\n\
         \x20   def request_uri(self) -> str:\n\
         \x20       return self.path or \"/\"\n",
        "fieldor.py",
    );
    assert!(
        out.contains("match __rython_field {")
            || out.contains("match __rython_field {")
            || out.contains("Some (__rython_inner) => __rython_inner"),
        "the self-field Option or-fold must unwrap to the inner: {}",
        out
    );
    assert!(
        out.contains("None => (\"/\") . to_string ()") || out.contains("None => (\"/\").to_string()"),
        "the concrete default must be owned: {}",
        out
    );
}

#[test]
fn option_and_call_narrows_the_inner_argument() {
    // Round 48: `ca_certs and os.path.expanduser(ca_certs)` where
    // ca_certs is `str | None` (urllib3) — the truthy arm passes the
    // UNWRAPPED inner string to the call (`expanduser` expects a path),
    // never the Option. The fold narrows the name for the operand's
    // re-render.
    let out = compile(
        "def exp(p: str) -> str:\n\
         \x20   return p\n\
         \ndef pick(ca: str | None) -> str | None:\n\
         \x20   return ca and exp(ca)\n",
        "andcall.py",
    );
    assert!(
        out.contains("(ca) . clone () . unwrap ()") || out.contains("(ca).clone().unwrap()"),
        "the truthy arm must pass the unwrapped inner: {}",
        out
    );
}

#[test]
fn none_seeded_local_or_call_folds_the_option_arm() {
    // Round 62: `conn or self._new_conn()` (urllib3's _get_conn) — a
    // local seeded `conn = None` (its Option-ness lives only in
    // optional_names; infer_type resolves the recorded None assignment to
    // PyObject) OR'd with an untyped call. The fold must take the Option
    // arm — Some-wrapping the call — never the loud `||` fallback (which
    // rustc rejects: Option has no bool operator).
    let out = compile(
        "def g():\n    return None\n\n\
         def f(cond) -> object:\n    conn = None\n    if cond:\n        conn = g()\n    return conn or g()\n",
        "connor.py",
    );
    assert!(
        out.contains("is_truthy () { __rython_or } else { Some (g () ?) }")
            || out.contains("is_truthy() { __rython_or } else { Some(g()?) }"),
        "the None-seeded local must fold through the Option arm: {}",
        out
    );
    assert!(
        !out.contains("(conn) || ("),
        "must not fall back to the || approximation: {}",
        out
    );
}

#[test]
fn option_dict_or_empty_dict_folds_the_option_arm() {
    // Round 62: `headers or {}` (urllib3's RequestMethods.__init__) — an
    // Option-typed dict parameter OR'd with an empty-dict literal (which
    // infers Dict(PyObject, PyObject)); the container types unify through
    // the same relation the rest of the codebase uses, so the fold Some-
    // wraps the literal instead of falling to `||`.
    let out = compile(
        "def f(headers: dict | None) -> None:\n    x = headers or {}\n    return None\n",
        "hdrs.py",
    );
    assert!(
        out.contains("is_truthy () { __rython_or } else { Some (PyDict :: from ([])) }")
            || out.contains("is_truthy() { __rython_or } else { Some(PyDict::from([])) }"),
        "the Option-dict or-fold must Some-wrap the empty dict: {}",
        out
    );
    assert!(
        !out.contains("(headers) || ("),
        "must not fall back to the || approximation: {}",
        out
    );
}

#[test]
fn option_receiver_subscript_store_unwraps_and_owns() {
    // Round 63: `headers["k"] = "v"` where headers is `Mapping[str, str]
    // | None` (urllib3's RequestMethods — guaranteed non-None after the
    // `if headers is None:` fill): the subscript STORE unwraps the Option
    // receiver with a loud §12.2 panic (CPython's TypeError on a None
    // receiver), and the String-keyed dict owns both the index literal and
    // the str-literal value (the receiver dict type is read THROUGH the
    // Option). The boxed-value twin (`dict[str, Any] | None` —
    // poolmanager's request_context) wraps the stored member in
    // PyValue::from.
    let out = compile(
        "def f(headers: dict[str, str] | None) -> None:\n    headers[\"k\"] = \"v\"\n    return None\n",
        "optstore.py",
    );
    assert!(
        out.contains("as_mut () . unwrap_or_else")
            || out.contains("as_mut().unwrap_or_else"),
        "the Option receiver must unwrap with the loud panic: {}",
        out
    );
    assert!(
        out.contains("does not support item assignment"),
        "the panic must carry CPython's TypeError message: {}",
        out
    );
    assert!(
        out.contains("(\"k\") . to_string ()") || out.contains("(\"k\").to_string()"),
        "the index literal must be owned: {}",
        out
    );
    assert!(
        out.contains("(\"v\") . to_string ()") || out.contains("(\"v\").to_string()"),
        "the str-literal value must be owned: {}",
        out
    );

    let out2 = compile(
        "def g(ctx: dict[str, object] | None) -> None:\n    ctx[\"blocksize\"] = 10\n    return None\n",
        "optstore2.py",
    );
    assert!(
        out2.contains("PyValue :: from (10)") || out2.contains("PyValue::from(10)"),
        "the member of a boxed-valued dict must wrap: {}",
        out2
    );
    assert!(
        out2.contains("(\"blocksize\") . to_string ()")
            || out2.contains("(\"blocksize\").to_string()"),
        "the boxed-valued dict's index must be owned: {}",
        out2
    );
}

#[test]
fn boxed_subscript_str_method_dispatches_on_the_member() {
    // Round 64: `context["scheme"].lower()` where context is `dict[str,
    // Any]` (urllib3's poolmanager) — the subscript read yields the
    // boxed PyValue, whose str method dispatches on the runtime member
    // (Str -> lowercase; anything else -> CPython's AttributeError
    // panic). The blanket PyStrOps needs AsRef<str>, which PyValue does
    // not satisfy (E0599). The dispatch also fires through an
    // OPTION-wrapped receiver (`dict[str, Any] | None` — the subscript's
    // infer_type sees the value type through the Option).
    let out = compile(
        "def f(ctx: dict[str, object]) -> str:\n    return ctx[\"scheme\"].lower()\n",
        "boxedlower.py",
    );
    assert!(
        out.contains("py_boxed_lower"),
        "the boxed receiver's lower must dispatch at runtime: {}",
        out
    );
    assert!(
        !out.contains("py_index (\"scheme\") ? . lower ()"),
        "must not emit the plain PyStrOps lower: {}",
        out
    );

    let out2 = compile(
        "def g(ctx: dict[str, object] | None) -> str:\n    return ctx[\"scheme\"].lower()\n",
        "boxedlower2.py",
    );
    assert!(
        out2.contains("py_boxed_lower"),
        "the Option-wrapped boxed receiver must dispatch too: {}",
        out2
    );
}

#[test]
fn unbound_builtin_str_method_applies_to_its_argument() {
    // Round 65: `str.title(header)` (urllib3's SKIPPABLE_HEADERS
    // titlecasing) — Python's `str.m(s)` is `s.m()`; the class-as-value
    // model has no `str.title` attribute (E0609/E0599 on the runtime
    // str() fn item), so the call lowers to the bound method on the
    // argument.
    let out = compile(
        "def f(headers: list[str]) -> list[str]:\n    return [str.title(header) for header in headers]\n",
        "strtitle.py",
    );
    assert!(
        out.contains("(header) . title ()") || out.contains("(header).title()"),
        "str.title(x) must lower to x.title(): {}",
        out
    );
    assert!(
        !out.contains("str . title") && !out.contains("str.title"),
        "must not emit the fn-item attribute: {}",
        out
    );

    // `map(str.lower, xs)` — the unbound method as a function argument
    // (urllib3's request): a closure applying the bound method.
    let out2 = compile(
        "def g(headers: dict[str, str]) -> bool:\n    return \"content-type\" in map(str.lower, headers.keys())\n",
        "strlowermap.py",
    );
    assert!(
        out2.contains("| __rython_x | (__rython_x) . lower ()")
            || out2.contains("|__rython_x| (__rython_x).lower()"),
        "map(str.lower, xs) must lower to a closure: {}",
        out2
    );
}

#[test]
fn tuple_literal_iterates_as_an_array_of_owned_strings() {
    // Round 66: `for key in ("headers", "_proxy_headers")` (urllib3's
    // poolmanager) — Python iterates the tuple; rython's tuple value is
    // a Rust tuple, which is not IntoIterator (E0277). An all-constant
    // tuple iterates as an array; STRING literals own themselves so the
    // loop target feeds String-keyed dict calls
    // (`request_context.pop(key, None)`).
    let out = compile(
        "def f(ctx: dict[str, object] | None) -> None:\n\
         \x20   for key in (\"scheme\", \"host\"):\n\
         \x20       ctx.pop(key, None)\n\
         \x20   return None\n",
        "tupleiter.py",
    );
    assert!(
        out.contains("for key in [(\"scheme\") . to_string () , (\"host\") . to_string ()]")
            || out.contains("for key in [(\"scheme\").to_string(), (\"host\").to_string()]"),
        "the tuple iterable must become an array of owned strings: {}",
        out
    );
}

#[test]
fn option_receiver_membership_unwraps_and_owns_the_key() {
    // Round 66: `key in request_context` where request_context is
    // `dict[str, Any] | None` (urllib3's poolmanager) — the membership
    // READ unwraps the Option receiver with a loud §12.2 panic (CPython's
    // TypeError on a None comparator), and a String-keyed dict owns the
    // &str member key (the dict type is read THROUGH the Option).
    let out = compile(
        "def f(ctx: dict[str, object] | None, key: str) -> bool:\n    return key in ctx\n",
        "optcontains.py",
    );
    assert!(
        out.contains("not iterable"),
        "the Option comparator must unwrap with CPython's TypeError: {}",
        out
    );
    assert!(
        out.contains("py_contains") && !out.contains("(ctx) . py_contains"),
        "the unwrapped comparator must run the membership test: {}",
        out
    );
}

#[test]
fn super_method_factory_local_resolves_its_class() {
    // Round 67: `r = super().make()` — an override assigning the BASE's
    // result, then reading a field of it that lives on the result class's
    // OWN base (`r.status` where the result class embeds its base — the
    // field needs the embedded-base chain rewrite). The factory-local
    // receiver resolution used to recognize only a bare `self` callee
    // (`r = self.make()`); the super-callee left the local untyped, the
    // field read emitted the bare name, and the embedded-base field was
    // an E0615 method-not-a-field.
    let resp = parse(
        "class RBase:\n\
         \x20   def __init__(self, status: int) -> None:\n\
         \x20       self.status = status\n\
         class Resp(RBase):\n\
         \x20   pass\n",
        "resp.py",
    )
    .unwrap();
    let base = parse(
        "from .resp import Resp\n\
         class Base:\n\
         \x20   def make(self) -> Resp:\n\
         \x20       return Resp(200)\n\
         class Sub(Base):\n\
         \x20   def make(self) -> Resp:\n\
         \x20       return super().make()\n\
         \x20   def read(self) -> int:\n\
         \x20       r = super().make()\n\
         \x20       return r.status\n",
        "base.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["resp".to_string()], std::rc::Rc::new(resp));
    defs.insert(vec!["base".to_string()], std::rc::Rc::new(base.clone()));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let symbols = base.clone().find_symbols(SymbolTableScopes::new());
    let out = base
        .to_rust(
            CodeGenContext::Module("base".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        out.contains("r . __rython_base . status") || out.contains("r.__rython_base.status"),
        "the super-factory local's embedded-base field read must chain: {}",
        out
    );
}

#[test]
fn direct_imported_factory_call_property_read_resolves() {
    // Round 67: `parse_url(url).netloc` — a PROPERTY of the return class
    // of an IMPORTED factory, read directly on the call (not through a
    // local): the factory-call receiver's class was unresolved, the read
    // emitted the bare name, and the property was E0615
    // method-not-a-field. The factory resolution now also covers the call
    // in place; the property read routes to the getter call.
    let url = parse(
        "class Url:\n\
         \x20   def __init__(self, netloc: str | None) -> None:\n\
         \x20       self._netloc = netloc\n\
         \x20   @property\n\
         \x20   def netloc(self) -> str | None:\n\
         \x20       return self._netloc\n\
         def parse_url(url: str) -> Url:\n\
         \x20   return Url(url)\n",
        "url.py",
    )
    .unwrap();
    let main = parse(
        "from .url import parse_url\n\
         def f(url: str) -> str | None:\n\
         \x20   return parse_url(url).netloc\n",
        "main.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["url".to_string()], std::rc::Rc::new(url));
    defs.insert(vec!["main".to_string()], std::rc::Rc::new(main.clone()));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let symbols = main.clone().find_symbols(SymbolTableScopes::new());
    let out = main
        .to_rust(
            CodeGenContext::Module("main".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        out.contains("parse_url (url) ? . netloc () ?")
            || out.contains("parse_url(url)?.netloc()?"),
        "the factory-call property read must route to the getter: {}",
        out
    );
}

#[test]
fn construction_call_property_read_resolves_imported_classes_too() {
    // Round 68: `Url(scheme=..., path=...).url` — a PROPERTY of a class
    // CONSTRUCTION read in place. The direct-call receiver resolution
    // covered imported-factory calls; a construction's func is the class
    // NAME, local or imported — both now resolve, so the property read
    // routes to the getter instead of E0615 method-not-a-field.
    let url = parse(
        "class Url:\n\
         \x20   def __init__(self, scheme: str | None) -> None:\n\
         \x20       self._scheme = scheme\n\
         \x20   @property\n\
         \x20   def url(self) -> str:\n\
         \x20       return self._scheme or \"\"\n",
        "url.py",
    )
    .unwrap();
    let main = parse(
        "from .url import Url\n\
         def f(scheme: str | None) -> str:\n\
         \x20   return Url(scheme).url\n",
        "main.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["url".to_string()], std::rc::Rc::new(url));
    defs.insert(vec!["main".to_string()], std::rc::Rc::new(main.clone()));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let symbols = main.clone().find_symbols(SymbolTableScopes::new());
    let out = main
        .to_rust(
            CodeGenContext::Module("main".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        out.contains(". url () ?") || out.contains(".url()?"),
        "the construction property read must route to the getter: {}",
        out
    );
}

#[test]
fn inherited_property_read_routes_to_the_getter() {
    // Round 69: `self.host` where `host` is a @property of a BASE class
    // (read from a derived method — the property check used to look at
    // the derived class's own methods only, so the read emitted the bare
    // name and the getter METHOD was an E0615 method-not-a-field). The
    // property check now walks the base chain.
    let out = compile(
        "class Base:\n\
         \x20   def __init__(self, host: str | None) -> None:\n\
         \x20       self._host = host\n\
         \x20   @property\n\
         \x20   def host(self) -> str | None:\n\
         \x20       return self._host\n\
         class Derived(Base):\n\
         \x20   def f(self) -> str | None:\n\
         \x20       return self.host\n",
        "inhprop.py",
    );
    assert!(
        out.contains("(self . host () ?)") || out.contains("(self.host()?)"),
        "the inherited property read must route to the getter: {}",
        out
    );
}

#[test]
fn annotated_local_widened_by_a_later_option_store() {
    // Round 70: `server_hostname: str = self.host()` then `server_hostname
    // = self._tunnel_host` (a `str | None` field of a base class, stored
    // inside an if): the Python local becomes None-able — the annotation
    // was a hint, not a constraint. The class-aware seeding now recurses
    // into nested bodies and walks the base chain for the field, so the
    // local widens to Option; the plain String store wraps in Some, and an
    // Option-slot ARGUMENT passes the local through unwrapped (never
    // Some-wrapped again).
    let out = compile(
        "class Base:\n\
         \x20   def __init__(self) -> None:\n\
         \x20       self._tunnel_host: str | None = None\n\
         \x20   def host(self) -> str:\n\
         \x20       return \"h\"\n\
         class Derived(Base):\n\
         \x20   def consume(self, h: str | None) -> None:\n\
         \x20       pass\n\
         \x20   def f(self) -> None:\n\
         \x20       server_hostname: str = self.host()\n\
         \x20       if self._tunnel_host is not None:\n\
         \x20           server_hostname = self._tunnel_host\n\
         \x20       self.consume(server_hostname)\n\
         \x20       return None\n",
        "widened.py",
    );
    assert!(
        out.contains("Some ({ (self) . host () ? })") || out.contains("Some({(self).host()?})"),
        "the plain String store must wrap into the widened local: {}",
        out
    );
    assert!(
        out.contains("consume (server_hostname)") || out.contains("consume(server_hostname)"),
        "the widened local must pass an Option slot unwrapped: {}",
        out
    );
    assert!(
        !out.contains("Some (server_hostname)") && !out.contains("Some(server_hostname)"),
        "the argument must not double-wrap: {}",
        out
    );
}

#[test]
fn compiled_regex_statics_type_and_dispatch() {
    // Round 72: `_TARGET_RE = re.compile(...)` module statics are typed
    // as the runtime's compiled Regex (not boxed in a PyValue that has no
    // regex methods), and `_RE.match(x)` / `_RE.search(x)` /
    // `_RE.fullmatch(x)` dispatch through the runtime's PyRegexOps
    // (anchored-at-start / anywhere / whole-text).
    let out = compile(
        "import re\n\
         _TARGET_RE = re.compile(\"a+\")\n\
         def f(target: str) -> bool:\n\
         \x20   return _TARGET_RE.match(target) is not None\n\
         def g(target: str) -> bool:\n\
         \x20   return _TARGET_RE.fullmatch(target) is not None\n",
        "regex.py",
    );
    assert!(
        out.contains("LazyLock < stdpython :: stdlib :: re :: Regex >")
            || out.contains("LazyLock<stdpython::stdlib::re::Regex>"),
        "the re.compile static must be typed as the runtime Regex: {}",
        out
    );
    assert!(
        out.contains("py_match") && !out.contains("r#match ("),
        "the .match() call must dispatch through py_match: {}",
        out
    );
    assert!(
        out.contains("py_fullmatch"),
        "the .fullmatch() call must dispatch through py_fullmatch: {}",
        out
    );
}

#[test]
fn field_walk_follows_imported_bases() {
    // Round 71: `self.headers` where `headers` is a field stored in an
    // IMPORTED base class (`PoolManager(RequestMethods)` — the struct
    // embeds the base): the base chain used a symbol-table-only walk that
    // could not follow imported bases, so the chain stopped at the
    // derived class and the field-walk missed the ancestor's field —
    // the generic-trait read emitted the bare name (E0615
    // method-not-a-field). The walk now resolves imported bases through
    // the module definitions.
    let base = parse(
        "class Base:\n\
         \x20   def __init__(self) -> None:\n\
         \x20       self.headers: dict[str, str] | None = None\n",
        "base.py",
    )
    .unwrap();
    let main = parse(
        "from .base import Base\n\
         class Derived(Base):\n\
         \x20   def f(self) -> dict[str, str] | None:\n\
         \x20       return self.headers\n",
        "main.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["base".to_string()], std::rc::Rc::new(base));
    defs.insert(vec!["main".to_string()], std::rc::Rc::new(main.clone()));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let symbols = main.clone().find_symbols(SymbolTableScopes::new());
    let out = main
        .to_rust(
            CodeGenContext::Module("main".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        out.contains("__rython_base . headers") || out.contains("__rython_base.headers"),
        "the imported-base field read must chain through the embedded base: {}",
        out
    );
}

#[test]
fn local_from_another_objects_option_field_is_option() {
    // Round 68: `destination_scheme = parsed_url.scheme` (a `str | None`
    // field of a factory-local object), then passed to a `str | None`
    // parameter — the local's Option-ness was not seeded (only
    // `self.<field>` reads were), so the argument adaptation wrapped the
    // already-Option local in `Some(...)` — Option<Option<String>> (E0308).
    // The class-aware seeding now also types locals from an option field
    // of ANY object whose class resolves.
    let url = parse(
        "class Url:\n\
         \x20   def __init__(self, scheme: str | None) -> None:\n\
         \x20       self.scheme = scheme\n\
         def parse_url(url: str) -> Url:\n\
         \x20   return Url(url)\n",
        "url.py",
    )
    .unwrap();
    let main = parse(
        "from .url import parse_url\n\
         class Client:\n\
         \x20   def consume(self, scheme: str | None) -> None:\n\
         \x20       pass\n\
         \x20   def f(self, url: str) -> None:\n\
         \x20       u = parse_url(url)\n\
         \x20       destination_scheme = u.scheme\n\
         \x20       self.consume(destination_scheme)\n\
         \x20       return None\n",
        "main.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["url".to_string()], std::rc::Rc::new(url));
    defs.insert(vec!["main".to_string()], std::rc::Rc::new(main.clone()));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let symbols = main.clone().find_symbols(SymbolTableScopes::new());
    let out = main
        .to_rust(
            CodeGenContext::Module("main".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    assert!(
        out.contains("(self) . consume (destination_scheme)")
            || out.contains("(self).consume(destination_scheme)"),
        "the option-typed local must pass through unwrapped, never Some-wrapped: {}",
        out
    );
    assert!(
        !out.contains("Some (destination_scheme)") && !out.contains("Some(destination_scheme)"),
        "the argument must not double-wrap: {}",
        out
    );
}

#[test]
fn option_lhs_compare_unwraps_with_equality_semantics() {
    // Round 43: `amt != 0` where amt is `int | None` (urllib3's
    // _read_next_chunk) — the Option LHS unwraps the inner for the
    // comparison; a None LHS answers Python's EQUALITY semantics
    // (`None == x` is False, `None != x` is True), while an ORDERED
    // compare on None is CPython's TypeError — a loud §12.2 panic with
    // the exact message.
    let out = compile(
        "def ne(amt: int | None) -> bool:\n    return amt != 0\n\n\
         def eq(amt: int | None) -> bool:\n    return amt == 0\n\n\
         def lt(amt: int | None) -> bool:\n    return amt < 5\n",
        "optcmp.py",
    );
    assert!(
        out.contains("Some (__rython_v) => (__rython_v) . py_ne (& (0))")
            || out.contains("Some(__rython_v) => (__rython_v).py_ne(&(0))"),
        "the inner value must compare: {}",
        out
    );
    assert!(
        out.contains("None => true") && out.contains("None => false"),
        "a None LHS must answer Python's equality semantics: {}",
        out
    );
    assert!(
        out.contains("'<' not supported between instances of 'NoneType' and 'int'"),
        "an ordered compare on None must panic with CPython's message: {}",
        out
    );
}

#[test]
fn option_both_sides_compare_unwrap_both() {
    // Round 43: `amt < self.chunk_left` where BOTH are `int | None`
    // (urllib3's _handle_chunk, guarded `is not None` on both): each
    // side unwraps with the loud panic.
    let out = compile(
        "class R:\n\
         \x20   def __init__(self):\n\
         \x20       self.chunk_left: int | None = None\n\
         \x20   def f(self, amt: int | None) -> bool:\n\
         \x20       if amt is not None and self.chunk_left is not None:\n\
         \x20           return amt < self.chunk_left\n\
         \x20       return False\n",
        "optcmp2.py",
    );
    assert!(
        out.contains("match (self . chunk_left) . clone ()")
            || out.contains("match (self.chunk_left()).clone()")
            || out.contains("match (self.chunk_left).clone()"),
        "the Option comparator must be unwrapped too: {}",
        out
    );
}

#[test]
fn optional_str_literal_arguments_own_themselves() {
    let out = compile(
        "def pick(ca: str | None) -> str | None:\n    return ca\n\ndef use() -> str | None:\n    return pick(\"x\")\n",
        "optstrlit.py",
    );
    assert!(
        out.contains("Some ((\"x\") . to_string ())"),
        "a str literal into an Option<String> slot must be owned: {}",
        out
    );
}

#[test]
fn a_none_stored_local_into_a_boxed_class_param_unwraps() {
    // Round 84: `conn = None` then `conn = self._get_conn(...)`, passed
    // to a method whose parameter is annotated with a TYPE_CHECKING-imported
    // Protocol stub (`conn: BaseHTTPConnection` — urllib3's
    // connectionpool). The stub resolves to the boxed PyValue (the same
    // authority the parameter's Rust type used), so the Option<PyValue>
    // binding must unwrap to the boxed value with Python's None passing
    // through as PyValue::None_ — never a raw Option into a PyValue slot
    // (the `PyValue | Option<PyValue>` family, ×18 fixed).
    let out = compile(
        concat!(
            "from http.client import HTTPConnection as _HTTPConnection\n",
            "\n",
            "class Conn:\n",
            "    def _prepare_proxy(self, conn: _HTTPConnection) -> None:\n",
            "        pass\n",
            "\n",
            "    def _make(self) -> _HTTPConnection:\n",
            "        return self._new_conn()\n",
            "\n",
            "    def _new_conn(self) -> _HTTPConnection:\n",
            "        raise NotImplementedError()\n",
            "\n",
            "    def f(self) -> None:\n",
            "        conn = None\n",
            "        conn = self._make()\n",
            "        self._prepare_proxy(conn)\n",
        ),
        "boxedparam.py",
    );
    assert!(
        out.contains("(conn) . unwrap_or (stdpython :: PyValue :: None_)"),
        "the None-stored local must unwrap to the boxed value with Python's None passing through: {}",
        out
    );
}

#[test]
fn a_caller_of_an_inferred_option_fn_narrows_and_unwraps() {
    // Round 85 (the return-type directive): `pick(flag: bool)` returning
    // `"yes"` | None INFERS `Option<String>` (no annotation — the body's
    // two return types are exactly T and None). The caller's store of the
    // result must learn the Option (call_return_typeinfo consults the
    // inferred return), so `if v is None:` narrows and the read unwraps —
    // the caller decides what to do with the None. A caller that returns
    // the Option into a concrete slot unhandled keeps the loud mismatch
    // (Python's likely-bug pattern — "throw an error rather than mangle").
    let out = compile(
        concat!(
            "def pick(flag: bool):\n",
            "    if flag:\n",
            "        return \"yes\"\n",
            "    return None\n",
            "\n",
            "def use(flag: bool) -> str:\n",
            "    v = pick(flag)\n",
            "    if v is None:\n",
            "        return \"none\"\n",
            "    return v\n",
        ),
        "optcaller.py",
    );
    assert!(
        out.contains("Result < Option < String > , PyException >"),
        "the unannotated T | None function must return Option<String>: {}",
        out
    );
    assert!(
        out.contains("(v) . clone () . unwrap ()"),
        "the narrowed read must unwrap the Option: {}",
        out
    );
    assert!(
        out.contains("return Ok ((v) . clone () . unwrap ())") || out.contains("return Ok ((v).clone().unwrap())"),
        "the narrowed return must unwrap: {}",
        out
    );
}

#[test]
fn an_option_callee_result_into_a_boxed_union_param_coerces() {
    // Round 86: `resolve_default_timeout(timeout)` returns `float | None`
    // and the result feeds a `_TYPE_TIMEOUT` parameter — a module-level
    // alias (`Union[float, str, None]`) that lowers to the boxed PyValue.
    // The syntax-only annotation mapping cannot see the alias, so the
    // GENERAL call-argument path (a plain `g(...)` call, not the
    // mapped-fill) must fall back to the symbols-aware authority — an
    // OPTION-typed argument coerces `Option<f64> → PyValue` via the
    // Some/None match, Python's None passing through as the boxed None.
    let out = compile(
        concat!(
            "from typing import Union\n",
            "\n",
            "_TYPE_TIMEOUT = Union[float, str, None]\n",
            "\n",
            "def resolve_default_timeout(timeout: _TYPE_TIMEOUT) -> float | None:\n",
            "    return None\n",
            "\n",
            "def g(timeout: _TYPE_TIMEOUT) -> None:\n",
            "    pass\n",
            "\n",
            "def f(timeout: _TYPE_TIMEOUT) -> None:\n",
            "    g(resolve_default_timeout(timeout))\n",
        ),
        "optunionarg.py",
    );
    assert!(
        out.contains("match (resolve_default_timeout (timeout) ?) {")
            || out.contains("match(resolve_default_timeout(timeout)?){"),
        "the Option-typed callee result must coerce into the boxed param: {}",
        out
    );
    assert!(
        out.contains("Some (__rython_v) => PyValue :: from ((__rython_v))")
            || out.contains("Some(__rython_v)=>PyValue::from((__rython_v))"),
        "the Some arm must box the inner: {}",
        out
    );
    assert!(
        out.contains("None => stdpython :: PyValue :: None_"),
        "the None arm must be the boxed None: {}",
        out
    );
}

#[test]
fn a_property_read_local_on_a_factory_local_coerces_into_a_boxed_union_param() {
    // Round 87: `timeout_obj = self._get_timeout()` (a `-> Timeout`
    // self-method call) seeds the local with the class; the PROPERTY read
    // `read_timeout = timeout_obj.read_timeout` (a `-> float | None`
    // accessor) then types the local as Option<f64> — and because the
    // read yields the Option (the read-flavored receiver resolution sees
    // the factory local where the conservative receiver_class returned
    // None on the attribute-callee Assign shape), the store passes it
    // through instead of double-wrapping `Some(timeout_obj.read_timeout()?)`.
    // The Option<f64> local into a `_TYPE_TIMEOUT` (boxed PyValue) param
    // then coerces via the Some/None match — Python's None passes through
    // as the boxed None.
    let out = compile(
        concat!(
            "from typing import Union\n",
            "\n",
            "_TYPE_TIMEOUT = Union[float, str, None]\n",
            "\n",
            "class Timeout:\n",
            "    @property\n",
            "    def read_timeout(self) -> float | None:\n",
            "        return None\n",
            "\n",
            "class Conn:\n",
            "    def _get_timeout(self) -> Timeout:\n",
            "        return Timeout()\n",
            "    def _raise(self, e: int, timeout_value: _TYPE_TIMEOUT) -> None:\n",
            "        pass\n",
            "    def f(self) -> None:\n",
            "        timeout_obj = self._get_timeout()\n",
            "        read_timeout = timeout_obj.read_timeout\n",
            "        self._raise(1, read_timeout)\n",
        ),
        "propfactory.py",
    );
    assert!(
        out.contains("read_timeout = timeout_obj . read_timeout () ?")
            || out.contains("read_timeout=timeout_obj.read_timeout()?"),
        "the property read must pass the Option through, not double-wrap: {}",
        out
    );
    assert!(
        !out.contains("read_timeout = Some (timeout_obj")
            && !out.contains("read_timeout=Some(timeout_obj"),
        "the store must NOT Some-wrap an already-Option property read: {}",
        out
    );
    assert!(
        out.contains("match (read_timeout) { Some (__rython_v) => PyValue :: from ((__rython_v)) , None => stdpython :: PyValue :: None_ , }")
            || out.contains("match(read_timeout){Some(__rython_v)=>PyValue::from((__rython_v)),None=>stdpython::PyValue::None_,}"),
        "the Option<f64> local must coerce into the boxed param: {}",
        out
    );
}

#[test]
fn a_float_typed_option_compares_with_an_int_literal_as_a_float() {
    // Round 87: `read_timeout == 0` where read_timeout is a `float | None`
    // property read — the Option-match comparison lowers the inner f64
    // against the integer literal, and Rust std has no int/float
    // cross-PartialEq. Python promotes the int operand to float, so the
    // literal renders `(0) as f64` (the same `as f64` the coercion
    // machinery accepts for numeric contexts).
    let out = compile(
        concat!(
            "class Timeout:\n",
            "    @property\n",
            "    def read_timeout(self) -> float | None:\n",
            "        return 0.5\n",
            "\n",
            "class Conn:\n",
            "    def _get_timeout(self) -> Timeout:\n",
            "        return Timeout()\n",
            "    def f(self) -> None:\n",
            "        timeout_obj = self._get_timeout()\n",
            "        read_timeout = timeout_obj.read_timeout\n",
            "        if read_timeout == 0:\n",
            "            print(\"zero\")\n",
        ),
        "floatcmp.py",
    );
    assert!(
        out.contains("py_eq (& ((0) as f64))") || out.contains("py_eq(&((0)as f64))"),
        "the integer literal must promote to the float operand: {}",
        out
    );
}

#[test]
fn a_dict_literal_string_value_is_owned_like_its_key() {
    // Round 87: `headers_ = {"Accept": "*/*"}` in a `-> Mapping[str, str]`
    // function — string-literal VALUES alone inferred &'static str, so the
    // literal rendered IndexMap<String, &str> and could never match the
    // IndexMap<String, String> return. The value normalizes to owned
    // String exactly like the key.
    let out = compile(
        concat!(
            "from typing import Mapping\n",
            "\n",
            "def headers() -> Mapping[str, str]:\n",
            "    headers_ = {\"Accept\": \"*/*\"}\n",
            "    return headers_\n",
        ),
        "strdict.py",
    );
    assert!(
        out.contains("(\"Accept\") . to_string () , (\"*/*\") . to_string ()")
            || out.contains("(\"Accept\").to_string(),(\"*/*\").to_string()"),
        "the dict literal must own its string value like its key: {}",
        out
    );
}

#[test]
fn an_annotated_option_return_keeps_the_option_against_a_plain_literal_body() {
    // Round 87: `-> float | None` with a `return 0.5` body — the body's
    // inferred `f64` must NOT shrink the annotated `Option<f64>` (the
    // return-site Some-wrap and the fn_return_is_option flag already
    // agree on the Option; the annotation is the contract).
    let out = compile(
        concat!(
            "class Timeout:\n",
            "    @property\n",
            "    def read_timeout(self) -> float | None:\n",
            "        return 0.5\n",
        ),
        "annopt.py",
    );
    assert!(
        out.contains("-> Result < Option < f64 > , PyException >")
            || out.contains("->Result<Option<f64>,PyException>"),
        "the annotated Option return must win over the body's inferred float: {}",
        out
    );
}

#[test]
fn a_reused_class_local_clones_via_the_qualified_std_clone() {
    // Round 88: a REUSED class-typed local (`timeout_obj` from a
    // `-> Timeout` factory, passed to two calls) must clone with the
    // QUALIFIED std Clone — the bare `(x).clone()` would resolve to the
    // class's OWN `clone` method when it defines one (urllib3's Timeout
    // does), a REAL semantic call where Python just re-reads the
    // variable (round 88).
    let out = compile(
        concat!(
            "class Timeout:\n",
            "    def clone(self) -> Timeout:\n",
            "        return Timeout()\n",
            "\n",
            "class Conn:\n",
            "    def _get_timeout(self) -> Timeout:\n",
            "        return Timeout()\n",
            "    def _consume(self, t: Timeout) -> None:\n",
            "        pass\n",
            "    def f(self) -> None:\n",
            "        timeout_obj = self._get_timeout()\n",
            "        self._consume(timeout_obj)\n",
            "        self._consume(timeout_obj)\n",
        ),
        "qualclone.py",
    );
    assert!(
        out.contains("Clone :: clone (& (timeout_obj))")
            || out.contains("Clone::clone(&(timeout_obj))"),
        "the reuse-clone must be the qualified std Clone, never the user clone method: {}",
        out
    );
    assert!(
        !out.contains("timeout_obj . clone ()")
            && !out.contains("timeout_obj.clone()"),
        "the user's clone method must not be called for an ownership clone: {}",
        out
    );
}

#[test]
fn a_dict_update_with_an_option_dict_argument_unwraps_loudly() {
    // Round 88: `headers.update(self.proxy_headers)` where proxy_headers
    // is a `Mapping[str, str] | None` field — the stdpython PyDictOps
    // update takes the other dict BY VALUE, and Python's update(None) is
    // a TypeError, so the Option argument coerces via the round-83 match
    // with the loud unhandled-None panic.
    let out = compile(
        concat!(
            "from typing import Mapping\n",
            "\n",
            "class P:\n",
            "    def __init__(self) -> None:\n",
            "        self.proxy_headers: Mapping[str, str] | None = {\"Accept\": \"*/*\"}\n",
            "    def urlopen(self, headers: Mapping[str, str] | None) -> None:\n",
            "        headers.update(self.proxy_headers)\n",
        ),
        "dictupdate.py",
    );
    assert!(
        out.contains("match ((self . proxy_headers) . clone ()) { Some (__rython_v) => __rython_v")
            || out.contains("match((self.proxy_headers).clone()){Some(__rython_v)=>__rython_v"),
        "the Option dict argument must unwrap via the Some/None match: {}",
        out
    );
    assert!(
        out.contains("an optional value was None where a concrete value was required"),
        "the None case must be the loud unhandled-Option panic: {}",
        out
    );
}

#[test]
fn an_option_field_read_passes_through_an_option_slot_without_double_wrapping() {
    // Round 89: `super().from_host(self.proxy.host, ...)` where Url's
    // `host`/`port`/`scheme` fields are `T | None` — the field READ
    // yields the Option (the accessor returns it), so an Option-slot
    // argument must pass it through; `Some(self.proxy().host)` would
    // double it into `Option<Option<String>>`.
    let out = compile(
        concat!(
            "class Url:\n",
            "    def __init__(self) -> None:\n",
            "        self.host: str | None = None\n",
            "        self.port: int | None = None\n",
            "        self.scheme: str | None = None\n",
            "\n",
            "class Base:\n",
            "    def from_host(self, host: str | None, port: int | None = None, scheme: str | None = None) -> None:\n",
            "        pass\n",
            "\n",
            "class PM(Base):\n",
            "    def __init__(self) -> None:\n",
            "        self.proxy: Url = Url()\n",
            "    def connection_from_host(self, host: str | None, port: int | None = None, scheme: str | None = None) -> None:\n",
            "        return super().from_host(self.proxy.host, self.proxy.port, self.proxy.scheme)\n",
        ),
        "fieldopt.py",
    );
    let i = out.find("connection_from_host").unwrap_or(0);
    let body = &out[i..];
    assert!(
        body.contains("self . proxy () . host") || body.contains("self.proxy().host"),
        "the Option field read must pass through unwrapped: {}",
        out
    );
    assert!(
        !body.contains("Some (self . proxy () . host)")
            && !body.contains("Some(self.proxy().host)"),
        "the Option field read must NOT double-wrap: {}",
        out
    );
}

#[test]
fn a_self_option_field_compares_via_the_option_match() {
    // Round 89: `self.length_remaining != 0` where the field is
    // `int | None` — infer_type cannot see through self-fields, so the
    // compare's Option-match trigger consults the FIELD TABLE for a
    // `self.<field>` accessor: the Option unwraps to the inner i64 and
    // the equality answers Python's None semantics (`None != 0` is True).
    let out = compile(
        concat!(
            "class Conn:\n",
            "    def __init__(self) -> None:\n",
            "        self.length_remaining: int | None = None\n",
            "    def f(self) -> bool:\n",
            "        return self.length_remaining != 0\n",
        ),
        "selfoptcmp.py",
    );
    assert!(
        out.contains("match (self . length_remaining) . clone () { Some (__rython_v) => (__rython_v) . py_ne (& (0)) , None => true")
            || out.contains("match(self.length_remaining).clone(){Some(__rython_v)=>(__rython_v).py_ne(&(0)),None=>true"),
        "the self-field Option must unwrap to the inner comparison: {}",
        out
    );
}

#[test]
fn a_factory_local_stored_into_a_field_resolves_the_receiver_class() {
    // Round 90: `proxy = parse_url(...); self.proxy = proxy` (urllib3's
    // ProxyManager.__init__) — field_class's param-only Name arm could not
    // name the field's class, so `self.proxy.host` never resolved its
    // receiver and the Option field reads double-wrapped
    // `Some(self.proxy().host)`. The arm now resolves a LOCAL store
    // through the factory call's return annotation.
    let out = compile(
        concat!(
            "class Url:\n",
            "    def __init__(self) -> None:\n",
            "        self.host: str | None = None\n",
            "        self.port: int | None = None\n",
            "        self.scheme: str | None = None\n",
            "\n",
            "def parse_url(u: str) -> Url:\n",
            "    return Url()\n",
            "\n",
            "class Base:\n",
            "    def from_host(self, host: str | None, port: int | None = None, scheme: str | None = None) -> None:\n",
            "        pass\n",
            "\n",
            "class PM(Base):\n",
            "    def __init__(self) -> None:\n",
            "        proxy = parse_url(\"x\")\n",
            "        self.proxy = proxy\n",
            "    def connection_from_host(self, host: str | None, port: int | None = None, scheme: str | None = None) -> None:\n",
            "        return super().from_host(self.proxy.host, self.proxy.port, self.proxy.scheme)\n",
        ),
        "fieldloc.py",
    );
    let i = out.find("connection_from_host").unwrap_or(0);
    let body = &out[i..];
    assert!(
        body.contains("self . proxy () . host") || body.contains("self.proxy().host"),
        "the factory-local field read must pass through unwrapped: {}",
        out
    );
    assert!(
        !body.contains("Some (self . proxy () . host)")
            && !body.contains("Some(self.proxy().host)"),
        "the factory-local field read must NOT double-wrap: {}",
        out
    );
}

#[test]
fn a_base_chain_option_field_compares_via_the_option_match() {
    // Round 91: `self.__rython_base._tunnel_scheme == "https"` — a
    // BASE-class `str | None` field read through the embedded base
    // struct — the compare's Option-match trigger only handled the
    // bare `self.<field>` shape, so the base-chain read compared the
    // raw Option (`py_eq(&("https"))` on Option<String> — E0308). The
    // trigger now walks a `self.<chain>.<field>` receiver chain and
    // looks the field up in every class of the base chain.
    let out = compile(
        concat!(
            "class Base:\n",
            "    def __init__(self) -> None:\n",
            "        self._tunnel_scheme: str | None = None\n",
            "\n",
            "class Conn(Base):\n",
            "    def f(self) -> bool:\n",
            "        return self._tunnel_scheme == \"https\"\n",
        ),
        "baseoptcmp.py",
    );
    assert!(
        out.contains("match (") && out.contains(") . clone () { Some (__rython_v) => (__rython_v) . py_eq (& (\"https\")) , None => false")
            || out.contains("match(") && out.contains(").clone(){Some(__rython_v)=>(__rython_v).py_eq(&(\"https\")),None=>false"),
        "the base-chain Option field must unwrap to the inner comparison: {}",
        out
    );
}

#[test]
fn a_plain_lhs_with_an_option_comparator_unwraps_the_comparator() {
    // Round 92: `total < amt` where total is a plain i64 and amt is
    // `int | None` (urllib3's `len(self._decoded_buffer) < amt` in
    // _read) — the Option-comparator unwrap lived ONLY inside the
    // LHS-Option branch, so a plain LHS compared the raw Option
    // (`py_lt(&(amt))` — E0277 on the PyLt<Option<i64>> bound). The
    // unwrap now applies to the py_* six ops regardless of the LHS:
    // the inner compares, and None is CPython's TypeError — the loud
    // §12.2 panic naming the LHS type.
    let out = compile(
        concat!(
            "class R:\n",
            "    def read(self, amt: int | None = None) -> None:\n",
            "        total = 10\n",
            "        while total < amt:\n",
            "            total += 1\n",
        ),
        "plainoptcmp.py",
    );
    assert!(
        out.contains("py_lt (& (match (amt) . clone () { Some (__rython_r) => __rython_r , None => panic")
            || out.contains("py_lt(&(match(amt).clone(){Some(__rython_r)=>__rython_r,None=>panic"),
        "the Option comparator must unwrap to the inner comparison: {}",
        out
    );
    assert!(
        out.contains("not supported between instances of 'int' and 'NoneType'"),
        "the None case must be CPython's ordered-compare TypeError text: {}",
        out
    );
}

#[test]
fn a_type_alias_annotated_param_stores_into_its_boxed_local() {
    // Round 93: `value: _TYPE_FIELD_VALUE` where
    // `_TYPE_FIELD_VALUE = Union[str, bytes]` — the parameter's recorded
    // type came from the bare-name "class" fallback (Class("_TYPE_FIELD_VALUE")),
    // disagreeing with the parameter's actual boxed PyValue Rust type, so
    // a store into the local (`value = "%s*=%s" % (name, value)`) went in
    // raw. The parameter annotation now resolves the alias FIRST, so the
    // boxed local wraps its stores.
    let out = compile(
        concat!(
            "from typing import Union\n",
            "\n",
            "_TYPE_FIELD_VALUE = Union[str, bytes]\n",
            "\n",
            "def format_header_param_rfc2231(name: str, value: _TYPE_FIELD_VALUE) -> str:\n",
            "    value = \"%s*=%s\" % (name, value)\n",
            "    return value\n",
        ),
        "aliasparam.py",
    );
    assert!(
        out.contains("value = PyValue :: from (py_mod") || out.contains("value=PyValue::from(py_mod"),
        "the boxed-alias local must wrap its stores: {}",
        out
    );
}

#[test]
fn a_module_qualified_typing_cast_is_a_runtime_identity() {
    // Round 94: `typing.cast(ProxyConfig, self.proxy_config)` — the
    // MODULE-QUALIFIED cast (urllib3's _connect_tls_proxy) previously
    // fell to the external-module drop (`proxy_config = PyValue::None_`),
    // breaking every field read on the local (E0609). The qualified form
    // lowers to its VALUE argument, exactly like the imported `cast` name.
    let out = compile(
        concat!(
            "from typing import cast\n",
            "\n",
            "class ProxyConfig:\n",
            "    def __init__(self) -> None:\n",
            "        self.ssl_context: str | None = None\n",
            "\n",
            "class R:\n",
            "    def __init__(self) -> None:\n",
            "        self.proxy_config: ProxyConfig = ProxyConfig()\n",
            "    def f(self) -> str | None:\n",
            "        proxy_config = cast(ProxyConfig, self.proxy_config)\n",
            "        return proxy_config.ssl_context\n",
        ),
        "typingcast.py",
    );
    let i = out.find("fn f").unwrap_or(0);
    let body = &out[i..];
    assert!(
        body.contains("proxy_config = self . proxy_config") || body.contains("proxy_config=self.proxy_config"),
        "the cast must pass the value through unchanged: {}",
        out
    );
    assert!(
        !body.contains("proxy_config = stdpython :: PyValue :: None_")
            && !body.contains("proxy_config=stdpython::PyValue::None_"),
        "the cast must NOT drop to the boxed None: {}",
        out
    );
}
#[test]
fn a_cast_assigned_option_field_local_unwraps_and_clones_on_read() {
    // Round 95: `proxy_config = cast(ProxyConfig, self.proxy_config)`
    // where the field is `ProxyConfig | None` — the cast-assigned local
    // must seed as the field's real Option type (the walk looks through
    // the identity cast), the Option-slot store must pass the value
    // through (the cast yields what its value yields) and clone it out
    // of `&self`, and field reads on the unwrapped local resolve the
    // inner class's fields.
    let out = compile(
        concat!(
            "from typing import cast\n",
            "\n",
            "class ProxyConfig:\n",
            "    def __init__(self) -> None:\n",
            "        self.ssl_context: str | None = None\n",
            "\n",
            "class R:\n",
            "    def __init__(self) -> None:\n",
            "        self.proxy_config: ProxyConfig | None = None\n",
            "    def f(self, other: str | None) -> str | None:\n",
            "        proxy_config = cast(ProxyConfig, self.proxy_config)\n",
            "        ssl_context = proxy_config.ssl_context\n",
            "        return other if ssl_context is None else ssl_context\n",
        ),
        "castoptfield.py",
    );
    let i = out.find("fn f").unwrap_or(0);
    let body = &out[i..];
    assert!(
        body.contains("proxy_config = (self . proxy_config) . clone ()")
            || body.contains("proxy_config=(self.proxy_config).clone()"),
        "the cast-assigned Option field must clone out of the receiver: {}",
        out
    );
    assert!(
        body.contains("unwrap_or_else (|| { panic ! (\"AttributeError: 'NoneType' object has no attribute '{}'\"")
            || body.contains("unwrap_or_else(||{panic!(\"AttributeError: 'NoneType' object has no attribute '{}'\""),
        "the read of the Option local must unwrap loudly: {}",
        out
    );
}

#[test]
fn a_boxed_static_promoted_from_a_scalar_initializer_wraps() {
    // Round 96: `_FAILEDTELL: Final[_TYPE_FAILEDTELL] =
    // _TYPE_FAILEDTELL.token` — an Enum sentinel member (an i64 const)
    // promoted to a boxed LazyLock static — the inferred-type promotion
    // path rendered the initializer RAW against LazyLock<PyValue>
    // (E0308). A boxed-typed static now wraps its initializer in
    // PyValue::from, matching the unknown-type path.
    let out = compile(
        concat!(
            "from enum import Enum\n",
            "from typing import Final\n",
            "\n",
            "class _TYPE_FAILEDTELL(Enum):\n",
            "    token = 0\n",
            "\n",
            "def use() -> int:\n",
            "    return 1 if _FAILEDTELL is _TYPE_FAILEDTELL.token else 0\n",
            "\n",
            "_FAILEDTELL: Final[_TYPE_FAILEDTELL] = _TYPE_FAILEDTELL.token\n",
        ),
        "enumstatic.py",
    );
    assert!(
        out.contains("LazyLock < stdpython :: PyValue > = std :: sync :: LazyLock :: new (|| stdpython :: PyValue :: from")
            || out.contains("LazyLock<stdpython::PyValue>=std::sync::LazyLock::new(||stdpython::PyValue::from"),
        "the boxed static must wrap its scalar initializer: {}",
        out
    );
}

#[test]
fn a_comprehension_filter_uses_the_if_statement_truthiness_authority() {
    // Issue #137, Directive 5: a comprehension `if` filter lowered its
    // condition RAW (`if !(w.strip())` — the unary `!` applied to a
    // String, E0600 in the idiom corpus's `[w.strip() for w in ... if
    // w.strip()]`). The filter now routes through condition_to_rust —
    // the same truthiness authority the if-statement uses
    // (`(#tokens).is_truthy()`).
    let out = compile(
        concat!(
            "def f(s: str) -> list[str]:\n",
            "    return [w.strip() for w in s.split(\",\") if w.strip()]\n",
        ),
        "compfilter.py",
    );
    assert!(
        out.contains("w . strip ()) . is_truthy ()))")
            || out.contains("w.strip()).is_truthy())"),
        "the comprehension filter must use is_truthy, never ! on a String: {}",
        out
    );
}

#[test]
fn a_dict_store_of_a_reused_instance_clones_value_and_key() {
    // Round 98: `self.items[item.name] = item` — the dict takes the
    // value BY OWNED VALUE while the key reads a field of the same
    // object: the value clones (the key read would borrow a moved
    // value), and the key itself clones (a reused non-self receiver's
    // field read moves the String out of the receiver). CPython
    // evaluates the value first, then the key — the emitted order
    // matches.
    let out = compile(
        concat!(
            "class Item:\n",
            "    def __init__(self, name: str) -> None:\n",
            "        self.name = name\n",
            "\n",
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items: dict[str, Item] = {}\n",
            "    def add(self, item: Item) -> None:\n",
            "        if item.name in self.items:\n",
            "            self.items[item.name].qty += 1\n",
            "        else:\n",
            "            self.items[item.name] = item\n",
        ),
        "dictstoreitem.py",
    );
    assert!(
        out.contains("let __rython_val = Clone :: clone (& (item)) ;")
            || out.contains("let __rython_val=Clone::clone(&(item));"),
        "the dict-store value must clone the reused instance: {}",
        out
    );
    // The key reads `item.name` AFTER the value binding — which is a
    // CLONE, so the receiver is intact and the plain read compiles.
    assert!(
        out.contains("py_set_index (item . name , __rython_val)")
            || out.contains("py_set_index(item.name,__rython_val)"),
        "the key reads the intact receiver (the value was cloned): {}",
        out
    );
}

#[test]
fn sum_over_a_generator_comprehension_sums_the_collected_list() {
    // Round 98: `sum(item.qty for item in self.items.values())` — the
    // generator collector ends its Vec with `.into_iter()`, so sum
    // received an IntoIter with no PySum impl (E0277 in the idiom
    // corpus's total). The runtime implements PySum for the numeric
    // IntoIter forms.
    let out = compile(
        concat!(
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items: dict[str, int] = {}\n",
            "    def total(self) -> int:\n",
            "        return sum(item for item in self.items.values())\n",
        ),
        "sumcomp.py",
    );
    assert!(
        out.contains("sum ({") || out.contains("sum({"),
        "the sum over a comprehension must lower to the runtime sum: {}",
        out
    );
}


#[test]
fn an_overriding_derived_argument_into_a_base_slot_is_loud() {
    // Round 99: `add(Perishable(...))` where `add(item: Item)` — the
    // generated From-slice would LOSE Perishable's label override (the
    // base-typed slot dispatches statically; CPython dispatches
    // dynamically through the base-typed container). Loud refusal.
    let err = compile_err(
        concat!(
            "class Item:\n",
            "    def __init__(self) -> None:\n",
            "        self.qty: int = 0\n",
            "    def label(self) -> str:\n",
            "        return \"x\"\n",
            "\n",
            "class Perishable(Item):\n",
            "    def label(self) -> str:\n",
            "        return \"p\"\n",
            "\n",
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items: dict[str, Item] = {}\n",
            "    def add(self, item: Item) -> None:\n",
            "        self.items[item.qty] = item\n",
            "\n",
            "def main() -> None:\n",
            "    bag = Bag()\n",
            "    bag.add(Perishable())\n",
        ),
        "override.py",
    );
    assert!(
        err.contains("passing `Perishable` where `Item` is expected is not supported yet")
            && err.contains("would lose the override"),
        "the override-loss refusal must name the construct: {}",
        err
    );
}

#[test]
fn sorted_over_class_valued_pairs_sorts_by_key_with_cpython_tie_panic() {
    // Round 99: sorted(d.items()) where the values are class instances —
    // the values have no ordering, so the sort routes to sorted_pairs:
    // by-key, with CPython's TypeError at a key tie (exact for dict keys,
    // which are unique).
    let out = compile(
        concat!(
            "class Item:\n",
            "    def __init__(self) -> None:\n",
            "        self.qty: int = 0\n",
            "\n",
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items: dict[str, Item] = {}\n",
            "    def names(self) -> list[str]:\n",
            "        return [name for (name, item) in sorted(self.items.items())]\n",
        ),
        "sortedpairs.py",
    );
    assert!(
        out.contains("sorted_pairs (&"),
        "the unordered-pair sort must route to sorted_pairs: {}",
        out
    );
}

#[test]
fn mutation_through_a_fetch_local_writes_back_to_the_container() {
    // Round 99 (Directive 4's borrowed-accessor increment):
    // `item = self.items.get(name)` then `item.qty -= qty` — the local is
    // a VIEW of the container slot in CPython, so the mutation must reach
    // it: the lowering writes back (mutate a copy, store the slot, rebind
    // the local) instead of the silent-loss form.
    let out = compile(
        concat!(
            "class Item:\n",
            "    def __init__(self) -> None:\n",
            "        self.qty: int = 0\n",
            "\n",
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items: dict[str, Item] = {}\n",
            "    def take(self, name: str, qty: int) -> int:\n",
            "        item = self.items.get(name)\n",
            "        if item is None:\n",
            "            raise KeyError(name)\n",
            "        item.qty -= qty\n",
            "        return item.qty\n",
        ),
        "writeback.py",
    );
    assert!(
        out.contains("py_set_index"),
        "the mutation must write back to the container slot: {}",
        out
    );
    assert!(
        out.contains("__rython_v . qty = (__rython_v . qty) . py_sub")
            || out.contains("__rython_v.qty = (__rython_v.qty).py_sub"),
        "the mutation applies to the copy, then the slot stores it: {}",
        out
    );
    assert!(
        out.contains("fn take (& mut self") || out.contains("fn take(&mut self"),
        "the write-back needs &mut self: {}",
        out
    );
}

#[test]
fn mutation_through_a_one_hop_fetch_method_writes_back() {
    // The idiom corpus's take: the fetch goes through a METHOD
    // (`self.find(name)` whose body is a single
    // `return self.items.get(name)`) — the provenance resolves the hop
    // and the write-back still fires (the loud aliasing error is gone).
    let out = compile(
        concat!(
            "class Item:\n",
            "    def __init__(self) -> None:\n",
            "        self.qty: int = 0\n",
            "\n",
            "class Bag:\n",
            "    def __init__(self) -> None:\n",
            "        self.items: dict[str, Item] = {}\n",
            "    def find(self, name: str) -> Item | None:\n",
            "        return self.items.get(name)\n",
            "    def take(self, name: str, qty: int) -> int:\n",
            "        item = self.find(name)\n",
            "        if item is None:\n",
            "            raise KeyError(name)\n",
            "        item.qty -= qty\n",
            "        return item.qty\n",
        ),
        "onehop.py",
    );
    assert!(
        out.contains("py_set_index"),
        "the one-hop fetch resolves to the container slot: {}",
        out
    );
    assert!(
        !out.contains("the derived class overrides"),
        "no loud error: {}",
        out
    );
}

// `__setitem__`/`__contains__` dunders receive the subscript store,
// membership test, and the collections.abc `.get` mixin synthesis —
// the class's methods ARE Python's behavior (including its exceptions
// and case-insensitivity). The routing fires only for WELL-TYPED
// dunders (a concrete first-argument annotation); an `Any`-typed dunder
// keeps the pre-existing loud py_index path. The `.get` synthesis is
// gated on the MutableMapping ABC base — a plain `__getitem__`-only
// class must not silently gain methods CPython raises AttributeError
// for.
// ---------------------------------------------------------------------

#[test]
fn a_class_getitem_getsetitem_and_contains_receive_the_operators() {
    let out = compile(
        concat!(
            "class HeaderDict:\n",
            "    def __getitem__(self, key: str) -> str:\n",
            "        return key\n",
            "\n",
            "    def __setitem__(self, key: str, val: str) -> None:\n",
            "        pass\n",
            "\n",
            "    def __contains__(self, key: str) -> bool:\n",
            "        return True\n",
            "\n",
            "    def probe(self) -> str:\n",
            "        x = self[\"a\"]\n",
            "        self[\"b\"] = \"c\"\n",
            "        if \"d\" in self:\n",
            "            return x\n",
            "        return \"\"\n",
        ),
        "dundertrio.py",
    );
    assert!(
        out.contains("(self) . __getitem__ (\"a\") ?"),
        "the subscript read must route to __getitem__: {}",
        out
    );
    assert!(
        out.contains("(self) . __setitem__ (\"b\" , \"c\") ?"),
        "the subscript store must route to __setitem__: {}",
        out
    );
    assert!(
        out.contains("(self) . __contains__ (\"d\") ?"),
        "the membership test must route to __contains__: {}",
        out
    );
}

#[test]
fn a_well_typed_getitem_without_the_mapping_abc_keeps_get_unrouted() {
    // The class defines __getitem__ but does NOT subclass MutableMapping:
    // CPython raises AttributeError on `.get` — the codegen must not
    // synthesize it (it would be a silent divergence).
    let out = compile(
        concat!(
            "class Plain:\n",
            "    def __getitem__(self, key: str) -> str:\n",
            "        return key\n",
            "\n",
            "    def use(self, k: str) -> str:\n",
            "        return self[k]\n",
        ),
        "plainget.py",
    );
    assert!(
        out.contains("(self) . __getitem__ (k) ?"),
        "the subscript still routes: {}",
        out
    );
}

#[test]
fn an_any_typed_dunder_keeps_the_loud_fallback() {
    // `__setitem__(self, key: Any, value: Any)` cannot coerce the call's
    // arguments either — routing would merely swap one loud error for
    // another; the py_index path stays.
    let out = compile(
        concat!(
            "from typing import Any\n",
            "\n",
            "class Pool:\n",
            "    def __setitem__(self, key: Any, value: Any) -> None:\n",
            "        pass\n",
            "\n",
            "    def put(self, k: str, v: str) -> None:\n",
            "        self[k] = v\n",
        ),
        "anydunder.py",
    );
    assert!(
        !out.contains("(self) . __setitem__"),
        "the Any-typed dunder must keep the py_set_index fallback: {}",
        out
    );
}

// ---------------------------------------------------------------------
// Issue #137's Option-aware access: a read, method call, or store
// THROUGH an Option-typed receiver (`self.timeout.connect_timeout()`
// where the field is `Timeout | None` — urllib3) unwraps the Option
// first. CPython raises AttributeError on a None receiver; rython
// lowers that as a loud §12.2 panic with CPython's message. Guarded
// access (`if x is not None:`) is narrowed and never reaches the
// unwrap; a &mut-taking method unwraps mutably, a &self method clones.
// ---------------------------------------------------------------------

#[test]
fn a_method_call_through_an_option_field_unwraps() {
    let out = compile(
        concat!(
            "class Timeout:\n",
            "    def __init__(self, value: float) -> None:\n",
            "        self._value = value\n",
            "\n",
            "    def connect_timeout(self) -> float:\n",
            "        return self._value\n",
            "\n",
            "class Conn:\n",
            "    def __init__(self, timeout: Timeout | None) -> None:\n",
            "        self.timeout = timeout\n",
            "\n",
            "    def total(self) -> float:\n",
            "        return self.timeout.connect_timeout()\n",
        ),
        "optmethod.py",
    );
    assert!(
        out.contains("(self . timeout) . clone () . unwrap_or_else"),
        "the method call must unwrap the Option field: {}",
        out
    );
}

#[test]
fn a_read_through_an_option_field_unwraps() {
    let out = compile(
        concat!(
            "class Timeout:\n",
            "    def __init__(self, value: float) -> None:\n",
            "        self._value = value\n",
            "\n",
            "class Conn:\n",
            "    def __init__(self, timeout: Timeout | None) -> None:\n",
            "        self.timeout = timeout\n",
            "\n",
            "    def label(self) -> float:\n",
            "        return self.timeout._value\n",
        ),
        "optread.py",
    );
    assert!(
        out.contains("(self . timeout) . clone () . unwrap_or_else"),
        "the read must unwrap the Option field: {}",
        out
    );
    assert!(
        out.contains("no attribute '{}'"),
        "the panic must carry CPython's AttributeError text: {}",
        out
    );
}

#[test]
fn a_guarded_option_read_still_unwraps_with_the_panic_net() {
    // `if self.timeout is not None:` — the guard renders as a
    // py_is_none test; attribute receivers are not NAME-narrowed, so the
    // branch read unwraps with the panic as the safety net (the panic
    // can only fire if the guard lied — CPython's AttributeError on a
    // None receiver, §12.2).
    let out = compile(
        concat!(
            "class Timeout:\n",
            "    def __init__(self, value: float) -> None:\n",
            "        self._value = value\n",
            "\n",
            "class Conn:\n",
            "    def __init__(self, timeout: Timeout | None) -> None:\n",
            "        self.timeout = timeout\n",
            "\n",
            "    def label(self) -> float:\n",
            "        if self.timeout is not None:\n",
            "            return self.timeout._value\n",
            "        return 0.0\n",
        ),
        "optguarded.py",
    );
    assert!(
        out.contains("unwrap_or_else"),
        "guarded attribute reads still unwrap (the panic never fires): {}",
        out
    );
}

#[test]
fn a_local_assigned_an_option_field_reads_unwrapped() {
    // `resp_options = self._response_options` (an Option field): the
    // local types as Option from the field table, and reads through it
    // unwrap (the local-seeding half).
    let out = compile(
        concat!(
            "class Opts:\n",
            "    def __init__(self) -> None:\n",
            "        self.request_url: str | None = None\n",
            "\n",
            "class Resp:\n",
            "    def __init__(self) -> None:\n",
            "        self._opts: Opts | None = None\n",
            "\n",
            "    def url(self) -> str:\n",
            "        opts = self._opts\n",
            "        return opts.request_url\n",
        ),
        "optlocal.py",
    );
    assert!(
        out.contains("(opts) . clone () . unwrap_or_else"),
        "reads through the field-seeded local must unwrap: {}",
        out
    );
}

#[test]
fn an_option_value_in_condition_position_tests_none_ness() {
    // `if conn:` where conn is `BaseHTTPConnection | None` — CPython's
    // truthiness of a user object is "not None"; the generic Truthy-for-
    // Option impl needs T: Truthy, which a user class lacks (E0599 ×12
    // in the corpus). `!(x).py_is_none()` works for Option and boxed
    // bindings alike (both have unconditional PyIsNone).
    let out = compile(
        concat!(
            "class Conn:\n",
            "    def __init__(self) -> None:\n",
            "        self._ok = True\n",
            "\n",
            "class Pool:\n",
            "    def __init__(self, conn: Conn | None) -> None:\n",
            "        self.conn = conn\n",
            "\n",
            "    def state(self) -> str:\n",
            "        if self.conn:\n",
            "            return \"open\"\n",
            "        return \"closed\"\n",
        ),
        "opttruth.py",
    );
    assert!(
        out.contains("! (self . conn) . py_is_none ()"),
        "an Option in condition position must test None-ness: {}",
        out
    );
}

#[test]
fn an_option_field_augmented_add_operates_on_the_inner() {
    // Round 83: `self._data += data` where the field is `bytes | None`
    // (urllib3's DeflateDecoder — the None stores widen `_data` to
    // Option<Vec<u8>>): the aug-add reads the INNER value, py_adds, and
    // stores back wrapped — a None here is CPython's TypeError
    // (`None + bytes`), a loud §12.2 panic with the message.
    let out = compile(
        concat!(
            "class Decoder:\n",
            "    def __init__(self) -> None:\n",
            "        self._data: bytes | None = b\"\"\n",
            "\n",
            "    def feed(self, data: bytes) -> bytes:\n",
            "        self._data += data\n",
            "        return data\n",
        ),
        "optaugadd.py",
    );
    assert!(
        out.contains("unsupported operand type(s) for +=: 'NoneType' and 'bytes'"),
        "the Option aug-add must carry CPython's TypeError text: {}",
        out
    );
    assert!(
        out.contains("Some ((__rython_v) . py_add"),
        "the Option aug-add must py_add the INNER value and store wrapped: {}",
        out
    );
}

#[test]
fn an_option_field_value_into_a_concrete_slot_unwraps_loudly() {
    // Round 83: `self.decompress(self._data)` where the field is
    // `bytes | None` (urllib3's DeflateDecoder reading the now-Option
    // `_data` into the `data: bytes` parameter): the Option unwraps with
    // the loud conversion panic — Python fails at use on a None value,
    // rython at the conversion (§12.2), mirroring the return site.
    let out = compile(
        concat!(
            "class Decoder:\n",
            "    def __init__(self) -> None:\n",
            "        self._data: bytes | None = b\"\"\n",
            "\n",
            "    def feed(self, data: bytes) -> bytes:\n",
            "        return self.decompress(self._data)\n",
            "\n",
            "    def decompress(self, data: bytes) -> bytes:\n",
            "        return data\n",
        ),
        "optarg.py",
    );
    assert!(
        out.contains("rython: an optional value was None where a concrete value was required"),
        "an Option value into a concrete slot must unwrap with the loud panic: {}",
        out
    );
    assert!(
        out.contains("Some (__rython_v) => __rython_v"),
        "the Some arm must yield the inner value: {}",
        out
    );
}

// ---------------------------------------------------------------------
// The bytes-display slice (issue #137): Python displays a bytes value as
// `b'ab'`, NOT the int-list the blanket Vec<T> display renders. print of
// a bytes-typed argument routes through the runtime's CPython-verified
// py_bytes_repr, and bytes + bytes infers Bytes (so a concatenated
// value keeps the bytes path).
// ---------------------------------------------------------------------

#[test]
fn printing_a_bytes_value_uses_the_bytes_repr() {
    let out = compile(
        "def show():\n    print(b\"ab\")\n\ndef add(x: bytes) -> bytes:\n    return x + b\"c\"\n",
        "bytesprint.py",
    );
    assert!(
        out.contains("py_bytes_repr"),
        "print of a bytes value must route through py_bytes_repr: {}",
        out
    );
    assert!(
        out.contains("fn add (x : Vec < u8 >) -> Result < Vec < u8 > , PyException >"),
        "bytes + bytes must infer bytes: {}",
        out
    );
}

#[test]
fn bytes_join_uses_the_runtime_helper() {
    let out = compile(
        "def assemble(parts: list[bytes]) -> bytes:\n    return b\"\".join(parts)\n",
        "bytesjoin.py",
    );
    assert!(
        out.contains("bytes_join"),
        "bytes join must route to the runtime helper: {}",
        out
    );
}

// ---------------------------------------------------------------------
// `raise X from None` (issue #137): CPython sets the cause to None —
// no cause text at all — and the None literal cannot format (E0277 ×17
// in the corpus). The from-None shape skips the §12.3 cause folding;
// `raise X from Y` (a real cause) keeps it.
// ---------------------------------------------------------------------

#[test]
fn raise_from_none_skips_the_cause_text() {
    let out = compile(
        concat!(
            "def fail():\n",
            "    raise KeyError(\"outer\") from None\n",
        ),
        "raisefromnone.py",
    );
    assert!(
        !out.contains("from None") || out.contains("KeyError"),
        "from None must not fold cause text into the message: {}",
        out
    );
    assert!(
        !out.contains("format ! (\"{}\" , None)"),
        "the None literal must not be formatted: {}",
        out
    );
}

#[test]
fn raise_from_a_real_cause_keeps_the_documented_folding() {
    let out = compile(
        concat!(
            "def fail(e):\n",
            "    raise KeyError(\"outer\") from e\n",
        ),
        "raisefromcause.py",
    );
    assert!(
        out.contains("from"),
        "the §12.3 cause folding stays for a real cause: {}",
        out
    );
}

#[test]
fn class_instances_display_through_str_or_default_repr() {
    // Round 34: str(x)/print(x)/f-string {x} on a class instance route
    // through py_display; the generated PyDisplay impl calls __str__ when
    // the class defines one, else emits the default object repr. And a
    // class instance in an exception message wraps in py_display (not raw
    // format!, which would demand a Rust Display).
    let out = compile(
        concat!(
            "class PoolError(Exception):\n",
            "    pass\n",
            "\n",
            "class Pool:\n",
            "    def __init__(self, host: str):\n",
            "        self._host = host\n",
            "\n",
            "    def __str__(self) -> str:\n",
            "        return f\"Pool(host={self._host!r})\"\n",
            "\n",
            "class Raw:\n",
            "    pass\n",
            "\n",
            "def boom():\n",
            "    raise PoolError(Pool(\"x\"), \"closed.\")\n",
            "\n",
            "print(str(Raw()))\n",
            "print(Pool(\"y\"))\n",
        ),
        "disp.py",
    );
    assert!(
        out.contains("impl stdpython :: PyDisplay for Pool")
            || out.contains("impl stdpython::PyDisplay for Pool"),
        "the class must get a PyDisplay impl: {}",
        out
    );
    assert!(
        out.contains("self . __str__ () . unwrap_or_else")
            || out.contains("self.__str__().unwrap_or_else"),
        "a class with __str__ must route display through it: {}",
        out
    );
    assert!(
        out.contains("\"<{} object>\"") && out.contains("\"Raw\""),
        "a class without __str__/__repr__ uses the default object repr: {}",
        out
    );
    assert!(
        out.contains("print (& ({ Pool :: new")
            || out.contains("print(&({ Pool::new"),
        "a class instance in print compiles through the PyDisplay bound: {}",
        out
    );
}

#[test]
fn hierarchy_trait_carries_py_display_bound() {
    // Round 41: a trait-DEFAULT body that formats `self` in an exception
    // message (`raise ClosedPoolError(self)` — urllib3's _get_conn)
    // lowers through py_display, which needs `Self: PyDisplay` — the
    // concrete class always carries the generated impl, so the trait
    // declares the bound (every implementor satisfies it).
    let out = compile(
        concat!(
            "class PoolError(Exception):\n",
            "    pass\n",
            "\n",
            "class Base:\n",
            "    def __init__(self):\n",
            "        self.x = 1\n",
            "\n",
            "class Pool(Base):\n",
            "    def _get(self):\n",
            "        raise PoolError(self)\n",
        ),
        "dispbound.py",
    );
    assert!(
        out.contains("pub trait PoolTrait : BaseTrait where Self : stdpython :: PyDisplay")
            || out.contains("pub trait PoolTrait: BaseTrait where Self: stdpython::PyDisplay"),
        "the hierarchy trait must declare Self: PyDisplay: {}",
        out
    );
    assert!(
        out.contains("pub trait BaseTrait where Self : stdpython :: PyDisplay")
            || out.contains("pub trait BaseTrait where Self: stdpython::PyDisplay"),
        "the base trait must declare Self: PyDisplay: {}",
        out
    );
}

#[test]
fn exception_message_args_wrap_class_instances_in_py_display() {
    // The raise-site message flattening wraps class instances, Options,
    // and boxed values in py_display (Python's str) — a raw format! would
    // fail to compile (no Rust Display on a class struct).
    let out = compile(
        concat!(
            "class PoolError(Exception):\n",
            "    pass\n",
            "\n",
            "class Pool:\n",
            "    def __init__(self, host: str):\n",
            "        self._host = host\n",
            "\n",
            "def boom():\n",
            "    raise PoolError(Pool(\"x\"), \"closed.\")\n",
        ),
        "disp.py",
    );
    assert!(
        out.contains("py_display (& ({ Pool :: new")
            || out.contains("py_display(&({ Pool::new"),
        "the message arg must wrap the class instance: {}",
        out
    );
}

#[test]
fn derived_trait_base_accessors_are_inherent_and_qualified() {
    // Issue #137's E0034 cluster: a derived class implements BOTH its own
    // trait and every ancestor trait, each declaring `base` with a
    // different return type — `self.base()` on the concrete receiver would
    // be ambiguous. The struct gets INHERENT base()/base_mut() accessors
    // (inherent methods win over trait ones), and generic trait-default
    // bodies qualify the first hop with the own trait.
    let out = compile(
        concat!(
            "class Base:\n",
            "    def __init__(self):\n",
            "        self.x = 0\n",
            "\n",
            "class Mid(Base):\n",
            "    def __init__(self):\n",
            "        super().__init__()\n",
            "        self.y = 0\n",
            "\n",
            "class Leaf(Mid):\n",
            "    def __init__(self):\n",
            "        super().__init__()\n",
            "        self.z = 0\n",
            "\n",
            "    def read_x(self) -> int:\n",
            "        return self.x\n",
        ),
        "inherit.py",
    );
    assert!(
        out.contains("pub (crate) fn base (& self) -> & Mid")
            || out.contains("pub(crate) fn base(&self) -> &Mid"),
        "the derived struct needs an inherent base accessor: {}",
        out
    );
    assert!(
        out.contains("< Self as LeafTrait > :: base (self)")
            || out.contains("<Self as LeafTrait>::base(self)"),
        "generic trait-default bodies must qualify the first base hop: {}",
        out
    );
}

#[test]
fn factory_local_property_reads_route_through_the_getter() {
    // Issue #137's E0615 cluster: `timeout_obj = self._get_timeout(
    // timeout)` (whose return annotation names Timeout) makes
    // `timeout_obj.connect_timeout` a PROPERTY read — the attribute path
    // resolves the factory local's class and routes the read through the
    // getter call, instead of emitting a method-as-value E0615.
    let out = compile(
        concat!(
            "class Timeout:\n",
            "    @property\n",
            "    def connect_timeout(self) -> float:\n",
            "        return 1.0\n",
            "\n",
            "class Pool:\n",
            "    def _get_timeout(self, timeout) -> Timeout:\n",
            "        return Timeout()\n",
            "\n",
            "    def go(self):\n",
            "        timeout_obj = self._get_timeout(1)\n",
            "        return timeout_obj.connect_timeout\n",
        ),
        "timeout.py",
    );
    assert!(
        out.contains("timeout_obj . connect_timeout ()")
            || out.contains("timeout_obj.connect_timeout()"),
        "the factory local's property read must call the getter: {}",
        out
    );
}

#[test]
fn type_self_construction_lowers_to_the_class_constructor() {
    // `result = type(self)(maybe_constructable)` (urllib3's
    // HTTPHeaderDict.__ror__): the class OBJECT is constructed — CPython
    // builds a new instance of the runtime class — so the call lowers to
    // the same-class construction (::new with the signature mapping),
    // never a class-name string being called (the old E0618 string-call).
    let out = compile(
        concat!(
            "class A:\n",
            "    def __init__(self, x: int = 0):\n",
            "        self.x = x\n",
            "\n",
            "    def duplicate(self) -> A:\n",
            "        return type(self)(self.x)\n",
        ),
        "cls.py",
    );
    assert!(
        out.contains("A :: new (self . x)")
            || out.contains("A::new(self.x)"),
        "type(self)(...) must construct the class: {}",
        out
    );
    assert!(
        !out.contains("stringify!") || !out.contains("to_string () ("),
        "the old string-call must be gone: {}",
        out
    );
}

#[test]
fn set_literal_call_uses_the_runtime_conversion() {
    // `set("abc")` — the set conversion of a string (urllib3's
    // `_UNRESERVED_CHARS = set("...")`): a set of one-char strings via
    // the runtime `set()` — never a `"set"(...)` string-call (the old
    // E0618 bug).
    let out = compile(
        concat!(
            "_UNRESERVED_CHARS = set(\"ABCabc\")\n",
            "\n",
            "def has(ch: str) -> bool:\n",
            "    return ch in _UNRESERVED_CHARS\n",
        ),
        "sets.py",
    );
    assert!(
        out.contains("set (\"ABCabc\")") || out.contains("set(\"ABCabc\")"),
        "set(...) must call the runtime conversion: {}",
        out
    );
    assert!(
        !out.contains("\"set\"("),
        "the string-call form is gone: {}",
        out
    );
}

#[test]
fn call_through_a_boxed_value_drops_loudly() {
    // Issue #122's callable-as-value divergence: a call whose callee is a
    // boxed/String/Option VALUE (a `pool_cls(...)`-style class-name local,
    // or an unresolvable self member like `self._tunnel()` — a method
    // inherited from an external base) lowers to the warned no-op (the
    // boxed None), never a `value(...)` E0618 string-call or an E0599.
    let out = compile_with_warnings(
        concat!(
            "def take(cb) -> str:\n",
            "    return cb(\"x\")\n",
            "\n",
            "def run():\n",
            "    f = take\n",
            "    return f(\"y\")\n",
        ),
        "cb.py",
    );
    assert!(
        out.0.contains("PyValue :: None_") || out.0.contains("PyValue::None_"),
        "the call through the callable value must lower to the boxed None: {}",
        out.0
    );
}

#[test]
fn unannotated_class_default_lowers_to_the_name_string() {
    // Round 56: `def merge_setting(request_setting, session_setting,
    // dict_class=OrderedDict)` (requests' sessions.py — dict_class has NO
    // annotation): the call site omits dict_class, so the inlined default
    // is the class object, whose rython value is its NAME STRING (round
    // 33). Previously the default rendered as a bare struct name — E0423
    // (the struct lives in the type namespace). All 21 requests E0423s.
    let out = compile(
        concat!(
            "from collections import OrderedDict\n",
            "def merge_setting(a, b, dict_class=OrderedDict):\n",
            "    return dict_class\n",
            "def f(x, y):\n",
            "    return merge_setting(x, y)\n",
        ),
        "classdefault.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("merge_setting(x,y,\"OrderedDict\".to_string())"),
        "the inlined class default must lower to the name string: {}",
        out
    );
}

#[test]
fn string_literal_annotation_unquotes_to_the_real_type() {
    // Round 56: `def _urllib3_request_context(request, verify: "bool |
    // str | None", ...)` — requests' adapters.py writes its annotations as
    // QUOTED STRINGS (CPython's typing.get_type_hints evaluates them). The
    // annotation authorities must re-parse the string content, or the
    // parameter renders as a literal `"bool | str | None"` type — a parse
    // error that breaks every use of the parameter (the 8 verify +
    // 6 client_cert E0425s).
    let out = compile(
        concat!(
            "def f(verify: \"bool | str | None\"):\n",
            "    if verify is None:\n",
            "        return 0\n",
            "    return 1\n",
        ),
        "strann.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("verify:stdpython::PyValue"),
        "the quoted union must unquote to the boxed PyValue param: {}",
        out
    );
    assert!(
        !flat.contains("\"bool|str|None\""),
        "the raw string must not render as a Rust type: {}",
        out
    );
}

#[test]
fn builtin_class_names_in_value_position_lower_to_name_strings() {
    // Round 56: a bare BUILTIN class name read as a VALUE (`basestring =
    // (str, bytes)`, `HEADER_VALIDATORS = {bytes: ..., str: ...}` —
    // requests' compat/_internal_utils): builtin classes are class
    // objects, and the class-as-value model names them by their name
    // string. Previously the tuple/dict rendered bare `str`/`bytes`
    // idents — E0425 (or, worse, resolved to the stdpython::str function
    // through the glob, silently wrong). Both requests errors.
    let out = compile(
        concat!(
            "builtin_str = str\n",
            "str = str\n",
            "bytes = bytes\n",
            "basestring = (str, bytes)\n",
            "numeric_types = (int, float)\n",
            "integer_types = (int,)\n",
            "HEADER_VALIDATORS = {bytes: 1, str: 2}\n",
        ),
        "builtinval.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("basestring=(\"str\".to_string(),\"bytes\".to_string())"),
        "a builtin-class tuple must lower to name strings: {}",
        out
    );
    assert!(
        flat.contains("numeric_types=(\"int\".to_string(),\"float\".to_string())"),
        "the int/float tuple must lower to name strings: {}",
        out
    );
    assert!(
        flat.contains("integer_types=(\"int\".to_string(),)"),
        "the 1-tuple must lower to the name string: {}",
        out
    );
    assert!(
        flat.contains("PyDict::from([(\"bytes\".to_string(),1),(\"str\".to_string(),2)])"),
        "builtin-class dict keys must lower to name strings: {}",
        out
    );
}

#[test]
fn compat_builtin_self_alias_import_drops_and_calls_dispatch_to_builtin() {
    // Round 56: `str = str` / `bytes = bytes` (requests' compat py2 shims)
    // drop as no-ops, so a sibling's `from .compat import str` has NO
    // runtime item to re-export — the import must drop (a `pub use
    // crate::compat::str` is E0603: nothing public behind it), and the
    // name still means the BUILTIN: `str(x)` / `str(x, encoding)` calls
    // dispatch to the builtin arms (decode_by_name for the encoding form).
    let compat = parse("str = str\nbytes = bytes\n", "compat.py").unwrap();
    let caller = parse(
        concat!(
            "from compat import str, bytes\n",
            "def conv(x: bytes, enc: str) -> str:\n",
            "    return str(x, encoding=enc)\n",
            "def conv2(x: str) -> str:\n",
            "    return str(x)\n",
            "def conv3(x: str) -> bytes:\n",
            "    return bytes(x)\n",
        ),
        "caller.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["compat".to_string()], std::rc::Rc::new(compat));
    // The defining module's runtime-item drop only applies in a
    // MULTI-module conversion (module_defs.len() > 1) — a lone module
    // must assume an unknown absolute import is a crate sibling.
    defs.insert(vec!["caller".to_string()], std::rc::Rc::new(caller.clone()));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let symbols = caller.clone().find_symbols(SymbolTableScopes::new());
    let out = caller
        .to_rust(
            CodeGenContext::Module("caller".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !flat.contains("pub use crate::compat::str")
            && !flat.contains("pub use crate::compat::bytes"),
        "the builtin self-alias imports must drop (no public item to point at): {}",
        out
    );
    assert!(
        flat.contains("decode_by_name(&(x),enc)?"),
        "str(x, encoding=enc) must dispatch to the decode arm: {}",
        out
    );
    assert!(
        flat.contains("returnOk(str(x))"),
        "str(x) must dispatch to the builtin str(): {}",
        out
    );
    assert!(
        flat.contains("(x).into_bytes_like()"),
        "bytes(x) must dispatch to the builtin bytes arm: {}",
        out
    );
}

#[test]
fn for_loop_target_read_only_in_del_keeps_its_name() {
    // Round 56: `for key in none_keys: del merged_setting[key]` —
    // requests' merge_setting. The del statement READS the loop target,
    // but the reference walk missed Delete targets, so the loop analysis
    // declared `key` unused and lowered the target to `_` while the
    // body's `py_pop(key)` still referenced it (E0425 in the generated
    // crate).
    let out = compile(
        concat!(
            "def merge_setting(d, none_keys):\n",
            "    for key in none_keys:\n",
            "        del d[key]\n",
            "    return d\n",
        ),
        "delloop.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("forkeyinnone_keys{"),
        "the loop target must keep its name (read by the del): {}",
        out
    );
    assert!(
        flat.contains("py_pop(key)?"),
        "the del must still read the named key: {}",
        out
    );
}

#[test]
fn range_annotation_maps_to_py_range() {
    // Round 56: `offsets: range` — the builtin range class as a
    // parameter annotation (charset_normalizer's cut_sequence_chunks).
    // The annotation authority mapped int/float/str/bytes but not range,
    // so the parameter rendered a bare `range` ident — E0573.
    let out = compile(
        concat!(
            "def cut(sequences: bytes, offsets: range, chunk_size: int):\n",
            "    for i in offsets:\n",
            "        yield i\n",
        ),
        "rangeann.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("offsets:PyRange"),
        "the range annotation must map to the runtime PyRange: {}",
        out
    );
}

#[test]
fn mixed_arity_tuple_list_boxes_heterogeneous_elements() {
    // Round 57: a list literal whose elements are tuples of DIFFERENT
    // arities (`[(0, "3"), (65, "M", "a"), (76, "V")]` — idna's _seg
    // tables) boxes each element as PyValue. The element-type fold was
    // order-dependent: a trailing 2-tuple re-absorbed the heterogeneous
    // result (`unify(PyObject, Tuple2)` snaps back to Tuple2), hiding the
    // mix from the boxable-union check — every 3-tuple then mismatched
    // the inferred Vec<(i64, &str)> (E0308 per row).
    let out = compile(
        concat!(
            "def _seg_0():\n",
            "    return [\n",
            "        (0, \"3\"),\n",
            "        (65, \"M\", \"a\"),\n",
            "        (76, \"V\"),\n",
            "    ]\n",
        ),
        "seglist.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("PyValue::from(((65,\"M\",\"a\")))")
            && flat.contains("PyValue::from(((76,\"V\")))"),
        "mixed-arity tuple list elements must box as PyValue: {}",
        out
    );
}

#[test]
fn list_of_union_tuples_return_annotation_boxes_elements() {
    // Round 57: `-> List[Union[Tuple[int, str], Tuple[int, str, str]]]`
    // (idna's _seg annotations) — the Union subscript resolved to
    // nothing (the annotation authority only knew the `A | B` spelling),
    // so the return type defaulted to `()` and the homogeneous segments
    // of a boxed-element list stayed Vec<(i64, &str)> (E0308). The Union
    // subscript maps to the boxed PyValue, and the RETURNING list boxes
    // each element.
    let out = compile(
        concat!(
            "from typing import List, Tuple, Union\n",
            "def _seg_0() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:\n",
            "    return [(0, \"3\"), (65, \"M\", \"a\")]\n",
        ),
        "segann.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("fn_seg_0()->Result<Vec<stdpython::PyValue>"),
        "the Union-typed return must resolve to Vec<PyValue>: {}",
        out
    );
    assert!(
        flat.contains("PyValue::from(((0,\"3\")))"),
        "the returning list must box each element: {}",
        out
    );
}

#[test]
fn module_tuple_unpack_promotes_names_read_by_functions() {
    // Round 57: a module-level TUPLE-UNPACK (`_STATUS_VALID,
    // _STATUS_MAPPED, ... = b"VMDI"` — idna's core.py) binds each name
    // to the value at its position; the names functions read must be
    // promoted to statics extracting their element (a module-init local
    // is invisible to function bodies — E0425).
    let out = compile(
        concat!(
            "_STATUS_VALID, _STATUS_MAPPED, _STATUS_DEVIATION, _STATUS_IGNORED = b\"VMDI\"\n",
            "def encode(domain):\n",
            "    if domain == _STATUS_VALID:\n",
            "        return 1\n",
            "    if domain == _STATUS_MAPPED:\n",
            "        return 2\n",
            "    return 0\n",
        ),
        "statusunpack.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("pubstatic_STATUS_VALID:std::sync::LazyLock<stdpython::PyValue>")
            && flat.contains("pubstatic_STATUS_MAPPED:"),
        "function-read unpack names must promote to statics: {}",
        out
    );
    assert!(
        flat.contains("py_index(0i64)") && flat.contains("py_index(1i64)"),
        "each promoted name's static must extract its element at its position: {}",
        out
    );
    assert!(
        !flat.contains("letmut_STATUS_VALID"),
        "promoted names must not remain module-init locals: {}",
        out
    );
}

#[test]
fn boxed_bool_fold_returns_the_python_operand_in_both_orders() {
    // The retrospective's shipped wrong-semantics finding on #260: the
    // (Bool, PyValue) / (PyValue, Bool) fold special-cased the bool-first
    // order and SWAPPED the arms for the value-first order — `y and x`
    // (y boxed, x bool) returned y on a truthy y instead of x, and
    // `y or x` returned x on a truthy y. Python `a and b` is a if a is
    // falsy else b; `a or b` is a if a is truthy else b — order-
    // independent. Verified against python3: for y=1, x=False, `y and x`
    // is False (x), `y or x` is 1 (y); for y=0, x=True, `y and x` is 0
    // (y), `y or x` is True (x).
    let out = compile(
        concat!(
            "def f(x: bool, y: object):\n",
            "    a = y and x\n",
            "    b = y or x\n",
            "    c = x and y\n",
            "    d = x or y\n",
            "    return a, b, c, d\n",
        ),
        "boxedbool.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    // `y and x`: truthy y -> the bool (x); `y or x`: truthy y -> the
    // value (y).
    assert!(
        flat.contains("let__rython_and=y;if(__rython_and).is_truthy(){PyValue::from(x)}else{PyValue::from((__rython_and).clone())}"),
        "value-first AND must return the bool on a truthy value (via the bound temp, evaluated once): {}",
        out
    );
    assert!(
        flat.contains("let__rython_or=y;if(__rython_or).is_truthy(){PyValue::from((__rython_or).clone())}else{PyValue::from(x)}"),
        "value-first OR must return the value on a truthy value (via the bound temp, evaluated once): {}",
        out
    );
    // `x and y` (bool first): truthy x -> the value (y); `x or y`:
    // truthy x -> the bool (x) — both via the bound temp.
    assert!(
        flat.contains("let__rython_and=x;if(__rython_and).is_truthy(){PyValue::from(y)}else{PyValue::from((__rython_and).clone())}"),
        "bool-first AND must return the value on a truthy bool: {}",
        out
    );
    assert!(
        flat.contains("let__rython_or=x;if(__rython_or).is_truthy(){PyValue::from((__rython_or).clone())}else{PyValue::from(y)}"),
        "bool-first OR must return the bool on a truthy bool: {}",
        out
    );
}

#[test]
fn boxed_return_list_annotation_does_not_retag_local_lists() {
    // Devin review on #263 (Finding 1): the first version set the forced
    // list element on the SHARED function options, so a `-> List[Union[
    // ...]]` return annotation boxed EVERY list literal in the function
    // — local lists and call arguments gained unintended element types.
    // The forced element now rides only on the Return statement's own
    // options clone: the returned list boxes, a local list keeps its
    // own inference.
    let out = compile(
        concat!(
            "from typing import List, Tuple, Union\n",
            "def f() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:\n",
            "    local = [1, 2]\n",
            "    return [(0, \"3\"), (65, \"M\", \"a\")]\n",
        ),
        "retlocal.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("PyValue::from(((0,\"3\")))"),
        "the RETURNED list must box each element: {}",
        out
    );
    assert!(
        flat.contains("local=vec![1,2]"),
        "a LOCAL list must keep its own Vec<i64> inference: {}",
        out
    );
    assert!(
        !flat.contains("local=vec![PyValue::from(1),PyValue::from(2)]"),
        "the local list must NOT be retagged as boxed: {}",
        out
    );
}

#[test]
fn boxed_return_list_annotation_spreads_starred_elements() {
    // Devin review on #263 (Finding 2): under a boxed-element return
    // annotation, `*xs` was emitted as ONE list element instead of
    // spreading the collection. The forced branch now interleaves
    // fixed elements and spreads in source order.
    let out = compile(
        concat!(
            "from typing import List, Tuple, Union\n",
            "def f(xs: list) -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:\n",
            "    return [(0, \"3\"), *xs, (65, \"M\", \"a\")]\n",
        ),
        "starret.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("__rython_list.extend("),
        "a starred element must SPREAD into the returned list: {}",
        out
    );
    assert!(
        flat.contains("__rython_list.push(PyValue::from(((65,\"M\",\"a\")))"),
        "the fixed elements after the spread must still box, in order: {}",
        out
    );
}

#[test]
fn module_tuple_unpack_emits_shared_rhs_static_and_typed_projections() {
    // Devin review on #263 (Findings 3+4): the first version of the
    // unpack promotion emitted one static PER NAME, each re-evaluating
    // the whole RHS (`a, b = make()` ran make() twice — side effects
    // repeat, names from different results) and truncating every element
    // through `as i64` (`a, b = (1.5, 2.5)` boxed 1 instead of 1.5).
    // One shared `__rython_unpack_N` static now evaluates the RHS once;
    // each name projects its element from it, boxed as-is.
    let out = compile(
        concat!(
            "a, b = (1.5, 2.5)\n",
            "def f():\n",
            "    return a, b\n",
        ),
        "unpackrhs.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("pubstatic__rython_unpack_0:std::sync::LazyLock<stdpython::PyValue>"),
        "one shared RHS static must hold the evaluated value: {}",
        out
    );
    assert!(
        flat.contains("(*__rython_unpack_0).clone().py_index(0i64)")
            && flat.contains("(*__rython_unpack_0).clone().py_index(1i64)"),
        "each name's static must PROJECT its element from the shared RHS: {}",
        out
    );
    assert!(
        !flat.contains("py_index(0i64){Ok(__rython_elt)=>PyValue::from(__rython_eltasi64)"),
        "the projection must NOT truncate the element through as i64: {}",
        out
    );
}

#[test]
fn option_typed_field_read_into_option_slot_does_not_double_wrap() {
    // The retrospective's R2 double-wrap family: a field whose type is
    // Option (`ca_cert_dir: str | None`) read into an Option-typed
    // parameter (`ca_cert_dir=self.ca_cert_dir` — urllib3's
    // _ssl_wrap_socket call sites) used to render `Some(self.ca_cert_dir
    // ())` — Option<Option<String>>. The field read already IS the
    // Option (the accessor returns it); the wrap must pass it through.
    let out = compile(
        concat!(
            "class C:\n",
            "    def __init__(self):\n",
            "        self.ca_cert_dir: str | None = None\n",
            "    def use(self):\n",
            "        return take(self.ca_cert_dir)\n",
            "def take(ca_cert_dir: str | None) -> str | None:\n",
            "    return ca_cert_dir\n",
        ),
        "optfield.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("take(self.ca_cert_dir)") || flat.contains("take((self.ca_cert_dir).clone())"),
        "an Option-typed field read must pass through an Option parameter unwrapped: {}",
        out
    );
    assert!(
        !flat.contains("Some(self.ca_cert_dir())"),
        "the Option field must not be double-wrapped: {}",
        out
    );
}

#[test]
fn option_typed_field_read_stored_into_option_local_does_not_double_wrap() {
    // The store twin of the double-wrap family: `destination_scheme =
    // parsed_url.scheme` where parsed_url is a Url instance and scheme
    // is an Option<String> field — the store into the Option local
    // (`destination_scheme` assigned None on another path) used to wrap
    // `Some(parsed_url.scheme)` — Option<Option<String>>.
    let out = compile(
        concat!(
            "class Url:\n",
            "    def __init__(self):\n",
            "        self.scheme: str | None = None\n",
            "def f(parsed_url: Url) -> str | None:\n",
            "    destination_scheme = parsed_url.scheme\n",
            "    return destination_scheme\n",
        ),
        "optstore.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !flat.contains("destination_scheme=Some("),
        "an Option-typed field store must not be double-wrapped: {}",
        out
    );
}

#[test]
fn imported_factory_option_field_crosses_modules_unwrapped() {
    // Devin review on #264: when only an IMPORTED factory exposes its
    // result class, the defining module's symbol table must survive the
    // resolution — `make() -> Result` in module A, `u = make()` in
    // module B, `u.field` (Option-typed in A) into an Option slot must
    // pass through without the double-wrap. The first version discarded
    // the defining symbols, so A's Result never resolved in B.
    let defs_mod = parse(
        concat!(
            "class Result:\n",
            "    def __init__(self):\n",
            "        self.field: str | None = None\n",
            "def make() -> Result:\n",
            "    return Result()\n",
        ),
        "resultmod.py",
    )
    .unwrap();
    let caller = parse(
        concat!(
            "from resultmod import make\n",
            "def take(field: str | None) -> str | None:\n",
            "    return field\n",
            "def f() -> str | None:\n",
            "    u = make()\n",
            "    return take(u.field)\n",
        ),
        "caller2.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        vec!["resultmod".to_string()],
        std::rc::Rc::new(defs_mod),
    );
    defs.insert(
        vec!["caller2".to_string()],
        std::rc::Rc::new(caller.clone()),
    );
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let symbols = caller.clone().find_symbols(SymbolTableScopes::new());
    let out = caller
        .to_rust(
            CodeGenContext::Module("caller2".to_string()),
            options,
            symbols,
        )
        .unwrap()
        .to_string();
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("take(u.field)") || flat.contains("take((u.field).clone())"),
        "an imported factory's Option field must pass through an Option slot unwrapped: {}",
        out
    );
    assert!(
        !flat.contains("take(Some(u.field)"),
        "the imported factory's Option field must not double-wrap: {}",
        out
    );
}

#[test]
fn option_field_store_of_a_reused_name_wraps_and_clones() {
    // Round 59: `self._last_printable_char = character` where the field
    // is `str | None` and character is read again later (charset_normalizer's
    // _count_suspicious): the reused-name CLONE arm preceded the Option-wrap
    // arm, so the store rendered `(character).clone()` — a bare String into
    // the Option field (E0308). The Option arm now runs first and clones
    // INTO the Some (`Some((character).clone())`).
    let out = compile(
        concat!(
            "class Mess:\n",
            "    def __init__(self):\n",
            "        self._last_printable_char: str | None = None\n",
            "    def count(self, character: str) -> str:\n",
            "        self._last_printable_char = character\n",
            "        return character\n",
        ),
        "optstore2.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("_last_printable_char=Some((character).clone())"),
        "a reused name stored into an Option field must clone into Some: {}",
        out
    );
    assert!(
        !flat.contains("_last_printable_char=(character).clone()"),
        "the bare clone must not bypass the Option wrap: {}",
        out
    );
}

#[test]
fn mapping_get_binds_the_default_before_the_match() {
    // Devin review on #267: the Mapping-get synthesis placed the
    // default ONLY in the KeyError arm — a present key skipped the
    // default's side effects, unlike Python (which evaluates receiver,
    // key, and default eagerly, exactly once, before entering get).
    // Verified against python3: `d.get('k', side_effect())` runs
    // side_effect even when 'k' is present. The generated code must bind
    // `__rython_default` before the match.
    let out = compile(
        concat!(
            "from typing import MutableMapping\n",
            "class M(MutableMapping[str, str]):\n",
            "    def __getitem__(self, key: str) -> str:\n",
            "        return \"v\"\n",
            "    def __len__(self) -> int:\n",
            "        return 0\n",
            "    def __iter__(self):\n",
            "        return iter([])\n",
            "def side_effect() -> str:\n",
            "    return \"d\"\n",
            "def f(m: M) -> str:\n",
            "    return m.get(\"k\", side_effect())\n",
        ),
        "mapget.py",
    );
    assert!(
        out.contains("__rython_default"),
        "the default must be bound before the match (eager, exactly once): {}",
        out
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    // Python's evaluation order: RECEIVER, KEY, DEFAULT — each bound
    // once before the __getitem__ invocation (Devin review on #267, the
    // reorder pass: the first fix bound the default first).
    let recv = flat.find("let__rython_recv=").unwrap_or(usize::MAX);
    let key = flat.find("let__rython_key=").unwrap_or(usize::MAX);
    let dflt = flat.find("let__rython_default=").unwrap_or(usize::MAX);
    assert!(
        recv < key && key < dflt,
        "receiver, key, default must bind in Python's order: {}",
        out
    );
    assert!(
        flat.contains("Err(__rython_e)if__rython_e.matches(\"KeyError\")=>__rython_default"),
        "the KeyError arm must use the pre-bound default: {}",
        out
    );
    assert!(
        flat.contains("__rython_recv.__getitem__(__rython_key)"),
        "the __getitem__ call must use the bound receiver and key: {}",
        out
    );
}

#[test]
fn mapping_get_with_an_option_typed_default_keeps_the_option() {
    // Round 83: `.get(k, default)` where the DEFAULT is `str | None`
    // (urllib3's getheader — `default: str | None = None` into the
    // mapping-get synthesis): the fallback IS the Option (the result is
    // Option<String>, matching the Some-wrapped Ok arm) — the round-83
    // Option→concrete unwrap must NOT fire on the default, or the arms
    // mismatch (`Option<String> | String`, getheader ×3).
    let out = compile(
        concat!(
            "from typing import MutableMapping\n",
            "class M(MutableMapping[str, str]):\n",
            "    def __getitem__(self, key: str) -> str:\n",
            "        return \"v\"\n",
            "    def __len__(self) -> int:\n",
            "        return 0\n",
            "    def __iter__(self):\n",
            "        return iter([])\n",
            "def f(m: M, default: str | None) -> str | None:\n",
            "    return m.get(\"k\", default)\n",
        ),
        "mapgetopt.py",
    );
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        flat.contains("let__rython_default=default;"),
        "the Option default must render as the Option itself, not the unwrap: {}",
        out
    );
    assert!(
        flat.contains("Ok(__rython_v)=>Some(__rython_v)"),
        "the Ok arm must Some-wrap to match the Option default: {}",
        out
    );
    assert!(
        !flat.contains("rython:anoptionalvaluewasNone"),
        "the Option default must NOT be unwrapped with the round-83 panic: {}",
        out
    );
}

#[test]
fn compiled_regex_groups_destructure_span_and_option_arg() {
    // Round 74: the compiled-regex match surface completes against real
    // urllib3 usage — m.span(i) routes to the group-indexed span, and a
    // truthiness-narrowed Option<String> argument unwraps before the
    // anchored match (CPython would raise TypeError on an actual None).
    let out = compile(
        "import re\n\
         _TARGET_RE = re.compile(r\"^([^?#]*)(?:\\?([^#]*))?.*$\")\n\
         def f(url: str) -> str:\n\
         \x20   m = _TARGET_RE.match(url)\n\
         \x20   path, query = m.groups()\n\
         \x20   return path + \"?\" + query\n\
         def g(host: str | None) -> str:\n\
         \x20   if host:\n\
         \x20       m = _TARGET_RE.match(host)\n\
         \x20       start, end = m.span(1)\n\
         \x20   return \"x\"\n",
        "regex2.py",
    );
    assert!(
        out.contains("span_group (1)"),
        "m.span(i) must route to the group-indexed span: {}",
        out
    );
    assert!(
        out.contains("unwrap_or_else") && out.contains("TypeError"),
        "a truthiness-narrowed Option<String> argument must unwrap before the match with Python's TypeError: {}",
        out
    );
}

#[test]
fn option_returning_functions_wrap_plain_members_and_narrowed_reads_unwrap() {
    // Round 74: a `-> T | None` function wraps its PLAIN returns in Some
    // (`return host.lower()` after the None guards), lowers `return None`
    // to the None member, and passes an already-Option value through
    // (`return host`); an Option-typed receiver SLICES through a loud
    // TypeError unwrap (`host[start:end]` after `if host:`); and an
    // `if v is not None:`-narrowed Option-typed name READS by unwrapping
    // (the PyValue as_str() path is for isinstance-narrowed boxed values
    // only).
    let out = compile(
        "import re\n\
         _ZONE = re.compile(r\"\\[(.*?)\\]\")\n\
         def normalize(host: str | None, scheme: str | None) -> str | None:\n\
         \x20   if host:\n\
         \x20       m = _ZONE.search(host)\n\
         \x20       if m is not None:\n\
         \x20           start, end = m.span(1)\n\
         \x20           return \"z:\" + host[start:end]\n\
         \x20       return host.lower()\n\
         \x20   return None\n\
         def show(v: str | None) -> str:\n\
         \x20   if v is not None:\n\
         \x20       return v\n\
         \x20   return \"none\"\n",
        "regex_opt.py",
    );
    assert!(
        out.contains("return Ok (Some (") || out.contains("return Ok(Some("),
        "plain members of an Option-returning function must wrap in Some: {}",
        out
    );
    assert!(
        out.contains("return Ok (None)") || out.contains("return Ok(None)"),
        "return None in an Option-returning function lowers to the None member: {}",
        out
    );
    assert!(
        out.contains("is not subscriptable"),
        "an Option-typed slice receiver unwraps with the TypeError panic: {}",
        out
    );
    assert!(
        out.contains("(v) . clone () . unwrap ()") || out.contains("(v).clone().unwrap()"),
        "an is-not-None-narrowed Option name reads by unwrapping: {}",
        out
    );
}

#[test]
fn imported_option_returning_callee_store_does_not_double_wrap() {
    // Round 75: a local assigned from an IMPORTED function whose return
    // annotation is `T | None` (`character_range = unicode_range(chunk)`
    // — charset_normalizer's cd.py, where utils.unicode_range returns
    // `str | None`) is itself the Option — the store must pass the
    // callee's result through, never Some-wrap it again
    // (Option<Option<String>>, the 19-error double-Option family).
    let utils = parse(
        "def unicode_range(character: str) -> str | None:\n\
         \x20   return character\n",
        "utils.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["utils".to_string()], std::rc::Rc::new(utils));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        ..Default::default()
    };
    let out = compile_with_options(
        "from .utils import unicode_range\n\
         def f(chunk: str) -> str | None:\n\
         \x20   character_range = unicode_range(chunk)\n\
         \x20   if character_range is not None:\n\
         \x20       return character_range\n\
         \x20   return None\n",
        "cd.py",
        options,
    )
    .expect("converts");
    assert!(
        !out.contains("Some (unicode_range") && !out.contains("Some(unicode_range"),
        "a store from an imported Option-returning callee must not Some-wrap: {}",
        out
    );
    assert!(
        out.contains("= unicode_range") && out.contains("return Ok (None)"),
        "the store passes the Option through and None lowers to the member: {}",
        out
    );
}

#[test]
fn lru_cache_optional_keys_stay_single_option() {
    // Round 75: an @lru_cache function with an Optional key parameter
    // (`lg_inclusion: Optional[str] = None` — charset_normalizer's
    // is_unicode_range_secondary) caches on the Option: the key type was
    // wrapped twice (Option<Option<String>>), breaking every cache hit
    // and miss path (the 34-error charset cluster).
    let out = compile(
        "from functools import lru_cache\n\
         @lru_cache(maxsize=2048)\n\
         def f(decoded: str, lg_inclusion: str | None = None) -> str | None:\n\
         \x20   return lg_inclusion\n",
        "lru_opt.py",
    );
    assert!(
        !out.contains("Option < Option < String > >") && !out.contains("Option<Option<String>>"),
        "an Optional cache key must stay a single Option: {}",
        out
    );
    assert!(
        out.contains("lg_inclusion : Option < String >") || out.contains("lg_inclusion: Option<String>"),
        "the key param is the single Option: {}",
        out
    );
}

#[test]
fn is_none_early_exit_guard_narrows_the_following_reads() {
    // Round 77: `if character_range is None: continue` (an is-None guard
    // whose body ALWAYS exits) narrows the FOLLOWING statements — they
    // are reachable only when the name is not None, so the reads unwrap
    // (`(character_range).clone().unwrap()`) instead of passing the
    // Option to a plain String parameter (charset_normalizer's
    // encoding_unicode_range: String: From<Option<String>>).
    let out = compile(
        "def opt() -> str | None:\n\
         \x20   return \"x\"\n\
         def f(chunk: str) -> bool:\n\
         \x20   character_range = opt()\n\
         \x20   if character_range is None:\n\
         \x20       return False\n\
         \x20   return \"a\" in character_range\n",
        "none_guard.py",
    );
    assert!(
        out.contains("clone () . unwrap ()") || out.contains("clone().unwrap()"),
        "an is-None early-exit guard must narrow the following reads: {}",
        out
    );
    assert!(
        !out.contains("From < Option"),
        "the narrowed read must not pass the raw Option to a plain parameter: {}",
        out
    );
}

#[test]
fn chained_is_none_and_none_assigning_else_do_not_narrow() {
    // Devin review on #282: the is-None early-exit narrowing must NOT
    // fire for a CHAINED compare (`x is None is y` — the single Compare
    // node carries two ops, whose truth depends on y) nor when the else
    // branch re-assigns the name (`else: x = None` — the following
    // statements can then see None).
    let out = compile(
        "def f(x: str | None, y: object) -> bool:\n\
         \x20   if x is None is y:\n\
         \x20       return False\n\
         \x20   return y is None\n\
         def g(x: str | None) -> str:\n\
         \x20   if x is None:\n\
         \x20       return \"a\"\n\
         \x20   else:\n\
         \x20       x = None\n\
         \x20   return \"b\"\n",
        "none_chain.py",
    );
    // The chained compare must not produce a narrowed x read (the unwrap
    // of an Option-typed x would be a compile error if x were narrowed
    // wrongly — assert the raw read shape instead).
    assert!(
        out.contains("py_is_none ()") || out.contains("py_is_none()"),
        "the guard compiles: {}",
        out
    );
    // Devin review on #283: a harmless `else: pass` keeps the narrowing —
    // the else writes nothing, so the following read of an Option-typed
    // x still unwraps.
    let out2 = compile(
        "def h(x: str | None) -> str:\n\
         \x20   if x is None:\n\
         \x20       return \"a\"\n\
         \x20   else:\n\
         \x20       pass\n\
         \x20   return x\n",
        "none_pass.py",
    );
    assert!(
        out2.contains("clone () . unwrap ()") || out2.contains("clone().unwrap()"),
        "an `else: pass` must keep the is-None narrowing: {}",
        out2
    );
    assert!(
        out.contains("x = None") || out.contains("x = None ;"),
        "the else branch re-assignment stays: {}",
        out
    );
    // Devin review on #284: a walrus in the else REBINDS the name
    // (`else: (x := None)`), so the narrowing must be discarded; an
    // attribute store (`else: x.attr = 1`) only mutates the object and
    // must KEEP it.
    let out3 = compile(
        "def w(x: str | None) -> str:\n\
         \x20   if x is None:\n\
         \x20       return \"a\"\n\
         \x20   else:\n\
         \x20       y = (x := None)\n\
         \x20   return \"b\"\n",
        "none_walrus.py",
    );
    assert!(
        !out3.contains("clone () . unwrap ()") && !out3.contains("clone().unwrap()"),
        "a walrus rebinding in the else must discard the narrowing: {}",
        out3
    );
    let out4 = compile(
        "def s(x: str | None) -> str:\n\
         \x20   if x is None:\n\
         \x20       return \"a\"\n\
         \x20   else:\n\
         \x20       x.attr = 1\n\
         \x20   return x\n",
        "none_attr.py",
    );
    assert!(
        out4.contains("clone () . unwrap ()") || out4.contains("clone().unwrap()"),
        "an attribute store in the else must keep the narrowing: {}",
        out4
    );
    // Devin review on #285: a walrus in an if/while TEST of the else
    // rebinds; a comprehension TARGET does not rebind the outer scope.
    let out5 = compile(
        "def t(x: str | None) -> str:\n\
         \x20   if x is None:\n\
         \x20       return \"a\"\n\
         \x20   else:\n\
         \x20       if (x := None):\n\
         \x20           pass\n\
         \x20   return \"b\"\n",
        "none_ifwalrus.py",
    );
    assert!(
        !out5.contains("clone () . unwrap ()") && !out5.contains("clone().unwrap()"),
        "a walrus in the else's if-test must discard the narrowing: {}",
        out5
    );
    let out6 = compile(
        "def c(x: str | None) -> str:\n\
         \x20   if x is None:\n\
         \x20       return \"a\"\n\
         \x20   else:\n\
         \x20       y = [z for z in (x, \"b\")]\n\
         \x20   return x\n",
        "none_comp.py",
    );
    assert!(
        out6.contains("clone () . unwrap ()") || out6.contains("clone().unwrap()"),
        "a comprehension target must keep the outer narrowing: {}",
        out6
    );
    // Round 78: a `-> list[str]` function's returning list LITERAL owns
    // its string literals (`return ["a", "b"]` lowers the elements), and
    // a module `Vec<String>` static with a list-literal init does too.
    let out_v = compile(
        "def langs() -> list[str]:\n\
         \x20   return [\"Latin\", \"Cyrillic\"]\n",
        "vecstr.py",
    );
    assert!(
        out_v.contains("to_string ()") || out_v.contains("to_string()"),
        "a returning list[str] literal must own its elements: {}",
        out_v
    );
    // Devin review on #286: a SPREAD of borrowed strings into the
    // returning list owns each spread element too.
    let out_vs = compile(
        "def langs(more: list[str]) -> list[str]:\n\
         \x20   return [\"a\", *more, \"b\"]\n",
        "vecspread.py",
    );
    assert!(
        out_vs.contains("into_iter () . map") || out_vs.contains("into_iter().map"),
        "a spread into a returning list[str] must own each element: {}",
        out_vs
    );
    // Devin review on #285 (2nd pass): a walrus in a def DEFAULT or a
    // class BASE in the else rebinds the guarded name.
    let out7 = compile(
        "def d(x: str | None) -> str:\n\
         \x20   if x is None:\n\
         \x20       return \"a\"\n\
         \x20   else:\n\
         \x20       def g(y=(x := None)):\n\
         \x20           return y\n\
         \x20   return \"b\"\n",
        "none_defdefault.py",
    );
    assert!(
        !out7.contains("clone () . unwrap ()") && !out7.contains("clone().unwrap()"),
        "a walrus in a nested def default must discard the narrowing: {}",
        out7
    );
}

#[test]
fn abstract_stub_call_with_defaultable_missing_args_dispatches_virtually() {
    // Round 79: `self.read(len(b))` inside a base's readinto where the
    // base read is a `raise NotImplementedError()` stub — the call was
    // DROPPED (boxed None) because the stub's arity exceeded the call's.
    // The stub's missing params all have defaults, so the call maps to
    // the full-arity invocation, which dispatches virtually to the
    // derived override (urllib3's BaseHTTPResponse.readinto →
    // HTTPResponse.read).
    let out = compile(
        "class Base:\n\
         \x20   def read(self, amt: int | None = None, decode_content: bool | None = None) -> bytes:\n\
         \x20       raise NotImplementedError()\n\
         \x20   def readinto(self, b: bytearray) -> int:\n\
         \x20       temp = self.read(len(b))\n\
         \x20       if len(temp) == 0:\n\
         \x20           return 0\n\
         \x20       b[: len(temp)] = temp\n\
         \x20       return len(temp)\n",
        "mro_stub.py",
    );
    assert!(
        !out.contains("PyValue :: None_") && !out.contains("PyValue::None_"),
        "the stub call must not drop to a boxed None: {}",
        out
    );
    assert!(
        out.contains("(self) . read (Some (len (& (b)) as i64) , None) ")
            || out.contains("(self).read(Some(len(&(b)) as i64), None)"),
        "the stub call must map to the full-arity virtual invocation: {}",
        out
    );
}

#[test]
fn reused_name_slice_assign_clones_the_value() {
    // Round 79: `b[:len(temp)] = temp; return len(temp)` — the
    // slice-assign MOVED temp into the receiver; the later read was a
    // use-after-move (E0382). A reused Name value now clones.
    let out = compile(
        "def f(b: bytearray, temp: bytes) -> int:\n\
         \x20   b[: len(temp)] = temp\n\
         \x20   return len(temp)\n",
        "slice_reuse.py",
    );
    assert!(
        out.contains("clone ()") || out.contains("clone()"),
        "a reused slice-assign value must clone: {}",
        out
    );
}

#[test]
fn and_chain_truthiness_narrows_the_later_operand_reads() {
    // Round 81 (the generics directive): `if conn and
    // is_connection_dropped(conn):` — the first conjunct proves the
    // Option-typed name non-None, so the SECOND conjunct's read of conn
    // unwraps (`(conn).clone().unwrap()` — the PyValue inner), where the
    // pre-round-81 output passed the raw Option and failed in rustc.
    let out = compile(
        "def drop(c) -> bool:\n\
         \x20   return False\n\
         def f(conn: bytes | None) -> bool:\n\
         \x20   if conn and drop(conn):\n\
         \x20       return True\n\
         \x20   return False\n",
        "and_narrow.py",
    );
    assert!(
        out.contains("conn) . clone () . unwrap ()")
            || out.contains("(conn).clone().unwrap()"),
        "the and-chain must narrow the later operand's reads: {}",
        out
    );
    assert!(
        !out.contains("drop (conn)") && !out.contains("drop(conn)"),
        "the later operand must NOT receive the raw Option: {}",
        out
    );
}

#[test]
fn boxed_value_return_in_a_typed_fn_converts_via_into() {
    // Round 81 (the generics directive): a `-> bytes` function returning
    // a local that was assigned a DROPPED call (`decompressed =
    // self._obj.decompress(data)` — the zlib receiver is boxed) has a
    // PyValue binding; the return converts via the reverse From<PyValue>
    // impl (`(decompressed).into()`) instead of leaving an E0308. The
    // conversion is LOUD on a wrong member (Python fails at use, rython
    // at the conversion).
    let out = compile(
        "class D:\n\
         \x20   def __init__(self):\n\
         \x20       self._obj = None\n\
         \x20   def decompress(self, data: bytes) -> bytes:\n\
         \x20       decompressed = self._obj.decompress(data)\n\
         \x20       if decompressed:\n\
         \x20           self._first = False\n\
         \x20       return decompressed\n",
        "dropped_ret.py",
    );
    assert!(
        out.contains("decompressed) . into ()") || out.contains("(decompressed).into()"),
        "the boxed-value return must convert via .into(): {}",
        out
    );
}

#[test]
fn boxed_argument_into_a_concrete_optional_slot_converts_the_inner() {
    // Round 81 (the generics directive): `create_urllib3_context(
    // cert_reqs=resolve_cert_reqs(cert_reqs))` — the callee's boxed
    // return feeds an `int | None` param (`Option<i64>`). The argument
    // wraps in Some AND converts the inner (`(v).into()`) — a loud
    // TypeError on a wrong member, never `Some(PyValue)` against
    // `Option<i64>` (E0308).
    let out = compile(
        "from typing import Any\n\
         def resolve_cert_reqs(candidate: int | None) -> Any:\n\
         \x20   if candidate is None:\n\
         \x20       return 2\n\
         \x20   return candidate\n\
         def create_urllib3_context(cert_reqs: int | None = None) -> None:\n\
         \x20   pass\n\
         def make_context(cert_reqs: int | None) -> None:\n\
         \x20   create_urllib3_context(cert_reqs=resolve_cert_reqs(cert_reqs))\n",
        "opt_inner.py",
    );
    assert!(
        (out.contains("Some ({") || out.contains("Some({"))
            && (out.contains("__rython_v) . into ()") || out.contains("(__rython_v).into()")),
        "the boxed argument must wrap in Some and convert the inner: {}",
        out
    );
}

#[test]
fn reused_boxed_name_store_clones_out_of_the_shared_value() {
    // Round 81: `context = ssl_context` then `elif ssl_context is None:`
    // — the boxed PyValue name is read AGAIN after the store; the store
    // must CLONE (the Arc reference copy, Python semantics) or the move
    // poisons the later read (E0382, newly exposed by the round-81
    // coerce fixes).
    let out = compile(
        "from typing import Any\n\
         def wrap(ssl_context: Any) -> Any:\n\
         \x20   context = ssl_context\n\
         \x20   if context is None:\n\
         \x20       return ssl_context\n\
         \x20   return context\n",
        "reused_boxed.py",
    );
    assert!(
        out.contains("context = (ssl_context) . clone ()")
            || out.contains("context = (ssl_context).clone()"),
        "a reused boxed name store must clone: {}",
        out
    );
}

#[test]
fn external_module_return_annotation_types_the_function_as_boxed() {
    // Round 82: `-> ssl.SSLSocket` (an external-module class annotation)
    // previously resolved to NOTHING — the function silently typed `()`
    // while its body returned a value, so every caller of the return
    // mismatched (the `() | PyValue` family). The symbols-aware authority
    // now resolves the annotation to the boxed PyValue (the external-object
    // divergence), so the signature carries the value.
    let out = compile(
        "import ssl\n\
         def wrap_socket(sock) -> ssl.SSLSocket:\n\
         \x20   return sock\n",
        "ssl_ret.py",
    );
    assert!(
        out.contains("-> Result < stdpython :: PyValue , PyException >")
            || out.contains("Result<stdpython::PyValue, PyException>"),
        "the external-module return annotation must box the return: {}",
        out
    );
}

#[test]
fn boxed_value_stored_into_a_concrete_inherited_field_converts() {
    // Round 82: `self.is_verified = sock_and_verified.is_verified` — a
    // boxed namedtuple member stored into the bool field inherited from
    // the BASE class (HTTPSConnection → HTTPConnection). The field type is
    // the BASE-MOST owner's (bool, the struct ground truth), not the
    // derived class's own boxed-join; the store converts via `.into()`.
    let out = compile(
        "class Base:\n\
         \x20   def __init__(self):\n\
         \x20       self.is_verified = False\n\
         class Child(Base):\n\
         \x20   def __init__(self, info):\n\
         \x20       super().__init__()\n\
         \x20       self.is_verified = info.is_verified\n",
        "inherit_field.py",
    );
    assert!(
        !out.contains("PyValue :: from (info . is_verified)") && !out.contains("PyValue::from(info.is_verified)"),
        "a boxed value into a concrete inherited field must not box-wrap: {}",
        out
    );
}

// The binder authority (issue #137, the round-99 evaluation's drift 3): a
// comprehension target, a `key=` lambda parameter, and a constructor call
// used directly as a receiver are typed the way a for-statement target
// is, so a user method call on them resolves its receiver's class and
// emits the `?` its Result needs. Before, the untyped receiver fell to
// the generic arm, the Result leaked, and rustc rejected it downstream
// (the idiom corpus's shapes: six of its eleven errors).
fn has_fallible_area_call(out: &str) -> bool {
    out.contains(". area () ?") || out.contains(".area()?")
}

#[test]
fn a_comprehension_target_over_a_class_list_types_its_method_calls() {
    let out = compile(
        concat!(
            "class Shape:\n",
            "    def area(self) -> float:\n",
            "        return 1.0\n",
            "\n",
            "def total(shapes: list[Shape]) -> float:\n",
            "    return sum([s.area() for s in shapes])\n",
        ),
        "comp_binder.py",
    );
    assert!(
        has_fallible_area_call(&out),
        "the comprehension target must resolve the class so the call propagates: {}",
        out
    );
}

#[test]
fn a_key_lambda_parameter_is_the_iterables_element() {
    let out = compile(
        concat!(
            "class Shape:\n",
            "    def area(self) -> float:\n",
            "        return 1.0\n",
            "\n",
            "def biggest(shapes: list[Shape]) -> Shape:\n",
            "    return max(shapes, key=lambda s: s.area())\n",
        ),
        "key_binder.py",
    );
    assert!(
        has_fallible_area_call(&out),
        "the key lambda's parameter must resolve the class so the call propagates: {}",
        out
    );
}

#[test]
fn a_constructor_call_as_receiver_resolves_its_class() {
    let out = compile(
        concat!(
            "class Shape:\n",
            "    def area(self) -> float:\n",
            "        return 1.0\n",
            "\n",
            "def main() -> None:\n",
            "    print(Shape().area())\n",
        ),
        "ctor_receiver.py",
    );
    assert!(
        has_fallible_area_call(&out),
        "a constructor call used as the receiver must resolve its class: {}",
        out
    );
}

#[test]
fn a_comprehension_element_type_is_the_body_type_in_the_comprehension_scope() {
    // The type side agrees with the lowering: `[s.name() for s in shapes]`
    // is a Vec<String>, so a later `"-".join(names)` sees strings, not an
    // unknown element.
    let out = compile(
        concat!(
            "class Shape:\n",
            "    def name(self) -> str:\n",
            "        return \"shape\"\n",
            "\n",
            "def names(shapes: list[Shape]) -> str:\n",
            "    ns = [s.name() for s in shapes]\n",
            "    return \"-\".join(ns)\n",
        ),
        "comp_type.py",
    );
    assert!(
        out.contains(". name () ?") || out.contains(".name()?"),
        "the comprehension body call must propagate: {}",
        out
    );
}

#[test]
fn a_local_bound_to_a_comprehension_carries_its_element_type() {
    // `squares = [s for s in shapes if ...]` is a list of shapes' element,
    // so a later comprehension over `squares` resolves the receiver and
    // propagates the method call's Result.
    let out = compile(
        concat!(
            "class Shape:\n",
            "    def name(self) -> str:\n",
            "        return \"shape\"\n",
            "    def big(self) -> bool:\n",
            "        return True\n",
            "\n",
            "def names(shapes: list[Shape]) -> list[str]:\n",
            "    bigs = [s for s in shapes if s.big()]\n",
            "    return [s.name() for s in bigs]\n",
        ),
        "comp_local.py",
    );
    assert!(
        out.contains(". name () ?") || out.contains(".name()?"),
        "the comprehension-bound local must carry the element type: {}",
        out
    );
}

#[test]
fn the_analysis_narrows_an_isinstance_branch_like_the_lowering() {
    // idna's ulabel: the else branch of `if not isinstance(label, bytes)`
    // reads the `str | bytes` parameter as bytes, so the alias
    // `label_bytes = label` is bytes — not the parameter's boxed union,
    // which would box every later byte operation (E0599 on the boxed
    // startswith/lower/slice).
    let out = compile(
        concat!(
            "def ulabel(label: str | bytes) -> str:\n",
            "    if not isinstance(label, bytes):\n",
            "        label_bytes = label.encode(\"ascii\")\n",
            "    else:\n",
            "        label_bytes = label\n",
            "    label_bytes = label_bytes.lower()\n",
            "    return label_bytes.decode(\"ascii\")\n",
        ),
        "narrow_analysis.py",
    );
    assert!(
        !out.contains("PyValue :: from") && !out.contains("PyValue::from"),
        "the alias must be typed by the narrowed branch, not boxed: {}",
        out
    );
}

#[test]
fn a_nested_comprehension_binds_each_generator_in_its_prefix_scope() {
    // Devin review on #318: generator i's iterable sees only the targets
    // bound before it, so `for s in group` resolves `group` from the outer
    // generator, and `s` types from it — the inner call propagates.
    let out = compile(
        concat!(
            "class Shape:\n",
            "    def name(self) -> str:\n",
            "        return \"shape\"\n",
            "\n",
            "def names(groups: list[list[Shape]]) -> list[str]:\n",
            "    return [s.name() for group in groups for s in group]\n",
        ),
        "nested_comp.py",
    );
    assert!(
        out.contains(". name () ?") || out.contains(".name()?"),
        "the inner generator's target must type from the outer target: {}",
        out
    );
}

#[test]
fn a_write_nested_under_a_condition_in_a_narrowed_branch_keeps_the_union() {
    // Devin review on #318: only a DEFINITE reassignment (every fall-through
    // path of both branches) retypes the name after the if; a write nested
    // under a further condition leaves the union in place on the other
    // path, so post-if reads stay boxed.
    let out = compile(
        concat!(
            "def norm(label: str | bytes, flag: bool) -> bool:\n",
            "    if isinstance(label, bytes):\n",
            "        if flag:\n",
            "            label = label.decode(\"ascii\")\n",
            "    else:\n",
            "        label = label.upper()\n",
            "    return isinstance(label, bytes)\n",
        ),
        "branch_partial_reassign.py",
    );
    // The post-if `isinstance` is a second runtime test on the union.
    assert!(
        out.matches("is_bytes ()").count() == 2,
        "a conditional write must not retype the name after the if: {}",
        out
    );
}

/// Devin review on #318: a factory imported through the package's own
/// root-qualified path (`from pkg.session import make`, keyed
/// ["session"]) resolves as a receiver when its result is used directly
/// (`make().run()`): the module key authority normalizes the path.
#[test]
fn a_root_qualified_imported_factory_result_is_a_method_receiver() {
    let session = parse(
        concat!(
            "class Session:\n",
            "    def run(self) -> int:\n",
            "        return 42\n",
            "\n",
            "def make() -> Session:\n",
            "    return Session()\n",
        ),
        "session.py",
    )
    .unwrap();
    let mut defs = std::collections::HashMap::new();
    defs.insert(vec!["session".to_string()], std::rc::Rc::new(session));
    let options = PythonOptions {
        module_defs: std::rc::Rc::new(defs),
        python_namespace: "pkg".to_string(),
        ..Default::default()
    };
    let usemod = parse(
        "from pkg.session import make\n\ndef go() -> int:\n    return make().run()\n",
        "usemod.py",
    )
    .unwrap();
    let symbols = usemod.clone().find_symbols(SymbolTableScopes::new());
    let out = usemod
        .to_rust(CodeGenContext::Module("usemod".to_string()), options, symbols)
        .unwrap()
        .to_string();
    assert!(
        out.contains(". run () ?"),
        "the factory's result must resolve its class and the call propagate: {}",
        out
    );
}

/// A comprehension target that reuses a NARROWED outer name (Devin
/// review on #318): the target is a fresh binding (Python 3 scopes the
/// comprehension), so the enclosing `if x is None: return` narrowing —
/// whose reads unwrap the Option parameter — must not apply to it.
#[test]
fn a_comprehension_target_shadowing_a_narrowed_name_is_a_fresh_binding() {
    let out = compile(
        "def f(x: int | None, xs: list[int]) -> list[int]:\n    if x is None:\n        return []\n    return [x * 2 for x in xs]\n",
        "shadow_comp.py",
    );
    assert!(
        !out.contains("unwrap"),
        "the fresh target reads plainly, never through the outer Option's unwrap: {}",
        out
    );
}

/// A key lambda's parameter that reuses a narrowed outer name is likewise
/// a fresh binding.
#[test]
fn a_key_lambda_parameter_shadowing_a_narrowed_name_is_a_fresh_binding() {
    let out = compile(
        "def f(s: str | None, xs: list[str]) -> list[str]:\n    if s is None:\n        return []\n    return sorted(xs, key=lambda s: s.lower())\n",
        "shadow_lambda.py",
    );
    assert!(
        !out.contains("unwrap"),
        "the fresh parameter reads plainly, never through the outer Option's unwrap: {}",
        out
    );
}
