# A Rust program with a module written in Python

This is an ordinary cargo binary crate whose geometry code lives in
[`src/geometry.py`](src/geometry.py) — real Python with classes, single
inheritance, `super()`, dynamically-dispatched method overrides, f-strings,
and three functions (`scale`, `clamp`, `lerp`) with **no type annotations
at all** — their generic Rust signatures are inferred from how the
parameters are used.

The one line that makes it work is in [`src/main.rs`](src/main.rs):

```rust
python_module!(geometry);
```

At build time the `python-mod` proc-macro reads `src/geometry.py`, compiles
it to Rust with the `python-ast` codegen, and splices the result in as a
regular Rust module. There is no interpreter, no GIL, and no Python
installation needed at runtime — `geometry::Rectangle` is a plain Rust
struct, and calling it costs the same as any other Rust code.

## Run it

```bash
cargo run
```

Expected output:

```
rectangle: area=12.0 perimeter=14.0
circle: area=3.141592653589793 perimeter=6.283185307179586
scale(2.5, 4.0)      = 10
scale(6, 7)          = 42
scale("na", 4)       = nananana
scale([1, 2], 3)     = [1, 2, 1, 2, 1, 2]
clamp(12, 0, 10)     = 10
clamp(0.2, 0.5, 2.0) = 0.5
clamp("m", "a", "f")  = f
lerp(0.0, 10.0, 0.25) = 2.5
lerp(100, 200, 2)     = 300
```

The module is still ordinary Python, so the same file runs under CPython —
which is exactly how you check the two agree:

```bash
cd src && python3 -c "
import geometry as g
print(g.Rectangle(3.0, 4.0).describe())
print(g.Circle(1.0).describe())"
```

## What the Python compiles to

- `class Shape:` becomes a Rust struct plus a `ShapeTrait` carrying its
  methods. `class Rectangle(Shape):` embeds a `Shape` and implements
  `ShapeTrait`, so inherited methods (like `describe`) and overridden ones
  (like `area`) resolve just as they do under CPython — `describe`'s
  internal `self.area()` call dispatches to the subclass override.
- `Rectangle(3.0, 4.0)` in Python is `Rectangle::new(3.0, 4.0)?` from Rust.
  Methods that can raise return `Result<T, PyException>`, so Python
  exceptions become ordinary Rust error handling.
- **Inferred generics, three flavors.** None of `scale`, `clamp`, `lerp`
  are annotated; rython derives the weakest trait bounds their bodies
  imply:
  - `scale(value, factor)` gets a `PyMul` bound from `value * factor`, so
    the one function multiplies floats, ints, strings (`"na" * 4`), and
    lists (`[1, 2] * 3`) — each call monomorphized by rustc.
  - `clamp(value, low, high)` can return any of its three parameters, so
    inference unifies them into one variable:
    `clamp<T>(value: T, low: T, high: T) -> Result<T, _>` — called above
    with ints, floats, and Strings.
  - `lerp(start, end, t)` chains three operators; the signature carries
    the intermediate `Output` bounds
    (`<B as PySub<A>>::Output: PyMul<C>`, ...) that make the composed
    return type well-formed.

## Notes

- The macro looks for `src/<name>.py` (or `src/<name>/__init__.py`) in the
  crate that invokes it.
- Only the rython subset of Python is accepted; anything outside it is a
  **compile-time** error pointing at the construct, never silently
  different behavior. See [`docs/spec.md`](../../docs/spec.md).
- `python_module!{name /* rust items */}` also accepts a Rust preamble to
  make `use` items available to hand-written code in the same module; see
  [`crates/python-mod`](../../crates/python-mod).
