use python_mod::python_module;
use stdpython::PyException;

// Compile `src/geometry.py` into a Rust module at build time. The Python
// classes become Rust structs (inheritance becomes trait-based embedding),
// and every function/method is an ordinary Rust item we can call below.
python_module!(geometry);

// The traits carry the inherited/overridable methods (`describe`, `area`,
// ...), so bring the whole module surface into scope like any Rust module.
use geometry::*;

fn run() -> Result<(), PyException> {
    // Python: rect = Rectangle(3.0, 4.0)
    let rect = Rectangle::new(3.0, 4.0)?;
    // `describe` is inherited from Shape; the `self.area()` inside it
    // dispatches to Rectangle's override, exactly like CPython.
    println!("{}", rect.describe()?);

    let circle = Circle::new(1.0)?;
    println!("{}", circle.describe()?);

    // None of the functions below have type annotations - rython infers
    // generic Rust signatures from how the parameters are used, so each
    // Python function is ONE definition serving many types.

    // `value * factor` covers everything Python's `*` covers:
    println!("scale(2.5, 4.0)      = {}", scale(2.5, 4.0)?);
    println!("scale(6, 7)          = {}", scale(6, 7)?);
    println!("scale(\"na\", 4)       = {}", scale("na".to_string(), 4)?);
    println!("scale([1, 2], 3)     = {:?}", scale(vec![1, 2], 3)?);

    // `clamp` returns any of its three parameters, so inference unifies
    // them into a single type variable T:
    println!("clamp(12, 0, 10)     = {}", clamp(12, 0, 10)?);
    println!("clamp(0.2, 0.5, 2.0) = {}", clamp(0.2, 0.5, 2.0)?);
    println!(
        "clamp(\"m\", \"a\", \"f\")  = {}",
        clamp("m".to_string(), "a".to_string(), "f".to_string())?
    );

    // `lerp` chains three operators; the inferred signature carries the
    // intermediate `Output` bounds, so floats and ints both work:
    println!("lerp(0.0, 10.0, 0.25) = {}", lerp(0.0, 10.0, 0.25)?);
    println!("lerp(100, 200, 2)     = {}", lerp(100, 200, 2)?);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
