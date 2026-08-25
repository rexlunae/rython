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

    // `scale` has no annotations - rython inferred a generic signature
    // from `value * factor`, so one Python function serves several types.
    println!("scale(2.5, 4.0) = {}", scale(2.5, 4.0)?);
    println!("scale(6, 7)     = {}", scale(6, 7)?);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
