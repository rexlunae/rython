//! Python copy module implementation
//!
//! In rython's value model every lowered type owns its data, so both
//! copy() and deepcopy() are a full clone. deepcopy() is exact; copy()'s
//! Python contract (nested containers stay SHARED with the original) has
//! no aliasing to preserve here — rython values never alias — so the
//! observable difference cannot arise inside generated code.

/// copy.deepcopy(x)
pub fn deepcopy<T: Clone>(x: &T) -> T {
    x.clone()
}

/// copy.copy(x)
pub fn copy<T: Clone>(x: &T) -> T {
    x.clone()
}
