//! A loud limit on lowering depth (issue #354).
//!
//! Lowering recurses once per AST node, so a deeply nested expression can
//! walk off the end of the thread's stack — a `SIGABRT` that names no file,
//! no line and no construct. A caller that knows its stack gives the
//! lowering a BUDGET with [`with_stack_budget`]; every expression and
//! statement lowering then checks how much of it is spent and, past the
//! budget, fails the conversion with an error naming the construct
//! instead. Without a budget nothing is checked.
//!
//! The measure is the distance between the address of a local at the
//! budget's entry and one at the check: the stack grows downward on every
//! target rython supports.

use std::cell::Cell;

thread_local! {
    /// (base address, budget in bytes) for this thread, when set.
    static GUARD: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// Restores the previous budget when the call returns or unwinds.
struct Restore(Option<(usize, usize)>);

impl Drop for Restore {
    fn drop(&mut self) {
        GUARD.with(|g| g.set(self.0));
    }
}

/// Run `f` with at most `budget` bytes of stack for lowering, counted from
/// here. Leave the rest of the thread's stack as headroom for the work
/// between two checks (type analyses recurse too).
#[inline(never)]
pub fn with_stack_budget<R>(budget: usize, f: impl FnOnce() -> R) -> R {
    let marker = 0u8;
    let base = std::hint::black_box(&marker) as *const u8 as usize;
    let _restore = Restore(GUARD.with(|g| g.replace(Some((base, budget)))));
    f()
}

/// Whether this thread's lowering has spent its budget.
#[inline(never)]
pub(crate) fn check() -> Result<(), Box<dyn std::error::Error>> {
    let Some((base, budget)) = GUARD.with(|g| g.get()) else {
        return Ok(());
    };
    let marker = 0u8;
    let here = std::hint::black_box(&marker) as *const u8 as usize;
    if base.saturating_sub(here) > budget {
        return Err(format!(
            "this code nests too deeply for rython to lower: converting it would \
             need more than the {} MiB of stack the conversion is given. Split the \
             nesting up — bind a long chained expression's parts to intermediate \
             variables, or flatten deeply nested calls, literals or blocks. rython \
             refuses rather than overflowing the stack",
            budget / (1024 * 1024)
        )
        .into());
    }
    Ok(())
}
