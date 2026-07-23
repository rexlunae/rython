//! Python heapq module implementation
//!
//! Min-heaps on plain Vecs, ported from CPython's heapq.py algorithms so
//! the LIST LAYOUT after every operation is identical to Python's — the
//! heap is an observable list in Python, not an opaque structure.
//! Comparisons use `<` exactly as CPython does (so NaN behaves the same
//! way it does in Python: `<` is simply false).

use crate::PyException;
use alloc::vec::Vec;

fn lt<T: PartialOrd>(a: &T, b: &T) -> bool {
    a < b
}

/// CPython's _siftdown: bubble heap[pos] toward the root.
fn siftdown<T: PartialOrd + Clone>(heap: &mut [T], startpos: usize, mut pos: usize) {
    let newitem = heap[pos].clone();
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        if lt(&newitem, &heap[parentpos]) {
            heap[pos] = heap[parentpos].clone();
            pos = parentpos;
            continue;
        }
        break;
    }
    heap[pos] = newitem;
}

/// CPython's _siftup: move the smaller child up until a leaf, then place
/// newitem and bubble it down.
fn siftup<T: PartialOrd + Clone>(heap: &mut [T], mut pos: usize) {
    let endpos = heap.len();
    let startpos = pos;
    let newitem = heap[pos].clone();
    let mut childpos = 2 * pos + 1;
    while childpos < endpos {
        let rightpos = childpos + 1;
        if rightpos < endpos && !lt(&heap[childpos], &heap[rightpos]) {
            childpos = rightpos;
        }
        heap[pos] = heap[childpos].clone();
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    siftdown(heap, startpos, pos);
}

/// heapq.heapify(x): in-place, O(n), CPython's exact traversal order.
pub fn heapify<T: PartialOrd + Clone>(heap: &mut Vec<T>) {
    let n = heap.len();
    for i in (0..n / 2).rev() {
        siftup(heap, i);
    }
}

/// heapq.heappush(heap, item)
pub fn heappush<T: PartialOrd + Clone>(heap: &mut Vec<T>, item: T) {
    heap.push(item);
    let last = heap.len() - 1;
    siftdown(heap, 0, last);
}

/// heapq.heappop(heap): the smallest item; IndexError on an empty heap,
/// with Python's message.
pub fn heappop<T: PartialOrd + Clone>(heap: &mut Vec<T>) -> Result<T, PyException> {
    let lastelt = heap
        .pop()
        .ok_or_else(|| PyException::new("IndexError", "index out of range"))?;
    if heap.is_empty() {
        return Ok(lastelt);
    }
    let returnitem = core::mem::replace(&mut heap[0], lastelt);
    siftup(heap, 0);
    Ok(returnitem)
}

/// heapq.heappushpop(heap, item): push then pop, faster and — like
/// Python — returning item itself when it's <= the heap's smallest.
pub fn heappushpop<T: PartialOrd + Clone>(heap: &mut Vec<T>, mut item: T) -> T {
    if !heap.is_empty() && lt(&heap[0], &item) {
        core::mem::swap(&mut heap[0], &mut item);
        siftup(heap, 0);
    }
    item
}

/// heapq.heapreplace(heap, item): pop then push; IndexError on empty.
pub fn heapreplace<T: PartialOrd + Clone>(heap: &mut Vec<T>, item: T) -> Result<T, PyException> {
    if heap.is_empty() {
        return Err(PyException::new("IndexError", "index out of range"));
    }
    let returnitem = core::mem::replace(&mut heap[0], item);
    siftup(heap, 0);
    Ok(returnitem)
}

/// heapq.nlargest(n, iterable): equivalent to sorted(iterable,
/// reverse=True)[:n], which is exactly how it's documented.
pub fn nlargest<T: PartialOrd + Clone>(n: usize, iterable: &[T]) -> Vec<T> {
    let mut sorted = crate::sorted_reverse(iterable, true);
    sorted.truncate(n);
    sorted
}

/// heapq.nsmallest(n, iterable): equivalent to sorted(iterable)[:n].
pub fn nsmallest<T: PartialOrd + Clone>(n: usize, iterable: &[T]) -> Vec<T> {
    let mut sorted = crate::sorted(iterable);
    sorted.truncate(n);
    sorted
}
