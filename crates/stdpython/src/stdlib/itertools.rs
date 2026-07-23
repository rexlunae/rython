//! Python itertools module implementation
//! 
//! This module provides functions creating iterators for efficient looping.
//! Implementation matches Python's itertools module API.

use alloc::collections::VecDeque;
use alloc::{vec, vec::Vec};

/// count - infinite iterator starting from start
#[derive(Debug, Clone)]
pub struct Count<T> {
    current: T,
    step: T,
}

impl<T> Count<T> 
where
    T: Clone + core::ops::Add<Output = T>,
{
    /// Create new count iterator
    pub fn new(start: T, step: T) -> Self {
        Self {
            current: start,
            step,
        }
    }
    
    /// Get next value
    pub fn next(&mut self) -> T {
        let result = self.current.clone();
        self.current = self.current.clone() + self.step.clone();
        result
    }
    
    /// Take n values from count
    pub fn take(&mut self, n: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            result.push(self.next());
        }
        result
    }
}

/// cycle - infinite iterator cycling through iterable
#[derive(Debug, Clone)]
pub struct Cycle<T> {
    items: Vec<T>,
    index: usize,
}

impl<T> Cycle<T> 
where
    T: Clone,
{
    /// Create new cycle iterator
    pub fn new<I>(iterable: I) -> Self 
    where
        I: IntoIterator<Item = T>,
    {
        Self {
            items: iterable.into_iter().collect(),
            index: 0,
        }
    }
    
    /// Get next value (cycles through items)
    pub fn next(&mut self) -> Option<T> {
        if self.items.is_empty() {
            return None;
        }
        
        let result = self.items[self.index].clone();
        self.index = (self.index + 1) % self.items.len();
        Some(result)
    }
    
    /// Take n values from cycle
    pub fn take(&mut self, n: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(item) = self.next() {
                result.push(item);
            }
        }
        result
    }
}

/// repeat - infinite iterator returning same value
#[derive(Debug, Clone)]
pub struct Repeat<T> {
    item: T,
    times: Option<usize>,
    count: usize,
}

impl<T> Repeat<T> 
where
    T: Clone,
{
    /// Create infinite repeat iterator
    pub fn infinite(item: T) -> Self {
        Self {
            item,
            times: None,
            count: 0,
        }
    }
    
    /// Create repeat iterator with limit
    pub fn times(item: T, times: usize) -> Self {
        Self {
            item,
            times: Some(times),
            count: 0,
        }
    }
    
    /// Get next value
    pub fn next(&mut self) -> Option<T> {
        if let Some(limit) = self.times {
            if self.count >= limit {
                return None;
            }
        }
        
        self.count += 1;
        Some(self.item.clone())
    }
    
    /// Take n values from repeat
    pub fn take(&mut self, n: usize) -> Vec<T> {
        let mut result = Vec::new();
        for _ in 0..n {
            if let Some(item) = self.next() {
                result.push(item);
            } else {
                break;
            }
        }
        result
    }
}

/// chain - iterator chaining multiple iterables
#[derive(Debug)]
pub struct Chain<T> {
    iterables: VecDeque<Vec<T>>,
    current: usize,
}

impl<T> Chain<T> 
where
    T: Clone,
{
    /// Create new chain iterator
    pub fn new() -> Self {
        Self {
            iterables: VecDeque::new(),
            current: 0,
        }
    }
    
    /// Add iterable to chain
    pub fn add<I>(&mut self, iterable: I) 
    where
        I: IntoIterator<Item = T>,
    {
        self.iterables.push_back(iterable.into_iter().collect());
    }
    
    /// Get next value from chain
    pub fn next(&mut self) -> Option<T> {
        while !self.iterables.is_empty() {
            if let Some(current_vec) = self.iterables.front_mut() {
                if self.current < current_vec.len() {
                    let result = current_vec.get(self.current).cloned();
                    self.current += 1;
                    return result;
                } else {
                    self.iterables.pop_front();
                    self.current = 0;
                }
            } else {
                break;
            }
        }
        None
    }
    
    /// Collect all remaining items
    pub fn collect(mut self) -> Vec<T> {
        let mut result = Vec::new();
        while let Some(item) = self.next() {
            result.push(item);
        }
        result
    }
}

impl<T> Default for Chain<T> 
where
    T: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// islice - iterator slice with start, stop, step
#[derive(Debug)]
pub struct ISlice<T> {
    items: Vec<T>,
    stop: Option<usize>,
    step: usize,
    current: usize,
}

impl<T> ISlice<T> 
where
    T: Clone,
{
    /// Create islice iterator
    pub fn new<I>(iterable: I, start: usize, stop: Option<usize>, step: usize) -> Self 
    where
        I: IntoIterator<Item = T>,
    {
        Self {
            items: iterable.into_iter().collect(),
            stop,
            step: step.max(1), // Ensure step is at least 1
            current: start,
        }
    }
    
    /// Get next value from slice
    pub fn next(&mut self) -> Option<T> {
        if self.current >= self.items.len() {
            return None;
        }
        
        if let Some(stop) = self.stop {
            if self.current >= stop {
                return None;
            }
        }
        
        let result = self.items.get(self.current).cloned();
        self.current += self.step;
        result
    }
    
    /// Collect all remaining items
    pub fn collect(mut self) -> Vec<T> {
        let mut result = Vec::new();
        while let Some(item) = self.next() {
            result.push(item);
        }
        result
    }
}

/// takewhile - iterator yielding items while predicate is true
#[derive(Debug)]
pub struct TakeWhile<T, F> {
    items: Vec<T>,
    predicate: F,
    index: usize,
    stopped: bool,
}

impl<T, F> TakeWhile<T, F> 
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    /// Create takewhile iterator
    pub fn new<I>(iterable: I, predicate: F) -> Self 
    where
        I: IntoIterator<Item = T>,
    {
        Self {
            items: iterable.into_iter().collect(),
            predicate,
            index: 0,
            stopped: false,
        }
    }
    
    /// Get next value while predicate is true
    pub fn next(&mut self) -> Option<T> {
        if self.stopped || self.index >= self.items.len() {
            return None;
        }
        
        if let Some(item) = self.items.get(self.index) {
            if (self.predicate)(item) {
                self.index += 1;
                Some(item.clone())
            } else {
                self.stopped = true;
                None
            }
        } else {
            None
        }
    }
    
    /// Collect all remaining items
    pub fn collect(mut self) -> Vec<T> {
        let mut result = Vec::new();
        while let Some(item) = self.next() {
            result.push(item);
        }
        result
    }
}

/// dropwhile - iterator dropping items while predicate is true
#[derive(Debug)]
pub struct DropWhile<T, F> {
    items: Vec<T>,
    predicate: F,
    index: usize,
    started: bool,
}

impl<T, F> DropWhile<T, F> 
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    /// Create dropwhile iterator
    pub fn new<I>(iterable: I, predicate: F) -> Self 
    where
        I: IntoIterator<Item = T>,
    {
        Self {
            items: iterable.into_iter().collect(),
            predicate,
            index: 0,
            started: false,
        }
    }
    
    /// Get next value after dropping initial items
    pub fn next(&mut self) -> Option<T> {
        if !self.started {
            // Skip items while predicate is true
            while self.index < self.items.len() {
                if let Some(item) = self.items.get(self.index) {
                    if !(self.predicate)(item) {
                        self.started = true;
                        break;
                    }
                    self.index += 1;
                } else {
                    break;
                }
            }
        }
        
        if self.index < self.items.len() {
            let result = self.items.get(self.index).cloned();
            self.index += 1;
            result
        } else {
            None
        }
    }
    
    /// Collect all remaining items
    pub fn collect(mut self) -> Vec<T> {
        let mut result = Vec::new();
        while let Some(item) = self.next() {
            result.push(item);
        }
        result
    }
}

/// filterfalse - iterator filtering items where predicate is false
#[derive(Debug)]
pub struct FilterFalse<T, F> {
    items: Vec<T>,
    predicate: F,
    index: usize,
}

impl<T, F> FilterFalse<T, F> 
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    /// Create filterfalse iterator
    pub fn new<I>(iterable: I, predicate: F) -> Self 
    where
        I: IntoIterator<Item = T>,
    {
        Self {
            items: iterable.into_iter().collect(),
            predicate,
            index: 0,
        }
    }
    
    /// Get next value where predicate is false
    pub fn next(&mut self) -> Option<T> {
        while self.index < self.items.len() {
            if let Some(item) = self.items.get(self.index) {
                self.index += 1;
                if !(self.predicate)(item) {
                    return Some(item.clone());
                }
            } else {
                break;
            }
        }
        None
    }
    
    /// Collect all remaining items
    pub fn collect(mut self) -> Vec<T> {
        let mut result = Vec::new();
        while let Some(item) = self.next() {
            result.push(item);
        }
        result
    }
}

/// compress - iterator selecting items based on selectors
#[derive(Debug)]
pub struct Compress<T> {
    data: Vec<T>,
    selectors: Vec<bool>,
    index: usize,
}

impl<T> Compress<T> 
where
    T: Clone,
{
    /// Create compress iterator
    pub fn new<I, S>(data: I, selectors: S) -> Self 
    where
        I: IntoIterator<Item = T>,
        S: IntoIterator<Item = bool>,
    {
        Self {
            data: data.into_iter().collect(),
            selectors: selectors.into_iter().collect(),
            index: 0,
        }
    }
    
    /// Get next selected value
    pub fn next(&mut self) -> Option<T> {
        while self.index < self.data.len().min(self.selectors.len()) {
            let current_index = self.index;
            self.index += 1;
            
            if self.selectors.get(current_index).copied().unwrap_or(false) {
                return self.data.get(current_index).cloned();
            }
        }
        None
    }
    
    /// Collect all remaining items
    pub fn collect(mut self) -> Vec<T> {
        let mut result = Vec::new();
        while let Some(item) = self.next() {
            result.push(item);
        }
        result
    }
}

// Module-level convenience functions

/// count - create count iterator
pub fn count<T>(start: T, step: T) -> Count<T> 
where
    T: Clone + core::ops::Add<Output = T>,
{
    Count::new(start, step)
}

/// cycle - create cycle iterator
pub fn cycle<T, I>(iterable: I) -> Cycle<T> 
where
    T: Clone,
    I: IntoIterator<Item = T>,
{
    Cycle::new(iterable)
}

/// repeat - create repeat iterator
pub fn repeat<T>(item: T) -> Repeat<T> 
where
    T: Clone,
{
    Repeat::infinite(item)
}

/// repeat_times - create limited repeat iterator
pub fn repeat_times<T>(item: T, times: usize) -> Repeat<T> 
where
    T: Clone,
{
    Repeat::times(item, times)
}

/// chain - chain multiple iterables
pub fn chain_from_iterable<T, I>(iterables: I) -> Vec<T> 
where
    I: IntoIterator<Item = Vec<T>>,
{
    let mut result = Vec::new();
    for iterable in iterables {
        result.extend(iterable);
    }
    result
}

/// islice - slice iterator
pub fn islice<T, I>(iterable: I, start: usize, stop: Option<usize>, step: usize) -> Vec<T> 
where
    T: Clone,
    I: IntoIterator<Item = T>,
{
    ISlice::new(iterable, start, stop, step).collect()
}

/// takewhile - take while predicate is true
pub fn takewhile<T, I, F>(iterable: I, predicate: F) -> Vec<T> 
where
    T: Clone,
    I: IntoIterator<Item = T>,
    F: Fn(&T) -> bool,
{
    TakeWhile::new(iterable, predicate).collect()
}

/// dropwhile - drop while predicate is true
pub fn dropwhile<T, I, F>(iterable: I, predicate: F) -> Vec<T> 
where
    T: Clone,
    I: IntoIterator<Item = T>,
    F: Fn(&T) -> bool,
{
    DropWhile::new(iterable, predicate).collect()
}

/// filterfalse - filter where predicate is false
pub fn filterfalse<T, I, F>(iterable: I, predicate: F) -> Vec<T> 
where
    T: Clone,
    I: IntoIterator<Item = T>,
    F: Fn(&T) -> bool,
{
    FilterFalse::new(iterable, predicate).collect()
}

/// compress - select items based on selectors
pub fn compress<T, I, S>(data: I, selectors: S) -> Vec<T> 
where
    T: Clone,
    I: IntoIterator<Item = T>,
    S: IntoIterator<Item = bool>,
{
    Compress::new(data, selectors).collect()
}

/// combinations - generate combinations of length r
pub fn combinations<T>(iterable: &[T], r: usize) -> Vec<Vec<T>> 
where
    T: Clone,
{
    if r == 0 {
        return vec![vec![]];
    }
    
    if r > iterable.len() {
        return vec![];
    }
    
    let mut result = Vec::new();
    combinations_helper(iterable, r, 0, &mut vec![], &mut result);
    result
}

fn combinations_helper<T>(
    iterable: &[T], 
    r: usize, 
    start: usize, 
    current: &mut Vec<T>, 
    result: &mut Vec<Vec<T>>
) where
    T: Clone,
{
    if current.len() == r {
        result.push(current.clone());
        return;
    }
    
    for i in start..iterable.len() {
        current.push(iterable[i].clone());
        combinations_helper(iterable, r, i + 1, current, result);
        current.pop();
    }
}

/// permutations - generate permutations of length r
pub fn permutations<T>(iterable: &[T], r: Option<usize>) -> Vec<Vec<T>> 
where
    T: Clone,
{
    let r = r.unwrap_or(iterable.len());
    
    if r == 0 {
        return vec![vec![]];
    }
    
    if r > iterable.len() {
        return vec![];
    }
    
    let mut result = Vec::new();
    let mut used = vec![false; iterable.len()];
    permutations_helper(iterable, r, &mut vec![], &mut used, &mut result);
    result
}

fn permutations_helper<T>(
    iterable: &[T], 
    r: usize, 
    current: &mut Vec<T>, 
    used: &mut [bool], 
    result: &mut Vec<Vec<T>>
) where
    T: Clone,
{
    if current.len() == r {
        result.push(current.clone());
        return;
    }
    
    for i in 0..iterable.len() {
        if !used[i] {
            current.push(iterable[i].clone());
            used[i] = true;
            permutations_helper(iterable, r, current, used, result);
            current.pop();
            used[i] = false;
        }
    }
}

/// product - cartesian product of iterables
pub fn product<T>(iterables: &[Vec<T>]) -> Vec<Vec<T>> 
where
    T: Clone,
{
    if iterables.is_empty() {
        return vec![vec![]];
    }
    
    let mut result = vec![vec![]];
    
    for iterable in iterables {
        let mut new_result = Vec::new();
        for existing in &result {
            for item in iterable {
                let mut new_combo = existing.clone();
                new_combo.push(item.clone());
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }
    
    result
}

/// accumulate - running totals
pub fn accumulate<T, F>(iterable: &[T], func: Option<F>) -> Vec<T> 
where
    T: Clone + core::ops::Add<Output = T>,
    F: Fn(&T, &T) -> T,
{
    if iterable.is_empty() {
        return vec![];
    }
    
    let mut result = vec![iterable[0].clone()];
    
    for i in 1..iterable.len() {
        let next_val = if let Some(ref f) = func {
            f(&result[i - 1], &iterable[i])
        } else {
            result[i - 1].clone() + iterable[i].clone()
        };
        result.push(next_val);
    }
    
    result
}

/// accumulate with only the default operator (running sums). A separate
/// entry point so codegen never has to name an uninferable Option<F>.
pub fn accumulate_sum<T>(iterable: &[T]) -> Vec<T>
where
    T: Clone + core::ops::Add<Output = T>,
{
    let mut result: Vec<T> = Vec::with_capacity(iterable.len());
    for x in iterable {
        match result.last() {
            Some(prev) => {
                let next = prev.clone() + x.clone();
                result.push(next);
            }
            None => result.push(x.clone()),
        }
    }
    result
}

/// accumulate(iterable, func)
pub fn accumulate_func<T, F>(iterable: &[T], mut func: F) -> Vec<T>
where
    T: Clone,
    F: FnMut(T, T) -> T,
{
    let mut result: Vec<T> = Vec::with_capacity(iterable.len());
    for x in iterable {
        match result.last() {
            Some(prev) => {
                let next = func(prev.clone(), x.clone());
                result.push(next);
            }
            None => result.push(x.clone()),
        }
    }
    result
}

/// accumulate(iterable, initial=v): the initial value leads the output,
/// so an empty iterable still yields [initial].
pub fn accumulate_sum_initial<T>(iterable: &[T], initial: T) -> Vec<T>
where
    T: Clone + core::ops::Add<Output = T>,
{
    let mut result = vec![initial];
    for x in iterable {
        let next = result.last().unwrap().clone() + x.clone();
        result.push(next);
    }
    result
}

/// accumulate(iterable, func, initial=v)
pub fn accumulate_func_initial<T, F>(iterable: &[T], mut func: F, initial: T) -> Vec<T>
where
    T: Clone,
    F: FnMut(T, T) -> T,
{
    let mut result = vec![initial];
    for x in iterable {
        let next = func(result.last().unwrap().clone(), x.clone());
        result.push(next);
    }
    result
}

/// product(a, b): typed pairs, so heterogeneous element types work.
pub fn product2<A: Clone, B: Clone>(a: &[A], b: &[B]) -> Vec<(A, B)> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for x in a {
        for y in b {
            out.push((x.clone(), y.clone()));
        }
    }
    out
}

/// product(a, b, c)
pub fn product3<A: Clone, B: Clone, C: Clone>(a: &[A], b: &[B], c: &[C]) -> Vec<(A, B, C)> {
    let mut out = Vec::with_capacity(a.len() * b.len() * c.len());
    for x in a {
        for y in b {
            for z in c {
                out.push((x.clone(), y.clone(), z.clone()));
            }
        }
    }
    out
}

/// product(iterable, repeat=2)
pub fn product_repeat2<T: Clone>(iterable: &[T]) -> Vec<(T, T)> {
    product2(iterable, iterable)
}

/// product(iterable, repeat=3)
pub fn product_repeat3<T: Clone>(iterable: &[T]) -> Vec<(T, T, T)> {
    product3(iterable, iterable, iterable)
}

/// combinations_with_replacement(iterable, r) — same Vec-of-Vec shape as
/// combinations(), in the same lexicographic-by-index order as Python.
pub fn combinations_with_replacement<T: Clone>(iterable: &[T], r: usize) -> Vec<Vec<T>> {
    let n = iterable.len();
    let mut result = Vec::new();
    if r == 0 {
        result.push(Vec::new());
        return result;
    }
    if n == 0 {
        return result;
    }
    // indices are non-decreasing; advance like CPython's implementation.
    let mut indices = vec![0usize; r];
    loop {
        result.push(indices.iter().map(|&i| iterable[i].clone()).collect());
        // Find the rightmost index that can still grow.
        let mut i = r;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if indices[i] != n - 1 {
                break;
            }
        }
        let next = indices[i] + 1;
        for j in i..r {
            indices[j] = next;
        }
    }
}

/// pairwise(iterable): consecutive overlapping pairs.
pub fn pairwise<T: Clone>(iterable: &[T]) -> Vec<(T, T)> {
    iterable
        .windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect()
}

/// zip_longest(a, b): exhausted sides fill with None, which is exactly
/// the Option in rython's None model.
pub fn zip_longest<A: Clone, B: Clone>(a: &[A], b: &[B]) -> Vec<(Option<A>, Option<B>)> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| (a.get(i).cloned(), b.get(i).cloned()))
        .collect()
}

/// zip_longest(a, b, fillvalue=v)
pub fn zip_longest_fill<T: Clone>(a: &[T], b: &[T], fill: T) -> Vec<(T, T)> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| {
            (
                a.get(i).cloned().unwrap_or_else(|| fill.clone()),
                b.get(i).cloned().unwrap_or_else(|| fill.clone()),
            )
        })
        .collect()
}

/// groupby(iterable): CONSECUTIVE runs of equal elements, like Python —
/// [1,1,2,1] yields three groups, not two. Groups are materialized.
pub fn groupby<T: Clone + PartialEq>(iterable: &[T]) -> Vec<(T, Vec<T>)> {
    groupby_key(iterable, |x| x.clone())
}

/// groupby(iterable, key=f)
pub fn groupby_key<T, K, F>(iterable: &[T], mut key: F) -> Vec<(K, Vec<T>)>
where
    T: Clone,
    K: Clone + PartialEq,
    F: FnMut(&T) -> K,
{
    let mut out: Vec<(K, Vec<T>)> = Vec::new();
    for x in iterable {
        let k = key(x);
        match out.last_mut() {
            Some((current, group)) if *current == k => group.push(x.clone()),
            _ => out.push((k, vec![x.clone()])),
        }
    }
    out
}

/// The tuple arities starmap() can splat into a function call.
pub trait StarArgs<F> {
    type Out;
    fn star_call(self, f: &mut F) -> Self::Out;
}

impl<A, B, U, F: FnMut(A, B) -> U> StarArgs<F> for (A, B) {
    type Out = U;
    fn star_call(self, f: &mut F) -> U {
        f(self.0, self.1)
    }
}

impl<A, B, C, U, F: FnMut(A, B, C) -> U> StarArgs<F> for (A, B, C) {
    type Out = U;
    fn star_call(self, f: &mut F) -> U {
        f(self.0, self.1, self.2)
    }
}

/// starmap(f, iterable-of-tuples): each tuple splats into f's parameters,
/// so `starmap(lambda a, b: a * b, pairs)` works for 2- and 3-tuples.
pub fn starmap<T, F>(mut f: F, iterable: &[T]) -> Vec<T::Out>
where
    T: StarArgs<F> + Clone,
{
    iterable
        .iter()
        .cloned()
        .map(|t| t.star_call(&mut f))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count() {
        let mut counter = count(0, 2);
        assert_eq!(counter.take(5), vec![0, 2, 4, 6, 8]);
    }
    
    #[test]
    fn test_cycle() {
        let mut cycler = cycle(vec![1, 2, 3]);
        assert_eq!(cycler.take(7), vec![1, 2, 3, 1, 2, 3, 1]);
    }
    
    #[test]
    fn test_repeat() {
        let mut repeater = repeat_times(5, 3);
        assert_eq!(repeater.take(5), vec![5, 5, 5]); // Only 3 items available
    }
    
    #[test]
    fn test_combinations() {
        let result = combinations(&[1, 2, 3, 4], 2);
        assert_eq!(result, vec![
            vec![1, 2], vec![1, 3], vec![1, 4],
            vec![2, 3], vec![2, 4], vec![3, 4]
        ]);
    }
    
    #[test]
    fn test_permutations() {
        let result = permutations(&[1, 2], None);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&vec![1, 2]));
        assert!(result.contains(&vec![2, 1]));
    }
    
    #[test]
    fn test_compress() {
        let result = compress(
            vec!['A', 'B', 'C', 'D'],
            vec![true, false, true, false]
        );
        assert_eq!(result, vec!['A', 'C']);
    }
}