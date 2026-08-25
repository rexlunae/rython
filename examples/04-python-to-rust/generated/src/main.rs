#![doc = "wordstats: a tiny word-frequency report.\\n\\nOrdinary Python - classes, inheritance with super(), overridden methods,\\ndicts, f-strings, and three fully unannotated (type-inferred) helpers -\\nused as the walkthrough program for converting Python to Rust with\\nrypip. See README.md next to this file; the crate it generates is\\nchecked in under generated/ for reading.\\n"]
#![doc = "Generated from Python file: wordstats.py"]
use stdpython::*;
#[doc = "Base class: a named accumulator."]
#[derive(Clone, Default)]
pub struct Tally {
    pub label: String,
    pub total: i64,
}
pub trait TallyTrait {
    fn label(&self) -> String;
    fn label_mut(&mut self) -> &mut String;
    fn total(&self) -> i64;
    fn total_mut(&mut self) -> &mut i64;
    fn add(&mut self, n: i64) -> Result<(), PyException> {
        *self.total_mut() = (self.total()).py_add(&(n));
        Ok(())
    }
    fn summary(&self) -> Result<String, PyException> {
        return Ok(format!(
            "{}: {}",
            py_display(&(self.label())),
            py_display(&(self.total()))
        ));
    }
    fn __rython_super_add(&mut self, n: i64) -> Result<(), PyException> {
        *self.total_mut() = (self.total()).py_add(&(n));
        Ok(())
    }
    fn __rython_super_summary(&self) -> Result<String, PyException> {
        return Ok(format!(
            "{}: {}",
            py_display(&(self.label())),
            py_display(&(self.total()))
        ));
    }
}
impl TallyTrait for Tally {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn label_mut(&mut self) -> &mut String {
        &mut self.label
    }
    fn total(&self) -> i64 {
        self.total.clone()
    }
    fn total_mut(&mut self) -> &mut i64 {
        &mut self.total
    }
}
impl Tally {
    pub fn new(label: impl Into<String>) -> Result<Self, PyException> {
        let mut __rython_self = Self::default();
        __rython_self.__init__(label)?;
        Ok(__rython_self)
    }
    pub(crate) fn __init__(&mut self, label: impl Into<String>) -> Result<(), PyException> {
        let label: String = label.into();
        self.label = label;
        self.total = 0;
        Ok(())
    }
    pub fn add(&mut self, n: i64) -> Result<(), PyException> {
        self.total = (self.total).py_add(&(n));
        Ok(())
    }
    pub fn summary(&self) -> Result<String, PyException> {
        return Ok(format!(
            "{}: {}",
            py_display(&(self.label)),
            py_display(&(self.total))
        ));
    }
}
#[doc = "Counts words and tracks per-word frequencies."]
#[derive(Clone, Default)]
pub struct WordTally {
    pub freq: PyDict<String, i64>,
    pub __rython_base: Tally,
}
pub trait WordTallyTrait: TallyTrait {
    fn base(&self) -> &Tally;
    fn base_mut(&mut self) -> &mut Tally;
    fn freq(&self) -> PyDict<String, i64>;
    fn freq_mut(&mut self) -> &mut PyDict<String, i64>;
    fn add_text(&mut self, text: impl Into<String>) -> Result<(), PyException> {
        let text: String = text.into();
        let words;
        words = (text).py_split_whitespace();
        {
            (self).add(len(&(words)) as i64)?
        };
        for w in words {
            {
                let __rython_val = ((self.freq()).py_get_default(&(w), 0)).py_add(&((1) as i64));
                (self.freq_mut()).py_set_index(w, __rython_val)?;
            };
        }
        Ok(())
    }
    fn top(&self) -> Result<String, PyException> {
        let mut best;
        let mut best_count;
        let mut count;
        best = ("").to_string();
        best_count = 0;
        for w in sorted(&((self.freq()).py_keys())) {
            count = (self.freq()).py_get_default(&(w), 0);
            if (count).py_gt(&(best_count)) {
                best = w;
                best_count = count;
            };
        }
        return Ok(format!(
            "{} x{}",
            py_display(&(best)),
            py_display(&(best_count))
        ));
    }
    fn __rython_super_add_text(&mut self, text: impl Into<String>) -> Result<(), PyException> {
        let text: String = text.into();
        let words;
        words = (text).py_split_whitespace();
        {
            (self).add(len(&(words)) as i64)?
        };
        for w in words {
            {
                let __rython_val = ((self.freq()).py_get_default(&(w), 0)).py_add(&((1) as i64));
                (self.freq_mut()).py_set_index(w, __rython_val)?;
            };
        }
        Ok(())
    }
    fn __rython_super_summary(&self) -> Result<String, PyException> {
        let base;
        let distinct;
        base = { <Self as TallyTrait>::__rython_super_summary(self)? };
        distinct = len(&(self.freq())) as i64;
        return Ok(format!(
            "{} ({} distinct)",
            py_display(&(base)),
            py_display(&(distinct))
        ));
    }
    fn __rython_super_top(&self) -> Result<String, PyException> {
        let mut best;
        let mut best_count;
        let mut count;
        best = ("").to_string();
        best_count = 0;
        for w in sorted(&((self.freq()).py_keys())) {
            count = (self.freq()).py_get_default(&(w), 0);
            if (count).py_gt(&(best_count)) {
                best = w;
                best_count = count;
            };
        }
        return Ok(format!(
            "{} x{}",
            py_display(&(best)),
            py_display(&(best_count))
        ));
    }
}
impl WordTallyTrait for WordTally {
    fn base(&self) -> &Tally {
        &self.__rython_base
    }
    fn base_mut(&mut self) -> &mut Tally {
        &mut self.__rython_base
    }
    fn freq(&self) -> PyDict<String, i64> {
        self.freq.clone()
    }
    fn freq_mut(&mut self) -> &mut PyDict<String, i64> {
        &mut self.freq
    }
}
impl TallyTrait for WordTally {
    fn label(&self) -> String {
        self.__rython_base.label.clone()
    }
    fn label_mut(&mut self) -> &mut String {
        &mut self.__rython_base.label
    }
    fn total(&self) -> i64 {
        self.__rython_base.total.clone()
    }
    fn total_mut(&mut self) -> &mut i64 {
        &mut self.__rython_base.total
    }
    fn summary(&self) -> Result<String, PyException> {
        let base;
        let distinct;
        base = { <Self as TallyTrait>::__rython_super_summary(self)? };
        distinct = len(&(self.freq)) as i64;
        return Ok(format!(
            "{} ({} distinct)",
            py_display(&(base)),
            py_display(&(distinct))
        ));
    }
}
impl WordTally {
    pub fn new() -> Result<Self, PyException> {
        let mut __rython_self = Self::default();
        __rython_self.__init__()?;
        Ok(__rython_self)
    }
    pub(crate) fn __init__(&mut self) -> Result<(), PyException> {
        {
            (self.__rython_base).__init__("words")?
        };
        self.freq = PyDict::from([]);
        Ok(())
    }
    pub fn add_text(&mut self, text: impl Into<String>) -> Result<(), PyException> {
        let text: String = text.into();
        let words;
        words = (text).py_split_whitespace();
        {
            (self).add(len(&(words)) as i64)?
        };
        for w in words {
            {
                let __rython_val = ((self.freq).py_get_default(&(w), 0)).py_add(&((1) as i64));
                (self.freq).py_set_index(w, __rython_val)?;
            };
        }
        Ok(())
    }
    pub fn summary(&self) -> Result<String, PyException> {
        let base;
        let distinct;
        base = { <Self as TallyTrait>::__rython_super_summary(self)? };
        distinct = len(&(self.freq)) as i64;
        return Ok(format!(
            "{} ({} distinct)",
            py_display(&(base)),
            py_display(&(distinct))
        ));
    }
    pub fn top(&self) -> Result<String, PyException> {
        let mut best;
        let mut best_count;
        let mut count;
        best = ("").to_string();
        best_count = 0;
        for w in sorted(&((self.freq).py_keys())) {
            count = (self.freq).py_get_default(&(w), 0);
            if (count).py_gt(&(best_count)) {
                best = w;
                best_count = count;
            };
        }
        return Ok(format!(
            "{} x{}",
            py_display(&(best)),
            py_display(&(best_count))
        ));
    }
}
#[doc = "No annotations: `for w in words` infers an iterable, and the\\n    accumulator's `best = \\\"\\\"` seed concretizes the element type - the\\n    signature is `longest<T: IntoIterator<Item = String>>(words: T) ->\\n    Result<String, _>`."]
pub fn longest<T>(words: T) -> Result<String, PyException>
where
    T: IntoIterator<Item = String>,
{
    let mut best;
    best = ("").to_string();
    for w in words {
        if (len(&(w)) as i64).py_gt(&(len(&(best)) as i64)) {
            best = w;
        };
    }
    return Ok(best);
}
#[doc = "No annotations: an integer-seeded accumulator over an inferred\\n    iterable, with `len(w)` bounding the elements."]
pub fn total_chars<A, B>(words: A) -> Result<i64, PyException>
where
    A: IntoIterator<Item = B>,
    B: Len,
{
    let mut n;
    n = 0;
    for w in words {
        n = (n).py_add(&(len(&(w)) as i64));
    }
    return Ok(n);
}
#[doc = "No annotations: rython infers a generic, comparison-bounded Rust\\n    signature from how the parameters are used."]
pub fn within<A, B, C>(value: A, low: B, high: C) -> Result<bool, PyException>
where
    A: PyLe<C, Output = bool>,
    A: Clone,
    B: PyLe<A, Output = bool>,
{
    return Ok(((low).py_le(&(value))) && ((value).py_le(&(high))));
}
fn main() {
    let __rython_result = (|| -> Result<(), PyException> {
        let mut tally;
        tally = { WordTally::new()? };
        {
            (tally).add_text("the quick brown fox jumps over the lazy dog")?
        };
        {
            (tally).add_text("the dog barks")?
        };
        print(&({ (tally).summary()? }));
        print(&(format!("top: {}", py_display(&({ (tally).top()? })))));
        print(
            &(format!(
                "longest: {}",
                py_display(&(longest(("the quick brown fox").py_split_whitespace())?))
            )),
        );
        print(
            &(format!(
                "chars: {}",
                py_display(&(total_chars(("the quick brown fox").py_split_whitespace())?))
            )),
        );
        print(
            &(format!(
                "tweet-sized: {}",
                py_display(&(within(tally.__rython_base.total, 1, 280)?))
            )),
        );
        Ok(())
    })();
    if let Err(e) = __rython_result {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
