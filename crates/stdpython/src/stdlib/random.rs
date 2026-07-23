//! Python random module implementation.
//!
//! The generator is a port of the Mersenne Twister (MT19937) exactly as
//! CPython uses it (Modules/_randommodule.c), and the distribution
//! functions follow Lib/random.py operation-for-operation. Integer-only
//! paths (`random()`, `randrange`, `randint`, `getrandbits`, `choice`,
//! `shuffle`, `sample`, weightless `choices`) reproduce CPython's seeded
//! sequences bit-for-bit; the continuous distributions perform the same
//! arithmetic and match to within libm rounding differences.

use crate::PyException;
use std::sync::Mutex;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The full generator state Python keeps per Random instance: the MT19937
/// word array and index, plus the cached second Box-Muller deviate used by
/// `gauss` (tied to the state so seeded sequences are reproducible, and
/// cleared by `seed`, exactly as CPython does).
struct PyRandom {
    mt: [u32; N],
    index: usize,
    gauss_next: Option<f64>,
    seeded: bool,
}

static RNG: Mutex<PyRandom> = Mutex::new(PyRandom {
    mt: [0; N],
    index: N + 1,
    gauss_next: None,
    seeded: false,
});

impl PyRandom {
    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = N.max(key.len());
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    /// Seed from a Python int, matching CPython's random_seed: the absolute
    /// value split into little-endian 32-bit words feeds init_by_array.
    fn seed_int(&mut self, n: i64) {
        let a = n.unsigned_abs();
        let lo = (a & 0xffff_ffff) as u32;
        let hi = (a >> 32) as u32;
        let key: &[u32] = if hi != 0 { &[lo, hi] } else { &[lo] };
        self.init_by_array(key);
        self.gauss_next = None;
        self.seeded = true;
    }

    /// Seed from OS entropy (std's RandomState draws from the OS), used for
    /// `seed()` with no argument and for first use of an unseeded generator.
    fn seed_from_entropy(&mut self) {
        use std::hash::{BuildHasher, Hasher};
        let mut key = [0u32; 4];
        for pair in 0..2 {
            let word = std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish();
            key[pair * 2] = (word & 0xffff_ffff) as u32;
            key[pair * 2 + 1] = (word >> 32) as u32;
        }
        self.init_by_array(&key);
        self.gauss_next = None;
        self.seeded = true;
    }

    fn ensure_seeded(&mut self) {
        if !self.seeded {
            self.seed_from_entropy();
        }
    }

    fn genrand_u32(&mut self) -> u32 {
        self.ensure_seeded();
        if self.index >= N {
            for i in 0..N {
                let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
                let mut next = self.mt[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.mt[i] = next;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython's genrand_res53: a 53-bit float uniform on [0, 1).
    fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64; // 27 bits
        let b = (self.genrand_u32() >> 6) as f64; // 26 bits
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// CPython's getrandbits: k bits assembled from 32-bit words,
    /// little-endian, each partial word taken from the TOP of a fresh
    /// genrand output.
    fn getrandbits(&mut self, k: u32) -> u64 {
        debug_assert!(k >= 1 && k <= 64);
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        let lo = self.genrand_u32() as u64;
        let hi_bits = k - 32;
        let hi = (self.genrand_u32() >> (32 - hi_bits)) as u64;
        lo | (hi << 32)
    }

    /// CPython's _randbelow_with_getrandbits: rejection-sample the smallest
    /// bit-width covering n.
    fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }
}

fn with_rng<T>(f: impl FnOnce(&mut PyRandom) -> T) -> T {
    // A poisoned lock only occurs if a panic happened mid-generation;
    // continuing with the recovered state matches Python best-effort.
    let mut guard = match RNG.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(&mut guard)
}

/// random.seed - initialize the generator. An integer seed reproduces
/// CPython's sequence for the same seed; None seeds from OS entropy.
pub fn seed<T>(a: Option<T>)
where
    T: Into<i64>,
{
    with_rng(|rng| match a {
        Some(n) => rng.seed_int(n.into()),
        None => rng.seed_from_entropy(),
    });
}

/// random.getstate - the generator state: 624 MT words, the index, and the
/// cached gauss deviate (presence flag + bits).
pub fn getstate() -> Vec<u64> {
    with_rng(|rng| {
        rng.ensure_seeded();
        let mut state: Vec<u64> = rng.mt.iter().map(|w| *w as u64).collect();
        state.push(rng.index as u64);
        match rng.gauss_next {
            Some(g) => {
                state.push(1);
                state.push(g.to_bits());
            }
            None => {
                state.push(0);
                state.push(0);
            }
        }
        state
    })
}

/// random.setstate - restore a state captured by getstate.
pub fn setstate(state: &[u64]) -> Result<(), PyException> {
    if state.len() != N + 3 {
        return Err(crate::value_error(&format!(
            "state vector must have {} elements, got {}",
            N + 3,
            state.len()
        )));
    }
    with_rng(|rng| {
        for (i, w) in state[..N].iter().enumerate() {
            rng.mt[i] = *w as u32;
        }
        rng.index = state[N] as usize;
        rng.gauss_next = if state[N + 1] == 1 {
            Some(f64::from_bits(state[N + 2]))
        } else {
            None
        };
        rng.seeded = true;
    });
    Ok(())
}

/// random.random - uniform float in [0.0, 1.0).
pub fn random() -> f64 {
    with_rng(|rng| rng.random())
}

/// random.uniform - random float between a and b.
pub fn uniform<T, U>(a: T, b: U) -> f64
where
    T: Into<f64>,
    U: Into<f64>,
{
    let a = a.into();
    let b = b.into();
    a + (b - a) * random()
}

/// random.triangular - triangular distribution (CPython's algorithm,
/// including the degenerate high == low case).
pub fn triangular<T, U, V>(low: T, high: U, mode: Option<V>) -> f64
where
    T: Into<f64>,
    U: Into<f64>,
    V: Into<f64>,
{
    let mut low = low.into();
    let mut high = high.into();
    let mut u = random();
    let mut c = match mode {
        None => 0.5,
        Some(m) => {
            if high == low {
                return low;
            }
            (m.into() - low) / (high - low)
        }
    };
    if u > c {
        u = 1.0 - u;
        c = 1.0 - c;
        std::mem::swap(&mut low, &mut high);
    }
    low + (high - low) * (u * c).sqrt()
}

/// random.normalvariate - normal distribution via Kinderman-Monahan
/// (CPython's algorithm; no hidden cache).
pub fn normalvariate<T, U>(mu: T, sigma: U) -> f64
where
    T: Into<f64>,
    U: Into<f64>,
{
    let mu = mu.into();
    let sigma = sigma.into();
    let nv_magicconst = 4.0 * (-0.5f64).exp() / 2.0f64.sqrt();
    loop {
        let u1 = random();
        let u2 = 1.0 - random();
        let z = nv_magicconst * (u1 - 0.5) / u2;
        if z * z / 4.0 <= -u2.ln() {
            return mu + z * sigma;
        }
    }
}

/// random.gauss - normal distribution via Box-Muller with the second
/// deviate cached in the generator state (cleared by seed), as CPython.
pub fn gauss<T, U>(mu: T, sigma: U) -> f64
where
    T: Into<f64>,
    U: Into<f64>,
{
    let mu = mu.into();
    let sigma = sigma.into();
    let z = with_rng(|rng| match rng.gauss_next.take() {
        Some(z) => z,
        None => {
            let x2pi = rng.random() * std::f64::consts::TAU;
            let g2rad = (-2.0 * (1.0 - rng.random()).ln()).sqrt();
            rng.gauss_next = Some(x2pi.sin() * g2rad);
            x2pi.cos() * g2rad
        }
    });
    mu + z * sigma
}

/// random.lognormvariate - log of the variate is normally distributed.
pub fn lognormvariate<T, U>(mu: T, sigma: U) -> f64
where
    T: Into<f64>,
    U: Into<f64>,
{
    normalvariate(mu, sigma).exp()
}

/// random.expovariate - exponential distribution. Python raises
/// ZeroDivisionError for lambd == 0 and permits negative lambd.
pub fn expovariate<T>(lambd: T) -> Result<f64, PyException>
where
    T: Into<f64>,
{
    let lambd = lambd.into();
    if lambd == 0.0 {
        return Err(PyException::new("ZeroDivisionError", "float division by zero"));
    }
    Ok(-(1.0 - random()).ln() / lambd)
}

/// random.gammavariate - gamma distribution, CPython's three-case
/// algorithm (Cheng for alpha > 1, exponential for alpha == 1,
/// ALGORITHM GS for alpha < 1).
pub fn gammavariate<T, U>(alpha: T, beta: U) -> Result<f64, PyException>
where
    T: Into<f64>,
    U: Into<f64>,
{
    let alpha = alpha.into();
    let beta = beta.into();
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(crate::value_error("gammavariate: alpha and beta must be > 0.0"));
    }
    let sg_magicconst = 1.0 + 4.5f64.ln();
    let log4 = 4.0f64.ln();
    if alpha > 1.0 {
        let ainv = (2.0 * alpha - 1.0).sqrt();
        let bbb = alpha - log4;
        let ccc = alpha + ainv;
        loop {
            let u1 = random();
            if !(1e-7 < u1 && u1 < 0.999_999_9) {
                continue;
            }
            let u2 = 1.0 - random();
            let v = (u1 / (1.0 - u1)).ln() / ainv;
            let x = alpha * v.exp();
            let z = u1 * u1 * u2;
            let r = bbb + ccc * v - x;
            if r + sg_magicconst - 4.5 * z >= 0.0 || r >= z.ln() {
                return Ok(x * beta);
            }
        }
    } else if alpha == 1.0 {
        Ok(-(1.0 - random()).ln() * beta)
    } else {
        loop {
            let u = random();
            let b = (std::f64::consts::E + alpha) / std::f64::consts::E;
            let p = b * u;
            let x = if p <= 1.0 {
                p.powf(1.0 / alpha)
            } else {
                -((b - p) / alpha).ln()
            };
            let u1 = random();
            let accept = if p > 1.0 {
                u1 <= x.powf(alpha - 1.0)
            } else {
                u1 <= (-x).exp()
            };
            if accept {
                return Ok(x * beta);
            }
        }
    }
}

/// random.betavariate - ratio of gamma variates (CPython's algorithm).
pub fn betavariate<T, U>(alpha: T, beta: U) -> Result<f64, PyException>
where
    T: Into<f64>,
    U: Into<f64>,
{
    let y = gammavariate(alpha.into(), 1.0)?;
    if y != 0.0 {
        Ok(y / (y + gammavariate(beta.into(), 1.0)?))
    } else {
        Ok(0.0)
    }
}

/// random.vonmisesvariate - circular data distribution (Fisher's method,
/// as in CPython).
pub fn vonmisesvariate<T, U>(mu: T, kappa: U) -> f64
where
    T: Into<f64>,
    U: Into<f64>,
{
    let mu = mu.into();
    let kappa = kappa.into();
    if kappa <= 1e-6 {
        return std::f64::consts::TAU * random();
    }
    let s = 0.5 / kappa;
    let r = s + (1.0 + s * s).sqrt();
    let z = loop {
        let u1 = random();
        let z = (std::f64::consts::PI * u1).cos();
        let d = z / (r + z);
        let u2 = random();
        if u2 < 1.0 - d * d || u2 <= (1.0 - d) * d.exp() {
            break z;
        }
    };
    let q = 1.0 / r;
    let f = (q + z) / (1.0 + q * z);
    let u3 = random();
    if u3 > 0.5 {
        (mu + f.acos()).rem_euclid(std::f64::consts::TAU)
    } else {
        (mu - f.acos()).rem_euclid(std::f64::consts::TAU)
    }
}

/// random.weibullvariate - Weibull distribution (CPython's algorithm).
pub fn weibullvariate<T, U>(alpha: T, beta: U) -> Result<f64, PyException>
where
    T: Into<f64>,
    U: Into<f64>,
{
    let alpha = alpha.into();
    let beta = beta.into();
    if beta == 0.0 {
        return Err(PyException::new("ZeroDivisionError", "float division by zero"));
    }
    let u = 1.0 - random();
    Ok(alpha * (-u.ln()).powf(1.0 / beta))
}

/// random.randrange - random integer from range(start, stop, step),
/// via _randbelow so seeded sequences match CPython.
pub fn randrange(start: i64, stop: Option<i64>, step: Option<i64>) -> Result<i64, PyException> {
    let (start, stop, step) = match (stop, step) {
        (None, None) => (0, start, 1),
        (Some(stop), None) => (start, stop, 1),
        (Some(stop), Some(step)) => {
            if step == 0 {
                return Err(crate::value_error("zero step for randrange()"));
            }
            (start, stop, step)
        }
        (None, Some(_)) => {
            return Err(crate::type_error("Missing stop argument for randrange()"))
        }
    };

    let width = stop - start;
    if step == 1 {
        if width > 0 {
            return Ok(start + with_rng(|rng| rng.randbelow(width as u64)) as i64);
        }
        return Err(crate::value_error("empty range for randrange()"));
    }

    let n = if step > 0 {
        (width + step - 1).div_euclid(step)
    } else {
        (width + step + 1).div_euclid(step)
    };
    if n <= 0 {
        return Err(crate::value_error("empty range for randrange()"));
    }
    Ok(start + step * with_rng(|rng| rng.randbelow(n as u64)) as i64)
}

/// random.randint - random integer in [a, b].
pub fn randint(a: i64, b: i64) -> Result<i64, PyException> {
    if a > b {
        return Err(crate::value_error("empty range for randrange()"));
    }
    randrange(a, Some(b + 1), None)
}

/// random.getrandbits - integer with k random bits. Python supports
/// arbitrary k; this implementation is limited to 64 bits and errors
/// loudly past that.
pub fn getrandbits(k: u32) -> Result<u64, PyException> {
    if k == 0 {
        return Ok(0);
    }
    if k > 64 {
        return Err(crate::value_error(
            "getrandbits() beyond 64 bits is not supported",
        ));
    }
    Ok(with_rng(|rng| rng.getrandbits(k)))
}

/// random.choice - random element of a non-empty sequence.
pub fn choice<T>(seq: &[T]) -> Result<&T, PyException> {
    if seq.is_empty() {
        return Err(crate::index_error("Cannot choose from an empty sequence"));
    }
    let index = with_rng(|rng| rng.randbelow(seq.len() as u64)) as usize;
    Ok(&seq[index])
}

/// random.choices - k elements with replacement, with optional weights,
/// following CPython (floor of random()*n without weights, bisect_right
/// over cumulative weights with them).
pub fn choices<T>(
    population: &[T],
    weights: Option<&[f64]>,
    cum_weights: Option<&[f64]>,
    k: usize,
) -> Result<Vec<T>, PyException>
where
    T: Clone,
{
    let n = population.len();
    let cum: Vec<f64>;
    let cum_weights = match (cum_weights, weights) {
        (Some(_), Some(_)) => {
            return Err(crate::type_error(
                "Cannot specify both weights and cumulative weights",
            ));
        }
        (Some(cw), None) => {
            if cw.len() != n {
                return Err(crate::value_error(
                    "The number of weights does not match the population",
                ));
            }
            cw
        }
        (None, Some(w)) => {
            if w.len() != n {
                return Err(crate::value_error(
                    "The number of weights does not match the population",
                ));
            }
            let mut total = 0.0;
            cum = w
                .iter()
                .map(|w| {
                    total += w;
                    total
                })
                .collect();
            &cum
        }
        (None, None) => {
            if n == 0 {
                return Err(crate::index_error("Cannot choose from an empty sequence"));
            }
            let nf = n as f64;
            return Ok((0..k)
                .map(|_| population[(random() * nf).floor() as usize].clone())
                .collect());
        }
    };
    let total = *cum_weights.last().unwrap_or(&0.0);
    if total <= 0.0 {
        return Err(crate::value_error("Total of weights must be greater than zero"));
    }
    if !total.is_finite() {
        return Err(crate::value_error("Total of weights must be finite"));
    }
    let hi = n - 1;
    Ok((0..k)
        .map(|_| {
            let x = random() * total;
            // bisect_right over cum_weights[0..hi]
            let mut lo = 0usize;
            let mut hi = hi;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if x < cum_weights[mid] {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            population[lo].clone()
        })
        .collect())
}

/// random.shuffle - Fisher-Yates driven by _randbelow, matching CPython's
/// seeded permutations.
pub fn shuffle<T>(seq: &mut [T]) {
    with_rng(|rng| {
        for i in (1..seq.len()).rev() {
            let j = rng.randbelow((i + 1) as u64) as usize;
            seq.swap(i, j);
        }
    });
}

/// random.sample - k unique elements, using CPython's pool/selection-set
/// algorithm so seeded results match.
pub fn sample<T>(population: &[T], k: usize) -> Result<Vec<T>, PyException>
where
    T: Clone,
{
    let n = population.len();
    if k > n {
        return Err(crate::value_error(
            "Sample larger than population or is negative",
        ));
    }
    let mut setsize: usize = 21;
    if k > 5 {
        setsize += 4usize.pow(((k * 3) as f64).log(4.0).ceil() as u32);
    }
    with_rng(|rng| {
        let mut result = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool: Vec<T> = population.to_vec();
            for i in 0..k {
                let j = rng.randbelow((n - i) as u64) as usize;
                result.push(pool[j].clone());
                pool[j] = pool[n - i - 1].clone();
            }
        } else {
            let mut selected = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = rng.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = rng.randbelow(n as u64) as usize;
                }
                selected.insert(j);
                result.push(population[j].clone());
            }
        }
        Ok(result)
    })
}

/// SystemRandom - draws from OS entropy (std's RandomState), independent
/// of the seeded generator, like Python's os.urandom-backed SystemRandom.
pub struct SystemRandom;

impl SystemRandom {
    pub fn new() -> Self {
        Self
    }

    fn entropy_word() -> u64 {
        use std::hash::{BuildHasher, Hasher};
        std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish()
    }

    /// Generate random bytes.
    pub fn randbytes(&self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let word = Self::entropy_word();
            for b in word.to_le_bytes() {
                if out.len() == n {
                    break;
                }
                out.push(b);
            }
        }
        out
    }

    /// Generate random integer in range [0, k).
    pub fn randbelow(&self, k: u64) -> Result<u64, PyException> {
        if k == 0 {
            return Err(crate::value_error("k must be positive"));
        }
        let bits = 64 - (k - 1).leading_zeros().min(63);
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        loop {
            let r = Self::entropy_word() & mask;
            if r < k {
                return Ok(r);
            }
        }
    }
}

impl Default for SystemRandom {
    fn default() -> Self {
        Self::new()
    }
}
