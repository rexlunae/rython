//! Python datetime module implementation
//! 
//! This module provides classes for manipulating dates and times.
//! Implementation matches Python's datetime module API.

use crate::PyException;
use std::time::{SystemTime, Duration, UNIX_EPOCH};
use std::fmt;

// Days in each month (non-leap year)
const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// date - represents a date (year, month, day)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct date {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl date {
    /// Create a new date
    pub fn new(year: i32, month: u32, day: u32) -> Result<Self, PyException> {
        if !(1..=12).contains(&month) {
            return Err(crate::value_error("month must be in 1..12"));
        }
        
        let max_day = if month == 2 && is_leap_year(year) { 29 } else { DAYS_IN_MONTH[month as usize - 1] };
        
        if !(1..=max_day).contains(&day) {
            return Err(crate::value_error(format!("day must be in 1..{}", max_day)));
        }
        
        Ok(Self { year, month, day })
    }
    
    /// Get today's date
    pub fn today() -> Self {
        let now = SystemTime::now();
        let duration = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0));
        let days_since_epoch = duration.as_secs() / 86400;
        days_to_date(days_since_epoch as i64 + 719163) // Unix epoch offset
    }
    
    /// Create date from ordinal day
    pub fn fromordinal(ordinal: i64) -> Result<Self, PyException> {
        if ordinal < 1 {
            return Err(crate::value_error("ordinal must be >= 1"));
        }
        Ok(days_to_date(ordinal))
    }
    
    /// Convert to ordinal day
    pub fn toordinal(&self) -> i64 {
        date_to_days(*self)
    }
    
    /// Get weekday (0=Monday, 6=Sunday)
    pub fn weekday(&self) -> u32 {
        // Ordinal 1 (0001-01-01) is a Monday, so subtract 1 before the mod.
        ((self.toordinal() - 1).rem_euclid(7)) as u32
    }
    
    /// Get ISO weekday (1=Monday, 7=Sunday)
    pub fn isoweekday(&self) -> u32 {
        self.weekday() + 1
    }
    
    /// Get ISO calendar (year, week, weekday)
    pub fn isocalendar(&self) -> (i32, u32, u32) {
        let year = self.year;
        let ordinal = self.toordinal();
        let jan1_ordinal = date::new(year, 1, 1).unwrap().toordinal();
        let jan1_weekday = (jan1_ordinal % 7) as u32;
        
        let week = ((ordinal - jan1_ordinal + jan1_weekday as i64 + 7) / 7) as u32;
        (year, week.max(1), self.isoweekday())
    }
    
    /// Format as ISO string
    pub fn isoformat(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
    
    /// Format with strftime
    pub fn strftime(&self, fmt: &str) -> String {
        // Simplified strftime implementation
        fmt.replace("%Y", &format!("{:04}", self.year))
           .replace("%m", &format!("{:02}", self.month))
           .replace("%d", &format!("{:02}", self.day))
           .replace("%B", month_name(self.month))
           .replace("%b", month_abbr(self.month))
           .replace("%A", weekday_name(self.weekday()))
           .replace("%a", weekday_abbr(self.weekday()))
    }
    
    /// Replace components
    pub fn replace(&self, year: Option<i32>, month: Option<u32>, day: Option<u32>) -> Result<Self, PyException> {
        Self::new(
            year.unwrap_or(self.year),
            month.unwrap_or(self.month),
            day.unwrap_or(self.day)
        )
    }
}

impl fmt::Display for date {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.isoformat())
    }
}

/// time - represents a time (hour, minute, second, microsecond)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct time {
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub microsecond: u32,
}

impl time {
    /// Create a new time
    pub fn new(hour: u32, minute: u32, second: Option<u32>, microsecond: Option<u32>) -> Result<Self, PyException> {
        if hour >= 24 {
            return Err(crate::value_error("hour must be in 0..23"));
        }
        if minute >= 60 {
            return Err(crate::value_error("minute must be in 0..59"));
        }
        let second = second.unwrap_or(0);
        if second >= 60 {
            return Err(crate::value_error("second must be in 0..59"));
        }
        let microsecond = microsecond.unwrap_or(0);
        if microsecond >= 1_000_000 {
            return Err(crate::value_error("microsecond must be in 0..999999"));
        }
        
        Ok(Self { hour, minute, second, microsecond })
    }
    
    /// Format as ISO string
    pub fn isoformat(&self, timespec: Option<&str>) -> String {
        match timespec {
            Some("hours") => format!("{:02}", self.hour),
            Some("minutes") => format!("{:02}:{:02}", self.hour, self.minute),
            Some("seconds") => format!("{:02}:{:02}:{:02}", self.hour, self.minute, self.second),
            _ => {
                if self.microsecond == 0 {
                    format!("{:02}:{:02}:{:02}", self.hour, self.minute, self.second)
                } else {
                    format!("{:02}:{:02}:{:02}.{:06}", self.hour, self.minute, self.second, self.microsecond)
                }
            }
        }
    }
    
    /// Format with strftime
    pub fn strftime(&self, fmt: &str) -> String {
        fmt.replace("%H", &format!("{:02}", self.hour))
           .replace("%M", &format!("{:02}", self.minute))
           .replace("%S", &format!("{:02}", self.second))
           .replace("%f", &format!("{:06}", self.microsecond))
    }
    
    /// Replace components
    pub fn replace(&self, hour: Option<u32>, minute: Option<u32>, second: Option<u32>, microsecond: Option<u32>) -> Result<Self, PyException> {
        Self::new(
            hour.unwrap_or(self.hour),
            minute.unwrap_or(self.minute),
            Some(second.unwrap_or(self.second)),
            Some(microsecond.unwrap_or(self.microsecond))
        )
    }
}

impl fmt::Display for time {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.isoformat(None))
    }
}

/// datetime - represents a datetime (date + time)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct datetime {
    pub date: date,
    pub time: time,
}

impl datetime {
    /// Create a new datetime
    pub fn new(year: i32, month: u32, day: u32, hour: Option<u32>, minute: Option<u32>, 
               second: Option<u32>, microsecond: Option<u32>) -> Result<Self, PyException> {
        let date = date::new(year, month, day)?;
        let time = time::new(
            hour.unwrap_or(0),
            minute.unwrap_or(0),
            second,
            microsecond
        )?;
        Ok(Self { date, time })
    }
    
    /// Get current datetime in LOCAL time, like Python's datetime.now()
    /// (via localtime on unix; non-unix hosts fall back to UTC).
    pub fn now() -> Self {
        let now = SystemTime::now();
        let duration = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0));
        let micros = duration.subsec_micros();
        Self::from_unix_local(duration.as_secs() as i64, micros)
            .unwrap_or_else(|_| Self::from_unix_utc(duration.as_secs() as i64, micros))
    }

    /// Get current datetime in UTC — decomposed from the UNIX clock, NOT an
    /// alias of now() (which is local time).
    pub fn utcnow() -> Self {
        let now = SystemTime::now();
        let duration = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0));
        Self::from_unix_utc(duration.as_secs() as i64, duration.subsec_micros())
    }

    /// Decompose UNIX seconds as UTC.
    fn from_unix_utc(secs: i64, microsecond: u32) -> Self {
        let days_since_epoch = secs.div_euclid(86400);
        let seconds_today = secs.rem_euclid(86400);
        let date = days_to_date(days_since_epoch + 719163);
        let hour = (seconds_today / 3600) as u32;
        let minute = ((seconds_today % 3600) / 60) as u32;
        let second = (seconds_today % 60) as u32;
        Self {
            date,
            time: time::new(hour, minute, Some(second), Some(microsecond))
                .expect("decomposed clock fields are in range"),
        }
    }

    /// Decompose UNIX seconds in the host's LOCAL timezone (unix only).
    #[cfg(unix)]
    fn from_unix_local(secs: i64, microsecond: u32) -> Result<Self, PyException> {
        let t: libc::time_t = secs as libc::time_t;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        let ok = unsafe { libc::localtime_r(&t, &mut tm) };
        if ok.is_null() {
            return Err(crate::value_error("timestamp out of range for localtime"));
        }
        Ok(Self {
            date: date::new(tm.tm_year + 1900, (tm.tm_mon + 1) as u32, tm.tm_mday as u32)?,
            time: time::new(
                tm.tm_hour as u32,
                tm.tm_min as u32,
                Some(tm.tm_sec as u32),
                Some(microsecond),
            )?,
        })
    }

    #[cfg(not(unix))]
    fn from_unix_local(secs: i64, microsecond: u32) -> Result<Self, PyException> {
        // No portable localtime without a timezone database; UTC is the
        // documented fallback on non-unix hosts.
        Ok(Self::from_unix_utc(secs, microsecond))
    }

    /// Create from timestamp, interpreted in LOCAL time like Python.
    /// Negative timestamps (pre-1970) are valid.
    pub fn fromtimestamp(timestamp: f64) -> Result<Self, PyException> {
        if !timestamp.is_finite() {
            return Err(crate::value_error("Invalid value NaN or Infinity for timestamp"));
        }
        let secs = timestamp.floor();
        let micros = ((timestamp - secs) * 1_000_000.0).round() as u32;
        let (secs, micros) = if micros >= 1_000_000 {
            (secs as i64 + 1, 0)
        } else {
            (secs as i64, micros)
        };
        Self::from_unix_local(secs, micros)
    }

    /// Convert to timestamp. A naive datetime is interpreted as LOCAL time
    /// (Python semantics), via mktime on unix; pre-1970 datetimes produce
    /// negative timestamps instead of wrapping.
    pub fn timestamp(&self) -> f64 {
        let micros = self.time.microsecond as f64 / 1_000_000.0;
        self.unix_seconds_local() as f64 + micros
    }

    #[cfg(unix)]
    fn unix_seconds_local(&self) -> i64 {
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        tm.tm_year = self.date.year - 1900;
        tm.tm_mon = self.date.month as i32 - 1;
        tm.tm_mday = self.date.day as i32;
        tm.tm_hour = self.time.hour as i32;
        tm.tm_min = self.time.minute as i32;
        tm.tm_sec = self.time.second as i32;
        tm.tm_isdst = -1; // let mktime resolve DST, like Python
        let t = unsafe { libc::mktime(&mut tm) };
        if t == -1 {
            // -1 is ambiguous: mktime's error value, but ALSO the valid
            // epoch-seconds for the local wall clock one second before
            // 1970. Disambiguate portably (errno's location differs per
            // platform) by decomposing -1 back and comparing fields.
            if let Ok(at_minus_one) = Self::from_unix_local(-1, 0) {
                if at_minus_one.date == self.date
                    && at_minus_one.time.hour == self.time.hour
                    && at_minus_one.time.minute == self.time.minute
                    && at_minus_one.time.second == self.time.second
                {
                    return -1;
                }
            }
            // A real mktime failure: fall back to the UTC computation
            // (signed, so pre-1970 stays negative instead of wrapping).
            return self.unix_seconds_utc();
        }
        t as i64
    }

    #[cfg(not(unix))]
    fn unix_seconds_local(&self) -> i64 {
        self.unix_seconds_utc()
    }

    fn unix_seconds_utc(&self) -> i64 {
        let days_since_epoch = self.date.toordinal() - 719163;
        let seconds_since_midnight = self.time.hour as i64 * 3600
            + self.time.minute as i64 * 60
            + self.time.second as i64;
        days_since_epoch * 86400 + seconds_since_midnight
    }
    
    /// Get date component
    pub fn date_component(&self) -> date {
        self.date
    }
    
    /// Get time component
    pub fn time_component(&self) -> time {
        self.time
    }
    
    /// Format as ISO string
    pub fn isoformat(&self, sep: Option<char>, timespec: Option<&str>) -> String {
        let sep = sep.unwrap_or('T');
        format!("{}{}{}", self.date.isoformat(), sep, self.time.isoformat(timespec))
    }
    
    /// Format with strftime
    pub fn strftime(&self, fmt: &str) -> String {
        self.date.strftime(&self.time.strftime(fmt))
    }
    
    /// Replace components
    pub fn replace(&self, year: Option<i32>, month: Option<u32>, day: Option<u32>,
                   hour: Option<u32>, minute: Option<u32>, second: Option<u32>, 
                   microsecond: Option<u32>) -> Result<Self, PyException> {
        let new_date = self.date.replace(year, month, day)?;
        let new_time = self.time.replace(hour, minute, second, microsecond)?;
        Ok(Self { date: new_date, time: new_time })
    }
}

impl fmt::Display for datetime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Python's str(datetime) separates date and time with a SPACE
        // (isoformat's default 'T' is only for isoformat()).
        write!(f, "{}", self.isoformat(Some(' '), None))
    }
}

/// timedelta - represents a duration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct timedelta {
    pub days: i64,
    pub seconds: i64,
    pub microseconds: i64,
}

impl timedelta {
    /// Create a new timedelta
    pub fn new(days: Option<i64>, seconds: Option<i64>, microseconds: Option<i64>,
               milliseconds: Option<i64>, minutes: Option<i64>, hours: Option<i64>,
               weeks: Option<i64>) -> Self {
        let mut total_days = days.unwrap_or(0);
        let mut total_seconds = seconds.unwrap_or(0);
        let mut total_microseconds = microseconds.unwrap_or(0);
        
        if let Some(ms) = milliseconds {
            total_microseconds += ms * 1000;
        }
        if let Some(min) = minutes {
            total_seconds += min * 60;
        }
        if let Some(hr) = hours {
            total_seconds += hr * 3600;
        }
        if let Some(wk) = weeks {
            total_days += wk * 7;
        }
        
        // Normalize
        if total_microseconds >= 1_000_000 {
            total_seconds += total_microseconds / 1_000_000;
            total_microseconds %= 1_000_000;
        } else if total_microseconds < 0 {
            total_seconds += (total_microseconds - 999_999) / 1_000_000;
            total_microseconds = ((total_microseconds % 1_000_000) + 1_000_000) % 1_000_000;
        }
        
        if total_seconds >= 86400 {
            total_days += total_seconds / 86400;
            total_seconds %= 86400;
        } else if total_seconds < 0 {
            total_days += (total_seconds - 86399) / 86400;
            total_seconds = ((total_seconds % 86400) + 86400) % 86400;
        }
        
        Self {
            days: total_days,
            seconds: total_seconds,
            microseconds: total_microseconds,
        }
    }
    
    /// Get total seconds
    pub fn total_seconds(&self) -> f64 {
        self.days as f64 * 86400.0 + self.seconds as f64 + self.microseconds as f64 / 1_000_000.0
    }

    /// Total duration in microseconds — the exact integer form the
    /// datetime operators compute with.
    fn total_micros(&self) -> i128 {
        self.days as i128 * 86_400_000_000 + self.seconds as i128 * 1_000_000
            + self.microseconds as i128
    }

    fn from_total_micros(micros: i128) -> Self {
        // Python's normalization: microseconds and seconds non-negative,
        // days carries the sign — exactly what new() produces.
        let days = micros.div_euclid(86_400_000_000);
        let rem = micros.rem_euclid(86_400_000_000);
        Self {
            days: days as i64,
            seconds: (rem / 1_000_000) as i64,
            microseconds: (rem % 1_000_000) as i64,
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic operators, so `d2 - d1` and `dt + timedelta(...)` lower to
// plain Rust operators. Python raises OverflowError when a result leaves
// date's range (year 1..9999); operator traits can't return Result, so
// that surfaces as a loud panic carrying the Python exception display.
// ---------------------------------------------------------------------------

/// Python date range in ordinal days: 0001-01-01 ..= 9999-12-31.
const MAX_ORDINAL: i64 = 3_652_059;

fn checked_ordinal(ordinal: i64) -> date {
    if !(1..=MAX_ORDINAL).contains(&ordinal) {
        panic!(
            "{}",
            crate::PyException::new("OverflowError", "date value out of range")
        );
    }
    days_to_date(ordinal)
}

impl core::ops::Sub for date {
    type Output = timedelta;
    fn sub(self, rhs: date) -> timedelta {
        timedelta {
            days: self.toordinal() - rhs.toordinal(),
            seconds: 0,
            microseconds: 0,
        }
    }
}

impl core::ops::Add<timedelta> for date {
    type Output = date;
    fn add(self, rhs: timedelta) -> date {
        // Python's date math uses only timedelta.days: the sub-day part
        // is ignored, so date(2024,1,1) + timedelta(hours=25) is Jan 2.
        checked_ordinal(self.toordinal() + rhs.days)
    }
}

impl core::ops::Sub<timedelta> for date {
    type Output = date;
    fn sub(self, rhs: timedelta) -> date {
        checked_ordinal(self.toordinal() - rhs.days)
    }
}

impl datetime {
    fn total_micros(&self) -> i128 {
        self.date.toordinal() as i128 * 86_400_000_000
            + (self.time.hour as i128 * 3600 + self.time.minute as i128 * 60
                + self.time.second as i128)
                * 1_000_000
            + self.time.microsecond as i128
    }

    fn from_total_micros(micros: i128) -> Self {
        let ordinal = micros.div_euclid(86_400_000_000);
        let rem = micros.rem_euclid(86_400_000_000);
        let date = checked_ordinal(ordinal as i64);
        let secs = rem / 1_000_000;
        Self {
            date,
            time: time::new(
                (secs / 3600) as u32,
                ((secs % 3600) / 60) as u32,
                Some((secs % 60) as u32),
                Some((rem % 1_000_000) as u32),
            )
            .expect("decomposed fields are in range"),
        }
    }
}

impl core::ops::Sub for datetime {
    type Output = timedelta;
    fn sub(self, rhs: datetime) -> timedelta {
        timedelta::from_total_micros(self.total_micros() - rhs.total_micros())
    }
}

impl core::ops::Add<timedelta> for datetime {
    type Output = datetime;
    fn add(self, rhs: timedelta) -> datetime {
        datetime::from_total_micros(self.total_micros() + rhs.total_micros())
    }
}

impl core::ops::Sub<timedelta> for datetime {
    type Output = datetime;
    fn sub(self, rhs: timedelta) -> datetime {
        datetime::from_total_micros(self.total_micros() - rhs.total_micros())
    }
}

impl core::ops::Add for timedelta {
    type Output = timedelta;
    fn add(self, rhs: timedelta) -> timedelta {
        timedelta::from_total_micros(self.total_micros() + rhs.total_micros())
    }
}

impl core::ops::Sub for timedelta {
    type Output = timedelta;
    fn sub(self, rhs: timedelta) -> timedelta {
        timedelta::from_total_micros(self.total_micros() - rhs.total_micros())
    }
}

impl core::ops::Neg for timedelta {
    type Output = timedelta;
    fn neg(self) -> timedelta {
        timedelta::from_total_micros(-self.total_micros())
    }
}

impl core::ops::Mul<i64> for timedelta {
    type Output = timedelta;
    fn mul(self, rhs: i64) -> timedelta {
        timedelta::from_total_micros(self.total_micros() * rhs as i128)
    }
}

impl core::ops::Mul<timedelta> for i64 {
    type Output = timedelta;
    fn mul(self, rhs: timedelta) -> timedelta {
        rhs * self
    }
}

// Python `+` lowers through PyAdd (it must handle string concatenation),
// so the date types implement it too, delegating to the operators above.
impl crate::PyAdd<timedelta> for date {
    type Output = date;
    fn py_add(&self, rhs: &timedelta) -> date {
        *self + *rhs
    }
}

impl crate::PyAdd<timedelta> for datetime {
    type Output = datetime;
    fn py_add(&self, rhs: &timedelta) -> datetime {
        *self + *rhs
    }
}

impl crate::PyAdd<timedelta> for timedelta {
    type Output = timedelta;
    fn py_add(&self, rhs: &timedelta) -> timedelta {
        *self + *rhs
    }
}

// ---------------------------------------------------------------------------
// strptime
// ---------------------------------------------------------------------------

impl datetime {
    /// Python datetime.strptime(text, format). Supported directives:
    /// %Y %m %d %H %M %S %f %b %B %I %p %%; anything else is a loud
    /// ValueError with Python's message. Missing fields default to
    /// 1900-01-01 00:00:00, as in Python.
    pub fn strptime(text: &str, format: &str) -> Result<Self, PyException> {
        let mismatch = || {
            crate::value_error(format!(
                "time data '{}' does not match format '{}'",
                text, format
            ))
        };
        let mut year: i32 = 1900;
        let mut month: u32 = 1;
        let mut day: u32 = 1;
        let mut hour: u32 = 0;
        let mut minute: u32 = 0;
        let mut second: u32 = 0;
        let mut microsecond: u32 = 0;
        let mut hour12: Option<u32> = None;
        let mut pm: Option<bool> = None;

        let input: Vec<char> = text.chars().collect();
        let mut pos = 0usize;
        let mut fmt = format.chars().peekable();

        // Parse 1..=max digits greedily; Err is the whole-input mismatch.
        let take_number = |pos: &mut usize, max: usize| -> Option<i64> {
            let start = *pos;
            while *pos < input.len() && *pos - start < max && input[*pos].is_ascii_digit() {
                *pos += 1;
            }
            if *pos == start {
                return None;
            }
            input[start..*pos].iter().collect::<String>().parse().ok()
        };

        while let Some(c) = fmt.next() {
            if c != '%' {
                if pos < input.len() && input[pos] == c {
                    pos += 1;
                    continue;
                }
                return Err(mismatch());
            }
            let directive = fmt.next().ok_or_else(|| {
                crate::value_error(format!("stray %% in format '{}'", format))
            })?;
            match directive {
                '%' => {
                    if pos < input.len() && input[pos] == '%' {
                        pos += 1;
                    } else {
                        return Err(mismatch());
                    }
                }
                'Y' => year = take_number(&mut pos, 4).ok_or_else(mismatch)? as i32,
                'm' => month = take_number(&mut pos, 2).ok_or_else(mismatch)? as u32,
                'd' => day = take_number(&mut pos, 2).ok_or_else(mismatch)? as u32,
                'H' => hour = take_number(&mut pos, 2).ok_or_else(mismatch)? as u32,
                'I' => hour12 = Some(take_number(&mut pos, 2).ok_or_else(mismatch)? as u32),
                'M' => minute = take_number(&mut pos, 2).ok_or_else(mismatch)? as u32,
                'S' => second = take_number(&mut pos, 2).ok_or_else(mismatch)? as u32,
                'f' => {
                    // 1..=6 digits, right-padded: ".25" is 250000 µs.
                    let start = pos;
                    let n = take_number(&mut pos, 6).ok_or_else(mismatch)?;
                    let width = pos - start;
                    microsecond = (n as u32) * 10u32.pow(6 - width as u32);
                }
                'b' | 'B' => {
                    let full = directive == 'B';
                    let mut found = None;
                    for m in 1..=12u32 {
                        let name = if full { month_name(m) } else { month_abbr(m) };
                        let matches = input[pos..]
                            .iter()
                            .take(name.chars().count())
                            .collect::<String>()
                            .eq_ignore_ascii_case(name);
                        if matches {
                            found = Some((m, name.chars().count()));
                            break;
                        }
                    }
                    let (m, len) = found.ok_or_else(mismatch)?;
                    month = m;
                    pos += len;
                }
                'p' => {
                    let rest: String = input[pos..].iter().take(2).collect();
                    if rest.eq_ignore_ascii_case("am") {
                        pm = Some(false);
                        pos += 2;
                    } else if rest.eq_ignore_ascii_case("pm") {
                        pm = Some(true);
                        pos += 2;
                    } else {
                        return Err(mismatch());
                    }
                }
                other => {
                    return Err(crate::value_error(format!(
                        "'{}' is a bad directive in format '{}'",
                        other, format
                    )));
                }
            }
        }
        if pos < input.len() {
            let rest: String = input[pos..].iter().collect();
            return Err(crate::value_error(format!(
                "unconverted data remains: {}",
                rest
            )));
        }
        if let Some(h12) = hour12 {
            if !(1..=12).contains(&h12) {
                return Err(mismatch());
            }
            hour = match pm {
                Some(true) => {
                    if h12 == 12 { 12 } else { h12 + 12 }
                }
                _ => {
                    if h12 == 12 { 0 } else { h12 }
                }
            };
        }
        let date = date::new(year, month, day).map_err(|_| mismatch())?;
        let time = time::new(hour, minute, Some(second), Some(microsecond))
            .map_err(|_| mismatch())?;
        Ok(Self { date, time })
    }
}

impl fmt::Display for timedelta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.days != 0 {
            // Python pluralizes on |days| — "-1 day, 1:00:00" is singular.
            let plural = if self.days.abs() == 1 { "" } else { "s" };
            write!(f, "{} day{}, ", self.days, plural)?;
        }
        
        let hours = self.seconds / 3600;
        let minutes = (self.seconds % 3600) / 60;
        let seconds = self.seconds % 60;
        
        if self.microseconds != 0 {
            write!(f, "{}:{:02}:{:02}.{:06}", hours, minutes, seconds, self.microseconds)
        } else {
            write!(f, "{}:{:02}:{:02}", hours, minutes, seconds)
        }
    }
}

// Helper functions
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_to_date(ordinal: i64) -> date {
    // Simplified algorithm - not historically accurate for very old dates
    let mut year = 1;
    let mut remaining_days = ordinal - 1;
    
    // Find the year
    while remaining_days >= 365 {
        let year_days = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < year_days {
            break;
        }
        remaining_days -= year_days;
        year += 1;
    }
    
    // Find the month and day
    let mut month = 1;
    while month <= 12 {
        let month_days = if month == 2 && is_leap_year(year) { 29 } else { DAYS_IN_MONTH[month as usize - 1] };
        if remaining_days < month_days as i64 {
            break;
        }
        remaining_days -= month_days as i64;
        month += 1;
    }
    
    date::new(year, month, remaining_days as u32 + 1).unwrap()
}

fn date_to_days(d: date) -> i64 {
    let mut days = 0i64;
    
    // Add days for complete years
    for y in 1..d.year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }
    
    // Add days for complete months in the current year
    for m in 1..d.month {
        let month_days = if m == 2 && is_leap_year(d.year) { 29 } else { DAYS_IN_MONTH[m as usize - 1] };
        days += month_days as i64;
    }
    
    // Add remaining days
    days + d.day as i64
}

fn month_name(month: u32) -> &'static str {
    match month {
        1 => "January", 2 => "February", 3 => "March", 4 => "April",
        5 => "May", 6 => "June", 7 => "July", 8 => "August",
        9 => "September", 10 => "October", 11 => "November", 12 => "December",
        _ => "Unknown"
    }
}

fn month_abbr(month: u32) -> &'static str {
    match month {
        1 => "Jan", 2 => "Feb", 3 => "Mar", 4 => "Apr",
        5 => "May", 6 => "Jun", 7 => "Jul", 8 => "Aug",
        9 => "Sep", 10 => "Oct", 11 => "Nov", 12 => "Dec",
        _ => "Unk"
    }
}

fn weekday_name(weekday: u32) -> &'static str {
    match weekday {
        0 => "Monday", 1 => "Tuesday", 2 => "Wednesday", 3 => "Thursday",
        4 => "Friday", 5 => "Saturday", 6 => "Sunday",
        _ => "Unknown"
    }
}

fn weekday_abbr(weekday: u32) -> &'static str {
    match weekday {
        0 => "Mon", 1 => "Tue", 2 => "Wed", 3 => "Thu",
        4 => "Fri", 5 => "Sat", 6 => "Sun",
        _ => "Unk"
    }
}

// Constants
pub const MINYEAR: i32 = 1;
pub const MAXYEAR: i32 = 9999;

// Module-level functions
/// Get current UTC time
pub fn utcnow() -> datetime {
    datetime::utcnow()
}