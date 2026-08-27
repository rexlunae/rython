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
        check_year(year)?;
        if !(1..=12).contains(&month) {
            return Err(crate::value_error("month must be in 1..12"));
        }
        
        let max_day = if month == 2 && is_leap_year(year) { 29 } else { DAYS_IN_MONTH[month as usize - 1] };
        
        if !(1..=max_day).contains(&day) {
            return Err(crate::value_error(format!("day must be in 1..{}", max_day)));
        }
        
        Ok(Self { year, month, day })
    }
    
    /// Get today's date in LOCAL time, like Python's date.today() (via
    /// localtime on unix; non-unix hosts fall back to UTC). The old
    /// implementation decomposed UTC seconds directly, which was off by a
    /// day for any timezone whose local date differs from UTC (issue #82).
    pub fn today() -> Self {
        let now = SystemTime::now();
        let duration = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0));
        datetime::from_unix_local(duration.as_secs() as i64, duration.subsec_micros())
            .unwrap_or_else(|_| datetime::from_unix_utc(duration.as_secs() as i64, duration.subsec_micros()))
            .date_component()
    }
    
    /// Create date from ordinal day
    pub fn fromordinal(ordinal: i64) -> Result<Self, PyException> {
        if ordinal < 1 {
            return Err(crate::value_error("ordinal must be >= 1"));
        }
        let d = days_to_date(ordinal);
        // CPython validates the RESULTING year: fromordinal(3652060)
        // (10000-01-01) raises "year must be in 1..9999, not 10000".
        check_year(d.year)?;
        Ok(d)
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
    
    /// Get ISO calendar (year, week, weekday). ISO weeks start on a
    /// Monday and week 1 is the one containing the first Thursday, so
    /// early-January and late-December dates can belong to the
    /// neighbouring ISO YEAR: 2023-01-01 is (2022, 52, 7) and 2024-12-30
    /// is (2025, 1, 1).
    pub fn isocalendar(&self) -> (i32, u32, u32) {
        // The Monday of this date's ISO week.
        let ordinal = self.toordinal();
        let weekday = self.weekday() as i64; // 0 = Monday
        let week_monday = ordinal - weekday;
        // The ISO year is the calendar year of that week's Thursday.
        let thursday = week_monday + 3;
        let iso_year = days_to_date(thursday).year;
        // Week 1 is the week whose Thursday falls in the ISO year, i.e.
        // it starts on the Monday on or before that year's January 4th.
        // Computed from ordinals (not date::new) because a late-December
        // date near MAXYEAR can have iso_year == 10000, which is out of
        // the validated range (issue #82).
        let jan4_ordinal = date_to_days(date { year: iso_year, month: 1, day: 4 });
        let jan4_weekday = (jan4_ordinal - 1).rem_euclid(7); // 0 = Monday
        let week1_monday = jan4_ordinal - jan4_weekday;
        let week = ((week_monday - week1_monday) / 7 + 1) as u32;
        (iso_year, week, self.isoweekday())
    }
    
    /// Format as ISO string
    pub fn isoformat(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
    
    /// Format with strftime
    pub fn strftime(&self, fmt: &str) -> String {
        strftime_pass(fmt, Some(*self), None)
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
        strftime_pass(fmt, None, Some(*self))
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

/// datetime - represents a datetime. The fields are FLAT, as in Python:
/// dt.year through dt.microsecond are attributes, while dt.date() and
/// dt.time() are methods. The derived ordering over the field order is
/// exactly chronological order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct datetime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub microsecond: u32,
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
        Ok(Self::from_parts(date, time))
    }

    fn from_parts(date: date, time: time) -> Self {
        Self {
            year: date.year,
            month: date.month,
            day: date.day,
            hour: time.hour,
            minute: time.minute,
            second: time.second,
            microsecond: time.microsecond,
        }
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
        Self::from_parts(
            date,
            time::new(hour, minute, Some(second), Some(microsecond))
                .expect("decomposed clock fields are in range"),
        )
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
        Ok(Self::from_parts(
            date::new(tm.tm_year + 1900, (tm.tm_mon + 1) as u32, tm.tm_mday as u32)?,
            time::new(
                tm.tm_hour as u32,
                tm.tm_min as u32,
                Some(tm.tm_sec as u32),
                Some(microsecond),
            )?,
        ))
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
        let micros = self.microsecond as f64 / 1_000_000.0;
        self.unix_seconds_local() as f64 + micros
    }

    #[cfg(unix)]
    fn unix_seconds_local(&self) -> i64 {
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        tm.tm_year = self.year - 1900;
        tm.tm_mon = self.month as i32 - 1;
        tm.tm_mday = self.day as i32;
        tm.tm_hour = self.hour as i32;
        tm.tm_min = self.minute as i32;
        tm.tm_sec = self.second as i32;
        tm.tm_isdst = -1; // let mktime resolve DST, like Python
        let t = unsafe { libc::mktime(&mut tm) };
        if t == -1 {
            // -1 is ambiguous: mktime's error value, but ALSO the valid
            // epoch-seconds for the local wall clock one second before
            // 1970. Disambiguate portably (errno's location differs per
            // platform) by decomposing -1 back and comparing fields.
            if let Ok(at_minus_one) = Self::from_unix_local(-1, 0) {
                if at_minus_one.date_component() == self.date_component()
                    && at_minus_one.hour == self.hour
                    && at_minus_one.minute == self.minute
                    && at_minus_one.second == self.second
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
        let days_since_epoch = self.date_component().toordinal() - 719163;
        let seconds_since_midnight = self.hour as i64 * 3600
            + self.minute as i64 * 60
            + self.second as i64;
        days_since_epoch * 86400 + seconds_since_midnight
    }
    
    /// Get date component
    pub fn date_component(&self) -> date {
        date {
            year: self.year,
            month: self.month,
            day: self.day,
        }
    }
    
    /// Get time component
    pub fn time_component(&self) -> time {
        time {
            hour: self.hour,
            minute: self.minute,
            second: self.second,
            microsecond: self.microsecond,
        }
    }

    /// Python dt.date()
    pub fn date(&self) -> date {
        self.date_component()
    }

    /// Python dt.time()
    pub fn time(&self) -> time {
        self.time_component()
    }
    
    /// Format as ISO string
    pub fn isoformat(&self, sep: Option<char>, timespec: Option<&str>) -> String {
        let sep = sep.unwrap_or('T');
        format!("{}{}{}", self.date_component().isoformat(), sep, self.time_component().isoformat(timespec))
    }
    
    /// Format with strftime
    pub fn strftime(&self, fmt: &str) -> String {
        strftime_pass(fmt, Some(self.date_component()), Some(self.time_component()))
    }
    
    /// Replace components
    pub fn replace(&self, year: Option<i32>, month: Option<u32>, day: Option<u32>,
                   hour: Option<u32>, minute: Option<u32>, second: Option<u32>, 
                   microsecond: Option<u32>) -> Result<Self, PyException> {
        let new_date = self.date_component().replace(year, month, day)?;
        let new_time = self.time_component().replace(hour, minute, second, microsecond)?;
        Ok(Self::from_parts(new_date, new_time))
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
        self.date_component().toordinal() as i128 * 86_400_000_000
            + (self.hour as i128 * 3600 + self.minute as i128 * 60
                + self.second as i128)
                * 1_000_000
            + self.microsecond as i128
    }

    fn from_total_micros(micros: i128) -> Self {
        let ordinal = micros.div_euclid(86_400_000_000);
        let rem = micros.rem_euclid(86_400_000_000);
        let date = checked_ordinal(ordinal as i64);
        let secs = rem / 1_000_000;
        Self::from_parts(
            date,
            time::new(
                (secs / 3600) as u32,
                ((secs % 3600) / 60) as u32,
                Some((secs % 60) as u32),
                Some((rem % 1_000_000) as u32),
            )
            .expect("decomposed fields are in range"),
        )
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

// Python `-` lowers through PySub (borrowed operands), so the date types
// implement it too, delegating to the native `-` operators above.
impl crate::PySub<date> for date {
    type Output = timedelta;
    fn py_sub(&self, rhs: &date) -> timedelta {
        *self - *rhs
    }
}
impl crate::PySub<timedelta> for date {
    type Output = date;
    fn py_sub(&self, rhs: &timedelta) -> date {
        *self - *rhs
    }
}
impl crate::PySub<datetime> for datetime {
    type Output = timedelta;
    fn py_sub(&self, rhs: &datetime) -> timedelta {
        *self - *rhs
    }
}
impl crate::PySub<timedelta> for datetime {
    type Output = datetime;
    fn py_sub(&self, rhs: &timedelta) -> datetime {
        *self - *rhs
    }
}
impl crate::PySub<timedelta> for timedelta {
    type Output = timedelta;
    fn py_sub(&self, rhs: &timedelta) -> timedelta {
        *self - *rhs
    }
}

// ---------------------------------------------------------------------------
// strptime
// ---------------------------------------------------------------------------

impl datetime {
    /// Python datetime.strptime(text, format). Supported directives:
    /// %Y %m %d %H %M %S %f %b %B %I %p %a %A %j %%; anything else is a
    /// loud ValueError with Python's message. Missing fields default to
    /// 1900-01-01 00:00:00, as in Python. A weekday name (%a/%A) is
    /// parsed but not validated against the date, exactly as CPython;
    /// %j resolves month and day from the year, rolling into the next
    /// year when it exceeds the year's length (Python computes the date
    /// by ordinal arithmetic).
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
        let mut julian: Option<i64> = None;

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
                'a' | 'A' => {
                    // CPython parses the weekday name and, with no
                    // week-number directive, IGNORES it — even one
                    // inconsistent with the date. Case-insensitive; %a
                    // matches only abbreviated names, %A only full ones.
                    let full = directive == 'A';
                    let mut consumed = None;
                    for w in 0..7u32 {
                        let name = if full { weekday_name(w) } else { weekday_abbr(w) };
                        let n = name.chars().count();
                        let matches = input[pos..]
                            .iter()
                            .take(n)
                            .collect::<String>()
                            .eq_ignore_ascii_case(name);
                        if matches {
                            consumed = Some(n);
                            break;
                        }
                    }
                    pos += consumed.ok_or_else(mismatch)?;
                }
                'j' => {
                    // Day of year, 1..=366. CPython's pattern matches the
                    // LONGEST digit prefix (up to 3) that stays in range:
                    // "367" matches "36" and leaves "7" unconverted.
                    let mut len = 0usize;
                    while pos + len < input.len()
                        && len < 3
                        && input[pos + len].is_ascii_digit()
                    {
                        len += 1;
                    }
                    let mut chosen = None;
                    while len > 0 {
                        let v: i64 = input[pos..pos + len]
                            .iter()
                            .collect::<String>()
                            .parse()
                            .expect("digits parse");
                        if (1..=366).contains(&v) {
                            chosen = Some((v, len));
                            break;
                        }
                        len -= 1;
                    }
                    let (v, len) = chosen.ok_or_else(mismatch)?;
                    julian = Some(v);
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
        if let Some(j) = julian {
            // Python: date.fromordinal(date(year,1,1).toordinal() + j-1),
            // so day 366 of a 365-day year is Jan 1 of the next year.
            let start = date::new(year, 1, 1).map_err(|_| mismatch())?;
            let resolved = checked_ordinal(start.toordinal() + j - 1);
            year = resolved.year;
            month = resolved.month;
            day = resolved.day;
        }
        let date = date::new(year, month, day).map_err(|_| mismatch())?;
        let time = time::new(hour, minute, Some(second), Some(microsecond))
            .map_err(|_| mismatch())?;
        Ok(Self::from_parts(date, time))
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
    // Exact civil-from-days algorithm; constructs the date directly so it
    // stays usable for internal decomposition (from_unix_*, isocalendar).
    // Public constructors validate the year afterwards (date::new,
    // date::fromordinal), matching CPython's error messages.
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
    
    date {
        year,
        month,
        day: remaining_days as u32 + 1,
    }
}

fn date_to_days(d: date) -> i64 {
    let mut days = 0i64;
    
    // Add days for complete years. Public constructors guarantee year >= 1
    // (date::new validates against MINYEAR), so `1..d.year` is never the
    // empty range that silently collapsed years <= 1 into year 1 (issue #82).
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

/// CPython's year-range check: "year must be in 1..9999, not 0".
fn check_year(year: i32) -> Result<(), PyException> {
    if !(MINYEAR..=MAXYEAR).contains(&year) {
        return Err(crate::value_error(format!(
            "year must be in {}..{}, not {}",
            MINYEAR, MAXYEAR, year
        )));
    }
    Ok(())
}

/// A single left-to-right strftime pass (issue #82). The old chained
/// `str::replace` re-processed inserted text — `%%d` became `%05` and
/// `100%%` stayed `100%%` — and left every unimplemented directive as
/// literal text. CPython's strftime scans once: `%%` is a literal percent,
/// a known directive is substituted with the component's value, and an
/// unknown directive stays as-is (`%q` -> `%q`).
///
/// A date-only receiver substitutes zeros for the time components and a
/// time-only receiver substitutes 1900-01-01 for the date components,
/// exactly like CPython (date.strftime('%H') -> "00",
/// time.strftime('%Y') -> "1900").
fn strftime_pass(fmt: &str, d: Option<date>, t: Option<time>) -> String {
    let d = d.unwrap_or(date { year: 1900, month: 1, day: 1 });
    let t = t.unwrap_or(time { hour: 0, minute: 0, second: 0, microsecond: 0 });

    let ord = d.toordinal();
    let jan1_ord = date_to_days(date { year: d.year, month: 1, day: 1 });
    let doy = ord - jan1_ord + 1; // 1-based day of year (%j, %U, %W)
    let jan1_mon = (jan1_ord - 1).rem_euclid(7); // 0 = Monday
    let jan1_sun = jan1_ord.rem_euclid(7); // 0 = Sunday
    let first_monday = ((7 - jan1_mon) % 7) + 1;
    let first_sunday = ((7 - jan1_sun) % 7) + 1;
    let week_monday = if doy < first_monday { 0 } else { (doy - first_monday) / 7 + 1 };
    let week_sunday = if doy < first_sunday { 0 } else { (doy - first_sunday) / 7 + 1 };
    let (iso_year, iso_week, _) = d.isocalendar();
    let hour12 = {
        let h = t.hour % 12;
        if h == 0 { 12 } else { h }
    };

    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(dir) = chars.next() else {
            out.push('%');
            break;
        };
        match dir {
            '%' => out.push('%'),
            'Y' => out.push_str(&format!("{:04}", d.year)),
            'y' => out.push_str(&format!("{:02}", d.year.rem_euclid(100))),
            'm' => out.push_str(&format!("{:02}", d.month)),
            'd' => out.push_str(&format!("{:02}", d.day)),
            'e' => out.push_str(&format!("{:2}", d.day)),
            'B' => out.push_str(month_name(d.month)),
            'b' => out.push_str(month_abbr(d.month)),
            'A' => out.push_str(weekday_name(d.weekday())),
            'a' => out.push_str(weekday_abbr(d.weekday())),
            'j' => out.push_str(&format!("{:03}", doy)),
            'w' => out.push_str(&format!("{}", ord.rem_euclid(7))), // 0 = Sunday
            'u' => out.push_str(&format!("{}", d.isoweekday())),
            'U' => out.push_str(&format!("{:02}", week_sunday)),
            'W' => out.push_str(&format!("{:02}", week_monday)),
            'G' => out.push_str(&format!("{:04}", iso_year)),
            'V' => out.push_str(&format!("{:02}", iso_week)),
            'H' => out.push_str(&format!("{:02}", t.hour)),
            'I' => out.push_str(&format!("{:02}", hour12)),
            'M' => out.push_str(&format!("{:02}", t.minute)),
            'S' => out.push_str(&format!("{:02}", t.second)),
            'f' => out.push_str(&format!("{:06}", t.microsecond)),
            'p' => out.push_str(if t.hour < 12 { "AM" } else { "PM" }),
            // A naive datetime has no tz: %z/%Z are empty, like CPython.
            'z' => {}
            'Z' => {}
            'c' => out.push_str(&format!(
                "{} {} {:2} {:02}:{:02}:{:02} {:04}",
                weekday_abbr(d.weekday()),
                month_abbr(d.month),
                d.day,
                t.hour,
                t.minute,
                t.second,
                d.year
            )),
            'x' => out.push_str(&format!(
                "{:02}/{:02}/{:02}",
                d.month,
                d.day,
                d.year.rem_euclid(100)
            )),
            'X' => out.push_str(&format!("{:02}:{:02}:{:02}", t.hour, t.minute, t.second)),
            other => {
                // Unknown directive: CPython leaves it literal (%q -> %q).
                out.push('%');
                out.push(other);
            }
        }
    }
    out
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
// str(x) of the datetime family is its isoformat-style Display, already
// verified against CPython; print defers to it.
impl crate::PyDisplay for date {
    fn py_display(&self) -> String {
        self.to_string()
    }
}

impl crate::PyDisplay for time {
    fn py_display(&self) -> String {
        self.to_string()
    }
}

impl crate::PyDisplay for datetime {
    fn py_display(&self) -> String {
        self.to_string()
    }
}

impl crate::PyDisplay for timedelta {
    fn py_display(&self) -> String {
        self.to_string()
    }
}

// ---------------------------------------------------------------------------
// Keyword-shaped replace()
// ---------------------------------------------------------------------------

/// Keyword arguments for replace(), as one struct so the CONVERTER can
/// emit a single lowering without knowing whether the receiver is a
/// datetime, date, or time — each receiver validates its own field set
/// and raises Python's exact TypeError for fields it does not have.
/// Fields are i64 (the converter's integer type); range errors are
/// ValueError, as in Python.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReplaceArgs {
    pub year: Option<i64>,
    pub month: Option<i64>,
    pub day: Option<i64>,
    pub hour: Option<i64>,
    pub minute: Option<i64>,
    pub second: Option<i64>,
    pub microsecond: Option<i64>,
}

/// dt.replace(field=...) for the datetime family.
pub trait PyReplace: Sized {
    fn py_replace(&self, args: ReplaceArgs) -> Result<Self, PyException>;
}

fn invalid_replace_keyword(field: &str) -> PyException {
    // CPython: TypeError: 'hour' is an invalid keyword argument for replace()
    PyException::new(
        "TypeError",
        format!("'{}' is an invalid keyword argument for replace()", field),
    )
}

fn narrow_i32(v: Option<i64>, field: &str) -> Result<Option<i32>, PyException> {
    v.map(|v| {
        i32::try_from(v)
            .map_err(|_| crate::value_error(format!("{} {} is out of range", field, v)))
    })
    .transpose()
}

fn narrow_u32(v: Option<i64>, field: &str) -> Result<Option<u32>, PyException> {
    v.map(|v| {
        u32::try_from(v)
            .map_err(|_| crate::value_error(format!("{} {} is out of range", field, v)))
    })
    .transpose()
}

impl PyReplace for datetime {
    fn py_replace(&self, args: ReplaceArgs) -> Result<Self, PyException> {
        self.replace(
            narrow_i32(args.year, "year")?,
            narrow_u32(args.month, "month")?,
            narrow_u32(args.day, "day")?,
            narrow_u32(args.hour, "hour")?,
            narrow_u32(args.minute, "minute")?,
            narrow_u32(args.second, "second")?,
            narrow_u32(args.microsecond, "microsecond")?,
        )
    }
}

impl PyReplace for date {
    fn py_replace(&self, args: ReplaceArgs) -> Result<Self, PyException> {
        for (field, value) in [
            ("hour", args.hour),
            ("minute", args.minute),
            ("second", args.second),
            ("microsecond", args.microsecond),
        ] {
            if value.is_some() {
                return Err(invalid_replace_keyword(field));
            }
        }
        self.replace(
            narrow_i32(args.year, "year")?,
            narrow_u32(args.month, "month")?,
            narrow_u32(args.day, "day")?,
        )
    }
}

impl PyReplace for time {
    fn py_replace(&self, args: ReplaceArgs) -> Result<Self, PyException> {
        for (field, value) in [
            ("year", args.year),
            ("month", args.month),
            ("day", args.day),
        ] {
            if value.is_some() {
                return Err(invalid_replace_keyword(field));
            }
        }
        self.replace(
            narrow_u32(args.hour, "hour")?,
            narrow_u32(args.minute, "minute")?,
            narrow_u32(args.second, "second")?,
            narrow_u32(args.microsecond, "microsecond")?,
        )
    }
}
