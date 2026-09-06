//! Python argparse module implementation
//!
//! The compiler evaluates ArgumentParser/add_argument calls at
//! conversion time (literal specs only) and lowers parse_args() into a
//! call to run_parser with the collected specs; the result feeds a
//! generated struct whose fields are the destinations. This module
//! reproduces CPython's observable behavior: the usage line, the help
//! layout (including the help-column computation), prefix
//! abbreviation, `--opt=value` and split forms, and the exact error
//! messages with exit code 2 (or 0 for --help), printed to the streams
//! Python uses.

use crate::PyException;

/// `argparse.FileType(mode)` — the file-opening coercion `type=` may
/// name. The converter reads the mode at conversion time (an ArgKind::
/// File / BinaryFile spec) and never constructs this; the item exists so
/// `from argparse import FileType` resolves. The field is the Python
/// mode string.
pub struct FileType(pub String);

#[derive(Clone, Copy, PartialEq)]
pub enum ArgKind {
    Str,
    Int,
    Float,
    StoreTrue,
    /// `type=FileType(mode)` in a TEXT mode: the value is the opened
    /// PyFile ("-" is stdin, read into a buffer, for a read mode).
    File(&'static str),
    /// `type=FileType(mode)` in a BINARY mode ('b' in the mode): the
    /// opened binary file.
    BinaryFile(&'static str),
    /// `action="version"`: `--version` prints the spec's default (the
    /// version string) and exits 0; the namespace has no field for it.
    Version,
}

/// How many values an argument takes (`nargs`): one (the default), one
/// or more (`"+"`), zero or more (`"*"`). Only positionals may be
/// variadic here, and at most one per parser (the converter refuses the
/// rest); the values are a ParsedValue::List.
#[derive(Clone, Copy, PartialEq)]
pub enum Nargs {
    One,
    Plus,
    Star,
}

/// A FileType argument's file, opened ONCE at parse time — the open that
/// validates the path (Python's argparse opens it then too, and reports
/// a failure as an argument error) is the handle the namespace gets, so
/// nothing is opened twice and a write mode truncates once (Devin review
/// on #339). `-` is the live standard stream for the mode: stdin for a
/// read mode, stdout for a write or append mode, text or binary.
#[derive(Clone)]
pub struct OpenedFile {
    pub path: String,
    pub mode: &'static str,
    handle: FileHandle,
}

#[derive(Clone)]
enum FileHandle {
    Text(crate::PyFile),
    Binary(crate::stdlib::io::PyBytesIO),
}

impl core::fmt::Debug for OpenedFile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "File({:?}, {:?})", self.path, self.mode)
    }
}

impl PartialEq for OpenedFile {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.mode == other.mode
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParsedValue {
    Str(String),
    Int(i64),
    Float(f64),
    Flag(bool),
    /// A FileType value: the file the parser opened (see [`OpenedFile`]).
    File(OpenedFile),
    List(Vec<ParsedValue>),
}

impl ParsedValue {
    pub fn into_str(self) -> String {
        match self {
            ParsedValue::Str(s) => s,
            other => panic!("argparse internal error: expected str, got {:?}", other),
        }
    }
    pub fn into_int(self) -> i64 {
        match self {
            ParsedValue::Int(i) => i,
            other => panic!("argparse internal error: expected int, got {:?}", other),
        }
    }
    pub fn into_float(self) -> f64 {
        match self {
            ParsedValue::Float(f) => f,
            other => panic!("argparse internal error: expected float, got {:?}", other),
        }
    }
    pub fn into_flag(self) -> bool {
        match self {
            ParsedValue::Flag(b) => b,
            other => panic!("argparse internal error: expected flag, got {:?}", other),
        }
    }
    /// The opened text file of a FileType argument: the handle the
    /// parse-time open produced (see [`OpenedFile`]).
    pub fn into_file(self) -> crate::PyFile {
        match self {
            ParsedValue::File(OpenedFile { handle: FileHandle::Text(file), .. }) => file,
            other => panic!("argparse internal error: expected text file, got {:?}", other),
        }
    }
    /// The opened binary file of a FileType argument (a 'b' mode).
    pub fn into_binary_file(self) -> crate::stdlib::io::PyBytesIO {
        match self {
            ParsedValue::File(OpenedFile { handle: FileHandle::Binary(file), .. }) => file,
            other => panic!("argparse internal error: expected binary file, got {:?}", other),
        }
    }
    /// The values of a variadic (`nargs`) argument.
    pub fn into_list(self) -> Vec<ParsedValue> {
        match self {
            ParsedValue::List(items) => items,
            other => panic!("argparse internal error: expected list, got {:?}", other),
        }
    }
}

pub struct ArgSpec {
    /// "count" for a positional, "--verbose" for an option.
    pub name: &'static str,
    /// The short alias of an option ("-c" for `add_argument("-c",
    /// "--contents")`); None for positionals and long-only options.
    pub short: Option<&'static str>,
    pub kind: ArgKind,
    /// `dest=` — the namespace attribute when it is not derived from the
    /// name (Python's metavar derives from it too); None otherwise.
    pub dest: Option<&'static str>,
    pub nargs: Nargs,
    /// Required for value-taking options (Python's None default cannot
    /// inhabit a typed field); positionals and store_true have implied
    /// handling.
    pub default: Option<ParsedValue>,
    pub help: Option<&'static str>,
}

impl ArgSpec {
    fn is_positional(&self) -> bool {
        !self.name.starts_with('-')
    }
    /// The attribute name on the namespace ("--scale" -> scale, or the
    /// explicit dest=).
    fn dest(&self) -> String {
        match self.dest {
            Some(d) => d.to_string(),
            None => self.name.trim_start_matches('-').replace('-', "_"),
        }
    }
    fn takes_value(&self) -> bool {
        !matches!(self.kind, ArgKind::StoreTrue | ArgKind::Version)
    }
    /// A positional's usage/help spelling by nargs: `files [files ...]`
    /// for "+", `[files ...]` for "*" (Python's formatter).
    fn positional_spelling(&self) -> String {
        match self.nargs {
            Nargs::One => self.name.to_string(),
            Nargs::Plus => format!("{} [{} ...]", self.name, self.name),
            Nargs::Star => format!("[{} ...]", self.name),
        }
    }
    /// The uppercase metavar of a value-taking option.
    fn metavar(&self) -> String {
        self.dest().to_uppercase()
    }
    /// How the argument appears in the USAGE line: positionals by name,
    /// options by their SHORT alias when they have one (Python's
    /// formatter uses the first option string).
    fn usage_invocation(&self) -> String {
        if self.is_positional() {
            self.positional_spelling()
        } else {
            let lead = self.short.unwrap_or(self.name);
            if !self.takes_value() {
                lead.to_string()
            } else {
                format!("{} {}", lead, self.metavar())
            }
        }
    }
    /// How the argument appears in the HELP list: positionals by name;
    /// options list every alias, value-taking ones with the metavar after
    /// each ("-s SCALE, --scale SCALE" — Python 3.11's format).
    fn invocation(&self) -> String {
        if self.is_positional() {
            return self.name.to_string();
        }
        let mut parts = Vec::new();
        for alias in self.short.iter().chain([&self.name]) {
            if !self.takes_value() {
                parts.push(alias.to_string());
            } else {
                parts.push(format!("{} {}", alias, self.metavar()));
            }
        }
        parts.join(", ")
    }
}

fn prog_name(explicit: Option<&str>) -> String {
    match explicit {
        Some(p) => p.to_string(),
        None => std::env::args()
            .next()
            .as_deref()
            .and_then(|p| p.rsplit('/').next().map(str::to_string))
            .unwrap_or_else(|| "prog".to_string()),
    }
}

fn usage_line(prog: &str, specs: &[ArgSpec]) -> String {
    let mut parts = vec![format!("usage: {} [-h]", prog)];
    for s in specs.iter().filter(|s| !s.is_positional()) {
        parts.push(format!("[{}]", s.usage_invocation()));
    }
    for s in specs.iter().filter(|s| s.is_positional()) {
        parts.push(s.positional_spelling());
    }
    parts.join(" ")
}

fn help_text(prog: &str, description: Option<&str>, specs: &[ArgSpec]) -> String {
    // Python's help column: two-space indent + the longest invocation
    // (capped at 24) + two spaces. Longer invocations push their help
    // onto the next line at that column.
    // Python's formula (HelpFormatter): the help column is indent(2) +
    // longest invocation + 2, capped at max_help_position=24.
    let help_spec = "-h, --help".to_string();
    let max_len = specs
        .iter()
        .map(|s| s.invocation().chars().count())
        .chain([help_spec.chars().count()])
        .max()
        .unwrap_or(0);
    let help_col = (2 + max_len + 2).min(24);

    let mut out = usage_line(prog, specs);
    out.push('\n');
    if let Some(d) = description {
        out.push('\n');
        out.push_str(&format_text(d, prog));
        out.push('\n');
    }
    let entry = |out: &mut String, invocation: &str, help: Option<&str>| {
        out.push_str("  ");
        out.push_str(invocation);
        match help {
            None => out.push('\n'),
            Some(h) => {
                let used = 2 + invocation.chars().count();
                if used + 2 > help_col {
                    out.push('\n');
                    out.push_str(&" ".repeat(help_col));
                } else {
                    out.push_str(&" ".repeat(help_col - used));
                }
                out.push_str(h);
                out.push('\n');
            }
        }
    };
    if specs.iter().any(|s| s.is_positional()) {
        out.push_str("\npositional arguments:\n");
        for s in specs.iter().filter(|s| s.is_positional()) {
            entry(&mut out, &s.invocation(), s.help.map(|h| expand_help(h, prog, s)).as_deref());
        }
    }
    out.push_str("\noptions:\n");
    entry(&mut out, &help_spec, Some("show this help message and exit"));
    for s in specs.iter().filter(|s| !s.is_positional()) {
        // Python's default help for action="version".
        let help = match (s.kind, s.help) {
            (ArgKind::Version, None) => Some("show program's version number and exit".to_string()),
            (_, h) => h.map(|h| expand_help(h, prog, s)),
        };
        entry(&mut out, &s.invocation(), help.as_deref());
    }
    out
}

/// `action="version"`: Python prints the version string to stdout and
/// Python's `text % params` for the `%(name)s` directives argparse
/// substitutes (`%(prog)s` in a version or description, `%(prog)s`,
/// `%(default)s`, `%(dest)s`, `%(type)s` in a help string) and `%%`.
/// Any other directive is what CPython raises for it (a KeyError for an
/// unknown name, a ValueError for an incomplete format) — loud.
fn percent_format(text: &str, params: &[(&str, String)]) -> Result<String, PyException> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('%') => out.push('%'),
            Some('(') => {
                let mut name = String::new();
                loop {
                    match chars.next() {
                        Some(')') => break,
                        Some(ch) => name.push(ch),
                        None => {
                            return Err(crate::value_error("incomplete format key"));
                        }
                    }
                }
                let conversion = chars.next();
                if conversion != Some('s') {
                    return Err(crate::value_error(&format!(
                        "unsupported format character {} in argparse text",
                        conversion.map(|c| py_repr(&c.to_string())).unwrap_or_else(|| "'(end)'".to_string())
                    )));
                }
                match params.iter().find(|(k, _)| *k == name) {
                    Some((_, value)) => out.push_str(value),
                    None => {
                        return Err(PyException::new("KeyError", py_repr(&name)));
                    }
                }
            }
            _ => return Err(crate::value_error("incomplete format")),
        }
    }
    Ok(out)
}

/// The formatter's `_format_text`: `%(prog)s` is substituted (and `%%`
/// collapsed) only when the text mentions `%(prog)` — a version or a
/// description without it prints verbatim, `100%` included.
fn format_text(text: &str, prog: &str) -> String {
    if !text.contains("%(prog)") {
        return text.to_string();
    }
    percent_format(text, &[("prog", prog.to_string())]).unwrap_or_else(|e| loud_exit(&e))
}

/// The formatter's `_expand_help`: a help string is ALWAYS formatted
/// (CPython raises for a stray `%`), with the action's attributes as
/// parameters — `prog`, `default`, `dest`, `type` here.
fn expand_help(help: &str, prog: &str, spec: &ArgSpec) -> String {
    let default = match (&spec.default, spec.kind) {
        (_, ArgKind::StoreTrue) => "False".to_string(),
        (Some(ParsedValue::Str(s)), _) => s.clone(),
        (Some(ParsedValue::Int(i)), _) => i.to_string(),
        (Some(ParsedValue::Float(f)), _) => crate::py_float_repr(*f),
        (Some(ParsedValue::Flag(b)), _) => if *b { "True" } else { "False" }.to_string(),
        (Some(other), _) => format!("{:?}", other),
        (None, _) => "None".to_string(),
    };
    let type_name = match spec.kind {
        ArgKind::Int => "int",
        ArgKind::Float => "float",
        ArgKind::Str | ArgKind::StoreTrue | ArgKind::Version => "None",
        ArgKind::File(_) | ArgKind::BinaryFile(_) => "FileType",
    };
    percent_format(
        help,
        &[
            ("prog", prog.to_string()),
            ("default", default),
            ("dest", spec.dest()),
            ("type", type_name.to_string()),
        ],
    )
    .unwrap_or_else(|e| loud_exit(&e))
}

/// A Python exception escaping parse_args itself (a bad `%` directive):
/// the traceback CPython would print, exit 1.
fn loud_exit(e: &PyException) -> ! {
    eprintln!("Traceback (most recent call last):");
    eprintln!("{}: {}", e.exception_type, e.message);
    std::process::exit(1);
}

/// `action="version"`: Python prints the version string — through the
/// formatter, so `%(prog)s` is the parser's name — to stdout and exits 0.
fn print_version_and_exit(prog: &str, spec: &ArgSpec) -> ! {
    let version = match &spec.default {
        Some(ParsedValue::Str(v)) => v.clone(),
        _ => String::new(),
    };
    println!("{}", format_text(&version, prog));
    std::process::exit(0);
}

fn exit_error(prog: &str, specs: &[ArgSpec], message: &str) -> ! {
    eprintln!("{}", usage_line(prog, specs));
    eprintln!("{}: error: {}", prog, message);
    std::process::exit(2);
}

impl ArgSpec {
    /// How an argument is named in error messages — CPython's
    /// `_get_action_name`: the option strings joined by `/` (`-n/--num`),
    /// a positional by its name.
    fn action_name(&self) -> String {
        match self.short {
            Some(short) if !self.is_positional() => format!("{}/{}", short, self.name),
            _ => self.name.to_string(),
        }
    }
}

/// Python's repr of a string with no quote inside — what a FileType
/// error prints for a mode or a value.
fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{}\"", s)
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

fn convert(
    prog: &str,
    specs: &[ArgSpec],
    spec: &ArgSpec,
    raw: &str,
) -> ParsedValue {
    match spec.kind {
        ArgKind::Str => ParsedValue::Str(raw.to_string()),
        ArgKind::Int => match raw.parse::<i64>() {
            Ok(i) => ParsedValue::Int(i),
            Err(_) => exit_error(
                prog,
                specs,
                &format!("argument {}: invalid int value: '{}'", spec.action_name(), raw),
            ),
        },
        ArgKind::Float => match raw.parse::<f64>() {
            Ok(f) => ParsedValue::Float(f),
            Err(_) => exit_error(
                prog,
                specs,
                &format!("argument {}: invalid float value: '{}'", spec.action_name(), raw),
            ),
        },
        ArgKind::StoreTrue | ArgKind::Version => ParsedValue::Flag(true),
        // Python opens the file at parse time and reports a failure as an
        // argument error: `argument files: can't open 'x': [Errno 2] No
        // such file or directory: 'x'`. The handle opened here IS the
        // namespace's (opened once; a write mode truncates once, as
        // Python's parse-time open does). `-` follows FileType.__call__
        // exactly: the standard input for a mode containing 'r', the
        // standard output for one containing 'w', 'a' or 'x', and for any
        // other mode the ValueError that `_get_value` reports as an
        // invalid value (Devin review on #339, round 2).
        ArgKind::File(mode) | ArgKind::BinaryFile(mode) => {
            let binary = matches!(spec.kind, ArgKind::BinaryFile(_));
            let handle = if raw == "-" {
                if mode.contains('r') {
                    Ok(if binary {
                        FileHandle::Binary(crate::stdlib::io::PyBytesIO::stdin())
                    } else {
                        FileHandle::Text(crate::PyFile::stdin())
                    })
                } else if mode.contains(['w', 'a', 'x']) {
                    Ok(if binary {
                        FileHandle::Binary(crate::stdlib::io::PyBytesIO::stdout())
                    } else {
                        FileHandle::Text(crate::PyFile::stdout())
                    })
                } else {
                    Err(crate::value_error(&format!(
                        "argument \"-\" with mode {}",
                        py_repr(mode)
                    )))
                }
            } else if binary {
                crate::open_binary(raw, mode).map(FileHandle::Binary)
            } else {
                crate::open(raw, Some(mode)).map(FileHandle::Text)
            };
            match handle {
                Ok(handle) => ParsedValue::File(OpenedFile {
                    path: raw.to_string(),
                    mode,
                    handle,
                }),
                // FileType.__call__ turns an OSError into "can't open
                // '<path>': <str(e)>"; a ValueError (an invalid mode, or
                // `-` with a mode that opens nothing) reaches _get_value
                // and is reported as an invalid value of the type's repr.
                Err(e) if e.exception_type == "ValueError" => exit_error(
                    prog,
                    specs,
                    &format!(
                        "argument {}: invalid FileType({}) value: {}",
                        spec.action_name(),
                        py_repr(mode),
                        py_repr(raw)
                    ),
                ),
                Err(e) => exit_error(
                    prog,
                    specs,
                    &format!(
                        "argument {}: can't open '{}': {}",
                        spec.action_name(),
                        raw,
                        e.message
                    ),
                ),
            }
        }
    }
}

/// What an option string names: the parser's own `-h`/`--help`, or one
/// of the specs.
#[derive(Clone, Copy, PartialEq)]
enum OptAction {
    Help,
    Spec(usize),
}

/// CPython's option tuple — `_parse_optional`'s answer for a token that
/// looks like an option: the action (None for an option string the
/// parser does not know), the option string it matched, the separator
/// (`=`, or the empty string for `-s2.5`'s attached value) and the
/// explicit argument.
struct OptionTuple {
    action: Option<OptAction>,
    option_string: String,
    sep: Option<String>,
    explicit_arg: Option<String>,
}

/// The parser's option strings in registration order — `-h`, `--help`,
/// then each option's -short and --long (CPython's
/// `_option_string_actions`, whose order the ambiguity message shows).
fn option_strings(specs: &[ArgSpec]) -> Vec<(&'static str, OptAction)> {
    let mut out = vec![("-h", OptAction::Help), ("--help", OptAction::Help)];
    for (i, s) in specs.iter().enumerate().filter(|(_, s)| !s.is_positional()) {
        if let Some(short) = s.short {
            out.push((short, OptAction::Spec(i)));
        }
        out.push((s.name, OptAction::Spec(i)));
    }
    out
}

/// CPython's `_negative_number_matcher`: `^-\d+$|^-\d*\.\d+$`.
fn looks_like_negative_number(s: &str) -> bool {
    let Some(rest) = s.strip_prefix('-') else {
        return false;
    };
    let digits = |t: &str| t.chars().all(|c| c.is_ascii_digit());
    match rest.split_once('.') {
        None => !rest.is_empty() && digits(rest),
        Some((int, frac)) => digits(int) && !frac.is_empty() && digits(frac),
    }
}

/// One element of a nargs pattern over the token classes: a greedy run
/// of any of the given classes, or exactly one.
enum Atom {
    Many(&'static [char]),
    One(char),
}

/// A positional's nargs pattern (`_get_nargs_pattern`, with the `-*`
/// kept — positionals absorb the `--` marker): `(-*A-*)` for one value,
/// `(-*[A-]*)` for `*`, `(-*A[A-]*)` for `+`.
fn positional_pattern(nargs: Nargs) -> Vec<Atom> {
    match nargs {
        Nargs::One => vec![Atom::Many(&['-']), Atom::One('A'), Atom::Many(&['-'])],
        Nargs::Star => vec![Atom::Many(&['-']), Atom::Many(&['A', '-'])],
        Nargs::Plus => vec![Atom::Many(&['-']), Atom::One('A'), Atom::Many(&['A', '-'])],
    }
}

/// `re.match` of the concatenated groups against the token-class string
/// (anchored at the start only), in the regex engine's greedy-then-
/// backtrack order; the length each group matched.
fn match_groups(groups: &[Vec<Atom>], pattern: &[char]) -> Option<Vec<usize>> {
    fn go(
        groups: &[Vec<Atom>],
        gi: usize,
        ai: usize,
        pos: usize,
        pattern: &[char],
        ends: &mut Vec<usize>,
    ) -> bool {
        if gi == groups.len() {
            return true;
        }
        let atoms = &groups[gi];
        if ai == atoms.len() {
            ends.push(pos);
            if go(groups, gi + 1, 0, pos, pattern, ends) {
                return true;
            }
            ends.pop();
            return false;
        }
        match &atoms[ai] {
            Atom::One(c) => {
                pattern.get(pos) == Some(c) && go(groups, gi, ai + 1, pos + 1, pattern, ends)
            }
            Atom::Many(set) => {
                let run = pattern[pos..].iter().take_while(|c| set.contains(c)).count();
                (0..=run)
                    .rev()
                    .any(|n| go(groups, gi, ai + 1, pos + n, pattern, ends))
            }
        }
    }
    let mut ends = Vec::new();
    if !go(groups, 0, 0, 0, pattern, &mut ends) {
        return None;
    }
    let mut lengths = Vec::with_capacity(ends.len());
    let mut previous = 0;
    for end in ends {
        lengths.push(end - previous);
        previous = end;
    }
    Some(lengths)
}

/// `_match_arguments_partial`: the longest prefix of the remaining
/// positionals whose combined pattern matches, and each one's count.
fn match_arguments_partial(specs: &[ArgSpec], positionals: &[usize], pattern: &[char]) -> Vec<usize> {
    for take in (1..=positionals.len()).rev() {
        let groups: Vec<Vec<Atom>> = positionals[..take]
            .iter()
            .map(|&i| positional_pattern(specs[i].nargs))
            .collect();
        if let Some(lengths) = match_groups(&groups, pattern) {
            return lengths;
        }
    }
    Vec::new()
}

/// The state of one `parse_args` run: CPython's `_parse_known_args`
/// locals, with `take_action`, `consume_optional` and
/// `consume_positionals` as methods.
struct Parse<'a> {
    prog: String,
    description: Option<&'a str>,
    specs: &'a [ArgSpec],
    arg_strings: Vec<String>,
    /// Every option string in registration order (`_option_string_actions`).
    options: Vec<(&'static str, OptAction)>,
    /// `_has_negative_number_optionals`: an option string that looks like
    /// a negative number (`-1`) makes every unmatched negative-number
    /// token an option (an unrecognized one), never a positional.
    has_negative_number_optionals: bool,
    /// Each token's class: `A` an argument, `O` an option, `-` the
    /// `--` marker.
    pattern: Vec<char>,
    /// The option tuple at each `O` index.
    option_at: Vec<Option<OptionTuple>>,
    values: Vec<Option<ParsedValue>>,
    seen: Vec<bool>,
    extras: Vec<String>,
    /// The positionals still to be consumed (spec indices).
    positionals: Vec<usize>,
}

impl<'a> Parse<'a> {
    fn lookup(&self, s: &str) -> Option<OptAction> {
        self.options.iter().find(|(name, _)| *name == s).map(|(_, a)| *a)
    }

    fn error(&self, message: &str) -> ! {
        exit_error(&self.prog, self.specs, message)
    }

    fn action_name(&self, action: OptAction) -> String {
        match action {
            OptAction::Help => "-h/--help".to_string(),
            OptAction::Spec(i) => self.specs[i].action_name(),
        }
    }

    /// `_get_option_tuples`: the interpretations of an option-looking
    /// token by prefix — a `--long` abbreviation, or a `-s` with its
    /// value attached (`-s2.5`).
    fn option_tuples(&self, token: &str) -> Vec<OptionTuple> {
        let mut result = Vec::new();
        if token.chars().nth(1) == Some('-') {
            let (prefix, sep, explicit) = match token.split_once('=') {
                Some((p, e)) => (p, Some("=".to_string()), Some(e.to_string())),
                None => (token, None, None),
            };
            for (name, action) in &self.options {
                if name.starts_with(prefix) {
                    result.push(OptionTuple {
                        action: Some(*action),
                        option_string: name.to_string(),
                        sep: sep.clone(),
                        explicit_arg: explicit.clone(),
                    });
                }
            }
        } else {
            let short_prefix: String = token.chars().take(2).collect();
            let short_explicit: String = token.chars().skip(2).collect();
            for (name, action) in &self.options {
                if *name == short_prefix {
                    result.push(OptionTuple {
                        action: Some(*action),
                        option_string: name.to_string(),
                        sep: Some(String::new()),
                        explicit_arg: Some(short_explicit.clone()),
                    });
                } else if name.starts_with(token) {
                    result.push(OptionTuple {
                        action: Some(*action),
                        option_string: name.to_string(),
                        sep: None,
                        explicit_arg: None,
                    });
                }
            }
        }
        result
    }

    /// `_parse_optional`: None for a token meant as a positional.
    fn parse_optional(&self, token: &str) -> Option<OptionTuple> {
        if !token.starts_with('-') {
            return None;
        }
        if let Some(action) = self.lookup(token) {
            return Some(OptionTuple {
                action: Some(action),
                option_string: token.to_string(),
                sep: None,
                explicit_arg: None,
            });
        }
        if token.chars().count() == 1 {
            return None;
        }
        if let Some((name, explicit)) = token.split_once('=')
            && let Some(action) = self.lookup(name)
        {
            return Some(OptionTuple {
                action: Some(action),
                option_string: name.to_string(),
                sep: Some("=".to_string()),
                explicit_arg: Some(explicit.to_string()),
            });
        }
        let mut tuples = self.option_tuples(token);
        if tuples.len() > 1 {
            let matches: Vec<&str> = tuples.iter().map(|t| t.option_string.as_str()).collect();
            self.error(&format!(
                "ambiguous option: {} could match {}",
                token,
                matches.join(", ")
            ));
        }
        if tuples.len() == 1 {
            return tuples.pop();
        }
        // A negative number is a positional unless an option looks like
        // one; a token with a space in it is a positional.
        if looks_like_negative_number(token) && !self.has_negative_number_optionals {
            return None;
        }
        if token.contains(' ') {
            return None;
        }
        Some(OptionTuple {
            action: None,
            option_string: token.to_string(),
            sep: None,
            explicit_arg: None,
        })
    }

    /// `_match_argument` for an option: the count of following tokens it
    /// takes — none for a flag, one argument (`A`) for a value-taking
    /// option, else CPython's error.
    fn match_argument(&self, action: OptAction, following: &[char]) -> usize {
        match action {
            OptAction::Help => 0,
            OptAction::Spec(i) if !self.specs[i].takes_value() => 0,
            OptAction::Spec(i) => {
                if following.first() == Some(&'A') {
                    1
                } else {
                    self.error(&format!(
                        "argument {}: expected one argument",
                        self.specs[i].action_name()
                    ))
                }
            }
        }
    }

    /// `take_action`: convert the values (opening files, reporting the
    /// first failure) and store them — or, for help and version, print
    /// and exit right here, in argv order.
    fn take_action(&mut self, action: OptAction, mut args: Vec<String>) {
        match action {
            OptAction::Help => {
                print!("{}", help_text(&self.prog, self.description, self.specs));
                std::process::exit(0);
            }
            OptAction::Spec(i) => {
                let spec = &self.specs[i];
                self.seen[i] = true;
                // `_get_values` drops one `--` marker from the values.
                if let Some(at) = args.iter().position(|a| a == "--") {
                    args.remove(at);
                }
                let value = match spec.kind {
                    ArgKind::Version => print_version_and_exit(&self.prog, spec),
                    ArgKind::StoreTrue => ParsedValue::Flag(true),
                    _ if spec.nargs == Nargs::One => {
                        let raw = args.first().expect("one value matched");
                        convert(&self.prog, self.specs, spec, raw)
                    }
                    _ => ParsedValue::List(
                        args.iter()
                            .map(|a| convert(&self.prog, self.specs, spec, a))
                            .collect(),
                    ),
                };
                self.values[i] = Some(value);
            }
        }
    }

    /// `consume_optional`: the option at this index, its explicit or
    /// following argument, and — for a single-dash flag with more
    /// letters attached (`-vn 3`) — the flags packed after it.
    fn consume_optional(&mut self, start_index: usize) -> usize {
        let tuple = self.option_at[start_index].as_ref().expect("an option index");
        let mut action = tuple.action;
        let mut option_string = tuple.option_string.clone();
        let mut sep = tuple.sep.clone();
        let mut explicit_arg = tuple.explicit_arg.clone();
        let mut action_tuples: Vec<(OptAction, Vec<String>)> = Vec::new();
        let stop;
        loop {
            let Some(current) = action else {
                self.extras.push(self.arg_strings[start_index].clone());
                return start_index + 1;
            };
            match explicit_arg.take() {
                Some(explicit) => {
                    let arg_count = self.match_argument(current, &['A']);
                    let single_dash = option_string.chars().nth(1) != Some('-');
                    if arg_count == 0 && single_dash && !explicit.is_empty() {
                        if sep.as_deref().is_some_and(|s| !s.is_empty())
                            || explicit.starts_with('-')
                        {
                            self.error(&format!(
                                "argument {}: ignored explicit argument {}",
                                self.action_name(current),
                                py_repr(&explicit)
                            ));
                        }
                        action_tuples.push((current, Vec::new()));
                        let mut chars = explicit.chars();
                        let next_letter = chars.next().expect("non-empty");
                        let rest: String = chars.collect();
                        option_string = format!("-{}", next_letter);
                        match self.lookup(&option_string) {
                            Some(next) => {
                                action = Some(next);
                                if rest.is_empty() {
                                    sep = None;
                                    explicit_arg = None;
                                } else if let Some(after_eq) = rest.strip_prefix('=') {
                                    sep = Some("=".to_string());
                                    explicit_arg = Some(after_eq.to_string());
                                } else {
                                    sep = Some(String::new());
                                    explicit_arg = Some(rest);
                                }
                            }
                            None => {
                                self.extras.push(format!("-{}", explicit));
                                stop = start_index + 1;
                                break;
                            }
                        }
                    } else if arg_count == 1 {
                        stop = start_index + 1;
                        action_tuples.push((current, vec![explicit]));
                        break;
                    } else {
                        self.error(&format!(
                            "argument {}: ignored explicit argument {}",
                            self.action_name(current),
                            py_repr(&explicit)
                        ));
                    }
                }
                None => {
                    let start = start_index + 1;
                    let arg_count = self.match_argument(current, &self.pattern[start..]);
                    stop = start + arg_count;
                    action_tuples.push((current, self.arg_strings[start..stop].to_vec()));
                    break;
                }
            }
        }
        for (action, args) in action_tuples {
            self.take_action(action, args);
        }
        stop
    }

    /// `consume_positionals`: as many of the remaining positionals as the
    /// token classes from here allow, each taking its matched slice.
    fn consume_positionals(&mut self, mut start_index: usize) -> usize {
        let counts =
            match_arguments_partial(self.specs, &self.positionals, &self.pattern[start_index..]);
        let taken: Vec<usize> = self.positionals.drain(..counts.len()).collect();
        for (spec_index, count) in taken.into_iter().zip(counts) {
            let args = self.arg_strings[start_index..start_index + count].to_vec();
            start_index += count;
            self.take_action(OptAction::Spec(spec_index), args);
        }
        start_index
    }
}

/// Parse std::env::args() against the specs, exactly as Python's
/// parse_args(): returns the value for every spec IN SPEC ORDER, or
/// prints help (exit 0) / usage + error (exit 2) like CPython. The
/// PyException in the signature keeps the call-site shape uniform;
/// errors exit instead, as Python's SystemExit reaching the top does.
///
/// This is a port of `ArgumentParser._parse_known_args` (CPython 3.11):
/// every token is classified as an argument (`A`), an option (`O`) or
/// the `--` marker; positionals are consumed at each option boundary
/// through the partial pattern match, options through
/// `consume_optional`, and every action is taken — converted, opened,
/// printed — in ARGV ORDER, so the error a bad positional raises comes
/// before a later option's, `--version` after a positional that cannot
/// open never prints, and the arguments a variadic positional takes are
/// exactly CPython's (Devin review on #339, round 2).
pub fn run_parser(
    prog: Option<&str>,
    description: Option<&str>,
    specs: &[ArgSpec],
    argv: Option<Vec<String>>,
) -> Result<Vec<ParsedValue>, PyException> {
    // parse_args(argv): an explicit argument list; None is sys.argv[1:].
    let arg_strings: Vec<String> = argv.unwrap_or_else(|| std::env::args().skip(1).collect());
    let mut parse = Parse {
        prog: prog_name(prog),
        description,
        specs,
        arg_strings,
        options: option_strings(specs),
        has_negative_number_optionals: option_strings(specs)
            .iter()
            .any(|(name, _)| looks_like_negative_number(name)),
        pattern: Vec::new(),
        option_at: Vec::new(),
        values: specs.iter().map(|_| None).collect(),
        seen: specs.iter().map(|_| false).collect(),
        extras: Vec::new(),
        positionals: specs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_positional())
            .map(|(i, _)| i)
            .collect(),
    };

    // Classify every token; everything after `--` is an argument.
    let mut after_marker = false;
    for token in parse.arg_strings.clone() {
        let (class, tuple) = if after_marker {
            ('A', None)
        } else if token == "--" {
            after_marker = true;
            ('-', None)
        } else {
            match parse.parse_optional(&token) {
                None => ('A', None),
                Some(tuple) => ('O', Some(tuple)),
            }
        };
        parse.pattern.push(class);
        parse.option_at.push(tuple);
    }

    // Consume positionals and options alternately up to the last option.
    let option_indices: Vec<usize> = parse
        .pattern
        .iter()
        .enumerate()
        .filter(|(_, c)| **c == 'O')
        .map(|(i, _)| i)
        .collect();
    let mut start_index = 0;
    let max_option_index = option_indices.last().copied();
    while max_option_index.is_some_and(|max| start_index <= max) {
        let next_option_index = option_indices
            .iter()
            .copied()
            .find(|&i| i >= start_index)
            .expect("an option index at or after start");
        if start_index != next_option_index {
            let positionals_end = parse.consume_positionals(start_index);
            if positionals_end > start_index {
                start_index = positionals_end;
                continue;
            }
            start_index = positionals_end;
        }
        if !option_indices.contains(&start_index) {
            let skipped = parse.arg_strings[start_index..next_option_index].to_vec();
            parse.extras.extend(skipped);
            start_index = next_option_index;
        }
        start_index = parse.consume_optional(start_index);
    }
    let stop_index = parse.consume_positionals(start_index);
    let leftovers = parse.arg_strings[stop_index..].to_vec();
    parse.extras.extend(leftovers);

    // Required arguments first (`_parse_known_args`), then the leftovers
    // (`parse_args`): a positional that takes zero or more is never
    // required.
    let missing: Vec<String> = specs
        .iter()
        .enumerate()
        .filter(|(i, s)| !parse.seen[*i] && s.is_positional() && s.nargs != Nargs::Star)
        .map(|(_, s)| s.action_name())
        .collect();
    if !missing.is_empty() {
        parse.error(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    if !parse.extras.is_empty() {
        parse.error(&format!("unrecognized arguments: {}", parse.extras.join(" ")));
    }

    Ok(parse
        .values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.or_else(|| specs[i].default.clone()).unwrap_or_else(|| {
                match specs[i].kind {
                    ArgKind::StoreTrue | ArgKind::Version => ParsedValue::Flag(false),
                    _ if specs[i].nargs == Nargs::Star => ParsedValue::List(Vec::new()),
                    // The converter requires default= on value-taking
                    // options, so this is unreachable for valid specs.
                    _ => panic!(
                        "argparse internal error: option {} has no value and no default",
                        specs[i].name
                    ),
                }
            })
        })
        .collect())
}
