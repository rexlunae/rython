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

#[derive(Clone, Debug, PartialEq)]
pub enum ParsedValue {
    Str(String),
    Int(i64),
    Float(f64),
    Flag(bool),
    /// A FileType value: the PATH the argument named and the mode; the
    /// parser validated it by opening it (Python opens at parse time and
    /// reports a failure as an argument error), and `into_file` opens it
    /// for the namespace.
    File(String, &'static str),
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
    /// The opened text file of a FileType argument. The path was
    /// validated at parse time; a failure here (the file vanished in
    /// between) is the module-level abort every failed initializer is.
    pub fn into_file(self) -> crate::PyFile {
        match self {
            ParsedValue::File(path, mode) if path == "-" => {
                // Python: "-" is sys.stdin for a read mode.
                let mut text = String::new();
                use std::io::Read;
                std::io::stdin()
                    .read_to_string(&mut text)
                    .unwrap_or_else(|e| panic!("argparse: reading stdin for '-': {}", e));
                let mut file = crate::stdlib::io::StringIO_seeded(&text);
                file.name = "<stdin>".to_string();
                let _ = mode;
                file
            }
            ParsedValue::File(path, mode) => crate::open(&path, Some(mode))
                .unwrap_or_else(|e| panic!("argparse: can't open '{}': {}", path, e)),
            other => panic!("argparse internal error: expected file, got {:?}", other),
        }
    }
    /// The opened binary file of a FileType argument (a 'b' mode).
    pub fn into_binary_file(self) -> crate::stdlib::io::PyBytesIO {
        match self {
            ParsedValue::File(path, _) if path == "-" => {
                let mut bytes = Vec::new();
                use std::io::Read;
                std::io::stdin()
                    .read_to_end(&mut bytes)
                    .unwrap_or_else(|e| panic!("argparse: reading stdin for '-': {}", e));
                let mut file = crate::stdlib::io::BytesIO_seeded(bytes);
                file.name = "<stdin>".to_string();
                file
            }
            ParsedValue::File(path, mode) => crate::open_binary(&path, mode)
                .unwrap_or_else(|e| panic!("argparse: can't open '{}': {}", path, e)),
            other => panic!("argparse internal error: expected file, got {:?}", other),
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
        out.push_str(d);
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
            entry(&mut out, &s.invocation(), s.help);
        }
    }
    out.push_str("\noptions:\n");
    entry(&mut out, &help_spec, Some("show this help message and exit"));
    for s in specs.iter().filter(|s| !s.is_positional()) {
        // Python's default help for action="version".
        let help = match (s.kind, s.help) {
            (ArgKind::Version, None) => Some("show program's version number and exit"),
            (_, h) => h,
        };
        entry(&mut out, &s.invocation(), help);
    }
    out
}

/// `action="version"`: Python prints the version string to stdout and
/// exits 0.
fn print_version_and_exit(spec: &ArgSpec) -> ! {
    let version = match &spec.default {
        Some(ParsedValue::Str(v)) => v.clone(),
        _ => String::new(),
    };
    println!("{}", version);
    std::process::exit(0);
}

fn exit_error(prog: &str, specs: &[ArgSpec], message: &str) -> ! {
    eprintln!("{}", usage_line(prog, specs));
    eprintln!("{}: error: {}", prog, message);
    std::process::exit(2);
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
                &format!("argument {}: invalid int value: '{}'", spec.name, raw),
            ),
        },
        ArgKind::Float => match raw.parse::<f64>() {
            Ok(f) => ParsedValue::Float(f),
            Err(_) => exit_error(
                prog,
                specs,
                &format!("argument {}: invalid float value: '{}'", spec.name, raw),
            ),
        },
        ArgKind::StoreTrue | ArgKind::Version => ParsedValue::Flag(true),
        // Python opens the file at parse time and reports a failure as an
        // argument error: `argument files: can't open 'x': [Errno 2] No
        // such file or directory: 'x'`. The handle is dropped here and
        // reopened for the namespace (into_file), so a write mode
        // truncates exactly as Python's parse-time open does.
        ArgKind::File(mode) | ArgKind::BinaryFile(mode) => {
            if raw != "-" {
                let opened = if matches!(spec.kind, ArgKind::BinaryFile(_)) {
                    crate::open_binary(raw, mode).map(|_| ())
                } else {
                    crate::open(raw, Some(mode)).map(|_| ())
                };
                if let Err(e) = opened {
                    // Python's message: str(e) — the exception's message
                    // without its type.
                    exit_error(
                        prog,
                        specs,
                        &format!("argument {}: can't open '{}': {}", spec.name, raw, e.message),
                    );
                }
            }
            ParsedValue::File(raw.to_string(), mode)
        }
    }
}

/// Parse std::env::args() against the specs, exactly as Python's
/// parse_args(): returns the value for every spec IN SPEC ORDER, or
/// prints help (exit 0) / usage + error (exit 2) like CPython. The
/// PyException in the signature keeps the call-site shape uniform;
/// errors exit instead, as Python's SystemExit reaching the top does.
pub fn run_parser(
    prog: Option<&str>,
    description: Option<&str>,
    specs: &[ArgSpec],
    argv: Option<Vec<String>>,
) -> Result<Vec<ParsedValue>, PyException> {
    let prog = prog_name(prog);
    // parse_args(argv): an explicit argument list; None is sys.argv[1:].
    let argv: Vec<String> = argv.unwrap_or_else(|| std::env::args().skip(1).collect());

    let mut values: Vec<Option<ParsedValue>> = specs.iter().map(|_| None).collect();
    let mut extras: Vec<String> = Vec::new();
    let positional_indices: Vec<usize> = specs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_positional())
        .map(|(i, _)| i)
        .collect();
    let mut positional_tokens: Vec<String> = Vec::new();

    let mut i = 0;
    while i < argv.len() {
        let token = &argv[i];
        if token == "-h" || token == "--help" {
            print!("{}", help_text(&prog, description, specs));
            std::process::exit(0);
        }
        if token.starts_with("--") {
            // --opt=value splits; prefix abbreviation resolves like
            // Python (unique prefix ok, ambiguous is an error).
            let (name, inline) = match token.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (token.clone(), None),
            };
            let matches: Vec<usize> = specs
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.is_positional() && s.name.starts_with(name.as_str()))
                .map(|(i, _)| i)
                .collect();
            let exact: Option<usize> = specs
                .iter()
                .position(|s| !s.is_positional() && s.name == name);
            let idx = match (exact, matches.as_slice()) {
                (Some(i), _) => i,
                (None, [single]) => *single,
                (None, []) => {
                    extras.push(token.clone());
                    i += 1;
                    continue;
                }
                (None, many) => {
                    let options: Vec<&str> =
                        many.iter().map(|&i| specs[i].name).collect();
                    exit_error(
                        &prog,
                        specs,
                        &format!(
                            "ambiguous option: {} could match {}",
                            name,
                            options.join(", ")
                        ),
                    );
                }
            };
            let spec = &specs[idx];
            if spec.kind == ArgKind::Version {
                print_version_and_exit(spec);
            }
            let value = if spec.kind == ArgKind::StoreTrue {
                if inline.is_some() {
                    exit_error(
                        &prog,
                        specs,
                        &format!("argument {}: ignored explicit argument", spec.name),
                    );
                }
                ParsedValue::Flag(true)
            } else {
                let raw = match inline {
                    Some(v) => v,
                    None => {
                        i += 1;
                        match argv.get(i) {
                            Some(v) => v.clone(),
                            None => exit_error(
                                &prog,
                                specs,
                                &format!("argument {}: expected one argument", spec.name),
                            ),
                        }
                    }
                };
                convert(&prog, specs, spec, &raw)
            };
            values[idx] = Some(value);
        } else if token.starts_with('-')
            && token.len() > 1
            && token.parse::<f64>().is_err()
        {
            // A SHORT option (-c, -s 2.5, -s2.5). Like Python, a token
            // that looks like an option (leading '-', not a negative
            // number) never fills a positional — an unknown one is an
            // "unrecognized arguments" error.
            let exact = specs
                .iter()
                .position(|s| s.short == Some(token.as_str()));
            if let Some(idx) = exact {
                let spec = &specs[idx];
                if spec.kind == ArgKind::Version {
                    print_version_and_exit(spec);
                }
                let value = if spec.kind == ArgKind::StoreTrue {
                    ParsedValue::Flag(true)
                } else {
                    i += 1;
                    match argv.get(i) {
                        Some(v) => convert(&prog, specs, spec, v),
                        None => exit_error(
                            &prog,
                            specs,
                            &format!("argument {}: expected one argument", spec.name),
                        ),
                    }
                };
                values[idx] = Some(value);
            } else {
                // Attached-value form (-s2.5) for value-taking shorts.
                let head: String = token.chars().take(2).collect();
                let attached = specs.iter().position(|s| {
                    s.short.as_deref() == Some(head.as_str()) && s.takes_value()
                });
                match attached {
                    Some(idx) => {
                        let raw: String = token.chars().skip(2).collect();
                        values[idx] = Some(convert(&prog, specs, &specs[idx], &raw));
                    }
                    None => extras.push(token.clone()),
                }
            }
        } else {
            // Positional tokens are distributed after the loop (a
            // variadic positional takes what the fixed ones leave).
            positional_tokens.push(token.clone());
        }
        i += 1;
    }

    // Distribute the positional tokens: fixed positionals before the
    // variadic one take one each from the front, fixed ones after it one
    // each from the back, the variadic takes the middle (Python's
    // pattern match over the positional sequence, for one variadic);
    // leftovers with no variadic are "unrecognized arguments".
    let variadic = positional_indices
        .iter()
        .position(|&i| specs[i].nargs != Nargs::One);
    let fixed = positional_indices.len() - usize::from(variadic.is_some());
    let mut tokens = positional_tokens.into_iter();
    let before = variadic.unwrap_or(positional_indices.len());
    let mut assigned = 0usize;
    for &idx in &positional_indices[..before] {
        match tokens.next() {
            Some(t) => {
                values[idx] = Some(convert(&prog, specs, &specs[idx], &t));
                assigned += 1;
            }
            None => break,
        }
    }
    let mut rest: Vec<String> = tokens.collect();
    if let Some(v) = variadic {
        let after = &positional_indices[v + 1..];
        let keep_for_after = after.len().min(rest.len());
        let tail: Vec<String> = rest.split_off(rest.len() - keep_for_after);
        let vidx = positional_indices[v];
        if assigned == before {
            let items: Vec<ParsedValue> = rest
                .iter()
                .map(|t| convert(&prog, specs, &specs[vidx], t))
                .collect();
            if !(items.is_empty() && specs[vidx].nargs == Nargs::Plus) {
                values[vidx] = Some(ParsedValue::List(items));
            }
        }
        for (&idx, t) in after.iter().zip(tail.iter()) {
            values[idx] = Some(convert(&prog, specs, &specs[idx], t));
        }
        rest = Vec::new();
    }
    let _ = fixed;
    extras.extend(rest);

    if !extras.is_empty() {
        exit_error(
            &prog,
            specs,
            &format!("unrecognized arguments: {}", extras.join(" ")),
        );
    }
    let missing: Vec<&str> = positional_indices
        .iter()
        .filter(|&&i| values[i].is_none() && specs[i].nargs != Nargs::Star)
        .map(|&i| specs[i].name)
        .collect();
    if !missing.is_empty() {
        exit_error(
            &prog,
            specs,
            &format!(
                "the following arguments are required: {}",
                missing.join(", ")
            ),
        );
    }

    Ok(values
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
