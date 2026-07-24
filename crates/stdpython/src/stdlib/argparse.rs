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

#[derive(Clone, Copy, PartialEq)]
pub enum ArgKind {
    Str,
    Int,
    Float,
    StoreTrue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParsedValue {
    Str(String),
    Int(i64),
    Float(f64),
    Flag(bool),
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
}

pub struct ArgSpec {
    /// "count" for a positional, "--verbose" for an option.
    pub name: &'static str,
    pub kind: ArgKind,
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
    /// The attribute name on the namespace ("--scale" -> scale).
    fn dest(&self) -> String {
        self.name.trim_start_matches('-').replace('-', "_")
    }
    /// How the argument appears in the help list: positionals by name,
    /// value-taking options with their uppercase metavar.
    fn invocation(&self) -> String {
        if self.is_positional() {
            self.name.to_string()
        } else if self.kind == ArgKind::StoreTrue {
            self.name.to_string()
        } else {
            format!("{} {}", self.name, self.dest().to_uppercase())
        }
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
        parts.push(format!("[{}]", s.invocation()));
    }
    for s in specs.iter().filter(|s| s.is_positional()) {
        parts.push(s.name.to_string());
    }
    parts.join(" ")
}

fn help_text(prog: &str, description: Option<&str>, specs: &[ArgSpec]) -> String {
    // Python's help column: two-space indent + the longest invocation
    // (capped at 24) + two spaces. Longer invocations push their help
    // onto the next line at that column.
    let help_spec = "-h, --help".to_string();
    let max_len = specs
        .iter()
        .map(|s| s.invocation().chars().count())
        .chain([help_spec.chars().count()])
        .max()
        .unwrap_or(0)
        .min(24);
    let help_col = 2 + max_len + 2;

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
        entry(&mut out, &s.invocation(), s.help);
    }
    out
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
                &format!("argument {}: invalid int value: '{}'", spec.dest(), raw),
            ),
        },
        ArgKind::Float => match raw.parse::<f64>() {
            Ok(f) => ParsedValue::Float(f),
            Err(_) => exit_error(
                prog,
                specs,
                &format!("argument {}: invalid float value: '{}'", spec.dest(), raw),
            ),
        },
        ArgKind::StoreTrue => ParsedValue::Flag(true),
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
) -> Result<Vec<ParsedValue>, PyException> {
    let prog = prog_name(prog);
    let argv: Vec<String> = std::env::args().skip(1).collect();

    let mut values: Vec<Option<ParsedValue>> = specs.iter().map(|_| None).collect();
    let mut extras: Vec<String> = Vec::new();
    let positional_indices: Vec<usize> = specs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_positional())
        .map(|(i, _)| i)
        .collect();
    let mut next_positional = 0usize;

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
        } else if next_positional < positional_indices.len() {
            let idx = positional_indices[next_positional];
            values[idx] = Some(convert(&prog, specs, &specs[idx], token));
            next_positional += 1;
        } else {
            extras.push(token.clone());
        }
        i += 1;
    }

    if !extras.is_empty() {
        exit_error(
            &prog,
            specs,
            &format!("unrecognized arguments: {}", extras.join(" ")),
        );
    }
    let missing: Vec<&str> = positional_indices[next_positional..]
        .iter()
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
                    ArgKind::StoreTrue => ParsedValue::Flag(false),
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
