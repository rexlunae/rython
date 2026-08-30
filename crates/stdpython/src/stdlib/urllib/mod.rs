//! Python urllib package. Only the `request` submodule has runtime items;
//! `urllib.error`'s URLError/HTTPError are string-tagged PyException
//! values matched by name through the exception hierarchy (URLError IS-A
//! OSError, HTTPError IS-A URLError), so no runtime module is needed for
//! them.

pub mod parse;
/// urllib.request is ureq-backed (http-ureq feature); parse has no
/// network dependency.
#[cfg(feature = "http-ureq")]
pub mod request;
