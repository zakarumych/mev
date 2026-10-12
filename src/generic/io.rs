use std::{error::Error, fmt};

/// Failure to create, encode, or execute native file I/O.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IoError {
    Unsupported,
    InvalidInput(String),
    Failure(String),
    Cancelled,
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("native file I/O is unsupported"),
            Self::InvalidInput(message) => write!(f, "invalid I/O input: {message}"),
            Self::Failure(message) => write!(f, "I/O failed: {message}"),
            Self::Cancelled => f.write_str("I/O cancelled"),
        }
    }
}

impl Error for IoError {}

