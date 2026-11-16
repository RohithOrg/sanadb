//! Typed metadata values.
//!
//! A record's metadata is a map from string keys to `Value`s. The first value
//! written under a key fixes that key's type for the store; a later write of
//! another type is `Error::TypeMismatch`. Keys are dynamic, there is no
//! schema to declare.

use std::collections::HashMap;
use std::fmt;

/// Metadata attached to a record.
pub type Metadata = HashMap<String, Value>;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// The type of a `Value`, without its payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    Str = 0,
    Int = 1,
    Float = 2,
    Bool = 3,
}

impl Kind {
    pub(crate) fn from_u8(b: u8) -> Option<Kind> {
        match b {
            0 => Some(Kind::Str),
            1 => Some(Kind::Int),
            2 => Some(Kind::Float),
            3 => Some(Kind::Bool),
            _ => None,
        }
    }
}

impl Value {
    pub fn kind(&self) -> Kind {
        match self {
            Value::Str(_) => Kind::Str,
            Value::Int(_) => Kind::Int,
            Value::Float(_) => Kind::Float,
            Value::Bool(_) => Kind::Bool,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Floats and ints both read as a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(s) => f.write_str(s),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::Bool(b) => write!(f, "{b}"),
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Value {
        Value::Int(i)
    }
}
impl From<i32> for Value {
    fn from(i: i32) -> Value {
        Value::Int(i as i64)
    }
}
impl From<u32> for Value {
    fn from(i: u32) -> Value {
        Value::Int(i as i64)
    }
}
impl From<f64> for Value {
    fn from(x: f64) -> Value {
        Value::Float(x)
    }
}
impl From<f32> for Value {
    fn from(x: f32) -> Value {
        Value::Float(x as f64)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}

impl PartialEq<str> for Value {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == Some(other)
    }
}
impl PartialEq<&str> for Value {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}
impl PartialEq<String> for Value {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == Some(other.as_str())
    }
}
