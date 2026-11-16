//! Filter expressions over metadata.
//!
//! A `Filter` is data, not a closure, so the engine can evaluate it against
//! the index's attribute columns without touching the metadata map, estimate
//! how many records pass, and pick a search strategy from that estimate.

use crate::{Metadata, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Key holds exactly this value. A missing key or a different type never
    /// matches.
    Eq(String, Value),
    /// Key is present and holds a different value of the same type.
    Ne(String, Value),
    /// Key holds one of these values.
    In(String, Vec<Value>),
    Lt(String, Value),
    Le(String, Value),
    Gt(String, Value),
    Ge(String, Value),
    /// String key starts with this prefix.
    Prefix(String, String),
    /// Key is present with any value.
    Exists(String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    pub fn eq(key: impl Into<String>, v: impl Into<Value>) -> Filter {
        Filter::Eq(key.into(), v.into())
    }
    pub fn ne(key: impl Into<String>, v: impl Into<Value>) -> Filter {
        Filter::Ne(key.into(), v.into())
    }
    pub fn is_in(key: impl Into<String>, vs: impl IntoIterator<Item = impl Into<Value>>) -> Filter {
        Filter::In(key.into(), vs.into_iter().map(Into::into).collect())
    }
    pub fn lt(key: impl Into<String>, v: impl Into<Value>) -> Filter {
        Filter::Lt(key.into(), v.into())
    }
    pub fn le(key: impl Into<String>, v: impl Into<Value>) -> Filter {
        Filter::Le(key.into(), v.into())
    }
    pub fn gt(key: impl Into<String>, v: impl Into<Value>) -> Filter {
        Filter::Gt(key.into(), v.into())
    }
    pub fn ge(key: impl Into<String>, v: impl Into<Value>) -> Filter {
        Filter::Ge(key.into(), v.into())
    }
    pub fn prefix(key: impl Into<String>, p: impl Into<String>) -> Filter {
        Filter::Prefix(key.into(), p.into())
    }
    pub fn exists(key: impl Into<String>) -> Filter {
        Filter::Exists(key.into())
    }
    pub fn and(fs: impl IntoIterator<Item = Filter>) -> Filter {
        Filter::And(fs.into_iter().collect())
    }
    pub fn or(fs: impl IntoIterator<Item = Filter>) -> Filter {
        Filter::Or(fs.into_iter().collect())
    }
    #[allow(clippy::should_implement_trait)]
    pub fn not(f: Filter) -> Filter {
        Filter::Not(Box::new(f))
    }

    /// Evaluate against an in-memory record. Comparisons across types are
    /// false, except that ints and floats compare numerically.
    pub fn eval(&self, m: &Metadata) -> bool {
        match self {
            Filter::Eq(k, v) => m.get(k).is_some_and(|x| same(x, v)),
            Filter::Ne(k, v) => m
                .get(k)
                .is_some_and(|x| x.kind() == v.kind() && !same(x, v)),
            Filter::In(k, vs) => m.get(k).is_some_and(|x| vs.iter().any(|v| same(x, v))),
            Filter::Lt(k, v) => m.get(k).and_then(|x| cmp(x, v)).is_some_and(|o| o.is_lt()),
            Filter::Le(k, v) => m.get(k).and_then(|x| cmp(x, v)).is_some_and(|o| o.is_le()),
            Filter::Gt(k, v) => m.get(k).and_then(|x| cmp(x, v)).is_some_and(|o| o.is_gt()),
            Filter::Ge(k, v) => m.get(k).and_then(|x| cmp(x, v)).is_some_and(|o| o.is_ge()),
            Filter::Prefix(k, p) => m
                .get(k)
                .and_then(Value::as_str)
                .is_some_and(|s| s.starts_with(p.as_str())),
            Filter::Exists(k) => m.contains_key(k),
            Filter::And(fs) => fs.iter().all(|f| f.eval(m)),
            Filter::Or(fs) => fs.iter().any(|f| f.eval(m)),
            Filter::Not(f) => !f.eval(m),
        }
    }
}

/// Equality with int/float interchange.
pub(crate) fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => *x as f64 == *y,
        _ => a == b,
    }
}

/// Ordering for same-typed values (numbers across int/float); `None` when
/// the types do not compare.
pub(crate) fn cmp(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
        (Value::Float(_), _) | (_, Value::Float(_)) => a.as_float()?.partial_cmp(&b.as_float()?),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn rec() -> Metadata {
        HashMap::from([
            ("doc".to_string(), Value::from("intro.md")),
            ("year".to_string(), Value::from(2024i64)),
            ("score".to_string(), Value::from(0.5f64)),
            ("public".to_string(), Value::from(true)),
        ])
    }

    #[test]
    fn evaluates_each_operator() {
        let m = rec();
        assert!(Filter::eq("doc", "intro.md").eval(&m));
        assert!(!Filter::eq("doc", "other.md").eval(&m));
        assert!(
            !Filter::eq("doc", 2024i64).eval(&m),
            "type mismatch is false, not an error"
        );
        assert!(
            Filter::eq("year", 2024.0f64).eval(&m),
            "ints compare with floats"
        );
        assert!(Filter::ne("year", 2023i64).eval(&m));
        assert!(
            !Filter::ne("missing", 1i64).eval(&m),
            "ne needs the key present"
        );
        assert!(Filter::is_in("doc", ["a.md", "intro.md"]).eval(&m));
        assert!(
            Filter::ge("year", 2024i64).eval(&m)
                && Filter::lt("year", 2025i64).eval(&m)
                && !Filter::gt("year", 2024i64).eval(&m)
        );
        assert!(Filter::le("score", 0.5f64).eval(&m) && !Filter::lt("score", 0.5f64).eval(&m));
        assert!(
            !Filter::lt("doc", 5i64).eval(&m),
            "string against int does not compare"
        );
        assert!(Filter::prefix("doc", "intro").eval(&m) && !Filter::prefix("year", "20").eval(&m));
        assert!(Filter::exists("public").eval(&m) && !Filter::exists("nope").eval(&m));
        assert!(Filter::and([Filter::eq("public", true), Filter::gt("year", 2000i64)]).eval(&m));
        assert!(Filter::or([Filter::eq("public", false), Filter::gt("year", 2000i64)]).eval(&m));
        assert!(Filter::not(Filter::eq("public", false)).eval(&m));
        assert!(Filter::and([]).eval(&m) && !Filter::or([]).eval(&m));
    }
}
