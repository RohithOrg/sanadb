//! Python bindings. `Db` is thread-safe on the Rust side, so every call that
//! does real work releases the GIL.

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use sanadb_core::{Batch, Filter as Expr, GraphParams, Metadata, Metric, Options, Storage, Value};

fn err(e: sanadb_core::Error) -> PyErr {
    match e {
        sanadb_core::Error::Io(io) => PyIOError::new_err(io.to_string()),
        sanadb_core::Error::DimMismatch { .. }
        | sanadb_core::Error::MetaTooLarge
        | sanadb_core::Error::HeaderMismatch { .. }
        | sanadb_core::Error::InvalidVector => PyValueError::new_err(e.to_string()),
        sanadb_core::Error::TypeMismatch { .. } => PyTypeError::new_err(e.to_string()),
        sanadb_core::Error::ReadOnly => pyo3::exceptions::PyPermissionError::new_err(e.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Accept a 1-D numpy array (any float dtype) or any sequence of numbers.
fn to_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(a) = obj.downcast::<PyArray1<f32>>() {
        return Ok(a.readonly().as_slice()?.to_vec());
    }
    if let Ok(a) = obj.downcast::<PyArray1<f64>>() {
        return Ok(a.readonly().as_slice()?.iter().map(|&x| x as f32).collect());
    }
    obj.extract::<Vec<f32>>()
}

/// `bool` is checked before `int` because Python's bool is an int subclass.
fn to_value(v: &Bound<'_, PyAny>) -> PyResult<Value> {
    if v.is_instance_of::<PyBool>() {
        Ok(Value::Bool(v.extract()?))
    } else if v.is_instance_of::<PyInt>() {
        Ok(Value::Int(v.extract()?))
    } else if v.is_instance_of::<PyFloat>() {
        Ok(Value::Float(v.extract()?))
    } else if v.is_instance_of::<PyString>() {
        Ok(Value::Str(v.extract()?))
    } else {
        Err(PyTypeError::new_err(format!(
            "metadata values must be str, int, float or bool, got {}",
            v.get_type().name()?
        )))
    }
}

fn to_meta(obj: Option<&Bound<'_, PyDict>>) -> PyResult<Metadata> {
    let mut m = Metadata::new();
    if let Some(d) = obj {
        for (k, v) in d.iter() {
            m.insert(k.extract::<String>()?, to_value(&v)?);
        }
    }
    Ok(m)
}

fn meta_to_py<'py>(py: Python<'py>, m: &Metadata) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in m {
        match v {
            Value::Str(s) => d.set_item(k, s)?,
            Value::Int(i) => d.set_item(k, *i)?,
            Value::Float(x) => d.set_item(k, *x)?,
            Value::Bool(b) => d.set_item(k, *b)?,
        }
    }
    Ok(d)
}

/// Either a filter dict (compiled to a `Filter`, evaluated in Rust on the
/// index's columns, GIL released) or a Python callable taking the metadata
/// dict (GIL held during the search, applied to visited nodes).
enum Filter {
    Expr(Expr),
    Callable(PyObject),
}

/// Parse the filter dict. `{"k": v}` is equality; `{"k": {"$op": v}}` applies
/// operators `$eq $ne $in $lt $le $lte $gt $ge $gte $prefix $exists`; the keys
/// `$and`, `$or` take lists of dicts and `$not` one dict. Several keys in one
/// dict are and-ed.
fn to_filter(d: &Bound<'_, PyDict>) -> PyResult<Expr> {
    let mut terms = Vec::with_capacity(d.len());
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let term = match key.as_str() {
            "$and" | "$or" => {
                let list = v.downcast::<PyList>().map_err(|_| {
                    PyValueError::new_err(format!("{key} takes a list of filter dicts"))
                })?;
                let fs = list
                    .iter()
                    .map(|o| to_filter(o.downcast::<PyDict>()?))
                    .collect::<PyResult<Vec<_>>>()?;
                if key == "$and" {
                    Expr::And(fs)
                } else {
                    Expr::Or(fs)
                }
            }
            "$not" => Expr::Not(Box::new(to_filter(
                v.downcast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("$not takes a filter dict"))?,
            )?)),
            _ if key.starts_with('$') => {
                return Err(PyValueError::new_err(format!(
                    "unknown filter operator {key}"
                )));
            }
            _ => match v.downcast::<PyDict>() {
                Err(_) => Expr::Eq(key, to_value(&v)?),
                Ok(ops) => {
                    let mut fs = Vec::with_capacity(ops.len());
                    for (op, arg) in ops.iter() {
                        let op: String = op.extract()?;
                        fs.push(match op.as_str() {
                            "$eq" => Expr::Eq(key.clone(), to_value(&arg)?),
                            "$ne" => Expr::Ne(key.clone(), to_value(&arg)?),
                            "$in" => Expr::In(
                                key.clone(),
                                arg.try_iter()?
                                    .map(|x| to_value(&x?))
                                    .collect::<PyResult<Vec<_>>>()?,
                            ),
                            "$lt" => Expr::Lt(key.clone(), to_value(&arg)?),
                            "$le" | "$lte" => Expr::Le(key.clone(), to_value(&arg)?),
                            "$gt" => Expr::Gt(key.clone(), to_value(&arg)?),
                            "$ge" | "$gte" => Expr::Ge(key.clone(), to_value(&arg)?),
                            "$prefix" => Expr::Prefix(key.clone(), arg.extract()?),
                            "$exists" => {
                                let want: bool = arg.extract()?;
                                if want {
                                    Expr::Exists(key.clone())
                                } else {
                                    Expr::Not(Box::new(Expr::Exists(key.clone())))
                                }
                            }
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "unknown filter operator {op} for key {key}"
                                )));
                            }
                        });
                    }
                    if fs.len() == 1 {
                        fs.pop().unwrap()
                    } else {
                        Expr::And(fs)
                    }
                }
            },
        };
        terms.push(term);
    }
    Ok(if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        Expr::And(terms)
    })
}

#[pyclass(name = "Db", module = "sanadb")]
struct PyDb {
    inner: Arc<sanadb_core::Db>,
}

#[pymethods]
impl PyDb {
    /// Open or create a store.
    ///
    /// metric: "l2" (squared Euclidean), "dot" (negative inner product, vectors
    /// stored as given) or "cosine" (negative cosine similarity; vectors are
    /// normalised on insert and query).
    ///
    /// storage="int8" stores indexed vectors with 8-bit scalar quantisation
    /// (a quarter of the bytes per hop; exact re-ranking and get() use the
    /// dequantised vector). The log keeps f32 until the next checkpoint.
    ///
    /// read_only=True opens an existing store beside its writer (any process):
    /// no lock is taken, reads follow the writer's appends and checkpoints,
    /// and writes raise PermissionError.
    #[new]
    #[pyo3(signature = (path, dim, *, metric = "l2", search_beam = 64, max_search_beam = 4096, io_width = 4, sync_every_write = false, pq = true, max_degree = 32, build_beam = 100, alpha = 1.2, read_only = false, storage = "f32"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: String,
        dim: u32,
        metric: &str,
        search_beam: u32,
        max_search_beam: u32,
        io_width: u32,
        sync_every_write: bool,
        pq: bool,
        max_degree: u32,
        build_beam: u32,
        alpha: f32,
        read_only: bool,
        storage: &str,
    ) -> PyResult<Self> {
        let metric = match metric {
            "l2" => Metric::L2,
            "dot" | "ip" => Metric::Dot,
            "cosine" => Metric::Cosine,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown metric {other:?}; use \"l2\", \"dot\" or \"cosine\""
                )));
            }
        };
        let storage = match storage {
            "f32" => Storage::F32,
            "int8" | "i8" => Storage::I8,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown storage {other:?}; use \"f32\" or \"int8\""
                )));
            }
        };
        let opts = Options::new(dim)
            .metric(metric)
            .search_beam(search_beam)
            .max_search_beam(max_search_beam)
            .io_width(io_width)
            .sync_every_write(sync_every_write)
            .graph(GraphParams {
                max_degree,
                build_beam,
                alpha,
                use_pq: pq,
                storage,
                ..GraphParams::default()
            });
        let db = py
            .allow_threads(|| {
                if read_only {
                    sanadb_core::Db::open_read_only(path, opts)
                } else {
                    sanadb_core::Db::open(path, opts)
                }
            })
            .map_err(err)?;
        Ok(PyDb {
            inner: Arc::new(db),
        })
    }

    #[getter]
    fn dim(&self) -> u32 {
        self.inner.dim()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Records held in the in-RAM delta (written since the last checkpoint).
    #[getter]
    fn delta_len(&self) -> usize {
        self.inner.delta_len()
    }

    /// Insert or replace one record. `meta` values are stored as strings.
    #[pyo3(signature = (id, vector, meta = None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: u64,
        vector: &Bound<'_, PyAny>,
        meta: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let v = to_vec(vector)?;
        let m = to_meta(meta)?;
        let db = self.inner.clone();
        py.allow_threads(move || db.insert(id, &v, m)).map_err(err)
    }

    /// Insert many records from a 2-D float32 array, one row per id.
    #[pyo3(signature = (ids, vectors, metas = None))]
    fn insert_many(
        &self,
        py: Python<'_>,
        ids: Vec<u64>,
        vectors: PyReadonlyArray2<'_, f32>,
        metas: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let shape = vectors.shape();
        if shape[0] != ids.len() {
            return Err(PyValueError::new_err(format!(
                "{} ids but {} rows",
                ids.len(),
                shape[0]
            )));
        }
        let metas: Vec<Metadata> = match metas {
            Some(l) => {
                if l.len() != ids.len() {
                    return Err(PyValueError::new_err("metas must have one entry per id"));
                }
                l.iter()
                    .map(|o| {
                        if o.is_none() {
                            Ok(Metadata::new())
                        } else {
                            to_meta(Some(o.downcast::<PyDict>()?))
                        }
                    })
                    .collect::<PyResult<_>>()?
            }
            None => vec![Metadata::new(); ids.len()],
        };
        let dim = shape[1];
        let flat: Vec<f32> = vectors.as_array().iter().copied().collect();
        let db = self.inner.clone();
        // One atomic batch: all rows land or none does, and a reader never
        // sees part of them.
        py.allow_threads(move || {
            let mut b = Batch::new();
            for (i, (id, m)) in ids.into_iter().zip(metas).enumerate() {
                b = b.insert(id, &flat[i * dim..(i + 1) * dim], m);
            }
            db.write_batch(b)
        })
        .map_err(err)
    }

    fn delete(&self, py: Python<'_>, id: u64) -> PyResult<bool> {
        let db = self.inner.clone();
        py.allow_threads(move || db.delete(id)).map_err(err)
    }

    /// Delete several records atomically. Ids that do not exist are ignored.
    fn delete_many(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<()> {
        let db = self.inner.clone();
        py.allow_threads(move || {
            db.write_batch(ids.into_iter().fold(Batch::new(), |b, id| b.delete(id)))
        })
        .map_err(err)
    }

    /// `(vector, meta)` or `None`.
    fn get<'py>(&self, py: Python<'py>, id: u64) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match self.inner.get(id) {
            None => Ok(None),
            Some((v, m)) => {
                let arr = PyArray1::from_vec(py, v);
                Ok(Some(PyTuple::new(
                    py,
                    [arr.into_any(), meta_to_py(py, &m)?.into_any()],
                )?))
            }
        }
    }

    /// k nearest neighbours as a list of `(id, score)`, closest first.
    ///
    /// `filter` is a dict of metadata pairs that must all match exactly, or a
    /// callable `meta -> bool`. Dict filters run with the GIL released.
    #[pyo3(signature = (query, k = 10, filter = None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let q = to_vec(query)?;
        let filter = match filter {
            None => None,
            Some(f) if f.is_instance_of::<PyDict>() => {
                Some(Filter::Expr(to_filter(f.downcast::<PyDict>()?)?))
            }
            Some(f) if f.is_callable() => Some(Filter::Callable(f.clone().unbind())),
            Some(_) => return Err(PyValueError::new_err("filter must be a dict or a callable")),
        };
        let db = self.inner.clone();
        let hits = match filter {
            None => py.allow_threads(move || db.search(&q, k)).map_err(err)?,
            Some(Filter::Expr(f)) => py
                .allow_threads(move || db.search_where(&q, k, &f))
                .map_err(err)?,
            Some(Filter::Callable(f)) => {
                let py_err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
                let hits = db
                    .search_filtered(&q, k, |m| {
                        match meta_to_py(py, m).and_then(|d| f.bind(py).call1((d,))?.is_truthy()) {
                            Ok(b) => b,
                            Err(e) => {
                                *py_err.borrow_mut() = Some(e);
                                false
                            }
                        }
                    })
                    .map_err(err)?;
                if let Some(e) = py_err.into_inner() {
                    return Err(e);
                }
                hits
            }
        };
        Ok(hits.into_iter().map(|h| (h.id, h.score)).collect())
    }

    /// k nearest neighbours for every row of a 2-D float32 array, computed
    /// across all cores with the GIL released. Returns `(ids, scores)`, two
    /// `(n, k)` arrays; rows with fewer than `k` results are padded with
    /// `ids = -1` and `scores = nan`. `filter` is a filter dict as for
    /// `search` (callables are not accepted here).
    #[pyo3(signature = (queries, k = 10, filter = None))]
    fn search_many<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        filter: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let n = queries.shape()[0];
        let flat: Vec<f32> = queries.as_slice()?.to_vec();
        let f = filter.map(to_filter).transpose()?;
        let db = self.inner.clone();
        let hits = py
            .allow_threads(move || match f {
                None => db.search_many(&flat, k),
                Some(f) => db.search_many_where(&flat, k, &f),
            })
            .map_err(err)?;
        let mut ids = vec![-1i64; n * k];
        let mut scores = vec![f32::NAN; n * k];
        for (row, hs) in hits.iter().enumerate() {
            for (j, h) in hs.iter().enumerate() {
                ids[row * k + j] = h.id as i64;
                scores[row * k + j] = h.score;
            }
        }
        Ok((
            PyArray1::from_vec(py, ids).reshape([n, k])?,
            PyArray1::from_vec(py, scores).reshape([n, k])?,
        ))
    }

    /// Read-only stores: bring the view up to date with the writer now. Every
    /// read does this anyway; call it to pick up writes without reading.
    fn refresh(&self, py: Python<'_>) -> PyResult<()> {
        let db = self.inner.clone();
        py.allow_threads(move || db.refresh()).map_err(err)
    }

    /// fsync the log so acknowledged writes survive power loss.
    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        let db = self.inner.clone();
        py.allow_threads(move || db.sync()).map_err(err)
    }

    /// Rebuild the graph index over all live records and compact the log.
    /// Reads and writes from other threads continue while it runs.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        let db = self.inner.clone();
        py.allow_threads(move || db.checkpoint()).map_err(err)
    }

    /// `{"nodes", "pq_subspaces", "tombstones"}` or `None` before the first checkpoint.
    fn index_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.index_stats() {
            None => Ok(None),
            Some(s) => {
                let d = PyDict::new(py);
                d.set_item("nodes", s.nodes)?;
                d.set_item("pq_subspaces", s.pq_subspaces)?;
                d.set_item("tombstones", s.tombstones)?;
                d.set_item(
                    "storage",
                    match s.storage {
                        Storage::F32 => "f32",
                        Storage::I8 => "int8",
                    },
                )?;
                Ok(Some(d))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "sanadb.Db(dim={}, len={}, delta={})",
            self.inner.dim(),
            self.inner.len(),
            self.inner.delta_len()
        )
    }
}

#[pymodule]
fn sanadb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDb>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
