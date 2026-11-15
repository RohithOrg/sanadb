//! Python bindings. `Db` is thread-safe on the Rust side, so every call that
//! does real work releases the GIL.

use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use sanadb_core::{GraphParams, Metadata, Metric, Options};

fn err(e: sanadb_core::Error) -> PyErr {
    match e {
        sanadb_core::Error::Io(io) => PyIOError::new_err(io.to_string()),
        sanadb_core::Error::DimMismatch { .. } | sanadb_core::Error::MetaTooLarge | sanadb_core::Error::HeaderMismatch { .. } => PyValueError::new_err(e.to_string()),
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

fn to_meta(obj: Option<&Bound<'_, PyDict>>) -> PyResult<Metadata> {
    let mut m = Metadata::new();
    if let Some(d) = obj {
        for (k, v) in d.iter() {
            m.insert(k.extract::<String>()?, v.str()?.to_string());
        }
    }
    Ok(m)
}

fn meta_to_py<'py>(py: Python<'py>, m: &Metadata) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in m {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Either a dict of exact-match pairs (evaluated in Rust, GIL released) or a
/// Python callable taking the metadata dict (GIL held during the search).
enum Filter {
    Equals(Metadata),
    Callable(PyObject),
}

#[pyclass(name = "Db", module = "sanadb")]
struct PyDb {
    inner: Arc<sanadb_core::Db>,
}

#[pymethods]
impl PyDb {
    /// Open or create a store.
    ///
    /// metric: "l2" (squared Euclidean) or "dot" (negative inner product;
    /// normalise your vectors for cosine).
    #[new]
    #[pyo3(signature = (path, dim, *, metric = "l2", search_beam = 64, max_search_beam = 4096, sync_every_write = false, pq = true, max_degree = 32, build_beam = 100, alpha = 1.2))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: String,
        dim: u32,
        metric: &str,
        search_beam: u32,
        max_search_beam: u32,
        sync_every_write: bool,
        pq: bool,
        max_degree: u32,
        build_beam: u32,
        alpha: f32,
    ) -> PyResult<Self> {
        let metric = match metric {
            "l2" => Metric::L2,
            "dot" | "ip" => Metric::Dot,
            other => return Err(PyValueError::new_err(format!("unknown metric {other:?}; use \"l2\" or \"dot\""))),
        };
        let opts = Options::new(dim)
            .metric(metric)
            .search_beam(search_beam)
            .max_search_beam(max_search_beam)
            .sync_every_write(sync_every_write)
            .graph(GraphParams { max_degree, build_beam, alpha, use_pq: pq });
        let db = py.allow_threads(|| sanadb_core::Db::open(path, opts)).map_err(err)?;
        Ok(PyDb { inner: Arc::new(db) })
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
    fn insert(&self, py: Python<'_>, id: u64, vector: &Bound<'_, PyAny>, meta: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let v = to_vec(vector)?;
        let m = to_meta(meta)?;
        let db = self.inner.clone();
        py.allow_threads(move || db.insert(id, &v, m)).map_err(err)
    }

    /// Insert many records from a 2-D float32 array, one row per id.
    #[pyo3(signature = (ids, vectors, metas = None))]
    fn insert_many(&self, py: Python<'_>, ids: Vec<u64>, vectors: PyReadonlyArray2<'_, f32>, metas: Option<&Bound<'_, PyList>>) -> PyResult<()> {
        let shape = vectors.shape();
        if shape[0] != ids.len() {
            return Err(PyValueError::new_err(format!("{} ids but {} rows", ids.len(), shape[0])));
        }
        let metas: Vec<Metadata> = match metas {
            Some(l) => {
                if l.len() != ids.len() {
                    return Err(PyValueError::new_err("metas must have one entry per id"));
                }
                l.iter().map(|o| if o.is_none() { Ok(Metadata::new()) } else { to_meta(Some(o.downcast::<PyDict>()?)) }).collect::<PyResult<_>>()?
            }
            None => vec![Metadata::new(); ids.len()],
        };
        let dim = shape[1];
        let flat: Vec<f32> = vectors.as_array().iter().copied().collect();
        let db = self.inner.clone();
        py.allow_threads(move || {
            for (i, (id, m)) in ids.into_iter().zip(metas).enumerate() {
                db.insert(id, &flat[i * dim..(i + 1) * dim], m)?;
            }
            Ok(())
        })
        .map_err(err)
    }

    fn delete(&self, py: Python<'_>, id: u64) -> PyResult<bool> {
        let db = self.inner.clone();
        py.allow_threads(move || db.delete(id)).map_err(err)
    }

    /// `(vector, meta)` or `None`.
    fn get<'py>(&self, py: Python<'py>, id: u64) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match self.inner.get(id) {
            None => Ok(None),
            Some((v, m)) => {
                let arr = PyArray1::from_vec(py, v);
                Ok(Some(PyTuple::new(py, [arr.into_any(), meta_to_py(py, &m)?.into_any()])?))
            }
        }
    }

    /// k nearest neighbours as a list of `(id, score)`, closest first.
    ///
    /// `filter` is a dict of metadata pairs that must all match exactly, or a
    /// callable `meta -> bool`. Dict filters run with the GIL released.
    #[pyo3(signature = (query, k = 10, filter = None))]
    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize, filter: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<(u64, f32)>> {
        let q = to_vec(query)?;
        let filter = match filter {
            None => None,
            Some(f) if f.is_instance_of::<PyDict>() => Some(Filter::Equals(to_meta(Some(f.downcast::<PyDict>()?))?)),
            Some(f) if f.is_callable() => Some(Filter::Callable(f.clone().unbind())),
            Some(_) => return Err(PyValueError::new_err("filter must be a dict or a callable")),
        };
        let db = self.inner.clone();
        let hits = match filter {
            None => py.allow_threads(move || db.search(&q, k)).map_err(err)?,
            Some(Filter::Equals(want)) => py
                .allow_threads(move || db.search_filtered(&q, k, |m| want.iter().all(|(k, v)| m.get(k) == Some(v))))
                .map_err(err)?,
            Some(Filter::Callable(f)) => {
                let py_err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
                let hits = db
                    .search_filtered(&q, k, |m| match meta_to_py(py, m).and_then(|d| f.bind(py).call1((d,))?.is_truthy()) {
                        Ok(b) => b,
                        Err(e) => {
                            *py_err.borrow_mut() = Some(e);
                            false
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
                Ok(Some(d))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("sanadb.Db(dim={}, len={}, delta={})", self.inner.dim(), self.inner.len(), self.inner.delta_len())
    }
}

#[pymodule]
fn sanadb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDb>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
