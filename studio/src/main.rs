//! `sanadb-studio <store.sana> [--port 7070] [--text-field KEY] [--read-only]
//!   [--embed-url URL --embed-model NAME [--embed-key-env VAR] | --embed-cmd CMD]`
//!
//! With an embedder attached the page searches by text and adds records
//! from text: `--embed-url` is an OpenAI-compatible `/v1/embeddings`
//! endpoint (OpenAI, Ollama, LM Studio, vLLM, text-embeddings-inference),
//! `--embed-cmd` any command that reads a JSON array of strings on stdin
//! and prints a JSON array of float arrays.
//!
//! Serves a single page on localhost that opens the store, lists and
//! searches records, draws a 2-D map of a sample of the vectors (PCA, with
//! k-means colours), shows the graph walk of a search on that map, and edits
//! metadata. One binary, no build step: the page is embedded.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use sanadb::{Db, Filter, Hit, Kind, Metric, Options, Value};
use serde_json::{Value as J, json};
use tiny_http::{Header, Method, Request, Response, Server};

const PAGE: &str = include_str!("../ui/index.html");
const FAVICON: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#2E4BE8"/><circle cx="11" cy="20" r="3.2" fill="#fff"/><circle cx="21" cy="11" r="3.2" fill="#fff"/><circle cx="22" cy="22" r="2.2" fill="#fff" opacity=".7"/><path d="M11 20 21 11M11 20l11 2" stroke="#fff" stroke-width="1.6" opacity=".8"/></svg>"##;

/// How text becomes a vector, when the user attached a model.
enum Embedder {
    Http {
        url: String,
        model: String,
        key: Option<String>,
    },
    Cmd(String),
}

impl Embedder {
    fn name(&self) -> String {
        match self {
            Embedder::Http { model, .. } => model.clone(),
            Embedder::Cmd(c) => format!("command: {c}"),
        }
    }

    fn describe(&self) -> J {
        match self {
            Embedder::Http { url, model, .. } => {
                json!({"kind": "http", "url": url, "model": model})
            }
            Embedder::Cmd(c) => json!({"kind": "command", "command": c}),
        }
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        match self {
            Embedder::Http { url, model, key } => {
                let mut req = ureq::post(url).header("Content-Type", "application/json");
                if let Some(k) = key {
                    req = req.header("Authorization", &format!("Bearer {k}"));
                }
                let body = json!({"model": model, "input": texts});
                let mut resp = req
                    .send_json(&body)
                    .map_err(|e| format!("embedding request to {url} failed: {e}"))?;
                let v: J = resp
                    .body_mut()
                    .read_json()
                    .map_err(|e| format!("embedding response is not JSON: {e}"))?;
                let data = v["data"].as_array().ok_or_else(|| {
                    format!(
                        "embedding response has no `data` array: {}",
                        short(&v.to_string())
                    )
                })?;
                let mut out: Vec<(usize, Vec<f32>)> = data
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let idx = d["index"].as_u64().map(|x| x as usize).unwrap_or(i);
                        let e = d["embedding"]
                            .as_array()
                            .ok_or("embedding entry has no `embedding` array")?
                            .iter()
                            .map(|x| {
                                x.as_f64()
                                    .map(|x| x as f32)
                                    .ok_or("non-numeric embedding value")
                            })
                            .collect::<Result<Vec<f32>, &str>>()?;
                        Ok((idx, e))
                    })
                    .collect::<Result<_, &str>>()
                    .map_err(|e| e.to_string())?;
                out.sort_by_key(|(i, _)| *i);
                if out.len() != texts.len() {
                    return Err(format!(
                        "asked for {} embeddings, got {}",
                        texts.len(),
                        out.len()
                    ));
                }
                Ok(out.into_iter().map(|(_, e)| e).collect())
            }
            Embedder::Cmd(cmd) => {
                use std::io::Write;
                let mut child = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(cmd)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::inherit())
                    .spawn()
                    .map_err(|e| format!("cannot run embed command: {e}"))?;
                child
                    .stdin
                    .take()
                    .unwrap()
                    .write_all(serde_json::to_string(texts).unwrap().as_bytes())
                    .map_err(|e| e.to_string())?;
                let out = child.wait_with_output().map_err(|e| e.to_string())?;
                if !out.status.success() {
                    return Err(format!("embed command exited with {}", out.status));
                }
                let v: Vec<Vec<f32>> = serde_json::from_slice(&out.stdout).map_err(|e| {
                    format!("embed command output is not a JSON array of float arrays: {e}")
                })?;
                if v.len() != texts.len() {
                    return Err(format!(
                        "asked for {} embeddings, got {}",
                        texts.len(),
                        v.len()
                    ));
                }
                Ok(v)
            }
        }
    }
}

fn short(s: &str) -> String {
    if s.len() > 200 {
        format!("{}…", &s[..200])
    } else {
        s.to_string()
    }
}

struct App {
    db: Db,
    path: String,
    dim: usize,
    metric: Metric,
    read_only: bool,
    embed: Option<Embedder>,
    /// Where `add` stores the text it embedded: the text field, else `text`.
    text_key: String,
    /// PCA basis of the last map: mean, three components, and the sampled
    /// points, so later searches can be projected onto the same axes.
    map: Mutex<Option<MapState>>,
}

struct MapState {
    mean: Vec<f32>,
    axes: [Vec<f32>; 3],
    scale: f32,
    points: J,
    /// The sampled records' metadata, cluster and position, for labelling clusters.
    sampled: Vec<(sanadb::Metadata, usize, [f32; 3])>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path = None;
    let mut port = 7070u16;
    let mut text_field: Option<String> = None;
    let mut read_only = false;
    let (mut embed_url, mut embed_model, mut embed_key_env, mut embed_cmd) =
        (None, None, None, None);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--embed-url" => {
                embed_url = Some(args[i + 1].clone());
                i += 1;
            }
            "--embed-model" => {
                embed_model = Some(args[i + 1].clone());
                i += 1;
            }
            "--embed-key-env" => {
                embed_key_env = Some(args[i + 1].clone());
                i += 1;
            }
            "--embed-cmd" => {
                embed_cmd = Some(args[i + 1].clone());
                i += 1;
            }
            "--port" => {
                port = args[i + 1].parse().expect("--port takes a number");
                i += 1;
            }
            "--text-field" => {
                text_field = Some(args[i + 1].clone());
                i += 1;
            }
            "--read-only" => read_only = true,
            a => path = Some(a.to_string()),
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!(
            "usage: sanadb-studio <store.sana> [--port 7070] [--text-field KEY] [--read-only] [--embed-url URL --embed-model NAME [--embed-key-env VAR] | --embed-cmd CMD]"
        );
        std::process::exit(2);
    };
    let embed = match (embed_url, embed_cmd) {
        (Some(url), _) => {
            let Some(model) = embed_model else {
                eprintln!("--embed-url needs --embed-model");
                std::process::exit(2);
            };
            let key = embed_key_env.and_then(|v| std::env::var(v).ok());
            Some(Embedder::Http { url, model, key })
        }
        (None, Some(cmd)) => Some(Embedder::Cmd(cmd)),
        (None, None) => None,
    };
    let text_key = text_field.clone().unwrap_or_else(|| "text".to_string());
    let (dim, metric) = Db::peek(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let mut opts = Options::new(dim).metric(metric);
    if let Some(f) = text_field {
        opts = opts.text_field(f);
    }
    let (db, read_only) = if read_only {
        (
            Db::open_read_only(&path, opts).expect("open read-only"),
            true,
        )
    } else {
        match Db::open(&path, opts.clone()) {
            Ok(db) => (db, false),
            Err(sanadb::Error::Locked) => {
                eprintln!("another writer holds {path}; opening read-only");
                (
                    Db::open_read_only(&path, opts).expect("open read-only"),
                    true,
                )
            }
            Err(e) => {
                eprintln!("{path}: {e}");
                std::process::exit(1);
            }
        }
    };
    // Attaching a model does not claim the store's vectors came from it;
    // the name is recorded the first time this model writes a record. But
    // say so when the store names a different one: vectors from two
    // models do not compare.
    if let (Some(e), Some(have)) = (&embed, db.property("embedding_model"))
        && have != e.name()
    {
        eprintln!(
            "warning: this store's vectors were embedded with {have:?}; the attached model is {:?}. Queries and new records will not be comparable.",
            e.name()
        );
    }
    let app = Arc::new(App {
        db,
        path: path.clone(),
        dim: dim as usize,
        metric,
        read_only,
        embed,
        text_key,
        map: Mutex::new(None),
    });
    let server = Server::http(("127.0.0.1", port)).expect("bind");
    println!(
        "sanadb studio {}: {path} ({} records, dim {dim}, {metric:?}){} at http://127.0.0.1:{port}/",
        env!("CARGO_PKG_VERSION"),
        app.db.len(),
        if app.read_only { ", read-only" } else { "" }
    );
    for req in server.incoming_requests() {
        let app = app.clone();
        std::thread::spawn(move || handle(app, req));
    }
}

fn handle(app: Arc<App>, mut req: Request) {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let query: HashMap<String, String> = url
        .split_once('?')
        .map(|(_, q)| {
            q.split('&')
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let mut body = String::new();
    let _ = req.as_reader().read_to_string(&mut body);
    let result = match (req.method(), path.as_str()) {
        (Method::Get, "/") => {
            let r = Response::from_string(PAGE).with_header(
                Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
            );
            let _ = req.respond(r);
            return;
        }
        (Method::Get, "/favicon.svg") | (Method::Get, "/favicon.ico") => {
            let r = Response::from_string(FAVICON)
                .with_header(Header::from_bytes("Content-Type", "image/svg+xml").unwrap());
            let _ = req.respond(r);
            return;
        }
        (Method::Get, "/api/info") => Ok(info(&app)),
        (Method::Get, "/api/ids") => Ok(ids(&app, &query)),
        (Method::Get, "/api/record") => record(&app, &query),
        (Method::Post, "/api/record") => update(&app, &query, &body),
        (Method::Delete, "/api/record") => remove(&app, &query),
        (Method::Post, "/api/search") => search(&app, &body),
        (Method::Post, "/api/add") => add(&app, &body),
        (Method::Get, "/api/map") => map(&app, &query),
        (Method::Get, "/api/labels") => labels(&app, &query),
        (Method::Get, "/api/files") => Ok(files(&app)),
        (Method::Post, "/api/checkpoint") => checkpoint(&app),
        _ => Err("not found".to_string()),
    };
    let (status, json) = match result {
        Ok(j) => (200, j),
        Err(e) => (400, json!({ "error": e })),
    };
    let r = Response::from_string(json.to_string())
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap());
    let _ = req.respond(r);
}

fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::L2 => "l2",
        Metric::Dot => "dot",
        Metric::Cosine => "cosine",
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Str => "str",
        Kind::Int => "int",
        Kind::Float => "float",
        Kind::Bool => "bool",
        Kind::Sparse => "sparse",
    }
}

fn info(app: &App) -> J {
    let s = app.db.stats();
    let ix = app.db.index_stats();
    json!({
        "path": app.path,
        "version": env!("CARGO_PKG_VERSION"),
        "dim": app.dim,
        "metric": metric_name(app.metric),
        "read_only": app.read_only,
        "embed": app.embed.as_ref().map(|e| e.describe()),
        "properties": app.db.properties(),
        "text_key": app.text_key,
        "len": app.db.len(),
        "delta_len": app.db.delta_len(),
        "index": ix.map(|x| json!({
            "nodes": x.nodes, "pq_subspaces": x.pq_subspaces, "tombstones": x.tombstones,
            "storage": format!("{:?}", x.storage).to_lowercase(),
            "file": app.db.index_path().map(|p| p.display().to_string()),
        })),
        "keys": app.db.keys().into_iter().map(|(k, t)| json!({"name": k, "kind": kind_name(t)})).collect::<Vec<_>>(),
        "stats": {
            "searches": s.searches, "graph_walks": s.graph_walks, "filtered_walks": s.filtered_walks,
            "exact_scans": s.exact_scans, "delta_only": s.delta_only, "text_searches": s.text_searches,
            "sparse_searches": s.sparse_searches, "inserts": s.inserts, "deletes": s.deletes,
            "batches": s.batches, "checkpoints": s.checkpoints, "incremental_checkpoints": s.incremental_checkpoints,
            "failed_checkpoints": s.failed_checkpoints, "last_checkpoint_ms": s.last_checkpoint_ms,
        },
    })
}

fn ids(app: &App, q: &HashMap<String, String>) -> J {
    let all = app.db.ids();
    let offset: usize = q.get("offset").and_then(|s| s.parse().ok()).unwrap_or(0);
    let limit: usize = q
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
        .min(1000);
    let page: Vec<J> = all
        .iter()
        .skip(offset)
        .take(limit)
        .map(|&id| {
            let meta = app
                .db
                .get(id)
                .map(|(_, m)| meta_json(&m))
                .unwrap_or(J::Null);
            json!({"id": id, "meta": meta})
        })
        .collect();
    json!({"total": all.len(), "offset": offset, "records": page})
}

fn meta_json(m: &sanadb::Metadata) -> J {
    let mut o = serde_json::Map::new();
    let mut keys: Vec<&String> = m.keys().collect();
    keys.sort();
    for k in keys {
        let v = match &m[k] {
            Value::Str(s) => json!(s),
            Value::Int(i) => json!(i),
            Value::Float(x) => json!(x),
            Value::Bool(b) => json!(b),
            Value::Sparse(p) => {
                json!({"indices": p.iter().map(|x| x.0).collect::<Vec<_>>(), "weights": p.iter().map(|x| x.1).collect::<Vec<_>>()})
            }
        };
        o.insert(k.clone(), v);
    }
    J::Object(o)
}

fn json_meta(j: &J) -> Result<sanadb::Metadata, String> {
    let J::Object(o) = j else {
        return Err("metadata must be an object".into());
    };
    let mut m = sanadb::Metadata::new();
    for (k, v) in o {
        m.insert(k.clone(), json_value(v)?);
    }
    Ok(m)
}

fn json_value(v: &J) -> Result<Value, String> {
    Ok(match v {
        J::String(s) => Value::Str(s.clone()),
        J::Bool(b) => Value::Bool(*b),
        J::Number(n) if n.is_i64() => Value::Int(n.as_i64().unwrap()),
        J::Number(n) => Value::Float(n.as_f64().unwrap()),
        J::Object(o) if o.contains_key("indices") => {
            let ids = o["indices"].as_array().ok_or("indices must be a list")?;
            let ws = o
                .get("weights")
                .and_then(|w| w.as_array())
                .ok_or("weights must be a list")?;
            if ids.len() != ws.len() {
                return Err("indices and weights differ in length".into());
            }
            Value::Sparse(
                ids.iter()
                    .zip(ws)
                    .map(|(i, w)| {
                        Ok((
                            i.as_u64().ok_or("bad index")? as u32,
                            w.as_f64().ok_or("bad weight")? as f32,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            )
        }
        other => return Err(format!("unsupported metadata value {other}")),
    })
}

fn record(app: &App, q: &HashMap<String, String>) -> Result<J, String> {
    let id: u64 = q
        .get("id")
        .and_then(|s| s.parse().ok())
        .ok_or("id required")?;
    let (v, m) = app.db.get(id).ok_or("no such record")?;
    Ok(json!({"id": id, "vector": v, "meta": meta_json(&m)}))
}

fn update(app: &App, q: &HashMap<String, String>, body: &str) -> Result<J, String> {
    if app.read_only {
        return Err("store opened read-only".into());
    }
    let id: u64 = q
        .get("id")
        .and_then(|s| s.parse().ok())
        .ok_or("id required")?;
    let j: J = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let (old_v, old_m) = app.db.get(id).ok_or("no such record")?;
    // New metadata, and optionally a new vector: numbers, or text to
    // re-embed with the attached model.
    let mut meta = if j["meta"].is_null() {
        old_m
    } else {
        json_meta(&j["meta"])?
    };
    let v = if let Some(arr) = j["vector"].as_array() {
        arr.iter()
            .map(|x| {
                x.as_f64()
                    .map(|x| x as f32)
                    .ok_or("vector values must be numbers")
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if let Some(t) = j["text"].as_str().filter(|t| !t.trim().is_empty()) {
        let e = app
            .embed
            .as_ref()
            .ok_or("no embedding model attached: start studio with --embed-url or --embed-cmd")?;
        meta.insert(app.text_key.clone(), Value::Str(t.to_string()));
        note_model(app, e);
        e.embed(std::slice::from_ref(&t.to_string()))?.remove(0)
    } else {
        old_v
    };
    if v.len() != app.dim {
        return Err(format!(
            "{} dimensions given, the store has {}",
            v.len(),
            app.dim
        ));
    }
    app.db.insert(id, &v, meta).map_err(|e| e.to_string())?;
    *app.map.lock().unwrap() = None;
    record(app, q)
}

fn remove(app: &App, q: &HashMap<String, String>) -> Result<J, String> {
    if app.read_only {
        return Err("store opened read-only".into());
    }
    let id: u64 = q
        .get("id")
        .and_then(|s| s.parse().ok())
        .ok_or("id required")?;
    let existed = app.db.delete(id).map_err(|e| e.to_string())?;
    Ok(json!({"id": id, "deleted": existed}))
}

/// `{"k": v}` equality, `{"k": {"$op": v}}` operators, `$and`/`$or` lists,
/// `$not` one object; several keys in one object are and-ed. Same grammar
/// as the Python bindings.
fn json_filter(j: &J) -> Result<Filter, String> {
    let J::Object(o) = j else {
        return Err("filter must be an object".into());
    };
    let mut terms = Vec::new();
    for (k, v) in o {
        let t = match k.as_str() {
            "$and" | "$or" => {
                let fs = v
                    .as_array()
                    .ok_or("$and/$or take a list")?
                    .iter()
                    .map(json_filter)
                    .collect::<Result<Vec<_>, _>>()?;
                if k == "$and" {
                    Filter::and(fs)
                } else {
                    Filter::or(fs)
                }
            }
            "$not" => Filter::not(json_filter(v)?),
            _ if k.starts_with('$') => return Err(format!("unknown operator {k}")),
            _ => match v {
                J::Object(ops) => {
                    let mut fs = Vec::new();
                    for (op, x) in ops {
                        fs.push(match op.as_str() {
                            "$eq" => Filter::eq(k, json_value(x)?),
                            "$ne" => Filter::ne(k, json_value(x)?),
                            "$lt" => Filter::lt(k, json_value(x)?),
                            "$le" | "$lte" => Filter::le(k, json_value(x)?),
                            "$gt" => Filter::gt(k, json_value(x)?),
                            "$ge" | "$gte" => Filter::ge(k, json_value(x)?),
                            "$in" => Filter::In(
                                k.clone(),
                                x.as_array()
                                    .ok_or("$in takes a list")?
                                    .iter()
                                    .map(json_value)
                                    .collect::<Result<Vec<_>, _>>()?,
                            ),
                            "$prefix" => {
                                Filter::prefix(k, x.as_str().ok_or("$prefix takes a string")?)
                            }
                            "$contains" | "$has" => {
                                Filter::contains(k, x.as_str().ok_or("$contains takes a string")?)
                            }
                            "$exists" => {
                                if x.as_bool().unwrap_or(true) {
                                    Filter::exists(k)
                                } else {
                                    Filter::not(Filter::exists(k))
                                }
                            }
                            other => return Err(format!("unknown operator {other}")),
                        });
                    }
                    Filter::and(fs)
                }
                _ => Filter::eq(k, json_value(v)?),
            },
        };
        terms.push(t);
    }
    Ok(if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        Filter::and(terms)
    })
}

fn hits_json(app: &App, hits: &[Hit], with_meta: bool) -> Vec<J> {
    let st = app.map.lock().unwrap();
    hits.iter()
        .map(|h| {
            let mut o = json!({"id": h.id, "score": h.score});
            if (with_meta || st.is_some())
                && let Some((v, m)) = app.db.get(h.id)
            {
                if with_meta {
                    o["meta"] = meta_json(&m);
                }
                if let Some(s) = st.as_ref() {
                    let [x, y, z] = project(s, &v);
                    o["x"] = json!(x);
                    o["y"] = json!(y);
                    o["z"] = json!(z);
                }
            }
            o
        })
        .collect()
}

#[derive(serde::Deserialize)]
struct SearchReq {
    id: Option<u64>,
    vector: Option<Vec<f32>>,
    /// Text to embed with the attached model and use as the query vector.
    query_text: Option<String>,
    text: Option<String>,
    sparse: Option<(Vec<u32>, Vec<f32>)>,
    filter: Option<J>,
    #[serde(default = "ten")]
    k: usize,
    #[serde(default)]
    walk: bool,
}
fn ten() -> usize {
    10
}

fn search(app: &App, body: &str) -> Result<J, String> {
    let r: SearchReq = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let filter = r
        .filter
        .as_ref()
        .filter(|f| !f.as_object().is_some_and(|o| o.is_empty()))
        .map(json_filter)
        .transpose()?;
    let vector = match (&r.vector, r.id, &r.query_text) {
        (Some(v), _, _) => Some(v.clone()),
        (None, Some(id), _) => Some(app.db.get(id).ok_or("no such record")?.0),
        (None, None, Some(t)) if !t.trim().is_empty() => {
            let e = app.embed.as_ref().ok_or(
                "no embedding model attached: start studio with --embed-url or --embed-cmd",
            )?;
            let v = e.embed(std::slice::from_ref(t))?.remove(0);
            if v.len() != app.dim {
                return Err(format!(
                    "the model returned {} dimensions, the store has {}",
                    v.len(),
                    app.dim
                ));
            }
            Some(v)
        }
        _ => None,
    };
    let t = std::time::Instant::now();
    let (hits, walk, mode) = match (&vector, &r.text, &r.sparse) {
        (Some(v), Some(text), _) => (
            app.db
                .search_hybrid(v, text, r.k, filter.as_ref())
                .map_err(|e| e.to_string())?,
            Vec::new(),
            "hybrid",
        ),
        (Some(v), None, Some((i, w))) => {
            let q: Vec<(u32, f32)> = i.iter().copied().zip(w.iter().copied()).collect();
            (
                app.db
                    .search_hybrid_sparse(v, &q, r.k, filter.as_ref())
                    .map_err(|e| e.to_string())?,
                Vec::new(),
                "hybrid sparse",
            )
        }
        (Some(v), None, None) => match &filter {
            Some(f) => (
                app.db.search_where(v, r.k, f).map_err(|e| e.to_string())?,
                Vec::new(),
                "filtered",
            ),
            None if r.walk => {
                let (h, w) = app.db.search_traced(v, r.k).map_err(|e| e.to_string())?;
                (h, w, "walk")
            }
            None => (
                app.db.search(v, r.k).map_err(|e| e.to_string())?,
                Vec::new(),
                "nearest",
            ),
        },
        (None, Some(text), _) => (
            app.db
                .search_text(text, r.k, filter.as_ref())
                .map_err(|e| e.to_string())?,
            Vec::new(),
            "keyword",
        ),
        (None, None, Some((i, w))) => {
            let q: Vec<(u32, f32)> = i.iter().copied().zip(w.iter().copied()).collect();
            (
                app.db
                    .search_sparse(&q, r.k, filter.as_ref())
                    .map_err(|e| e.to_string())?,
                Vec::new(),
                "sparse",
            )
        }
        (None, None, None) => {
            return Err("give an id, text to embed, keyword text or a sparse query".into());
        }
    };
    let micros = t.elapsed().as_micros();
    Ok(json!({
        "mode": mode,
        "micros": micros,
        "hits": hits_json(app, &hits, true),
        "walk": hits_json(app, &walk, false),
        "query_xyz": vector.as_ref().and_then(|v| app.map.lock().unwrap().as_ref().map(|s| project(s, v))),
    }))
}

#[derive(serde::Deserialize)]
struct AddReq {
    id: Option<u64>,
    text: Option<String>,
    vector: Option<Vec<f32>>,
    #[serde(default)]
    meta: J,
}

/// Add one record from text (embedded with the attached model) or from a
/// vector. The text is kept in the metadata under the store's text key.
fn add(app: &App, body: &str) -> Result<J, String> {
    if app.read_only {
        return Err("store opened read-only".into());
    }
    let r: AddReq = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let mut meta = if r.meta.is_null() {
        sanadb::Metadata::new()
    } else {
        json_meta(&r.meta)?
    };
    let vector = match (&r.vector, &r.text) {
        (Some(v), _) => v.clone(),
        (None, Some(t)) if !t.trim().is_empty() => {
            let e = app.embed.as_ref().ok_or(
                "no embedding model attached: start studio with --embed-url or --embed-cmd",
            )?;
            meta.entry(app.text_key.clone())
                .or_insert_with(|| Value::Str(t.clone()));
            note_model(app, e);
            e.embed(std::slice::from_ref(t))?.remove(0)
        }
        _ => return Err("give text to embed or a vector".into()),
    };
    if vector.len() != app.dim {
        return Err(format!(
            "{} dimensions given, the store has {}",
            vector.len(),
            app.dim
        ));
    }
    let id = match r.id {
        Some(id) => id,
        None => app.db.ids().last().map_or(0, |m| m + 1),
    };
    if r.id.is_some() && app.db.get(id).is_some() {
        return Err(format!(
            "record {id} exists; delete it or leave the id empty for a new one"
        ));
    }
    app.db
        .insert(id, &vector, meta)
        .map_err(|e| e.to_string())?;
    *app.map.lock().unwrap() = None;
    let (v, m) = app.db.get(id).ok_or("inserted record not readable")?;
    Ok(json!({"id": id, "vector_len": v.len(), "meta": meta_json(&m)}))
}

/// Record the model's name in the store the first time it writes a vector.
fn note_model(app: &App, e: &Embedder) {
    if app.db.property("embedding_model").is_none() {
        let _ = app.db.set_property("embedding_model", e.name());
        let _ = app.db.set_property("embedding_dim", app.dim.to_string());
    }
}

fn project(s: &MapState, v: &[f32]) -> [f32; 3] {
    let mut out = [0f32; 3];
    for (i, &c) in v.iter().enumerate() {
        let d = c - s.mean[i];
        for (o, axis) in out.iter_mut().zip(&s.axes) {
            *o += d * axis[i];
        }
    }
    out.map(|c| c / s.scale)
}

/// A sample of up to `n` records: PCA to three dimensions (power iteration
/// on the sample covariance; the page draws two or all three) and k-means
/// colours on the first two. Cached until the next call with a different `n`.
fn map(app: &App, q: &HashMap<String, String>) -> Result<J, String> {
    let n: usize = q
        .get("n")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000)
        .clamp(50, 20000);
    let clusters: usize = q
        .get("clusters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
        .clamp(1, 32);
    if let Some(s) = app.map.lock().unwrap().as_ref()
        && s.points["n"] == n
        && s.points["clusters"] == clusters
        && q.get("refresh").is_none()
    {
        return Ok(s.points.clone());
    }
    let all = app.db.ids();
    if all.is_empty() {
        return Err("empty store".into());
    }
    // A deterministic but non-periodic sample: a regular stride aliases
    // with any periodic structure in the ids (synthetic labels, sharded
    // imports), so jitter each pick within its stride.
    let step = (all.len() / n).max(1);
    let ids: Vec<u64> = (0..n.min(all.len()))
        .map(|i| {
            let jitter = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
            all[(i * step + (jitter as usize % step)).min(all.len() - 1)]
        })
        .collect();
    let dim = app.dim;
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
    let mut kept = Vec::with_capacity(ids.len());
    let mut metas = Vec::with_capacity(ids.len());
    for &id in &ids {
        if let Some((v, m)) = app.db.get(id) {
            vecs.push(v);
            kept.push(id);
            metas.push(m);
        }
    }
    let m = vecs.len() as f32;
    let mut mean = vec![0f32; dim];
    for v in &vecs {
        for (a, b) in mean.iter_mut().zip(v) {
            *a += b;
        }
    }
    for a in mean.iter_mut() {
        *a /= m;
    }
    let centred: Vec<Vec<f32>> = vecs
        .iter()
        .map(|v| v.iter().zip(&mean).map(|(a, b)| a - b).collect())
        .collect();
    let a0 = pca_axis(&centred, &[]);
    let a1 = pca_axis(&centred, std::slice::from_ref(&a0));
    let a2 = pca_axis(&centred, &[a0.clone(), a1.clone()]);
    let axes = [a0, a1, a2];
    let coords: Vec<[f32; 3]> = centred
        .iter()
        .map(|v| [dot(v, &axes[0]), dot(v, &axes[1]), dot(v, &axes[2])])
        .collect();
    // The share of the sample's variance each axis carries: how much of
    // the data the picture shows.
    let total_var: f32 = centred.iter().map(|v| dot(v, v)).sum::<f32>() / m;
    let variance: Vec<f32> = (0..3)
        .map(|k| coords.iter().map(|c| c[k] * c[k]).sum::<f32>() / m / total_var.max(1e-12))
        .collect();
    // The scale comes from the first two axes so the 2-D view is unchanged
    // by the third; z is on the same scale.
    let scale = coords
        .iter()
        .map(|c| c[0].abs().max(c[1].abs()))
        .fold(1e-6f32, f32::max);
    let flat: Vec<(f32, f32)> = coords.iter().map(|c| (c[0], c[1])).collect();
    let labels = kmeans(&flat, clusters);
    let points: Vec<J> = kept
        .iter()
        .zip(&coords)
        .zip(&labels)
        .map(|((id, c), k)| json!([id, c[0] / scale, c[1] / scale, c[2] / scale, k]))
        .collect();
    let out = json!({"n": n, "clusters": clusters, "sampled": kept.len(), "total": all.len(), "variance": variance, "points": points});
    let sampled = metas
        .into_iter()
        .zip(&coords)
        .zip(&labels)
        .map(|((m, c), k)| (m, *k, c.map(|x| x / scale)))
        .collect();
    *app.map.lock().unwrap() = Some(MapState {
        mean,
        axes,
        scale,
        points: out.clone(),
        sampled,
    });
    Ok(out)
}

/// For each cluster of the current map, the most common value of `key`
/// among its sampled records, with its share, at the cluster's centroid.
fn labels(app: &App, q: &HashMap<String, String>) -> Result<J, String> {
    let key = q.get("key").ok_or("key required")?;
    let st = app.map.lock().unwrap();
    let st = st.as_ref().ok_or("build the map first")?;
    let clusters = st.points["clusters"].as_u64().unwrap_or(0) as usize;
    let mut out = Vec::new();
    for c in 0..clusters {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let (mut n, mut sx, mut sy, mut sz) = (0usize, 0f32, 0f32, 0f32);
        for (m, mc, [x, y, z]) in &st.sampled {
            if *mc != c {
                continue;
            }
            n += 1;
            sx += x;
            sy += y;
            sz += z;
            if let Some(v) = m.get(key.as_str()) {
                let text = match v {
                    Value::Sparse(p) => format!("sparse[{}]", p.len()),
                    other => other.to_string(),
                };
                *counts.entry(text).or_insert(0) += 1;
            }
        }
        if n == 0 {
            continue;
        }
        let best = counts.into_iter().max_by_key(|(_, k)| *k);
        out.push(json!({
            "cluster": c, "size": n, "x": sx / n as f32, "y": sy / n as f32, "z": sz / n as f32,
            "value": best.as_ref().map(|(v, _)| v.clone()),
            "share": best.as_ref().map_or(0.0, |(_, k)| *k as f64 / n as f64),
        }));
    }
    Ok(json!({"key": key, "labels": out}))
}

/// The store's files with sizes and the facts a user wants to see about
/// them: the log's reader level, the index's name, level and snapshot id.
fn files(app: &App) -> J {
    let size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let byte13 = |p: &std::path::Path| -> Option<u8> {
        let mut f = std::fs::File::open(p).ok()?;
        let mut b = [0u8; 16];
        std::io::Read::read_exact(&mut f, &mut b).ok()?;
        Some(b[13])
    };
    let log = std::path::PathBuf::from(&app.path);
    let mut lock = log.as_os_str().to_owned();
    lock.push(".lock");
    let lock = std::path::PathBuf::from(lock);
    let mut out = vec![json!({
        "role": "log", "path": log.display().to_string(), "bytes": size(&log),
        "level": byte13(&log),
        "note": "write-ahead log: header, then CRC-framed insert, delete, snapshot, begin and commit frames; replayed on open",
    })];
    if let Some(idx) = app.db.index_path() {
        let name = idx
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let snapshot = name
            .rsplit(".idx.")
            .next()
            .filter(|s| s.len() == 16)
            .map(|s| s.to_string());
        out.push(json!({
            "role": "index", "path": idx.display().to_string(), "bytes": size(&idx),
            "level": byte13(&idx), "snapshot": snapshot,
            "note": "graph index named by the log's snapshot frame: record blocks, PQ codes, attribute columns, id table, text and sparse sections, footer; memory-mapped",
        }));
    }
    if lock.exists() {
        out.push(json!({"role": "lock", "path": lock.display().to_string(), "bytes": size(&lock), "note": "empty; the writer holds an exclusive flock on it"}));
    }
    json!({"files": out})
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Leading eigenvector of the sample covariance by power iteration,
/// orthogonal to every axis in `against`.
fn pca_axis(rows: &[Vec<f32>], against: &[Vec<f32>]) -> Vec<f32> {
    let dim = rows[0].len();
    let mut v: Vec<f32> = (0..dim)
        .map(|i| ((i * 7919) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let deflate = |v: &mut Vec<f32>| {
        for o in against {
            let p = dot(v, o);
            for (a, b) in v.iter_mut().zip(o) {
                *a -= p * b;
            }
        }
    };
    for _ in 0..30 {
        deflate(&mut v);
        let mut next = vec![0f32; dim];
        for r in rows {
            let p = dot(r, &v);
            for (a, b) in next.iter_mut().zip(r) {
                *a += p * b;
            }
        }
        let norm = dot(&next, &next).sqrt().max(1e-12);
        for a in next.iter_mut() {
            *a /= norm;
        }
        v = next;
    }
    if !against.is_empty() {
        deflate(&mut v);
        let norm = dot(&v, &v).sqrt().max(1e-12);
        for a in v.iter_mut() {
            *a /= norm;
        }
    }
    v
}

fn kmeans(pts: &[(f32, f32)], k: usize) -> Vec<usize> {
    let k = k.min(pts.len()).max(1);
    let mut cents: Vec<(f32, f32)> = (0..k).map(|i| pts[i * pts.len() / k]).collect();
    let mut labels = vec![0usize; pts.len()];
    for _ in 0..25 {
        for (l, p) in labels.iter_mut().zip(pts) {
            *l = (0..k)
                .min_by(|&a, &b| {
                    let da = (p.0 - cents[a].0).powi(2) + (p.1 - cents[a].1).powi(2);
                    let db = (p.0 - cents[b].0).powi(2) + (p.1 - cents[b].1).powi(2);
                    da.total_cmp(&db)
                })
                .unwrap();
        }
        let mut sum = vec![(0f32, 0f32, 0usize); k];
        for (l, p) in labels.iter().zip(pts) {
            sum[*l].0 += p.0;
            sum[*l].1 += p.1;
            sum[*l].2 += 1;
        }
        for (c, s) in cents.iter_mut().zip(&sum) {
            if s.2 > 0 {
                *c = (s.0 / s.2 as f32, s.1 / s.2 as f32);
            }
        }
    }
    labels
}

fn checkpoint(app: &App) -> Result<J, String> {
    if app.read_only {
        return Err("store opened read-only".into());
    }
    let t = std::time::Instant::now();
    app.db.checkpoint().map_err(|e| e.to_string())?;
    *app.map.lock().unwrap() = None;
    Ok(json!({"ms": t.elapsed().as_millis() as u64}))
}
