//! Text in, vectors out: the hook that lets a store be used with text when
//! the user has an embedding model. The core never ships a model; it
//! ships the trait, two `Db` helpers that use it, and, behind the
//! `embed-http` feature, a client for any OpenAI-compatible
//! `/v1/embeddings` endpoint (OpenAI, Ollama, LM Studio, vLLM,
//! text-embeddings-inference).

use crate::{Db, Error, Filter, Hit, Metadata, Result, Value};

/// Turns texts into vectors of the store's dimension. Implement it over
/// whatever model you run; batch calls are one request where the backend
/// allows it.
pub trait Embedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

impl<F> Embedder for F
where
    F: Fn(&[&str]) -> Result<Vec<Vec<f32>>>,
{
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self(texts)
    }
}

impl Db {
    /// Embed `text` with `model`, store it under `id` with `meta` plus the
    /// text itself under `text_key` (so `search_text` and `get` see it),
    /// and return the vector written.
    pub fn add_text(
        &self,
        id: u64,
        text: &str,
        mut meta: Metadata,
        text_key: &str,
        model: &impl Embedder,
    ) -> Result<Vec<f32>> {
        let mut v = model.embed(&[text])?;
        let v = v
            .pop()
            .ok_or(Error::Corrupt("embedder returned no vector"))?;
        meta.entry(text_key.to_string())
            .or_insert_with(|| Value::Str(text.to_string()));
        self.insert(id, &v, meta)?;
        Ok(v)
    }

    /// Embed `text` and search with the vector, with an optional filter.
    pub fn search_text_embedded(
        &self,
        text: &str,
        k: usize,
        filter: Option<&Filter>,
        model: &impl Embedder,
    ) -> Result<Vec<Hit>> {
        let mut v = model.embed(&[text])?;
        let v = v
            .pop()
            .ok_or(Error::Corrupt("embedder returned no vector"))?;
        match filter {
            Some(f) => self.search_where(&v, k, f),
            None => self.search(&v, k),
        }
    }
}

/// An OpenAI-compatible `/v1/embeddings` client. Feature `embed-http`.
#[cfg(feature = "embed-http")]
pub struct OpenAiEmbedder {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
}

#[cfg(feature = "embed-http")]
impl OpenAiEmbedder {
    /// `url` is the full endpoint, for instance
    /// `http://localhost:11434/v1/embeddings` for Ollama.
    pub fn new(url: impl Into<String>, model: impl Into<String>) -> Self {
        OpenAiEmbedder {
            url: url.into(),
            model: model.into(),
            api_key: None,
        }
    }
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

#[cfg(feature = "embed-http")]
impl Embedder for OpenAiEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let io = |m: String| Error::Io(std::io::Error::other(m));
        let mut req = ureq::post(&self.url).header("Content-Type", "application/json");
        if let Some(k) = &self.api_key {
            req = req.header("Authorization", &format!("Bearer {k}"));
        }
        let body = serde_json::json!({"model": self.model, "input": texts});
        let mut resp = req
            .send_json(&body)
            .map_err(|e| io(format!("embedding request to {} failed: {e}", self.url)))?;
        let v: serde_json::Value = resp
            .body_mut()
            .read_json()
            .map_err(|e| io(format!("embedding response is not JSON: {e}")))?;
        let data = v["data"]
            .as_array()
            .ok_or_else(|| io("embedding response has no `data` array".into()))?;
        let mut out: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for (i, d) in data.iter().enumerate() {
            let idx = d["index"].as_u64().map_or(i, |x| x as usize);
            let e = d["embedding"]
                .as_array()
                .ok_or_else(|| io("embedding entry has no `embedding` array".into()))?
                .iter()
                .map(|x| x.as_f64().map(|x| x as f32))
                .collect::<Option<Vec<f32>>>()
                .ok_or_else(|| io("non-numeric embedding value".into()))?;
            out.push((idx, e));
        }
        out.sort_by_key(|(i, _)| *i);
        if out.len() != texts.len() {
            return Err(io(format!(
                "asked for {} embeddings, got {}",
                texts.len(),
                out.len()
            )));
        }
        Ok(out.into_iter().map(|(_, e)| e).collect())
    }
}
