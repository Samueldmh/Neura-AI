use crate::config::{COLLECTION_NAME, SEARCH_STOP_WORDS};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::{
    Condition, Filter, Match, MatchText, MatchValue, QueryPointsBuilder, ScoredPoint,
};
use regex::Regex;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};

#[derive(Clone)]
pub struct RagEngine {
    pub qdrant: Arc<QdrantClient>,
    pub embedder: Arc<Mutex<TextEmbedding>>,
}

impl RagEngine {
    pub fn new(qdrant: Arc<QdrantClient>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
        )?;
        Ok(Self {
            qdrant,
            embedder: Arc::new(Mutex::new(model)),
        })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let embedder = self.embedder.lock().await;
        let embeddings = embedder.embed(vec![text.to_string()], None)?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| "Empty embeddings returned".into())
    }

    pub async fn search_qdrant(
        &self,
        query_text: &str,
        limit: u64,
        preferred_books: &[String],
    ) -> Vec<ScoredPoint> {
        let query_vector = match self.embed(query_text).await {
            Ok(v) => v,
            Err(e) => {
                error!("Embedding generation error: {}", e);
                return Vec::new();
            }
        };

        if preferred_books.is_empty() {
            let query = QueryPointsBuilder::new(COLLECTION_NAME)
                .query(query_vector)
                .limit(limit)
                .with_payload(true);

            return match self.qdrant.query(query).await {
                Ok(res) => res.result,
                Err(e) => {
                    error!("Qdrant global search error: {}", e);
                    Vec::new()
                }
            };
        }

        let mut all_points = Vec::new();

        for book in preferred_books {
            if book.is_empty() || book.starts_with("Skip") {
                continue;
            }

            // Attempt 1: Exact Match
            let filter = Filter::all(vec![Condition::matches(
                "book_title",
                MatchValue {
                    value: Some(qdrant_client::qdrant::match_value::Value::Keyword(
                        book.clone(),
                    )),
                },
            )]);

            let query = QueryPointsBuilder::new(COLLECTION_NAME)
                .query(query_vector.clone())
                .filter(filter)
                .limit(limit)
                .with_payload(true);

            let mut hits = match self.qdrant.query(query).await {
                Ok(res) => res.result,
                Err(_) => Vec::new(),
            };

            // Attempt 2: Fuzzy keyword fallback
            if hits.is_empty() {
                let b_lower = book.to_lowercase();
                let book_kw = if b_lower.contains("lippincott") {
                    "lippincott"
                } else if b_lower.contains("robbins") {
                    "robbins"
                } else if b_lower.contains("haematology") || b_lower.contains("hoffbrand") {
                    "haematology"
                } else if b_lower.contains("microbiology") || b_lower.contains("jawetz") {
                    "microbiology"
                } else if b_lower.contains("sembulingam") {
                    "sembulingam"
                } else if b_lower.contains("moore") || b_lower.contains("anatomy") {
                    "moore"
                } else {
                    ""
                };

                if !book_kw.is_empty() {
                    let fuzzy_filter = Filter::all(vec![Condition::matches(
                        "book_title",
                        MatchText {
                            text: book_kw.to_string(),
                        },
                    )]);

                    let fuzzy_query = QueryPointsBuilder::new(COLLECTION_NAME)
                        .query(query_vector.clone())
                        .filter(fuzzy_filter)
                        .limit(limit)
                        .with_payload(true);

                    if let Ok(res) = self.qdrant.query(fuzzy_query).await {
                        hits = res.result;
                    }
                }
            }

            all_points.extend(hits);
        }

        all_points.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_points
    }

    pub async fn multi_search_qdrant(
        &self,
        search_terms: &[String],
        preferred_books: &[String],
    ) -> Vec<ScoredPoint> {
        let mut seen_texts = HashSet::new();
        let mut all_results = Vec::new();

        let mut tasks = Vec::new();
        for term in search_terms {
            let engine = self.clone();
            let t = term.clone();
            let books = preferred_books.to_vec();
            tasks.push(tokio::spawn(async move {
                engine.search_qdrant(&t, 4, &books).await
            }));
        }

        for task in tasks {
            if let Ok(points) = task.await {
                for point in points {
                    let text = point
                        .payload
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let key: String = text.chars().take(100).collect();
                    if !seen_texts.contains(&key) {
                        seen_texts.insert(key);
                        all_results.push(point);
                    }
                }
            }
        }

        all_results.truncate(10);
        info!(
            "📚 Multi-search returned {} unique chunks from {} keyword(s)",
            all_results.len(),
            search_terms.len()
        );
        all_results
    }
}

pub fn extract_medical_terms(user_msg: &str) -> Vec<String> {
    let re_punct = Regex::new(r"[^\w\s]").unwrap();
    let cleaned = re_punct.replace_all(user_msg, " ");
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    let stop_words: HashSet<&str> = SEARCH_STOP_WORDS.iter().cloned().collect();

    let meaningful: Vec<&str> = words
        .into_iter()
        .filter(|w| !stop_words.contains(w.to_lowercase().as_str()) && w.len() > 2)
        .collect();

    if meaningful.is_empty() {
        return vec![user_msg.to_string()];
    }

    let mut phrases = Vec::new();
    let mut current_phrase = Vec::new();

    for w in meaningful {
        let lower = w.to_lowercase();
        if lower == "and" || lower == "or" || lower == "vs" || lower == "versus" {
            if !current_phrase.is_empty() {
                phrases.push(current_phrase.join(" "));
                current_phrase.clear();
            }
        } else {
            current_phrase.push(w);
        }
    }

    if !current_phrase.is_empty() {
        phrases.push(current_phrase.join(" "));
    }

    if !phrases.contains(&user_msg.to_string()) {
        phrases.push(user_msg.to_string());
    }

    phrases
}

pub fn get_explicit_book_override(user_msg: &str, preferred_books: &[String]) -> Vec<String> {
    let msg_lower = user_msg.to_lowercase();
    let mut override_books = Vec::new();

    for b in preferred_books {
        if b.is_empty() || b.starts_with("Skip") {
            continue;
        }
        let b_lower = b.to_lowercase();
        if msg_lower.contains("pharmacology") && b_lower.contains("pharmacology") {
            override_books.push(b.clone());
        } else if msg_lower.contains("pathology")
            && (b_lower.contains("pathology") || b_lower.contains("robbins"))
        {
            override_books.push(b.clone());
        } else if msg_lower.contains("anatomy") && b_lower.contains("anatomy") {
            override_books.push(b.clone());
        } else if msg_lower.contains("physiology")
            && (b_lower.contains("physiology") || b_lower.contains("sembulingam"))
        {
            override_books.push(b.clone());
        } else if msg_lower.contains("biochemistry") && b_lower.contains("biochemistry") {
            override_books.push(b.clone());
        } else if msg_lower.contains("haematology")
            && (b_lower.contains("haematology") || b_lower.contains("hoffbrand"))
        {
            override_books.push(b.clone());
        } else if msg_lower.contains("microbiology")
            && (b_lower.contains("microbiology") || b_lower.contains("jawetz"))
        {
            override_books.push(b.clone());
        } else if msg_lower.contains("lippincott") && b_lower.contains("lippincott") {
            override_books.push(b.clone());
        } else if msg_lower.contains("robbins") && b_lower.contains("robbins") {
            override_books.push(b.clone());
        } else if msg_lower.contains("sembulingam") && b_lower.contains("sembulingam") {
            override_books.push(b.clone());
        }
    }

    if override_books.is_empty() {
        preferred_books.to_vec()
    } else {
        override_books
    }
}
