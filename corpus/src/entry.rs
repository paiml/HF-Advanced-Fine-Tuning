//! Corpus entry types.
//!
//! Uses deterministic UUID v5 for reproducibility (per Spec Section 3.1).

use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Custom namespace UUID for corpus entries (deterministic).
/// Generated once: `uuid -v4` -> fixed for reproducibility.
const CORPUS_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1,
    0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
]);

/// Fixed extraction date for reproducibility (2026-01-21T00:00:00Z).
fn deterministic_extraction_date() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339("2026-01-21T00:00:00Z")
        .expect("valid date")
        .with_timezone(&chrono::Utc)
}

/// Documentation category classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    /// Function-level documentation (`/// Description`)
    Function,
    /// Argument documentation (`/// # Arguments`)
    Argument,
    /// Example documentation (`/// # Examples`)
    Example,
    /// Error documentation (`/// # Errors`, `/// # Panics`)
    Error,
    /// Module-level documentation (`//!`)
    Module,
}

impl Category {
    /// Returns the string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Argument => "argument",
            Self::Example => "example",
            Self::Error => "error",
            Self::Module => "module",
        }
    }

    /// Parses a category from a string.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "function" => Self::Function,
            "argument" => Self::Argument,
            "example" => Self::Example,
            "error" => Self::Error,
            "module" => Self::Module,
            _ => Self::Function,
        }
    }
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single corpus entry representing an input/output pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CorpusEntry {
    /// UUID v4 identifier
    pub id: String,
    /// Function/struct signature (the prompt)
    pub input: String,
    /// Documentation comment (the completion)
    pub output: String,
    /// Documentation category
    pub category: Category,
    /// Source repository (e.g., "clap-rs/clap")
    pub source_repo: String,
    /// Git commit SHA (7 chars)
    pub source_commit: String,
    /// Relative file path
    pub source_file: String,
    /// Line number in source file
    pub source_line: u32,
    /// Token count for input
    pub tokens_input: u32,
    /// Token count for output
    pub tokens_output: u32,
    /// Quality score [0.0, 1.0]
    pub quality_score: f32,
    /// Extraction timestamp
    #[serde(with = "chrono::serde::ts_seconds")]
    pub extraction_date: chrono::DateTime<chrono::Utc>,
}

impl CorpusEntry {
    /// Creates a new corpus entry with deterministic UUID v5.
    ///
    /// The UUID is derived from the content hash (input + output) using UUID v5,
    /// ensuring identical content always produces identical IDs (reproducibility).
    #[must_use]
    pub fn new(input: String, output: String, category: Category) -> Self {
        // Deterministic UUID v5: same content -> same UUID
        let content_for_hash = format!("{}\0{}", input, output);
        let id = Uuid::new_v5(&CORPUS_NAMESPACE, content_for_hash.as_bytes()).to_string();

        let tokens_input = Self::estimate_tokens(&input);
        let tokens_output = Self::estimate_tokens(&output);

        Self {
            id,
            input,
            output,
            category,
            source_repo: String::new(),
            source_commit: String::new(),
            source_file: String::new(),
            source_line: 0,
            tokens_input,
            tokens_output,
            quality_score: 0.0,
            extraction_date: deterministic_extraction_date(),
        }
    }

    /// Creates an entry with full provenance.
    #[must_use]
    pub fn with_provenance(
        input: String,
        output: String,
        category: Category,
        repo: String,
        commit: String,
        file: String,
        line: u32,
    ) -> Self {
        let mut entry = Self::new(input, output, category);
        entry.source_repo = repo;
        entry.source_commit = commit;
        entry.source_file = file;
        entry.source_line = line;
        entry
    }

    /// Computes a content hash for deduplication.
    #[must_use]
    pub fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.input.as_bytes());
        hasher.update(self.output.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Estimates token count (simple heuristic: ~4 chars per token).
    fn estimate_tokens(text: &str) -> u32 {
        (text.len() as f32 / 4.0).ceil() as u32
    }

    /// Returns the input/output ratio.
    #[must_use]
    pub fn io_ratio(&self) -> f64 {
        if self.tokens_input == 0 {
            0.0
        } else {
            f64::from(self.tokens_output) / f64::from(self.tokens_input)
        }
    }

    /// Returns total token count.
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        self.tokens_input + self.tokens_output
    }
}

impl Default for CorpusEntry {
    fn default() -> Self {
        Self::new(String::new(), String::new(), Category::Function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_roundtrip() {
        for cat in [
            Category::Function,
            Category::Argument,
            Category::Example,
            Category::Error,
            Category::Module,
        ] {
            let s = cat.as_str();
            let parsed = Category::from_str(s);
            assert_eq!(cat, parsed);
        }
    }

    #[test]
    fn test_entry_new_generates_uuid() {
        let entry = CorpusEntry::new(
            "fn test() {}".to_string(),
            "/// Test.".to_string(),
            Category::Function,
        );
        assert!(!entry.id.is_empty());
        assert!(entry.id.contains('-')); // UUID format
    }

    #[test]
    fn test_content_hash_deterministic() {
        let entry1 = CorpusEntry::new(
            "fn test() {}".to_string(),
            "/// Test.".to_string(),
            Category::Function,
        );
        let entry2 = CorpusEntry::new(
            "fn test() {}".to_string(),
            "/// Test.".to_string(),
            Category::Function,
        );

        assert_eq!(entry1.content_hash(), entry2.content_hash());
    }

    #[test]
    fn test_content_hash_different() {
        let entry1 = CorpusEntry::new(
            "fn test1() {}".to_string(),
            "/// Test 1.".to_string(),
            Category::Function,
        );
        let entry2 = CorpusEntry::new(
            "fn test2() {}".to_string(),
            "/// Test 2.".to_string(),
            Category::Function,
        );

        assert_ne!(entry1.content_hash(), entry2.content_hash());
    }

    #[test]
    fn test_io_ratio() {
        let mut entry = CorpusEntry::default();
        entry.tokens_input = 10;
        entry.tokens_output = 50;

        assert!((entry.io_ratio() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_io_ratio_zero_input() {
        let entry = CorpusEntry::default();
        assert_eq!(entry.io_ratio(), 0.0);
    }
}
