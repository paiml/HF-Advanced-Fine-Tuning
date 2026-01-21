//! Rust CLI Documentation Corpus
//!
//! A scientifically rigorous corpus for fine-tuning LLMs to generate
//! idiomatic `///` documentation comments for Rust CLI tools.
//!
//! # Design Philosophy
//!
//! This corpus follows the Toyota Way principles:
//! - **Genchi Genbutsu**: Data from real production CLI tools
//! - **Jidoka**: Built-in quality gates
//! - **Muda**: No waste, no redundancy
//!
//! # Example
//!
//! ```no_run
//! use rust_cli_docs_corpus::{Corpus, CorpusValidator};
//!
//! let corpus = Corpus::load_from_parquet("data/train.parquet")?;
//! let validator = CorpusValidator::new();
//! let result = validator.validate_corpus(&corpus);
//!
//! println!("Score: {}/100", result.score());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod entry;
pub mod error;
pub mod extractor;
pub mod filter;
pub mod publisher;
pub mod schema;
pub mod validator;

pub use extractor::{DocExtractor, ExtractorConfig, RepoSpec};
pub use filter::CorpusFilter;
pub use publisher::{HfPublisher, PublishResult, PublisherConfig};

pub use entry::{Category, CorpusEntry};
pub use error::{CorpusError, Result};
pub use validator::{CorpusValidator, ValidationResult, ValidationScore};

use arrow::array::{Array, Float32Array, Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Corpus metadata with provenance information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CorpusMetadata {
    /// Semantic version of the corpus
    pub version: String,
    /// SHA-256 hash of the corpus content
    pub output_hash: Option<String>,
    /// Extraction timestamp
    pub extraction_date: chrono::DateTime<chrono::Utc>,
    /// Tool versions used for extraction
    pub tool_versions: HashMap<String, String>,
    /// Source repository commits
    pub source_commits: HashMap<String, String>,
}

impl Default for CorpusMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            output_hash: None,
            extraction_date: chrono::Utc::now(),
            tool_versions: HashMap::new(),
            source_commits: HashMap::new(),
        }
    }
}

/// The main corpus container.
#[derive(Debug)]
pub struct Corpus {
    entries: Vec<CorpusEntry>,
    metadata: CorpusMetadata,
}

impl Corpus {
    /// Creates a new empty corpus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            metadata: CorpusMetadata::default(),
        }
    }

    /// Creates a corpus from a vector of entries.
    #[must_use]
    pub fn from_entries(entries: Vec<CorpusEntry>) -> Self {
        Self {
            entries,
            metadata: CorpusMetadata::default(),
        }
    }

    /// Loads a corpus from a Parquet file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load_from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut entries = Vec::new();

        for batch_result in reader {
            let batch = batch_result?;
            let batch_entries = Self::batch_to_entries(&batch)?;
            entries.extend(batch_entries);
        }

        let mut corpus = Self {
            entries,
            metadata: CorpusMetadata::default(),
        };

        // Compute and set the hash in metadata
        let hash = corpus.compute_hash();
        corpus.metadata.output_hash = Some(hash);

        Ok(corpus)
    }

    /// Saves the corpus to a Parquet file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_to_parquet<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let batch = self.to_record_batch()?;
        let file = std::fs::File::create(path.as_ref())?;

        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(Default::default()))
            .build();

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Returns an iterator over corpus entries.
    pub fn entries(&self) -> impl Iterator<Item = &CorpusEntry> {
        self.entries.iter()
    }

    /// Returns the number of entries in the corpus.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the corpus is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the corpus schema.
    #[must_use]
    pub fn schema(&self) -> CorpusSchema {
        CorpusSchema::default()
    }

    /// Returns the corpus metadata.
    #[must_use]
    pub fn metadata(&self) -> &CorpusMetadata {
        &self.metadata
    }

    /// Sets the corpus metadata.
    pub fn set_metadata(&mut self, metadata: CorpusMetadata) {
        self.metadata = metadata;
    }

    /// Adds an entry to the corpus.
    pub fn add_entry(&mut self, entry: CorpusEntry) {
        self.entries.push(entry);
    }

    /// Computes the SHA-256 hash of the corpus content.
    #[must_use]
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        for entry in &self.entries {
            hasher.update(entry.id.as_bytes());
            hasher.update(entry.input.as_bytes());
            hasher.update(entry.output.as_bytes());
        }
        hex::encode(hasher.finalize())
    }

    /// Converts the corpus to an Arrow RecordBatch.
    fn to_record_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(schema::arrow_schema());

        let ids: Vec<&str> = self.entries.iter().map(|e| e.id.as_str()).collect();
        let inputs: Vec<&str> = self.entries.iter().map(|e| e.input.as_str()).collect();
        let outputs: Vec<&str> = self.entries.iter().map(|e| e.output.as_str()).collect();
        let categories: Vec<&str> = self.entries.iter().map(|e| e.category.as_str()).collect();
        let repos: Vec<&str> = self.entries.iter().map(|e| e.source_repo.as_str()).collect();
        let commits: Vec<&str> = self.entries.iter().map(|e| e.source_commit.as_str()).collect();
        let files: Vec<&str> = self.entries.iter().map(|e| e.source_file.as_str()).collect();
        let lines: Vec<i32> = self.entries.iter().map(|e| e.source_line as i32).collect();
        let tokens_in: Vec<i32> = self.entries.iter().map(|e| e.tokens_input as i32).collect();
        let tokens_out: Vec<i32> = self.entries.iter().map(|e| e.tokens_output as i32).collect();
        let scores: Vec<f32> = self.entries.iter().map(|e| e.quality_score).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(inputs)),
                Arc::new(StringArray::from(outputs)),
                Arc::new(StringArray::from(categories)),
                Arc::new(StringArray::from(repos)),
                Arc::new(StringArray::from(commits)),
                Arc::new(StringArray::from(files)),
                Arc::new(Int32Array::from(lines)),
                Arc::new(Int32Array::from(tokens_in)),
                Arc::new(Int32Array::from(tokens_out)),
                Arc::new(Float32Array::from(scores)),
            ],
        )?;

        Ok(batch)
    }

    /// Converts a RecordBatch to corpus entries.
    fn batch_to_entries(batch: &RecordBatch) -> Result<Vec<CorpusEntry>> {
        let ids = batch
            .column_by_name("id")
            .ok_or_else(|| CorpusError::Schema("missing 'id' column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CorpusError::Schema("'id' not a string".into()))?;

        let inputs = batch
            .column_by_name("input")
            .ok_or_else(|| CorpusError::Schema("missing 'input' column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CorpusError::Schema("'input' not a string".into()))?;

        let outputs = batch
            .column_by_name("output")
            .ok_or_else(|| CorpusError::Schema("missing 'output' column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CorpusError::Schema("'output' not a string".into()))?;

        let categories = batch
            .column_by_name("category")
            .ok_or_else(|| CorpusError::Schema("missing 'category' column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CorpusError::Schema("'category' not a string".into()))?;

        let repos = batch
            .column_by_name("source_repo")
            .ok_or_else(|| CorpusError::Schema("missing 'source_repo' column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CorpusError::Schema("'source_repo' not a string".into()))?;

        let commits = batch
            .column_by_name("source_commit")
            .ok_or_else(|| CorpusError::Schema("missing 'source_commit' column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CorpusError::Schema("'source_commit' not a string".into()))?;

        let files = batch
            .column_by_name("source_file")
            .ok_or_else(|| CorpusError::Schema("missing 'source_file' column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CorpusError::Schema("'source_file' not a string".into()))?;

        let lines = batch
            .column_by_name("source_line")
            .ok_or_else(|| CorpusError::Schema("missing 'source_line' column".into()))?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| CorpusError::Schema("'source_line' not an int".into()))?;

        let tokens_in = batch
            .column_by_name("tokens_input")
            .ok_or_else(|| CorpusError::Schema("missing 'tokens_input' column".into()))?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| CorpusError::Schema("'tokens_input' not an int".into()))?;

        let tokens_out = batch
            .column_by_name("tokens_output")
            .ok_or_else(|| CorpusError::Schema("missing 'tokens_output' column".into()))?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| CorpusError::Schema("'tokens_output' not an int".into()))?;

        let scores = batch
            .column_by_name("quality_score")
            .ok_or_else(|| CorpusError::Schema("missing 'quality_score' column".into()))?
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| CorpusError::Schema("'quality_score' not a float".into()))?;

        let mut entries = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            let entry = CorpusEntry {
                id: ids.value(i).to_string(),
                input: inputs.value(i).to_string(),
                output: outputs.value(i).to_string(),
                category: Category::from_str(categories.value(i)),
                source_repo: repos.value(i).to_string(),
                source_commit: commits.value(i).to_string(),
                source_file: files.value(i).to_string(),
                source_line: lines.value(i) as u32,
                tokens_input: tokens_in.value(i) as u32,
                tokens_output: tokens_out.value(i) as u32,
                quality_score: scores.value(i),
                extraction_date: chrono::Utc::now(),
            };
            entries.push(entry);
        }

        Ok(entries)
    }
}

impl Default for Corpus {
    fn default() -> Self {
        Self::new()
    }
}

/// Schema wrapper for validation.
#[derive(Debug)]
pub struct CorpusSchema {
    fields: Vec<String>,
}

impl CorpusSchema {
    /// Checks if the schema contains a field.
    #[must_use]
    pub fn contains_field(&self, name: &str) -> bool {
        self.fields.contains(&name.to_string())
    }
}

impl Default for CorpusSchema {
    fn default() -> Self {
        Self {
            fields: vec![
                "id".to_string(),
                "input".to_string(),
                "output".to_string(),
                "category".to_string(),
                "source_repo".to_string(),
                "source_commit".to_string(),
                "source_file".to_string(),
                "source_line".to_string(),
                "tokens_input".to_string(),
                "tokens_output".to_string(),
                "quality_score".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_new() {
        let corpus = Corpus::new();
        assert!(corpus.is_empty());
        assert_eq!(corpus.len(), 0);
    }

    #[test]
    fn test_corpus_add_entry() {
        let mut corpus = Corpus::new();
        let entry = CorpusEntry::new(
            "pub fn test() {}".to_string(),
            "/// Test function.".to_string(),
            Category::Function,
        );
        corpus.add_entry(entry);
        assert_eq!(corpus.len(), 1);
    }

    #[test]
    fn test_schema_contains_fields() {
        let schema = CorpusSchema::default();
        assert!(schema.contains_field("id"));
        assert!(schema.contains_field("input"));
        assert!(schema.contains_field("output"));
        assert!(!schema.contains_field("nonexistent"));
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let mut corpus = Corpus::new();
        let entry = CorpusEntry::new(
            "fn test() {}".to_string(),
            "/// Test.".to_string(),
            Category::Function,
        );
        corpus.add_entry(entry);

        let hash1 = corpus.compute_hash();
        let hash2 = corpus.compute_hash();
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 hex
    }
}
