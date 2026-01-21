//! Error types for corpus operations.

use thiserror::Error;

/// Result type alias for corpus operations.
pub type Result<T> = std::result::Result<T, CorpusError>;

/// Errors that can occur during corpus operations.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Parquet error
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// Arrow error
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// Schema validation error
    #[error("Schema error: {0}")]
    Schema(String),

    /// Validation error
    #[error("Validation failed: {0}")]
    Validation(String),

    /// Git operation error
    #[error("Git error: {0}")]
    Git(String),

    /// Extraction error
    #[error("Extraction error: {0}")]
    Extraction(String),

    /// Publication error
    #[error("Publication error: {0}")]
    Publication(String),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}

impl CorpusError {
    /// Creates a new validation error.
    #[must_use]
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Creates a new schema error.
    #[must_use]
    pub fn schema(msg: impl Into<String>) -> Self {
        Self::Schema(msg.into())
    }

    /// Creates a new git error.
    #[must_use]
    pub fn git(msg: impl Into<String>) -> Self {
        Self::Git(msg.into())
    }

    /// Creates a new extraction error.
    #[must_use]
    pub fn extraction(msg: impl Into<String>) -> Self {
        Self::Extraction(msg.into())
    }

    /// Creates a new publication error.
    #[must_use]
    pub fn publication(msg: impl Into<String>) -> Self {
        Self::Publication(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CorpusError::validation("test error");
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_error_variants() {
        let _ = CorpusError::schema("schema issue");
        let _ = CorpusError::git("git issue");
        let _ = CorpusError::extraction("extraction issue");
        let _ = CorpusError::publication("publication issue");
    }
}
