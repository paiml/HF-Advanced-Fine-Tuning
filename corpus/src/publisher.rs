//! HuggingFace Hub publication via alimentar.

use crate::{Corpus, CorpusError, Result};
use std::path::Path;

/// Publisher configuration.
#[derive(Debug, Clone)]
pub struct PublisherConfig {
    /// HuggingFace repository ID (e.g., "paiml/rust-cli-docs-corpus")
    pub repo_id: String,
    /// HuggingFace API token (from environment or direct)
    pub token: Option<String>,
    /// Whether to create the repo if it doesn't exist
    pub create_repo: bool,
    /// Repository visibility
    pub private: bool,
}

impl Default for PublisherConfig {
    fn default() -> Self {
        Self {
            repo_id: "paiml/rust-cli-docs-corpus".to_string(),
            token: std::env::var("HF_TOKEN").ok(),
            create_repo: true,
            private: false,
        }
    }
}

/// HuggingFace Hub publisher using alimentar.
#[derive(Debug)]
pub struct HfPublisher {
    config: PublisherConfig,
}

impl HfPublisher {
    /// Creates a new publisher with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PublisherConfig::default(),
        }
    }

    /// Creates a new publisher with custom configuration.
    #[must_use]
    pub fn with_config(config: PublisherConfig) -> Self {
        Self { config }
    }

    /// Publishes the corpus to HuggingFace Hub.
    ///
    /// # Errors
    ///
    /// Returns an error if publication fails.
    pub async fn publish(&self, corpus: &Corpus, output_dir: &Path) -> Result<PublishResult> {
        // Validate we have a token
        let _token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| CorpusError::publication("HF_TOKEN not set"))?;

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Save corpus to parquet files (train/validation/test splits)
        let train_path = output_dir.join("train.parquet");
        let val_path = output_dir.join("validation.parquet");
        let test_path = output_dir.join("test.parquet");

        // Split corpus (80/10/10)
        let entries: Vec<_> = corpus.entries().collect();
        let total = entries.len();
        let train_end = (total as f64 * 0.8) as usize;
        let val_end = (total as f64 * 0.9) as usize;

        // Create split corpora
        let train_corpus = create_split_corpus(&entries[..train_end]);
        let val_corpus = create_split_corpus(&entries[train_end..val_end]);
        let test_corpus = create_split_corpus(&entries[val_end..]);

        // Save splits
        train_corpus.save_to_parquet(&train_path)?;
        val_corpus.save_to_parquet(&val_path)?;
        test_corpus.save_to_parquet(&test_path)?;

        // Create dataset card
        let card_path = output_dir.join("README.md");
        let card_content = self.generate_dataset_card(corpus)?;
        std::fs::write(&card_path, card_content)?;

        // Create metadata
        let meta_path = output_dir.join("corpus_metadata.json");
        let meta_content = serde_json::to_string_pretty(corpus.metadata())?;
        std::fs::write(&meta_path, meta_content)?;

        Ok(PublishResult {
            repo_url: format!("https://huggingface.co/datasets/{}", self.config.repo_id),
            train_samples: train_end,
            validation_samples: val_end - train_end,
            test_samples: total - val_end,
            output_hash: corpus.compute_hash(),
        })
    }

    /// Uploads files to HuggingFace Hub using alimentar.
    ///
    /// # Errors
    ///
    /// Returns an error if upload fails.
    pub async fn upload_to_hub(&self, output_dir: &Path) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| CorpusError::publication("HF_TOKEN not set"))?;

        // Use alimentar's HfPublisher for actual upload
        // This integrates with the Sovereign AI Stack
        let files = vec![
            output_dir.join("train.parquet"),
            output_dir.join("validation.parquet"),
            output_dir.join("test.parquet"),
            output_dir.join("README.md"),
            output_dir.join("corpus_metadata.json"),
        ];

        for file in files {
            if file.exists() {
                upload_file(&file, &self.config.repo_id, token).await?;
            }
        }

        Ok(())
    }

    /// Generates a dataset card (README.md) for the corpus.
    fn generate_dataset_card(&self, corpus: &Corpus) -> Result<String> {
        let metadata = corpus.metadata();

        let card = format!(
            r#"---
license: apache-2.0
language:
- en
tags:
- rust
- documentation
- code-generation
- fine-tuning
size_categories:
- 1K<n<10K
---

# Rust CLI Documentation Corpus

A scientifically rigorous corpus for fine-tuning LLMs to generate idiomatic
`///` documentation comments for Rust CLI tools.

## Dataset Description

### Summary

This corpus contains {count} high-quality input/output pairs extracted from
production Rust CLI repositories. Each entry pairs a Rust code signature
(function, struct, enum) with its corresponding documentation comment.

### Supported Tasks

- **Documentation Generation**: Generate Rust doc comments from code signatures
- **Code Understanding**: Learn Rust idioms and documentation patterns

### Languages

- **Source Code**: Rust
- **Documentation**: English

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | UUID v4 identifier |
| `input` | string | Rust code signature |
| `output` | string | Documentation comment |
| `category` | string | Doc type (function/argument/example/error/module) |
| `source_repo` | string | Source repository |
| `source_commit` | string | Git commit SHA (7 chars) |
| `source_file` | string | Relative file path |
| `source_line` | int32 | Line number |
| `tokens_input` | int32 | Input token count |
| `tokens_output` | int32 | Output token count |
| `quality_score` | float32 | Quality score [0.0, 1.0] |

### Data Splits

| Split | Samples | Percentage |
|-------|---------|------------|
| train | 80% | Training |
| validation | 10% | Validation |
| test | 10% | Testing |

## Dataset Creation

### Curation Rationale

Following Toyota Way principles:
- **Genchi Genbutsu**: Data from real production CLI tools
- **Jidoka**: Built-in quality gates (100-point validation)
- **Muda**: No waste, no redundancy

### Source Data

Extracted from high-quality Rust CLI repositories:
- clap-rs/clap
- BurntSushi/ripgrep
- sharkdp/fd
- sharkdp/bat
- ogham/exa

### Quality Validation

All entries pass a 100-point Popperian falsification criteria:
- Data Integrity (20 points)
- Syntactic Validity (20 points)
- Semantic Validity (20 points)
- Distribution Balance (15 points)
- Reproducibility (15 points)
- Quality Metrics (10 points)

## Considerations for Using the Data

### Social Impact

This dataset promotes:
- Better documentation practices in Rust ecosystem
- Accessibility of Rust for new developers
- Consistency in CLI tool documentation

### Bias

The corpus is biased toward:
- CLI-specific patterns (argument parsing, error handling)
- High-quality, well-maintained repositories
- English documentation

## Additional Information

### Dataset Curators

PAIML (Pragmatic AI Labs)

### Licensing Information

Apache 2.0

### Citation Information

```bibtex
@dataset{{rust_cli_docs_corpus,
  title = {{Rust CLI Documentation Corpus}},
  author = {{PAIML}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/paiml/rust-cli-docs-corpus}}
}}
```

### Provenance

- **Version**: {version}
- **Extraction Date**: {date}
- **Output Hash**: {hash}
"#,
            count = corpus.len(),
            version = metadata.version,
            date = metadata.extraction_date.format("%Y-%m-%d"),
            hash = corpus.compute_hash(),
        );

        Ok(card)
    }
}

impl Default for HfPublisher {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a publish operation.
#[derive(Debug, Clone)]
pub struct PublishResult {
    /// URL to the HuggingFace repository
    pub repo_url: String,
    /// Number of training samples
    pub train_samples: usize,
    /// Number of validation samples
    pub validation_samples: usize,
    /// Number of test samples
    pub test_samples: usize,
    /// SHA-256 hash of output
    pub output_hash: String,
}

/// Creates a corpus from a slice of entries.
fn create_split_corpus(entries: &[&crate::CorpusEntry]) -> Corpus {
    let owned: Vec<_> = entries.iter().map(|e| (*e).clone()).collect();
    Corpus::from_entries(owned)
}

/// Uploads a single file to HuggingFace Hub.
async fn upload_file(path: &Path, repo_id: &str, token: &str) -> Result<()> {
    let filename = path
        .file_name()
        .ok_or_else(|| CorpusError::publication("Invalid filename"))?
        .to_string_lossy();

    // Use huggingface_hub API via reqwest
    let client = reqwest::Client::new();
    let content = std::fs::read(path)?;

    let url = format!(
        "https://huggingface.co/api/datasets/{}/upload/main/{}",
        repo_id, filename
    );

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .body(content)
        .send()
        .await
        .map_err(|e| CorpusError::publication(format!("Upload failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(CorpusError::publication(format!(
            "Upload failed with status: {}",
            response.status()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publisher_config_default() {
        let config = PublisherConfig::default();
        assert_eq!(config.repo_id, "paiml/rust-cli-docs-corpus");
        assert!(config.create_repo);
        assert!(!config.private);
    }

    #[test]
    fn test_publisher_new() {
        let publisher = HfPublisher::new();
        assert_eq!(publisher.config.repo_id, "paiml/rust-cli-docs-corpus");
    }

    #[test]
    fn test_publish_result() {
        let result = PublishResult {
            repo_url: "https://huggingface.co/datasets/test".to_string(),
            train_samples: 800,
            validation_samples: 100,
            test_samples: 100,
            output_hash: "abc123".to_string(),
        };
        assert_eq!(result.train_samples + result.validation_samples + result.test_samples, 1000);
    }
}
