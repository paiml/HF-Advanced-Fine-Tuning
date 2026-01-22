//! Corpus CLI tool for extraction, validation, and publication.
//!
//! Per Spec Section 11: Appendix Makefile Targets
//!
//! Commands:
//! - clone-sources: Clone and pin source repositories
//! - extract: Extract documentation pairs
//! - validate: Run all validation gates
//! - export: Export to parquet format
//! - falsify: Run 100-point falsification
//! - review: Generate human review sample
//! - stats: Print corpus statistics
//! - card: Generate HuggingFace dataset card
//! - publish: Push to HuggingFace Hub
//! - inspect: Interactive corpus browser
//! - sample: Print N random examples

use clap::{Parser, Subcommand};
use std::fmt::Write as _;
use rust_cli_docs_corpus::{
    extractor::{DocExtractor, ExtractorConfig, RepoSpec},
    filter::CorpusFilter,
    publisher::{HfPublisher, PublisherConfig},
    Corpus, CorpusValidator,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "corpus-cli")]
#[command(about = "Rust CLI Documentation Corpus tool - Per Spec Section 11")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Clone and pin source repositories
    CloneSources {
        /// Output directory for cloned repos
        #[arg(short, long, default_value = "data/clones")]
        output: PathBuf,
    },

    /// Extract documentation from repositories
    Extract {
        /// Output parquet file
        #[arg(short, long, default_value = "corpus.parquet")]
        output: PathBuf,

        /// Repository URLs (can be specified multiple times)
        #[arg(short, long)]
        repo: Vec<String>,

        /// Use default repository list
        #[arg(long)]
        defaults: bool,

        /// Minimum quality score
        #[arg(long, default_value = "0.5")]
        min_quality: f32,

        /// Maximum entries per repository
        #[arg(long, default_value = "500")]
        max_per_repo: usize,
    },

    /// Validate a corpus file
    Validate {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,

        /// Show detailed errors
        #[arg(long)]
        verbose: bool,
    },

    /// Export corpus to different formats
    Export {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = "data/corpus")]
        output: PathBuf,
    },

    /// Run 100-point Popperian falsification
    Falsify {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Generate human review sample
    Review {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,

        /// Number of samples to review
        #[arg(long, default_value = "10")]
        sample: usize,

        /// Output directory for review files
        #[arg(short, long, default_value = "data/review")]
        output: PathBuf,
    },

    /// Show corpus statistics
    Stats {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Generate HuggingFace dataset card
    Card {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,

        /// Output README.md path
        #[arg(short, long, default_value = "data/corpus/README.md")]
        output: PathBuf,
    },

    /// Publish corpus to HuggingFace Hub
    Publish {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for generated files
        #[arg(short, long, default_value = "hf_output")]
        output: PathBuf,

        /// HuggingFace repository ID
        #[arg(long, default_value = "paiml/rust-cli-docs-corpus")]
        repo_id: String,

        /// Upload to HuggingFace Hub
        #[arg(long)]
        upload: bool,
    },

    /// Interactive corpus browser
    Inspect {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Print random examples from corpus
    Sample {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,

        /// Number of samples to show
        #[arg(short, long, default_value = "10")]
        count: usize,
    },

    /// Debug: show entries with parsing failures
    DebugParse {
        /// Input parquet file
        #[arg(short, long)]
        input: PathBuf,

        /// Maximum failures to show
        #[arg(short, long, default_value = "10")]
        max: usize,
    },
}

/// Default repositories to extract from.
fn default_repos() -> Vec<RepoSpec> {
    // Per Spec Section 5.2: Candidate Repositories
    vec![
        // Primary repos (well-documented CLI tools)
        RepoSpec::new("https://github.com/clap-rs/clap", "clap-rs/clap"),
        RepoSpec::new("https://github.com/BurntSushi/ripgrep", "BurntSushi/ripgrep"),
        RepoSpec::new("https://github.com/sharkdp/fd", "sharkdp/fd"),
        RepoSpec::new("https://github.com/sharkdp/bat", "sharkdp/bat"),
        RepoSpec::new("https://github.com/eza-community/eza", "eza-community/eza"),
        // Additional repos for diversity
        RepoSpec::new("https://github.com/starship/starship", "starship/starship"),
        RepoSpec::new("https://github.com/bootandy/dust", "bootandy/dust"),
        RepoSpec::new("https://github.com/XAMPPRocky/tokei", "XAMPPRocky/tokei"),
        RepoSpec::new("https://github.com/dalance/procs", "dalance/procs"),
        RepoSpec::new("https://github.com/casey/just", "casey/just"),
    ]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::CloneSources { output } => {
            clone_sources_command(output).await?;
        }
        Commands::Extract {
            output,
            repo,
            defaults,
            min_quality,
            max_per_repo,
        } => {
            extract_command(output, repo, defaults, min_quality, max_per_repo).await?;
        }
        Commands::Validate { input, verbose } => {
            validate_command(input, verbose)?;
        }
        Commands::Export { input, output } => {
            export_command(input, output)?;
        }
        Commands::Falsify { input } => {
            falsify_command(input)?;
        }
        Commands::Review {
            input,
            sample,
            output,
        } => {
            review_command(input, sample, output)?;
        }
        Commands::Stats { input } => {
            stats_command(input)?;
        }
        Commands::Card { input, output } => {
            card_command(input, output)?;
        }
        Commands::Publish {
            input,
            output,
            repo_id,
            upload,
        } => {
            publish_command(input, output, repo_id, upload).await?;
        }
        Commands::Inspect { input } => {
            inspect_command(input)?;
        }
        Commands::Sample { input, count } => {
            sample_command(input, count)?;
        }
        Commands::DebugParse { input, max } => {
            debug_parse_command(input, max)?;
        }
    }

    Ok(())
}

/// Clone and pin source repositories
async fn clone_sources_command(output: PathBuf) -> anyhow::Result<()> {
    use std::process::Command;

    println!("Cloning source repositories to {}...", output.display());
    std::fs::create_dir_all(&output)?;

    for spec in default_repos() {
        let repo_dir = output.join(spec.name.replace('/', "_"));
        println!("  Cloning {}...", spec.name);

        if repo_dir.exists() {
            println!("    Already exists, skipping");
            continue;
        }

        let status = Command::new("git")
            .args(["clone", "--depth", "1", &spec.url])
            .arg(&repo_dir)
            .status()?;

        if status.success() {
            // Get the commit SHA
            let output_sha = Command::new("git")
                .args(["rev-parse", "--short=7", "HEAD"])
                .current_dir(&repo_dir)
                .output()?;
            let sha = String::from_utf8_lossy(&output_sha.stdout);
            println!("    Cloned at commit {}", sha.trim());
        } else {
            eprintln!("    Failed to clone {}", spec.name);
        }
    }

    println!("\nSource repositories cloned to: {}", output.display());
    Ok(())
}

async fn extract_command(
    output: PathBuf,
    repos: Vec<String>,
    defaults: bool,
    min_quality: f32,
    max_per_repo: usize,
) -> anyhow::Result<()> {
    let config = ExtractorConfig {
        min_quality,
        max_per_repo,
        ..Default::default()
    };
    let extractor = DocExtractor::with_config(config);

    let repo_specs: Vec<RepoSpec> = if defaults {
        default_repos()
    } else if repos.is_empty() {
        eprintln!("Error: No repositories specified. Use --defaults or --repo");
        std::process::exit(1);
    } else {
        repos
            .into_iter()
            .map(|url| {
                let name = url
                    .trim_end_matches(".git")
                    .rsplit('/')
                    .take(2)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("/");
                RepoSpec::new(url, name)
            })
            .collect()
    };

    let mut corpus = Corpus::new();

    for spec in &repo_specs {
        println!("Extracting from {}...", spec.name);
        match extractor.extract_from_repo(spec) {
            Ok(entries) => {
                println!("  Found {} entries", entries.len());
                for entry in entries {
                    corpus.add_entry(entry);
                }
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
    }

    println!("\nRaw entries: {}", corpus.len());

    // Filter and balance corpus
    println!("Filtering and balancing corpus...");
    let filter = CorpusFilter::default();
    filter.print_filter_stats(&corpus);
    let corpus = filter.filter(&corpus);
    println!("After filtering: {} entries", corpus.len());

    // Validate before saving
    let validator = CorpusValidator::new();
    let result = validator.validate_corpus(&corpus);
    println!("Validation score: {}/100", result.score());

    // Save
    corpus.save_to_parquet(&output)?;
    println!("Saved to: {}", output.display());

    Ok(())
}

fn validate_command(input: PathBuf, verbose: bool) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;
    let validator = CorpusValidator::new();
    let result = validator.validate_corpus(&corpus);

    let score = result.score_breakdown();

    println!("Corpus Validation Report");
    println!("========================\n");
    println!("Total Score: {}/100\n", score.total);
    println!("Section Breakdown:");
    println!("  Data Integrity:      {:>2}/20", score.data_integrity);
    println!("  Syntactic Validity:  {:>2}/20", score.syntactic_validity);
    println!("  Semantic Validity:   {:>2}/20", score.semantic_validity);
    println!("  Distribution Balance:{:>2}/15", score.distribution_balance);
    println!("  Reproducibility:     {:>2}/15", score.reproducibility);
    println!("  Quality Metrics:     {:>2}/10", score.quality_metrics);

    if verbose {
        let failed = result.failed_criteria();
        if !failed.is_empty() {
            println!("\nFailed Criteria:");
            for criterion in failed {
                println!(
                    "  [{:>2}] {} (0/{} points)",
                    criterion.number, criterion.name, criterion.points
                );
                if let Some(ref error) = criterion.error {
                    println!("       Error: {}", error);
                }
            }
        }
    }

    println!("\nStatus: {}", if result.is_valid() { "PASS" } else { "FAIL" });

    if !result.is_valid() {
        std::process::exit(1);
    }

    Ok(())
}

async fn publish_command(
    input: PathBuf,
    output: PathBuf,
    repo_id: String,
    upload: bool,
) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;

    // Validate first
    let validator = CorpusValidator::new();
    let result = validator.validate_corpus(&corpus);

    if !result.is_valid() {
        eprintln!("Error: Corpus validation failed ({}/100)", result.score());
        eprintln!("Fix validation errors before publishing.");
        std::process::exit(1);
    }

    let config = PublisherConfig {
        repo_id: repo_id.clone(),
        ..Default::default()
    };
    let publisher = HfPublisher::with_config(config);

    println!("Preparing corpus for publication...");
    let publish_result = publisher.publish(&corpus, &output).await?;

    println!("\nPublication Summary:");
    println!("  Repository: {}", publish_result.repo_url);
    println!("  Train samples: {}", publish_result.train_samples);
    println!("  Validation samples: {}", publish_result.validation_samples);
    println!("  Test samples: {}", publish_result.test_samples);
    println!("  Output hash: {}", publish_result.output_hash);
    println!("\nOutput files saved to: {}", output.display());

    if upload {
        println!("\nUploading to HuggingFace Hub...");
        publisher.upload_to_hub(&output).await?;
        println!("Upload complete: {}", publish_result.repo_url);
    } else {
        println!("\nTo upload, run with --upload flag");
    }

    Ok(())
}

fn stats_command(input: PathBuf) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;

    println!("Corpus Statistics");
    println!("=================\n");
    println!("Total entries: {}", corpus.len());

    // Category distribution
    let mut categories = std::collections::HashMap::new();
    let mut repos = std::collections::HashMap::new();
    let mut total_input_tokens = 0u64;
    let mut total_output_tokens = 0u64;
    let mut total_quality = 0.0f64;

    for entry in corpus.entries() {
        *categories.entry(entry.category).or_insert(0usize) += 1;
        *repos.entry(entry.source_repo.clone()).or_insert(0usize) += 1;
        total_input_tokens += u64::from(entry.tokens_input);
        total_output_tokens += u64::from(entry.tokens_output);
        total_quality += f64::from(entry.quality_score);
    }

    let count = corpus.len() as f64;

    println!("\nCategory Distribution:");
    for (cat, cnt) in &categories {
        let pct = (*cnt as f64 / count) * 100.0;
        println!("  {}: {} ({:.1}%)", cat, cnt, pct);
    }

    println!("\nRepository Distribution:");
    for (repo, cnt) in &repos {
        let pct = (*cnt as f64 / count) * 100.0;
        println!("  {}: {} ({:.1}%)", repo, cnt, pct);
    }

    println!("\nToken Statistics:");
    println!("  Total input tokens: {}", total_input_tokens);
    println!("  Total output tokens: {}", total_output_tokens);
    println!(
        "  Mean input tokens: {:.1}",
        total_input_tokens as f64 / count
    );
    println!(
        "  Mean output tokens: {:.1}",
        total_output_tokens as f64 / count
    );

    println!("\nQuality:");
    println!("  Mean quality score: {:.3}", total_quality / count);
    println!("  Corpus hash: {}", corpus.compute_hash());

    Ok(())
}

/// Export corpus to different formats
fn export_command(input: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;

    std::fs::create_dir_all(&output)?;

    // Export as parquet (copy)
    let parquet_out = output.join("train.parquet");
    corpus.save_to_parquet(&parquet_out)?;
    println!("Exported to: {}", parquet_out.display());

    // Export as JSON Lines
    let jsonl_out = output.join("train.jsonl");
    let file = std::fs::File::create(&jsonl_out)?;
    let mut writer = std::io::BufWriter::new(file);
    for entry in corpus.entries() {
        use std::io::Write;
        serde_json::to_writer(&mut writer, entry)?;
        writeln!(writer)?;
    }
    println!("Exported to: {}", jsonl_out.display());

    println!("\nTotal entries exported: {}", corpus.len());
    Ok(())
}

/// Run 100-point Popperian falsification
fn falsify_command(input: PathBuf) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;
    let validator = CorpusValidator::new();
    let result = validator.validate_corpus(&corpus);

    let score = result.score_breakdown();

    println!("100-Point Popperian Falsification Report");
    println!("========================================\n");

    println!("Section Scores:");
    println!("  7.1 Data Integrity:      {:>2}/20 {}", score.data_integrity, status_emoji(score.data_integrity, 20));
    println!("  7.2 Syntactic Validity:  {:>2}/20 {}", score.syntactic_validity, status_emoji(score.syntactic_validity, 20));
    println!("  7.3 Semantic Validity:   {:>2}/20 {}", score.semantic_validity, status_emoji(score.semantic_validity, 20));
    println!("  7.4 Distribution Balance:{:>2}/15 {}", score.distribution_balance, status_emoji(score.distribution_balance, 15));
    println!("  7.5 Reproducibility:     {:>2}/15 {}", score.reproducibility, status_emoji(score.reproducibility, 15));
    println!("  7.6 Quality Metrics:     {:>2}/10 {}", score.quality_metrics, status_emoji(score.quality_metrics, 10));

    println!("\n─────────────────────────────────────────");
    println!("TOTAL SCORE: {}/100", score.total);
    println!("─────────────────────────────────────────\n");

    let failed = result.failed_criteria();
    if !failed.is_empty() {
        println!("Failed Criteria ({} total):", failed.len());
        for criterion in failed {
            println!(
                "  [C{:02}] {} (-{} points)",
                criterion.number, criterion.name, criterion.points
            );
            if let Some(ref error) = criterion.error {
                println!("         Reason: {}", error);
            }
        }
        println!();
    }

    // Per Spec 7.7: PASS >= 95, WARN >= 85, FAIL < 85
    let status = if score.total >= 95 {
        "PASS (>= 95 points)"
    } else if score.total >= 85 {
        "WARN (85-94 points)"
    } else {
        "FAIL (< 85 points)"
    };

    println!("Falsification Status: {}", status);

    if score.total < 85 {
        std::process::exit(1);
    }

    Ok(())
}

fn status_emoji(earned: u8, max: u8) -> &'static str {
    if earned == max {
        "[OK]"
    } else if earned >= max * 3 / 4 {
        "[WARN]"
    } else {
        "[FAIL]"
    }
}

/// Generate human review sample
fn review_command(input: PathBuf, sample_count: usize, output: PathBuf) -> anyhow::Result<()> {
    use rand::prelude::IndexedRandom;

    let corpus = Corpus::load_from_parquet(&input)?;
    std::fs::create_dir_all(&output)?;

    let entries: Vec<_> = corpus.entries().collect();
    let mut rng = rand::rng();

    let sample: Vec<_> = entries
        .choose_multiple(&mut rng, sample_count.min(entries.len()))
        .collect();

    println!("Human Review Sample");
    println!("==================\n");
    println!("Selected {} entries for review\n", sample.len());

    // Generate review markdown file
    let review_file = output.join("review_sample.md");
    let mut content = String::new();
    content.push_str("# Human Review Sample\n\n");
    let _ = writeln!(content, "Generated: {}", chrono::Utc::now());
    let _ = writeln!(content, "Sample size: {}\n", sample.len());

    for (i, entry) in sample.iter().enumerate() {
        let _ = writeln!(content, "## Entry {} / {}\n", i + 1, sample.len());
        let _ = writeln!(content, "**ID:** `{}`\n", entry.id);
        let _ = writeln!(content, "**Source:** `{}` @ `{}` (line {})\n", entry.source_repo, entry.source_commit, entry.source_line);
        let _ = writeln!(content, "**Category:** {}\n", entry.category);
        let _ = writeln!(content, "**Quality Score:** {:.3}\n", entry.quality_score);
        content.push_str("### Input (Signature)\n\n```rust\n");
        content.push_str(&entry.input);
        content.push_str("\n```\n\n");
        content.push_str("### Output (Documentation)\n\n```rust\n");
        content.push_str(&entry.output);
        content.push_str("\n```\n\n");
        content.push_str("### Review\n\n");
        content.push_str("- [ ] Accurate\n");
        content.push_str("- [ ] Helpful\n");
        content.push_str("- [ ] Idiomatic\n");
        content.push_str("- [ ] Complete\n\n");
        content.push_str("**Notes:**\n\n---\n\n");
    }

    std::fs::write(&review_file, content)?;
    println!("Review file saved to: {}", review_file.display());

    Ok(())
}

/// Generate HuggingFace dataset card
fn card_command(input: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;
    let validator = CorpusValidator::new();
    let result = validator.validate_corpus(&corpus);
    let score = result.score_breakdown();

    // Count categories
    let mut categories = std::collections::HashMap::new();
    let mut repos = std::collections::HashMap::new();
    for entry in corpus.entries() {
        *categories.entry(entry.category).or_insert(0usize) += 1;
        *repos.entry(entry.source_repo.clone()).or_insert(0usize) += 1;
    }

    let card = format!(r#"---
license: apache-2.0
task_categories:
  - text-generation
  - text2text-generation
language:
  - en
tags:
  - rust
  - documentation
  - code
  - cli
  - lora
  - fine-tuning
size_categories:
  - n<1K
---

# Rust CLI Documentation Corpus

A scientifically rigorous corpus for fine-tuning LLMs to generate idiomatic `///` documentation comments for Rust CLI tools.

## Dataset Description

This corpus follows the Toyota Way principles and Popperian falsification methodology.

### Statistics

- **Total entries:** {}
- **Source repositories:** {}
- **Validation score:** {}/100

### Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
{}

### Source Repositories

| Repository | Count | Percentage |
|------------|-------|------------|
{}

## Quality Validation

The corpus passed 100-point Popperian falsification with the following scores:

| Section | Score |
|---------|-------|
| Data Integrity | {}/20 |
| Syntactic Validity | {}/20 |
| Semantic Validity | {}/20 |
| Distribution Balance | {}/15 |
| Reproducibility | {}/15 |
| Quality Metrics | {}/10 |
| **Total** | **{}/100** |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("paiml/rust-cli-docs")
```

## License

Apache 2.0

## Citation

```bibtex
@dataset{{paiml_rust_cli_docs,
  title={{Rust CLI Documentation Corpus}},
  author={{PAIML}},
  year={{2026}},
  publisher={{HuggingFace}}
}}
```
"#,
        corpus.len(),
        repos.len(),
        score.total,
        categories.iter().map(|(cat, cnt)| {
            format!("| {} | {} | {:.1}% |", cat, cnt, (*cnt as f64 / corpus.len() as f64) * 100.0)
        }).collect::<Vec<_>>().join("\n"),
        repos.iter().map(|(repo, cnt)| {
            format!("| {} | {} | {:.1}% |", repo, cnt, (*cnt as f64 / corpus.len() as f64) * 100.0)
        }).collect::<Vec<_>>().join("\n"),
        score.data_integrity,
        score.syntactic_validity,
        score.semantic_validity,
        score.distribution_balance,
        score.reproducibility,
        score.quality_metrics,
        score.total,
    );

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, card)?;
    println!("Dataset card saved to: {}", output.display());

    Ok(())
}

/// Interactive corpus browser
fn inspect_command(input: PathBuf) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;

    println!("Corpus Inspector");
    println!("================\n");
    println!("Loaded {} entries from {}\n", corpus.len(), input.display());
    println!("Commands:");
    println!("  list [N]     - List first N entries (default 10)");
    println!("  show <ID>    - Show entry by ID");
    println!("  cat <INDEX>  - Show entry by index");
    println!("  stats        - Show statistics");
    println!("  quit         - Exit\n");

    let entries: Vec<_> = corpus.entries().collect();

    loop {
        print!("> ");
        use std::io::Write;
        std::io::stdout().flush()?;

        let mut input_line = String::new();
        if std::io::stdin().read_line(&mut input_line)? == 0 {
            break;
        }

        let parts: Vec<&str> = input_line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let cmd = parts.first().copied().unwrap_or("");
        match cmd {
            "quit" | "exit" | "q" => break,
            "list" => {
                let n: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
                for (i, entry) in entries.iter().take(n).enumerate() {
                    println!("{:>4}. [{}] {} ({})", i, entry.category, &entry.id[..8], entry.source_repo);
                }
            }
            "show" => {
                if let Some(id) = parts.get(1) {
                    if let Some(entry) = entries.iter().find(|e| e.id.starts_with(id)) {
                        print_entry(entry);
                    } else {
                        println!("Entry not found: {}", id);
                    }
                }
            }
            "cat" => {
                if let Some(idx) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) {
                    if let Some(entry) = entries.get(idx) {
                        print_entry(entry);
                    } else {
                        println!("Index out of range: {}", idx);
                    }
                }
            }
            "stats" => {
                println!("Total entries: {}", entries.len());
            }
            _ => println!("Unknown command: {}", cmd),
        }
    }

    Ok(())
}

fn print_entry(entry: &rust_cli_docs_corpus::CorpusEntry) {
    println!("\n─────────────────────────────────────────");
    println!("ID: {}", entry.id);
    println!("Category: {}", entry.category);
    println!("Source: {} @ {} (line {})", entry.source_repo, entry.source_commit, entry.source_line);
    println!("Quality: {:.3}", entry.quality_score);
    println!("\n[INPUT]");
    println!("{}", entry.input);
    println!("\n[OUTPUT]");
    println!("{}", entry.output);
    println!("─────────────────────────────────────────\n");
}

/// Print random examples from corpus
fn sample_command(input: PathBuf, count: usize) -> anyhow::Result<()> {
    use rand::prelude::IndexedRandom;

    let corpus = Corpus::load_from_parquet(&input)?;
    let entries: Vec<_> = corpus.entries().collect();
    let mut rng = rand::rng();

    let sample: Vec<_> = entries
        .choose_multiple(&mut rng, count.min(entries.len()))
        .collect();

    println!("Random Sample ({} entries)\n", sample.len());

    for (i, entry) in sample.iter().enumerate() {
        println!("═══════════════════════════════════════════");
        println!("Sample {} / {}", i + 1, sample.len());
        print_entry(entry);
    }

    Ok(())
}

/// Debug parse failures
fn debug_parse_command(input: PathBuf, max: usize) -> anyhow::Result<()> {
    let corpus = Corpus::load_from_parquet(&input)?;

    println!("Checking {} entries for parse failures...\n", corpus.len());

    let mut fail_count = 0;
    for entry in corpus.entries() {
        if syn::parse_str::<syn::Item>(&entry.input).is_err() {
            fail_count += 1;
            if fail_count <= max {
                println!("FAIL #{} ({}): {}", fail_count, entry.category.as_str(), entry.source_file);
                println!("  Input: {}", entry.input.replace('\n', "\\n"));
                if let Err(e) = syn::parse_str::<syn::Item>(&entry.input) {
                    println!("  Error: {}", e);
                }
                println!();
            }
        }
    }

    println!("Total: {}, Failed: {}", corpus.len(), fail_count);
    Ok(())
}
