# Rust CLI Documentation Corpus Specification

**Document:** RCDC-SPEC-001
**Version:** 1.0.0
**Status:** Draft
**Date:** January 2026
**Philosophy:** The Toyota Way (Lean Principles)
**Target Dataset:** `paiml/rust-cli-docs`

---

## 1. Executive Summary

This specification defines a reproducible, scientifically rigorous corpus for fine-tuning large language models to generate idiomatic `///` documentation comments for Rust CLI tools. The corpus enables QLoRA fine-tuning of Qwen2.5-Coder-7B on consumer hardware (24GB VRAM).

**Core Thesis:** The transformation from "knows Rust" to "writes your doc style" is a narrow, learnable diff. A small, high-quality corpus (100-500 examples) with LoRA rank 8 is sufficient for style transfer.

**Publication Target:** [huggingface.co/datasets/paiml/rust-cli-docs](https://huggingface.co/datasets/paiml/rust-cli-docs)

---

## 2. The Toyota Way: Design Principles

### 2.1 Genchi Genbutsu (Go and See)

**Principle:** Go to the source to find facts.

**Application:** We extract documentation from *real* production CLI tools, not synthetic examples. The source of truth is:
1. `rustc` - Does the code compile?
2. `cargo doc` - Does the documentation render correctly?
3. Human review - Is this documentation actually helpful?

**Validation:** Each corpus entry must pass all three gates.

### 2.2 Jidoka (Built-in Quality)

**Principle:** Build quality into the process; stop when problems occur.

**Application:** The extraction pipeline implements automated quality gates:
- Syntax validation (valid `///` format)
- Content validation (non-empty, non-trivial)
- Semantic validation (matches function signature)

**Andon Cord:** If any example fails validation, the pipeline halts and reports. No silent failures.

### 2.3 Muda (Waste Elimination)

**Principle:** Eliminate waste in all forms.

**Application:**
- **Over-processing waste:** Simple style transfer doesn't need 10,000 examples. 100-500 high-quality examples suffice.
- **Inventory waste:** No storing redundant examples. Each entry must be unique.
- **Defect waste:** Automated gates catch errors before they enter the corpus.

### 2.4 Heijunka (Leveling)

**Principle:** Balance workload across categories.

**Application:** The corpus maintains balanced representation across:

| Category | Target % | Description |
|----------|----------|-------------|
| Function docs | 40% | `/// Description` for functions |
| Argument docs | 25% | `/// # Arguments` sections |
| Example docs | 20% | `/// # Examples` with code blocks |
| Error docs | 10% | `/// # Errors` and `/// # Panics` |
| Module docs | 5% | `//!` module-level documentation |

### 2.5 Kaizen (Continuous Improvement)

**Principle:** Continuously improve through small, incremental changes.

**Application:**
- Version-controlled corpus with semantic versioning
- Quarterly human review of random 5% sample
- Feedback loop from model outputs to corpus refinement

---

## 3. Scientific Reproducibility

### 3.1 Idempotency Guarantee

Running the extraction pipeline twice with the same inputs produces byte-identical outputs.

| Component | Reproducibility Mechanism |
|-----------|---------------------------|
| Source repos | Pinned by git commit SHA |
| Extraction tool | `alimentar` version pinned in `Cargo.lock` |
| Output format | Apache Parquet with deterministic row ordering |
| Compression | zstd level 3 (deterministic) |
| Random sampling | Fixed seed (42) |

### 3.2 Provenance Tracking

Every corpus entry includes:

```json
{
  "source_repo": "clap-rs/clap",
  "source_commit": "a1b2c3d4e5f6",
  "source_file": "src/builder/command.rs",
  "source_line": 142,
  "extraction_date": "2026-01-21T00:00:00Z",
  "extractor_version": "alimentar 0.5.0"
}
```

### 3.3 Environment Specification

```toml
[environment]
rust_version = "1.83.0"
alimentar_version = "0.5.0"
platform = "x86_64-unknown-linux-gnu"

[sources]
clap = { repo = "clap-rs/clap", commit = "..." }
tokio = { repo = "tokio-rs/tokio", commit = "..." }
serde = { repo = "serde-rs/serde", commit = "..." }
# ... additional sources
```

---

## 4. Data Schema

### 4.1 Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | UUID v4 for deduplication |
| `input` | string | Function/struct signature (prompt) |
| `output` | string | Documentation comment (completion) |
| `category` | string | {function, argument, example, error, module} |
| `source_repo` | string | GitHub repository |
| `source_commit` | string | Git SHA (7 chars) |
| `source_file` | string | Relative file path |
| `source_line` | int32 | Line number |
| `tokens_input` | int32 | Token count (tiktoken cl100k) |
| `tokens_output` | int32 | Token count (tiktoken cl100k) |
| `quality_score` | float32 | Automated quality score [0-1] |

### 4.2 Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "input": "pub fn parse_args(args: &[String]) -> Result<Config, Error> {",
  "output": "/// Parses command-line arguments into application configuration.\n///\n/// # Arguments\n///\n/// * `args` - Raw command-line arguments, typically from `std::env::args()`\n///\n/// # Returns\n///\n/// A `Config` struct containing parsed options, or an `Error` if parsing fails.\n///\n/// # Examples\n///\n/// ```\n/// let args = vec![\"app\".into(), \"--verbose\".into()];\n/// let config = parse_args(&args)?;\n/// assert!(config.verbose);\n/// ```",
  "category": "function",
  "source_repo": "example/cli-tool",
  "source_commit": "a1b2c3d",
  "source_file": "src/cli.rs",
  "source_line": 42,
  "tokens_input": 23,
  "tokens_output": 89,
  "quality_score": 0.95
}
```

---

## 5. Extraction Pipeline

### 5.1 Source Selection Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| GitHub stars | ≥100 | Community validation |
| Recent activity | Commit in last 6 months | Maintained codebase |
| License | OSI-approved | Legal clarity |
| Doc coverage | ≥50% public items documented | Sufficient examples |
| CLI focus | Uses clap/structopt/argh | Domain relevance |

### 5.2 Candidate Repositories

| Repository | Stars | CLI Framework | Category Focus |
|------------|-------|---------------|----------------|
| clap-rs/clap | 14k+ | clap | Argument parsing |
| BurntSushi/ripgrep | 48k+ | clap | Search tool |
| sharkdp/fd | 34k+ | clap | File finding |
| sharkdp/bat | 50k+ | clap | File viewing |
| ogham/exa | 24k+ | clap | File listing |
| starship/starship | 45k+ | clap | Shell prompt |
| bootandy/dust | 9k+ | clap | Disk usage |
| XAMPPRocky/tokei | 11k+ | clap | Code statistics |
| dalance/procs | 5k+ | clap | Process viewer |
| casey/just | 20k+ | clap | Command runner |

### 5.3 Extraction Commands

```bash
# Clone and pin sources
make corpus-clone-sources

# Extract documentation pairs
make corpus-extract

# Validate and filter
make corpus-validate

# Export to parquet
make corpus-export

# Full pipeline (idempotent)
make corpus-build
```

### 5.4 Quality Gates

```
┌─────────────────┐
│  Raw Extraction │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Syntax Check    │ ← Valid /// format?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Content Check   │ ← Non-trivial? (>10 chars)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Check  │ ← Matches signature?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Deduplication   │ ← Unique by content hash?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Balance Check   │ ← Category distribution?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Corpus   │
└─────────────────┘
```

---

## 6. Annotated Bibliography

### 6.1 Fine-Tuning Methodology

**1. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv:2106.09685*.**
- **Summary:** Introduces LoRA, demonstrating that fine-tuning can be achieved by training low-rank decomposition matrices rather than full model weights.
- **Relevance:** Foundational technique for this corpus. Validates that style transfer requires only ~0.1% of parameters.

**2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.**
- **Summary:** Combines 4-bit quantization with LoRA, enabling 65B model fine-tuning on single 48GB GPU.
- **Relevance:** Enables our target: 7B model fine-tuning on 24GB consumer GPU.

**3. Zhou, C., et al. (2023). "LIMA: Less Is More for Alignment." *NeurIPS 2023*.**
- **Summary:** Demonstrates that 1,000 carefully curated examples can outperform 50,000 lower-quality examples.
- **Relevance:** Validates our small-corpus approach. Quality > quantity for style transfer.

### 6.2 Code Documentation

**4. Clement, C., et al. (2020). "PyMT5: Multi-mode Translation of Natural Language and Python Code." *EMNLP 2020*.**
- **Summary:** Trains models to translate between code and documentation bidirectionally.
- **Relevance:** Establishes that code↔documentation is a learnable translation task.

**5. Mastropaolo, A., et al. (2021). "Studying the Usage of Text-To-Text Transfer Transformer to Support Code-Related Tasks." *ICSE 2021*.**
- **Summary:** Evaluates T5 on code summarization, finding that domain-specific fine-tuning dramatically improves results.
- **Relevance:** Supports fine-tuning on Rust-specific documentation patterns.

**6. Chen, M., et al. (2021). "Evaluating Large Language Models Trained on Code." *arXiv:2107.03374 (Codex)*.**
- **Summary:** Demonstrates that code LLMs benefit from diverse, high-quality training data.
- **Relevance:** Validates sourcing from multiple high-quality CLI repositories.

### 6.3 Data Quality

**7. Just, R., et al. (2014). "Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies." *ISSTA 2014*.**
- **Summary:** Establishes reproducibility standards for software engineering research corpora.
- **Relevance:** Model for our provenance tracking and reproducibility guarantees.

**8. Allamanis, M., et al. (2019). "The Adverse Effects of Code Duplication in Machine Learning Models of Code." *OOPSLA 2019*.**
- **Summary:** Shows that code duplication between train/test sets inflates metrics by 100%+.
- **Relevance:** Justifies our strict deduplication gate.

### 6.4 Lean Principles

**9. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill.**
- **Summary:** Foundational text on Lean manufacturing principles.
- **Relevance:** Framework for corpus design (Genchi Genbutsu, Jidoka, Muda).

**10. Poppendieck, M. & Poppendieck, T. (2003). *Lean Software Development*. Addison-Wesley.**
- **Summary:** Adapts Toyota Production System to software development.
- **Relevance:** Validates applying Lean principles to ML corpus engineering.

### 6.5 Scientific Methodology

**11. Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.**
- **Summary:** Establishes falsificationism as the demarcation criterion for scientific theories.
- **Relevance:** Framework for our 100-point falsification criteria.

**12. Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*.**
- **Summary:** ML reproducibility checklist adopted by major conferences.
- **Relevance:** Informs our reproducibility guarantees and provenance tracking.

---

## 7. Popperian Falsification Criteria

Following Popper's falsificationism, a scientific corpus must specify conditions under which it would be considered **invalid**. Each criterion is testable and must pass for the corpus to be valid.

### 7.1 Data Integrity (20 points)

| # | Criterion | Test | Points |
|---|-----------|------|--------|
| 1 | All source commits exist | `git cat-file -e <sha>` | 2 |
| 2 | All source files exist at specified commit | `git show <sha>:<path>` | 2 |
| 3 | All line numbers valid | Line exists in file | 2 |
| 4 | No null/empty inputs | `input.len() > 0` | 2 |
| 5 | No null/empty outputs | `output.len() > 0` | 2 |
| 6 | All UUIDs unique | `COUNT(DISTINCT id) = COUNT(*)` | 2 |
| 7 | All UUIDs valid v4 | Regex validation | 2 |
| 8 | Parquet schema matches spec | Schema comparison | 2 |
| 9 | No duplicate content hashes | SHA-256 uniqueness | 2 |
| 10 | Timestamps in valid range | 2020-01-01 to extraction date | 2 |

### 7.2 Syntactic Validity (20 points)

| # | Criterion | Test | Points |
|---|-----------|------|--------|
| 11 | All inputs parse as Rust | `syn::parse_str()` | 2 |
| 12 | All outputs start with `///` or `//!` | Prefix check | 2 |
| 13 | All outputs valid UTF-8 | Encoding validation | 2 |
| 14 | No malformed markdown in docs | `pulldown-cmark` parse | 2 |
| 15 | Code blocks in docs compile | `rustc --check` | 2 |
| 16 | No unbalanced delimiters | Bracket matching | 2 |
| 17 | No control characters | `!char.is_control()` | 2 |
| 18 | Line lengths ≤ 100 chars | Length check | 2 |
| 19 | Consistent line endings (LF) | `\r` absence | 2 |
| 20 | No trailing whitespace | Trim comparison | 2 |

### 7.3 Semantic Validity (20 points)

| # | Criterion | Test | Points |
|---|-----------|------|--------|
| 21 | Function docs describe function | NLP relevance ≥0.5 | 2 |
| 22 | Argument docs match parameters | Name extraction | 2 |
| 23 | Return docs match return type | Type extraction | 2 |
| 24 | Error docs mention error types | Type mention check | 2 |
| 25 | Examples use documented item | AST reference check | 2 |
| 26 | No hallucinated parameters | Param existence check | 2 |
| 27 | No hallucinated types | Type existence check | 2 |
| 28 | Docs in English | Language detection | 2 |
| 29 | No profanity/inappropriate content | Content filter | 2 |
| 30 | No PII (emails, names) | PII detection | 2 |

### 7.4 Distribution Balance (15 points)

| # | Criterion | Test | Points |
|---|-----------|------|--------|
| 31 | Function docs 35-45% | Category count | 2 |
| 32 | Argument docs 20-30% | Category count | 2 |
| 33 | Example docs 15-25% | Category count | 2 |
| 34 | Error docs 5-15% | Category count | 2 |
| 35 | Module docs 3-7% | Category count | 2 |
| 36 | ≥5 source repositories | Repo count | 2 |
| 37 | No repo >40% of corpus | Repo distribution | 3 |

### 7.5 Reproducibility (15 points)

| # | Criterion | Test | Points |
|---|-----------|------|--------|
| 38 | Extraction idempotent | Hash equality on re-run | 3 |
| 39 | All deps version-pinned | Lock file check | 2 |
| 40 | Environment documented | Env spec exists | 2 |
| 41 | Sources cloneable | Git clone succeeds | 2 |
| 42 | Pipeline runnable | `make corpus-build` succeeds | 3 |
| 43 | Output hash documented | SHA-256 in metadata | 3 |

### 7.6 Quality Metrics (10 points)

| # | Criterion | Test | Points |
|---|-----------|------|--------|
| 44 | Mean quality score ≥0.7 | Aggregate check | 2 |
| 45 | No examples with score <0.3 | Minimum threshold | 2 |
| 46 | Token count reasonable (10-500) | Range check | 2 |
| 47 | Input/output ratio 1:2 to 1:10 | Ratio calculation | 2 |
| 48 | Human review approval ≥90% | Sample review | 2 |

### 7.7 Validation Score Calculation

```
Total Score = Σ(passed criteria × points) / 100

PASS:  Score ≥ 95 (≤5 points failed)
WARN:  Score ≥ 85 (≤15 points failed)
FAIL:  Score < 85 (>15 points failed)
```

### 7.8 Falsification Statement

**This corpus is INVALID if any of the following are true:**

1. Total validation score < 85/100
2. Any Data Integrity criterion fails (criteria 1-10)
3. Any Reproducibility criterion fails (criteria 38-43)
4. Human review approval < 90% (criterion 48)

**Falsification is PERMANENT until the corpus is re-extracted and re-validated.**

---

## 8. HuggingFace Publication

### 8.1 Dataset Card

```yaml
---
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
```

### 8.2 Publication Workflow

```bash
# 1. Build and validate corpus
make corpus-build
make corpus-validate

# 2. Generate dataset card
make corpus-card

# 3. Login to HuggingFace
huggingface-cli login

# 4. Push to hub
make corpus-publish
# Equivalent to:
# huggingface-cli upload paiml/rust-cli-docs ./data/corpus/
```

### 8.3 Versioning

| Version | Date | Records | Notes |
|---------|------|---------|-------|
| 1.0.0 | 2026-Q1 | ~300 | Initial release |
| 1.1.0 | 2026-Q2 | ~400 | Additional repos |
| 2.0.0 | 2026-Q3 | ~500 | Schema revision |

---

## 9. Usage

### 9.1 Loading the Dataset

```rust
// Using Rust (with hf-hub crate)
use hf_hub::api::sync::Api;

let api = Api::new()?;
let repo = api.dataset("paiml/rust-cli-docs");
let path = repo.get("data/train.parquet")?;
```

```python
# Using Python (for comparison/validation)
from datasets import load_dataset

ds = load_dataset("paiml/rust-cli-docs")
print(f"Training examples: {len(ds['train'])}")
```

### 9.2 Fine-Tuning Configuration

```toml
[model]
base = "Qwen/Qwen2.5-Coder-7B"
load_in_4bit = true

[lora]
rank = 8
alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
dropout = 0.05

[training]
epochs = 3
batch_size = 4
learning_rate = 2e-4
warmup_ratio = 0.03
```

---

## 10. Appendix: Makefile Targets

```makefile
# Corpus management
corpus-clone-sources    # Clone and pin source repositories
corpus-extract          # Extract documentation pairs
corpus-validate         # Run all validation gates
corpus-export           # Export to parquet format
corpus-build            # Full pipeline (idempotent)

# Quality assurance
corpus-falsify          # Run 100-point falsification
corpus-review           # Generate human review sample
corpus-stats            # Print corpus statistics

# Publication
corpus-card             # Generate HuggingFace dataset card
corpus-publish          # Push to HuggingFace Hub

# Development
corpus-inspect          # Interactive corpus browser
corpus-sample N=10      # Print N random examples
```

---

## 11. Changelog

### v1.0.0 (2026-01-21)
- Initial specification
- 100-point Popperian falsification criteria
- Toyota Way design principles
- 12 peer-reviewed citations

---

**Document Control:**
- **Author:** Noah Gift / Claude Code (Opus 4.5)
- **Review:** Pending
- **Approval:** Pending
