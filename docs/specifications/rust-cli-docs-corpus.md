# Rust CLI Documentation Corpus Specification

**Document:** RCDC-SPEC-001
**Version:** 1.0.1
**Status:** Draft (Reviewed by Dr. Popper)
**Date:** January 2026
**Philosophy:** The Toyota Way (Lean Principles) & Critical Rationalism
**Publication Target:** [huggingface.co/datasets/paiml/rust-cli-docs](https://huggingface.co/datasets/paiml/rust-cli-docs)

---

## 1. Executive Summary (The Conjecture)

This specification defines a reproducible, scientifically rigorous corpus for fine-tuning large language models to generate idiomatic `///` documentation comments for Rust CLI tools.

**Core Conjecture:** The transformation from "knows Rust" to "writes your doc style" is a narrow, learnable diff. We conjecture that a small, rigorously curated corpus (100-500 examples) with LoRA rank 8 is sufficient for style transfer, provided it survives our falsification attempts.

**Methodology:** We do not seek to *verify* that the corpus is "good" (an impossible task). Instead, we seek to identify and eliminate "bad" data through aggressive falsification tests. The corpus remains valid only as long as it withstands these tests.

---

## 2. The Toyota Way: Design Principles

### 2.1 Genchi Genbutsu (Go and See)

**Principle:** Go to the source to find facts.

**Application:** We extract documentation from *real* production CLI tools. These serve as our empirical reality.
1. `rustc` - The absolute falsifier. If it doesn't compile, it is rejected.
2. `cargo doc` - The rendering falsifier. If it breaks HTML generation, it is rejected.
3. Human review - The semantic falsifier. Is this helpful, or merely syntactically correct noise?

**Validation:** Each corpus entry is a hypothesis that must survive these three gates.

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

### 3.4 The Problem of Induction

We explicitly acknowledge that no amount of passing tests proves the corpus "perfect" (the problem of induction). We can only state that it has not yet been falsified by our 100-point verification matrix. All claims of quality are provisional.

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
  "output": "/// Parses command-line arguments into application configuration.\n///\n/// # Arguments\n///\n/// * `args` - Raw command-line arguments, typically from `std::env::args()`\n///\n/// # Returns\n///\n/// A `Config` struct containing parsed options, or an `Error` if parsing fails.\n///\n/// # Examples\n///\n/// ```\n/// let args = vec![\"app\".into(), \"--verbose\".into()];\n/// let config = parse_args(&args)?;
/// assert!(config.verbose);
/// ```",
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
- **Summary:** Shows that code duplication between train/test sets inflates metrics by 100%*.*
- **Relevance:** Justifies our strict deduplication gate.

### 6.4 Lean Principles

**9. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill.**
- **Summary:** Foundational text on Lean manufacturing principles.
- **Relevance:** Framework for corpus design (Genchi Genbutsu, Jidoka, Muda).

**10. Poppendieck, M. & Poppendieck, T. (2003). *Lean Software Development*. Addison-Wesley.**
- **Summary:** Adapts Toyota Production System to software development.
- **Relevance:** Validates applying Lean principles to ML corpus engineering.

### 6.5 Scientific Methodology (The Karl Popper Extension)

**11. Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.**
- **Summary:** Establishes falsificationism as the demarcation criterion for scientific theories.
- **Relevance:** We treat the corpus not as "truth" but as a set of hypotheses. Every validation step is an attempt to falsify the hypothesis that "this is good data."

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
| 46 | Token count strictly bounded | `10 <= count <= 500` | 2 |
| 47 | Input/output ratio valid | `2.0 <= (output/input) <= 10.0` | 2 |
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

### 8.2 Publication Workflow (Sovereign AI Stack)

**Tooling:** All publication uses `alimentar` from the batuta Sovereign AI Stack. No Python dependencies.

```bash
# 1. Build and validate corpus
make corpus-build
make corpus-validate

# 2. Generate dataset card
make corpus-card

# 3. Set HuggingFace token (environment variable)
export HF_TOKEN="hf_..."  # From huggingface.co/settings/tokens

# 4. Push to hub via alimentar
make corpus-publish
# Equivalent to:
# alimentar hub push data/corpus/train.parquet paiml/rust-cli-docs \
#   --path-in-repo data/train.parquet \
#   --readme data/corpus/README.md \
#   --message "Release v1.0.0"
```

**alimentar capabilities:**
- `alimentar hub push` - Upload parquet files (uses LFS for binary)
- `alimentar hub push --readme` - Upload dataset card with validation
- Validates `task_categories` against official HuggingFace schema
- Pure Rust, no Python runtime required

### 8.3 Versioning

| Version | Date | Records | Notes |
|---------|------|---------|-------|
| 1.0.0 | 2026-Q1 | ~300 | Initial release |
| 1.1.0 | 2026-Q2 | ~400 | Additional repos |
| 2.0.0 | 2026-Q3 | ~500 | Schema revision |

---

## 9. Usage

### 9.1 Loading the Dataset (Sovereign AI Stack)

**CLI (alimentar):**
```bash
# Download from HuggingFace Hub
alimentar import hf paiml/rust-cli-docs -o data/train.parquet --split train

# Inspect the dataset
alimentar info data/train.parquet
alimentar head data/train.parquet -n 5
```

**Library (alimentar crate):**
```rust
use alimentar::{hf_hub::HfDataset, Dataset};

// Import from HuggingFace Hub
let hf = HfDataset::builder("paiml/rust-cli-docs")
    .split("train")
    .build()?;

let dataset = hf.download()?;
println!("Loaded {} examples", dataset.len());

// Access data
for batch in dataset.iter_batches(32) {
    let inputs = batch.column("input")?;
    let outputs = batch.column("output")?;
    // Process...
}
```

**Alternative (DuckDB for SQL queries):**
```bash
# Direct query without download
duckdb -c "
SELECT input, output, category
FROM 'https://huggingface.co/datasets/paiml/rust-cli-docs/resolve/main/data/train.parquet'
LIMIT 10
"
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

## 10. Sovereign AI Stack Dependencies

This specification uses **ONLY** tools from the batuta Sovereign AI Stack. No Python dependencies.

### 10.1 Required Tools

| Tool | Crate | Purpose | Install |
|------|-------|---------|---------|
| `alimentar` | [alimentar](https://crates.io/crates/alimentar) | Data loading, HuggingFace Hub | `cargo install alimentar` |
| `batuta` | [batuta](https://crates.io/crates/batuta) | Orchestration, analysis | `cargo install batuta` |
| `pacha` | [pacha](https://crates.io/crates/pacha) | Model/data registry | `cargo install pacha` |
| `entrenar` | [entrenar](https://crates.io/crates/entrenar) | Fine-tuning execution | `cargo install entrenar` |
| `certeza` | [certeza](https://crates.io/crates/certeza) | Quality validation | `cargo install certeza` |

### 10.2 Version Pinning

```toml
# Cargo.toml for corpus tooling
[dependencies]
alimentar = "0.5"
batuta = "0.4"
pacha = "0.1"

[build-dependencies]
# Pinned in Cargo.lock for reproducibility
```

### 10.3 Why Pure Rust?

1. **Reproducibility**: Single `cargo install` vs. complex Python environments
2. **Performance**: Zero-copy Arrow for large datasets
3. **Security**: Memory-safe data handling, no pickle vulnerabilities
4. **Sovereignty**: No external cloud dependencies, local-first design

---

## 11. Appendix: Makefile Targets

```makefile
# Corpus management (uses alimentar)
corpus-clone-sources    # Clone and pin source repositories
corpus-extract          # Extract documentation pairs (alimentar doctest)
corpus-validate         # Run all validation gates
corpus-export           # Export to parquet format
corpus-build            # Full pipeline (idempotent)

# Quality assurance (uses certeza)
corpus-falsify          # Run 100-point falsification
corpus-review           # Generate human review sample
corpus-stats            # Print corpus statistics

# Publication (uses alimentar hub)
corpus-card             # Generate HuggingFace dataset card
corpus-publish          # Push to HuggingFace Hub via alimentar

# Development
corpus-inspect          # Interactive corpus browser (alimentar repl)
corpus-sample N=10      # Print N random examples
```

---

## 12. Changelog

### v1.1.0 (2026-01-21) - Corpus Expansion
- **Expanded extraction parameters:** Lowered `min_quality` from 0.5 to 0.4
- **Deepened repository mining:** All 10 candidate repos now actively extracted
- **Target:** 100+ entries (previously 50)
- **Strategy:** Options 1+2 from expansion analysis (add missing repos + relax filters)
- Added Appendix G: Corpus Expansion Strategy

### v1.0.1 (2026-01-21) - Dr. Popper Review
- Refined Executive Summary to emphasize "Conjecture and Refutation".
- Added "The Problem of Induction" (Sec 3.4).
- Sharpened "Quality Metrics" (Sec 7.6) with strict boundaries.
- Added Appendix E: Team Prompts.
- Added Appendix F: The Null Hypothesis

### v1.0.0 (2026-01-21)
- Initial specification
- 100-point Popperian falsification criteria
- Toyota Way design principles
- 12 peer-reviewed citations

---

## Appendix E: Team Prompts (The Critical Rationalist's Toolkit)

Use these prompts to maintain rigorous skepticism during implementation:

1.  **The "Anti-Induction" Prompt:**
    > "I am currently assuming this data extraction logic is correct because it worked on the first 5 files. Dr. Popper, how would you design a test specifically to break this logic on the 6th file?"

2.  **The "Falsification" Prompt:**
    > "We have a new candidate repo. Instead of checking if it fits, write a script that tries to prove it *doesn't* fit our quality standards (e.g., check for non-English comments, broken markdown, or excessive length). Only if it survives this attack do we accept it."

3.  **The "Demarcation" Prompt:**
    > "I am unsure if this doc comment is 'good'. Instead of using my intuition, help me define a strict, testable rule (a demarcation line) that cleanly separates 'good' from 'bad' in this context, so we can automate the decision."

4.  **The "Crucial Experiment" Prompt:**
    > "We have two competing extraction strategies (A and B). Design a 'crucial experiment'—a single test case where A and B predict opposite results—so we can decisively refute one of them."

---

## Appendix F: The Null Hypothesis

In scientific testing, we must define what we are trying to disprove.

**Null Hypothesis ($H_0$):**
> "The generated documentation from our fine-tuned model is statistically indistinguishable from random documentation snippets or generic 'AI' filler text."

**Alternative Hypothesis ($H_1$):**
> "The generated documentation specifically reflects the idiomatic style, structure, and content patterns of high-quality Rust CLI tools found in the corpus."

**Falsification Strategy:**
We do not try to prove $H_1$. We try to reject $H_0$ by:
1.  **Blind A/B Testing:** Can human experts distinguish real corpus examples from model output? (If they can't, we fail to reject $H_0$ regarding quality, or we have successfully achieved mimicry).
2.  **Functional Testing:** Does the generated code in `/// # Examples` actually compile? (Random noise ($H_0$) would not compile).

---

## Appendix G: Corpus Expansion Strategy (v1.1.0)

### Problem Statement

Initial extraction yielded 50 entries. The LIMA paper and spec conjecture require 100-500 entries for effective style transfer.

### Expansion Analysis

| Strategy | Entries Gained | Implementation |
|----------|---------------|----------------|
| 1. Add missing spec repos | +30-50 | starship, dust, procs, just |
| 2. Relax quality filters | +20-40 | min_quality 0.5→0.4 |
| 3. Category-targeted mining | +10-20 | Focus on underrepresented categories |
| 4. Adjacent domain repos | +20-30 | nushell, helix, zellij, zoxide |
| 5. Manual curation sprint | +10-20 | Human review top candidates |

### Selected Strategy: Options 1 + 2

**Rationale:**
- Options 1+2 are automatable and maintain reproducibility
- Combined expected yield: 50-90 entries → total 100-140
- Preserves Popperian falsification methodology
- No manual curation bias introduced

### Extraction Parameters

```toml
# Before (v1.0.x)
[extraction]
min_quality = 0.5
max_per_repo = 500

# After (v1.1.0)
[extraction]
min_quality = 0.4
max_per_repo = 500
```

### Expected Outcomes

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Total entries | 50 | 100+ | 100-500 |
| Active repos | 6 | 10 | ≥5 |
| Validation score | 96/100 | ≥95/100 | ≥95 |

### Verification

Triple-Build Attack must pass after expansion to confirm determinism is preserved.

---

**Document Control:**
- **Author:** Noah Gift / Claude Code (Opus 4.5)
- **Advisor:** Dr. Karl Popper (Gemini CLI)
- **Review:** Completed (v1.1.0)
- **Approval:** Pending