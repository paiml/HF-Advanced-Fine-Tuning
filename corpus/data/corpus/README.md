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

# Rust CLI Documentation Corpus

A scientifically rigorous corpus for fine-tuning LLMs to generate idiomatic `///` documentation comments for Rust CLI tools.

## Dataset Description

This corpus follows the Toyota Way principles and Popperian falsification methodology.

### Statistics

- **Total entries:** 50
- **Source repositories:** 6
- **Validation score:** 96/100

### Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| function | 22 | 44.0% |
| error | 3 | 6.0% |
| module | 3 | 6.0% |
| example | 6 | 12.0% |
| argument | 16 | 32.0% |

### Source Repositories

| Repository | Count | Percentage |
|------------|-------|------------|
| clap-rs/clap | 16 | 32.0% |
| BurntSushi/ripgrep | 17 | 34.0% |
| sharkdp/bat | 1 | 2.0% |
| XAMPPRocky/tokei | 2 | 4.0% |
| sharkdp/fd | 4 | 8.0% |
| eza-community/eza | 10 | 20.0% |

## Quality Validation

The corpus passed 100-point Popperian falsification with the following scores:

| Section | Score |
|---------|-------|
| Data Integrity | 20/20 |
| Syntactic Validity | 20/20 |
| Semantic Validity | 20/20 |
| Distribution Balance | 11/15 |
| Reproducibility | 15/15 |
| Quality Metrics | 10/10 |
| **Total** | **96/100** |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("paiml/rust-cli-docs")
```

## License

Apache 2.0

## Citation

```bibtex
@dataset{paiml_rust_cli_docs,
  title={Rust CLI Documentation Corpus},
  author={PAIML},
  year={2026},
  publisher={HuggingFace}
}
```
