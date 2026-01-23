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
  - fine-tuning
  - rustdoc
  - sovereign-ai
size_categories:
  - n<1K
pretty_name: "Rust CLI Documentation Corpus"
---

# Rust CLI Documentation Corpus

**RCDC-SPEC-001 v3.0.0 | APPROVED FOR PUBLICATION**

A scientifically rigorous corpus for fine-tuning LLMs to generate idiomatic `///` documentation comments for Rust CLI tools. Follows rustdoc conventions (RFC 1574, RFC 1946).

## Dataset Description

This corpus follows the Toyota Way principles and Popperian falsification methodology.

### Statistics

| Metric | Value |
|--------|-------|
| **Total entries** | 100 |
| **Source repositories** | 7 |
| **Validation score** | 100/100 |
| **Specification** | RCDC-SPEC-001 v3.0.0 |

### Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| function | 46 | 46.0% |
| argument | 29 | 29.0% |
| example | 14 | 14.0% |
| module | 6 | 6.0% |
| error | 5 | 5.0% |

### Source Repositories

| Repository | Count | Stars |
|------------|-------|-------|
| BurntSushi/ripgrep | 38 | 48k+ |
| clap-rs/clap | 26 | 14k+ |
| eza-community/eza | 17 | 12k+ |
| starship/starship | 7 | 45k+ |
| sharkdp/fd | 5 | 34k+ |
| XAMPPRocky/tokei | 4 | 11k+ |
| sharkdp/bat | 3 | 49k+ |

## Data Format

```json
{
  "text": "<code>fn parse_args() -> Config {}</code>\n<doc>/// Parses command-line arguments...</doc>"
}
```

## Quality Validation

100-point Popperian falsification checklist verified:

| Section | Score | Status |
|---------|-------|--------|
| Data Integrity | 20/20 | PASSED |
| Syntactic Validity | 20/20 | PASSED |
| Semantic Validity | 20/20 | PASSED |
| Distribution Balance | 15/15 | PASSED |
| Reproducibility | 15/15 | PASSED |
| Quality Metrics | 10/10 | PASSED |
| **Total** | **100/100** | **VERIFIED** |

## Training Results

Model trained on this corpus achieved:

| Metric | Value |
|--------|-------|
| Final Loss | 0.602 |
| Best Loss | 0.472 |
| Perplexity | 1.83 |
| Loss Reduction | 96.38% |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("paiml/rust-cli-docs")
print(dataset["train"][0]["text"])
```

## License

Apache 2.0

## Citation

```bibtex
@dataset{rust-cli-docs-corpus-2026,
  title={Rust CLI Documentation Corpus},
  author={Gift, Noah},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/paiml/rust-cli-docs},
  note={RCDC-SPEC-001 v3.0.0}
}
```

## Provenance

| Field | Value |
|-------|-------|
| **Specification** | RCDC-SPEC-001 |
| **Version** | 3.0.0 |
| **Date** | 2026-01-23 |
| **Status** | APPROVED FOR PUBLICATION |

---

*Built with the Sovereign AI Stack (entrenar/realizar)*
