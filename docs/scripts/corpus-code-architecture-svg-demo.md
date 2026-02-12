# Corpus Code Architecture SVG Demo Script

**Duration:** ~3 minutes
**Visual:** `docs/images/week2/corpus-code-architecture.svg`
**Format:** SVG walkthrough only (no live code)

---

## INTRO (15 seconds)

> "This diagram shows the internal architecture of our corpus crate—a pure Rust implementation for building ML training datasets. Let me walk you through each component."

---

## SECTION 1: CorpusEntry Struct (30 seconds)

**[Point to top-left green panel]**

> "CorpusEntry is the atomic data unit. Twelve typed fields organized in four groups:
>
> Identity: UUID v5 for deterministic deduplication, plus a category enum—Function, Argument, Example, Error, or Module.
>
> Content: The input field holds function signatures, output holds rustdoc comments.
>
> Provenance: Source repo, commit SHA, file path, line number. Full traceability.
>
> Quality: A float score from 0 to 1, computed by heuristics."

---

## SECTION 2: Module Architecture (45 seconds)

**[Point to top-right blue panel]**

> "Four modules handle the pipeline stages:
>
> Extractor parses Rust files using the syn crate—real AST parsing, not regex. It walks syntax trees and pairs signatures with their doc comments.
>
> Filter applies seven quality gates: score thresholds, token limits, balanced delimiters, no control characters. It also handles deduplication and category balancing.
>
> Validator runs our 100-point Popperian falsification suite. Six categories: Data Integrity, Syntactic, Semantic, Distribution, Reproducibility, and Quality Metrics. Pass threshold is 85.
>
> Publisher generates Parquet files with 80/10/10 train/val/test splits, stratified by category. Uses the alimentar crate for HuggingFace uploads."

---

## SECTION 3: CLI Commands (30 seconds)

**[Point to middle-left orange panel]**

> "Twelve CLI commands built with clap:
>
> The grid shows all subcommands. Key ones: clone-sources for repo setup, extract for parsing, falsify for validation, publish for HuggingFace upload.
>
> Also includes inspect—an interactive TUI browser—and sample for quick spot-checks.
>
> Each command is type-safe with helpful error messages."

---

## SECTION 4: Data Flow (30 seconds)

**[Point to bottom-left purple panel]**

> "The data flow shows how entries move through the system:
>
> GitHub repos feed into the Extractor. Entries pass through Filter, then Validator. The validated corpus goes to Publisher, which outputs to HuggingFace Hub.
>
> The Arrow schema at bottom shows the five output columns: id, source, input, output, and quality_score. All stored in Parquet format."

---

## SECTION 5: Demo Commands (20 seconds)

**[Point to bottom-right teal panel]**

> "Three commands to try it yourself:
>
> Extract with defaults pulls from seven curated Rust CLI repos.
>
> Falsify runs the full 100-point test suite.
>
> Stats shows category distribution and quality histograms.
>
> Pre-build with cargo build --release to avoid compile waits during demos."

---

## CONCLUSION (10 seconds)

> "That's the corpus crate architecture—extraction, filtering, validation, and publishing in pure Rust. No Python dependencies, fully reproducible, falsification-tested."

---

## TIMING SUMMARY

| Section | Duration |
|---------|----------|
| Intro | 0:15 |
| CorpusEntry Struct | 0:30 |
| Module Architecture | 0:45 |
| CLI Commands | 0:30 |
| Data Flow | 0:30 |
| Demo Commands | 0:20 |
| Conclusion | 0:10 |
| **Total** | **~3:00** |

---

## PRESENTER NOTES

- Keep pointer visible when referencing panels
- Follow the natural reading order: top-left → top-right → middle → bottom
- The color coding helps: green=data, blue=modules, orange=CLI, purple=flow, teal=demos
- Key phrase to repeat: "pure Rust, no Python"
- If questions about performance: extraction ~30s for 10 repos
- If questions about falsification: 96/100 typical score, 85 minimum to pass
