# Corpus Creation & Publishing Pipeline Demo Script

**Duration:** ~4 minutes
**Audience:** ML practitioners, data engineers, Rust developers
**Visual:** `docs/images/week2/corpus-pipeline.svg`

---

## INTRO (20 seconds)

> "Today I'll walk you through how we build a high-quality training corpus from scratch—starting with source code repositories and ending with a published dataset on HuggingFace.
>
> This pipeline is fully automated, rigorously tested, and produces falsification-validated data ready for fine-tuning."

---

## STEP 1: SOURCE REPOS (30 seconds)

**[Point to top-left panel with blue border]**

> "Everything starts with real production code. We pull from 7 popular Rust CLI repositories—ripgrep, clap, eza, starship, fd, tokei, and bat.
>
> The distribution is intentional: ripgrep contributes 38% because it has exemplary documentation. Clap provides 26%—essential for CLI argument parsing patterns.
>
> We clone each repo and extract function signatures paired with their rustdoc comments. This gives us real-world examples of how experienced Rust developers document their code."

---

## STEP 2: EXTRACTION (30 seconds)

**[Point to second panel with purple border]**

> "The extraction phase parses Rust source files using syn-serde for AST analysis.
>
> For each documented function, we create an instruction-response pair. The INPUT is the function signature—name, parameters, return type. The OUTPUT is the documentation we want the model to learn to generate.
>
> This isn't scraping markdown files. We're parsing actual Rust code to understand the relationship between signatures and their documentation."

---

## STEP 3: FALSIFICATION (45 seconds)

**[Point to third panel with orange border]**

> "Here's what makes this corpus special—Popperian falsification testing. Every example must survive rigorous quality gates.
>
> Data Integrity: 20 out of 20. No corrupted entries, no malformed pairs.
>
> Syntactic Validity: 20 out of 20. Every function signature parses correctly, every doc comment is valid rustdoc.
>
> Semantic Validity: 20 out of 20. The documentation actually describes what the function does—no hallucinations, no mismatches.
>
> Distribution: 11 out of 15. We're slightly over on function docs, slightly under on examples—acceptable variance.
>
> Total: 73 tests pass, 96 out of 100 quality score. Only examples that survive falsification attempts make the cut."

---

## STEP 4: CORPUS OUTPUT (30 seconds)

**[Point to fourth panel with green border]**

> "The validated corpus is saved as a Parquet file—columnar storage optimized for ML workloads.
>
> Each entry has five fields: a unique ID, the source crate name, the input signature, the output documentation, and a quality score.
>
> Final result: 100 curated entries from 7 crates. Small but high-quality beats large and noisy every time."

---

## STEP 5: PUBLISH TO HUGGINGFACE (30 seconds)

**[Point to fifth panel with teal border]**

> "Publishing uses alimentar, our Rust-native HuggingFace client. One command: `cargo run --release` with alimentar push.
>
> The dataset goes live at huggingface.co/datasets/paiml/rust-cli-docs-corpus. Apache 2.0 licensed, immediately usable by anyone.
>
> No Python dependencies. Pure Rust from source code to published dataset."

---

## CATEGORY DISTRIBUTION (20 seconds)

**[Point to bottom-left section]**

> "The corpus covers four documentation categories: function docs at 46%, argument descriptions at 29%, code examples at 14%, and module-level docs plus error handling at 11%.
>
> This distribution mirrors real Rust project documentation structure."

---

## TOOL CHAIN (20 seconds)

**[Point to bottom-center section]**

> "Three tools power this pipeline. syn-serde for Rust AST parsing, the corpus crate for testing and validation, and alimentar for HuggingFace publishing.
>
> Underneath: Arrow 57 for columnar data, Parquet 57 for file format, and hf-hub for the HuggingFace API. All pure Rust."

---

## QUALITY PIPELINE (30 seconds)

**[Point to Quality Pipeline section]**

> "Watch the funnel effect. We start with roughly 500 raw candidates extracted from source repos.
>
> Syntax filtering drops us to 250—only valid AST structures pass.
>
> Semantic filtering reduces to 150—documentation must be meaningful, not just present.
>
> Quality scoring above 0.8 gives us the final 100 entries.
>
> 80% of candidates are filtered out. That's the cost of quality—and it's worth it."

---

## CONCLUSION (20 seconds)

> "That's the complete corpus creation pipeline—from git clone to HuggingFace dataset in five automated steps.
>
> Key insight: quality gates matter more than volume. 100 falsification-tested examples outperform 10,000 scraped ones.
>
> The dataset is live. Clone it, fine-tune with it, build better documentation generators. Link in the description."

---

## TIMING SUMMARY

| Section | Duration |
|---------|----------|
| Intro | 0:20 |
| Step 1: Source Repos | 0:30 |
| Step 2: Extraction | 0:30 |
| Step 3: Falsification | 0:45 |
| Step 4: Corpus Output | 0:30 |
| Step 5: Publish | 0:30 |
| Category Distribution | 0:20 |
| Tool Chain | 0:20 |
| Quality Pipeline | 0:30 |
| Conclusion | 0:20 |
| **Total** | **~4:15** |

---

## PRESENTER NOTES

- Keep pointer visible when referencing panels
- Emphasize "falsification-tested"—this is the differentiator
- The 80% filter rate is memorable—repeat it
- For live demos, have the HuggingFace dataset page open
- Key stats to remember: 7 repos, 100 entries, 73 tests, 96/100 score
- If asked about scaling: quality gates are the bottleneck by design
