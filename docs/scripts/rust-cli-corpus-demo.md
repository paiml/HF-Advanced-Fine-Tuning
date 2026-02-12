# Rust CLI Documentation Corpus Demo Script

**Duration:** ~3 minutes
**Audience:** ML practitioners, Rust developers, fine-tuning enthusiasts
**Visual:** `docs/images/week2/corpus-pipeline.svg`
**Live Dataset:** https://huggingface.co/datasets/paiml/rust-cli-docs-corpus

---

## INTRO (20 seconds)

> "Today I'm showing you the Rust CLI Documentation Corpus—a curated dataset for fine-tuning language models to generate Rust CLI documentation.
>
> This is live on HuggingFace under paiml/rust-cli-docs-corpus. Let's explore what's in it."

---

## DATASET OVERVIEW (30 seconds)

**[Point to Dataset card tab]**

> "The corpus contains 100 high-quality instruction-response pairs extracted from 7 popular Rust CLI repositories.
>
> Each entry teaches the model to generate proper rustdoc-style documentation—complete with argument descriptions, return values, and working examples.
>
> Source crates include ripgrep, clap, bat, fd, tokio, and serde—real production code that millions of developers rely on."

---

## DATASET VIEWER (45 seconds)

**[Point to Dataset Viewer rows]**

> "Let's look at the actual data. Each row has an ID identifying the source crate, and a text column with the instruction-response pair.
>
> Here's an example from clap: the instruction is a function signature for `parse_args`, and the response is complete rustdoc with the Arguments section, Returns description, and a code example.
>
> Notice the quality—these aren't auto-generated. Each example was validated through Popperian falsification testing to ensure structural correctness and semantic accuracy.
>
> The 'crate' column tracks provenance so you know exactly where each example originated."

---

## DATA DISTRIBUTION (30 seconds)

**[Point to Files and versions tab]**

> "The distribution is intentional. Ripgrep contributes 38% of examples—it has exemplary documentation. Clap provides 26%—critical for CLI argument parsing patterns.
>
> Category-wise: 46% function documentation, 29% argument docs, 14% examples, and 11% error handling. This matches real-world Rust documentation structure."

---

## USAGE (30 seconds)

**[Point to 'Use this dataset' button]**

> "Using the corpus is simple. Click 'Use this dataset' and you get the HuggingFace datasets code:
>
> ```python
> from datasets import load_dataset
> ds = load_dataset('paiml/rust-cli-docs-corpus')
> ```
>
> Or with the Sovereign AI Stack in pure Rust:
>
> ```bash
> alimentar pull paiml/rust-cli-docs-corpus
> entrenar train --corpus rust-cli-docs
> ```
>
> No Python dependencies required."

---

## QUALITY ASSURANCE (30 seconds)

> "What makes this corpus special is the quality pipeline.
>
> Every example passes 73 falsification tests—data integrity, syntactic validity, semantic accuracy. The corpus scored 96 out of 100 on our Popperian quality metric.
>
> We reject hallucinations. If a generated doc mentions a flag that doesn't exist, it's filtered out. Only examples that survive rigorous testing make the cut."

---

## CONCLUSION (15 seconds)

> "That's the Rust CLI Documentation Corpus—100 curated examples from real Rust projects, falsification-tested, ready for fine-tuning.
>
> Clone it, fine-tune a LoRA adapter, and generate documentation for your own CLI tools. Link is in the description."

---

## TIMING SUMMARY

| Section | Duration |
|---------|----------|
| Intro | 0:20 |
| Dataset Overview | 0:30 |
| Dataset Viewer | 0:45 |
| Data Distribution | 0:30 |
| Usage | 0:30 |
| Quality Assurance | 0:30 |
| Conclusion | 0:15 |
| **Total** | **~3:20** |

---

## PRESENTER NOTES

- Have HuggingFace page open in browser before starting
- Scroll through Dataset Viewer to show variety of examples
- Emphasize "falsification-tested"—this differentiates from scraped datasets
- For live coding, have `alimentar` and `entrenar` installed
- Key stat to remember: 100 examples, 7 crates, 96/100 quality score
