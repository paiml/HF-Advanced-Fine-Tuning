# Corpus Code Architecture Demo Script

**Duration:** ~6 minutes
**Audience:** Rust developers, ML engineers, software architects
**Visual:** `docs/images/week2/corpus-code-architecture.svg`
**Live Code:** `corpus/` directory

---

## INTRO (20 seconds)

> "Let me show you the internals of our corpus crate—a pure Rust implementation for extracting, validating, and publishing ML training data. No Python, no notebooks, just type-safe Rust with rigorous quality gates."

---

## PART 1: CORE DATA TYPE (45 seconds)

**[Point to CorpusEntry struct in diagram, then show code]**

```bash
# Show the entry struct
cat corpus/src/entry.rs | head -50
```

> "CorpusEntry is our atomic unit. Twelve fields, fully typed.
>
> The `id` is a UUID v5—deterministic from input plus output. Same content always gets the same ID. This enables deduplication across runs.
>
> `input` holds the function signature, `output` holds the documentation. `category` is an enum: Function, Argument, Example, Error, or Module.
>
> Source provenance: repo name, 7-character commit SHA, file path, line number. Full traceability to the original code.
>
> `quality_score` is a float from 0 to 1—computed by our scoring heuristics, not arbitrary."

**Key file:** `corpus/src/entry.rs:15-45`

---

## PART 2: EXTRACTOR MODULE (60 seconds)

**[Point to extractor.rs section, then demo]**

```bash
# Show the extract_from_file function
grep -A 30 "fn extract_from_file" corpus/src/extractor.rs | head -40
```

> "The extractor uses the `syn` crate for AST parsing. We don't regex-scrape documentation—we parse actual Rust syntax trees.
>
> `extract_from_file()` walks every item in a Rust file. For each function, struct, or enum with doc comments, we extract the signature and docs as a pair.
>
> Watch this—let me show you `format_fn_signature()`. It cleans the raw AST into a readable signature string, stripping unnecessary whitespace and formatting.
>
> The `score_entry()` function computes quality. Base score is 0.6. Bonuses for: documentation length over 50 chars, presence of examples, Arguments section, Errors section. Penalty for docs under 20 chars. Clamped to [0, 1].
>
> Category detection: if docs contain '# Example' or code blocks, it's Example. If they mention '# Arguments', it's Argument. Content-based, not filename-based."

**Key files:**
- `corpus/src/extractor.rs:180-250` (extract_from_file)
- `corpus/src/extractor.rs:320-380` (score_entry)

**Live demo:**
```bash
# Run extraction on a single repo
cargo run extract --repo https://github.com/clap-rs/clap --output demo.json --max-per-repo 20
```

---

## PART 3: FILTER MODULE (45 seconds)

**[Point to filter.rs section]**

```bash
# Show validity checks
grep -A 20 "fn is_valid_entry" corpus/src/filter.rs
```

> "The filter is where quality gates live. Seven hard checks:
>
> 1. Quality score at least 0.4—we reject low-quality docs.
> 2. Line length under 100 characters—no minified garbage.
> 3. Balanced delimiters—parentheses, brackets, braces must match.
> 4. Balanced code blocks—no unclosed triple backticks.
> 5. No control characters except newlines and tabs.
> 6. Token count between 10 and 500—not too short, not too long.
> 7. Input/output ratio between 1 and 15—docs should be longer than signatures.
>
> After validity filtering: deduplication by content hash, category balancing to hit our targets, and repo dominance limiting—no single repo can exceed 38% of the corpus."

**Key file:** `corpus/src/filter.rs:45-120`

---

## PART 4: VALIDATOR MODULE (60 seconds)

**[Point to validator.rs section, then run falsification]**

```bash
# Run the 100-point falsification
cargo run falsify --input corpus.json
```

> "This is the Popperian falsification engine—100 points across six categories.
>
> Data Integrity: 20 points. Are source commits valid 7-character hex? Do files end in .rs? Are line numbers positive? Are UUIDs valid v5 format?
>
> Syntactic Validity: 20 points. Does every input parse as valid Rust? Does every output start with `///` or `//!`? Valid UTF-8? Balanced markdown?
>
> Semantic Validity: 20 points. Do docs actually describe the function? Any PII like email addresses?
>
> Distribution: 15 points. Are categories within target ranges? At least 5 repos represented? No repo over 40%?
>
> Reproducibility: 15 points. Cargo.lock present? rust-toolchain.toml pinned? Makefile exists?
>
> Quality Metrics: 10 points. Mean quality above 0.7? No entry below 0.3?
>
> Pass threshold is 85. We consistently hit 96."

**Key file:** `corpus/src/validator.rs:50-200`

---

## PART 5: PUBLISHER MODULE (45 seconds)

**[Point to publisher.rs section]**

```bash
# Show the publish function
grep -A 25 "pub fn publish" corpus/src/publisher.rs | head -30
```

> "Publisher creates the final dataset artifacts. Three parquet files: 80% train, 10% validation, 10% test. Stratified by category so each split has the same distribution.
>
> We use the `alimentar` crate for HuggingFace integration—pure Rust, no Python SDK.
>
> The dataset card is auto-generated: license, task categories, statistics, field descriptions, usage examples, even a BibTeX citation.
>
> `PublishResult` returns the repo URL and a SHA-256 hash of the output. Fully deterministic—same input corpus produces identical output."

**Key file:** `corpus/src/publisher.rs:80-150`

**Live demo:**
```bash
# Publish (dry run without --upload)
cargo run publish --input corpus.json --output hf_output --repo-id test/demo
```

---

## PART 6: CLI WALKTHROUGH (45 seconds)

**[Show the CLI commands in terminal]**

```bash
# Full pipeline demo
cargo run -- --help
```

> "Twelve CLI commands cover the full workflow:
>
> `clone-sources`: Git clone all default repos with depth=1.
>
> `extract`: Parse repos, generate corpus entries.
>
> `validate`: Run basic validity checks.
>
> `falsify`: The full 100-point Popperian test suite.
>
> `stats`: Print category distribution, repo breakdown, quality histogram.
>
> `inspect`: Interactive TUI browser—arrow keys to navigate entries.
>
> `sample`: Print N random entries to stdout.
>
> `publish`: Generate parquet files and upload to HuggingFace.
>
> Each command is a clap subcommand. Type-safe arguments, helpful error messages."

**Live demo:**
```bash
# Interactive inspection
cargo run inspect --input corpus.json

# Quick sample
cargo run sample --input corpus.json --count 3
```

---

## CONCLUSION (30 seconds)

> "That's the corpus crate architecture—700 lines of Rust that extracts, filters, validates, and publishes ML training data.
>
> Key design decisions: syn for AST parsing not regex, deterministic UUIDs for deduplication, 100-point falsification for quality, alimentar for pure-Rust HuggingFace integration.
>
> The result: a reproducible pipeline that produces falsification-tested training data. Same inputs, same outputs, every time."

---

## TIMING SUMMARY

| Section | Duration |
|---------|----------|
| Intro | 0:20 |
| Part 1: CorpusEntry | 0:45 |
| Part 2: Extractor | 1:00 |
| Part 3: Filter | 0:45 |
| Part 4: Validator | 1:00 |
| Part 5: Publisher | 0:45 |
| Part 6: CLI | 0:45 |
| Conclusion | 0:30 |
| **Total** | **~6:00** |

---

## KEY CODE LOCATIONS

| Concept | File | Lines |
|---------|------|-------|
| CorpusEntry struct | `src/entry.rs` | 15-45 |
| Category enum | `src/entry.rs` | 50-70 |
| AST extraction | `src/extractor.rs` | 180-250 |
| Quality scoring | `src/extractor.rs` | 320-380 |
| Validity checks | `src/filter.rs` | 45-120 |
| Category balancing | `src/filter.rs` | 200-280 |
| Falsification criteria | `src/validator.rs` | 50-200 |
| Score breakdown | `src/validator.rs` | 220-280 |
| HF publishing | `src/publisher.rs` | 80-150 |
| CLI commands | `src/bin/cli.rs` | 30-100 |

---

## LIVE DEMO COMMANDS

```bash
# 1. Show entry structure
cat corpus/src/entry.rs | head -50

# 2. Extract from one repo (fast)
cargo run extract --repo https://github.com/sharkdp/fd --output demo.json --max-per-repo 10

# 3. Run falsification
cargo run falsify --input demo.json

# 4. Show stats
cargo run stats --input demo.json

# 5. Interactive browse
cargo run inspect --input demo.json

# 6. Sample entries
cargo run sample --input demo.json --count 2
```

---

## PRESENTER NOTES

- Have `corpus/` directory open in editor before starting
- Pre-build with `cargo build --release` to avoid compile waits
- Keep a small demo.json (10-20 entries) for fast iteration
- The `inspect` command is visually impressive—use it
- Emphasize "pure Rust, no Python" multiple times
- If asked about performance: extraction takes ~30s for 10 repos
- Key differentiator: syn parsing vs regex scraping
