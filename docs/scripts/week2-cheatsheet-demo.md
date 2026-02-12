# Week 2 Cheat Sheet Demo Script

**Duration:** ~4 minutes
**Visual:** `docs/images/week2/week2-cheatsheet.svg`
**Format:** Summary/wrap-up presentation

---

## INTRO (15 seconds)

> "Let's wrap up Week 2 with a cheat sheet covering everything we've learned—from LoRA fundamentals to corpus engineering to HuggingFace publishing."

---

## BOX 1: LoRA Fundamentals (30 seconds)

**[Point to top-left green box]**

> "LoRA decomposes weight updates into low-rank matrices. The formula: W-prime equals W plus B times A.
>
> With rank 8, we train just 0.01% of parameters. Alpha controls scaling—alpha 16 with rank 8 gives a scaling factor of 2.
>
> Target modules are typically q_proj and v_proj in the attention layers. The insight: low rank captures task-specific adaptations while the base model stays frozen."

---

## BOX 2: QLoRA (30 seconds)

**[Point to top-middle purple box]**

> "QLoRA combines 4-bit quantization with LoRA adapters.
>
> A 7B model drops from 28GB to 4GB VRAM—a 7x reduction. Base weights are frozen in NF4 format, while LoRA weights stay in FP16 for precision.
>
> The insight: quantize the base for memory efficiency, keep adapters in full precision for training quality. This lets you fine-tune 70B models on consumer GPUs."

---

## BOX 3: Corpus Extraction (30 seconds)

**[Point to top-right blue box]**

> "Extraction uses the syn crate for AST parsing—not regex scraping.
>
> Input is the function signature: name, parameters, return type. Output is the rustdoc comments.
>
> UUID v5 generates deterministic IDs from content, enabling deduplication across runs. Same content always produces the same UUID."

---

## BOX 4: Quality Filtering (30 seconds)

**[Point to bottom-left orange box]**

> "Seven hard quality gates filter candidates:
>
> Score at least 0.4, tokens between 10 and 500, lines under 100 characters. Balanced delimiters, no control characters, valid UTF-8, input-output ratio between 1 and 15.
>
> Plus: no single repository can exceed 38% of the corpus. The insight: 80% of candidates get filtered out. Quality gates matter more than volume."

---

## BOX 5: Popperian Falsification (30 seconds)

**[Point to bottom-middle red box]**

> "Our 100-point falsification suite tries to break the corpus:
>
> Data Integrity and Syntactic Validity: 20 points each. Semantic Validity: 20 points. Distribution and Reproducibility: 15 points each. Quality Metrics: 10 points.
>
> Pass threshold is 85. We consistently score 96. The insight: try to falsify, not verify. Surviving falsification attempts builds justified confidence."

---

## BOX 6: HuggingFace Publishing (30 seconds)

**[Point to bottom-right teal box]**

> "The alimentar crate handles publishing—pure Rust, no Python SDK.
>
> Parquet splits: 80% train, 10% validation, 10% test. Stratified by category so each split has the same distribution.
>
> Auto-generates the dataset card with license, statistics, usage examples, and BibTeX citation. Fully deterministic: same input corpus produces identical output."

---

## FOOTER: The Big Picture (45 seconds)

**[Point to dark footer section]**

> "The complete pipeline from source code to production model:
>
> Extract with syn AST parsing. Filter through 7 quality gates. Falsify with our 100-point suite. Publish to HuggingFace.
>
> Then for fine-tuning: Quantize to 4-bit NF4, train LoRA adapters with rank 8 and alpha 16, merge back into the base model.
>
> Key metrics along the bottom: 0.01% parameters trained, 4GB VRAM for a 7B model, 96/100 falsification score, 80% filtered out, 100% pure Rust.
>
> The Sovereign AI Stack: entrenar for training, alimentar for publishing, corpus for data engineering, trueno for compute."

---

## CONCLUSION (15 seconds)

> "That's Week 2—PEFT fundamentals and corpus engineering. From curated source code to falsification-tested datasets to efficient fine-tuning. Pure Rust, fully reproducible, production-ready."

---

## TIMING SUMMARY

| Section | Duration |
|---------|----------|
| Intro | 0:15 |
| Box 1: LoRA Fundamentals | 0:30 |
| Box 2: QLoRA | 0:30 |
| Box 3: Corpus Extraction | 0:30 |
| Box 4: Quality Filtering | 0:30 |
| Box 5: Falsification | 0:30 |
| Box 6: HuggingFace | 0:30 |
| Footer: Big Picture | 0:45 |
| Conclusion | 0:15 |
| **Total** | **~4:15** |

---

## PRESENTER NOTES

- This is a wrap-up—assume audience has seen the detailed content
- Move briskly through boxes, emphasize key numbers
- The footer pipeline is the payoff—spend time here
- Key metrics to hammer: 0.01%, 4GB, 96/100, 80% filtered, pure Rust
- Color coding matches earlier diagrams: green=LoRA, purple=QLoRA, blue=extract, orange=filter, red=falsify, teal=publish
- End with "pure Rust, fully reproducible"—the differentiator
- If time is tight, abbreviate boxes 3-6 and expand on footer
