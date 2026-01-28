# Course 4: Advanced Fine-Tuning with Sovereign AI Stack

**3 Weeks × 3 Lessons = 9 Lessons Total (26 Videos)**

---

## Overview

- **Duration:** 3 Weeks | 12 Hours | 26 Videos
- **Stack:** entrenar (training), realizar (inference), aprender (ML)
- **Hub:** HuggingFace integration via `apr import/export`
- **Model:** Qwen2.5-Coder (tiered by hardware)
- **Inference:** apr CLI (pure Rust, GGUF native)

---

## Model Tiers

Select based on available hardware:

| Tier | Model | VRAM | Use Case |
|------|-------|------|----------|
| Tiny | Qwen2.5-Coder-0.5B-Instruct | ~2GB | Quick iteration, debugging |
| **Small** | Qwen2.5-Coder-1.5B-Instruct | ~4GB | **Colab free tier (T4 16GB)** |
| Medium | Qwen2.5-Coder-7B-Instruct | ~8GB | Production fine-tuning |
| Large | Qwen2.5-Coder-32B-Instruct | ~20GB | Full capability |

**Default: Small (1.5B)** — Fits Colab free tier, full instruction-following, fast iteration.

```bash
# Verify setup with apr CLI
apr showcase --tier small --auto-verify
```

---

## Module 1: ML Foundations & Compute (Week 1)

**Module Description:**
This module establishes the foundational knowledge required for understanding fine-tuning at a deep level. Learners will explore core ML concepts including parameters, VRAM constraints, and gradients, then progress to understanding data shapes and their mapping to hardware. The module concludes with transformer architecture fundamentals and the inference pipeline, preparing learners for the technical depth required in subsequent weeks.

**Learning Objectives:**
By the end of this module, learners will be able to:
- Explain the five core building blocks of fine-tuning: parameters, VRAM, gradients, fine-tuning, and rank
- Differentiate between scalar, vector, and matrix operations and their optimal hardware mappings
- Compare training parallelism versus inference bottlenecks caused by softmax and layer normalization
- Describe the six-step transformer inference pipeline: tokenize, embed, transform, LM head, sample, decode
- Identify where the 80% of parameters live within a transformer architecture

---

### Lesson 1.1: Core Concepts & Data Shapes

| Filename | Title |
|----------|-------|
| `1.1.1-core-concepts.mp4` | Core Concepts: Parameters, VRAM, Gradients, and LoRA |
| `1.1.2-data-shapes.mp4` | Data Shapes: Scalars, Vectors, and Matrices |

**Key Concepts:**
- Parameters as "knobs on a mixing board" - models have billions of tunable weights
- VRAM as workspace limitation - bigger models require bigger GPU memory
- Gradients as the "compass pointing downhill" for optimization
- LoRA rank as matrix compression - 16M numbers reduced to 131K through low-rank factorization
- Scalar operations for CPU control flow, vectors for SIMD, matrices for GPU parallelization
- Hardware mapping: CPU handles scalar, SIMD handles 4-8 width vectors, GPU handles thousands of parallel matrix operations

---

### Lesson 1.2: ML Foundations & Hardware Compute

| Filename | Title |
|----------|-------|
| `1.2.1-ml-foundations.mp4` | ML Foundations: Neural Networks, Transformers, and Loss Functions |
| `1.2.2-scalar-simd-gpu-demo.mp4` | Scalar vs SIMD vs GPU Compute Demo |
| `1.2.3-training-vs-inference.mp4` | Training vs Inference: Parallelism and Bottlenecks |

**Key Concepts:**
- Neural network layers: input, hidden, output with weighted connections and activations
- Pre-training (expensive, internet-scale) vs fine-tuning (cheap, task-specific)
- Loss function as "how wrong the model is" - gradient descent minimizes loss
- GPU penalty for small workloads due to memory transfer latency
- SIMD optimal for medium datasets, GPU optimal for large matrix multiplication (512×512+)
- Training processes all tokens simultaneously (parallel), inference generates one token at a time (sequential)
- Softmax and layer normalization require global reduction - fundamental inference bottlenecks

---

### Lesson 1.3: Transformer Architecture & Pipeline

| Filename | Title |
|----------|-------|
| `1.3.1-llm-architecture.mp4` | LLM Architecture: Where Parameters Live |
| `1.3.2-transformer-pipeline-demo.mp4` | Building a Toy Transformer Pipeline Demo |
| `1.3.3-bpe-tokenization.mp4` | BPE vs Word Tokenization Theory |

**Key Concepts:**
- Transformer contains 80% of parameters but 100% of intelligence
- Tokenizer (0 params), Embeddings (15%), FFN+QKV (80%), LM Head (tied), Sampler (0 params)
- Six-step pipeline: tokenize → embed → transform → LM head → sample → decode
- Temperature controls sampling creativity vs determinism
- BPE subword tokenization handles unknown words gracefully (infinite coverage)
- Word tokenization has fatal OOV flaw - 100% meaning loss for unseen words
- BPE trades token count for coverage - more tokens but never "I don't know"

---

## Module 2: Transformer Internals & LoRA Introduction (Week 2)

**Module Description:**
This module dives deep into the internal mechanisms of transformers, covering tokenization implementation, the attention mechanism with QKV projections, and feed-forward networks where two-thirds of model parameters reside. The module bridges into fine-tuning by introducing LoRA fundamentals, showing how to train only 0.1% of parameters while achieving full fine-tuning results.

**Learning Objectives:**
By the end of this module, learners will be able to:
- Implement and compare BPE vs word tokenizers, understanding subword merge rules
- Explain the attention mechanism: Query (what am I looking for), Key (what do I have), Value (what info do I provide)
- Describe how softmax converts arbitrary scores to valid probabilities summing to one
- Analyze feed-forward network expansion/compression and GELU non-linearity
- Apply LoRA concepts: freezing base model, training A×B matrix diff, rank selection for task complexity

---

### Lesson 2.1: Tokenization & Attention

| Filename | Title |
|----------|-------|
| `2.1.1-bpe-tokenizer-demo.mp4` | BPE vs Word Tokenizer Implementation Demo |
| `2.1.2-attention-theory.mp4` | How Attention Works: QKV and Softmax |
| `2.1.3-attention-demo.mp4` | Attention Mechanism Step-by-Step Demo |

**Key Concepts:**
- BPE learns character pair merges from corpus frequency (TH, THE, AT, CAT)
- Word tokenizer: closed vocabulary, fails on novel words; BPE: open vocabulary, decomposes gracefully
- "scattered" → Word: UNKNOWN; BPE: S-CAT-T-E-R-E-D (finds "cat" inside)
- Q (search query) × K (book titles) → attention scores → softmax → probabilities
- Softmax normalizes any numbers to 0-1 range summing to 1
- Attention matrix: each row is one token's attention distribution across all tokens
- Without softmax: arbitrary values, uninterpretable; with softmax: valid probability distribution

---

### Lesson 2.2: Feed-Forward Networks

| Filename | Title |
|----------|-------|
| `2.2.1-ffn-theory.mp4` | Feed-Forward Networks: Where Thinking Happens |
| `2.2.2-ffn-lemonade-analogy.mp4` | FFN Lemonade Analogy: Non-Linearity Explained |
| `2.2.3-ffn-lemonade-demo.mp4` | FFN Lemonade Demo: Taste Test Implementation |
| `2.2.4-week1-summary.mp4` | Week 1 Transformer Foundations Summary |

**Key Concepts:**
- Attention gathers context ("what ingredients"), FFN processes understanding ("makes the recipe")
- FFN: expand (768→3072), GELU non-linearity, compress back (3072→768)
- Two-thirds of transformer parameters live in FFN - primary fine-tuning target
- Lemonade analogy: attention selects ingredients (50% lemon, 30% water, 20% sugar), FFN creates the flavor
- Non-linearity enables learning complex relationships - linear algebra alone insufficient
- Transformer layer = Attention + FFN, repeated N times
- Full pipeline: string → token → embedding → attention → FFN → LM head → sample → string

---

### Lesson 2.3: Introduction to LoRA

| Filename | Title |
|----------|-------|
| `2.3.1-lora-concepts.mp4` | LoRA Key Concepts: Learning the Diff |
| `2.3.2-lora-diff-training.mp4` | LoRA: Train the Diff, Not the Whole Model |
| `2.3.3-lora-rank-demo.mp4` | LoRA Rank Ablation Demo |

**Key Concepts:**
- LoRA: "Don't rewrite the whole file, just learn a diff"
- Freeze base model (quantized 4-bit read-only), add trainable bypass with A×B matrices
- 7 billion parameters → 7 million trainable (0.1%), fits in 8GB VRAM
- A compresses rank, B expands back: ΔW = A × B
- Rank selection matches task complexity: rank 4 (one trick), rank 8 (one skill like CLI docs), rank 16+ (multiple skills)
- Target attention layers (Q, K, V, O projections) - model already knows code, just steering style
- Base brain + diff = customized output in your voice

---

## Module 3: QLoRA & Corpus Engineering (Week 3)

**Module Description:**
This module covers the complete production fine-tuning workflow from quantization techniques through corpus creation and publication. Learners will understand how QLoRA combines 4-bit quantization with LoRA adapters for 7× memory reduction, then build quality training datasets using AST parsing, falsification testing, and proper train/validation/test splits. The module concludes with HuggingFace publishing workflows.

**Learning Objectives:**
By the end of this module, learners will be able to:
- Explain quantization trade-offs: 16-bit (perfect), 8-bit (1% loss), 4-bit (3-5% loss compensated by LoRA)
- Configure QLoRA to fit 7B parameter models in 4GB VRAM
- Design training corpora using AST extraction with function signatures paired to documentation
- Apply Popperian falsification: data integrity, syntactic validity, semantic validity, distribution checks
- Publish datasets to HuggingFace with proper splits (80/10/10) and dataset cards

---

### Lesson 3.1: Quantization & QLoRA

| Filename | Title |
|----------|-------|
| `3.1.1-quantization.mp4` | Quantization: Fewer Bits, Less Precision |
| `3.1.2-qlora-demo.mp4` | QLoRA Demo: Memory-Efficient Fine-Tuning |
| `3.1.3-fine-tuning-pipeline.mp4` | Complete Fine-Tuning Pipeline Overview |

**Key Concepts:**
- Quantization: 16-bit (65K values, perfect) → 8-bit (256 values, ~1% loss) → 4-bit (16 values, 3-5% loss)
- Weight 0.2385: 16-bit exact, 8-bit rounds to 0.24, 4-bit rounds to 0.2
- QLoRA: 4-bit base (imperfect but close), full-precision LoRA adapters learn to correct rounding errors
- 7B model: 28GB → 4GB VRAM (7× reduction), fits RTX 4090 or laptop GPU
- Training config: rank 8, alpha 16, target QKV+O projections, 65K trainable parameters
- Evaluation metrics: structural checks (format), content accuracy (no hallucinations), BLEU score
- Production commands: `apr import` → `entrenar train` → `apr merge` → `apr run`

---

### Lesson 3.2: HuggingFace Corpus Publishing

| Filename | Title |
|----------|-------|
| `3.2.1-dataset-card.mp4` | HuggingFace Dataset Card Overview |
| `3.2.2-corpus-pipeline.mp4` | Corpus Creation Pipeline: Extraction to Publication |

**Key Concepts:**
- Dataset card: 100 high-quality instruction-response pairs from 7 Rust CLI repositories
- Source distribution intentional: ripgrep 38% (best docs), clap 28%, bat, tokio, serde
- Extraction: AST parsing (not regex) pairs function signatures with rust doc comments
- Quality scoring 0-1 for each entry, threshold filtering for training data
- Category distribution: function docs 29%, argument docs 14%, examples 11%, error handling
- Parquet format: columnar storage optimized for ML workloads
- Dataset splits: 80% train, 10% validation, 10% test, stratified by category

---

### Lesson 3.3: Corpus Engineering & Implementation

| Filename | Title |
|----------|-------|
| `3.3.1-corpus-architecture.mp4` | Corpus Crate Architecture Deep Dive |
| `3.3.2-corpus-demo.mp4` | Corpus Creation CLI Demo |
| `3.3.3-week2-summary.mp4` | Week 2 Fine-Tuning Summary |

**Key Concepts:**
- CorpusEntry struct: UUID, category enum, input/output content, provenance (repo, commit, path, line), quality score
- Four pipeline modules: extractor (AST parsing), filter (7 quality gates), validator (falsification), publisher (parquet + HF)
- 12 CLI commands: clone, extract, falsify, inspect, publish, stats, human-review
- Popperian falsification: data integrity, syntactic validity, semantic validity, distribution, reproducibility
- Threshold passing: 85% minimum across all falsification tests
- Sovereign AI workflow: clone → extract → falsify → publish, runs entirely offline
- Key insight: quality data (100 curated examples) beats quantity (10,000 noisy ones)

---

## Video Mapping Summary

| Lesson | Filename | Title |
|--------|----------|-------|
| 1.1 | `1.1.1-core-concepts.mp4` | Core Concepts: Parameters, VRAM, Gradients, and LoRA |
| 1.1 | `1.1.2-data-shapes.mp4` | Data Shapes: Scalars, Vectors, and Matrices |
| 1.2 | `1.2.1-ml-foundations.mp4` | ML Foundations: Neural Networks, Transformers, and Loss Functions |
| 1.2 | `1.2.2-scalar-simd-gpu-demo.mp4` | Scalar vs SIMD vs GPU Compute Demo |
| 1.2 | `1.2.3-training-vs-inference.mp4` | Training vs Inference: Parallelism and Bottlenecks |
| 1.3 | `1.3.1-llm-architecture.mp4` | LLM Architecture: Where Parameters Live |
| 1.3 | `1.3.2-transformer-pipeline-demo.mp4` | Building a Toy Transformer Pipeline Demo |
| 1.3 | `1.3.3-bpe-tokenization.mp4` | BPE vs Word Tokenization Theory |
| 2.1 | `2.1.1-bpe-tokenizer-demo.mp4` | BPE vs Word Tokenizer Implementation Demo |
| 2.1 | `2.1.2-attention-theory.mp4` | How Attention Works: QKV and Softmax |
| 2.1 | `2.1.3-attention-demo.mp4` | Attention Mechanism Step-by-Step Demo |
| 2.2 | `2.2.1-ffn-theory.mp4` | Feed-Forward Networks: Where Thinking Happens |
| 2.2 | `2.2.2-ffn-lemonade-analogy.mp4` | FFN Lemonade Analogy: Non-Linearity Explained |
| 2.2 | `2.2.3-ffn-lemonade-demo.mp4` | FFN Lemonade Demo: Taste Test Implementation |
| 2.2 | `2.2.4-week1-summary.mp4` | Week 1 Transformer Foundations Summary |
| 2.3 | `2.3.1-lora-concepts.mp4` | LoRA Key Concepts: Learning the Diff |
| 2.3 | `2.3.2-lora-diff-training.mp4` | LoRA: Train the Diff, Not the Whole Model |
| 2.3 | `2.3.3-lora-rank-demo.mp4` | LoRA Rank Ablation Demo |
| 3.1 | `3.1.1-quantization.mp4` | Quantization: Fewer Bits, Less Precision |
| 3.1 | `3.1.2-qlora-demo.mp4` | QLoRA Demo: Memory-Efficient Fine-Tuning |
| 3.1 | `3.1.3-fine-tuning-pipeline.mp4` | Complete Fine-Tuning Pipeline Overview |
| 3.2 | `3.2.1-dataset-card.mp4` | HuggingFace Dataset Card Overview |
| 3.2 | `3.2.2-corpus-pipeline.mp4` | Corpus Creation Pipeline: Extraction to Publication |
| 3.3 | `3.3.1-corpus-architecture.mp4` | Corpus Crate Architecture Deep Dive |
| 3.3 | `3.3.2-corpus-demo.mp4` | Corpus Creation CLI Demo |
| 3.3 | `3.3.3-week2-summary.mp4` | Week 2 Fine-Tuning Summary |

**Total: 26 videos → 9 lessons → 3 modules**

---

## Module Summary Table

| Module | Title | Lessons | Videos | Focus |
|--------|-------|---------|--------|-------|
| 1 | ML Foundations & Compute | 3 | 8 | Core concepts, data shapes, hardware mapping, training vs inference, transformer architecture |
| 2 | Transformer Internals & LoRA Introduction | 3 | 10 | BPE tokenization, attention mechanism, feed-forward networks, LoRA fundamentals |
| 3 | QLoRA & Corpus Engineering | 3 | 8 | Quantization, QLoRA, corpus extraction, falsification, HuggingFace publishing |

---

## Labs

| Week | Lab | Tier | Description |
|------|-----|------|-------------|
| 1 | `lab-transformer-trace` | Tiny | Trace token through full pipeline |
| 1 | `lab-attention-viz` | Tiny | Visualize attention weights |
| 2 | `lab-cli-help-data` | — | Collect CLI help from Rust tools |
| 2 | `lab-lora-cli-help` | Small | LoRA fine-tune on CLI help task |
| 2 | `lab-qlora-cli-help` | Tiny | QLoRA 4-bit on CLI help task |
| 2 | `lab-rank-ablation` | Small | Compare r=4,8,16,32 on CLI help |
| 3 | `lab-sft-chat` | Small | SFT on code instruction dataset |
| 3 | `lab-dpo-preference` | Small | DPO with code preference pairs |

### Lab Workflow

```bash
# 1. Import model from HuggingFace
apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o base.apr

# 2. Train with entrenar (Rust-native LoRA/QLoRA)
entrenar train --model base.apr --config labs/qlora.toml --output adapter.apr

# 3. Merge adapter
apr merge base.apr adapter.apr --strategy lora -o merged.apr

# 4. Inference with apr (pure Rust, no Python deps)
apr run merged.apr --prompt "def fibonacci(n):"

# 5. Export to HuggingFace Hub
apr export merged.apr --format gguf -o model.gguf
apr push paiml/my-fine-tuned-qwen --file model.gguf
```

---

## Prerequisites

- Transformers fundamentals (Course 1)
- Dataset handling (Course 2)
- **Colab free tier (T4 16GB)** or local GPU with 4GB+ VRAM

## Resources

### Sovereign AI Stack

- [aprender](https://crates.io/crates/aprender) — ML library with HF Hub integration
- [entrenar](https://crates.io/crates/entrenar) — Training: autograd, LoRA/QLoRA, quantization
- [realizar](https://crates.io/crates/realizar) — Inference engine (GGUF, SafeTensors, APR)
- [apr-cli](https://crates.io/crates/apr-cli) — CLI for import/export/run/serve
- [batuta](https://crates.io/crates/batuta) — Stack orchestration

### HuggingFace

- [Qwen2.5-Coder Models](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [HF Hub API](https://huggingface.co/docs/hub)
