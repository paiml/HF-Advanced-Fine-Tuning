# Course 4: Advanced Fine-Tuning

## Overview
- 3 Weeks | 12 Hours | 5 Key Concepts
- Stack: entrenar (training), realizar (inference), aprender (ML)
- Hub: HuggingFace integration via `apr import/export`
- Model: Qwen2.5-Coder (tiered by hardware)
- Inference: apr CLI (pure Rust, GGUF native)

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

## Week 1: Transformer Foundations

### 1.1 Compute Fundamentals
- Scalar vs SIMD vs GPU execution
- Memory-bound vs compute-bound workloads
- When GPU wins (matmul) vs loses (small ops)
- **Demo:** `make demo-scalar-simd-gpu`

### 1.2 Training vs Inference
- Global reductions (softmax, layernorm)
- Sequential dependencies (autoregressive generation)
- Why token N+1 waits for token N
- **Demo:** `make demo-training-vs-inference`

### 1.3 Inference Pipeline
- 6-step flow: tokenize → embed → transform → lm_head → sample → decode
- String in, tensors in between, string out
- **Demo:** `make demo-inference-pipeline`

### 1.4 Tokenization
- Word vs BPE tokenization
- Why BPE never says "unknown"
- Subword decomposition
- **Demo:** `make demo-bpe-vs-word`

### 1.5 Attention Mechanism
- Q/K/V projections
- Softmax as probability distribution
- Attention = soft dictionary lookup
- **Demo:** `make demo-attention`

### 1.6 Feed-Forward Network
- The Lemonade Stand analogy
- Expand → GELU (taste test) → Contract
- Why 2/3 of params live in FFN
- **Demo:** `make demo-feed-forward`

---

## Week 2: Parameter-Efficient Fine-Tuning (PEFT)

**Task:** Fine-tune Qwen2.5-Coder to generate CLI help text for Rust tools.

**Why CLI help:**
- Narrow domain, consistent structure (clap-derived)
- 50-200 examples sufficient
- Easy to evaluate (structure, flags, descriptions)
- Meta: model learns to document your own tools

**Data sources:** apr-cli, pmat, bashrs, ripgrep, fd, bat, cargo subcommands

### 2.1 Full Fine-Tuning Baseline
- Why training all params is expensive
- Memory: model + gradients + optimizer states
- The problem: 7B model needs ~60GB for training
- **Demo:** `make demo-full-finetune-cost`

### 2.2 LoRA Fundamentals
- Low-rank decomposition: W + ΔW = W + A×B
- Rank selection (r=4, 8, 16, 32, 64)
- Target modules (q_proj, v_proj, k_proj, o_proj)
- 1000x fewer trainable params, same quality
- **Demo:** `make demo-lora-math`

### 2.3 QLoRA
- 4-bit NF4 quantization + LoRA
- Double quantization
- Fine-tune 70B on consumer GPU (24GB)
- **Demo:** `make demo-qlora`

### 2.4 Rank Selection
- r=8 vs r=64: capacity vs efficiency
- Task complexity determines optimal rank
- CLI help = narrow task = low rank sufficient
- **Demo:** `make demo-lora-rank-ablation`

### 2.5 Training on CLI Help
- Dataset: command + flags → help text
- Format: clap-style output
- Evaluation: structure match, flag coverage
- **Demo:** `make demo-cli-help-train`

### 2.6 Merging & Deployment
- Merge LoRA back into base model
- Export to GGUF via `apr export`
- Inference: `apr run` generates help for new commands
- **Demo:** `make demo-lora-merge`

---

## Week 3: Alignment Training

### 3.1 Supervised Fine-Tuning (SFT)
- Dataset formats (instruction, chat)
- ChatML template (Qwen2.5 native format)
- SFTTrainer configuration
- Packing for efficiency

### 3.2 Direct Preference Optimization (DPO)
- DPO loss function
- Beta parameter tuning
- DPO vs RLHF tradeoffs
- No reward model needed

### 3.3 RLHF Pipeline
- Reward modeling (Bradley-Terry)
- PPO fundamentals (policy gradient, GAE)
- Reference model management
- KL divergence constraints

### 3.4 Evaluation & Deployment
- Win rate metrics
- Human evaluation protocols
- Merged model export
- Deployment with `apr serve`

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
