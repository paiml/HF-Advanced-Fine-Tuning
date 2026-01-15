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

## Week 1: Parameter-Efficient Fine-Tuning

### 1.1 LoRA Fundamentals
- Low-rank decomposition of weight updates
- Rank selection (r=8, 16, 32)
- Target modules (q_proj, v_proj, k_proj, o_proj)
- Qwen2.5-Coder architecture specifics

### 1.2 QLoRA
- 4-bit NF4 quantization
- Double quantization
- Paged optimizers for memory efficiency
- Memory calculation: 1.5B model in ~4GB

### 1.3 Quantization with bitsandbytes
- 8-bit vs 4-bit tradeoffs
- NF4 vs FP4 data types
- Memory footprint calculation
- Quantization-aware training considerations

### 1.4 PEFT Adapters
- LoRA, LoHA, LoKr variants
- Adapter merging strategies
- Multi-adapter inference
- Export to GGUF via `apr export`

---

## Week 2: Alignment Training

### 2.1 Supervised Fine-Tuning (SFT)
- Dataset formats (instruction, chat)
- ChatML template (Qwen2.5 native format)
- SFTTrainer configuration
- Packing for efficiency

### 2.2 Reward Modeling
- Preference data collection
- Bradley-Terry model
- Reward model training
- Code-specific reward signals

### 2.3 Direct Preference Optimization (DPO)
- DPO loss function
- Beta parameter tuning
- DPO vs RLHF tradeoffs
- Code quality preferences

### 2.4 Advanced DPO Variants
- IPO (Identity Preference Optimization)
- KTO (Kahneman-Tversky Optimization)
- ORPO (Odds Ratio Preference Optimization)

---

## Week 3: Reinforcement Learning

### 3.1 PPO Fundamentals
- Policy gradient methods
- Advantage estimation (GAE)
- Clipping and KL constraints

### 3.2 RLHF Pipeline
- Reference model management
- Value head architecture
- Reward shaping
- Code execution feedback

### 3.3 Training Stability
- KL divergence monitoring
- Learning rate scheduling
- Gradient accumulation
- Checkpoint management

### 3.4 Evaluation & Deployment
- Win rate metrics
- Human evaluation protocols
- Merged model export
- Deployment with `apr serve`

---

## Labs

| Week | Lab | Tier | Description |
|------|-----|------|-------------|
| 1 | `lab-qlora-qwen` | Tiny | QLoRA fine-tune Qwen2.5-Coder-0.5B (fast iteration) |
| 1 | `lab-adapter-merge` | Small | Merge multiple LoRA adapters |
| 2 | `lab-sft-chat` | Small | SFT on code instruction dataset |
| 2 | `lab-dpo-preference` | Small | DPO with code preference pairs |
| 3 | `lab-ppo-rlhf` | Small | Full RLHF pipeline |
| 3 | `lab-evaluation` | Small | Model evaluation + apr benchmark |

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
