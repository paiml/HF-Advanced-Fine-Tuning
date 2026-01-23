<div align="center">

<p align="center">
  <img src=".github/hero.svg" alt="Advanced Fine-Tuning" width="800">
</p>

<h1 align="center">HF-Advanced-Fine-Tuning</h1>

<p align="center">
  <b>Course 4: Advanced Fine-Tuning with Sovereign AI Stack</b>
</p>

<p align="center">
  <a href="https://github.com/paiml/HF-Advanced-Fine-Tuning/actions/workflows/ci.yml"><img src="https://github.com/paiml/HF-Advanced-Fine-Tuning/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct"><img src="https://img.shields.io/badge/Model-Qwen2.5--Coder-blue" alt="Model"></a>
  <a href="https://crates.io/crates/apr-cli"><img src="https://img.shields.io/badge/Format-.apr-purple" alt="APR Format"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

</div>

---

Fine-tune Qwen2.5-Coder models using the **Sovereign AI Stack** — pure Rust tools for privacy-preserving ML with pure Rust inference engine.

## Installation

```bash
# Install Sovereign AI Stack tools
cargo install entrenar realizar apr-cli pmat

# Or use make setup
make setup

# Verify installation
entrenar --version
apr --version
```

## Usage

```bash
# Import model from HuggingFace
apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o base.apr

# Fine-tune with LoRA
entrenar train --model base.apr --config labs/qlora.toml -o adapter.apr

# Merge adapter with base model
apr merge base.apr adapter.apr --strategy lora -o merged.apr

# Run inference
apr run merged.apr --prompt "def fibonacci(n):"

# Serve as API
apr serve merged.apr --port 8080
```

## Published Model

The fine-tuned model is available on HuggingFace Hub:

```bash
# Download and run the fine-tuned model
apr pull paiml/rust-cli-docs-qwen
apr run paiml/rust-cli-docs-qwen --prompt "How do I use clap for CLI arguments?"
```

| Model | Format | Size | Description |
|-------|--------|------|-------------|
| [paiml/rust-cli-docs-qwen](https://huggingface.co/paiml/rust-cli-docs-qwen) | `.apr` | 6.6 GB | Qwen2 fine-tuned on Rust CLI documentation |

## Quick Start

```bash
# Install tools
make setup

# Import model from HuggingFace
apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o base.apr

# Fine-tune with LoRA
entrenar train --model base.apr --config labs/qlora.toml -o adapter.apr

# Merge and run
apr merge base.apr adapter.apr --strategy lora -o merged.apr
apr run merged.apr --prompt "def fibonacci(n):"
```

## Model Tiers

| Tier | Model | VRAM | Throughput | Use Case |
|------|-------|------|------------|----------|
| Tiny | Qwen2.5-Coder-0.5B | ~2GB | 1000 tok/s | Quick iteration |
| **Small** | Qwen2.5-Coder-1.5B | ~4GB | **788 tok/s** | **Colab free tier** |
| Medium | Qwen2.5-Coder-7B | ~8GB | 400 tok/s | Production |
| Large | Qwen2.5-Coder-32B | ~20GB | 150 tok/s | Full capability |

**Default: Small (1.5B)** — Fits Google Colab free tier (T4 16GB), validated at 788 tok/s.

## Course Structure

### Week 1: Parameter-Efficient Fine-Tuning
- LoRA fundamentals and rank selection
- QLoRA with 4-bit NF4 quantization
- PEFT adapters and merging strategies

### Week 2: Alignment Training
- Supervised Fine-Tuning (SFT) with ChatML
- Direct Preference Optimization (DPO)
- Advanced variants: IPO, KTO, ORPO

### Week 3: Reinforcement Learning
- PPO fundamentals and RLHF pipeline
- Training stability and evaluation
- Deployment with `apr serve`

## Stack Components

```
┌─────────────────────────────────────────────────────────┐
│                    Course Tools                          │
├─────────────────────────────────────────────────────────┤
│  entrenar     │  realizar     │  apr-cli    │  pmat     │
│  (Training)   │  (Inference)  │  (CLI)      │  (Quality)│
├─────────────────────────────────────────────────────────┤
│              aprender (ML Library)                       │
├─────────────────────────────────────────────────────────┤
│              trueno (SIMD/GPU Compute)                   │
└─────────────────────────────────────────────────────────┘
```

| Component | Description |
|-----------|-------------|
| [entrenar](https://crates.io/crates/entrenar) | Training: autograd, LoRA/QLoRA, quantization |
| [realizar](https://crates.io/crates/realizar) | Inference engine for GGUF/SafeTensors/APR |
| [apr-cli](https://crates.io/crates/apr-cli) | CLI: import, export, run, serve, merge |
| [aprender](https://crates.io/crates/aprender) | ML library with HF Hub integration |
| [pmat](https://crates.io/crates/pmat) | Quality gates and ComputeBrick scoring |

## The `.apr` Format

Native model format optimized for the Sovereign AI Stack:

- **Security**: AES-256-GCM encryption, Ed25519 signatures
- **Performance**: Zero-copy mmap loading, 600x faster than pickle
- **Compression**: LZ4/ZSTD with quantization (int4, int8, fp16)
- **Interop**: Export to GGUF, SafeTensors for HuggingFace

```bash
# Import from HuggingFace
apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o model.apr

# Export to GGUF
apr export model.apr --format gguf -o model.gguf

# Push to HuggingFace Hub
apr push paiml/my-model --file model.apr
```

## Labs

| Week | Lab | Tier | Description |
|------|-----|------|-------------|
| 1 | `lab-qlora-qwen` | Tiny | QLoRA fine-tune Qwen2.5-Coder-0.5B |
| 1 | `lab-adapter-merge` | Small | Merge multiple LoRA adapters |
| 2 | `lab-sft-chat` | Small | SFT on code instruction dataset |
| 2 | `lab-dpo-preference` | Small | DPO with code preference pairs |
| 3 | `lab-ppo-rlhf` | Small | Full RLHF pipeline |
| 3 | `lab-evaluation` | Small | Model evaluation + benchmark |

## Interactive Demos

| Week | Demo | Description | Run |
|------|------|-------------|-----|
| 1 | Scalar vs SIMD vs GPU | Compare compute backends on small vs large operations | `cd demos && make demo-scalar-simd-gpu` |
| 1 | Training vs Inference | Why training is parallel but inference is sequential | `cd demos && make demo-training-vs-inference` |

**Demo 1 - Scalar vs SIMD vs GPU:**
- Small ops (dot product): SIMD wins, GPU loses to transfer overhead
- Large ops (matmul): GPU wins with massive parallelism

**Demo 2 - Training vs Inference:**
- Softmax: Global reduction - must sum all before normalizing any
- LayerNorm: Global reduction - must compute μ,σ from all dimensions
- Autoregressive: Token N+1 cannot exist until Token N is sampled

## ComputeBrick Profiling

Profile inference performance against baseline:

```bash
# Run profile
make profile

# All tiers
make profile-all

# View scores
make brick-score
```

**Targets:**
- Tiny (0.5B): 1000 tok/s
- Small (1.5B): pure Rust GGUF
- Medium (7B): 400 tok/s
- Large (32B): 150 tok/s

## Development

```bash
# Setup
make setup

# Lint shell scripts
make lint

# Run all quality checks
make check

# PMAT compliance
make compliance
```

## Prerequisites

- **Hardware**: Colab free tier (T4 16GB) or local GPU with 4GB+ VRAM
- **Courses**: Transformers fundamentals (Course 1), Dataset handling (Course 2)

## Resources

### Sovereign AI Stack
- [aprender](https://crates.io/crates/aprender) — ML library
- [entrenar](https://crates.io/crates/entrenar) — Training
- [realizar](https://crates.io/crates/realizar) — Inference
- [apr-cli](https://crates.io/crates/apr-cli) — CLI tools
- [batuta](https://crates.io/crates/batuta) — Orchestration

### HuggingFace
- [paiml/rust-cli-docs-qwen](https://huggingface.co/paiml/rust-cli-docs-qwen) — Fine-tuned model (6.6 GB APR format)
- [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) — Base model
- [HF Hub API](https://huggingface.co/docs/hub)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Write** tests for new functionality
4. **Ensure** all quality gates pass: `make check`
5. **Submit** a pull request

### Quality Standards

- PMAT compliance required: `pmat check`
- Test coverage minimum: 85%
- Documentation for public APIs

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built by <a href="https://paiml.com">Pragmatic AI Labs</a> — Sovereign AI for Everyone</sub>
</p>
