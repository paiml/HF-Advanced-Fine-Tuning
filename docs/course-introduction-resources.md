# Course 4: Advanced Fine-Tuning — Introduction & Resources

## Course Introduction

This course teaches production-grade fine-tuning using the **Sovereign AI Stack**—a pure Rust alternative to Python-based ML tooling. You will learn to fine-tune Qwen2.5-Coder models using LoRA and QLoRA techniques, build quality training corpora with Popperian falsification, and deploy models without Python dependencies.

**What makes this course different:**
- **No Python.** All ML operations use Rust crates from crates.io
- **Production-first.** Real code, real models, real deployment
- **Scientific rigor.** Falsification-based validation, not just "it works"
- **Hardware-aware.** Explicit trade-offs for consumer GPUs (8-24GB VRAM)

**By the end, you will:**
1. Understand transformer internals at the tensor level (attention, FFN, tokenization)
2. Apply LoRA/QLoRA to fine-tune 0.5B-7B models on consumer hardware
3. Build and validate training corpora using AST extraction and quality gates
4. Contribute to production Rust ML libraries with extreme TDD practices

---

## Course Repository

**GitHub:** [https://github.com/paiml/HF-Advanced-Fine-Tuning](https://github.com/paiml/HF-Advanced-Fine-Tuning)

```bash
git clone https://github.com/paiml/HF-Advanced-Fine-Tuning.git
cd HF-Advanced-Fine-Tuning
```

### Repository Structure

```
HF-Advanced-Fine-Tuning/
├── demos/
│   ├── week1/          # Transformer foundations demos
│   └── week2/          # LoRA/QLoRA demos
├── corpus/             # Rust CLI documentation corpus project
├── labs/               # Training configurations (YAML)
├── docs/
│   ├── images/         # SVG diagrams (1920x1080)
│   ├── scripts/        # Demo walkthrough scripts
│   └── specifications/ # Technical specifications
└── Makefile            # Build and run targets
```

---

## Key Demo Sections

### Week 1: Transformer Foundations

| Demo | Command | Purpose |
|------|---------|---------|
| **Scalar vs SIMD vs GPU** | `make demo-scalar-simd-gpu` | When GPU wins vs loses (transfer overhead) |
| **Training vs Inference** | `make demo-training-vs-inference` | Why training is parallel, inference is sequential |
| **Inference Pipeline** | `make demo-inference-pipeline` | 6-step flow: tokenize → embed → transform → head → sample → decode |
| **BPE vs Word** | `make demo-bpe-vs-word` | Subword tokenization handles unknown words |
| **Attention** | `make demo-attention` | QKV projections and softmax normalization |
| **Feed-Forward** | `make demo-feed-forward` | Expand → GELU → Contract (2/3 of params live here) |

### Week 2: Parameter-Efficient Fine-Tuning

| Demo | Command | Purpose |
|------|---------|---------|
| **Full Fine-Tune Cost** | `make demo-full-finetune-cost` | Why training all params needs 60GB+ for 7B |
| **LoRA Math** | `make demo-lora-math` | W' = W + A×B decomposition visualization |
| **QLoRA** | `make demo-qlora` | 4-bit quantization + LoRA adapters |
| **Rank Ablation** | `make demo-lora-rank-ablation` | r=4,8,16,32,64 trade-offs |
| **CLI Help Train** | `make demo-cli-help-train` | Fine-tune for CLI documentation |
| **LoRA Merge** | `make demo-lora-merge` | Merge adapter back to base model |

### Corpus Project

| Command | Purpose |
|---------|---------|
| `cargo run -- clone-sources` | Clone 7 Rust CLI repositories |
| `cargo run -- extract` | Parse AST, extract fn signatures + doc comments |
| `cargo run -- falsify` | Run 100-point Popperian falsification suite |
| `cargo run -- publish` | Push validated corpus to HuggingFace Hub |
| `cargo run -- inspect` | Interactive TUI corpus browser |

---

## Sovereign AI Stack

All course tooling uses the Sovereign AI Stack—pure Rust ML infrastructure:

| Crate | Version | Purpose | Crates.io |
|-------|---------|---------|-----------|
| **trueno** | 0.14+ | SIMD/GPU tensor operations | [trueno](https://crates.io/crates/trueno) |
| **aprender** | 0.24+ | ML library, HuggingFace Hub integration | [aprender](https://crates.io/crates/aprender) |
| **entrenar** | 0.5+ | Training: autograd, LoRA/QLoRA, quantization | [entrenar](https://crates.io/crates/entrenar) |
| **realizar** | 0.6+ | Inference engine (GGUF, SafeTensors, APR) | [realizar](https://crates.io/crates/realizar) |
| **alimentar** | 0.2+ | Data processing, corpus management | [alimentar](https://crates.io/crates/alimentar) |
| **apr-cli** | latest | CLI: import, export, merge, serve | [apr-cli](https://crates.io/crates/apr-cli) |

### Installation

```bash
# Install the apr CLI
cargo install apr-cli

# Verify installation
apr --version

# Pull a model from HuggingFace
apr pull Qwen/Qwen2.5-Coder-0.5B-Instruct
```

---

## GitHub Repositories

### Course & Demos

| Repository | Description |
|------------|-------------|
| [paiml/HF-Advanced-Fine-Tuning](https://github.com/paiml/HF-Advanced-Fine-Tuning) | **This course repository** — demos, labs, corpus project |

### Sovereign AI Stack (Core)

| Repository | Description |
|------------|-------------|
| [paiml/trueno](https://github.com/paiml/trueno) | SIMD-accelerated tensor operations, GPU compute |
| [paiml/aprender](https://github.com/paiml/aprender) | ML library with .apr format and HF Hub client |
| [paiml/entrenar](https://github.com/paiml/entrenar) | Training framework — LoRA, QLoRA, autograd |
| [paiml/realizar](https://github.com/paiml/realizar) | Inference engine — GGUF, SafeTensors support |
| [paiml/alimentar](https://github.com/paiml/alimentar) | Data pipelines, corpus extraction, HF datasets |

### Corpus Source Repositories

These repositories provide training data for the Rust CLI documentation corpus:

| Repository | Contribution | Quality |
|------------|--------------|---------|
| [BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep) | 38% | Best-in-class documentation |
| [clap-rs/clap](https://github.com/clap-rs/clap) | 28% | Argument parser, extensive examples |
| [sharkdp/bat](https://github.com/sharkdp/bat) | 10% | Cat clone with syntax highlighting |
| [sharkdp/fd](https://github.com/sharkdp/fd) | 8% | Find alternative |
| [tokio-rs/tokio](https://github.com/tokio-rs/tokio) | 8% | Async runtime |
| [serde-rs/serde](https://github.com/serde-rs/serde) | 5% | Serialization framework |
| [dtolnay/syn](https://github.com/dtolnay/syn) | 3% | Rust parser for procedural macros |

---

## Models Used

| Model | Parameters | VRAM | Use Case |
|-------|------------|------|----------|
| [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) | 0.5B | ~2GB | Quick iteration, CI testing |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 1.5B | ~4GB | **Default** — Colab T4, consumer GPUs |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 7B | ~8GB | Production fine-tuning |
| [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 32B | ~20GB | Full capability |

### Model Operations

```bash
# Import from HuggingFace
apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o base.apr

# Fine-tune with LoRA
entrenar train --model base.apr --config labs/qwen2.5-coder-1.5b-finetune.yaml

# Merge adapter
apr merge base.apr adapter.apr --strategy lora -o merged.apr

# Run inference
apr run merged.apr --prompt "/// Parses command-line arguments"

# Export to GGUF for llama.cpp compatibility
apr export merged.apr --format gguf -o model.gguf
```

---

## Lab Configurations

Located in `labs/`:

| Config | Model | Purpose |
|--------|-------|---------|
| `tiny_model_config.json` | Custom 2-layer | Kernel overhead testing |
| `itp-ent-tiny.yaml` | Tiny dummy | Falsification experiments |
| `qwen2.5-coder-1.5b-finetune.yaml` | Qwen 1.5B | Production fine-tuning (RTX 4090) |

---

## SVG Diagrams

All diagrams are 1920×1080 (16:9) with dark theme (`#0f172a` background).

### Week 1 Diagrams (`docs/images/week1/`)

| Diagram | Topic |
|---------|-------|
| `core-concepts-finetuning.svg` | Parameters, VRAM, gradients, LoRA overview |
| `data-shapes-backends.svg` | Scalar → SIMD → GPU mapping |
| `ml-foundations.svg` | Neural networks, transformers, loss |
| `training-vs-inference.svg` | Parallel training vs sequential inference |
| `transformers-architecture.svg` | Full transformer block diagram |
| `attention-mechanism.svg` | QKV projections, softmax |
| `feed-forward.svg` | FFN expand/contract with GELU |
| `bpe-vs-word.svg` | Tokenization comparison |
| `week1-cheatsheet.svg` | Quick reference summary |

### Week 2 Diagrams (`docs/images/week2/`)

| Diagram | Topic |
|---------|-------|
| `week2-peft-intro.svg` | Parameter-efficient fine-tuning intro |
| `week2-why-lora.svg` | LoRA efficiency explanation |
| `week2-lora-bypass.svg` | Frozen base + trainable adapter |
| `week2-quantization.svg` | 16-bit → 8-bit → 4-bit trade-offs |
| `fine-tuning-pipeline.svg` | End-to-end training workflow |
| `corpus-pipeline.svg` | Extraction → Filter → Validate → Publish |
| `corpus-code-architecture.svg` | Corpus crate internal design |
| `week2-cheatsheet.svg` | Quick reference summary |

---

## Reference Papers

| Paper | Topic | Link |
|-------|-------|------|
| LoRA | Low-Rank Adaptation of Large Language Models | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| QLoRA | Efficient Finetuning of Quantized LLMs | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| DoRA | Weight-Decomposed Low-Rank Adaptation | [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) |
| Attention | Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| BPE | Neural Machine Translation of Rare Words | [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) |

---

## Prerequisites

### Required

- **Rust 1.75+** — `rustup update stable`
- **Git** — For cloning repositories
- **4GB+ VRAM** — GTX 1070 minimum, RTX 3060+ recommended

### Recommended

- **CUDA 12.0+** — For GPU acceleration
- **24GB VRAM** — RTX 4090 for comfortable 7B fine-tuning
- **Colab Pro** — Alternative to local GPU

### Verify Setup

```bash
# Check Rust version
rustc --version  # Should be 1.75+

# Check CUDA (optional)
nvcc --version

# Clone and build course repo
git clone https://github.com/paiml/HF-Advanced-Fine-Tuning.git
cd HF-Advanced-Fine-Tuning
cargo build --release

# Run first demo
make demo-scalar-simd-gpu
```

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/paiml/HF-Advanced-Fine-Tuning.git
cd HF-Advanced-Fine-Tuning

# 2. Build all demos
make build

# 3. Run Week 1 demos (transformer foundations)
make demo-scalar-simd-gpu
make demo-training-vs-inference
make demo-inference-pipeline

# 4. Run Week 2 demos (LoRA/QLoRA)
make demo-lora-math
make demo-qlora
make demo-lora-rank-ablation

# 5. Build corpus project
cd corpus && cargo build --release

# 6. Extract documentation corpus
cargo run --release -- extract --repo ripgrep

# 7. Run falsification tests
cargo run --release -- falsify
```

---

## Support

- **Course Forum:** [Discussions](https://github.com/paiml/HF-Advanced-Fine-Tuning/discussions)
- **Bug Reports:** [Issues](https://github.com/paiml/HF-Advanced-Fine-Tuning/issues)
- **Stack Overflow:** Tag `sovereign-ai-stack` or `entrenar`

---

## License

MIT License — See [LICENSE](../LICENSE)

## Citation

```bibtex
@misc{hf-advanced-finetuning,
  title={Advanced Fine-Tuning with Sovereign AI Stack},
  author={PAIML},
  year={2025},
  howpublished={\url{https://github.com/paiml/HF-Advanced-Fine-Tuning}}
}
```
