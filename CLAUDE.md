# HF-Advanced-Fine-Tuning

Course 4: Advanced Fine-Tuning with Sovereign AI Stack

## Project Structure

```
docs/images/week{1,2,3}/  - SVG diagrams for each week
demos/week{1,2}/          - Rust demo binaries
labs/                     - Lab configurations (TOML)
```

## SVG Diagram Standards

All course diagrams follow these conventions:

- **Resolution**: 1920x1080 (16:9, optimized for 1080p displays)
- **Background**: Dark theme (`#0f172a` base, `#020617` header/footer)
- **Fonts**: Inter for UI text, JetBrains Mono for code
- **Color Palette**:
  - Red `#ef4444` - Problems, warnings
  - Purple `#7c3aed` - Original/baseline (full fine-tune)
  - Blue `#1e3a8a` - Frozen weights
  - Green `#22c55e` - LoRA/trainable adapters
  - Yellow `#eab308` - Operations (sum, merge)
  - Teal `#14b8a6` - Attention layers
  - Orange `#f59e0b` - FFN layers
- **Spacing**: Clear separation between sections, no overlapping elements
- **Text**: High contrast, minimum 16px for body, 20px+ for labels

## Week Topics

- **Week 1**: PEFT fundamentals (LoRA, QLoRA, rank selection)
- **Week 2**: Alignment (SFT, DPO, IPO, KTO, ORPO)
- **Week 3**: RLHF (PPO, deployment)

## Sovereign AI Stack

| Tool | Purpose |
|------|---------|
| entrenar | Training (autograd, LoRA/QLoRA) |
| realizar | Inference (GGUF/SafeTensors/APR) |
| apr-cli | CLI (import, export, merge, serve) |
| aprender | ML library |
| trueno | SIMD/GPU compute |

## Commands

```bash
# Build demos
cd demos && cargo build --release

# Run specific demo
cargo run --release --bin lora_math

# Lint
make lint

# Full check
make check
```
