# Week 1 Demos

## Demo 1: Scalar vs SIMD vs GPU

**Run**: `make demo-scalar-simd-gpu`

Shows when each compute backend wins:
- **Small ops (dot product)**: SIMD wins, GPU loses to transfer overhead
- **Large ops (matmul)**: GPU wins with massive parallelism

| Operation | Winner | Why |
|-----------|--------|-----|
| Dot product (n=10K) | SIMD | No transfer, 8 floats/instruction |
| Matrix multiply (512×512) | GPU | O(n³) compute amortizes transfer |

---

## Demo 2: Training vs Inference

**Run**: `make demo-training-vs-inference`

Shows why training is parallel but inference is sequential.

### Qwen2.5-Coder-1.5B Assembly Line

**Config**: 1.5B params, 28 layers, hidden=1536, heads=12, vocab=151936

```
Input: "def fib"
       ↓
[Token IDs]        →  [3, 1025, 12456]           # 3 integers
       ↓
[Embedding]        →  [3, 1536]                  # 3 × 1536 floats
       ↓
┌──────────────────────────────────────────────┐
│  × 28 LAYERS                                  │
│                                               │
│  [LayerNorm]     →  μ,σ over 1536 dims       │ ← LAYERNORM #1
│        ↓                                      │
│  [Q,K,V proj]    →  [3, 1536] each           │
│        ↓                                      │
│  [Attention]                                  │
│    QK^T          →  [12, 3, 3] raw scores    │
│    ÷√128         →  [12, 3, 3] scaled        │
│    +mask         →  [12, 3, 3] masked        │
│    Softmax       →  [12, 3, 3] probs 0-1     │ ← SOFTMAX #1
│    ×V            →  [3, 1536]                │
│        ↓                                      │
│  [LayerNorm]     →  μ,σ over 1536 dims       │ ← LAYERNORM #2
│        ↓                                      │
│  [FFN]           →  [3, 1536]                │
│                                               │
└──────────────────────────────────────────────┘
       ↓
[Final LayerNorm]  →  μ,σ over 1536 dims        ← LAYERNORM #3
       ↓
[LM Head]          →  [151936] logits           # one per vocab word
       ↓
[Softmax]          →  [151936] probs sum=1      ← SOFTMAX #2
       ↓
[Sample]           →  token_id = 8234           # "onacci"
       ↓
REPEAT (autoregressive)                         ← AUTOREGRESSIVE
```

### ELI5 Each Operation

| Op | Where | Before | After | ELI5 |
|----|-------|--------|-------|------|
| **Softmax** | Attention (×28) | `[2.1, -0.5, 1.8]` raw scores | `[0.48, 0.04, 0.48]` probs | "How much to look at each word" - must sum to 1 |
| **Softmax** | Output | `[5.2, -1.0, ..., 3.1]` 151936 logits | `[0.002, 0.0001, ..., 0.001]` probs | "Pick next word" - must sum to 1 |
| **LayerNorm** | Pre-attention (×28) | `[0.5, -0.2, 0.8, ...]` 1536 floats | `[0.3, -1.2, 1.1, ...]` normalized | "Keep numbers from exploding" - mean=0, std=1 |
| **Autoregressive** | Outer loop | `["def", "fib"]` | `["def", "fib", "onacci"]` | "Can't write word 4 until word 3 exists" |

### Why Inference is Slow

```
One token = 28 layers × (2 layernorms + 1 softmax) + final softmax
          = 57 global reductions PER TOKEN

Generate "def fibonacci(n):" (7 tokens)
          = 7 × 57 = 399 global reductions
          = 399 "wait for everyone" barriers
```

**Training**: See all 7 tokens at once → 1 pass
**Inference**: Generate 1 token, wait, generate next → 7 passes

### The Three Bottlenecks

1. **Softmax** - Global reduction. Can't compute `exp(x)/sum` until you have `sum`.
2. **LayerNorm** - Global reduction. Can't normalize until you compute μ,σ from all 1536 dims.
3. **Autoregressive** - Fundamental constraint. Token N+1 literally cannot exist until token N is sampled.

**TL;DR**: Training 1T tokens = days. Generating 1K tokens = seconds but feels slow.
