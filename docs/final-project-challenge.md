# Final Project Challenge: Qwen Fine-Tuning Enhancement

**1 Module × 1 Lesson = Capstone Project**

---

## Module 4: Final Project Challenge (Week 4)

**Module Description:**
This capstone project challenges learners to apply all concepts from the course by running, analyzing, and enhancing a real Qwen2.5-Coder fine-tuning pipeline in the entrenar Rust training library. Students will work with production-grade code, implement meaningful improvements, and validate their changes using the library's comprehensive test suite. This hands-on project bridges theory to practice, demonstrating mastery of LoRA, QLoRA, corpus engineering, and the Sovereign AI stack.

**Learning Objectives:**
By the end of this module, learners will be able to:
- Execute a complete Qwen2.5-Coder fine-tuning pipeline using pure Rust tooling
- Analyze LoRA configuration trade-offs (rank, alpha, target modules) in production code
- Implement and validate enhancements to an existing training pipeline
- Apply Popperian falsification principles to verify code correctness
- Contribute production-quality code following extreme TDD practices (95%+ coverage)

---

### Lesson 4.1: Capstone Project - Enhance Qwen Fine-Tuning Pipeline

| Phase | Title | Duration |
|-------|-------|----------|
| Phase 1 | Environment Setup & Baseline Run | 1-2 hours |
| Phase 2 | Code Analysis & Enhancement Selection | 2-3 hours |
| Phase 3 | Implementation & Testing | 4-6 hours |
| Phase 4 | Validation & Documentation | 2-3 hours |

**Total Estimated Time:** 10-14 hours

---

## Phase 1: Environment Setup & Baseline Run

### 1.1 Prerequisites

Verify your environment meets these requirements:

```bash
# Rust toolchain (1.75+)
rustc --version

# CUDA toolkit (optional, for GPU acceleration)
nvcc --version

# Clone entrenar if not present
cd ~/src
git clone https://github.com/paiml/entrenar.git
cd entrenar

# Verify build
cargo build --release
```

### 1.2 Run Baseline Fine-Tuning Example

Execute the existing Qwen2.5-Coder fine-tuning example:

```bash
cd ~/src/entrenar

# Run the test generation fine-tuning example
cargo run --release --example finetune_test_gen

# Or run the real end-to-end example (requires model download)
apr pull Qwen/Qwen2.5-Coder-0.5B-Instruct
cargo run --release --example finetune_real
```

### 1.3 Baseline Deliverables

Document your baseline run:
- [ ] Screenshot of successful compilation
- [ ] Training output (loss curves, final metrics)
- [ ] Memory usage observations
- [ ] Any errors encountered and how resolved

---

## Phase 2: Code Analysis & Enhancement Selection

### 2.1 Key Files to Study

| File | Purpose | Lines |
|------|---------|-------|
| `examples/finetune_test_gen.rs` | Qwen2.5-Coder test generation fine-tuning | ~670 |
| `examples/finetune_real.rs` | Real end-to-end fine-tuning with TUI | ~1100 |
| `src/lora/config.rs` | LoRA configuration and targeting | ~376 |
| `src/lora/qlora.rs` | QLoRA 4-bit quantization layer | ~180 |
| `crates/entrenar-lora/src/memory.rs` | Memory estimation and planning | ~340 |
| `crates/entrenar-lora/src/optimizer.rs` | Automatic rank optimization | ~426 |
| `src/transformer/config.rs` | Qwen2 architecture configuration | ~150 |

### 2.2 Enhancement Options

Select **ONE** enhancement from each tier (minimum: 1 Core + 1 Advanced):

#### Tier 1: Core Enhancements (Required - Pick One)

| Enhancement | Description | Difficulty |
|-------------|-------------|------------|
| **A. Rank Ablation Study** | Implement automated rank comparison (r=4,8,16,32,64) with metrics logging | Medium |
| **B. New Target Module Strategy** | Add `target_mlp_projections()` for gate/up/down FFN layers | Medium |
| **C. Learning Rate Finder** | Implement LR range test with automatic optimal LR selection | Medium |
| **D. Gradient Accumulation** | Add gradient accumulation for effective larger batch sizes | Medium |

#### Tier 2: Advanced Enhancements (Required - Pick One)

| Enhancement | Description | Difficulty |
|-------------|-------------|------------|
| **E. DoRA Implementation** | Implement Weight-Decomposed Low-Rank Adaptation | Hard |
| **F. Quantization-Aware LoRA** | Add QAT support during LoRA training | Hard |
| **G. Multi-Adapter Support** | Enable training multiple LoRA adapters simultaneously | Hard |
| **H. Curriculum Learning** | Implement difficulty-based training data ordering | Hard |

#### Tier 3: Bonus Enhancements (Optional)

| Enhancement | Description | Difficulty |
|-------------|-------------|------------|
| **I. TUI Dashboard Enhancement** | Add loss visualization graphs to monitor TUI | Medium |
| **J. Checkpoint Resume** | Implement training checkpoint save/resume | Medium |
| **K. Custom Corpus Integration** | Integrate your own Rust CLI corpus from Week 3 | Medium |
| **L. Benchmark Suite** | Create comprehensive benchmarks comparing methods | Hard |

### 2.3 Enhancement Selection Deliverable

Document your chosen enhancements:
- [ ] Core enhancement selected: ___
- [ ] Advanced enhancement selected: ___
- [ ] Bonus enhancement (optional): ___
- [ ] Brief justification for each choice (2-3 sentences)

---

## Phase 3: Implementation & Testing

### 3.1 Implementation Guidelines

Follow entrenar's coding standards:

```rust
// 1. All public functions require documentation
/// Calculates optimal LoRA rank for given VRAM constraint.
///
/// # Arguments
/// * `vram_gb` - Available GPU memory in gigabytes
/// * `model_params` - Total model parameters
///
/// # Returns
/// Optimal rank value (4, 8, 16, 32, or 64)
///
/// # Example
/// ```rust
/// let rank = calculate_optimal_rank(24.0, 500_000_000);
/// assert!(rank >= 4 && rank <= 64);
/// ```
pub fn calculate_optimal_rank(vram_gb: f64, model_params: u64) -> u32 {
    // Implementation
}

// 2. Unit tests for all new code
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_rank_low_vram() {
        let rank = calculate_optimal_rank(4.0, 500_000_000);
        assert_eq!(rank, 4); // Low VRAM → low rank
    }

    #[test]
    fn test_optimal_rank_high_vram() {
        let rank = calculate_optimal_rank(48.0, 500_000_000);
        assert_eq!(rank, 64); // High VRAM → can use higher rank
    }
}

// 3. Property-based tests for complex logic
#[cfg(test)]
mod proptests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn rank_always_valid(vram in 1.0f64..100.0, params in 1u64..10_000_000_000) {
            let rank = calculate_optimal_rank(vram, params);
            prop_assert!(rank >= 4);
            prop_assert!(rank <= 64);
            prop_assert!(rank.is_power_of_two() || rank == 4);
        }
    }
}
```

### 3.2 Quality Gates

Your implementation must pass all quality gates:

```bash
# 1. All tests pass
cargo test --all

# 2. No clippy warnings
cargo clippy --all-targets -- -D warnings

# 3. Code formatted
cargo fmt --check

# 4. Documentation builds
cargo doc --no-deps

# 5. Coverage check (target: 95%+)
cargo llvm-cov --all-features
```

### 3.3 Implementation Deliverables

- [ ] Feature branch created: `feature/your-enhancement-name`
- [ ] All new code has documentation
- [ ] Unit tests written (minimum 5 per enhancement)
- [ ] Property tests where applicable
- [ ] All quality gates pass

---

## Phase 4: Validation & Documentation

### 4.1 Popperian Falsification Tests

Apply falsification principles to your enhancement:

```rust
// Example falsification tests for rank ablation
#[cfg(test)]
mod falsification_tests {
    use super::*;

    /// Falsification: Higher rank should not decrease quality
    /// (if it does, our hypothesis about rank-quality relationship is wrong)
    #[test]
    fn falsify_rank_quality_monotonicity() {
        let results_r4 = train_with_rank(4);
        let results_r8 = train_with_rank(8);
        let results_r16 = train_with_rank(16);

        // Higher rank should maintain or improve quality
        assert!(results_r8.final_loss <= results_r4.final_loss * 1.1,
            "Rank 8 significantly worse than rank 4 - falsified!");
        assert!(results_r16.final_loss <= results_r8.final_loss * 1.1,
            "Rank 16 significantly worse than rank 8 - falsified!");
    }

    /// Falsification: Memory estimate should bound actual usage
    #[test]
    fn falsify_memory_estimation_accuracy() {
        let estimated = estimate_memory(rank, batch_size);
        let actual = measure_actual_memory(rank, batch_size);

        // Estimate should be within 20% of actual
        let error = (estimated - actual).abs() / actual;
        assert!(error < 0.2,
            "Memory estimation off by {:.1}% - falsified!", error * 100.0);
    }
}
```

### 4.2 Benchmark Results

Run and document benchmarks:

```bash
# Run benchmarks
cargo bench --bench lora_bench

# Generate comparison report
cargo run --release --example benchmark_report > results/benchmark.md
```

### 4.3 Final Deliverables Checklist

#### Code Deliverables
- [ ] Clean git history with meaningful commits
- [ ] All changes in feature branch
- [ ] PR-ready with description of changes

#### Documentation Deliverables
- [ ] `ENHANCEMENT.md` - Description of your enhancement (500-1000 words)
- [ ] API documentation for all public items
- [ ] Usage example in `examples/` directory

#### Validation Deliverables
- [ ] Falsification test results (all pass)
- [ ] Benchmark comparison (before/after)
- [ ] Memory usage comparison (if applicable)

#### Presentation Deliverables
- [ ] 5-minute demo video or live presentation
- [ ] Key metrics visualization
- [ ] Lessons learned summary

---

## Grading Rubric

| Category | Points | Criteria |
|----------|--------|----------|
| **Baseline Run** | 10 | Successfully ran existing example, documented output |
| **Code Quality** | 25 | Follows Rust idioms, documented, formatted, no warnings |
| **Test Coverage** | 25 | Unit tests, property tests, falsification tests, 95%+ coverage |
| **Enhancement Quality** | 25 | Meaningful improvement, correct implementation, benchmarked |
| **Documentation** | 15 | Clear ENHANCEMENT.md, API docs, usage examples |
| **Bonus** | +10 | Additional enhancement from Tier 3 |

**Total: 100 points (+10 bonus)**

| Grade | Points |
|-------|--------|
| A | 90-110 |
| B | 80-89 |
| C | 70-79 |
| D | 60-69 |
| F | <60 |

---

## Submission Instructions

### Option A: Pull Request (Recommended)

1. Fork the entrenar repository
2. Create feature branch from `main`
3. Implement enhancement with tests
4. Submit PR with description following template
5. Share PR link for review

### Option B: Archive Submission

1. Create archive of your changes:
```bash
git archive --format=zip HEAD -o enhancement-submission.zip
```
2. Include `ENHANCEMENT.md` at root
3. Submit via course portal

### Deadline

- **Submission Due:** End of Week 4
- **Late Policy:** -10 points per day, max 3 days late

---

## Resources

### Entrenar Documentation
- [Crate Documentation](https://docs.rs/entrenar)
- [Book/Guide](../entrenar/book/)
- [Examples](../entrenar/examples/)

### Reference Papers
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

### Sovereign AI Stack
- [aprender](https://crates.io/crates/aprender) - ML library
- [entrenar](https://crates.io/crates/entrenar) - Training library
- [realizar](https://crates.io/crates/realizar) - Inference engine
- [trueno](https://crates.io/crates/trueno) - SIMD compute

---

## Key Concepts Summary

| Concept | Application in Project |
|---------|----------------------|
| LoRA Rank Selection | Tier 1A: Rank ablation study |
| Target Module Strategy | Tier 1B: FFN layer targeting |
| QLoRA Memory Efficiency | Tier 2F: QAT during LoRA |
| Popperian Falsification | Phase 4: Validation tests |
| Corpus Engineering | Tier 3K: Custom corpus |
| TUI Monitoring | Tier 3I: Dashboard enhancement |

---

## FAQ

**Q: Can I use Python for any part?**
A: No. This course uses the Sovereign AI stack (pure Rust). All code must be Rust.

**Q: What if I don't have a GPU?**
A: The examples support CPU fallback. Use `--features cpu-fallback` and expect slower training.

**Q: Can I work in a team?**
A: Individual submissions only. You may discuss concepts but code must be your own.

**Q: What if my enhancement doesn't improve metrics?**
A: Document why. Negative results with proper analysis are valuable. Focus on correct implementation and thorough testing.

**Q: How do I get help?**
A:
1. Check entrenar documentation and examples
2. Review course materials from Weeks 1-3
3. Post in course forum with specific questions
4. Office hours available weekly
