# Demos Makefile
# All: make all (lint && test-fast && coverage)
# Individual: make demo-scalar-simd-gpu

.PHONY: all lint test-fast coverage clean help
.PHONY: demo-scalar-simd-gpu demo-scalar-simd-gpu-stdout

# Default: full pipeline
all: lint test-fast coverage

help:
	@printf '=== Week 1 Demos ===\n\n'
	@printf 'Pipeline:\n'
	@printf '  make all           Run lint && test-fast && coverage (95%%+)\n'
	@printf '  make lint          Clippy checks\n'
	@printf '  make test-fast     Fast unit tests\n'
	@printf '  make coverage      Coverage report\n\n'
	@printf 'Demos:\n'
	@printf '  make demo-scalar-simd-gpu         TUI mode\n'
	@printf '  make demo-scalar-simd-gpu-stdout  stdout mode (CI)\n\n'
	@printf 'Demo shows:\n'
	@printf '  1. Dot product (small)  - SIMD wins, GPU loses (transfer overhead)\n'
	@printf '  2. Matrix multiply      - GPU wins (compute-bound)\n'

# =============================================================================
# Pipeline Targets
# =============================================================================

lint:
	@printf '=== Lint ===\n'
	@cargo clippy --manifest-path week1/Cargo.toml -- -D warnings

test-fast:
	@printf '=== Test (fast) ===\n'
	@cargo test --manifest-path week1/Cargo.toml --lib

coverage:
	@printf '=== Coverage ===\n'
	@cargo llvm-cov --manifest-path week1/Cargo.toml --lib --summary-only

clean:
	@cargo clean --manifest-path week1/Cargo.toml

# =============================================================================
# Individual Demos - TUI Mode (default)
# =============================================================================

demo-scalar-simd-gpu:
	@cargo run --manifest-path week1/Cargo.toml --release --bin demo-scalar-simd-gpu

# =============================================================================
# Individual Demos - Stdout Mode (for CI)
# =============================================================================

demo-scalar-simd-gpu-stdout:
	@cargo run --manifest-path week1/Cargo.toml --release --bin demo-scalar-simd-gpu -- --stdout
