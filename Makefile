# HF-Advanced-Fine-Tuning Makefile
# Course 4: Advanced Fine-Tuning with Sovereign AI Stack
#
# bashrs compliance: make purify verified

.SUFFIXES:
.DELETE_ON_ERROR:
.PHONY: all help setup lint svg-lint test profile brick-score clean
.PHONY: docs serve compliance check
.PHONY: demos demo-test demo-scalar-simd-gpu demo-training-vs-inference
.PHONY: demo-inference-pipeline demo-bpe-vs-word demo-attention demo-feed-forward
.PHONY: demo-test-w2 demo-full-finetune-cost demo-lora-math demo-qlora
.PHONY: demo-lora-rank-ablation demo-cli-help-train demo-lora-merge demo-eval-cli-help

# Default target
all: lint compliance

# Help
help:
	@printf '=== HF-Advanced-Fine-Tuning ===\n\n'
	@printf 'Setup:\n'
	@printf '  make setup        Install required tools (apr-cli, pmat)\n\n'
	@printf 'Development:\n'
	@printf '  make lint         Lint shell scripts with bashrs\n'
	@printf '  make docs         Build documentation\n'
	@printf '  make serve        Serve docs locally\n\n'
	@printf 'Demos:\n'
	@printf '  make demos        List all available demos\n'
	@printf '  make demo-test    Run Week 1 demos in CI mode\n'
	@printf '  make demo-test-w2 Run Week 2 demos in CI mode\n'
	@printf '  make demo-X       Run specific demo (see: make demos)\n\n'
	@printf 'Profiling:\n'
	@printf '  make profile      Run ComputeBrick profile (small tier)\n'
	@printf '  make profile-tiny Run profile with tiny tier (0.5B)\n'
	@printf '  make profile-all  Run all tier profiles\n'
	@printf '  make brick-score  Calculate brick score from latest profile\n\n'
	@printf 'Quality:\n'
	@printf '  make compliance   Run pmat compliance check\n'
	@printf '  make check        Run all quality checks\n'
	@printf '  make clean        Remove generated files\n'

# Setup
setup:
	@printf 'Installing apr-cli...\n'
	cargo install apr-cli --locked || true
	@printf 'Installing pmat...\n'
	cargo install pmat --locked || true
	@printf 'Installing bashrs...\n'
	cargo install bashrs --locked || true
	@printf 'Initializing pmat hooks...\n'
	pmat hooks install || true
	pmat hooks cache init || true

# Week 1 Demos
DEMO_DIR := demos/week1

demos:
	@printf '=== Available Demos ===\n'
	@printf 'Week 1 - Foundations:\n'
	@printf '  make demo-scalar-simd-gpu      Compute parallelism\n'
	@printf '  make demo-training-vs-inference Forward vs backward pass\n'
	@printf '  make demo-inference-pipeline   6-step token flow\n'
	@printf '  make demo-bpe-vs-word          Tokenization comparison\n'
	@printf '  make demo-attention            Q/K/V + softmax\n'
	@printf '  make demo-feed-forward         FFN: gathered to understood\n'
	@printf '\nWeek 2 - PEFT (LoRA/QLoRA):\n'
	@printf '  make demo-full-finetune-cost   Why full fine-tune is expensive\n'
	@printf '  make demo-lora-math            LoRA A×B decomposition\n'
	@printf '  make demo-qlora                4-bit quantization + LoRA\n'
	@printf '  make demo-lora-rank-ablation   Rank selection guide\n'
	@printf '  make demo-cli-help-train       CLI help training data format\n'
	@printf '  make demo-lora-merge           Merge and deployment\n'
	@printf '  make demo-eval-cli-help        Evaluation framework\n'
	@printf '\nRun all: make demo-test (W1) | make demo-test-w2 (W2)\n'

demo-test:
	@printf '=== Running All Demos (CI Mode) ===\n'
	cd $(DEMO_DIR) && cargo run --bin demo-scalar-simd-gpu -- --stdout
	cd $(DEMO_DIR) && cargo run --bin demo-training-vs-inference -- --stdout
	cd $(DEMO_DIR) && cargo run --bin demo-inference-pipeline -- --stdout
	cd $(DEMO_DIR) && cargo run --bin demo-bpe-vs-word -- --stdout
	cd $(DEMO_DIR) && cargo run --bin demo-attention -- --stdout
	cd $(DEMO_DIR) && cargo run --bin demo-feed-forward -- --stdout
	@printf '=== All Demos Passed ===\n'

demo-scalar-simd-gpu:
	cd $(DEMO_DIR) && cargo run --bin demo-scalar-simd-gpu

demo-training-vs-inference:
	cd $(DEMO_DIR) && cargo run --bin demo-training-vs-inference

demo-inference-pipeline:
	cd $(DEMO_DIR) && cargo run --bin demo-inference-pipeline

demo-bpe-vs-word:
	cd $(DEMO_DIR) && cargo run --bin demo-bpe-vs-word

demo-attention:
	cd $(DEMO_DIR) && cargo run --bin demo-attention

demo-feed-forward:
	cd $(DEMO_DIR) && cargo run --bin demo-feed-forward

# Week 2 Demos
DEMO_DIR_W2 := demos/week2

demo-test-w2:
	@printf '=== Running Week 2 Demos (CI Mode) ===\n'
	cd $(DEMO_DIR_W2) && cargo run --bin demo-full-finetune-cost -- --stdout
	cd $(DEMO_DIR_W2) && cargo run --bin demo-lora-math -- --stdout
	cd $(DEMO_DIR_W2) && cargo run --bin demo-qlora -- --stdout
	cd $(DEMO_DIR_W2) && cargo run --bin demo-lora-rank-ablation -- --stdout
	cd $(DEMO_DIR_W2) && cargo run --bin demo-cli-help-train -- --stdout
	cd $(DEMO_DIR_W2) && cargo run --bin demo-lora-merge -- --stdout
	cd $(DEMO_DIR_W2) && cargo run --bin demo-eval-cli-help -- --stdout
	@printf '=== Week 2 Demos Passed ===\n'

demo-full-finetune-cost:
	cd $(DEMO_DIR_W2) && cargo run --bin demo-full-finetune-cost

demo-lora-math:
	cd $(DEMO_DIR_W2) && cargo run --bin demo-lora-math

demo-qlora:
	cd $(DEMO_DIR_W2) && cargo run --bin demo-qlora

demo-lora-rank-ablation:
	cd $(DEMO_DIR_W2) && cargo run --bin demo-lora-rank-ablation

demo-cli-help-train:
	cd $(DEMO_DIR_W2) && cargo run --bin demo-cli-help-train

demo-lora-merge:
	cd $(DEMO_DIR_W2) && cargo run --bin demo-lora-merge

demo-eval-cli-help:
	cd $(DEMO_DIR_W2) && cargo run --bin demo-eval-cli-help

# Lint shell scripts (exit 2 = error, exit 1 = warning, exit 0 = clean)
lint:
	@printf '=== Linting Shell Scripts ===\n'
	@bashrs lint brick/profile.sh; ret=$$?; if [ $$ret -eq 2 ]; then exit 1; fi
	@printf '=== Linting SVG Files ===\n'
	@./scripts/svg-lint.sh docs/images

# SVG linting only
svg-lint:
	@./scripts/svg-lint.sh docs/images

# Documentation
docs:
	@printf '=== Building Documentation ===\n'
	@if command -v mdbook >/dev/null 2>&1; then \
		mdbook build docs || true; \
	else \
		printf 'mdbook not installed. Install with: cargo install mdbook\n'; \
	fi

serve:
	@printf '=== Serving Documentation ===\n'
	@if command -v mdbook >/dev/null 2>&1; then \
		mdbook serve docs; \
	else \
		printf 'mdbook not installed. Install with: cargo install mdbook\n'; \
	fi

# Profiling
profile: profile-small

profile-tiny:
	@printf '=== Profiling Tiny Tier (0.5B) ===\n'
	./brick/profile.sh tiny

profile-small:
	@printf '=== Profiling Small Tier (1.5B) ===\n'
	./brick/profile.sh small

profile-medium:
	@printf '=== Profiling Medium Tier (7B) ===\n'
	./brick/profile.sh medium

profile-large:
	@printf '=== Profiling Large Tier (32B) ===\n'
	./brick/profile.sh large

profile-all: profile-tiny profile-small profile-medium profile-large

brick-score:
	@printf '=== ComputeBrick Score ===\n'
	@if ls brick/profiles/*.json 1>/dev/null 2>&1; then \
		LATEST=$$(ls -t brick/profiles/*.json | head -1); \
		pmat brick-score --input "$$LATEST"; \
	else \
		printf 'No profile data. Run: make profile\n'; \
	fi

# Quality
compliance:
	@printf '=== PMAT Compliance ===\n'
	pmat comply check

check: lint compliance
	@printf '=== All Checks Complete ===\n'

# Clean
clean:
	@printf '=== Cleaning Generated Files ===\n'
	rm -f brick/profiles/*.json
	rm -rf docs/book
