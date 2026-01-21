# Rust CLI Documentation Corpus - Makefile
# Toyota Way: Automation eliminates Muda (waste)
# Per Spec Section 11: Appendix Makefile Targets

.PHONY: all build test lint fmt clean help
.PHONY: corpus-clone-sources corpus-extract corpus-validate corpus-export corpus-build
.PHONY: corpus-falsify corpus-review corpus-stats
.PHONY: corpus-card corpus-publish corpus-inspect corpus-sample
.PHONY: dev pre-commit ci doc coverage

# Default target
all: build test lint

# Build the corpus CLI
build:
	cargo build --release

# Run all tests
test:
	cargo test

# Run tests with coverage (requires cargo-llvm-cov)
coverage:
	cargo llvm-cov --html
	@echo "Coverage report: target/llvm-cov/html/index.html"

# Run clippy lints
lint:
	cargo clippy -- -D warnings

# Format code
fmt:
	cargo fmt

# Format check
fmt-check:
	cargo fmt -- --check

# Clean build artifacts
clean:
	cargo clean
	rm -rf hf_output/
	rm -rf data/clones/
	rm -f corpus.parquet
	rm -f data/corpus/*.parquet

# ============================================================================
# Corpus Management (uses alimentar) - Per Spec Section 11
# ============================================================================

# Clone and pin source repositories
corpus-clone-sources:
	@echo "Cloning source repositories..."
	@mkdir -p data/clones
	cargo run --release -- clone-sources --output data/clones

# Extract documentation pairs (alimentar doctest)
corpus-extract:
	cargo run --release -- extract --defaults --output data/corpus/train.parquet

# Run all validation gates
corpus-validate:
	cargo run --release -- validate --input data/corpus/train.parquet --verbose

# Export to parquet format
corpus-export:
	cargo run --release -- export --input data/corpus/train.parquet --output data/corpus/

# Full pipeline (idempotent)
corpus-build: corpus-clone-sources corpus-extract corpus-validate corpus-export
	@echo "Corpus build complete."

# ============================================================================
# Quality Assurance (uses certeza) - Per Spec Section 11
# ============================================================================

# Run 100-point falsification
corpus-falsify:
	cargo run --release -- falsify --input data/corpus/train.parquet
	@echo "Running test suite as falsification..."
	cargo test -- --test-threads=1

# Generate human review sample
corpus-review:
	cargo run --release -- review --input data/corpus/train.parquet --sample 10 --output data/review/

# Print corpus statistics
corpus-stats:
	cargo run --release -- stats --input data/corpus/train.parquet

# ============================================================================
# Publication (uses alimentar hub) - Per Spec Section 11
# ============================================================================

# Generate HuggingFace dataset card
corpus-card:
	cargo run --release -- card --input data/corpus/train.parquet --output data/corpus/README.md

# Push to HuggingFace Hub via alimentar
corpus-publish:
	@test -n "$(HF_TOKEN)" || (echo "Error: HF_TOKEN not set" && exit 1)
	cargo run --release -- publish --input data/corpus/train.parquet --output hf_output --upload

# ============================================================================
# Development - Per Spec Section 11
# ============================================================================

# Interactive corpus browser (alimentar repl)
corpus-inspect:
	cargo run --release -- inspect --input data/corpus/train.parquet

# Print N random examples (default N=10)
N ?= 10
corpus-sample:
	cargo run --release -- sample --input data/corpus/train.parquet --count $(N)

# ============================================================================
# Legacy Aliases (for backwards compatibility)
# ============================================================================

extract: corpus-extract
validate: corpus-validate
stats: corpus-stats
publish: corpus-publish

# ============================================================================
# Development Workflow
# ============================================================================

# Full development cycle
dev: fmt lint test

# Pre-commit checks
pre-commit: fmt-check lint test

# CI pipeline
ci: fmt-check lint test coverage

# ============================================================================
# Documentation
# ============================================================================

# Generate documentation
doc:
	cargo doc --no-deps --open

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Rust CLI Documentation Corpus - Makefile"
	@echo "Per Spec Section 11: Appendix Makefile Targets"
	@echo ""
	@echo "Build & Test:"
	@echo "  make build              - Build release binary"
	@echo "  make test               - Run tests"
	@echo "  make coverage           - Run tests with coverage"
	@echo "  make lint               - Run clippy"
	@echo "  make fmt                - Format code"
	@echo "  make clean              - Clean artifacts"
	@echo ""
	@echo "Corpus Management (alimentar):"
	@echo "  make corpus-clone-sources - Clone and pin source repositories"
	@echo "  make corpus-extract       - Extract documentation pairs"
	@echo "  make corpus-validate      - Run all validation gates"
	@echo "  make corpus-export        - Export to parquet format"
	@echo "  make corpus-build         - Full pipeline (idempotent)"
	@echo ""
	@echo "Quality Assurance (certeza):"
	@echo "  make corpus-falsify       - Run 100-point falsification"
	@echo "  make corpus-review        - Generate human review sample"
	@echo "  make corpus-stats         - Print corpus statistics"
	@echo ""
	@echo "Publication (alimentar hub):"
	@echo "  make corpus-card          - Generate HuggingFace dataset card"
	@echo "  make corpus-publish       - Push to HuggingFace Hub"
	@echo ""
	@echo "Development:"
	@echo "  make corpus-inspect       - Interactive corpus browser"
	@echo "  make corpus-sample N=10   - Print N random examples"
	@echo "  make dev                  - Full dev cycle (fmt, lint, test)"
	@echo "  make pre-commit           - Pre-commit checks"
	@echo "  make ci                   - CI pipeline"
	@echo "  make doc                  - Generate documentation"
