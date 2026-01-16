# Interactive Demos

Rust-based demos for understanding ML concepts. All demos support dual output modes:
- **TUI** (default): Interactive terminal UI with colors
- **stdout**: Plain text for CI/scripts (`--stdout` flag)

## Quick Start

```bash
cd demos
make help          # Show all demos
make all           # Run lint + tests + coverage (95%+)
```

## Week 1: Parameter-Efficient Fine-Tuning

| Demo | Command | Description |
|------|---------|-------------|
| [Scalar vs SIMD vs GPU](week1/src/bin/README.md#demo-1-scalar-vs-simd-vs-gpu) | `make demo-scalar-simd-gpu` | When each compute backend wins |
| [Training vs Inference](week1/src/bin/README.md#demo-2-training-vs-inference) | `make demo-training-vs-inference` | Why inference is sequential |

## Quality Gates

All demos require:
- **95%+ test coverage**
- **Clippy clean** (no warnings)
- **Pre-commit hooks** enforce `make all` passes

```bash
make lint      # Clippy checks
make test-fast # Unit tests
make coverage  # Coverage report
```

## Architecture

```
demos/
├── Makefile           # Top-level targets
├── Cargo.toml         # Workspace root
└── week1/
    ├── Cargo.toml     # Week 1 crate
    └── src/
        ├── lib.rs                    # Module exports
        ├── scalar_simd_gpu.rs        # Demo 1 implementation
        ├── training_vs_inference.rs  # Demo 2 implementation
        └── bin/
            ├── README.md             # Demo documentation
            ├── scalar_simd_gpu.rs    # Demo 1 entry point
            └── training_vs_inference.rs  # Demo 2 entry point
```

## Adding New Demos

1. Create `week{N}/src/{demo_name}.rs` with implementation + tests
2. Add `pub mod {demo_name};` to `lib.rs`
3. Create `week{N}/src/bin/{demo_name}.rs` entry point
4. Add `[[bin]]` to `Cargo.toml`
5. Add targets to `Makefile`
6. Update `README.md` table
7. Ensure `make all` passes (95%+ coverage)
