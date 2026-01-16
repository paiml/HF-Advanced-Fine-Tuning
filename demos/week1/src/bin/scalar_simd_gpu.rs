//! Demo: Scalar vs SIMD vs GPU
//!
//! Shows two scenarios:
//! 1. Small dot product - SIMD wins (GPU transfer overhead)
//! 2. Large matmul - GPU wins (compute-bound)
//!
//! Usage:
//!   demo-scalar-simd-gpu              # TUI mode (default)
//!   demo-scalar-simd-gpu --stdout     # stdout mode (CI)

use clap::Parser;
use week1_demos::{scalar_simd_gpu, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-scalar-simd-gpu")]
#[command(about = "Compare Scalar, SIMD, and GPU compute backends")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,

    /// Dot product vector size (small = SIMD wins)
    #[arg(long, default_value = "10000")]
    dot_size: usize,

    /// Matrix size for matmul (large = GPU wins)
    #[arg(long, default_value = "512")]
    matmul_size: usize,

    /// Number of iterations per benchmark
    #[arg(short, long, default_value = "20")]
    iterations: usize,
}

fn main() {
    let args = Args::parse();
    let results = scalar_simd_gpu::run(args.dot_size, args.matmul_size, args.iterations);

    if args.demo.use_tui() {
        scalar_simd_gpu::render_tui(&results);
    } else {
        scalar_simd_gpu::print_stdout(&results);
    }
}
