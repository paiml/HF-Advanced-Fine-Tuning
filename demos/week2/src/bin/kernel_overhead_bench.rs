//! Kernel Overhead Benchmark Binary
//!
//! Falsification experiment for the "Kernel Launch Overhead" hypothesis.
//!
//! Usage:
//!   cargo run --bin kernel_overhead_bench
//!   cargo run --bin kernel_overhead_bench -- --layers 4 --hidden 128

use clap::Parser;
use week2_demos::kernel_overhead_bench::{self, BenchConfig};

#[derive(Parser)]
#[command(name = "kernel_overhead_bench")]
#[command(about = "Benchmark kernel launch overhead vs compute time")]
struct Args {
    /// Hidden dimension size
    #[arg(long, default_value = "64")]
    hidden: usize,

    /// Sequence length
    #[arg(long, default_value = "32")]
    seq_len: usize,

    /// Number of transformer layers
    #[arg(long, default_value = "2")]
    layers: usize,
}

fn main() {
    let args = Args::parse();

    let config = BenchConfig {
        hidden_size: args.hidden,
        seq_len: args.seq_len,
        num_layers: args.layers,
    };

    let report = kernel_overhead_bench::run(config.clone());
    kernel_overhead_bench::print_results(&config, &report);
}
