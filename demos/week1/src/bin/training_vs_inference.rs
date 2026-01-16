//! Demo: Training vs Inference
//!
//! Shows why training is parallel but inference is sequential:
//! 1. Softmax - global reduction
//! 2. LayerNorm - global reduction
//! 3. Autoregressive - token N+1 depends on token N
//!
//! Usage:
//!   demo-training-vs-inference              # TUI mode (default)
//!   demo-training-vs-inference --stdout     # stdout mode (CI)

use clap::Parser;
use week1_demos::{training_vs_inference, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-training-vs-inference")]
#[command(about = "Why training is parallel but inference is sequential")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let results = training_vs_inference::run();

    if args.demo.use_tui() {
        training_vs_inference::render_tui(&results);
    } else {
        training_vs_inference::print_stdout(&results);
    }
}
