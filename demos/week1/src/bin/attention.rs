//! Attention Mechanism Demo Binary
//!
//! Shows how self-attention works with toy Q/K/V matrices:
//! - Q = "What am I looking for?"
//! - K = "What do I contain?"
//! - V = "What info do I provide?"
//! - Softmax = converts scores → probabilities (0-1, sum to 1)
//!
//! Usage:
//!   demo-attention              # Normal run (TUI)
//!   demo-attention --stdout     # CI mode
//!   demo-attention --error      # Skip softmax (shows why it matters)

use clap::Parser;
use week1_demos::{attention, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-attention")]
#[command(about = "Self-attention mechanism with Q/K/V and softmax")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,

    /// Skip softmax (shows broken attention weights)
    #[arg(long)]
    error: bool,
}

fn main() {
    let args = Args::parse();
    let result = attention::run(args.error);

    if args.demo.use_tui() {
        attention::render_tui(&result);
    } else {
        attention::print_stdout(&result);
    }
}
