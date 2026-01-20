//! Full Fine-Tuning Cost Demo
//!
//! Shows memory requirements for training all parameters.
//!
//! Usage:
//!   demo-full-finetune-cost           # Normal run
//!   demo-full-finetune-cost --stdout  # CI mode

use clap::Parser;
use week2_demos::{full_finetune_cost, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-full-finetune-cost")]
#[command(about = "Why full fine-tuning is expensive")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let result = full_finetune_cost::run();

    if args.demo.use_tui() {
        full_finetune_cost::render_tui(&result);
    } else {
        full_finetune_cost::print_stdout(&result);
    }
}
