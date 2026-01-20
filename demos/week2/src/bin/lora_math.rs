//! LoRA Math Demo
//!
//! Shows the A×B low-rank decomposition.
//!
//! Usage:
//!   demo-lora-math           # Normal run
//!   demo-lora-math --stdout  # CI mode

use clap::Parser;
use week2_demos::{lora_math, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-lora-math")]
#[command(about = "LoRA: W' = W + A × B")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let result = lora_math::run();

    if args.demo.use_tui() {
        lora_math::render_tui(&result);
    } else {
        lora_math::print_stdout(&result);
    }
}
