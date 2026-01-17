//! Inference Pipeline Demo Binary
//!
//! Shows step-by-step data flow through an LLM:
//! TOKENIZE → EMBED → TRANSFORMER → LM_HEAD → SAMPLE → DECODE
//!
//! Usage:
//!   demo-inference-pipeline              # Normal run (TUI)
//!   demo-inference-pipeline --stdout     # CI mode
//!   demo-inference-pipeline --error      # Simulate APR-TOK-001

use clap::Parser;
use week1_demos::{inference_pipeline, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-inference-pipeline")]
#[command(about = "Step-by-step data flow through an LLM")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,

    /// Inject an error (APR-TOK-001 simulation)
    #[arg(long)]
    error: bool,
}

fn main() {
    let args = Args::parse();
    let prompt = "def add(x,y):";
    let trace = inference_pipeline::run_pipeline(prompt, args.error);

    if args.demo.use_tui() {
        inference_pipeline::render_tui(&trace);
    } else {
        inference_pipeline::print_stdout(&trace);
    }
}
