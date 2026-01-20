//! LoRA Rank Ablation Demo Binary
use clap::Parser;
use week2_demos::{lora_rank, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-lora-rank-ablation")]
#[command(about = "LoRA rank selection: r=4 vs r=64")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let result = lora_rank::run();
    if args.demo.use_tui() {
        lora_rank::render_tui(&result);
    } else {
        lora_rank::print_stdout(&result);
    }
}
