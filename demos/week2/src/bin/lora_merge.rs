//! LoRA Merge Demo Binary
use clap::Parser;
use week2_demos::{lora_merge, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-lora-merge")]
#[command(about = "LoRA merge and deployment")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let result = lora_merge::run();
    if args.demo.use_tui() {
        lora_merge::render_tui(&result);
    } else {
        lora_merge::print_stdout(&result);
    }
}
