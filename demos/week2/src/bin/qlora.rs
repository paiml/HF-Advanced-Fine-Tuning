//! QLoRA Demo Binary
use clap::Parser;
use week2_demos::{qlora, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-qlora")]
#[command(about = "QLoRA: 4-bit quantization + LoRA")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let result = qlora::run();
    if args.demo.use_tui() {
        qlora::render_tui(&result);
    } else {
        qlora::print_stdout(&result);
    }
}
