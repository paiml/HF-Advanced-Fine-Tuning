//! CLI Help Evaluation Demo Binary
use clap::Parser;
use week2_demos::{eval_cli_help, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-eval-cli-help")]
#[command(about = "Evaluation framework for CLI help generation")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let result = eval_cli_help::run();
    if args.demo.use_tui() {
        eval_cli_help::render_tui(&result);
    } else {
        eval_cli_help::print_stdout(&result);
    }
}
