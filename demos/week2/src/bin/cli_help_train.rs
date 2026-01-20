//! CLI Help Training Data Demo Binary
use clap::Parser;
use week2_demos::{cli_help, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-cli-help-train")]
#[command(about = "CLI help training data format")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,
}

fn main() {
    let args = Args::parse();
    let result = cli_help::run();
    if args.demo.use_tui() {
        cli_help::render_tui(&result);
    } else {
        cli_help::print_stdout(&result);
    }
}
