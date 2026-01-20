//! The Lemonade Stand: FFN Demo
//!
//! Making lemonade as a metaphor for FFN:
//! - Attention gathers ingredients from the pantry
//! - FFN tastes and adjusts (expand → GELU taste test → contract)
//!
//! Usage:
//!   demo-feed-forward                  # Normal run (with taste test)
//!   demo-feed-forward --stdout         # CI mode
//!   demo-feed-forward --skip-taste     # Skip taste test (shows why GELU matters)

use clap::Parser;
use week1_demos::{feed_forward, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-feed-forward")]
#[command(about = "The Lemonade Stand: FFN from gathered to blended")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,

    /// Skip taste test (shows FFN collapses without GELU)
    #[arg(long)]
    skip_taste: bool,

    /// Alias for --skip-taste (backward compat)
    #[arg(long, hide = true)]
    error: bool,
}

fn main() {
    let args = Args::parse();
    let skip = args.skip_taste || args.error;
    let result = feed_forward::run(skip);

    if args.demo.use_tui() {
        feed_forward::render_tui(&result);
    } else {
        feed_forward::print_stdout(&result);
    }
}
