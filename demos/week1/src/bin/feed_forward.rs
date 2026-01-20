//! Feed-Forward Network Demo Binary
//!
//! Shows why FFN is needed after attention:
//! - Attention mixes (weighted blend)
//! - FFN thinks (expand → GELU → contract)
//!
//! Usage:
//!   demo-feed-forward              # Normal run (TUI)
//!   demo-feed-forward --stdout     # CI mode
//!   demo-feed-forward --error      # Skip non-linearity (shows why GELU matters)

use clap::Parser;
use week1_demos::{feed_forward, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-feed-forward")]
#[command(about = "Feed-forward network: from gathered to understood")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,

    /// Skip non-linearity (shows FFN collapses to linear)
    #[arg(long)]
    error: bool,
}

fn main() {
    let args = Args::parse();
    let result = feed_forward::run(args.error);

    if args.demo.use_tui() {
        feed_forward::render_tui(&result);
    } else {
        feed_forward::print_stdout(&result);
    }
}
