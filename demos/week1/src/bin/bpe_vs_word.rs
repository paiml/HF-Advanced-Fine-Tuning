//! BPE vs Word Tokenization Demo Binary
//!
//! Shows why BPE is superior to word tokenizers:
//! - Word: fixed vocab, OOV → <UNK>
//! - BPE: subword merges, handles any input
//!
//! Usage:
//!   demo-bpe-vs-word              # Normal run (TUI)
//!   demo-bpe-vs-word --stdout     # CI mode
//!   demo-bpe-vs-word --error      # Show typo handling ("doge" vs "dog")

use clap::Parser;
use week1_demos::{bpe_vs_word, DemoArgs};

#[derive(Parser)]
#[command(name = "demo-bpe-vs-word")]
#[command(about = "Why BPE handles unknown words gracefully")]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,

    /// Use typo example ("doge" instead of unknown "rat")
    #[arg(long)]
    error: bool,
}

fn main() {
    let args = Args::parse();
    let results = bpe_vs_word::run(args.error);

    if args.demo.use_tui() {
        bpe_vs_word::render_tui(&results);
    } else {
        bpe_vs_word::print_stdout(&results);
    }
}
