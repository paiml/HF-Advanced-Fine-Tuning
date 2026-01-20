//! Week 2: Parameter-Efficient Fine-Tuning Demos
//!
//! Educational demos showing PEFT concepts:
//! - Full fine-tune cost (why it's expensive)
//! - LoRA math (A×B low-rank decomposition)
//! - QLoRA (4-bit quantization + LoRA)
//! - Rank ablation (r=4 vs r=64)
//! - CLI help format (the training task)
//! - LoRA merge (combine adapters)
//!
//! Actual training: entrenar
//! Actual inference: apr run/serve/chat

use clap::Args;

pub mod full_finetune_cost;
pub mod lora_math;
pub mod qlora;
pub mod lora_rank;
pub mod cli_help;
pub mod lora_merge;

/// Common CLI args for all demos
#[derive(Args, Debug, Clone)]
pub struct DemoArgs {
    /// Print to stdout instead of TUI
    #[arg(long)]
    pub stdout: bool,
}

impl DemoArgs {
    pub fn use_tui(&self) -> bool {
        !self.stdout
    }
}
