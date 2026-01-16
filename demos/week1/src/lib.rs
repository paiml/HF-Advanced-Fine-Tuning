//! Week 1: Parameter-Efficient Fine-Tuning Demos
//!
//! All demos support dual output modes:
//! - `--stdout` : Print results to stdout (CI/scripts)
//! - `--tui`    : Interactive terminal UI (default)

use clap::Parser;

pub mod scalar_simd_gpu;
pub mod training_vs_inference;

/// Common CLI args for all demos
#[derive(Parser, Debug, Clone, Default)]
pub struct DemoArgs {
    /// Output to stdout only (no TUI)
    #[arg(long)]
    pub stdout: bool,

    /// Run TUI mode (default)
    #[arg(long)]
    pub tui: bool,
}

impl DemoArgs {
    /// Create new DemoArgs
    #[must_use]
    pub fn new(stdout: bool, tui: bool) -> Self {
        Self { stdout, tui }
    }

    /// Returns true if TUI should be shown
    #[must_use]
    pub fn use_tui(&self) -> bool {
        !self.stdout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_args_default() {
        let args = DemoArgs::default();
        assert!(!args.stdout);
        assert!(!args.tui);
    }

    #[test]
    fn test_demo_args_new() {
        let args = DemoArgs::new(true, false);
        assert!(args.stdout);
        assert!(!args.tui);
    }

    #[test]
    fn test_use_tui_default() {
        let args = DemoArgs::default();
        assert!(args.use_tui());
    }

    #[test]
    fn test_use_tui_stdout_mode() {
        let args = DemoArgs::new(true, false);
        assert!(!args.use_tui());
    }

    #[test]
    fn test_use_tui_explicit_tui() {
        let args = DemoArgs::new(false, true);
        assert!(args.use_tui());
    }
}
