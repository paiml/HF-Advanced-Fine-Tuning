//! CLI Help Training Data Demo
//!
//! Shows the format for fine-tuning on CLI help text:
//! - Input: command signature
//! - Output: clap-style help text
//!
//! Data sources: apr-cli, pmat, ripgrep, fd, bat, cargo

use std::fmt;

/// A CLI help training example
#[derive(Debug, Clone)]
pub struct HelpExample {
    pub tool: &'static str,
    pub command: &'static str,
    pub help_text: &'static str,
}

pub const EXAMPLES: &[HelpExample] = &[
    HelpExample {
        tool: "apr",
        command: "apr import <SOURCE> -o <OUTPUT>",
        help_text: r#"Import model from HuggingFace Hub to local APR format.

Usage: apr import <SOURCE> -o <OUTPUT>

Arguments:
  <SOURCE>  HuggingFace model path (e.g., hf://Qwen/Qwen2.5-Coder-1.5B)

Options:
  -o, --output <PATH>  Output file path [required]
  -q, --quantize       Apply 4-bit NF4 quantization
  -h, --help           Print help"#,
    },
    HelpExample {
        tool: "apr",
        command: "apr run <MODEL> --prompt <TEXT>",
        help_text: r#"Run inference on a model with a prompt.

Usage: apr run <MODEL> --prompt <TEXT>

Arguments:
  <MODEL>  Path to APR or GGUF model file

Options:
  -p, --prompt <TEXT>   Input prompt [required]
  -t, --temperature <F> Sampling temperature [default: 0.7]
  -n, --max-tokens <N>  Maximum tokens to generate [default: 256]
  -h, --help            Print help"#,
    },
    HelpExample {
        tool: "pmat",
        command: "pmat comply check",
        help_text: r#"Run compliance checks on project.

Usage: pmat comply check [OPTIONS]

Options:
  -c, --config <PATH>  Config file path [default: .pmat/project.toml]
  -f, --fix            Attempt to auto-fix violations
  -v, --verbose        Show detailed output
  -h, --help           Print help"#,
    },
    HelpExample {
        tool: "rg",
        command: "rg <PATTERN> [PATH]",
        help_text: r#"Recursively search for a pattern in files.

Usage: rg [OPTIONS] <PATTERN> [PATH]...

Arguments:
  <PATTERN>  Regular expression pattern to search
  [PATH]...  Files or directories to search [default: .]

Options:
  -i, --ignore-case    Case insensitive search
  -w, --word-regexp    Only match whole words
  -c, --count          Show count of matches per file
  -l, --files-with-matches  Only show file names
  -h, --help           Print help"#,
    },
];

/// Training data format
#[derive(Debug, Clone)]
pub struct TrainingFormat {
    pub input_template: &'static str,
    pub output_template: &'static str,
    pub example_input: String,
    pub example_output: String,
}

/// Demo results
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub examples: Vec<HelpExample>,
    pub format: TrainingFormat,
    pub data_sources: Vec<&'static str>,
    pub estimated_examples: usize,
}

pub fn run() -> DemoResults {
    let format = TrainingFormat {
        input_template: "Generate help text for: {command}",
        output_template: "{help_text}",
        example_input: format!("Generate help text for: {}", EXAMPLES[0].command),
        example_output: EXAMPLES[0].help_text.to_string(),
    };

    DemoResults {
        examples: EXAMPLES.to_vec(),
        format,
        data_sources: vec![
            "apr-cli (import, run, serve, export, merge)",
            "pmat (comply, hooks, brick-score)",
            "ripgrep (rg)",
            "fd (fd-find)",
            "bat (syntax highlighting cat)",
            "cargo (build, test, run, clippy, fmt)",
            "git (common subcommands)",
        ],
        estimated_examples: 150,
    }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║         CLI HELP TRAINING DATA                                   ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Task: command signature → clap-style help text                  ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Data sources
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ DATA SOURCES                                                   │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for source in &self.data_sources {
            writeln!(f, "  • {}", source)?;
        }
        writeln!(f)?;
        writeln!(f, "  Estimated total: ~{} examples", self.estimated_examples)?;
        writeln!(f)?;

        // Format
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TRAINING FORMAT                                                │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  Input:  {}", self.format.input_template)?;
        writeln!(f, "  Output: {}", self.format.output_template)?;
        writeln!(f)?;

        // Example
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ EXAMPLE: {} ({})                                   │",
            self.examples[0].command.split_whitespace().take(2).collect::<Vec<_>>().join(" "),
            self.examples[0].tool)?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f)?;
        writeln!(f, "  INPUT:")?;
        writeln!(f, "  > {}", self.format.example_input)?;
        writeln!(f)?;
        writeln!(f, "  OUTPUT:")?;
        for line in self.examples[0].help_text.lines() {
            writeln!(f, "  {}", line)?;
        }
        writeln!(f)?;

        // Pattern analysis
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ WHY CLI HELP IS LEARNABLE                                      │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  Consistent structure:")?;
        writeln!(f, "    1. One-line description")?;
        writeln!(f, "    2. Usage: command [OPTIONS] <ARGS>")?;
        writeln!(f, "    3. Arguments: (positional)")?;
        writeln!(f, "    4. Options: (flags)")?;
        writeln!(f)?;
        writeln!(f, "  Predictable patterns:")?;
        writeln!(f, "    • -s, --long <VALUE>  Description [default: X]")?;
        writeln!(f, "    • <ARG>               Description")?;
        writeln!(f, "    • -h, --help          Print help")?;
        writeln!(f)?;

        // Collection command
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ DATA COLLECTION:                                                 ║")?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║   # Collect help from installed Rust CLI tools                   ║")?;
        writeln!(f, "║   for bin in ~/.cargo/bin/*; do                                  ║")?;
        writeln!(f, "║     echo \"---\"                                                   ║")?;
        writeln!(f, "║     echo \"COMMAND: $bin --help\"                                  ║")?;
        writeln!(f, "║     $bin --help 2>/dev/null                                      ║")?;
        writeln!(f, "║   done > cli_help_data.txt                                       ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;

        Ok(())
    }
}

pub fn print_stdout(result: &DemoResults) {
    println!("{result}");
}

pub fn render_tui(result: &DemoResults) {
    println!("{result}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_examples_not_empty() {
        assert!(!EXAMPLES.is_empty());
    }

    #[test]
    fn test_help_has_usage() {
        for ex in EXAMPLES {
            assert!(ex.help_text.contains("Usage:"));
        }
    }

    #[test]
    fn test_help_has_options() {
        for ex in EXAMPLES {
            assert!(ex.help_text.contains("Options:") || ex.help_text.contains("-h, --help"));
        }
    }

    #[test]
    fn test_display_runs() {
        let result = run();
        let display = format!("{}", result);
        assert!(display.contains("CLI HELP"));
        assert!(display.contains("TRAINING FORMAT"));
    }
}
