//! LoRA Rank Ablation Demo
//!
//! Shows effect of rank (r) on:
//! - Parameter count
//! - Memory usage
//! - Capacity (expressiveness)
//!
//! Key insight: Task complexity determines optimal rank.
//! CLI help = narrow domain = low rank sufficient (r=8-16)

use std::fmt;

/// Rank configuration with memory and capacity estimates
#[derive(Debug, Clone)]
pub struct RankConfig {
    pub rank: usize,
    pub params: usize,
    pub memory_mb: f64,
    pub capacity: &'static str,
    pub use_case: &'static str,
}

/// Model config for calculations
pub struct ModelConfig {
    pub name: &'static str,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_projections: usize, // q, k, v, o = 4
}

pub const QWEN_1_5B: ModelConfig = ModelConfig {
    name: "Qwen2.5-1.5B",
    hidden_dim: 1536,
    num_layers: 28,
    num_projections: 4,
};

pub const RANKS: &[usize] = &[4, 8, 16, 32, 64, 128];

/// Calculate LoRA params for a rank
pub fn calculate_rank_config(model: &ModelConfig, rank: usize) -> RankConfig {
    // LoRA params per projection: 2 × hidden_dim × rank (A and B matrices)
    let params_per_proj = 2 * model.hidden_dim * rank;
    let total_params = params_per_proj * model.num_projections * model.num_layers;

    // Memory: fp16 = 2 bytes per param
    let memory_mb = (total_params as f64 * 2.0) / 1e6;

    let (capacity, use_case) = match rank {
        r if r <= 4 => ("Very Low", "Simple patterns, style transfer"),
        r if r <= 8 => ("Low", "Narrow domains (CLI help, SQL)"),
        r if r <= 16 => ("Medium", "Most fine-tuning tasks"),
        r if r <= 32 => ("High", "Complex reasoning, multi-task"),
        r if r <= 64 => ("Very High", "Near full fine-tune quality"),
        _ => ("Maximum", "Diminishing returns"),
    };

    RankConfig {
        rank,
        params: total_params,
        memory_mb,
        capacity,
        use_case,
    }
}

/// Task complexity examples
#[derive(Debug, Clone)]
pub struct TaskExample {
    pub task: &'static str,
    pub recommended_rank: &'static str,
    pub reason: &'static str,
}

pub const TASK_EXAMPLES: &[TaskExample] = &[
    TaskExample {
        task: "CLI help generation",
        recommended_rank: "r=8",
        reason: "Consistent structure, narrow vocabulary",
    },
    TaskExample {
        task: "Code style adaptation",
        recommended_rank: "r=8-16",
        reason: "Pattern-based, limited variation",
    },
    TaskExample {
        task: "Instruction following",
        recommended_rank: "r=16-32",
        reason: "Diverse formats, reasoning required",
    },
    TaskExample {
        task: "Mathematical reasoning",
        recommended_rank: "r=32-64",
        reason: "Complex patterns, precision critical",
    },
    TaskExample {
        task: "General chat improvement",
        recommended_rank: "r=64+",
        reason: "Broad domain, many capabilities",
    },
];

#[derive(Debug, Clone)]
pub struct DemoResults {
    pub model_name: &'static str,
    pub configs: Vec<RankConfig>,
    pub tasks: Vec<TaskExample>,
    pub cli_help_recommendation: usize,
}

pub fn run() -> DemoResults {
    let configs: Vec<RankConfig> = RANKS.iter()
        .map(|&r| calculate_rank_config(&QWEN_1_5B, r))
        .collect();

    DemoResults {
        model_name: QWEN_1_5B.name,
        configs,
        tasks: TASK_EXAMPLES.to_vec(),
        cli_help_recommendation: 8,
    }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           LoRA RANK SELECTION                                    ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  \"How much capacity do you need?\"                                ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Rank comparison table
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ RANK vs PARAMETERS ({})                              │", self.model_name)?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  {:>6}  {:>12}  {:>10}  {:>12}  {}",
            "Rank", "Params", "Memory", "Capacity", "Use Case")?;
        writeln!(f, "  {:>6}  {:>12}  {:>10}  {:>12}  {}",
            "────", "──────", "──────", "────────", "────────")?;

        for cfg in &self.configs {
            let params_str = if cfg.params >= 1_000_000 {
                format!("{:.1}M", cfg.params as f64 / 1e6)
            } else {
                format!("{:.0}K", cfg.params as f64 / 1e3)
            };

            writeln!(f, "  r={:<4}  {:>12}  {:>9.1}MB  {:>12}  {}",
                cfg.rank, params_str, cfg.memory_mb, cfg.capacity, cfg.use_case)?;
        }
        writeln!(f)?;

        // Visual comparison
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ PARAMETER GROWTH                                               │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        let max_params = self.configs.last().map(|c| c.params).unwrap_or(1);
        for cfg in &self.configs {
            let bar_len = (cfg.params as f64 / max_params as f64 * 40.0) as usize;
            let bar: String = "█".repeat(bar_len.max(1));
            writeln!(f, "  r={:<4} {} {:.1}M",
                cfg.rank, bar, cfg.params as f64 / 1e6)?;
        }
        writeln!(f)?;

        // Task recommendations
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TASK → RECOMMENDED RANK                                        │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for task in &self.tasks {
            writeln!(f, "  {:30} {:>8}  ({})",
                task.task, task.recommended_rank, task.reason)?;
        }
        writeln!(f)?;

        // CLI help specific
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ FOR CLI HELP GENERATION:                                         ║")?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║  Recommended: r={}                                               ║", self.cli_help_recommendation)?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║  Why? CLI help has:                                              ║")?;
        writeln!(f, "║  • Consistent structure (Usage, Options, Examples)               ║")?;
        writeln!(f, "║  • Limited vocabulary (flags, commands, descriptions)            ║")?;
        writeln!(f, "║  • Predictable patterns (clap-style formatting)                  ║")?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║  Higher ranks waste parameters on unused capacity.               ║")?;
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
    fn test_params_increase_with_rank() {
        let configs: Vec<RankConfig> = RANKS.iter()
            .map(|&r| calculate_rank_config(&QWEN_1_5B, r))
            .collect();

        for i in 1..configs.len() {
            assert!(configs[i].params > configs[i-1].params);
        }
    }

    #[test]
    fn test_memory_proportional_to_params() {
        for &rank in RANKS {
            let cfg = calculate_rank_config(&QWEN_1_5B, rank);
            // Memory in MB should be ~2x params in millions (fp16)
            let expected_mb = cfg.params as f64 * 2.0 / 1e6;
            assert!((cfg.memory_mb - expected_mb).abs() < 0.1);
        }
    }

    #[test]
    fn test_r8_under_10mb() {
        let cfg = calculate_rank_config(&QWEN_1_5B, 8);
        assert!(cfg.memory_mb < 10.0);
    }

    #[test]
    fn test_display_runs() {
        let result = run();
        let display = format!("{}", result);
        assert!(display.contains("RANK"));
        assert!(display.contains("CLI HELP"));
    }
}
