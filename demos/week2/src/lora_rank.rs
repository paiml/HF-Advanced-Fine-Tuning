//! LoRA Rank Ablation Demo
//!
//! Shows effect of rank (r) on:
//! - Parameter count
//! - Memory usage
//! - Learning capacity
//!
//! Key insight: Match rank to task complexity.
//! Simple task = low rank. Complex task = high rank.

use std::fmt;

/// Rank configuration with memory and capacity estimates
#[derive(Debug, Clone)]
pub struct RankConfig {
    pub rank: usize,
    pub params: usize,
    pub memory_mb: f64,
    pub brain_size: &'static str,
    pub good_for: &'static str,
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

    let (brain_size, good_for) = match rank {
        r if r <= 4 => ("Tiny", "One trick (formatting, simple style)"),
        r if r <= 8 => ("Small", "One skill (CLI docs, SQL queries)"),
        r if r <= 16 => ("Medium", "A few skills (general fine-tuning)"),
        r if r <= 32 => ("Large", "Many skills (reasoning, multi-task)"),
        r if r <= 64 => ("XL", "Lots of skills (near full fine-tune)"),
        _ => ("XXL", "Overkill for most tasks"),
    };

    RankConfig {
        rank,
        params: total_params,
        memory_mb,
        brain_size,
        good_for,
    }
}

/// Task complexity examples
#[derive(Debug, Clone)]
pub struct TaskExample {
    pub task: &'static str,
    pub rank: &'static str,
    pub why: &'static str,
}

pub const TASK_EXAMPLES: &[TaskExample] = &[
    TaskExample {
        task: "CLI docs (/// comments)",
        rank: "r=8",
        why: "Same format every time",
    },
    TaskExample {
        task: "Code style tweaks",
        rank: "r=8-16",
        why: "Predictable patterns",
    },
    TaskExample {
        task: "Follow instructions",
        rank: "r=16-32",
        why: "Needs some flexibility",
    },
    TaskExample {
        task: "Math problems",
        rank: "r=32-64",
        why: "Complex reasoning",
    },
    TaskExample {
        task: "General chat",
        rank: "r=64+",
        why: "Everything and anything",
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
        writeln!(f, "║           PICKING A RANK                                         ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Rank = how much the model can learn                             ║")?;
        writeln!(f, "║  Low rank = one trick. High rank = many tricks.                  ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Rank comparison table
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ RANK OPTIONS ({})                                    │", self.model_name)?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  {:>6}  {:>10}  {:>10}  {:>8}  {}",
            "Rank", "Params", "Memory", "Size", "Good For")?;
        writeln!(f, "  {:>6}  {:>10}  {:>10}  {:>8}  {}",
            "────", "──────", "──────", "────", "────────")?;

        for cfg in &self.configs {
            let params_str = if cfg.params >= 1_000_000 {
                format!("{:.1}M", cfg.params as f64 / 1e6)
            } else {
                format!("{:.0}K", cfg.params as f64 / 1e3)
            };

            writeln!(f, "  r={:<4}  {:>10}  {:>9.1}MB  {:>8}  {}",
                cfg.rank, params_str, cfg.memory_mb, cfg.brain_size, cfg.good_for)?;
        }
        writeln!(f)?;

        // Visual comparison
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ BIGGER RANK = MORE TO TRAIN                                    │")?;
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
        writeln!(f, "│ MATCH RANK TO TASK                                             │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for task in &self.tasks {
            writeln!(f, "  {:25} {:>8}  — {}",
                task.task, task.rank, task.why)?;
        }
        writeln!(f)?;

        // CLI help specific
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ FOR RUST CLI DOCS:  r={}                                         ║", self.cli_help_recommendation)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║  It's one trick:                                                 ║")?;
        writeln!(f, "║    • Same /// format every time                                  ║")?;
        writeln!(f, "║    • Same words (flags, args, options)                           ║")?;
        writeln!(f, "║    • Same structure (Usage, Examples)                            ║")?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║  Don't rent a mansion when you need a studio.                    ║")?;
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
        assert!(display.contains("RUST CLI"));
    }
}
