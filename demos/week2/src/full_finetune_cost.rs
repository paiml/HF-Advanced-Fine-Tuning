//! Full Fine-Tuning Cost Demo
//!
//! Shows why training all parameters is expensive:
//! - Model weights (fp16/bf16)
//! - Gradients (same size as weights)
//! - Optimizer states (Adam: 2x weights for momentum + variance)
//!
//! Total: ~16-20 bytes per parameter for training

use std::fmt;

/// Model size tiers (in billions of parameters)
#[derive(Debug, Clone, Copy)]
pub struct ModelTier {
    pub name: &'static str,
    pub params_b: f64,
    pub typical_gpu: &'static str,
}

pub const TIERS: &[ModelTier] = &[
    ModelTier { name: "Tiny",   params_b: 0.5,  typical_gpu: "RTX 3060 (12GB)" },
    ModelTier { name: "Small",  params_b: 1.5,  typical_gpu: "RTX 3080 (10GB)" },
    ModelTier { name: "Medium", params_b: 7.0,  typical_gpu: "RTX 4090 (24GB)" },
    ModelTier { name: "Large",  params_b: 32.0, typical_gpu: "A100 (80GB)" },
    ModelTier { name: "XL",     params_b: 70.0, typical_gpu: "8x A100 (640GB)" },
];

/// Memory breakdown for training
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    pub tier: &'static str,
    pub params_b: f64,
    pub weights_gb: f64,
    pub gradients_gb: f64,
    pub optimizer_gb: f64,
    pub activations_gb: f64,
    pub total_gb: f64,
}

/// Calculate memory needed for full fine-tuning
pub fn calculate_memory(tier: &ModelTier) -> MemoryBreakdown {
    let params = tier.params_b * 1e9;

    // Weights: 2 bytes per param (fp16/bf16)
    let weights_gb = (params * 2.0) / 1e9;

    // Gradients: same as weights
    let gradients_gb = weights_gb;

    // Optimizer (Adam): 2x for momentum + variance (fp32 each = 8 bytes total)
    let optimizer_gb = (params * 8.0) / 1e9;

    // Activations: rough estimate ~2x weights for batch_size=1
    let activations_gb = weights_gb * 2.0;

    let total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb;

    MemoryBreakdown {
        tier: tier.name,
        params_b: tier.params_b,
        weights_gb,
        gradients_gb,
        optimizer_gb,
        activations_gb,
        total_gb,
    }
}

/// Full demo results
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub breakdowns: Vec<MemoryBreakdown>,
    pub bytes_per_param: f64,
}

/// Run the demo
pub fn run() -> DemoResults {
    let breakdowns: Vec<MemoryBreakdown> = TIERS.iter().map(calculate_memory).collect();

    // Calculate average bytes per parameter
    let bytes_per_param = if let Some(b) = breakdowns.first() {
        (b.total_gb * 1e9) / (b.params_b * 1e9)
    } else {
        0.0
    };

    DemoResults {
        breakdowns,
        bytes_per_param,
    }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║         FULL FINE-TUNING: MEMORY COST                            ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  \"Why can't I just train all the parameters?\"                    ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Formula
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ MEMORY FORMULA (Full Fine-Tuning with Adam)                    │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  Weights:      2 bytes/param (fp16/bf16)")?;
        writeln!(f, "  Gradients:    2 bytes/param (same as weights)")?;
        writeln!(f, "  Optimizer:    8 bytes/param (Adam: momentum + variance, fp32)")?;
        writeln!(f, "  Activations:  ~4 bytes/param (varies with batch size)")?;
        writeln!(f, "  ─────────────────────────────────────────")?;
        writeln!(f, "  TOTAL:        ~{:.0} bytes/param", self.bytes_per_param)?;
        writeln!(f)?;

        // Table header
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ MEMORY BY MODEL SIZE                                           │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  {:8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Tier", "Params", "Weights", "Grads", "Optim", "Acts", "TOTAL")?;
        writeln!(f, "  {:8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "────", "──────", "───────", "─────", "─────", "────", "─────")?;

        for b in &self.breakdowns {
            writeln!(f, "  {:8} {:>7.1}B {:>7.1}GB {:>7.1}GB {:>7.1}GB {:>7.1}GB {:>9.1}GB",
                b.tier,
                b.params_b,
                b.weights_gb,
                b.gradients_gb,
                b.optimizer_gb,
                b.activations_gb,
                b.total_gb)?;
        }
        writeln!(f)?;

        // Reality check
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ REALITY CHECK: What GPU Do You Need?                           │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for (tier, b) in TIERS.iter().zip(&self.breakdowns) {
            let fits = if b.total_gb <= 24.0 { "✓" } else { "✗" };
            writeln!(f, "  {} {:8} ({:>5.1}B): {:>6.1}GB needed → {}",
                fits, tier.name, tier.params_b, b.total_gb, tier.typical_gpu)?;
        }
        writeln!(f)?;

        // The problem
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ THE PROBLEM:                                                     ║")?;
        writeln!(f, "║ • 7B model needs ~112GB — most GPUs are 8-24GB                   ║")?;
        writeln!(f, "║ • 70B model needs ~1.1TB — need a cluster                        ║")?;
        writeln!(f, "║ • Even 1.5B needs ~24GB — barely fits RTX 4090                   ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ THE SOLUTION: LoRA                                               ║")?;
        writeln!(f, "║ • Freeze base model (no gradients, no optimizer states)          ║")?;
        writeln!(f, "║ • Train tiny adapter: 0.1% of params                             ║")?;
        writeln!(f, "║ • 7B trainable on 8GB GPU                                        ║")?;
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
    fn test_tiers_exist() {
        assert!(!TIERS.is_empty());
    }

    #[test]
    fn test_memory_increases_with_params() {
        let tiny = calculate_memory(&TIERS[0]);
        let large = calculate_memory(&TIERS[3]);
        assert!(large.total_gb > tiny.total_gb);
    }

    #[test]
    fn test_memory_components_positive() {
        for tier in TIERS {
            let mem = calculate_memory(tier);
            assert!(mem.weights_gb > 0.0);
            assert!(mem.gradients_gb > 0.0);
            assert!(mem.optimizer_gb > 0.0);
            assert!(mem.activations_gb > 0.0);
        }
    }

    #[test]
    fn test_total_equals_sum() {
        for tier in TIERS {
            let mem = calculate_memory(tier);
            let sum = mem.weights_gb + mem.gradients_gb + mem.optimizer_gb + mem.activations_gb;
            assert!((mem.total_gb - sum).abs() < 0.01);
        }
    }

    #[test]
    fn test_bytes_per_param_reasonable() {
        let result = run();
        // Should be roughly 16 bytes per param
        assert!(result.bytes_per_param > 10.0);
        assert!(result.bytes_per_param < 25.0);
    }

    #[test]
    fn test_7b_needs_over_100gb() {
        let medium = calculate_memory(&TIERS[2]); // 7B
        assert!(medium.total_gb > 100.0);
    }

    #[test]
    fn test_display_runs() {
        let result = run();
        let display = format!("{}", result);
        assert!(display.contains("FULL FINE-TUNING"));
        assert!(display.contains("THE PROBLEM"));
        assert!(display.contains("LoRA"));
    }
}
