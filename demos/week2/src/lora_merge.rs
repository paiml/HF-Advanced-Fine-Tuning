//! LoRA Merge Demo
//!
//! Shows how to merge LoRA adapters back into base model:
//! - W' = W + scale × A × B
//! - Combine multiple LoRAs
//! - Export to GGUF for deployment

use std::fmt;

/// Merge strategies
#[derive(Debug, Clone)]
pub struct MergeStrategy {
    pub name: &'static str,
    pub formula: &'static str,
    pub description: &'static str,
    pub use_case: &'static str,
}

pub const STRATEGIES: &[MergeStrategy] = &[
    MergeStrategy {
        name: "Standard",
        formula: "W' = W + α × A × B",
        description: "Simple addition with scaling factor α",
        use_case: "Single adapter deployment",
    },
    MergeStrategy {
        name: "Linear",
        formula: "W' = W + Σ(αᵢ × Aᵢ × Bᵢ)",
        description: "Sum multiple adapters with weights",
        use_case: "Combine task-specific adapters",
    },
    MergeStrategy {
        name: "TIES",
        formula: "W' = W + trim(sign × mean(|ΔW|))",
        description: "Trim, elect sign, merge magnitudes",
        use_case: "Reduce interference between adapters",
    },
    MergeStrategy {
        name: "DARE",
        formula: "W' = W + rescale(drop(ΔW, p))",
        description: "Randomly drop and rescale",
        use_case: "Improve generalization",
    },
];

/// Toy example of merging
#[derive(Debug, Clone)]
pub struct ToyMerge {
    pub w: Vec<Vec<f32>>,
    pub a: Vec<Vec<f32>>,
    pub b: Vec<Vec<f32>>,
    pub scale: f32,
    pub ab: Vec<Vec<f32>>,
    pub w_merged: Vec<Vec<f32>>,
}

pub fn create_toy_merge() -> ToyMerge {
    let w = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    let a = vec![
        vec![0.2],
        vec![0.0],
        vec![0.1],
    ];

    let b = vec![
        vec![1.0, 0.5, 0.0],
    ];

    let scale = 1.0;

    // A × B
    let ab = matmul(&a, &b);

    // W' = W + scale × A × B
    let w_merged = mat_add_scaled(&w, &ab, scale);

    ToyMerge { w, a, b, scale, ab, w_merged }
}

fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let m = a.len();
    let n = b[0].len();
    let k = b.len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

fn mat_add_scaled(a: &[Vec<f32>], b: &[Vec<f32>], scale: f32) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| {
            row_a.iter().zip(row_b.iter()).map(|(x, y)| x + scale * y).collect()
        })
        .collect()
}

fn fmt_matrix(m: &[Vec<f32>], indent: &str) -> String {
    m.iter()
        .map(|row| {
            let vals: Vec<String> = row.iter().map(|v| format!("{:>5.2}", v)).collect();
            format!("{}[{}]", indent, vals.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Demo results
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub strategies: Vec<MergeStrategy>,
    pub toy: ToyMerge,
    pub deployment_steps: Vec<&'static str>,
}

pub fn run() -> DemoResults {
    DemoResults {
        strategies: STRATEGIES.to_vec(),
        toy: create_toy_merge(),
        deployment_steps: vec![
            "1. Train LoRA adapter with entrenar",
            "2. Merge: apr merge base.apr adapter.lora -o merged.apr",
            "3. Export: apr export merged.apr --format gguf -o model.gguf",
            "4. Serve: apr serve model.gguf --port 8080",
            "5. Use: apr chat model.gguf",
        ],
    }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           LoRA MERGE & DEPLOYMENT                                ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  \"From adapter to production in 3 commands\"                      ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Merge strategies
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ MERGE STRATEGIES                                               │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for strat in &self.strategies {
            writeln!(f, "  {} ({})", strat.name, strat.formula)?;
            writeln!(f, "    {}", strat.description)?;
            writeln!(f, "    Use: {}", strat.use_case)?;
            writeln!(f)?;
        }

        // Toy example
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TOY EXAMPLE: Standard Merge                                    │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;

        writeln!(f, "\n  W (base weights, frozen during training):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.w, "    "))?;

        writeln!(f, "\n  A (LoRA down-projection, trained):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.a, "    "))?;

        writeln!(f, "\n  B (LoRA up-projection, trained):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.b, "    "))?;

        writeln!(f, "\n  A × B (the learned delta):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.ab, "    "))?;

        writeln!(f, "\n  W' = W + {} × (A × B) (merged weights):", self.toy.scale)?;
        writeln!(f, "{}", fmt_matrix(&self.toy.w_merged, "    "))?;
        writeln!(f)?;

        // Key insight
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ WHY MERGE?                                                     │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  During training:  Keep W frozen, train A and B separately")?;
        writeln!(f, "  At inference:     W' = W + AB computed once, no extra cost")?;
        writeln!(f)?;
        writeln!(f, "  Benefit: Same inference speed as base model!")?;
        writeln!(f)?;

        // Deployment
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ DEPLOYMENT WORKFLOW (apr-cli)                                    ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        for step in &self.deployment_steps {
            writeln!(f, "║  {}{}║",
                step,
                " ".repeat(62 - step.len()))?;
        }
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ apr run/serve/chat \"just works\" with merged GGUF                 ║")?;
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
    fn test_strategies_not_empty() {
        assert!(!STRATEGIES.is_empty());
    }

    #[test]
    fn test_merge_dimensions() {
        let toy = create_toy_merge();
        assert_eq!(toy.w.len(), toy.w_merged.len());
        assert_eq!(toy.w[0].len(), toy.w_merged[0].len());
    }

    #[test]
    fn test_ab_rank_1() {
        let toy = create_toy_merge();
        // A is 3×1, B is 1×3, so AB is 3×3
        assert_eq!(toy.ab.len(), 3);
        assert_eq!(toy.ab[0].len(), 3);
    }

    #[test]
    fn test_merged_differs_from_original() {
        let toy = create_toy_merge();
        let mut differs = false;
        for i in 0..toy.w.len() {
            for j in 0..toy.w[0].len() {
                if (toy.w[i][j] - toy.w_merged[i][j]).abs() > 1e-6 {
                    differs = true;
                }
            }
        }
        assert!(differs);
    }

    #[test]
    fn test_display_runs() {
        let result = run();
        let display = format!("{}", result);
        assert!(display.contains("MERGE"));
        assert!(display.contains("apr"));
    }
}
