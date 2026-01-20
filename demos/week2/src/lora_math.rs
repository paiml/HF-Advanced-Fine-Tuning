//! LoRA Math Demo
//!
//! Shows the low-rank decomposition: W' = W + ΔW = W + A × B
//!
//! Key insight:
//! - Original: d×d matrix = d² parameters
//! - LoRA: d×r + r×d = 2×d×r parameters
//! - With r << d, massive parameter reduction

use std::fmt;

/// Dimensions for the demo
pub const D: usize = 4096;  // Hidden dimension (like Qwen 1.5B)
pub const R_VALUES: &[usize] = &[4, 8, 16, 32, 64];

/// LoRA configuration
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub d: usize,           // Hidden dimension
    pub r: usize,           // Rank
    pub original_params: usize,
    pub lora_params: usize,
    pub reduction: f64,     // How much smaller
    pub percent: f64,       // Percentage of original
}

/// Calculate LoRA parameters
pub fn calculate_lora(d: usize, r: usize) -> LoraConfig {
    let original_params = d * d;
    let lora_params = 2 * d * r;  // A: d×r, B: r×d
    let reduction = original_params as f64 / lora_params as f64;
    let percent = (lora_params as f64 / original_params as f64) * 100.0;

    LoraConfig {
        d,
        r,
        original_params,
        lora_params,
        reduction,
        percent,
    }
}

/// Toy example showing actual matrix math
#[derive(Debug, Clone)]
pub struct ToyExample {
    pub d: usize,
    pub r: usize,
    pub w: Vec<Vec<f32>>,       // Original weights (frozen)
    pub a: Vec<Vec<f32>>,       // LoRA A matrix
    pub b: Vec<Vec<f32>>,       // LoRA B matrix
    pub delta_w: Vec<Vec<f32>>, // A × B
    pub w_prime: Vec<Vec<f32>>, // W + A × B
}

/// Create a toy example with small matrices
pub fn create_toy_example() -> ToyExample {
    let d = 4;
    let r = 2;

    // Original weights (frozen)
    let w = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    // LoRA A: d×r (learns "what to adapt")
    let a = vec![
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![0.1, 0.1],
        vec![0.0, 0.0],
    ];

    // LoRA B: r×d (learns "how to adapt")
    let b = vec![
        vec![1.0, 0.0, 0.5, 0.0],
        vec![0.0, 1.0, 0.5, 0.0],
    ];

    // Compute ΔW = A × B
    let delta_w = matmul(&a, &b);

    // Compute W' = W + ΔW
    let w_prime = mat_add(&w, &delta_w);

    ToyExample { d, r, w, a, b, delta_w, w_prime }
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

fn mat_add(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| {
            row_a.iter().zip(row_b.iter()).map(|(x, y)| x + y).collect()
        })
        .collect()
}

/// Demo results
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub configs: Vec<LoraConfig>,
    pub toy: ToyExample,
}

pub fn run() -> DemoResults {
    let configs: Vec<LoraConfig> = R_VALUES.iter().map(|&r| calculate_lora(D, r)).collect();
    let toy = create_toy_example();
    DemoResults { configs, toy }
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

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║               LoRA: LOW-RANK ADAPTATION                          ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  W' = W + ΔW = W + A × B                                         ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // The key insight
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ THE KEY INSIGHT                                                │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  Original weight matrix W: [d × d] = d² parameters")?;
        writeln!(f, "  LoRA adapters A and B:    [d × r] + [r × d] = 2dr parameters")?;
        writeln!(f)?;
        writeln!(f, "  When r << d:  2dr << d²")?;
        writeln!(f, "  Example: d=4096, r=8 → 65K params vs 16M params (256× smaller)")?;
        writeln!(f)?;

        // Parameter comparison table
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ PARAMETER COMPARISON (d={})                               │", D)?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  {:>6}  {:>12}  {:>12}  {:>10}  {:>8}",
            "Rank", "Original", "LoRA", "Reduction", "% of W")?;
        writeln!(f, "  {:>6}  {:>12}  {:>12}  {:>10}  {:>8}",
            "────", "────────", "────", "─────────", "──────")?;

        for cfg in &self.configs {
            writeln!(f, "  r={:<4}  {:>12}  {:>12}  {:>9.0}×  {:>7.2}%",
                cfg.r,
                format_params(cfg.original_params),
                format_params(cfg.lora_params),
                cfg.reduction,
                cfg.percent)?;
        }
        writeln!(f)?;

        // Toy example
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TOY EXAMPLE (d={}, r={})                                        │", self.toy.d, self.toy.r)?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;

        writeln!(f, "\n  W (original, FROZEN):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.w, "    "))?;

        writeln!(f, "\n  A (trainable, d×r):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.a, "    "))?;

        writeln!(f, "\n  B (trainable, r×d):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.b, "    "))?;

        writeln!(f, "\n  ΔW = A × B (the learned update):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.delta_w, "    "))?;

        writeln!(f, "\n  W' = W + ΔW (final weights at inference):")?;
        writeln!(f, "{}", fmt_matrix(&self.toy.w_prime, "    "))?;
        writeln!(f)?;

        // Summary
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ WHY IT WORKS:                                                    ║")?;
        writeln!(f, "║ • W is frozen — no gradients, no optimizer states               ║")?;
        writeln!(f, "║ • Only A and B are trained — tiny memory footprint              ║")?;
        writeln!(f, "║ • At inference: merge W' = W + AB — no extra latency            ║")?;
        writeln!(f, "║ • Hypothesis: weight updates have low \"intrinsic rank\"          ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;

        Ok(())
    }
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
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
    fn test_lora_params_smaller() {
        for &r in R_VALUES {
            let cfg = calculate_lora(D, r);
            assert!(cfg.lora_params < cfg.original_params);
        }
    }

    #[test]
    fn test_reduction_increases_with_smaller_rank() {
        let r8 = calculate_lora(D, 8);
        let r64 = calculate_lora(D, 64);
        assert!(r8.reduction > r64.reduction);
    }

    #[test]
    fn test_toy_dimensions() {
        let toy = create_toy_example();
        assert_eq!(toy.w.len(), toy.d);
        assert_eq!(toy.a.len(), toy.d);
        assert_eq!(toy.a[0].len(), toy.r);
        assert_eq!(toy.b.len(), toy.r);
        assert_eq!(toy.b[0].len(), toy.d);
    }

    #[test]
    fn test_delta_w_from_ab() {
        let toy = create_toy_example();
        // ΔW should have same dimensions as W
        assert_eq!(toy.delta_w.len(), toy.d);
        assert_eq!(toy.delta_w[0].len(), toy.d);
    }

    #[test]
    fn test_w_prime_sum() {
        let toy = create_toy_example();
        // W'[i][j] = W[i][j] + ΔW[i][j]
        for i in 0..toy.d {
            for j in 0..toy.d {
                let expected = toy.w[i][j] + toy.delta_w[i][j];
                assert!((toy.w_prime[i][j] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_percent_small() {
        for &r in R_VALUES {
            let cfg = calculate_lora(D, r);
            assert!(cfg.percent < 5.0);  // Always under 5% for these ranks
        }
    }

    #[test]
    fn test_display_contains_formula() {
        let result = run();
        let display = format!("{}", result);
        assert!(display.contains("W + A × B"));
        assert!(display.contains("FROZEN"));
    }
}
