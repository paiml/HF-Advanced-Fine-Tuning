//! Feed-Forward Network Demo
//!
//! Shows why FFN is needed after attention:
//! 1. After attention: weighted blend (gathered, not understood)
//! 2. FFN: expand → non-linearity → contract
//! 3. After FFN: computed meaning (ready for next layer)
//!
//! Key insight: Attention mixes. FFN thinks.

use std::fmt;

/// Embedding dimension (small for visibility)
pub const HIDDEN_DIM: usize = 4;

/// FFN intermediate dimension (4× expansion)
pub const INTERMEDIATE_DIM: usize = 16;

/// Token labels
pub const TOKENS: &[&str] = &["The", "cat", "sat"];

/// Number of tokens
pub const SEQ_LEN: usize = 3;

/// Simulated "after attention" embeddings
/// These represent weighted blends from attention
pub const AFTER_ATTENTION: [[f32; HIDDEN_DIM]; SEQ_LEN] = [
    [0.46, 0.41, 0.75, 0.40], // "The" - blended
    [0.40, 0.48, 0.78, 0.40], // "cat" - blended
    [0.39, 0.44, 0.79, 0.44], // "sat" - blended
];

/// Attention weights that produced these blends (for display)
pub const ATTENTION_WEIGHTS: [[f32; SEQ_LEN]; SEQ_LEN] = [
    [0.37, 0.31, 0.32], // "The" attended to
    [0.29, 0.38, 0.33], // "cat" attended to
    [0.29, 0.32, 0.39], // "sat" attended to
];

/// W1: expand from hidden_dim to intermediate_dim
/// Shape: [HIDDEN_DIM][INTERMEDIATE_DIM]
pub fn make_w1() -> Vec<Vec<f32>> {
    // Initialize with simple pattern for reproducibility
    (0..HIDDEN_DIM)
        .map(|i| {
            (0..INTERMEDIATE_DIM)
                .map(|j| ((i + j) % 3) as f32 * 0.1 + 0.05)
                .collect()
        })
        .collect()
}

/// W2: contract from intermediate_dim to hidden_dim
/// Shape: [INTERMEDIATE_DIM][HIDDEN_DIM]
pub fn make_w2() -> Vec<Vec<f32>> {
    (0..INTERMEDIATE_DIM)
        .map(|i| {
            (0..HIDDEN_DIM)
                .map(|j| ((i + j) % 4) as f32 * 0.05 + 0.02)
                .collect()
        })
        .collect()
}

/// GELU activation function
/// Approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.797_884_6;
    let coeff = 0.044715;
    x * 0.5 * (1.0 + (sqrt_2_over_pi * (x + coeff * x.powi(3))).tanh())
}

/// ReLU activation (simpler, for --error mode comparison)
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Linear: no activation (for --error mode)
pub fn linear(x: f32) -> f32 {
    x
}

/// Matrix-vector multiplication
fn matvec(matrix: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; matrix[0].len()];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[j] += vec[i] * val;
        }
    }
    result
}

/// FFN step results
#[derive(Debug, Clone)]
pub struct FfnStep {
    /// Input vector (after attention)
    pub input: Vec<f32>,
    /// After W1 (expanded)
    pub after_w1: Vec<f32>,
    /// After activation
    pub after_activation: Vec<f32>,
    /// After W2 (output)
    pub output: Vec<f32>,
    /// Activation name used
    pub activation_name: String,
}

/// Full demo results
#[derive(Debug, Clone)]
pub struct DemoResults {
    /// Token labels
    pub tokens: Vec<String>,
    /// Attention weights (for context)
    pub attention_weights: Vec<Vec<f32>>,
    /// FFN results per token
    pub ffn_steps: Vec<FfnStep>,
    /// W1 shape for display
    pub w1_shape: (usize, usize),
    /// W2 shape for display
    pub w2_shape: (usize, usize),
    /// Whether non-linearity was skipped
    pub skip_nonlinearity: bool,
}

/// Run FFN on a single vector
pub fn run_ffn<F>(input: &[f32], w1: &[Vec<f32>], w2: &[Vec<f32>], activation: F, activation_name: &str) -> FfnStep
where
    F: Fn(f32) -> f32,
{
    // Step 1: Expand (input × W1)
    let after_w1 = matvec(w1, input);

    // Step 2: Apply activation
    let after_activation: Vec<f32> = after_w1.iter().map(|&x| activation(x)).collect();

    // Step 3: Contract (intermediate × W2)
    let after_w2 = matvec(w2, &after_activation);

    FfnStep {
        input: input.to_vec(),
        after_w1,
        after_activation,
        output: after_w2,
        activation_name: activation_name.to_string(),
    }
}

/// Run the demo
pub fn run(skip_nonlinearity: bool) -> DemoResults {
    let w1 = make_w1();
    let w2 = make_w2();

    let tokens: Vec<String> = TOKENS.iter().map(|s| (*s).to_string()).collect();
    let attention_weights: Vec<Vec<f32>> = ATTENTION_WEIGHTS.iter().map(|row| row.to_vec()).collect();

    let ffn_steps: Vec<FfnStep> = AFTER_ATTENTION
        .iter()
        .map(|input| {
            if skip_nonlinearity {
                run_ffn(input, &w1, &w2, linear, "LINEAR (none)")
            } else {
                run_ffn(input, &w1, &w2, gelu, "GELU")
            }
        })
        .collect();

    DemoResults {
        tokens,
        attention_weights,
        ffn_steps,
        w1_shape: (HIDDEN_DIM, INTERMEDIATE_DIM),
        w2_shape: (INTERMEDIATE_DIM, HIDDEN_DIM),
        skip_nonlinearity,
    }
}

/// Format vector for display
fn fmt_vec(v: &[f32], max_show: usize) -> String {
    if v.len() <= max_show {
        let items: Vec<String> = v.iter().map(|x| format!("{x:.2}")).collect();
        format!("[{}]", items.join(", "))
    } else {
        let first: Vec<String> = v.iter().take(3).map(|x| format!("{x:.2}")).collect();
        let last: Vec<String> = v.iter().rev().take(2).rev().map(|x| format!("{x:.2}")).collect();
        format!("[{}, ... {}]", first.join(", "), last.join(", "))
    }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           FEED-FORWARD NETWORK DEMO                          ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  \"Attention mixes. FFN thinks.\"                              ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Context: what attention produced
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ CONTEXT: After Attention (weighted blends)                  │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        for (i, step) in self.ffn_steps.iter().enumerate() {
            let weights = &self.attention_weights[i];
            writeln!(
                f,
                "  {:>4} = {:.0}% The + {:.0}% cat + {:.0}% sat → {}",
                self.tokens[i],
                weights[0] * 100.0,
                weights[1] * 100.0,
                weights[2] * 100.0,
                fmt_vec(&step.input, 6)
            )?;
        }
        writeln!(f)?;
        writeln!(f, "  Problem: Just weighted averages. No computation yet.")?;
        writeln!(f)?;

        // FFN architecture
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ FFN ARCHITECTURE                                            │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        writeln!(
            f,
            "  W1: [{} × {}]  (expand 4×)",
            self.w1_shape.0, self.w1_shape.1
        )?;
        writeln!(f, "  Activation: {}", self.ffn_steps[0].activation_name)?;
        writeln!(
            f,
            "  W2: [{} × {}]  (contract)",
            self.w2_shape.0, self.w2_shape.1
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  Formula: FFN(x) = W2 · {}(W1 · x)",
            if self.skip_nonlinearity { "LINEAR" } else { "GELU" }
        )?;
        writeln!(f)?;

        // Step-by-step for each token
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        if self.skip_nonlinearity {
            writeln!(f, "│ ⚠️  FFN STEPS (NO NON-LINEARITY - ERROR MODE)                │")?;
        } else {
            writeln!(f, "│ FFN STEPS (per token)                                       │")?;
        }
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;

        for (i, step) in self.ffn_steps.iter().enumerate() {
            writeln!(f, "\n  ── {} ──", self.tokens[i])?;
            writeln!(f, "  Input ({}D):     {}", HIDDEN_DIM, fmt_vec(&step.input, 6))?;
            writeln!(
                f,
                "  After W1 ({}D): {}",
                INTERMEDIATE_DIM,
                fmt_vec(&step.after_w1, 6)
            )?;
            writeln!(
                f,
                "  After {}:    {}",
                if self.skip_nonlinearity { "LINEAR" } else { "GELU  " },
                fmt_vec(&step.after_activation, 6)
            )?;
            writeln!(f, "  Output ({}D):    {}", HIDDEN_DIM, fmt_vec(&step.output, 6))?;
        }
        writeln!(f)?;

        // Compare input vs output
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ BEFORE vs AFTER FFN                                         │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        for (i, step) in self.ffn_steps.iter().enumerate() {
            let input_norm: f32 = step.input.iter().map(|x| x * x).sum::<f32>().sqrt();
            let output_norm: f32 = step.output.iter().map(|x| x * x).sum::<f32>().sqrt();
            writeln!(
                f,
                "  {:>4}: {} → {} (norm: {:.2} → {:.2})",
                self.tokens[i],
                fmt_vec(&step.input, 6),
                fmt_vec(&step.output, 6),
                input_norm,
                output_norm
            )?;
        }
        writeln!(f)?;

        // Summary
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        if self.skip_nonlinearity {
            writeln!(f, "║ ⚠️  WITHOUT NON-LINEARITY:                                   ║")?;
            writeln!(f, "║ • FFN is just two matrix multiplies = one big matrix        ║")?;
            writeln!(f, "║ • W2 · W1 could be pre-computed → no \"thinking\"              ║")?;
            writeln!(f, "║ • Entire transformer collapses to linear function           ║")?;
        } else {
            writeln!(f, "║ KEY INSIGHT:                                                 ║")?;
            writeln!(f, "║ • Attention gathered context (who to listen to)             ║")?;
            writeln!(f, "║ • FFN computed meaning (what it means)                      ║")?;
            writeln!(f, "║ • GELU enables non-linear patterns                          ║")?;
            writeln!(f, "║ • 2/3 of transformer params live in FFN (W1 + W2)           ║")?;
        }
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;

        Ok(())
    }
}

/// Print to stdout (CI mode)
pub fn print_stdout(result: &DemoResults) {
    println!("{result}");
}

/// Render TUI (same as stdout for now)
pub fn render_tui(result: &DemoResults) {
    println!("{result}");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Constants tests ==========

    #[test]
    fn test_dimensions() {
        assert_eq!(INTERMEDIATE_DIM, HIDDEN_DIM * 4);
        assert_eq!(TOKENS.len(), SEQ_LEN);
        assert_eq!(AFTER_ATTENTION.len(), SEQ_LEN);
        assert_eq!(ATTENTION_WEIGHTS.len(), SEQ_LEN);
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        for row in &ATTENTION_WEIGHTS {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Row sum {} should be ~1.0", sum);
        }
    }

    #[test]
    fn test_after_attention_dimensions() {
        for row in &AFTER_ATTENTION {
            assert_eq!(row.len(), HIDDEN_DIM);
        }
    }

    // ========== Weight matrix tests ==========

    #[test]
    fn test_w1_shape() {
        let w1 = make_w1();
        assert_eq!(w1.len(), HIDDEN_DIM);
        assert_eq!(w1[0].len(), INTERMEDIATE_DIM);
    }

    #[test]
    fn test_w2_shape() {
        let w2 = make_w2();
        assert_eq!(w2.len(), INTERMEDIATE_DIM);
        assert_eq!(w2[0].len(), HIDDEN_DIM);
    }

    #[test]
    fn test_w1_values_bounded() {
        let w1 = make_w1();
        for row in &w1 {
            for &val in row {
                assert!(val >= 0.0 && val <= 1.0, "W1 value {} out of bounds", val);
            }
        }
    }

    #[test]
    fn test_w2_values_bounded() {
        let w2 = make_w2();
        for row in &w2 {
            for &val in row {
                assert!(val >= 0.0 && val <= 1.0, "W2 value {} out of bounds", val);
            }
        }
    }

    // ========== Activation function tests ==========

    #[test]
    fn test_gelu_zero() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let x = 1.0;
        let y = gelu(x);
        // GELU(1) ≈ 0.841
        assert!(y > 0.8 && y < 0.9, "GELU(1) = {} should be ~0.841", y);
    }

    #[test]
    fn test_gelu_negative() {
        let x = -1.0;
        let y = gelu(x);
        // GELU(-1) ≈ -0.159
        assert!(y > -0.2 && y < -0.1, "GELU(-1) = {} should be ~-0.159", y);
    }

    #[test]
    fn test_gelu_large_positive() {
        let x = 5.0;
        let y = gelu(x);
        // GELU(x) ≈ x for large positive x
        assert!((y - x).abs() < 0.01, "GELU({}) should be ~{}", x, x);
    }

    #[test]
    fn test_gelu_large_negative() {
        let x = -5.0;
        let y = gelu(x);
        // GELU(x) ≈ 0 for large negative x
        assert!(y.abs() < 0.01, "GELU({}) = {} should be ~0", x, y);
    }

    #[test]
    fn test_relu_zero() {
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_relu_positive() {
        assert_eq!(relu(5.0), 5.0);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(relu(-5.0), 0.0);
    }

    #[test]
    fn test_linear_passthrough() {
        assert_eq!(linear(3.14), 3.14);
        assert_eq!(linear(-2.5), -2.5);
    }

    // ========== Matrix-vector multiplication tests ==========

    #[test]
    fn test_matvec_identity() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![3.0, 4.0];
        let result = matvec(&identity, &v);
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn test_matvec_simple() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![1.0, 1.0];
        let result = matvec(&m, &v);
        // [1,1] × [[1,2],[3,4]] = [1*1+1*3, 1*2+1*4] = [4, 6]
        assert_eq!(result, vec![4.0, 6.0]);
    }

    #[test]
    fn test_matvec_expansion() {
        // 2D → 4D
        let m = vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]];
        let v = vec![2.0, 3.0];
        let result = matvec(&m, &v);
        assert_eq!(result.len(), 4);
    }

    // ========== FFN step tests ==========

    #[test]
    fn test_run_ffn_output_dimension() {
        let w1 = make_w1();
        let w2 = make_w2();
        let input = AFTER_ATTENTION[0].to_vec();
        let step = run_ffn(&input, &w1, &w2, gelu, "GELU");
        assert_eq!(step.output.len(), HIDDEN_DIM);
    }

    #[test]
    fn test_run_ffn_intermediate_dimension() {
        let w1 = make_w1();
        let w2 = make_w2();
        let input = AFTER_ATTENTION[0].to_vec();
        let step = run_ffn(&input, &w1, &w2, gelu, "GELU");
        assert_eq!(step.after_w1.len(), INTERMEDIATE_DIM);
        assert_eq!(step.after_activation.len(), INTERMEDIATE_DIM);
    }

    #[test]
    fn test_run_ffn_input_preserved() {
        let w1 = make_w1();
        let w2 = make_w2();
        let input = AFTER_ATTENTION[0].to_vec();
        let step = run_ffn(&input, &w1, &w2, gelu, "GELU");
        assert_eq!(step.input, input);
    }

    #[test]
    fn test_run_ffn_activation_name() {
        let w1 = make_w1();
        let w2 = make_w2();
        let input = AFTER_ATTENTION[0].to_vec();
        let step = run_ffn(&input, &w1, &w2, gelu, "GELU");
        assert_eq!(step.activation_name, "GELU");
    }

    #[test]
    fn test_run_ffn_linear_vs_gelu_different() {
        let w1 = make_w1();
        let w2 = make_w2();
        let input = AFTER_ATTENTION[0].to_vec();
        let step_gelu = run_ffn(&input, &w1, &w2, gelu, "GELU");
        let step_linear = run_ffn(&input, &w1, &w2, linear, "LINEAR");
        // Outputs should differ
        let diff: f32 = step_gelu
            .output
            .iter()
            .zip(step_linear.output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "GELU and LINEAR should produce different outputs");
    }

    // ========== Run function tests ==========

    #[test]
    fn test_run_normal_mode() {
        let result = run(false);
        assert!(!result.skip_nonlinearity);
        assert_eq!(result.ffn_steps.len(), SEQ_LEN);
    }

    #[test]
    fn test_run_error_mode() {
        let result = run(true);
        assert!(result.skip_nonlinearity);
    }

    #[test]
    fn test_run_tokens_match() {
        let result = run(false);
        assert_eq!(result.tokens, vec!["The", "cat", "sat"]);
    }

    #[test]
    fn test_run_shapes_correct() {
        let result = run(false);
        assert_eq!(result.w1_shape, (HIDDEN_DIM, INTERMEDIATE_DIM));
        assert_eq!(result.w2_shape, (INTERMEDIATE_DIM, HIDDEN_DIM));
    }

    #[test]
    fn test_run_activation_name_in_steps() {
        let result = run(false);
        for step in &result.ffn_steps {
            assert_eq!(step.activation_name, "GELU");
        }
        let result_error = run(true);
        for step in &result_error.ffn_steps {
            assert_eq!(step.activation_name, "LINEAR (none)");
        }
    }

    // ========== Display tests ==========

    #[test]
    fn test_fmt_vec_short() {
        let v = vec![1.0, 2.0, 3.0];
        let s = fmt_vec(&v, 6);
        assert_eq!(s, "[1.00, 2.00, 3.00]");
    }

    #[test]
    fn test_fmt_vec_long() {
        let v: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let s = fmt_vec(&v, 6);
        assert!(s.contains("..."));
    }

    #[test]
    fn test_display_normal_mode() {
        let result = run(false);
        let display = format!("{}", result);
        assert!(display.contains("FEED-FORWARD"));
        assert!(display.contains("GELU"));
        assert!(display.contains("KEY INSIGHT"));
    }

    #[test]
    fn test_display_error_mode() {
        let result = run(true);
        let display = format!("{}", result);
        assert!(display.contains("ERROR MODE"));
        assert!(display.contains("LINEAR"));
        assert!(display.contains("collapses"));
    }

    #[test]
    fn test_display_contains_tokens() {
        let result = run(false);
        let display = format!("{}", result);
        for token in TOKENS {
            assert!(display.contains(token));
        }
    }

    #[test]
    fn test_display_shows_shapes() {
        let result = run(false);
        let display = format!("{}", result);
        assert!(display.contains(&format!("[{} × {}]", HIDDEN_DIM, INTERMEDIATE_DIM)));
    }

    // ========== Output function tests ==========

    #[test]
    fn test_print_stdout_runs() {
        let result = run(false);
        print_stdout(&result);
    }

    #[test]
    fn test_render_tui_runs() {
        let result = run(false);
        render_tui(&result);
    }

    // ========== Integration tests ==========

    #[test]
    fn test_ffn_changes_representation() {
        let result = run(false);
        for step in &result.ffn_steps {
            // Output should differ from input
            let diff: f32 = step
                .input
                .iter()
                .zip(step.output.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(diff > 0.01, "FFN should change the representation");
        }
    }

    #[test]
    fn test_expansion_increases_dimension() {
        let result = run(false);
        for step in &result.ffn_steps {
            assert!(
                step.after_w1.len() > step.input.len(),
                "W1 should expand dimensions"
            );
        }
    }

    #[test]
    fn test_contraction_restores_dimension() {
        let result = run(false);
        for step in &result.ffn_steps {
            assert_eq!(
                step.output.len(),
                step.input.len(),
                "Output should match input dimension"
            );
        }
    }
}
