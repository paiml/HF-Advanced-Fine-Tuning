//! Attention Mechanism Demo
//!
//! Shows how self-attention works with toy Q/K/V matrices:
//! 1. Input embeddings → Q, K, V projections
//! 2. Attention scores = Q × K^T (who looks at whom)
//! 3. Softmax → probabilities (0-1, sum to 1)
//! 4. Output = attention_weights × V
//!
//! Special emphasis on softmax: converts arbitrary scores to valid probabilities.

use std::fmt;

/// Toy sentence for demo
pub const DEMO_SENTENCE: &str = "The cat sat";

/// Token labels for display
pub const TOKENS: &[&str] = &["The", "cat", "sat"];

/// Embedding dimension (small for visibility)
pub const EMBED_DIM: usize = 4;

/// Number of tokens
pub const SEQ_LEN: usize = 3;

/// Input embeddings for each token (hand-crafted for clear demo)
/// Each row is a token's embedding vector
pub const INPUT_EMBEDDINGS: [[f32; EMBED_DIM]; SEQ_LEN] = [
    [1.0, 0.0, 0.5, 0.2], // "The" - article, low semantic content
    [0.2, 1.0, 0.8, 0.1], // "cat" - noun, subject
    [0.1, 0.3, 1.0, 0.9], // "sat" - verb, action
];

/// Weight matrix for Query projection (EMBED_DIM × EMBED_DIM)
pub const W_Q: [[f32; EMBED_DIM]; EMBED_DIM] = [
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.5],
];

/// Weight matrix for Key projection
pub const W_K: [[f32; EMBED_DIM]; EMBED_DIM] = [
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.5],
];

/// Weight matrix for Value projection
pub const W_V: [[f32; EMBED_DIM]; EMBED_DIM] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Result of attention computation
#[derive(Debug, Clone)]
pub struct AttentionResult {
    /// Input embeddings [SEQ_LEN × EMBED_DIM]
    pub inputs: Vec<Vec<f32>>,
    /// Query vectors [SEQ_LEN × EMBED_DIM]
    pub queries: Vec<Vec<f32>>,
    /// Key vectors [SEQ_LEN × EMBED_DIM]
    pub keys: Vec<Vec<f32>>,
    /// Value vectors [SEQ_LEN × EMBED_DIM]
    pub values: Vec<Vec<f32>>,
    /// Raw attention scores [SEQ_LEN × SEQ_LEN] (before softmax)
    pub raw_scores: Vec<Vec<f32>>,
    /// Attention weights [SEQ_LEN × SEQ_LEN] (after softmax)
    pub attention_weights: Vec<Vec<f32>>,
    /// Output embeddings [SEQ_LEN × EMBED_DIM]
    pub outputs: Vec<Vec<f32>>,
    /// Whether softmax was skipped (error mode)
    pub skip_softmax: bool,
}

/// Softmax explanation step
#[derive(Debug, Clone)]
pub struct SoftmaxStep {
    /// Raw scores for one token
    pub raw: Vec<f32>,
    /// e^x for each score
    pub exp: Vec<f32>,
    /// Sum of e^x values
    pub sum: f32,
    /// Final probabilities
    pub probs: Vec<f32>,
}

/// Matrix multiplication: A[m×n] × B[n×p] = C[m×p]
fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let m = a.len();
    let n = b.len();
    let p = b[0].len();

    let mut result = vec![vec![0.0; p]; m];
    for i in 0..m {
        for j in 0..p {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Transpose a matrix
fn transpose(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if m.is_empty() {
        return vec![];
    }
    let rows = m.len();
    let cols = m[0].len();
    let mut result = vec![vec![0.0; rows]; cols];
    for (i, row) in m.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[j][i] = val;
        }
    }
    result
}

/// Convert 2D array to Vec<Vec<f32>>
fn array_to_vec<const R: usize, const C: usize>(arr: &[[f32; C]; R]) -> Vec<Vec<f32>> {
    arr.iter().map(|row| row.to_vec()).collect()
}

/// Softmax: e^x / Σe^x (converts scores to probabilities)
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    // Subtract max for numerical stability
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores.iter().map(|&x| (x - max_score).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    exp_scores.iter().map(|&x| x / sum).collect()
}

/// Explain softmax step-by-step for one row
pub fn explain_softmax(raw_scores: &[f32]) -> SoftmaxStep {
    let max_score = raw_scores
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = raw_scores.iter().map(|&x| (x - max_score).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();

    SoftmaxStep {
        raw: raw_scores.to_vec(),
        exp,
        sum,
        probs,
    }
}

/// Run attention computation
pub fn run(skip_softmax: bool) -> AttentionResult {
    // Convert constants to vectors
    let inputs = array_to_vec(&INPUT_EMBEDDINGS);
    let w_q = array_to_vec(&W_Q);
    let w_k = array_to_vec(&W_K);
    let w_v = array_to_vec(&W_V);

    // Step 1: Project to Q, K, V
    // Q = X × W_Q, K = X × W_K, V = X × W_V
    let queries = matmul(&inputs, &w_q);
    let keys = matmul(&inputs, &w_k);
    let values = matmul(&inputs, &w_v);

    // Step 2: Compute attention scores = Q × K^T
    let keys_t = transpose(&keys);
    let raw_scores = matmul(&queries, &keys_t);

    // Step 3: Apply softmax (or skip in error mode)
    let attention_weights = if skip_softmax {
        // Error mode: use raw scores directly (BAD!)
        raw_scores.clone()
    } else {
        // Normal: softmax each row
        raw_scores.iter().map(|row| softmax(row)).collect()
    };

    // Step 4: Compute output = attention_weights × V
    let outputs = matmul(&attention_weights, &values);

    AttentionResult {
        inputs,
        queries,
        keys,
        values,
        raw_scores,
        attention_weights,
        outputs,
        skip_softmax,
    }
}

/// Format a vector as a string with fixed precision
fn fmt_vec(v: &[f32], precision: usize) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.precision$}")).collect();
    format!("[{}]", items.join(", "))
}

/// Format a matrix for display
fn fmt_matrix(m: &[Vec<f32>], precision: usize, labels: Option<&[&str]>) -> String {
    let mut lines = Vec::new();
    for (i, row) in m.iter().enumerate() {
        let label = labels.map_or(String::new(), |l| format!("{:>4} ", l[i]));
        lines.push(format!("  {}{}", label, fmt_vec(row, precision)));
    }
    lines.join("\n")
}

impl fmt::Display for AttentionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           SELF-ATTENTION MECHANISM DEMO                      ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Sentence: \"{}\"                                          ║", DEMO_SENTENCE)?;
        writeln!(f, "║  Tokens: {:?}                                     ║", TOKENS)?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Input embeddings
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ STEP 1: INPUT EMBEDDINGS                                    │")?;
        writeln!(f, "│ Each token → vector of {} numbers                           │", EMBED_DIM)?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "{}", fmt_matrix(&self.inputs, 2, Some(TOKENS)))?;
        writeln!(f)?;

        // Q, K, V projections
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ STEP 2: Q/K/V PROJECTIONS                                   │")?;
        writeln!(f, "│ Q = \"What am I looking for?\"                                │")?;
        writeln!(f, "│ K = \"What do I contain?\"                                    │")?;
        writeln!(f, "│ V = \"What info do I provide?\"                               │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  Query (Q):")?;
        writeln!(f, "{}", fmt_matrix(&self.queries, 2, Some(TOKENS)))?;
        writeln!(f, "  Key (K):")?;
        writeln!(f, "{}", fmt_matrix(&self.keys, 2, Some(TOKENS)))?;
        writeln!(f, "  Value (V):")?;
        writeln!(f, "{}", fmt_matrix(&self.values, 2, Some(TOKENS)))?;
        writeln!(f)?;

        // Raw attention scores
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ STEP 3: RAW ATTENTION SCORES (Q × K^T)                      │")?;
        writeln!(f, "│ Higher = more similar = pay more attention                  │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "              The    cat    sat")?;
        for (i, row) in self.raw_scores.iter().enumerate() {
            writeln!(
                f,
                "  {:>4} → [{:>6.2}, {:>6.2}, {:>6.2}]",
                TOKENS[i], row[0], row[1], row[2]
            )?;
        }
        writeln!(f)?;

        // Softmax explanation
        if self.skip_softmax {
            writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
            writeln!(f, "│ ⚠️  STEP 4: SOFTMAX SKIPPED (ERROR MODE)                     │")?;
            writeln!(f, "│ Using raw scores as weights - THIS IS WRONG!                │")?;
            writeln!(f, "│ • Scores don't sum to 1                                     │")?;
            writeln!(f, "│ • Can't interpret as \"attention distribution\"               │")?;
            writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        } else {
            writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
            writeln!(f, "│ STEP 4: SOFTMAX → PROBABILITIES                             │")?;
            writeln!(f, "│ Formula: softmax(x)_i = e^x_i / Σe^x_j                      │")?;
            writeln!(f, "│ • Converts ANY numbers → probabilities (0-1)                │")?;
            writeln!(f, "│ • Always sums to 1.0                                        │")?;
            writeln!(f, "│ • Preserves ranking (higher score = higher prob)            │")?;
            writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
            // Show softmax step-by-step for first token
            let step = explain_softmax(&self.raw_scores[0]);
            writeln!(f, "  Example for \"The\" attending to all tokens:")?;
            writeln!(f, "    raw scores:  {}", fmt_vec(&step.raw, 2))?;
            writeln!(f, "    e^x:         {}", fmt_vec(&step.exp, 2))?;
            writeln!(f, "    sum(e^x):    {:.2}", step.sum)?;
            writeln!(f, "    e^x / sum:   {}", fmt_vec(&step.probs, 2))?;
        }
        writeln!(f)?;

        // Attention weights
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        if self.skip_softmax {
            writeln!(f, "│ ATTENTION WEIGHTS (RAW - INVALID!)                          │")?;
        } else {
            writeln!(f, "│ ATTENTION WEIGHTS (after softmax)                           │")?;
            writeln!(f, "│ Each row sums to 1.0 = valid probability distribution       │")?;
        }
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  \"Who does each token attend to?\"")?;
        writeln!(f, "              The    cat    sat    sum")?;
        for (i, row) in self.attention_weights.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            writeln!(
                f,
                "  {:>4} → [{:>6.2}, {:>6.2}, {:>6.2}] = {:.2}",
                TOKENS[i], row[0], row[1], row[2], sum
            )?;
        }
        writeln!(f)?;

        // Output
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ STEP 5: OUTPUT = attention_weights × V                      │")?;
        writeln!(f, "│ Weighted combination of all token values                    │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "{}", fmt_matrix(&self.outputs, 2, Some(TOKENS)))?;
        writeln!(f)?;

        // Summary
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        if self.skip_softmax {
            writeln!(f, "║ ⚠️  ERROR: Without softmax, attention weights are invalid    ║")?;
            writeln!(f, "║ • Row sums ≠ 1.0 (not probability distributions)            ║")?;
            writeln!(f, "║ • Outputs are scaled incorrectly                            ║")?;
        } else {
            writeln!(f, "║ KEY INSIGHT: Attention = \"soft\" dictionary lookup           ║")?;
            writeln!(f, "║ • Q asks: \"what am I looking for?\"                          ║")?;
            writeln!(f, "║ • K answers: \"here's what I have\"                           ║")?;
            writeln!(f, "║ • Softmax: \"how much to attend to each?\" (sums to 1)        ║")?;
            writeln!(f, "║ • V provides: \"here's my information\"                       ║")?;
        }
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;

        Ok(())
    }
}

/// Print to stdout (CI mode)
pub fn print_stdout(result: &AttentionResult) {
    println!("{result}");
}

/// Render TUI (same as stdout for now)
pub fn render_tui(result: &AttentionResult) {
    println!("{result}");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Constants tests ==========

    #[test]
    fn test_constants_dimensions() {
        assert_eq!(TOKENS.len(), SEQ_LEN);
        assert_eq!(INPUT_EMBEDDINGS.len(), SEQ_LEN);
        assert_eq!(INPUT_EMBEDDINGS[0].len(), EMBED_DIM);
        assert_eq!(W_Q.len(), EMBED_DIM);
        assert_eq!(W_Q[0].len(), EMBED_DIM);
    }

    #[test]
    fn test_demo_sentence_matches_tokens() {
        let words: Vec<&str> = DEMO_SENTENCE.split_whitespace().collect();
        assert_eq!(words, TOKENS);
    }

    // ========== Matrix operation tests ==========

    #[test]
    fn test_matmul_identity() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = matmul(&a, &identity);
        assert_eq!(result, a);
    }

    #[test]
    fn test_matmul_simple() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![3.0], vec![4.0]];
        let result = matmul(&a, &b);
        assert_eq!(result, vec![vec![11.0]]); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_transpose_square() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let t = transpose(&m);
        assert_eq!(t, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    }

    #[test]
    fn test_transpose_rectangular() {
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = transpose(&m);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert_eq!(t[0], vec![1.0, 4.0]);
    }

    #[test]
    fn test_transpose_empty() {
        let m: Vec<Vec<f32>> = vec![];
        let t = transpose(&m);
        assert!(t.is_empty());
    }

    #[test]
    fn test_array_to_vec() {
        let arr = [[1.0, 2.0], [3.0, 4.0]];
        let v = array_to_vec(&arr);
        assert_eq!(v, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    // ========== Softmax tests ==========

    #[test]
    fn test_softmax_uniform() {
        let scores = vec![1.0, 1.0, 1.0];
        let probs = softmax(&scores);
        // All equal inputs → all equal outputs
        assert!((probs[0] - probs[1]).abs() < 1e-6);
        assert!((probs[1] - probs[2]).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax(&scores);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_preserves_order() {
        let scores = vec![1.0, 3.0, 2.0];
        let probs = softmax(&scores);
        // 3.0 should have highest prob
        assert!(probs[1] > probs[0]);
        assert!(probs[1] > probs[2]);
        // 2.0 should be second
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_softmax_extreme_values() {
        let scores = vec![0.0, 100.0, 0.0];
        let probs = softmax(&scores);
        // Middle value dominates
        assert!(probs[1] > 0.99);
        assert!(probs[0] < 0.01);
    }

    #[test]
    fn test_softmax_negative_values() {
        let scores = vec![-1.0, -2.0, -3.0];
        let probs = softmax(&scores);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // -1.0 should have highest prob
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_single_element() {
        let scores = vec![5.0];
        let probs = softmax(&scores);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    // ========== Softmax explanation tests ==========

    #[test]
    fn test_explain_softmax_structure() {
        let raw = vec![1.0, 2.0, 3.0];
        let step = explain_softmax(&raw);
        assert_eq!(step.raw, raw);
        assert_eq!(step.exp.len(), 3);
        assert!(step.sum > 0.0);
        assert_eq!(step.probs.len(), 3);
    }

    #[test]
    fn test_explain_softmax_probs_match_softmax() {
        let raw = vec![0.5, 1.5, 2.5];
        let step = explain_softmax(&raw);
        let direct = softmax(&raw);
        for (a, b) in step.probs.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ========== Run function tests ==========

    #[test]
    fn test_run_normal_mode() {
        let result = run(false);
        assert!(!result.skip_softmax);
        assert_eq!(result.inputs.len(), SEQ_LEN);
        assert_eq!(result.queries.len(), SEQ_LEN);
        assert_eq!(result.attention_weights.len(), SEQ_LEN);
    }

    #[test]
    fn test_run_error_mode() {
        let result = run(true);
        assert!(result.skip_softmax);
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let result = run(false);
        for row in &result.attention_weights {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row sum {} should be 1.0",
                sum
            );
        }
    }

    #[test]
    fn test_error_mode_weights_dont_sum_to_one() {
        let result = run(true);
        // In error mode, raw scores are used - they shouldn't sum to 1
        let any_not_one = result
            .attention_weights
            .iter()
            .any(|row| (row.iter().sum::<f32>() - 1.0).abs() > 0.1);
        assert!(any_not_one, "Error mode should have invalid row sums");
    }

    #[test]
    fn test_output_dimensions() {
        let result = run(false);
        assert_eq!(result.outputs.len(), SEQ_LEN);
        assert_eq!(result.outputs[0].len(), EMBED_DIM);
    }

    #[test]
    fn test_qkv_dimensions() {
        let result = run(false);
        assert_eq!(result.queries.len(), SEQ_LEN);
        assert_eq!(result.queries[0].len(), EMBED_DIM);
        assert_eq!(result.keys.len(), SEQ_LEN);
        assert_eq!(result.keys[0].len(), EMBED_DIM);
        assert_eq!(result.values.len(), SEQ_LEN);
        assert_eq!(result.values[0].len(), EMBED_DIM);
    }

    #[test]
    fn test_raw_scores_dimensions() {
        let result = run(false);
        // Q × K^T should be SEQ_LEN × SEQ_LEN
        assert_eq!(result.raw_scores.len(), SEQ_LEN);
        assert_eq!(result.raw_scores[0].len(), SEQ_LEN);
    }

    // ========== Display tests ==========

    #[test]
    fn test_fmt_vec() {
        let v = vec![1.0, 2.5, 3.0];
        let s = fmt_vec(&v, 1);
        assert_eq!(s, "[1.0, 2.5, 3.0]");
    }

    #[test]
    fn test_fmt_vec_precision() {
        let v = vec![1.234, 5.678];
        let s = fmt_vec(&v, 2);
        assert_eq!(s, "[1.23, 5.68]");
    }

    #[test]
    fn test_fmt_matrix_with_labels() {
        let m = vec![vec![1.0, 2.0]];
        let s = fmt_matrix(&m, 1, Some(&["A"]));
        assert!(s.contains("A"));
        assert!(s.contains("[1.0, 2.0]"));
    }

    #[test]
    fn test_fmt_matrix_without_labels() {
        let m = vec![vec![1.0, 2.0]];
        let s = fmt_matrix(&m, 1, None);
        assert!(s.contains("[1.0, 2.0]"));
    }

    #[test]
    fn test_display_normal_mode() {
        let result = run(false);
        let display = format!("{}", result);
        assert!(display.contains("SELF-ATTENTION"));
        assert!(display.contains("SOFTMAX"));
        assert!(display.contains("softmax(x)"));
    }

    #[test]
    fn test_display_error_mode() {
        let result = run(true);
        let display = format!("{}", result);
        assert!(display.contains("SKIPPED"));
        assert!(display.contains("ERROR"));
    }

    #[test]
    fn test_display_contains_tokens() {
        let result = run(false);
        let display = format!("{}", result);
        for token in TOKENS {
            assert!(display.contains(token));
        }
    }

    // ========== Output function tests ==========

    #[test]
    fn test_print_stdout_runs() {
        let result = run(false);
        // Just verify it doesn't panic
        print_stdout(&result);
    }

    #[test]
    fn test_render_tui_runs() {
        let result = run(false);
        // Just verify it doesn't panic
        render_tui(&result);
    }

    // ========== Integration tests ==========

    #[test]
    fn test_attention_is_self_attention() {
        // Each token attends to itself and others
        let result = run(false);
        // Diagonal should have non-zero attention (self-attention)
        for i in 0..SEQ_LEN {
            assert!(
                result.attention_weights[i][i] > 0.0,
                "Token {} should attend to itself",
                TOKENS[i]
            );
        }
    }

    #[test]
    fn test_attention_weights_are_probabilities() {
        let result = run(false);
        for row in &result.attention_weights {
            for &w in row {
                assert!(w >= 0.0, "Weights should be non-negative");
                assert!(w <= 1.0, "Weights should be at most 1.0");
            }
        }
    }
}
