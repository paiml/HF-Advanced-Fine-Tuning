//! Demo 2: Training vs Inference - Why inference is slow
//!
//! Shows three operations that force sequential execution:
//! 1. Softmax - global reduction (must sum all before normalizing any)
//! 2. LayerNorm - global reduction (must compute μ,σ before normalizing)
//! 3. Autoregressive - token N+1 depends on token N

/// Softmax: exp(x_i) / sum(exp(x_j))
/// Global reduction - cannot compute any output until all inputs are seen
#[must_use]
pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    // Numerical stability: subtract max
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exp_vals.iter().map(|v| v / sum).collect()
}

/// LayerNorm: (x - μ) / σ
/// Global reduction - cannot normalize any element until μ,σ computed from all
#[must_use]
pub fn layernorm(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

/// LayerNorm stats for display
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNormStats {
    pub mean: f32,
    pub var: f32,
    pub std: f32,
}

/// Compute layernorm with stats for visualization
#[must_use]
pub fn layernorm_with_stats(x: &[f32], eps: f32) -> (Vec<f32>, LayerNormStats) {
    if x.is_empty() {
        return (
            vec![],
            LayerNormStats {
                mean: 0.0,
                var: 0.0,
                std: 0.0,
            },
        );
    }
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    let normalized = x.iter().map(|v| (v - mean) / std).collect();
    (normalized, LayerNormStats { mean, var, std })
}

/// Softmax stats for display
#[derive(Debug, Clone, PartialEq)]
pub struct SoftmaxStats {
    pub max: f32,
    pub sum_exp: f32,
}

/// Compute softmax with stats for visualization
#[must_use]
pub fn softmax_with_stats(x: &[f32]) -> (Vec<f32>, SoftmaxStats) {
    if x.is_empty() {
        return (
            vec![],
            SoftmaxStats {
                max: 0.0,
                sum_exp: 0.0,
            },
        );
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let result = if sum_exp == 0.0 {
        vec![1.0 / x.len() as f32; x.len()]
    } else {
        exp_vals.iter().map(|v| v / sum_exp).collect()
    };
    (result, SoftmaxStats { max, sum_exp })
}

/// Token for autoregressive demo
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub text: String,
}

impl Token {
    #[must_use]
    pub fn new(id: u32, text: &str) -> Self {
        Self {
            id,
            text: text.to_string(),
        }
    }
}

/// Simple vocabulary for demo
pub struct Vocab {
    tokens: Vec<String>,
}

impl Vocab {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: vec![
                "The".to_string(),
                "capital".to_string(),
                "of".to_string(),
                "France".to_string(),
                "is".to_string(),
                "Paris".to_string(),
                ".".to_string(),
            ],
        }
    }

    #[must_use]
    pub fn get(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated forward pass (toy model)
/// In real transformer: embedding → (attention+softmax → layernorm → FFN → layernorm) × 32
#[must_use]
pub fn toy_forward(tokens: &[u32], vocab_size: usize) -> Vec<f32> {
    // Simulate logits based on simple pattern matching
    let mut logits = vec![0.0_f32; vocab_size];

    // Hardcoded "The capital of France is Paris." pattern
    let next_token = match tokens {
        [] => 0,              // "The"
        [0] => 1,             // "capital"
        [0, 1] => 2,          // "of"
        [0, 1, 2] => 3,       // "France"
        [0, 1, 2, 3] => 4,    // "is"
        [0, 1, 2, 3, 4] => 5, // "Paris"
        _ => 6,               // "."
    };

    // Make the "correct" token have highest logit
    for (i, logit) in logits.iter_mut().enumerate() {
        *logit = if i == next_token {
            2.0
        } else {
            -1.0 + (i as f32) * 0.1
        };
    }

    logits
}

/// Sample from probability distribution (argmax for determinism in demo)
#[must_use]
pub fn sample_argmax(probs: &[f32]) -> u32 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Autoregressive generation step
#[derive(Debug, Clone)]
pub struct GenerationStep {
    pub step: usize,
    pub input_tokens: Vec<u32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub sampled_token: u32,
}

/// Generate tokens autoregressively with full trace
#[must_use]
pub fn generate_with_trace(prompt: &[u32], steps: usize, vocab_size: usize) -> Vec<GenerationStep> {
    let mut tokens = prompt.to_vec();
    let mut trace = Vec::new();

    for step in 0..steps {
        let logits = toy_forward(&tokens, vocab_size);
        let probs = softmax(&logits);
        let next = sample_argmax(&probs);

        trace.push(GenerationStep {
            step,
            input_tokens: tokens.clone(),
            logits: logits.clone(),
            probs,
            sampled_token: next,
        });

        tokens.push(next);
    }

    trace
}

/// Results for TUI display
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub softmax_input: Vec<f32>,
    pub softmax_output: Vec<f32>,
    pub softmax_stats: SoftmaxStats,
    pub layernorm_input: Vec<f32>,
    pub layernorm_output: Vec<f32>,
    pub layernorm_stats: LayerNormStats,
    pub generation_trace: Vec<GenerationStep>,
    pub vocab: Vec<String>,
}

/// Run the demo
#[must_use]
pub fn run() -> DemoResults {
    // Softmax demo
    let softmax_input = vec![2.1, 0.5, 1.8, 0.2];
    let (softmax_output, softmax_stats) = softmax_with_stats(&softmax_input);

    // LayerNorm demo
    let layernorm_input = vec![0.5, -0.2, 0.8, 0.1];
    let (layernorm_output, layernorm_stats) = layernorm_with_stats(&layernorm_input, 1e-5);

    // Autoregressive demo
    let vocab = Vocab::new();
    let generation_trace = generate_with_trace(&[], 6, vocab.len());

    DemoResults {
        softmax_input,
        softmax_output,
        softmax_stats,
        layernorm_input,
        layernorm_output,
        layernorm_stats,
        generation_trace,
        vocab: (0..vocab.len())
            .filter_map(|i| vocab.get(i as u32).map(|s| s.to_string()))
            .collect(),
    }
}

/// Print to stdout (CI mode)
pub fn print_stdout(results: &DemoResults) {
    println!("=== Training vs Inference ===\n");

    println!("--- SOFTMAX (Global Reduction) ---");
    println!("Input:  {:?}", results.softmax_input);
    println!(
        "Max:    {:.2} (subtract for stability)",
        results.softmax_stats.max
    );
    println!(
        "Sum:    {:.2} (must compute before dividing)",
        results.softmax_stats.sum_exp
    );
    println!(
        "Output: {:?}\n",
        results
            .softmax_output
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    println!("--- LAYERNORM (Global Reduction) ---");
    println!("Input:  {:?}", results.layernorm_input);
    println!(
        "Mean:   {:.3} (must see all values)",
        results.layernorm_stats.mean
    );
    println!(
        "Std:    {:.3} (must see all values)",
        results.layernorm_stats.std
    );
    println!(
        "Output: {:?}\n",
        results
            .layernorm_output
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    println!("--- AUTOREGRESSIVE (Sequential Dependency) ---");
    for step in &results.generation_trace {
        let prev: String = step
            .input_tokens
            .iter()
            .filter_map(|&id| results.vocab.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        let default = "?".to_string();
        let next = results
            .vocab
            .get(step.sampled_token as usize)
            .unwrap_or(&default);
        println!("Step {}: [{}] → \"{}\"", step.step + 1, prev, next);
    }
    println!("\nKey: Token N+1 CANNOT be computed until Token N exists.");
}

/// Render TUI
pub fn render_tui(results: &DemoResults) {
    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let yellow = "\x1b[33m";
    let cyan = "\x1b[36m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";
    let dim = "\x1b[90m";

    println!("\n{bold}=== Training vs Inference ==={reset}");
    println!("{dim}Why training is parallel but inference is sequential{reset}\n");

    // Softmax
    println!("{bold}{red}SOFTMAX{reset} {dim}(Global Reduction){reset}");
    println!("{dim}Can't normalize until you sum all values{reset}");
    print!("  Input:  ");
    for v in &results.softmax_input {
        print!("{yellow}[{:.1}]{reset} ", v);
    }
    println!();
    println!(
        "  {dim}→ max={:.2}, sum_exp={:.2}{reset}",
        results.softmax_stats.max, results.softmax_stats.sum_exp
    );
    print!("  Output: ");
    for v in &results.softmax_output {
        print!("{green}[{:.2}]{reset} ", v);
    }
    println!("\n");

    // LayerNorm
    println!("{bold}{yellow}LAYERNORM{reset} {dim}(Global Reduction){reset}");
    println!("{dim}Can't normalize until you compute μ,σ from all{reset}");
    print!("  Input:  ");
    for v in &results.layernorm_input {
        print!("{yellow}[{:.1}]{reset} ", v);
    }
    println!();
    println!(
        "  {dim}→ μ={:.3}, σ={:.3}{reset}",
        results.layernorm_stats.mean, results.layernorm_stats.std
    );
    print!("  Output: ");
    for v in &results.layernorm_output {
        print!("{green}[{:.2}]{reset} ", v);
    }
    println!("\n");

    // Autoregressive
    println!("{bold}{cyan}AUTOREGRESSIVE{reset} {dim}(Sequential Dependency){reset}");
    println!("{dim}Token N+1 cannot exist until Token N is sampled{reset}");
    print!("  ");
    let default_token = "?".to_string();
    for step in &results.generation_trace {
        let token = results
            .vocab
            .get(step.sampled_token as usize)
            .unwrap_or(&default_token);
        print!("{green}\"{}\"{reset}", token);
        if step.step < results.generation_trace.len() - 1 {
            print!(" {dim}→{reset} ");
        }
    }
    println!("\n");

    println!("{bold}TL;DR:{reset}");
    println!("  {green}Training:{reset}   All tokens at once → {bold}parallel{reset}");
    println!("  {red}Inference:{reset}  One token at a time → {bold}sequential{reset}");
    println!("  {dim}Each token = full 32-layer pass with softmax+layernorm barriers{reset}\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== SOFTMAX TESTS ====================

    #[test]
    fn test_softmax_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        assert_eq!(result.len(), 3);
        // Sum should be 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let x: Vec<f32> = vec![];
        let result = softmax(&x);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let x = vec![5.0];
        let result = softmax(&x);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_uniform() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let result = softmax(&x);
        for v in &result {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_ordering() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_softmax_with_stats() {
        let x = vec![2.1, 0.5, 1.8, 0.2];
        let (result, stats) = softmax_with_stats(&x);
        assert!((stats.max - 2.1).abs() < 1e-5);
        assert!(stats.sum_exp > 0.0);
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_with_stats_empty() {
        let x: Vec<f32> = vec![];
        let (result, stats) = softmax_with_stats(&x);
        assert!(result.is_empty());
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.sum_exp, 0.0);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let x = vec![1000.0, 1001.0, 1002.0];
        let result = softmax(&x);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ==================== LAYERNORM TESTS ====================

    #[test]
    fn test_layernorm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = layernorm(&x, 1e-5);
        assert_eq!(result.len(), 4);
        // Mean of normalized should be ~0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_layernorm_empty() {
        let x: Vec<f32> = vec![];
        let result = layernorm(&x, 1e-5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_layernorm_single() {
        let x = vec![5.0];
        let result = layernorm(&x, 1e-5);
        assert_eq!(result.len(), 1);
        // Single value normalized is 0 (x - mean = 0)
        assert!(result[0].abs() < 1e-3);
    }

    #[test]
    fn test_layernorm_uniform() {
        let x = vec![3.0, 3.0, 3.0, 3.0];
        let result = layernorm(&x, 1e-5);
        for v in &result {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn test_layernorm_with_stats() {
        let x = vec![0.5, -0.2, 0.8, 0.1];
        let (result, stats) = layernorm_with_stats(&x, 1e-5);
        assert_eq!(result.len(), 4);
        assert!((stats.mean - 0.3).abs() < 1e-5);
        assert!(stats.var > 0.0);
        assert!(stats.std > 0.0);
    }

    #[test]
    fn test_layernorm_with_stats_empty() {
        let x: Vec<f32> = vec![];
        let (result, stats) = layernorm_with_stats(&x, 1e-5);
        assert!(result.is_empty());
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.var, 0.0);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_layernorm_std_is_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = layernorm(&x, 1e-5);
        // Std of normalized should be ~1
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        let var: f32 = result.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / result.len() as f32;
        let std = var.sqrt();
        assert!((std - 1.0).abs() < 1e-4);
    }

    // ==================== TOKEN/VOCAB TESTS ====================

    #[test]
    fn test_token_new() {
        let t = Token::new(5, "Paris");
        assert_eq!(t.id, 5);
        assert_eq!(t.text, "Paris");
    }

    #[test]
    fn test_vocab_new() {
        let v = Vocab::new();
        assert!(!v.is_empty());
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_vocab_get() {
        let v = Vocab::new();
        assert_eq!(v.get(0), Some("The"));
        assert_eq!(v.get(5), Some("Paris"));
        assert_eq!(v.get(100), None);
    }

    #[test]
    fn test_vocab_default() {
        let v = Vocab::default();
        assert_eq!(v.len(), 7);
    }

    // ==================== GENERATION TESTS ====================

    #[test]
    fn test_toy_forward() {
        let logits = toy_forward(&[], 7);
        assert_eq!(logits.len(), 7);
        // First token should be highest
        assert!(logits[0] > logits[1]);
    }

    #[test]
    fn test_toy_forward_sequence() {
        let logits = toy_forward(&[0, 1, 2], 7);
        // Token 3 ("France") should be highest
        assert!(logits[3] > logits[0]);
        assert!(logits[3] > logits[1]);
    }

    #[test]
    fn test_sample_argmax() {
        let probs = vec![0.1, 0.2, 0.5, 0.2];
        assert_eq!(sample_argmax(&probs), 2);
    }

    #[test]
    fn test_sample_argmax_first() {
        let probs = vec![0.9, 0.05, 0.05];
        assert_eq!(sample_argmax(&probs), 0);
    }

    #[test]
    fn test_generate_with_trace() {
        let trace = generate_with_trace(&[], 3, 7);
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0].step, 0);
        assert_eq!(trace[1].step, 1);
        assert_eq!(trace[2].step, 2);
    }

    #[test]
    fn test_generate_with_trace_tokens() {
        let trace = generate_with_trace(&[], 6, 7);
        // Should generate "The capital of France is Paris"
        assert_eq!(trace[0].sampled_token, 0); // The
        assert_eq!(trace[1].sampled_token, 1); // capital
        assert_eq!(trace[2].sampled_token, 2); // of
        assert_eq!(trace[3].sampled_token, 3); // France
        assert_eq!(trace[4].sampled_token, 4); // is
        assert_eq!(trace[5].sampled_token, 5); // Paris
    }

    #[test]
    fn test_generate_trace_has_probs() {
        let trace = generate_with_trace(&[], 1, 7);
        assert!(!trace[0].probs.is_empty());
        let sum: f32 = trace[0].probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ==================== DEMO RESULTS TESTS ====================

    #[test]
    fn test_run() {
        let results = run();
        assert!(!results.softmax_input.is_empty());
        assert!(!results.softmax_output.is_empty());
        assert!(!results.layernorm_input.is_empty());
        assert!(!results.layernorm_output.is_empty());
        assert!(!results.generation_trace.is_empty());
        assert!(!results.vocab.is_empty());
    }

    #[test]
    fn test_run_softmax_sums_to_one() {
        let results = run();
        let sum: f32 = results.softmax_output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_run_layernorm_mean_zero() {
        let results = run();
        let mean: f32 =
            results.layernorm_output.iter().sum::<f32>() / results.layernorm_output.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    // ==================== OUTPUT TESTS ====================

    #[test]
    fn test_print_stdout_no_panic() {
        let results = run();
        print_stdout(&results);
    }

    #[test]
    fn test_render_tui_no_panic() {
        let results = run();
        render_tui(&results);
    }

    // ==================== STRUCT EQUALITY TESTS ====================

    #[test]
    fn test_softmax_stats_eq() {
        let a = SoftmaxStats {
            max: 1.0,
            sum_exp: 2.0,
        };
        let b = SoftmaxStats {
            max: 1.0,
            sum_exp: 2.0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_layernorm_stats_eq() {
        let a = LayerNormStats {
            mean: 1.0,
            var: 2.0,
            std: 3.0,
        };
        let b = LayerNormStats {
            mean: 1.0,
            var: 2.0,
            std: 3.0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_token_eq() {
        let a = Token::new(1, "test");
        let b = Token::new(1, "test");
        assert_eq!(a, b);
    }

    #[test]
    fn test_generation_step_clone() {
        let step = GenerationStep {
            step: 0,
            input_tokens: vec![1, 2],
            logits: vec![0.1, 0.2],
            probs: vec![0.3, 0.7],
            sampled_token: 1,
        };
        let cloned = step.clone();
        assert_eq!(cloned.step, step.step);
        assert_eq!(cloned.input_tokens, step.input_tokens);
    }
}
