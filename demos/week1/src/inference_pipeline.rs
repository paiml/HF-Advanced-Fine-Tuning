//! Demo: Inference Pipeline - Step-by-step data flow through an LLM
//!
//! Shows how data transforms at each stage:
//! 1. TOKENIZE: string → token IDs
//! 2. EMBED: token IDs → vectors
//! 3. TRANSFORMER: vectors → vectors (×N layers)
//! 4. LM_HEAD: last vector → logits
//! 5. SAMPLE: logits → token ID
//! 6. DECODE: token ID → string

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Toy model configuration
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
}

impl Config {
    #[must_use]
    pub fn toy() -> Self {
        Self {
            vocab_size: 32,
            hidden_dim: 8,
            num_layers: 2,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::toy()
    }
}

// ============================================================================
// STEP 1: TOKENIZE
// ============================================================================

/// Tokenizer output
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizeOutput {
    pub input_text: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
}

/// Toy BPE-like tokenizer
/// Maps known tokens to IDs, unknown chars get ID 0 (UNK)
#[must_use]
pub fn tokenize(text: &str) -> TokenizeOutput {
    // Toy vocabulary (subset)
    let vocab: Vec<(&str, u32)> = vec![
        ("<unk>", 0),
        ("<pad>", 1),
        ("def", 2),
        (" ", 3),
        ("add", 4),
        ("(", 5),
        ("x", 6),
        (",", 7),
        ("y", 8),
        (")", 9),
        (":", 10),
        ("return", 11),
        ("+", 12),
        ("sub", 13),
        ("-", 14),
        ("mul", 15),
        ("*", 16),
        ("div", 17),
        ("/", 18),
        ("z", 19),
        ("a", 20),
        ("b", 21),
        ("c", 22),
        ("0", 23),
        ("1", 24),
        ("2", 25),
        ("\n", 26),
        ("\t", 27),
        ("    ", 28), // 4 spaces (indent)
        ("print", 29),
        ("if", 30),
        ("else", 31),
    ];

    let mut tokens = Vec::new();
    let mut token_ids = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Try longest match first
        let mut matched = false;
        for (tok, id) in vocab.iter().rev() {
            if remaining.starts_with(tok) {
                tokens.push(tok.to_string());
                token_ids.push(*id);
                remaining = &remaining[tok.len()..];
                matched = true;
                break;
            }
        }
        if !matched {
            // Unknown character - use UNK and skip one char
            tokens.push(remaining.chars().next().unwrap().to_string());
            token_ids.push(0); // UNK
            remaining = &remaining[1..];
        }
    }

    TokenizeOutput {
        input_text: text.to_string(),
        tokens,
        token_ids,
    }
}

// ============================================================================
// STEP 2: EMBED
// ============================================================================

/// Embedding output
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedOutput {
    pub token_ids: Vec<u32>,
    pub embeddings: Vec<Vec<f32>>,
    pub shape: (usize, usize), // (seq_len, hidden_dim)
}

/// Toy embedding lookup
/// Creates deterministic pseudo-embeddings based on token ID
#[must_use]
pub fn embed(token_ids: &[u32], config: &Config) -> EmbedOutput {
    let embeddings: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&id| {
            // Deterministic pseudo-embedding
            (0..config.hidden_dim)
                .map(|i| {
                    let seed = (id as f32 + 1.0) * (i as f32 + 1.0);
                    (seed.sin() * 0.5).clamp(-1.0, 1.0)
                })
                .collect()
        })
        .collect();

    let shape = (token_ids.len(), config.hidden_dim);

    EmbedOutput {
        token_ids: token_ids.to_vec(),
        embeddings,
        shape,
    }
}

// ============================================================================
// STEP 3: TRANSFORMER LAYERS
// ============================================================================

/// Single transformer layer output
#[derive(Debug, Clone, PartialEq)]
pub struct LayerOutput {
    pub layer_num: usize,
    pub input_shape: (usize, usize),
    pub attention_weights: Vec<Vec<f32>>, // [seq_len, seq_len]
    pub post_attention: Vec<Vec<f32>>,
    pub post_ffn: Vec<Vec<f32>>,
    pub output_shape: (usize, usize),
}

/// Transformer output (all layers)
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerOutput {
    pub input_shape: (usize, usize),
    pub layers: Vec<LayerOutput>,
    pub final_hidden: Vec<Vec<f32>>,
    pub output_shape: (usize, usize),
}

/// Toy softmax for attention
#[must_use]
pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exp_vals.iter().map(|v| v / sum).collect()
}

/// Toy attention mechanism
#[must_use]
pub fn toy_attention(hidden: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let seq_len = hidden.len();
    if seq_len == 0 {
        return (vec![], vec![]);
    }

    // Compute attention weights (dot product + softmax)
    let mut attn_weights = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let mut scores = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            // Causal: only attend to previous tokens
            if j <= i {
                let dot: f32 = hidden[i]
                    .iter()
                    .zip(hidden[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                scores.push(dot / (hidden[0].len() as f32).sqrt());
            } else {
                scores.push(f32::NEG_INFINITY); // Mask future
            }
        }
        attn_weights.push(softmax(&scores));
    }

    // Apply attention
    let hidden_dim = hidden[0].len();
    let output: Vec<Vec<f32>> = attn_weights
        .iter()
        .enumerate()
        .map(|(i, weights)| {
            let mut out_vec = vec![0.0; hidden_dim];
            for (j, &weight) in weights.iter().enumerate().take(i + 1) {
                for (k, out_k) in out_vec.iter_mut().enumerate() {
                    *out_k += weight * hidden[j][k];
                }
            }
            out_vec
        })
        .collect();

    (attn_weights, output)
}

/// Toy FFN: expand → ReLU → contract
#[must_use]
pub fn toy_ffn(hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
    hidden
        .iter()
        .map(|h| {
            // Simple nonlinear transform
            h.iter()
                .map(|&v| {
                    let expanded = v * 2.0 + 0.1;
                    let activated = expanded.max(0.0); // ReLU
                    activated * 0.5 // Contract
                })
                .collect()
        })
        .collect()
}

/// Run transformer layers
#[must_use]
pub fn transformer(embeddings: &[Vec<f32>], config: &Config) -> TransformerOutput {
    let input_shape = (embeddings.len(), config.hidden_dim);
    let mut hidden = embeddings.to_vec();
    let mut layers = Vec::with_capacity(config.num_layers);

    for layer_num in 0..config.num_layers {
        let (attn_weights, post_attn) = toy_attention(&hidden);
        let post_ffn = toy_ffn(&post_attn);

        layers.push(LayerOutput {
            layer_num,
            input_shape: (hidden.len(), config.hidden_dim),
            attention_weights: attn_weights,
            post_attention: post_attn,
            post_ffn: post_ffn.clone(),
            output_shape: (post_ffn.len(), config.hidden_dim),
        });

        hidden = post_ffn;
    }

    TransformerOutput {
        input_shape,
        layers,
        final_hidden: hidden.clone(),
        output_shape: (hidden.len(), config.hidden_dim),
    }
}

// ============================================================================
// STEP 4: LM HEAD
// ============================================================================

/// LM Head output
#[derive(Debug, Clone, PartialEq)]
pub struct LmHeadOutput {
    pub last_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub input_shape: usize,  // hidden_dim
    pub output_shape: usize, // vocab_size
}

/// Project last hidden state to vocabulary logits
#[must_use]
pub fn lm_head(final_hidden: &[Vec<f32>], config: &Config) -> LmHeadOutput {
    // Take last token's hidden state
    let last_hidden = final_hidden.last().cloned().unwrap_or_default();

    // Project to vocab size (toy linear projection)
    let logits: Vec<f32> = (0..config.vocab_size)
        .map(|v| {
            // Deterministic pseudo-projection
            last_hidden
                .iter()
                .enumerate()
                .map(|(i, &h)| {
                    let weight = ((v as f32 + 1.0) * (i as f32 + 1.0)).cos() * 0.3;
                    h * weight
                })
                .sum()
        })
        .collect();

    LmHeadOutput {
        last_hidden,
        logits,
        input_shape: config.hidden_dim,
        output_shape: config.vocab_size,
    }
}

// ============================================================================
// STEP 5: SAMPLE
// ============================================================================

/// Sample output
#[derive(Debug, Clone, PartialEq)]
pub struct SampleOutput {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub sampled_id: u32,
    pub top_k: Vec<(u32, f32)>, // (token_id, probability)
}

/// Sample next token (argmax for determinism)
#[must_use]
pub fn sample(logits: &[f32], inject_error: bool) -> SampleOutput {
    let probabilities = softmax(logits);

    // Get top-k
    let mut indexed: Vec<(u32, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as u32, p))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k: Vec<(u32, f32)> = indexed.into_iter().take(5).collect();

    // Argmax
    let sampled_id = if inject_error {
        // APR-TOK-001: Return out-of-bounds token ID
        99999
    } else {
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    };

    SampleOutput {
        logits: logits.to_vec(),
        probabilities,
        sampled_id,
        top_k,
    }
}

// ============================================================================
// STEP 6: DECODE
// ============================================================================

/// Decode output
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeOutput {
    pub token_id: u32,
    pub token_text: Option<String>,
    pub error: Option<String>,
}

/// Decode token ID back to string
#[must_use]
pub fn decode(token_id: u32, vocab_size: usize) -> DecodeOutput {
    // Same vocab as tokenize
    let vocab: Vec<&str> = vec![
        "<unk>", "<pad>", "def", " ", "add", "(", "x", ",", "y", ")", ":", "return", "+", "sub",
        "-", "mul", "*", "div", "/", "z", "a", "b", "c", "0", "1", "2", "\n", "\t", "    ",
        "print", "if", "else",
    ];

    if (token_id as usize) >= vocab_size {
        DecodeOutput {
            token_id,
            token_text: None,
            error: Some(format!(
                "Token ID {} out of bounds (vocab_size={})",
                token_id, vocab_size
            )),
        }
    } else if let Some(&text) = vocab.get(token_id as usize) {
        DecodeOutput {
            token_id,
            token_text: Some(text.to_string()),
            error: None,
        }
    } else {
        DecodeOutput {
            token_id,
            token_text: None,
            error: Some(format!("Token ID {} not in vocabulary", token_id)),
        }
    }
}

// ============================================================================
// FULL PIPELINE
// ============================================================================

/// Complete pipeline trace
#[derive(Debug, Clone)]
pub struct PipelineTrace {
    pub config: Config,
    pub tokenize: TokenizeOutput,
    pub embed: EmbedOutput,
    pub transformer: TransformerOutput,
    pub lm_head: LmHeadOutput,
    pub sample: SampleOutput,
    pub decode: DecodeOutput,
}

/// Run the full pipeline
#[must_use]
pub fn run_pipeline(text: &str, inject_error: bool) -> PipelineTrace {
    let config = Config::toy();

    // Step 1: Tokenize
    let tokenize_out = tokenize(text);

    // Step 2: Embed
    let embed_out = embed(&tokenize_out.token_ids, &config);

    // Step 3: Transformer
    let transformer_out = transformer(&embed_out.embeddings, &config);

    // Step 4: LM Head
    let lm_head_out = lm_head(&transformer_out.final_hidden, &config);

    // Step 5: Sample
    let sample_out = sample(&lm_head_out.logits, inject_error);

    // Step 6: Decode
    let decode_out = decode(sample_out.sampled_id, config.vocab_size);

    PipelineTrace {
        config,
        tokenize: tokenize_out,
        embed: embed_out,
        transformer: transformer_out,
        lm_head: lm_head_out,
        sample: sample_out,
        decode: decode_out,
    }
}

// ============================================================================
// TUI RENDERING
// ============================================================================

/// Format a vector for display (truncated)
fn fmt_vec(v: &[f32], max_items: usize) -> String {
    let items: Vec<String> = v
        .iter()
        .take(max_items)
        .map(|x| format!("{:.2}", x))
        .collect();
    if v.len() > max_items {
        format!("[{}, ...]", items.join(", "))
    } else {
        format!("[{}]", items.join(", "))
    }
}

/// Render the pipeline trace to TUI
pub fn render_tui(trace: &PipelineTrace) {
    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let yellow = "\x1b[33m";
    let cyan = "\x1b[36m";
    let magenta = "\x1b[35m";
    let blue = "\x1b[34m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";
    let dim = "\x1b[90m";

    println!("\n{bold}=== Inference Pipeline ==={reset}");
    println!("{dim}Data flow through an LLM, step by step{reset}");
    println!(
        "{dim}Config: vocab_size={}, hidden_dim={}, layers={}{reset}\n",
        trace.config.vocab_size, trace.config.hidden_dim, trace.config.num_layers
    );

    // Step 1: Tokenize
    println!("{bold}{cyan}[1/6] TOKENIZE{reset}");
    println!("  {dim}string → token IDs{reset}");
    println!("  Input:  {yellow}\"{}\"{reset}", trace.tokenize.input_text);
    println!("  Tokens: {green}{:?}{reset}", trace.tokenize.tokens);
    println!("  IDs:    {green}{:?}{reset}", trace.tokenize.token_ids);
    println!(
        "  Shape:  {dim}[{}]{reset}\n",
        trace.tokenize.token_ids.len()
    );

    // Step 2: Embed
    println!("{bold}{blue}[2/6] EMBED{reset}");
    println!("  {dim}token IDs → vectors{reset}");
    println!("  Input:  {dim}{:?}{reset}", trace.embed.token_ids);
    println!("  Output: {dim}(showing first token's embedding){reset}");
    if let Some(first) = trace.embed.embeddings.first() {
        println!("          {green}{}{reset}", fmt_vec(first, 6));
    }
    println!(
        "  Shape:  {dim}[{}, {}] (seq_len, hidden_dim){reset}\n",
        trace.embed.shape.0, trace.embed.shape.1
    );

    // Step 3: Transformer
    println!("{bold}{magenta}[3/6] TRANSFORMER{reset}");
    println!(
        "  {dim}vectors → vectors (×{} layers){reset}",
        trace.config.num_layers
    );
    for layer in &trace.transformer.layers {
        println!("  Layer {}:", layer.layer_num + 1);
        println!(
            "    Attention: {dim}[{}, {}] weights{reset}",
            layer.attention_weights.len(),
            layer
                .attention_weights
                .first()
                .map(|v| v.len())
                .unwrap_or(0)
        );
        if let Some(last_attn) = layer.attention_weights.last() {
            println!("    Last row:  {yellow}{}{reset}", fmt_vec(last_attn, 6));
        }
        if let Some(last_ffn) = layer.post_ffn.last() {
            println!("    FFN out:   {green}{}{reset}", fmt_vec(last_ffn, 6));
        }
    }
    println!(
        "  Shape:  {dim}[{}, {}] → [{}, {}]{reset}\n",
        trace.transformer.input_shape.0,
        trace.transformer.input_shape.1,
        trace.transformer.output_shape.0,
        trace.transformer.output_shape.1
    );

    // Step 4: LM Head
    println!("{bold}{yellow}[4/6] LM_HEAD{reset}");
    println!("  {dim}last hidden → logits{reset}");
    println!(
        "  Input:  {dim}{}{reset}",
        fmt_vec(&trace.lm_head.last_hidden, 6)
    );
    println!(
        "  Output: {green}{}{reset}",
        fmt_vec(&trace.lm_head.logits, 6)
    );
    println!(
        "  Shape:  {dim}[{}] → [{}]{reset}\n",
        trace.lm_head.input_shape, trace.lm_head.output_shape
    );

    // Step 5: Sample
    println!("{bold}{green}[5/6] SAMPLE{reset}");
    println!("  {dim}logits → token ID{reset}");
    println!("  Top-5 probabilities:");
    for (id, prob) in &trace.sample.top_k {
        let token = decode(*id, trace.config.vocab_size);
        let token_str = token.token_text.unwrap_or_else(|| "?".to_string());
        println!(
            "    {dim}ID {:2}{reset} = {yellow}{:.1}%{reset}  {dim}\"{}\"{reset}",
            id,
            prob * 100.0,
            token_str
        );
    }
    println!("  Sampled: {green}ID {}{reset}\n", trace.sample.sampled_id);

    // Step 6: Decode
    println!("{bold}[6/6] DECODE{reset}");
    println!("  {dim}token ID → string{reset}");
    println!("  Input:  {dim}ID {}{reset}", trace.decode.token_id);
    if let Some(ref text) = trace.decode.token_text {
        println!("  Output: {green}\"{}\"{reset}", text);
    }
    if let Some(ref err) = trace.decode.error {
        println!("  {red}ERROR: {}{reset}", err);
        println!("  {red}This is APR-TOK-001: vocab mismatch!{reset}");
    }
    println!();

    // Summary
    println!("{bold}PIPELINE SUMMARY{reset}");
    println!("  {cyan}\"{}\" {reset}{dim}→{reset} {:?} {dim}→{reset} [{},{}] {dim}→{reset} [{},{}] {dim}→{reset} [{}] {dim}→{reset} ID {} {dim}→{reset} {}",
        trace.tokenize.input_text,
        trace.tokenize.token_ids,
        trace.embed.shape.0, trace.embed.shape.1,
        trace.transformer.output_shape.0, trace.transformer.output_shape.1,
        trace.lm_head.output_shape,
        trace.sample.sampled_id,
        trace.decode.token_text.as_deref().unwrap_or("ERROR")
    );
    println!();
}

/// Print to stdout (CI mode)
pub fn print_stdout(trace: &PipelineTrace) {
    println!("=== Inference Pipeline ===\n");
    println!("Input: \"{}\"", trace.tokenize.input_text);
    println!();
    println!(
        "[1] TOKENIZE: \"{}\" → {:?}",
        trace.tokenize.input_text, trace.tokenize.token_ids
    );
    println!(
        "[2] EMBED:    {:?} → [{}, {}]",
        trace.tokenize.token_ids, trace.embed.shape.0, trace.embed.shape.1
    );
    println!(
        "[3] TRANSFORMER: [{}, {}] → [{}, {}] (×{} layers)",
        trace.transformer.input_shape.0,
        trace.transformer.input_shape.1,
        trace.transformer.output_shape.0,
        trace.transformer.output_shape.1,
        trace.config.num_layers
    );
    println!(
        "[4] LM_HEAD:  [{}] → [{}]",
        trace.lm_head.input_shape, trace.lm_head.output_shape
    );
    println!(
        "[5] SAMPLE:   [{}] → ID {}",
        trace.lm_head.output_shape, trace.sample.sampled_id
    );
    if let Some(ref text) = trace.decode.token_text {
        println!("[6] DECODE:   ID {} → \"{}\"", trace.decode.token_id, text);
    } else if let Some(ref err) = trace.decode.error {
        println!(
            "[6] DECODE:   ID {} → ERROR: {}",
            trace.decode.token_id, err
        );
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== CONFIG TESTS ====================

    #[test]
    fn test_config_toy() {
        let config = Config::toy();
        assert_eq!(config.vocab_size, 32);
        assert_eq!(config.hidden_dim, 8);
        assert_eq!(config.num_layers, 2);
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config, Config::toy());
    }

    #[test]
    fn test_config_clone() {
        let config = Config::toy();
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    // ==================== TOKENIZE TESTS ====================

    #[test]
    fn test_tokenize_basic() {
        let out = tokenize("def add");
        assert_eq!(out.tokens, vec!["def", " ", "a", "d", "d"]);
    }

    #[test]
    fn test_tokenize_full_prompt() {
        let out = tokenize("def add(x,y):");
        assert!(!out.token_ids.is_empty());
        assert_eq!(out.input_text, "def add(x,y):");
    }

    #[test]
    fn test_tokenize_empty() {
        let out = tokenize("");
        assert!(out.tokens.is_empty());
        assert!(out.token_ids.is_empty());
    }

    #[test]
    fn test_tokenize_unknown() {
        let out = tokenize("@#$");
        // All should be UNK (0)
        assert!(out.token_ids.iter().all(|&id| id == 0));
    }

    #[test]
    fn test_tokenize_known_tokens() {
        let out = tokenize("def");
        assert_eq!(out.token_ids, vec![2]); // "def" = 2
    }

    #[test]
    fn test_tokenize_output_eq() {
        let a = tokenize("x");
        let b = tokenize("x");
        assert_eq!(a, b);
    }

    // ==================== EMBED TESTS ====================

    #[test]
    fn test_embed_basic() {
        let config = Config::toy();
        let out = embed(&[2, 3, 4], &config);
        assert_eq!(out.embeddings.len(), 3);
        assert_eq!(out.embeddings[0].len(), config.hidden_dim);
    }

    #[test]
    fn test_embed_shape() {
        let config = Config::toy();
        let out = embed(&[1, 2, 3, 4, 5], &config);
        assert_eq!(out.shape, (5, config.hidden_dim));
    }

    #[test]
    fn test_embed_empty() {
        let config = Config::toy();
        let out = embed(&[], &config);
        assert!(out.embeddings.is_empty());
        assert_eq!(out.shape, (0, config.hidden_dim));
    }

    #[test]
    fn test_embed_deterministic() {
        let config = Config::toy();
        let a = embed(&[5], &config);
        let b = embed(&[5], &config);
        assert_eq!(a.embeddings, b.embeddings);
    }

    #[test]
    fn test_embed_different_ids() {
        let config = Config::toy();
        let a = embed(&[1], &config);
        let b = embed(&[2], &config);
        assert_ne!(a.embeddings, b.embeddings);
    }

    #[test]
    fn test_embed_output_eq() {
        let config = Config::toy();
        let a = embed(&[1, 2], &config);
        let b = embed(&[1, 2], &config);
        assert_eq!(a, b);
    }

    // ==================== SOFTMAX TESTS ====================

    #[test]
    fn test_softmax_sums_to_one() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let result = softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_ordering() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    // ==================== ATTENTION TESTS ====================

    #[test]
    fn test_toy_attention_basic() {
        let hidden = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (weights, output) = toy_attention(&hidden);
        assert_eq!(weights.len(), 2);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_toy_attention_empty() {
        let (weights, output) = toy_attention(&[]);
        assert!(weights.is_empty());
        assert!(output.is_empty());
    }

    #[test]
    fn test_toy_attention_causal() {
        let hidden = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let (weights, _) = toy_attention(&hidden);
        // First token can only attend to itself
        assert!((weights[0][0] - 1.0).abs() < 1e-5);
        // Future positions should be 0 (masked)
        assert!(weights[0][1] < 1e-5);
        assert!(weights[0][2] < 1e-5);
    }

    #[test]
    fn test_toy_attention_weights_sum_to_one() {
        let hidden = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let (weights, _) = toy_attention(&hidden);
        for row in &weights {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    // ==================== FFN TESTS ====================

    #[test]
    fn test_toy_ffn_basic() {
        let hidden = vec![vec![1.0, 2.0]];
        let output = toy_ffn(&hidden);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 2);
    }

    #[test]
    fn test_toy_ffn_empty() {
        let output = toy_ffn(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_toy_ffn_relu() {
        // Negative input should be zeroed by ReLU then halved
        let hidden = vec![vec![-1.0]];
        let output = toy_ffn(&hidden);
        // -1.0 * 2.0 + 0.1 = -1.9, ReLU = 0, * 0.5 = 0
        assert!((output[0][0] - 0.0).abs() < 1e-5);
    }

    // ==================== TRANSFORMER TESTS ====================

    #[test]
    fn test_transformer_basic() {
        let config = Config::toy();
        let embeddings = vec![vec![0.5; config.hidden_dim]; 3];
        let out = transformer(&embeddings, &config);
        assert_eq!(out.layers.len(), config.num_layers);
    }

    #[test]
    fn test_transformer_shapes() {
        let config = Config::toy();
        let embeddings = vec![vec![0.5; config.hidden_dim]; 5];
        let out = transformer(&embeddings, &config);
        assert_eq!(out.input_shape, (5, config.hidden_dim));
        assert_eq!(out.output_shape, (5, config.hidden_dim));
    }

    #[test]
    fn test_transformer_layer_outputs() {
        let config = Config::toy();
        let embeddings = vec![vec![0.5; config.hidden_dim]; 2];
        let out = transformer(&embeddings, &config);
        for (i, layer) in out.layers.iter().enumerate() {
            assert_eq!(layer.layer_num, i);
            assert!(!layer.attention_weights.is_empty());
            assert!(!layer.post_ffn.is_empty());
        }
    }

    #[test]
    fn test_transformer_output_eq() {
        let config = Config::toy();
        let embeddings = vec![vec![0.5; config.hidden_dim]; 2];
        let a = transformer(&embeddings, &config);
        let b = transformer(&embeddings, &config);
        assert_eq!(a, b);
    }

    // ==================== LM HEAD TESTS ====================

    #[test]
    fn test_lm_head_basic() {
        let config = Config::toy();
        let hidden = vec![vec![0.5; config.hidden_dim]];
        let out = lm_head(&hidden, &config);
        assert_eq!(out.logits.len(), config.vocab_size);
    }

    #[test]
    fn test_lm_head_shapes() {
        let config = Config::toy();
        let hidden = vec![vec![0.5; config.hidden_dim]];
        let out = lm_head(&hidden, &config);
        assert_eq!(out.input_shape, config.hidden_dim);
        assert_eq!(out.output_shape, config.vocab_size);
    }

    #[test]
    fn test_lm_head_empty() {
        let config = Config::toy();
        let out = lm_head(&[], &config);
        assert!(out.last_hidden.is_empty());
    }

    #[test]
    fn test_lm_head_output_eq() {
        let config = Config::toy();
        let hidden = vec![vec![0.5; config.hidden_dim]];
        let a = lm_head(&hidden, &config);
        let b = lm_head(&hidden, &config);
        assert_eq!(a, b);
    }

    // ==================== SAMPLE TESTS ====================

    #[test]
    fn test_sample_basic() {
        let logits = vec![1.0, 2.0, 5.0, 0.5];
        let out = sample(&logits, false);
        assert_eq!(out.sampled_id, 2); // Highest logit
    }

    #[test]
    fn test_sample_probabilities_sum_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let out = sample(&logits, false);
        let sum: f32 = out.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_top_k() {
        let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let out = sample(&logits, false);
        assert_eq!(out.top_k.len(), 5);
        assert_eq!(out.top_k[0].0, 1); // ID 1 has highest logit (5.0)
    }

    #[test]
    fn test_sample_error_injection() {
        let logits = vec![1.0, 2.0, 3.0];
        let out = sample(&logits, true);
        assert_eq!(out.sampled_id, 99999); // Out of bounds
    }

    #[test]
    fn test_sample_output_eq() {
        let logits = vec![1.0, 2.0, 3.0];
        let a = sample(&logits, false);
        let b = sample(&logits, false);
        assert_eq!(a, b);
    }

    // ==================== DECODE TESTS ====================

    #[test]
    fn test_decode_valid() {
        let out = decode(2, 32);
        assert_eq!(out.token_text, Some("def".to_string()));
        assert!(out.error.is_none());
    }

    #[test]
    fn test_decode_out_of_bounds() {
        let out = decode(99999, 32);
        assert!(out.token_text.is_none());
        assert!(out.error.is_some());
    }

    #[test]
    fn test_decode_unk() {
        let out = decode(0, 32);
        assert_eq!(out.token_text, Some("<unk>".to_string()));
    }

    #[test]
    fn test_decode_output_eq() {
        let a = decode(5, 32);
        let b = decode(5, 32);
        assert_eq!(a, b);
    }

    // ==================== PIPELINE TESTS ====================

    #[test]
    fn test_run_pipeline_basic() {
        let trace = run_pipeline("def add(x,y):", false);
        assert!(!trace.tokenize.token_ids.is_empty());
        assert!(trace.decode.error.is_none());
    }

    #[test]
    fn test_run_pipeline_with_error() {
        let trace = run_pipeline("def", true);
        assert!(trace.decode.error.is_some());
    }

    #[test]
    fn test_run_pipeline_shapes_consistent() {
        let trace = run_pipeline("x + y", false);
        // Embed output shape should match transformer input
        assert_eq!(trace.embed.shape, trace.transformer.input_shape);
        // LM head input should be hidden_dim
        assert_eq!(trace.lm_head.input_shape, trace.config.hidden_dim);
        // LM head output should be vocab_size
        assert_eq!(trace.lm_head.output_shape, trace.config.vocab_size);
    }

    // ==================== RENDER TESTS ====================

    #[test]
    fn test_render_tui_no_panic() {
        let trace = run_pipeline("def add(x,y):", false);
        render_tui(&trace);
    }

    #[test]
    fn test_render_tui_with_error_no_panic() {
        let trace = run_pipeline("def", true);
        render_tui(&trace);
    }

    #[test]
    fn test_print_stdout_no_panic() {
        let trace = run_pipeline("def add(x,y):", false);
        print_stdout(&trace);
    }

    #[test]
    fn test_print_stdout_with_error_no_panic() {
        let trace = run_pipeline("def", true);
        print_stdout(&trace);
    }

    // ==================== FMT_VEC TESTS ====================

    #[test]
    fn test_fmt_vec_short() {
        let v = vec![1.0, 2.0, 3.0];
        let s = fmt_vec(&v, 5);
        assert_eq!(s, "[1.00, 2.00, 3.00]");
    }

    #[test]
    fn test_fmt_vec_truncated() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = fmt_vec(&v, 3);
        assert_eq!(s, "[1.00, 2.00, 3.00, ...]");
    }

    #[test]
    fn test_fmt_vec_empty() {
        let v: Vec<f32> = vec![];
        let s = fmt_vec(&v, 5);
        assert_eq!(s, "[]");
    }
}
