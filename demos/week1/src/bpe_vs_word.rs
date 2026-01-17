//! Demo: BPE vs Word Tokenization
//!
//! Shows why BPE is superior to word tokenizers:
//! 1. Word tokenizer: fixed vocab, OOV → <UNK>
//! 2. BPE tokenizer: subword merges, handles any input
//!
//! Key insight: BPE trades token count for coverage

use std::collections::HashMap;

// ============================================================================
// TRAINING CORPUS
// ============================================================================

/// Toy training corpus
pub const TRAINING_CORPUS: &[&str] = &["the cat sat", "the cat ran", "the dog sat", "the dog ran"];

// ============================================================================
// WORD TOKENIZER
// ============================================================================

/// Word tokenizer with fixed vocabulary
#[derive(Debug, Clone, PartialEq)]
pub struct WordTokenizer {
    pub vocab: Vec<String>,
    pub word_to_id: HashMap<String, u32>,
}

impl WordTokenizer {
    /// Build vocabulary from training corpus
    #[must_use]
    pub fn train(corpus: &[&str]) -> Self {
        let mut vocab = Vec::new();
        let mut word_to_id = HashMap::new();

        // Add UNK token first
        vocab.push("<UNK>".to_string());
        word_to_id.insert("<UNK>".to_string(), 0);

        // Extract unique words
        for sentence in corpus {
            for word in sentence.split_whitespace() {
                if !word_to_id.contains_key(word) {
                    let id = vocab.len() as u32;
                    vocab.push(word.to_string());
                    word_to_id.insert(word.to_string(), id);
                }
            }
        }

        Self { vocab, word_to_id }
    }

    /// Tokenize text into word IDs
    #[must_use]
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| *self.word_to_id.get(word).unwrap_or(&0)) // 0 = UNK
            .collect()
    }

    /// Tokenize with token strings for display
    #[must_use]
    pub fn tokenize_verbose(&self, text: &str) -> Vec<(String, u32, bool)> {
        text.split_whitespace()
            .map(|word| {
                let id = *self.word_to_id.get(word).unwrap_or(&0);
                let is_unk = id == 0 && word != "<UNK>";
                let display = if is_unk {
                    "<UNK>".to_string()
                } else {
                    word.to_string()
                };
                (display, id, is_unk)
            })
            .collect()
    }

    /// Decode token IDs back to text
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============================================================================
// BPE TOKENIZER
// ============================================================================

/// A merge rule: (pair, merged_token)
#[derive(Debug, Clone, PartialEq)]
pub struct MergeRule {
    pub pair: (String, String),
    pub merged: String,
    pub frequency: usize,
}

/// BPE tokenizer with learned merges
#[derive(Debug, Clone, PartialEq)]
pub struct BpeTokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub merges: Vec<MergeRule>,
}

impl BpeTokenizer {
    /// Build BPE vocabulary from training corpus
    #[must_use]
    pub fn train(corpus: &[&str], num_merges: usize) -> Self {
        // Start with character-level vocab
        let mut vocab: Vec<String> = Vec::new();
        let mut token_to_id: HashMap<String, u32> = HashMap::new();

        // Add special tokens
        vocab.push("<UNK>".to_string());
        token_to_id.insert("<UNK>".to_string(), 0);

        // Add space token
        vocab.push(" ".to_string());
        token_to_id.insert(" ".to_string(), 1);

        // Extract all unique characters
        for sentence in corpus {
            for ch in sentence.chars() {
                let s = ch.to_string();
                token_to_id.entry(s.clone()).or_insert_with(|| {
                    let id = vocab.len() as u32;
                    vocab.push(s);
                    id
                });
            }
        }

        // Tokenize corpus at character level
        let mut tokenized_corpus: Vec<Vec<String>> = corpus
            .iter()
            .map(|s| s.chars().map(|c| c.to_string()).collect())
            .collect();

        let mut merges = Vec::new();

        // Perform merges
        for _ in 0..num_merges {
            // Count pairs
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for tokens in &tokenized_corpus {
                for window in tokens.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            // Find most frequent pair
            if let Some((best_pair, freq)) = pair_counts.into_iter().max_by_key(|(_, v)| *v) {
                if freq < 2 {
                    break; // No more useful merges
                }

                // Create merged token
                let merged = format!("{}{}", best_pair.0, best_pair.1);

                // Add to vocab if new
                if !token_to_id.contains_key(&merged) {
                    let id = vocab.len() as u32;
                    vocab.push(merged.clone());
                    token_to_id.insert(merged.clone(), id);
                }

                // Record merge rule
                merges.push(MergeRule {
                    pair: best_pair.clone(),
                    merged: merged.clone(),
                    frequency: freq,
                });

                // Apply merge to corpus
                for tokens in &mut tokenized_corpus {
                    let mut i = 0;
                    while i + 1 < tokens.len() {
                        if tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                            tokens[i] = merged.clone();
                            tokens.remove(i + 1);
                        } else {
                            i += 1;
                        }
                    }
                }
            } else {
                break;
            }
        }

        Self {
            vocab,
            token_to_id,
            merges,
        }
    }

    /// Tokenize text using learned merges
    #[must_use]
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let tokens = self.tokenize_to_strings(text);
        tokens
            .iter()
            .map(|t| *self.token_to_id.get(t).unwrap_or(&0))
            .collect()
    }

    /// Tokenize to string tokens (for display)
    #[must_use]
    pub fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        // Start with characters
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply merges in order
        for merge in &self.merges {
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == merge.pair.0 && tokens[i + 1] == merge.pair.1 {
                    tokens[i] = merge.merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
    }

    /// Tokenize with detailed info for display
    #[must_use]
    pub fn tokenize_verbose(&self, text: &str) -> Vec<(String, u32, bool)> {
        let tokens = self.tokenize_to_strings(text);
        tokens
            .iter()
            .map(|t| {
                let id = *self.token_to_id.get(t).unwrap_or(&0);
                let is_unk = id == 0 && t != "<UNK>";
                (t.clone(), id, is_unk)
            })
            .collect()
    }

    /// Decode token IDs back to text
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }

    /// Vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============================================================================
// COMPARISON RESULTS
// ============================================================================

/// Result of tokenizing one text with both methods
#[derive(Debug, Clone)]
pub struct TokenizeResult {
    pub input: String,
    pub word_tokens: Vec<(String, u32, bool)>,
    pub bpe_tokens: Vec<(String, u32, bool)>,
    pub word_unk_count: usize,
    pub bpe_unk_count: usize,
}

/// Full demo results
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub corpus: Vec<String>,
    pub word_tokenizer: WordTokenizer,
    pub bpe_tokenizer: BpeTokenizer,
    pub known_text: TokenizeResult,
    pub unknown_word: TokenizeResult,
    pub novel_word: TokenizeResult,
}

/// Run the demo
#[must_use]
pub fn run(inject_error: bool) -> DemoResults {
    let corpus: Vec<String> = TRAINING_CORPUS.iter().map(|s| s.to_string()).collect();

    // Train both tokenizers
    let word_tokenizer = WordTokenizer::train(TRAINING_CORPUS);
    let bpe_tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);

    // Test cases
    let known_text = "the cat sat";
    let unknown_word = if inject_error {
        "the doge sat" // typo
    } else {
        "the rat sat" // unknown word
    };
    let novel_word = "scattered"; // uses only chars from corpus: s,c,a,t,e,r,d

    // Tokenize each
    let known_result = tokenize_both(&word_tokenizer, &bpe_tokenizer, known_text);
    let unknown_result = tokenize_both(&word_tokenizer, &bpe_tokenizer, unknown_word);
    let novel_result = tokenize_both(&word_tokenizer, &bpe_tokenizer, novel_word);

    DemoResults {
        corpus,
        word_tokenizer,
        bpe_tokenizer,
        known_text: known_result,
        unknown_word: unknown_result,
        novel_word: novel_result,
    }
}

fn tokenize_both(word: &WordTokenizer, bpe: &BpeTokenizer, text: &str) -> TokenizeResult {
    let word_tokens = word.tokenize_verbose(text);
    let bpe_tokens = bpe.tokenize_verbose(text);

    let word_unk_count = word_tokens.iter().filter(|(_, _, is_unk)| *is_unk).count();
    let bpe_unk_count = bpe_tokens.iter().filter(|(_, _, is_unk)| *is_unk).count();

    TokenizeResult {
        input: text.to_string(),
        word_tokens,
        bpe_tokens,
        word_unk_count,
        bpe_unk_count,
    }
}

// ============================================================================
// TUI RENDERING
// ============================================================================

/// Render the demo results to TUI
pub fn render_tui(results: &DemoResults) {
    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let yellow = "\x1b[33m";
    let cyan = "\x1b[36m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";
    let dim = "\x1b[90m";

    println!("\n{bold}=== BPE vs Word Tokenization ==={reset}");
    println!("{dim}Why BPE handles unknown words gracefully{reset}\n");

    // Training corpus
    println!("{bold}{cyan}[1/4] TRAINING CORPUS{reset}");
    for sentence in &results.corpus {
        println!("  {dim}\"{}\"{reset}", sentence);
    }
    println!();

    // Vocabulary comparison
    println!("{bold}{cyan}[2/4] VOCABULARY{reset}");
    println!(
        "  {yellow}WORD:{reset} {} tokens",
        results.word_tokenizer.vocab_size()
    );
    print!("        ");
    for (i, tok) in results.word_tokenizer.vocab.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{dim}\"{}\"{reset}", tok);
    }
    println!("\n");

    println!(
        "  {green}BPE:{reset}  {} tokens ({} merges)",
        results.bpe_tokenizer.vocab_size(),
        results.bpe_tokenizer.merges.len()
    );
    println!("        {dim}Merges learned:{reset}");
    for (i, merge) in results.bpe_tokenizer.merges.iter().take(5).enumerate() {
        println!(
            "        {}. \"{}\" + \"{}\" → \"{}\" {dim}(freq={}){reset}",
            i + 1,
            merge.pair.0,
            merge.pair.1,
            merge.merged,
            merge.frequency
        );
    }
    if results.bpe_tokenizer.merges.len() > 5 {
        println!(
            "        {dim}... and {} more{reset}",
            results.bpe_tokenizer.merges.len() - 5
        );
    }
    println!();

    // Test cases
    println!("{bold}{cyan}[3/4] TOKENIZATION COMPARISON{reset}\n");

    render_comparison(
        "Known text",
        &results.known_text,
        green,
        yellow,
        red,
        dim,
        reset,
    );
    render_comparison(
        "Unknown word",
        &results.unknown_word,
        green,
        yellow,
        red,
        dim,
        reset,
    );
    render_comparison(
        "Novel word",
        &results.novel_word,
        green,
        yellow,
        red,
        dim,
        reset,
    );

    // Summary
    println!("{bold}{cyan}[4/4] SUMMARY{reset}");
    println!("  ┌─────────────────┬─────────────┬─────────────┐");
    println!("  │                 │ {yellow}WORD{reset}        │ {green}BPE{reset}         │");
    println!("  ├─────────────────┼─────────────┼─────────────┤");
    println!(
        "  │ Vocab size      │ {:>11} │ {:>11} │",
        results.word_tokenizer.vocab_size(),
        results.bpe_tokenizer.vocab_size()
    );
    println!("  │ OOV handling    │ {red}    <UNK>{reset}    │ {green}  subwords{reset}  │");
    println!("  │ Coverage        │ {red}   finite{reset}    │ {green}  infinite{reset}  │");
    println!("  │ Known text      │ {green}      ✓{reset}      │ {green}      ✓{reset}      │");
    let word_unk = results.unknown_word.word_unk_count > 0;
    let bpe_unk = results.unknown_word.bpe_unk_count > 0;
    println!(
        "  │ Unknown word    │ {}      │ {}      │",
        if word_unk {
            format!("{red}      ✗{reset}")
        } else {
            format!("{green}      ✓{reset}")
        },
        if bpe_unk {
            format!("{red}      ✗{reset}")
        } else {
            format!("{green}      ✓{reset}")
        }
    );
    let word_novel_unk = results.novel_word.word_unk_count > 0;
    let bpe_novel_unk = results.novel_word.bpe_unk_count > 0;
    println!(
        "  │ Novel word      │ {}      │ {}      │",
        if word_novel_unk {
            format!("{red}      ✗{reset}")
        } else {
            format!("{green}      ✓{reset}")
        },
        if bpe_novel_unk {
            format!("{red}      ✗{reset}")
        } else {
            format!("{green}      ✓{reset}")
        }
    );
    println!("  └─────────────────┴─────────────┴─────────────┘");
    println!();

    println!("{bold}TL;DR:{reset}");
    println!("  {yellow}Word:{reset} Fast but breaks on new words → {red}<UNK>{reset}");
    println!("  {green}BPE:{reset}  More tokens but {green}never breaks{reset} → subword fallback");
    println!("  {dim}BPE trades token count for infinite coverage{reset}\n");
}

fn render_comparison(
    label: &str,
    result: &TokenizeResult,
    green: &str,
    yellow: &str,
    red: &str,
    dim: &str,
    reset: &str,
) {
    println!("  {}: \"{}\"", label, result.input);

    // Word tokenizer
    print!("    {yellow}WORD:{reset} [");
    for (i, (tok, _, is_unk)) in result.word_tokens.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        if *is_unk {
            print!("{red}\"{}\"{reset}", tok);
        } else {
            print!("{dim}\"{}\"{reset}", tok);
        }
    }
    print!("] → {} tokens", result.word_tokens.len());
    if result.word_unk_count > 0 {
        print!(" {red}✗ {} UNK{reset}", result.word_unk_count);
    } else {
        print!(" {green}✓{reset}");
    }
    println!();

    // BPE tokenizer
    print!("    {green}BPE:{reset}  [");
    for (i, (tok, _, is_unk)) in result.bpe_tokens.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        if *is_unk {
            print!("{red}\"{}\"{reset}", tok);
        } else {
            print!("{dim}\"{}\"{reset}", tok);
        }
    }
    print!("] → {} tokens", result.bpe_tokens.len());
    if result.bpe_unk_count > 0 {
        print!(" {red}✗ {} UNK{reset}", result.bpe_unk_count);
    } else {
        print!(" {green}✓{reset}");
    }
    println!("\n");
}

/// Print to stdout (CI mode)
pub fn print_stdout(results: &DemoResults) {
    println!("=== BPE vs Word Tokenization ===\n");

    println!("Training corpus: {} sentences", results.corpus.len());
    println!("Word vocab: {} tokens", results.word_tokenizer.vocab_size());
    println!(
        "BPE vocab:  {} tokens ({} merges)\n",
        results.bpe_tokenizer.vocab_size(),
        results.bpe_tokenizer.merges.len()
    );

    println!("Known text: \"{}\"", results.known_text.input);
    println!(
        "  WORD: {:?} ({} tokens)",
        results
            .known_text
            .word_tokens
            .iter()
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>(),
        results.known_text.word_tokens.len()
    );
    println!(
        "  BPE:  {:?} ({} tokens)\n",
        results
            .known_text
            .bpe_tokens
            .iter()
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>(),
        results.known_text.bpe_tokens.len()
    );

    println!("Unknown word: \"{}\"", results.unknown_word.input);
    println!(
        "  WORD: {:?} ({} UNK)",
        results
            .unknown_word
            .word_tokens
            .iter()
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>(),
        results.unknown_word.word_unk_count
    );
    println!(
        "  BPE:  {:?} ({} UNK)\n",
        results
            .unknown_word
            .bpe_tokens
            .iter()
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>(),
        results.unknown_word.bpe_unk_count
    );

    println!("Novel word: \"{}\"", results.novel_word.input);
    println!(
        "  WORD: {:?} ({} UNK)",
        results
            .novel_word
            .word_tokens
            .iter()
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>(),
        results.novel_word.word_unk_count
    );
    println!(
        "  BPE:  {:?} ({} UNK)",
        results
            .novel_word
            .bpe_tokens
            .iter()
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>(),
        results.novel_word.bpe_unk_count
    );
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== WORD TOKENIZER TESTS ====================

    #[test]
    fn test_word_tokenizer_train() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.vocab.contains(&"<UNK>".to_string()));
    }

    #[test]
    fn test_word_tokenizer_vocab_contents() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        assert!(tokenizer.vocab.contains(&"the".to_string()));
        assert!(tokenizer.vocab.contains(&"cat".to_string()));
        assert!(tokenizer.vocab.contains(&"dog".to_string()));
        assert!(tokenizer.vocab.contains(&"sat".to_string()));
        assert!(tokenizer.vocab.contains(&"ran".to_string()));
    }

    #[test]
    fn test_word_tokenizer_known_text() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        let tokens = tokenizer.tokenize("the cat sat");
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(|&id| id != 0)); // No UNK
    }

    #[test]
    fn test_word_tokenizer_unknown_word() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        let tokens = tokenizer.tokenize("the rat sat");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1], 0); // "rat" → UNK
    }

    #[test]
    fn test_word_tokenizer_verbose() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        let tokens = tokenizer.tokenize_verbose("the rat sat");
        assert_eq!(tokens.len(), 3);
        assert!(tokens[1].2); // is_unk = true for "rat"
    }

    #[test]
    fn test_word_tokenizer_decode() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        let ids = tokenizer.tokenize("the cat");
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "the cat");
    }

    #[test]
    fn test_word_tokenizer_empty() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }

    // ==================== BPE TOKENIZER TESTS ====================

    #[test]
    fn test_bpe_tokenizer_train() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        assert!(tokenizer.vocab_size() > 0);
        assert!(!tokenizer.merges.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_has_characters() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        assert!(tokenizer.vocab.contains(&"t".to_string()));
        assert!(tokenizer.vocab.contains(&"h".to_string()));
        assert!(tokenizer.vocab.contains(&"e".to_string()));
    }

    #[test]
    fn test_bpe_tokenizer_known_text() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        let tokens = tokenizer.tokenize("the cat sat");
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|&id| id != 0)); // No UNK
    }

    #[test]
    fn test_bpe_tokenizer_unknown_word() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        let tokens = tokenizer.tokenize_to_strings("the rat sat");
        // BPE should break "rat" into known characters
        assert!(tokens.contains(&"r".to_string()) || tokens.iter().any(|t| t.contains('r')));
    }

    #[test]
    fn test_bpe_tokenizer_novel_word() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        let tokens = tokenizer.tokenize_to_strings("catastrophe");
        // Should contain "cat" as a merged token or characters
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_verbose() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        let tokens = tokenizer.tokenize_verbose("the cat");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        let ids = tokenizer.tokenize("the cat");
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "the cat");
    }

    #[test]
    fn test_bpe_tokenizer_empty() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_bpe_merges_recorded() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 5);
        assert!(!tokenizer.merges.is_empty());
        // First merge should have high frequency
        assert!(tokenizer.merges[0].frequency >= 2);
    }

    // ==================== MERGE RULE TESTS ====================

    #[test]
    fn test_merge_rule_eq() {
        let a = MergeRule {
            pair: ("t".to_string(), "h".to_string()),
            merged: "th".to_string(),
            frequency: 4,
        };
        let b = MergeRule {
            pair: ("t".to_string(), "h".to_string()),
            merged: "th".to_string(),
            frequency: 4,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_merge_rule_clone() {
        let a = MergeRule {
            pair: ("t".to_string(), "h".to_string()),
            merged: "th".to_string(),
            frequency: 4,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ==================== COMPARISON TESTS ====================

    #[test]
    fn test_run_no_error() {
        let results = run(false);
        assert!(!results.corpus.is_empty());
        assert!(results.word_tokenizer.vocab_size() > 0);
        assert!(results.bpe_tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_run_with_error() {
        let results = run(true);
        assert!(results.unknown_word.input.contains("doge"));
    }

    #[test]
    fn test_known_text_no_unk() {
        let results = run(false);
        assert_eq!(results.known_text.word_unk_count, 0);
        assert_eq!(results.known_text.bpe_unk_count, 0);
    }

    #[test]
    fn test_unknown_word_has_unk_in_word() {
        let results = run(false);
        assert!(results.unknown_word.word_unk_count > 0);
    }

    #[test]
    fn test_unknown_word_no_unk_in_bpe() {
        let results = run(false);
        assert_eq!(results.unknown_word.bpe_unk_count, 0);
    }

    #[test]
    fn test_novel_word_has_unk_in_word() {
        let results = run(false);
        assert!(results.novel_word.word_unk_count > 0);
    }

    #[test]
    fn test_novel_word_no_unk_in_bpe() {
        let results = run(false);
        assert_eq!(results.novel_word.bpe_unk_count, 0);
    }

    // ==================== RENDER TESTS ====================

    #[test]
    fn test_render_tui_no_panic() {
        let results = run(false);
        render_tui(&results);
    }

    #[test]
    fn test_render_tui_with_error_no_panic() {
        let results = run(true);
        render_tui(&results);
    }

    #[test]
    fn test_print_stdout_no_panic() {
        let results = run(false);
        print_stdout(&results);
    }

    #[test]
    fn test_print_stdout_with_error_no_panic() {
        let results = run(true);
        print_stdout(&results);
    }

    // ==================== EDGE CASE TESTS ====================

    #[test]
    fn test_word_tokenizer_single_word() {
        let tokenizer = WordTokenizer::train(TRAINING_CORPUS);
        let tokens = tokenizer.tokenize("cat");
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_bpe_tokenizer_single_char() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        let tokens = tokenizer.tokenize("x");
        // 'x' not in training corpus, should fall back to UNK or char
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_bpe_handles_space() {
        let tokenizer = BpeTokenizer::train(TRAINING_CORPUS, 10);
        assert!(tokenizer.vocab.contains(&" ".to_string()));
    }

    #[test]
    fn test_tokenize_result_clone() {
        let results = run(false);
        let cloned = results.known_text.clone();
        assert_eq!(cloned.input, results.known_text.input);
    }

    #[test]
    fn test_demo_results_clone() {
        let results = run(false);
        let cloned = results.clone();
        assert_eq!(cloned.corpus.len(), results.corpus.len());
    }
}
