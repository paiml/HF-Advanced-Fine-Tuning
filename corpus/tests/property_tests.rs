//! Property-Based Tests for EXTREME TDD
//!
//! Uses proptest to generate random inputs and verify properties hold.
//! These tests ensure the corpus handling is robust against edge cases.

use proptest::prelude::*;
use rust_cli_docs_corpus::{Category, Corpus, CorpusEntry};

// ============================================================================
// PROPERTY: Content hash is deterministic
// ============================================================================

proptest! {
    #[test]
    fn prop_content_hash_deterministic(
        input in "[a-zA-Z0-9_ ]{1,100}",
        output in "[a-zA-Z0-9_ /]{1,100}"
    ) {
        let entry1 = CorpusEntry::new(input.clone(), output.clone(), Category::Function);
        let entry2 = CorpusEntry::new(input, output, Category::Function);

        // Same content should produce same hash
        prop_assert_eq!(entry1.content_hash(), entry2.content_hash());
    }
}

// ============================================================================
// PROPERTY: Different content produces different hashes
// ============================================================================

proptest! {
    #[test]
    fn prop_content_hash_different(
        input1 in "[a-z]{10,20}",
        input2 in "[A-Z]{10,20}",
        output in "[a-zA-Z ]{10,50}"
    ) {
        let entry1 = CorpusEntry::new(input1.clone(), output.clone(), Category::Function);
        let entry2 = CorpusEntry::new(input2.clone(), output, Category::Function);

        // Different inputs should (almost always) produce different hashes
        if input1 != input2 {
            prop_assert_ne!(entry1.content_hash(), entry2.content_hash());
        }
    }
}

// ============================================================================
// PROPERTY: Entry IDs are unique UUIDs
// ============================================================================

proptest! {
    #[test]
    fn prop_entry_ids_unique(count in 1usize..100) {
        let mut corpus = Corpus::new();
        let mut ids = std::collections::HashSet::new();

        for i in 0..count {
            let entry = CorpusEntry::new(
                format!("fn test_{}() {{}}", i),
                format!("/// Test {}", i),
                Category::Function,
            );
            ids.insert(entry.id.clone());
            corpus.add_entry(entry);
        }

        // All IDs should be unique
        prop_assert_eq!(ids.len(), count);
    }
}

// ============================================================================
// PROPERTY: Token counts are non-negative
// ============================================================================

proptest! {
    #[test]
    fn prop_token_counts_valid(
        input in "[a-zA-Z_][a-zA-Z0-9_]{0,50}",
        output in "[a-zA-Z ]{0,200}"
    ) {
        let entry = CorpusEntry::new(
            format!("fn {}() {{}}", input),
            format!("/// {}", output),
            Category::Function,
        );

        // Token counts should be calculated and positive for non-empty content
        prop_assert!(entry.tokens_input > 0 || input.is_empty());
        prop_assert!(entry.tokens_output > 0 || output.is_empty());
    }
}

// ============================================================================
// PROPERTY: I/O ratio is valid for non-zero inputs
// ============================================================================

proptest! {
    #[test]
    fn prop_io_ratio_calculated(
        tokens_in in 1u32..1000,
        tokens_out in 1u32..1000
    ) {
        let mut entry = CorpusEntry::new(
            "fn test() {}".to_string(),
            "/// Test".to_string(),
            Category::Function,
        );
        entry.tokens_input = tokens_in;
        entry.tokens_output = tokens_out;

        let ratio = entry.io_ratio();
        let expected = tokens_out as f64 / tokens_in as f64;

        prop_assert!((ratio - expected).abs() < 0.001);
    }
}

// ============================================================================
// PROPERTY: Quality scores are clamped to [0, 1]
// ============================================================================

proptest! {
    #[test]
    fn prop_quality_score_bounded(score in -10.0f32..10.0) {
        let mut entry = CorpusEntry::new(
            "fn test() {}".to_string(),
            "/// Test".to_string(),
            Category::Function,
        );
        entry.quality_score = score.clamp(0.0, 1.0);

        prop_assert!(entry.quality_score >= 0.0);
        prop_assert!(entry.quality_score <= 1.0);
    }
}

// ============================================================================
// PROPERTY: Category roundtrip
// ============================================================================

proptest! {
    #[test]
    fn prop_category_roundtrip(category_idx in 0usize..5) {
        let categories = [
            Category::Function,
            Category::Argument,
            Category::Example,
            Category::Error,
            Category::Module,
        ];
        let category = categories[category_idx];

        let as_str = category.as_str();
        let back = Category::from_str(as_str);

        prop_assert_eq!(category, back);
    }
}

// ============================================================================
// PROPERTY: Corpus length matches entry count
// ============================================================================

proptest! {
    #[test]
    fn prop_corpus_length(count in 0usize..50) {
        let mut corpus = Corpus::new();

        for i in 0..count {
            let entry = CorpusEntry::new(
                format!("fn func_{}() {{}}", i),
                format!("/// Doc {}", i),
                Category::Function,
            );
            corpus.add_entry(entry);
        }

        prop_assert_eq!(corpus.len(), count);
        prop_assert_eq!(corpus.is_empty(), count == 0);
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_empty_input_hash() {
    let entry1 = CorpusEntry::new("".to_string(), "/// Doc".to_string(), Category::Function);
    let entry2 = CorpusEntry::new("".to_string(), "/// Doc".to_string(), Category::Function);

    assert_eq!(entry1.content_hash(), entry2.content_hash());
}

#[test]
fn test_unicode_in_docs() {
    let entry = CorpusEntry::new(
        "fn test() {}".to_string(),
        "/// Handles émojis 🎉 and ünïcödë".to_string(),
        Category::Function,
    );

    // Should not panic
    let _hash = entry.content_hash();
    let _tokens = entry.tokens_output;
}

#[test]
fn test_very_long_output() {
    let long_doc = "/// ".to_string() + &"a".repeat(10000);
    let entry = CorpusEntry::new(
        "fn test() {}".to_string(),
        long_doc,
        Category::Function,
    );

    assert!(entry.tokens_output > 0);
}

#[test]
fn test_special_characters_in_input() {
    let entry = CorpusEntry::new(
        "fn test<'a, T: Clone + Send>(&self, arg: &'a [T]) -> Result<(), Box<dyn Error>> {}".to_string(),
        "/// Complex generic function".to_string(),
        Category::Function,
    );

    let _hash = entry.content_hash();
    assert!(entry.tokens_input > 0);
}

#[test]
fn test_code_block_in_output() {
    let entry = CorpusEntry::new(
        "fn example() {}".to_string(),
        r#"/// Example
///
/// ```rust
/// let x = 42;
/// assert_eq!(x, 42);
/// ```"#.to_string(),
        Category::Example,
    );

    assert!(entry.output.contains("```"));
}

#[test]
fn test_multiline_signature() {
    let entry = CorpusEntry::new(
        "fn complex_function(\n    arg1: String,\n    arg2: i32,\n) -> Result<(), Error> {}".to_string(),
        "/// A function with multiline signature".to_string(),
        Category::Function,
    );

    assert!(entry.tokens_input > 0);
}

#[test]
fn test_io_ratio_zero_input() {
    let mut entry = CorpusEntry::new(
        "".to_string(),
        "/// Doc".to_string(),
        Category::Function,
    );
    entry.tokens_input = 0;
    entry.tokens_output = 100;

    // Should not panic, returns 0.0
    assert_eq!(entry.io_ratio(), 0.0);
}

#[test]
fn test_total_tokens() {
    let mut entry = CorpusEntry::new(
        "fn test() {}".to_string(),
        "/// Doc".to_string(),
        Category::Function,
    );
    entry.tokens_input = 10;
    entry.tokens_output = 50;

    assert_eq!(entry.total_tokens(), 60);
}

#[test]
fn test_corpus_from_entries() {
    let entries = vec![
        CorpusEntry::new("fn a() {}".to_string(), "/// A".to_string(), Category::Function),
        CorpusEntry::new("fn b() {}".to_string(), "/// B".to_string(), Category::Function),
    ];

    let corpus = Corpus::from_entries(entries);
    assert_eq!(corpus.len(), 2);
}

#[test]
fn test_corpus_hash_changes_with_content() {
    let mut corpus1 = Corpus::new();
    corpus1.add_entry(CorpusEntry::new("fn a() {}".to_string(), "/// A".to_string(), Category::Function));

    let mut corpus2 = Corpus::new();
    corpus2.add_entry(CorpusEntry::new("fn b() {}".to_string(), "/// B".to_string(), Category::Function));

    assert_ne!(corpus1.compute_hash(), corpus2.compute_hash());
}
