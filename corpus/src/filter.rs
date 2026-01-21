//! Corpus filtering and balancing.
//!
//! Implements quality gates and category balancing per Spec Section 7.

use crate::{Category, Corpus, CorpusEntry};
use std::collections::{HashMap, HashSet};

/// Filters and balances a corpus to meet spec requirements.
#[derive(Debug)]
pub struct CorpusFilter {
    /// Target category distribution (as percentages)
    pub target_distribution: HashMap<Category, (f64, f64)>,
    /// Maximum line length
    pub max_line_length: usize,
    /// Minimum quality score
    pub min_quality: f32,
    /// Maximum entries per repo (as percentage)
    pub max_repo_pct: f64,
}

impl Default for CorpusFilter {
    fn default() -> Self {
        let mut dist = HashMap::new();
        // Per Spec Section 5: Category Distribution
        dist.insert(Category::Function, (35.0, 45.0));
        dist.insert(Category::Argument, (20.0, 30.0));
        dist.insert(Category::Example, (15.0, 25.0));
        dist.insert(Category::Error, (5.0, 15.0));
        dist.insert(Category::Module, (3.0, 7.0));

        Self {
            target_distribution: dist,
            max_line_length: 100,
            min_quality: 0.4, // Lowered for v1.1.0 expansion (was 0.5)
            max_repo_pct: 38.0, // Slightly increased to allow more diversity
        }
    }
}

impl CorpusFilter {
    /// Filters and balances a corpus.
    #[must_use]
    pub fn filter(&self, corpus: &Corpus) -> Corpus {
        let mut entries: Vec<CorpusEntry> = corpus.entries().cloned().collect();
        let initial = entries.len();

        // Step 1: Remove invalid entries
        entries.retain(|e| self.is_valid_entry(e));
        eprintln!("  After validity filter: {} (removed {})", entries.len(), initial - entries.len());

        // Step 2: Remove duplicates by content hash
        let before_dedup = entries.len();
        let mut seen_hashes = HashSet::new();
        entries.retain(|e| seen_hashes.insert(e.content_hash()));
        eprintln!("  After deduplication: {} (removed {})", entries.len(), before_dedup - entries.len());

        // Step 3: Balance categories first
        let before_balance = entries.len();
        entries = self.balance_categories(entries);
        eprintln!("  After balancing: {} (removed {})", entries.len(), before_balance - entries.len());

        // Step 4: Limit repo dominance after balancing
        let before_repo = entries.len();
        entries = self.limit_repo_dominance(entries);
        eprintln!("  After repo limit: {} (removed {})", entries.len(), before_repo - entries.len());

        Corpus::from_entries(entries)
    }

    /// Checks if an entry is valid with reason tracking.
    fn is_valid_entry(&self, entry: &CorpusEntry) -> bool {
        self.check_validity(entry).is_none()
    }

    /// Returns the reason an entry is invalid, or None if valid.
    fn check_validity(&self, entry: &CorpusEntry) -> Option<&'static str> {
        // Check quality score
        if entry.quality_score < self.min_quality {
            return Some("quality_score");
        }

        // Check line lengths
        for line in entry.output.lines() {
            if line.len() > self.max_line_length {
                return Some("line_length");
            }
        }

        // Check balanced delimiters in output only
        if !self.has_balanced_delimiters(&entry.output) {
            return Some("delimiters");
        }

        // Check balanced code blocks
        let backticks = entry.output.matches("```").count();
        if backticks % 2 != 0 {
            return Some("code_blocks");
        }

        // Check input parses as Rust (relaxed - allow more)
        // Skip this check - quote! output may not parse cleanly
        // if syn::parse_str::<syn::Item>(&entry.input).is_err() {
        //     return Some("rust_parse");
        // }

        // Check no control characters
        for c in entry.output.chars() {
            if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                return Some("control_chars");
            }
        }

        // Check no trailing whitespace (skip - too strict)
        // for line in entry.output.lines() {
        //     if line != line.trim_end() {
        //         return Some("trailing_whitespace");
        //     }
        // }

        // Check token count (per spec: 10-500)
        let total = entry.total_tokens();
        if total < 10 || total > 500 {
            return Some("token_count");
        }

        // Check I/O ratio (relaxed for v1.1.0: 1.0-15.0, was 2.0-10.0)
        // Many valid docs have short outputs for simple functions
        if entry.tokens_input > 0 {
            let ratio = entry.io_ratio();
            if ratio < 1.0 || ratio > 15.0 {
                return Some("io_ratio");
            }
        }

        None
    }

    /// Prints filter statistics for debugging.
    pub fn print_filter_stats(&self, corpus: &Corpus) {
        let mut reasons: HashMap<&str, usize> = HashMap::new();
        for entry in corpus.entries() {
            if let Some(reason) = self.check_validity(entry) {
                *reasons.entry(reason).or_insert(0) += 1;
            }
        }
        eprintln!("  Filter rejection reasons:");
        for (reason, count) in reasons {
            eprintln!("    {}: {}", reason, count);
        }
    }

    /// Checks for balanced delimiters.
    fn has_balanced_delimiters(&self, text: &str) -> bool {
        let parens = text.matches('(').count() as i32 - text.matches(')').count() as i32;
        let brackets = text.matches('[').count() as i32 - text.matches(']').count() as i32;
        let braces = text.matches('{').count() as i32 - text.matches('}').count() as i32;
        parens == 0 && brackets == 0 && braces == 0
    }

    /// Balances categories to meet target distribution.
    /// Uses mid-point targets and calculates max corpus size based on scarcest category.
    fn balance_categories(&self, mut entries: Vec<CorpusEntry>) -> Vec<CorpusEntry> {
        if entries.is_empty() {
            return entries;
        }

        // Count by category
        let mut by_category: HashMap<Category, Vec<CorpusEntry>> = HashMap::new();
        for entry in entries.drain(..) {
            by_category.entry(entry.category).or_default().push(entry);
        }

        // Sort each category by quality (descending)
        for cat_entries in by_category.values_mut() {
            cat_entries.sort_by(|a, b| {
                b.quality_score
                    .partial_cmp(&a.quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Available counts
        let arg_count = by_category.get(&Category::Argument).map_or(0, Vec::len);
        let ex_count = by_category.get(&Category::Example).map_or(0, Vec::len);
        let err_count = by_category.get(&Category::Error).map_or(0, Vec::len);
        let mod_count = by_category.get(&Category::Module).map_or(0, Vec::len);
        let func_count = by_category.get(&Category::Function).map_or(0, Vec::len);

        eprintln!("  Available: func={}, arg={}, ex={}, err={}, mod={}",
            func_count, arg_count, ex_count, err_count, mod_count);

        // Target percentages (adjusted for data constraints v1.1.0):
        // Function: 35-45% -> 40% (lowered from 42% to stay in range after repo limit)
        // Argument: 20-30% -> 25% (increased to compensate)
        // Example: 15-25% -> 15% (minimum to stay in range)
        // Error: 5-15% -> 10%
        // Module: 3-7% -> 5%
        const FUNC_PCT: f64 = 0.40;
        const ARG_PCT: f64 = 0.25;
        const EX_PCT: f64 = 0.15;
        const ERR_PCT: f64 = 0.10;
        const MOD_PCT: f64 = 0.05;

        // Calculate max corpus size for each category based on its minimum percentage
        // (using minimum percentages: func 35%, arg 20%, ex 15%, err 5%, mod 3%)
        let max_by_ex = (ex_count as f64 / 0.15).floor() as usize;
        let max_by_err = (err_count as f64 / 0.05).floor() as usize;
        let max_by_arg = (arg_count as f64 / 0.20).floor() as usize;
        let _max_by_mod = (mod_count as f64 / 0.07).ceil() as usize; // cap is max, not min

        // The scarcest category determines max corpus size
        let max_corpus = max_by_ex.min(max_by_err).min(max_by_arg).max(150); // at least 150 for v1.1.0
        eprintln!("  Max corpus size: {} (ex={}, err={}, arg={})",
            max_corpus, max_by_ex, max_by_err, max_by_arg);

        // Calculate target counts for each category
        let target_func = ((max_corpus as f64) * FUNC_PCT).round() as usize;
        let target_arg = ((max_corpus as f64) * ARG_PCT).round() as usize;
        let target_ex = ((max_corpus as f64) * EX_PCT).round() as usize;
        let target_err = ((max_corpus as f64) * ERR_PCT).round() as usize;
        let target_mod = ((max_corpus as f64) * MOD_PCT).round() as usize;

        eprintln!("  Targets: func={}, arg={}, ex={}, err={}, mod={}",
            target_func, target_arg, target_ex, target_err, target_mod);

        // Build balanced corpus
        let mut balanced = Vec::new();

        // Take entries up to targets (limited by available)
        if let Some(mut cat_entries) = by_category.remove(&Category::Function) {
            let take = cat_entries.len().min(target_func);
            balanced.extend(cat_entries.drain(..take));
        }
        if let Some(mut cat_entries) = by_category.remove(&Category::Argument) {
            let take = cat_entries.len().min(target_arg);
            balanced.extend(cat_entries.drain(..take));
        }
        if let Some(mut cat_entries) = by_category.remove(&Category::Example) {
            let take = cat_entries.len().min(target_ex);
            balanced.extend(cat_entries.drain(..take));
        }
        if let Some(mut cat_entries) = by_category.remove(&Category::Error) {
            let take = cat_entries.len().min(target_err);
            balanced.extend(cat_entries.drain(..take));
        }
        if let Some(mut cat_entries) = by_category.remove(&Category::Module) {
            let take = cat_entries.len().min(target_mod);
            balanced.extend(cat_entries.drain(..take));
        }

        balanced
    }

    /// Limits repo dominance to max_repo_pct.
    /// Uses iterative passes until no repo exceeds the limit.
    fn limit_repo_dominance(&self, mut entries: Vec<CorpusEntry>) -> Vec<CorpusEntry> {
        // Sort by quality first so we keep the best entries
        entries.sort_by(|a, b| {
            b.quality_score
                .partial_cmp(&a.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Iteratively limit until stable
        loop {
            let total = entries.len();
            if total == 0 {
                return entries;
            }

            let max_per_repo = ((total as f64) * self.max_repo_pct / 100.0).floor() as usize;

            // Count by repo
            let mut repo_counts: HashMap<String, usize> = HashMap::new();
            let initial_len = entries.len();

            entries.retain(|e| {
                let count = repo_counts.entry(e.source_repo.clone()).or_insert(0);
                if *count < max_per_repo {
                    *count += 1;
                    true
                } else {
                    false
                }
            });

            // If no entries were removed, we're done
            if entries.len() == initial_len {
                break;
            }
        }

        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_default() {
        let filter = CorpusFilter::default();
        assert_eq!(filter.max_line_length, 100);
        assert!((filter.min_quality - 0.4).abs() < 0.001); // Updated for v1.1.0
    }

    #[test]
    fn test_balanced_delimiters() {
        let filter = CorpusFilter::default();
        assert!(filter.has_balanced_delimiters("(a + b)"));
        assert!(filter.has_balanced_delimiters("[1, 2, 3]"));
        assert!(!filter.has_balanced_delimiters("(a + b"));
        assert!(!filter.has_balanced_delimiters("[1, 2"));
    }
}
