//! CLI Help Evaluation Framework
//!
//! How do we know the fine-tuned model "worked"?
//! - Structural validation (format correctness)
//! - Content accuracy (flag/arg correctness)
//! - Held-out test set (generalization)

use std::fmt;

/// Evaluation metrics for CLI help generation
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    pub structural_score: f64,   // Format correctness (0-1)
    pub content_accuracy: f64,   // Flag/arg accuracy (0-1)
    pub exact_match: f64,        // Exact match rate on test set
    pub bleu_score: f64,         // BLEU-4 score
}

/// Structural checks for help text
#[derive(Debug, Clone)]
pub struct StructuralChecks {
    pub has_description: bool,
    pub has_usage: bool,
    pub has_options: bool,
    pub has_help_flag: bool,
    pub valid_flag_format: bool,
    pub score: f64,
}

/// Check structural validity of generated help text
pub fn check_structure(help_text: &str) -> StructuralChecks {
    let has_description = !help_text.is_empty() &&
        help_text.lines().next().map(|l| !l.starts_with("Usage")).unwrap_or(false);
    let has_usage = help_text.contains("Usage:");
    let has_options = help_text.contains("Options:") || help_text.contains("Arguments:");
    let has_help_flag = help_text.contains("-h, --help") || help_text.contains("--help");

    // Check for valid flag format: -x, --long or --long
    let valid_flag_format = help_text.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("-") &&
        (trimmed.contains(", --") || trimmed.starts_with("--"))
    });

    let checks = [has_description, has_usage, has_options, has_help_flag, valid_flag_format];
    let score = checks.iter().filter(|&&x| x).count() as f64 / checks.len() as f64;

    StructuralChecks {
        has_description,
        has_usage,
        has_options,
        has_help_flag,
        valid_flag_format,
        score,
    }
}

/// Content validation: check if generated flags exist in reference
pub fn check_content_accuracy(generated: &str, reference: &str) -> f64 {
    let gen_flags = extract_flags(generated);
    let ref_flags = extract_flags(reference);

    if ref_flags.is_empty() {
        return if gen_flags.is_empty() { 1.0 } else { 0.0 };
    }

    let correct = gen_flags.iter().filter(|f| ref_flags.contains(f)).count();
    let hallucinated = gen_flags.iter().filter(|f| !ref_flags.contains(f)).count();

    // Penalize hallucinated flags
    let precision = if gen_flags.is_empty() { 0.0 } else {
        correct as f64 / gen_flags.len() as f64
    };
    let recall = correct as f64 / ref_flags.len() as f64;

    // F1 score with hallucination penalty
    if precision + recall == 0.0 {
        0.0
    } else {
        let f1 = 2.0 * precision * recall / (precision + recall);
        let penalty = 1.0 - (hallucinated as f64 * 0.1).min(0.5);
        f1 * penalty
    }
}

fn extract_flags(text: &str) -> Vec<String> {
    let mut flags = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('-') {
            // Extract --long-flag pattern
            for word in trimmed.split_whitespace() {
                if word.starts_with("--") {
                    let flag = word.trim_end_matches(',')
                        .trim_end_matches('>')
                        .trim_end_matches(']');
                    if flag.len() > 2 {
                        flags.push(flag.to_string());
                    }
                }
            }
        }
    }
    flags
}

/// Simple BLEU-like score (unigram + bigram)
pub fn bleu_score(generated: &str, reference: &str) -> f64 {
    let gen_tokens: Vec<&str> = generated.split_whitespace().collect();
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();

    if gen_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    // Unigram precision
    let unigram_matches = gen_tokens.iter()
        .filter(|t| ref_tokens.contains(t))
        .count();
    let unigram_precision = unigram_matches as f64 / gen_tokens.len() as f64;

    // Bigram precision
    let gen_bigrams: Vec<_> = gen_tokens.windows(2).collect();
    let ref_bigrams: Vec<_> = ref_tokens.windows(2).collect();

    let bigram_matches = gen_bigrams.iter()
        .filter(|b| ref_bigrams.contains(b))
        .count();
    let bigram_precision = if gen_bigrams.is_empty() { 0.0 } else {
        bigram_matches as f64 / gen_bigrams.len() as f64
    };

    // Brevity penalty
    let brevity = if gen_tokens.len() >= ref_tokens.len() {
        1.0
    } else {
        (gen_tokens.len() as f64 / ref_tokens.len() as f64).exp()
    };

    // Geometric mean of precisions
    let bleu = if unigram_precision > 0.0 && bigram_precision > 0.0 {
        brevity * (unigram_precision * bigram_precision).sqrt()
    } else {
        brevity * unigram_precision * 0.5
    };

    bleu.min(1.0)
}

/// Example evaluation on a test case
#[derive(Debug, Clone)]
pub struct TestCase {
    pub command: &'static str,
    pub reference: &'static str,
    pub generated_good: &'static str,
    pub generated_bad: &'static str,
}

pub fn example_test_case() -> TestCase {
    TestCase {
        command: "rg <PATTERN> [PATH]",
        reference: r#"Search for PATTERN in files recursively.

Usage: rg [OPTIONS] <PATTERN> [PATH]...

Arguments:
  <PATTERN>  The pattern to search for
  [PATH]...  Files or directories to search

Options:
  -i, --ignore-case     Search case insensitively
  -v, --invert-match    Invert matching
  -c, --count           Show count of matches
  -n, --line-number     Show line numbers
  -H, --with-filename   Show file names
  -h, --help            Print help"#,
        generated_good: r#"Search for patterns in files recursively.

Usage: rg [OPTIONS] <PATTERN> [PATH]...

Arguments:
  <PATTERN>  Pattern to search for
  [PATH]...  Paths to search

Options:
  -i, --ignore-case     Case insensitive search
  -c, --count           Only show match counts
  -n, --line-number     Show line numbers
  -h, --help            Print help"#,
        generated_bad: r#"rg is a tool

Options:
  --fast          Make it faster
  --turbo         Maximum speed
  --help          Show help"#,
    }
}

/// Demo results
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub test_case: TestCase,
    pub good_structural: StructuralChecks,
    pub bad_structural: StructuralChecks,
    pub good_content: f64,
    pub bad_content: f64,
    pub good_bleu: f64,
    pub bad_bleu: f64,
}

pub fn run() -> DemoResults {
    let test_case = example_test_case();

    let good_structural = check_structure(test_case.generated_good);
    let bad_structural = check_structure(test_case.generated_bad);

    let good_content = check_content_accuracy(test_case.generated_good, test_case.reference);
    let bad_content = check_content_accuracy(test_case.generated_bad, test_case.reference);

    let good_bleu = bleu_score(test_case.generated_good, test_case.reference);
    let bad_bleu = bleu_score(test_case.generated_bad, test_case.reference);

    DemoResults {
        test_case,
        good_structural,
        bad_structural,
        good_content,
        bad_content,
        good_bleu,
        bad_bleu,
    }
}

fn score_bar(score: f64, width: usize) -> String {
    let filled = (score * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}",
        "█".repeat(filled),
        "░".repeat(empty))
}

fn check_mark(b: bool) -> &'static str {
    if b { "✓" } else { "✗" }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║         EVALUATION: Did Fine-Tuning Work?                        ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  \"How do we know it's not just random?\"                          ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // The framework
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ EVALUATION FRAMEWORK                                           │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  1. STRUCTURAL VALIDATION")?;
        writeln!(f, "     Does output have correct format?")?;
        writeln!(f, "     • Has description line")?;
        writeln!(f, "     • Has Usage: section")?;
        writeln!(f, "     • Has Options:/Arguments: sections")?;
        writeln!(f, "     • Has -h, --help flag")?;
        writeln!(f, "     • Valid flag format (-x, --long)")?;
        writeln!(f)?;
        writeln!(f, "  2. CONTENT ACCURACY")?;
        writeln!(f, "     Are the flags/arguments real?")?;
        writeln!(f, "     • Precision: generated flags that exist in reference")?;
        writeln!(f, "     • Recall: reference flags found in generated")?;
        writeln!(f, "     • Hallucination penalty: invented flags")?;
        writeln!(f)?;
        writeln!(f, "  3. TEXT SIMILARITY (BLEU)")?;
        writeln!(f, "     How similar to reference?")?;
        writeln!(f, "     • Unigram/bigram overlap")?;
        writeln!(f, "     • Brevity penalty")?;
        writeln!(f)?;

        // Test case
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TEST CASE: {}                                            │", self.test_case.command)?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f)?;

        // Comparison
        writeln!(f, "┌─────────────────────────────────┬─────────────────────────────────┐")?;
        writeln!(f, "│ GOOD OUTPUT (trained model)     │ BAD OUTPUT (random/untrained)   │")?;
        writeln!(f, "├─────────────────────────────────┴─────────────────────────────────┤")?;
        writeln!(f)?;

        writeln!(f, "  STRUCTURAL CHECKS:")?;
        writeln!(f, "    Good: {} description {} usage {} options {} help {} format",
            check_mark(self.good_structural.has_description),
            check_mark(self.good_structural.has_usage),
            check_mark(self.good_structural.has_options),
            check_mark(self.good_structural.has_help_flag),
            check_mark(self.good_structural.valid_flag_format))?;
        writeln!(f, "    Bad:  {} description {} usage {} options {} help {} format",
            check_mark(self.bad_structural.has_description),
            check_mark(self.bad_structural.has_usage),
            check_mark(self.bad_structural.has_options),
            check_mark(self.bad_structural.has_help_flag),
            check_mark(self.bad_structural.valid_flag_format))?;
        writeln!(f)?;

        // Scores comparison
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ SCORES                                                         │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "                              Good          Bad")?;
        writeln!(f, "  Structural:          {} {:>5.1}%   {} {:>5.1}%",
            score_bar(self.good_structural.score, 10),
            self.good_structural.score * 100.0,
            score_bar(self.bad_structural.score, 10),
            self.bad_structural.score * 100.0)?;
        writeln!(f, "  Content Accuracy:    {} {:>5.1}%   {} {:>5.1}%",
            score_bar(self.good_content, 10),
            self.good_content * 100.0,
            score_bar(self.bad_content, 10),
            self.bad_content * 100.0)?;
        writeln!(f, "  BLEU Score:          {} {:>5.1}%   {} {:>5.1}%",
            score_bar(self.good_bleu, 10),
            self.good_bleu * 100.0,
            score_bar(self.bad_bleu, 10),
            self.bad_bleu * 100.0)?;
        writeln!(f)?;

        // Train/test split
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TRAIN/TEST METHODOLOGY                                         │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  Data: ~150 CLI help examples")?;
        writeln!(f)?;
        writeln!(f, "  Split:")?;
        writeln!(f, "    Train: 120 examples (80%) — model learns from these")?;
        writeln!(f, "    Test:   30 examples (20%) — held out, never seen")?;
        writeln!(f)?;
        writeln!(f, "  Key insight: Test on UNSEEN commands")?;
        writeln!(f, "    • Train on: apr, pmat, cargo, git")?;
        writeln!(f, "    • Test on:  rg, fd, bat (held out)")?;
        writeln!(f)?;
        writeln!(f, "  If test scores ≈ train scores → model generalized")?;
        writeln!(f, "  If test scores << train scores → overfitting")?;
        writeln!(f)?;

        // Success criteria
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ SUCCESS CRITERIA                                                 ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Structural Score:    > 80%  (correct format)                    ║")?;
        writeln!(f, "║  Content Accuracy:    > 70%  (real flags, no hallucinations)     ║")?;
        writeln!(f, "║  BLEU Score:          > 40%  (similar to reference)              ║")?;
        writeln!(f, "║  Test/Train Gap:      < 10%  (generalization, not memorization)  ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Random baseline: ~10-20% on all metrics                         ║")?;
        writeln!(f, "║  Good fine-tune:  ~70-90% on all metrics                         ║")?;
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
    fn test_structure_check_good() {
        let test = example_test_case();
        let checks = check_structure(test.generated_good);
        assert!(checks.has_usage);
        assert!(checks.has_options);
        assert!(checks.score > 0.8);
    }

    #[test]
    fn test_structure_check_bad() {
        let test = example_test_case();
        let checks = check_structure(test.generated_bad);
        assert!(!checks.has_usage);  // Missing Usage: section
        // Bad still passes some structural checks (has Options:, --help)
        // The real differentiation is in content accuracy and BLEU
    }

    #[test]
    fn test_content_accuracy_good() {
        let test = example_test_case();
        let score = check_content_accuracy(test.generated_good, test.reference);
        assert!(score > 0.5);
    }

    #[test]
    fn test_content_accuracy_bad() {
        let test = example_test_case();
        let score = check_content_accuracy(test.generated_bad, test.reference);
        assert!(score < 0.3);
    }

    #[test]
    fn test_bleu_good_higher() {
        let test = example_test_case();
        let good = bleu_score(test.generated_good, test.reference);
        let bad = bleu_score(test.generated_bad, test.reference);
        assert!(good > bad);
    }

    #[test]
    fn test_extract_flags() {
        let text = "  -i, --ignore-case  Ignore case\n  -v, --verbose  Be verbose";
        let flags = extract_flags(text);
        assert!(flags.contains(&"--ignore-case".to_string()));
        assert!(flags.contains(&"--verbose".to_string()));
    }

    #[test]
    fn test_display_runs() {
        let result = run();
        let display = format!("{}", result);
        assert!(display.contains("EVALUATION"));
        assert!(display.contains("SUCCESS CRITERIA"));
    }
}
