//! Feed-Forward Network Demo: The Lemonade Stand
//!
//! Making lemonade as a metaphor for FFN:
//! 1. ATTENTION = Gather ingredients from the pantry (who to listen to)
//! 2. FFN EXPAND = Break down into flavor components (what we're working with)
//! 3. GELU = Taste test! Too sour? Squash it. Good? Let it through.
//! 4. FFN CONTRACT = Final blend decision (the output)
//!
//! Key insight: Attention gathers. FFN tastes and adjusts.

use std::fmt;

/// Ingredient dimensions: L=Lemon, W=Water, S=Sugar, T=Tartness
pub const INGREDIENT_DIM: usize = 4;
pub const INGREDIENT_LABELS: [&str; INGREDIENT_DIM] = ["L", "W", "S", "T"];

/// Flavor component dimension (4× expansion = 16 taste receptors)
pub const FLAVOR_DIM: usize = 16;

/// Three batches of lemonade we're making
pub const BATCHES: &[&str] = &["Batch1", "Batch2", "Batch3"];

/// Number of batches
pub const NUM_BATCHES: usize = 3;

/// After ATTENTION: gathered ingredients from pantry
/// Each batch pulled different amounts from available supplies
/// Values: [Lemon, Water, Sugar, Tartness]
pub const GATHERED_INGREDIENTS: [[f32; INGREDIENT_DIM]; NUM_BATCHES] = [
    [0.46, 0.41, 0.75, 0.40], // Batch1: sweet-forward
    [0.40, 0.48, 0.78, 0.40], // Batch2: more water, sweeter
    [0.39, 0.44, 0.79, 0.44], // Batch3: sweetest, bit tart
];

/// How much each batch "listened to" each supply source
pub const GATHERING_WEIGHTS: [[f32; NUM_BATCHES]; NUM_BATCHES] = [
    [0.37, 0.31, 0.32], // Batch1 gathered from
    [0.29, 0.38, 0.33], // Batch2 gathered from
    [0.29, 0.32, 0.39], // Batch3 gathered from
];

/// W1: "Flavor Analyzer" - breaks ingredients into 16 taste components
/// Like: lemon → (sour, citrus, bright, acidic, ...)
/// Shape: [INGREDIENT_DIM][FLAVOR_DIM]
pub fn make_w1() -> Vec<Vec<f32>> {
    (0..INGREDIENT_DIM)
        .map(|i| {
            (0..FLAVOR_DIM)
                .map(|j| ((i + j) % 3) as f32 * 0.1 + 0.05)
                .collect()
        })
        .collect()
}

/// W2: "Blend Decider" - combines taste components back to final recipe
/// 16 taste signals → 4 output adjustments (L/W/S/T)
/// Shape: [FLAVOR_DIM][INGREDIENT_DIM]
pub fn make_w2() -> Vec<Vec<f32>> {
    (0..FLAVOR_DIM)
        .map(|i| {
            (0..INGREDIENT_DIM)
                .map(|j| ((i + j) % 4) as f32 * 0.05 + 0.02)
                .collect()
        })
        .collect()
}

/// GELU: The Taste Test!
/// - Positive values (good flavors) → pass through mostly unchanged
/// - Negative values (bad flavors) → squashed toward zero
/// - "Too sour? Nope. Good balance? Yes!"
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.797_884_6;
    let coeff = 0.044715;
    x * 0.5 * (1.0 + (sqrt_2_over_pi * (x + coeff * x.powi(3))).tanh())
}

/// ReLU: Binary taste test (anything negative = reject completely)
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Linear: No taste test (skip quality control) - for --error mode
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

/// Recipe step: one batch going through the kitchen
#[derive(Debug, Clone)]
pub struct RecipeStep {
    /// Gathered ingredients [L, W, S, T]
    pub ingredients: Vec<f32>,
    /// After flavor analysis (16 taste components)
    pub flavor_profile: Vec<f32>,
    /// After taste test (GELU filtered)
    pub taste_tested: Vec<f32>,
    /// Final blend decision [L, W, S, T]
    pub final_blend: Vec<f32>,
    /// Taste test method used
    pub taste_test_name: String,
}

/// Full lemonade stand results
#[derive(Debug, Clone)]
pub struct DemoResults {
    /// Batch names
    pub batches: Vec<String>,
    /// How each batch gathered ingredients
    pub gathering_weights: Vec<Vec<f32>>,
    /// Recipe results per batch
    pub recipe_steps: Vec<RecipeStep>,
    /// Flavor analyzer shape
    pub analyzer_shape: (usize, usize),
    /// Blend decider shape
    pub decider_shape: (usize, usize),
    /// Whether taste test was skipped
    pub skip_taste_test: bool,
}

/// Process one batch through the kitchen
pub fn make_batch<F>(ingredients: &[f32], analyzer: &[Vec<f32>], decider: &[Vec<f32>], taste_test: F, taste_test_name: &str) -> RecipeStep
where
    F: Fn(f32) -> f32,
{
    // Step 1: Analyze flavors (expand to 16 taste components)
    let flavor_profile = matvec(analyzer, ingredients);

    // Step 2: Taste test! (GELU gates bad flavors)
    let taste_tested: Vec<f32> = flavor_profile.iter().map(|&x| taste_test(x)).collect();

    // Step 3: Decide final blend (contract back to L/W/S/T)
    let final_blend = matvec(decider, &taste_tested);

    RecipeStep {
        ingredients: ingredients.to_vec(),
        flavor_profile,
        taste_tested,
        final_blend,
        taste_test_name: taste_test_name.to_string(),
    }
}

/// Run the lemonade stand demo
pub fn run(skip_taste_test: bool) -> DemoResults {
    let analyzer = make_w1(); // Flavor Analyzer
    let decider = make_w2();  // Blend Decider

    let batches: Vec<String> = BATCHES.iter().map(|s| (*s).to_string()).collect();
    let gathering_weights: Vec<Vec<f32>> = GATHERING_WEIGHTS.iter().map(|row| row.to_vec()).collect();

    let recipe_steps: Vec<RecipeStep> = GATHERED_INGREDIENTS
        .iter()
        .map(|ingredients| {
            if skip_taste_test {
                make_batch(ingredients, &analyzer, &decider, linear, "NONE (no taste test)")
            } else {
                make_batch(ingredients, &analyzer, &decider, gelu, "GELU (taste test)")
            }
        })
        .collect();

    DemoResults {
        batches,
        gathering_weights,
        recipe_steps,
        analyzer_shape: (INGREDIENT_DIM, FLAVOR_DIM),
        decider_shape: (FLAVOR_DIM, INGREDIENT_DIM),
        skip_taste_test,
    }
}

/// Format ingredients with labels
fn fmt_ingredients(v: &[f32]) -> String {
    if v.len() == INGREDIENT_DIM {
        format!(
            "[L:{:.2} W:{:.2} S:{:.2} T:{:.2}]",
            v[0], v[1], v[2], v[3]
        )
    } else {
        fmt_vec(v, 6)
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
        writeln!(f, "║        THE LEMONADE STAND: FFN Demo                          ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  \"Attention gathers ingredients. FFN tastes and adjusts.\"    ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Legend
        writeln!(f, "  Ingredients: L=Lemon, W=Water, S=Sugar, T=Tartness")?;
        writeln!(f)?;

        // Step 1: What attention gathered
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ STEP 1: ATTENTION GATHERED (from the pantry)               │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        for (i, step) in self.recipe_steps.iter().enumerate() {
            let weights = &self.gathering_weights[i];
            writeln!(
                f,
                "  {} = {:.0}%A + {:.0}%B + {:.0}%C → {}",
                self.batches[i],
                weights[0] * 100.0,
                weights[1] * 100.0,
                weights[2] * 100.0,
                fmt_ingredients(&step.ingredients)
            )?;
        }
        writeln!(f)?;
        writeln!(f, "  Status: Ingredients gathered, but not yet tasted!")?;
        writeln!(f)?;

        // Step 2: Kitchen equipment
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ STEP 2: THE KITCHEN (FFN Architecture)                      │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        writeln!(
            f,
            "  Flavor Analyzer: [{} × {}]  (4 ingredients → 16 taste signals)",
            self.analyzer_shape.0, self.analyzer_shape.1
        )?;
        writeln!(f, "  Taste Test: {}", self.recipe_steps[0].taste_test_name)?;
        writeln!(
            f,
            "  Blend Decider:   [{} × {}]  (16 signals → 4 adjustments)",
            self.decider_shape.0, self.decider_shape.1
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  Recipe: output = Decider( {}( Analyzer(ingredients) ) )",
            if self.skip_taste_test { "skip" } else { "GELU" }
        )?;
        writeln!(f)?;

        // Step 3: Processing each batch
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        if self.skip_taste_test {
            writeln!(f, "│ STEP 3: PROCESSING (⚠️  NO TASTE TEST!)                     │")?;
        } else {
            writeln!(f, "│ STEP 3: PROCESSING (taste test each batch)                 │")?;
        }
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;

        for (i, step) in self.recipe_steps.iter().enumerate() {
            writeln!(f, "\n  ── {} ──", self.batches[i])?;
            writeln!(f, "  Gathered:        {}", fmt_ingredients(&step.ingredients))?;
            writeln!(
                f,
                "  Flavor Profile:  {} (16 taste signals)",
                fmt_vec(&step.flavor_profile, 5)
            )?;
            if self.skip_taste_test {
                writeln!(f, "  (skipped taste)  {}", fmt_vec(&step.taste_tested, 5))?;
            } else {
                writeln!(f, "  After Taste:     {} (bad filtered)", fmt_vec(&step.taste_tested, 5))?;
            }
            writeln!(f, "  Final Blend:     {}", fmt_ingredients(&step.final_blend))?;
        }
        writeln!(f)?;

        // Before vs after
        writeln!(f, "┌─────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ GATHERED vs FINAL BLEND                                     │")?;
        writeln!(f, "└─────────────────────────────────────────────────────────────┘")?;
        for (i, step) in self.recipe_steps.iter().enumerate() {
            writeln!(
                f,
                "  {}: {} → {}",
                self.batches[i],
                fmt_ingredients(&step.ingredients),
                fmt_ingredients(&step.final_blend)
            )?;
        }
        writeln!(f)?;

        // Key insight
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        if self.skip_taste_test {
            writeln!(f, "║ ⚠️  WITHOUT TASTE TEST (no GELU):                            ║")?;
            writeln!(f, "║ • Just blindly mixed ingredients → no quality control       ║")?;
            writeln!(f, "║ • Analyzer × Decider = one big matrix (no decisions)        ║")?;
            writeln!(f, "║ • Can't learn \"too sour\" or \"too sweet\" → bad lemonade      ║")?;
        } else {
            writeln!(f, "║ THE LEMONADE INSIGHT:                                        ║")?;
            writeln!(f, "║ • Attention = gathered ingredients (L, W, S, T)              ║")?;
            writeln!(f, "║ • FFN = tasted and adjusted (GELU said \"no\" to bad flavors) ║")?;
            writeln!(f, "║ • Non-linearity = \"too sour? squash it!\"                     ║")?;
            writeln!(f, "║ • 2/3 of transformer params = the kitchen (Analyzer+Decider)║")?;
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

    // ========== Lemonade constants tests ==========

    #[test]
    fn test_dimensions() {
        assert_eq!(FLAVOR_DIM, INGREDIENT_DIM * 4);
        assert_eq!(BATCHES.len(), NUM_BATCHES);
        assert_eq!(GATHERED_INGREDIENTS.len(), NUM_BATCHES);
        assert_eq!(GATHERING_WEIGHTS.len(), NUM_BATCHES);
    }

    #[test]
    fn test_gathering_weights_sum_to_one() {
        for row in &GATHERING_WEIGHTS {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Row sum {} should be ~1.0", sum);
        }
    }

    #[test]
    fn test_gathered_ingredients_dimensions() {
        for row in &GATHERED_INGREDIENTS {
            assert_eq!(row.len(), INGREDIENT_DIM);
        }
    }

    #[test]
    fn test_ingredient_labels() {
        assert_eq!(INGREDIENT_LABELS, ["L", "W", "S", "T"]);
    }

    // ========== Kitchen equipment tests ==========

    #[test]
    fn test_analyzer_shape() {
        let analyzer = make_w1();
        assert_eq!(analyzer.len(), INGREDIENT_DIM);
        assert_eq!(analyzer[0].len(), FLAVOR_DIM);
    }

    #[test]
    fn test_decider_shape() {
        let decider = make_w2();
        assert_eq!(decider.len(), FLAVOR_DIM);
        assert_eq!(decider[0].len(), INGREDIENT_DIM);
    }

    #[test]
    fn test_analyzer_values_bounded() {
        let analyzer = make_w1();
        for row in &analyzer {
            for &val in row {
                assert!(val >= 0.0 && val <= 1.0, "Analyzer value {} out of bounds", val);
            }
        }
    }

    #[test]
    fn test_decider_values_bounded() {
        let decider = make_w2();
        for row in &decider {
            for &val in row {
                assert!(val >= 0.0 && val <= 1.0, "Decider value {} out of bounds", val);
            }
        }
    }

    // ========== Taste test (activation) tests ==========

    #[test]
    fn test_gelu_zero() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_good_flavor_passes() {
        // Positive = good flavor → passes through
        let x = 1.0;
        let y = gelu(x);
        assert!(y > 0.8 && y < 0.9, "Good flavor {} should pass ~0.841", y);
    }

    #[test]
    fn test_gelu_bad_flavor_squashed() {
        // Negative = bad flavor → squashed
        let x = -1.0;
        let y = gelu(x);
        assert!(y > -0.2 && y < -0.1, "Bad flavor {} should be squashed to ~-0.159", y);
    }

    #[test]
    fn test_gelu_very_good_passes_fully() {
        let x = 5.0;
        let y = gelu(x);
        assert!((y - x).abs() < 0.01, "Very good flavor passes fully");
    }

    #[test]
    fn test_gelu_very_bad_squashed_to_zero() {
        let x = -5.0;
        let y = gelu(x);
        assert!(y.abs() < 0.01, "Very bad flavor squashed to ~0");
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
    fn test_linear_no_taste_test() {
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
        assert_eq!(result, vec![4.0, 6.0]);
    }

    #[test]
    fn test_matvec_expansion() {
        let m = vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]];
        let v = vec![2.0, 3.0];
        let result = matvec(&m, &v);
        assert_eq!(result.len(), 4);
    }

    // ========== Recipe step (make_batch) tests ==========

    #[test]
    fn test_make_batch_final_blend_dimension() {
        let analyzer = make_w1();
        let decider = make_w2();
        let ingredients = GATHERED_INGREDIENTS[0].to_vec();
        let step = make_batch(&ingredients, &analyzer, &decider, gelu, "GELU");
        assert_eq!(step.final_blend.len(), INGREDIENT_DIM);
    }

    #[test]
    fn test_make_batch_flavor_profile_dimension() {
        let analyzer = make_w1();
        let decider = make_w2();
        let ingredients = GATHERED_INGREDIENTS[0].to_vec();
        let step = make_batch(&ingredients, &analyzer, &decider, gelu, "GELU");
        assert_eq!(step.flavor_profile.len(), FLAVOR_DIM);
        assert_eq!(step.taste_tested.len(), FLAVOR_DIM);
    }

    #[test]
    fn test_make_batch_ingredients_preserved() {
        let analyzer = make_w1();
        let decider = make_w2();
        let ingredients = GATHERED_INGREDIENTS[0].to_vec();
        let step = make_batch(&ingredients, &analyzer, &decider, gelu, "GELU");
        assert_eq!(step.ingredients, ingredients);
    }

    #[test]
    fn test_make_batch_taste_test_name() {
        let analyzer = make_w1();
        let decider = make_w2();
        let ingredients = GATHERED_INGREDIENTS[0].to_vec();
        let step = make_batch(&ingredients, &analyzer, &decider, gelu, "GELU taste test");
        assert_eq!(step.taste_test_name, "GELU taste test");
    }

    #[test]
    fn test_taste_test_vs_no_taste_test_different() {
        let analyzer = make_w1();
        let decider = make_w2();
        let ingredients = GATHERED_INGREDIENTS[0].to_vec();
        let with_taste = make_batch(&ingredients, &analyzer, &decider, gelu, "GELU");
        let no_taste = make_batch(&ingredients, &analyzer, &decider, linear, "NONE");
        let diff: f32 = with_taste
            .final_blend
            .iter()
            .zip(no_taste.final_blend.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Taste test should change the blend");
    }

    // ========== Run function tests ==========

    #[test]
    fn test_run_with_taste_test() {
        let result = run(false);
        assert!(!result.skip_taste_test);
        assert_eq!(result.recipe_steps.len(), NUM_BATCHES);
    }

    #[test]
    fn test_run_without_taste_test() {
        let result = run(true);
        assert!(result.skip_taste_test);
    }

    #[test]
    fn test_run_batches_match() {
        let result = run(false);
        assert_eq!(result.batches, vec!["Batch1", "Batch2", "Batch3"]);
    }

    #[test]
    fn test_run_shapes_correct() {
        let result = run(false);
        assert_eq!(result.analyzer_shape, (INGREDIENT_DIM, FLAVOR_DIM));
        assert_eq!(result.decider_shape, (FLAVOR_DIM, INGREDIENT_DIM));
    }

    #[test]
    fn test_run_taste_test_name_in_steps() {
        let result = run(false);
        for step in &result.recipe_steps {
            assert!(step.taste_test_name.contains("GELU"));
        }
        let result_skip = run(true);
        for step in &result_skip.recipe_steps {
            assert!(step.taste_test_name.contains("NONE"));
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
    fn test_fmt_ingredients() {
        let v = vec![0.5, 0.3, 0.8, 0.2];
        let s = fmt_ingredients(&v);
        assert!(s.contains("L:"));
        assert!(s.contains("W:"));
        assert!(s.contains("S:"));
        assert!(s.contains("T:"));
    }

    #[test]
    fn test_display_lemonade_mode() {
        let result = run(false);
        let display = format!("{}", result);
        assert!(display.contains("LEMONADE"));
        assert!(display.contains("GELU"));
        assert!(display.contains("taste"));
    }

    #[test]
    fn test_display_no_taste_test_mode() {
        let result = run(true);
        let display = format!("{}", result);
        assert!(display.contains("NO TASTE TEST"));
        assert!(display.contains("blindly"));
    }

    #[test]
    fn test_display_contains_batches() {
        let result = run(false);
        let display = format!("{}", result);
        for batch in BATCHES {
            assert!(display.contains(batch));
        }
    }

    #[test]
    fn test_display_shows_shapes() {
        let result = run(false);
        let display = format!("{}", result);
        assert!(display.contains(&format!("[{} × {}]", INGREDIENT_DIM, FLAVOR_DIM)));
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
    fn test_ffn_changes_blend() {
        let result = run(false);
        for step in &result.recipe_steps {
            let diff: f32 = step
                .ingredients
                .iter()
                .zip(step.final_blend.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(diff > 0.01, "FFN should change the blend");
        }
    }

    #[test]
    fn test_flavor_expansion() {
        let result = run(false);
        for step in &result.recipe_steps {
            assert!(
                step.flavor_profile.len() > step.ingredients.len(),
                "Analyzer should expand to 16 flavors"
            );
        }
    }

    #[test]
    fn test_blend_contraction() {
        let result = run(false);
        for step in &result.recipe_steps {
            assert_eq!(
                step.final_blend.len(),
                step.ingredients.len(),
                "Decider should contract back to 4 ingredients"
            );
        }
    }
}
