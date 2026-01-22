# Fine-Tuning Pipeline Demo Script

**Duration:** ~5 minutes
**Audience:** ML practitioners, developers learning fine-tuning
**Visual:** `docs/images/week2/fine-tuning-pipeline.svg`

---

## INTRO (30 seconds)

> "Welcome to the fine-tuning pipeline overview. Today I'll show you how to take a pre-trained language model and adapt it to your specific task—using just 0.1% of the parameters.
>
> This diagram shows the complete end-to-end pipeline we use in the Sovereign AI Stack. Let's walk through each step."

---

## PANEL 1: CORPUS (45 seconds)

**[Point to top-left panel with blue border]**

> "Everything starts with data. Our corpus contains 100 carefully curated examples from 7 popular Rust CLI repositories—ripgrep, clap, bat, and others.
>
> Each example is a pair: the INPUT is a function signature, and the OUTPUT is the documentation we want the model to generate.
>
> For example, given `fn parse_args` returning a `Result`, we want the model to produce proper rustdoc comments with Arguments sections, Returns descriptions, and Examples.
>
> Notice the quality score of 0.90—we use Popperian falsification to ensure only high-quality examples make it into training."

---

## PANEL 2: LoRA CONFIG (45 seconds)

**[Point to second panel with green border]**

> "Here's where the magic happens. Instead of training all 7 billion parameters, we use LoRA—Low-Rank Adaptation.
>
> The config is simple YAML: rank 8, alpha 16, and we target the attention projection layers—q, k, v, and o proj.
>
> The key insight? We're only training 0.1% of the parameters. That's roughly 65,000 trainable parameters instead of 7 billion. This means you can fine-tune on a laptop GPU instead of needing a data center."

---

## PANEL 3: TRAINING (45 seconds)

**[Point to third panel with purple border]**

> "Training is handled by entrenar, our Rust-native training library.
>
> You simply run `entrenar train train.yaml` and it handles everything—loading the config, creating batches, running the training loop.
>
> We train for 3 epochs with a learning rate of 0.0002 and batch size of 4. The loss decreases as the model learns to generate documentation in our style.
>
> When training completes, the model checkpoint is saved automatically."

---

## PANEL 4: EVALUATION (45 seconds)

**[Point to fourth panel with orange border]**

> "How do we know it worked? We evaluate on three metrics.
>
> Structural checks whether the output has the right format—description, usage, arguments sections.
>
> Content accuracy measures whether the generated flags and parameters actually exist—no hallucinations.
>
> BLEU score measures text similarity to reference documentation.
>
> We need all three metrics above 70% to pass. Our model hits 100% structural, 80% content, and 100% BLEU—that's a pass."

---

## PANEL 5: LoRA MATH (45 seconds)

**[Point to middle-left section]**

> "Let me explain why LoRA works mathematically.
>
> The original weight matrix W is huge—d by d, that's 16.8 million parameters for a 4096-dimension model.
>
> Instead of updating W directly, we learn two small matrices: A is d by r, and B is r by d. With rank 8, that's just 65,000 parameters.
>
> The update delta-W equals A times B. At inference, we merge: W-prime equals W plus A-B. No extra latency, 256x fewer parameters to train.
>
> The hypothesis is that weight updates have low intrinsic rank—and empirically, it works remarkably well."

---

## PANEL 6: PRODUCTION WORKFLOW (30 seconds)

**[Point to middle-right section]**

> "Here's the production workflow in three commands.
>
> First, import from HuggingFace: `apr import` pulls Qwen 2.5 Coder and converts to our format.
>
> Second, fine-tune with LoRA: `entrenar train` runs the training we just discussed.
>
> Third, merge and run: `apr merge` combines base and adapter, then `apr run` generates documentation for any function signature you give it.
>
> Pure Rust, no Python dependencies, runs anywhere."

---

## PANEL 7: DISTRIBUTIONS (30 seconds)

**[Point to bottom-left and bottom-center]**

> "The bottom panels show our data distribution.
>
> Repository distribution: ripgrep contributes 38%, clap 26%—we draw from real production CLI tools.
>
> Category distribution: 46% function docs, 29% argument docs, 14% examples. This matches how documentation is actually structured in Rust projects."

---

## PANEL 8: FALSIFICATION TESTS (30 seconds)

**[Point to bottom-right section]**

> "Finally, we validate everything with Popperian falsification—96 out of 100 points.
>
> Data integrity, syntactic validity, semantic validity—all perfect 20 out of 20.
>
> Distribution got 11 out of 15—slightly over on function docs, slightly under on examples.
>
> 73 total tests pass: 54 validation tests, 18 property-based tests, and 1 doc test.
>
> This rigorous testing ensures the corpus survives falsification attempts."

---

## CONCLUSION (30 seconds)

> "That's the complete fine-tuning pipeline.
>
> Key takeaways: LoRA lets you fine-tune with 0.1% of parameters. Quality data matters more than quantity—100 curated examples beat 10,000 noisy ones. And rigorous evaluation ensures your model actually learned what you intended.
>
> The Sovereign AI Stack—entrenar, apr-cli, and friends—makes this pipeline pure Rust, reproducible, and runnable on commodity hardware.
>
> Try it yourself: clone the repo, run `make demo-lora-math`, and see LoRA in action. Thanks for watching."

---

## TIMING SUMMARY

| Section | Duration |
|---------|----------|
| Intro | 0:30 |
| Panel 1: Corpus | 0:45 |
| Panel 2: LoRA Config | 0:45 |
| Panel 3: Training | 0:45 |
| Panel 4: Evaluation | 0:45 |
| Panel 5: LoRA Math | 0:45 |
| Panel 6: Production | 0:30 |
| Panel 7: Distributions | 0:30 |
| Panel 8: Falsification | 0:30 |
| Conclusion | 0:30 |
| **Total** | **~5:30** |

---

## PRESENTER NOTES

- Keep cursor/pointer visible when referencing panels
- Pause briefly when transitioning between panels
- For live demos, have terminal ready with `make demo-lora-math`
- Emphasize "0.1% of parameters" as the key insight
- The 256x reduction is memorable—repeat it
