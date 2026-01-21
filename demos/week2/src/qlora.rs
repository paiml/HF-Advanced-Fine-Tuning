//! QLoRA Demo
//!
//! Shows 4-bit quantization + LoRA:
//! - Shrink model from 14GB to 3.5GB
//! - Train only the diff at full precision
//!
//! Result: 7B model trainable on consumer GPU

use std::fmt;

/// Memory calculation for different quantization levels
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub name: &'static str,
    pub bits: u8,
    pub bytes_per_param: f64,
    pub analogy: &'static str,
}

pub const QUANT_LEVELS: &[QuantConfig] = &[
    QuantConfig { name: "FP32",  bits: 32, bytes_per_param: 4.0, analogy: "RAW photo" },
    QuantConfig { name: "FP16",  bits: 16, bytes_per_param: 2.0, analogy: "High-res JPEG" },
    QuantConfig { name: "INT8",  bits: 8,  bytes_per_param: 1.0, analogy: "Normal JPEG" },
    QuantConfig { name: "NF4",   bits: 4,  bytes_per_param: 0.5, analogy: "Thumbnail" },
];

/// Memory breakdown for QLoRA
#[derive(Debug, Clone)]
pub struct QLoraMemory {
    pub model_params_b: f64,
    pub quant_name: &'static str,
    pub base_model_gb: f64,
    pub lora_params: usize,
    pub lora_gb: f64,
    pub overhead_gb: f64,
    pub total_gb: f64,
}

/// Calculate QLoRA memory for a model
pub fn calculate_qlora(params_b: f64, quant: &QuantConfig, lora_rank: usize, num_layers: usize) -> QLoraMemory {
    let params = params_b * 1e9;

    // Base model in quantized format
    let base_model_gb = (params * quant.bytes_per_param) / 1e9;

    // LoRA params: assume adapting q,k,v,o projections per layer
    // Each projection: hidden_dim × hidden_dim, but LoRA: 2 × hidden_dim × rank
    // Rough estimate: 4 projections × num_layers × 2 × hidden_dim × rank
    let hidden_dim = 4096; // Approximate for 7B model
    let lora_params = 4 * num_layers * 2 * hidden_dim * lora_rank;
    let lora_gb = (lora_params as f64 * 2.0) / 1e9; // fp16

    // Overhead: gradients for LoRA only + optimizer states
    let overhead_gb = lora_gb * 6.0; // grads + adam states

    let total_gb = base_model_gb + lora_gb + overhead_gb;

    QLoraMemory {
        model_params_b: params_b,
        quant_name: quant.name,
        base_model_gb,
        lora_params,
        lora_gb,
        overhead_gb,
        total_gb,
    }
}

/// Comparison across model sizes
#[derive(Debug, Clone)]
pub struct DemoResults {
    pub comparisons: Vec<(f64, Vec<QLoraMemory>)>, // (params_b, memories for each quant)
}

pub fn run() -> DemoResults {
    let model_sizes = [1.5, 7.0, 13.0, 70.0];
    let lora_rank = 16;
    let num_layers = 32;

    let comparisons: Vec<(f64, Vec<QLoraMemory>)> = model_sizes
        .iter()
        .map(|&params_b| {
            let memories: Vec<QLoraMemory> = QUANT_LEVELS
                .iter()
                .map(|q| calculate_qlora(params_b, q, lora_rank, num_layers))
                .collect();
            (params_b, memories)
        })
        .collect();

    DemoResults { comparisons }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           QLoRA: SHRINK THE MODEL, TRAIN THE DIFF                ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Problem: 7B model = 14GB. Training needs 62GB.                  ║")?;
        writeln!(f, "║  Solution: Shrink to 4-bit (3.5GB), train diff at full precision.║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // Quantization as compression
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ QUANTIZATION = COMPRESSION                                     │")?;
        writeln!(f, "│ Fewer bits per number = smaller file = less precise            │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for q in QUANT_LEVELS {
            let bar_len = (q.bytes_per_param * 10.0) as usize;
            let bar: String = "█".repeat(bar_len);
            writeln!(f, "  {:5} {:2}-bit  {:>4.1} bytes  {}  ({})",
                q.name, q.bits, q.bytes_per_param, bar, q.analogy)?;
        }
        writeln!(f)?;

        // The shrink
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ HOW MUCH DOES IT SHRINK?                                       │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  {:>8}     {:>10}  →  {:>10}     {:>8}",
            "Model", "FP16", "4-bit", "Shrink")?;
        writeln!(f, "  {:>8}     {:>10}  →  {:>10}     {:>8}",
            "─────", "────", "─────", "──────")?;

        for (params_b, memories) in &self.comparisons {
            let fp16 = &memories[1];
            let nf4 = &memories[3];
            let shrink = fp16.base_model_gb / nf4.base_model_gb;
            writeln!(f, "  {:>7.1}B     {:>9.1}GB  →  {:>9.1}GB     {:>7.0}×",
                params_b, fp16.base_model_gb, nf4.base_model_gb, shrink)?;
        }
        writeln!(f)?;

        // Memory comparison for training
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TRAINING MEMORY: WITHOUT vs WITH QLoRA                         │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  {:>8}  {:>12}  {:>12}  {:>10}",
            "Model", "Full Train", "QLoRA", "Fits 24GB?")?;
        writeln!(f, "  {:>8}  {:>12}  {:>12}  {:>10}",
            "─────", "──────────", "─────", "──────────")?;

        for (params_b, memories) in &self.comparisons {
            let nf4 = &memories[3];

            // Full training: ~16 bytes/param
            let full_gb = params_b * 16.0;

            let fits = if nf4.total_gb <= 24.0 { "YES ✓" }
                      else if nf4.total_gb <= 48.0 { "48GB card" }
                      else { "NO ✗" };

            writeln!(f, "  {:>7.1}B  {:>11.0}GB  {:>11.1}GB  {:>10}",
                params_b, full_gb, nf4.total_gb, fits)?;
        }
        writeln!(f)?;

        // QLoRA breakdown for 7B
        if let Some((_, memories)) = self.comparisons.iter().find(|(p, _)| (*p - 7.0).abs() < 0.1) {
            let nf4 = &memories[3];
            writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
            writeln!(f, "│ WHERE DOES THE MEMORY GO? (7B model)                           │")?;
            writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
            writeln!(f, "  Frozen model (4-bit):   {:>5.1}GB   ← shrunk, read-only", nf4.base_model_gb)?;
            writeln!(f, "  The diff (LoRA):        {:>5.2}GB   ← tiny, this is what learns", nf4.lora_gb)?;
            writeln!(f, "  Optimizer for diff:     {:>5.2}GB   ← gradients + Adam", nf4.overhead_gb)?;
            writeln!(f, "  ───────────────────────────────")?;
            writeln!(f, "  TOTAL:                  {:>5.1}GB   ← fits your 24GB GPU", nf4.total_gb)?;
            writeln!(f)?;
        }

        // Summary
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ QLoRA IN 4 STEPS:                                                ║")?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║  1. Shrink base model to 4-bit (14GB → 3.5GB)                    ║")?;
        writeln!(f, "║  2. Freeze it — no training, just reading                        ║")?;
        writeln!(f, "║  3. Add tiny LoRA diff — this is what learns                     ║")?;
        writeln!(f, "║  4. Train the diff at full precision — fixes quantization errors ║")?;
        writeln!(f, "║                                                                  ║")?;
        writeln!(f, "║  4-bit is blurry. LoRA sharpens it for your task.                ║")?;
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
    fn test_nf4_smallest() {
        let nf4 = &QUANT_LEVELS[3];
        assert_eq!(nf4.bits, 4);
        assert!(nf4.bytes_per_param < 1.0);
    }

    #[test]
    fn test_qlora_smaller_than_full() {
        let result = run();
        for (params_b, memories) in &result.comparisons {
            let full_fp16_gb = params_b * 16.0;
            let nf4 = &memories[3];
            assert!(nf4.total_gb < full_fp16_gb);
        }
    }

    #[test]
    fn test_7b_fits_24gb() {
        let result = run();
        if let Some((_, memories)) = result.comparisons.iter().find(|(p, _)| (*p - 7.0).abs() < 0.1) {
            let nf4 = &memories[3];
            assert!(nf4.total_gb < 24.0);
        }
    }

    #[test]
    fn test_display_runs() {
        let result = run();
        let display = format!("{}", result);
        assert!(display.contains("QLoRA"));
        assert!(display.contains("SHRINK"));
    }
}
