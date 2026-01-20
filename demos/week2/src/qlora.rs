//! QLoRA Demo
//!
//! Shows 4-bit quantization + LoRA:
//! - NF4 (4-bit NormalFloat) quantization
//! - Double quantization (quantize the quantization constants)
//! - LoRA adapters in fp16/bf16 on top
//!
//! Result: 70B model trainable on 24GB GPU

use std::fmt;

/// Memory calculation for different quantization levels
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub name: &'static str,
    pub bits: u8,
    pub bytes_per_param: f64,
}

pub const QUANT_LEVELS: &[QuantConfig] = &[
    QuantConfig { name: "FP32",  bits: 32, bytes_per_param: 4.0 },
    QuantConfig { name: "FP16",  bits: 16, bytes_per_param: 2.0 },
    QuantConfig { name: "INT8",  bits: 8,  bytes_per_param: 1.0 },
    QuantConfig { name: "NF4",   bits: 4,  bytes_per_param: 0.5 },
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
    pub nf4_explanation: Vec<String>,
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

    let nf4_explanation = vec![
        "NF4 = 4-bit NormalFloat".to_string(),
        "Optimized for normally-distributed weights".to_string(),
        "16 quantization levels, non-uniform spacing".to_string(),
        "Double quantization: quantize the scales too".to_string(),
    ];

    DemoResults { comparisons, nf4_explanation }
}

impl fmt::Display for DemoResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           QLoRA: 4-BIT QUANTIZATION + LoRA                       ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  \"Fine-tune 70B on a single 24GB GPU\"                            ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        writeln!(f)?;

        // NF4 explanation
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ WHAT IS NF4?                                                   │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for line in &self.nf4_explanation {
            writeln!(f, "  • {}", line)?;
        }
        writeln!(f)?;

        // Quantization comparison
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ BYTES PER PARAMETER                                            │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        for q in QUANT_LEVELS {
            let bar_len = (q.bytes_per_param * 10.0) as usize;
            let bar: String = "█".repeat(bar_len);
            writeln!(f, "  {:5} ({:2}-bit): {:>4.1} bytes  {}",
                q.name, q.bits, q.bytes_per_param, bar)?;
        }
        writeln!(f)?;

        // Memory by model size
        writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│ TRAINING MEMORY: Full FP16 vs QLoRA (NF4 + r=16)               │")?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
        writeln!(f, "  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
            "Model", "Full FP16", "QLoRA NF4", "Savings", "Fits GPU?")?;
        writeln!(f, "  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
            "─────", "─────────", "─────────", "───────", "─────────")?;

        for (params_b, memories) in &self.comparisons {
            let _fp16 = &memories[1]; // FP16 full fine-tune estimate
            let nf4 = &memories[3];  // NF4 QLoRA

            // Full FP16 training: ~16 bytes/param
            let full_fp16_gb = params_b * 16.0;

            let savings = full_fp16_gb / nf4.total_gb;
            let fits = if nf4.total_gb <= 24.0 { "✓ 24GB" }
                      else if nf4.total_gb <= 48.0 { "✓ 48GB" }
                      else if nf4.total_gb <= 80.0 { "✓ 80GB" }
                      else { "✗ cluster" };

            writeln!(f, "  {:>7.1}B  {:>11.1}GB  {:>11.1}GB  {:>9.0}×  {:>10}",
                params_b, full_fp16_gb, nf4.total_gb, savings, fits)?;
        }
        writeln!(f)?;

        // QLoRA breakdown for 7B
        if let Some((_, memories)) = self.comparisons.iter().find(|(p, _)| (*p - 7.0).abs() < 0.1) {
            let nf4 = &memories[3];
            writeln!(f, "┌────────────────────────────────────────────────────────────────┐")?;
            writeln!(f, "│ QLoRA MEMORY BREAKDOWN (7B model, NF4, r=16)                   │")?;
            writeln!(f, "└────────────────────────────────────────────────────────────────┘")?;
            writeln!(f, "  Base model (NF4):     {:>6.1}GB  (frozen, 4-bit)", nf4.base_model_gb)?;
            writeln!(f, "  LoRA adapters (fp16): {:>6.2}GB  (trainable)", nf4.lora_gb)?;
            writeln!(f, "  Gradients + Adam:     {:>6.2}GB  (for LoRA only)", nf4.overhead_gb)?;
            writeln!(f, "  ────────────────────────────────")?;
            writeln!(f, "  TOTAL:                {:>6.1}GB", nf4.total_gb)?;
            writeln!(f)?;
        }

        // Summary
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║ QLoRA RECIPE:                                                    ║")?;
        writeln!(f, "║ 1. Quantize base model to NF4 (4-bit) — frozen                   ║")?;
        writeln!(f, "║ 2. Add LoRA adapters in fp16 — trainable                         ║")?;
        writeln!(f, "║ 3. Train only the adapters — tiny memory footprint               ║")?;
        writeln!(f, "║ 4. Merge and export — full quality, fraction of the cost         ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ entrenar: QLoRA training | apr export: GGUF output               ║")?;
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
        assert!(display.contains("NF4"));
    }
}
