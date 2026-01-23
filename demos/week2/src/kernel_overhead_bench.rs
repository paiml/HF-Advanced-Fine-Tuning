//! Kernel Overhead Benchmark (Falsification Experiment)
//!
//! This benchmark tests the "Kernel Launch Overhead" hypothesis:
//! - Hypothesis: >50% of forward pass time is spent in launch/transfer overhead
//! - If CORROBORATED: Need kernel fusion for production
//! - If FALSIFIED: Current per-op CUDA is sufficient
//!
//! Uses InferenceTracer for empirical measurement.

use crate::inference_trace::{TRACER, TraceStep};

/// Simulate a matmul operation (CPU version for demo)
fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

/// Simulated overhead (memory allocation, kernel dispatch, etc.)
fn simulate_kernel_overhead() {
    // Simulate ~1ms of driver/launch overhead per operation
    std::thread::sleep(std::time::Duration::from_micros(100));
}

/// Run a single "layer" simulation
fn run_layer(hidden_size: usize, seq_len: usize) {
    let input = vec![0.1f32; seq_len * hidden_size];
    let weight = vec![0.01f32; hidden_size * hidden_size];

    // QKV Projection (3 matmuls)
    for proj in ["Q", "K", "V"] {
        TRACER.start(TraceStep::Overhead);
        simulate_kernel_overhead();
        TRACER.end(TraceStep::Overhead, format!("{} launch", proj));

        TRACER.start(TraceStep::KernelMatmul);
        let _ = matmul_cpu(&input, &weight, seq_len, hidden_size, hidden_size);
        TRACER.end(TraceStep::KernelMatmul, format!("{} proj", proj));
    }

    // Attention (Q@K^T, softmax@V)
    TRACER.start(TraceStep::Overhead);
    simulate_kernel_overhead();
    TRACER.end(TraceStep::Overhead, "attn launch");

    TRACER.start(TraceStep::KernelAttention);
    let _ = matmul_cpu(&input, &input, seq_len, hidden_size, seq_len); // Q@K^T
    TRACER.end(TraceStep::KernelAttention, "Q@K^T");

    // Output projection
    TRACER.start(TraceStep::Overhead);
    simulate_kernel_overhead();
    TRACER.end(TraceStep::Overhead, "O launch");

    TRACER.start(TraceStep::KernelMatmul);
    let _ = matmul_cpu(&input, &weight, seq_len, hidden_size, hidden_size);
    TRACER.end(TraceStep::KernelMatmul, "O proj");
}

/// Configuration for the benchmark
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub hidden_size: usize,
    pub seq_len: usize,
    pub num_layers: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,  // Tiny for demo
            seq_len: 32,
            num_layers: 2,
        }
    }
}

/// Run the kernel overhead benchmark
pub fn run(config: BenchConfig) -> String {
    // Clear any previous measurements
    let tracer = &*TRACER;

    TRACER.start(TraceStep::Forward);

    for _layer_idx in 0..config.num_layers {
        run_layer(config.hidden_size, config.seq_len);
    }

    TRACER.end(TraceStep::Forward, format!("{} layers", config.num_layers));

    // Generate report with Dr. Popper analysis
    tracer.report()
}

/// Pretty print results
pub fn print_results(config: &BenchConfig, report: &str) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     KERNEL OVERHEAD BENCHMARK (Phase 19 Falsification)           ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Config:                                                         ║");
    println!("║    Hidden Size: {:>6}                                           ║", config.hidden_size);
    println!("║    Seq Length:  {:>6}                                           ║", config.seq_len);
    println!("║    Num Layers:  {:>6}                                           ║", config.num_layers);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("{}", report);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_cpu() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let c = matmul_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_benchmark_runs() {
        let config = BenchConfig::default();
        let report = run(config);
        assert!(report.contains("Forward"));
    }
}
