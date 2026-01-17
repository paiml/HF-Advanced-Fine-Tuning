//! Demo 1: Scalar vs SIMD vs GPU computation
//!
//! Demonstrates the performance hierarchy of compute backends using trueno.
//! Shows two scenarios:
//! 1. Small dot product - SIMD wins (GPU transfer overhead dominates)
//! 2. Large matmul - GPU wins (compute-bound, high arithmetic intensity)

use std::time::Instant;
use trueno::backends::gpu::GpuDevice;
use trueno::{Matrix, Vector};

/// Benchmark result for one backend
#[derive(Debug, Clone, PartialEq)]
pub struct BenchResult {
    pub name: &'static str,
    pub duration_us: f64,
    pub speedup: f64,
    pub available: bool,
}

impl BenchResult {
    /// Create a new benchmark result
    #[must_use]
    pub fn new(name: &'static str, duration_us: f64, speedup: f64) -> Self {
        Self {
            name,
            duration_us,
            speedup,
            available: true,
        }
    }

    /// Create an unavailable benchmark result
    #[must_use]
    pub fn unavailable(name: &'static str) -> Self {
        Self {
            name,
            duration_us: 0.0,
            speedup: 0.0,
            available: false,
        }
    }
}

/// Results for one operation type
#[derive(Debug, Clone)]
pub struct OpResults {
    pub name: &'static str,
    pub description: String,
    pub backends: Vec<BenchResult>,
}

impl OpResults {
    #[must_use]
    pub fn new(name: &'static str, description: String, backends: Vec<BenchResult>) -> Self {
        Self {
            name,
            description,
            backends,
        }
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&BenchResult> {
        self.backends.iter().find(|b| b.name == name)
    }
}

/// All benchmark results
#[derive(Debug, Clone)]
pub struct Results {
    pub operations: Vec<OpResults>,
    pub iterations: usize,
}

impl Results {
    #[must_use]
    pub fn new(operations: Vec<OpResults>, iterations: usize) -> Self {
        Self {
            operations,
            iterations,
        }
    }
}

// =============================================================================
// Scalar implementations
// =============================================================================

/// Scalar dot product
#[inline(never)]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar matrix multiplication (naive O(n³))
#[inline(never)]
pub fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// =============================================================================
// SIMD implementations (trueno)
// =============================================================================

/// SIMD dot product via trueno
#[inline(never)]
pub fn dot_simd(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
    a.dot(b).expect("SIMD dot product failed")
}

/// SIMD matmul via trueno
#[inline(never)]
pub fn matmul_simd(a: &Matrix<f32>, b: &Matrix<f32>) -> Matrix<f32> {
    a.matmul(b).expect("SIMD matmul failed")
}

// =============================================================================
// GPU implementations (trueno)
// =============================================================================

/// GPU dot product via trueno
#[inline(never)]
pub fn dot_gpu(a: &[f32], b: &[f32], device: &GpuDevice) -> Option<f32> {
    device.dot(a, b).ok()
}

/// GPU matmul via trueno
#[inline(never)]
pub fn matmul_gpu(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> bool {
    device.matmul(a, b, c, m, k, n).is_ok()
}

/// Check if GPU is available
#[must_use]
pub fn gpu_available() -> bool {
    GpuDevice::is_available()
}

/// Try to create a GPU device
#[must_use]
pub fn try_gpu_device() -> Option<GpuDevice> {
    GpuDevice::new().ok()
}

// =============================================================================
// Benchmarking
// =============================================================================

fn bench_op<F>(iterations: usize, mut op: F) -> f64
where
    F: FnMut(),
{
    // Warmup
    op();

    let start = Instant::now();
    for _ in 0..iterations {
        op();
        std::hint::black_box(());
    }
    start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64
}

/// Run small dot product benchmark
fn bench_dot(size: usize, iterations: usize, gpu: Option<&GpuDevice>) -> OpResults {
    let a_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    let b_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();
    let a_simd = Vector::from_slice(&a_vec);
    let b_simd = Vector::from_slice(&b_vec);

    let mut backends = Vec::new();

    // Scalar
    let scalar_us = bench_op(iterations, || {
        std::hint::black_box(dot_scalar(&a_vec, &b_vec));
    });
    backends.push(BenchResult::new("Scalar", scalar_us, 1.0));

    // SIMD
    let simd_us = bench_op(iterations, || {
        std::hint::black_box(dot_simd(&a_simd, &b_simd));
    });
    backends.push(BenchResult::new("SIMD", simd_us, scalar_us / simd_us));

    // GPU
    if let Some(device) = gpu {
        if dot_gpu(&a_vec, &b_vec, device).is_some() {
            let gpu_us = bench_op(iterations, || {
                std::hint::black_box(dot_gpu(&a_vec, &b_vec, device));
            });
            backends.push(BenchResult::new("GPU", gpu_us, scalar_us / gpu_us));
        } else {
            backends.push(BenchResult::unavailable("GPU"));
        }
    } else {
        backends.push(BenchResult::unavailable("GPU"));
    }

    OpResults::new(
        "Dot Product (small)",
        format!("n={size} (memory-bound, low arithmetic intensity)"),
        backends,
    )
}

/// Run large matmul benchmark
fn bench_matmul(size: usize, iterations: usize, gpu: Option<&GpuDevice>) -> OpResults {
    let m = size;
    let k = size;
    let n = size;

    let a_vec: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
    let b_vec: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
    let mut c_scalar = vec![0.0_f32; m * n];
    let mut c_gpu = vec![0.0_f32; m * n];

    let a_mat = Matrix::from_slice(m, k, &a_vec).expect("matrix A");
    let b_mat = Matrix::from_slice(k, n, &b_vec).expect("matrix B");

    let mut backends = Vec::new();

    // Scalar
    let scalar_us = bench_op(iterations, || {
        matmul_scalar(&a_vec, &b_vec, &mut c_scalar, m, k, n);
    });
    backends.push(BenchResult::new("Scalar", scalar_us, 1.0));

    // SIMD
    let simd_us = bench_op(iterations, || {
        std::hint::black_box(matmul_simd(&a_mat, &b_mat));
    });
    backends.push(BenchResult::new("SIMD", simd_us, scalar_us / simd_us));

    // GPU
    if let Some(device) = gpu {
        if matmul_gpu(&a_vec, &b_vec, &mut c_gpu, m, k, n, device) {
            let gpu_us = bench_op(iterations, || {
                matmul_gpu(&a_vec, &b_vec, &mut c_gpu, m, k, n, device);
            });
            backends.push(BenchResult::new("GPU", gpu_us, scalar_us / gpu_us));
        } else {
            backends.push(BenchResult::unavailable("GPU"));
        }
    } else {
        backends.push(BenchResult::unavailable("GPU"));
    }

    OpResults::new(
        "Matrix Multiply (large)",
        format!("{size}x{size} (compute-bound, high arithmetic intensity)"),
        backends,
    )
}

/// Run all benchmarks
#[must_use]
pub fn run(dot_size: usize, matmul_size: usize, iterations: usize) -> Results {
    let gpu = try_gpu_device();

    let operations = vec![
        bench_dot(dot_size, iterations, gpu.as_ref()),
        bench_matmul(matmul_size, iterations, gpu.as_ref()),
    ];

    Results::new(operations, iterations)
}

// =============================================================================
// Output formatting
// =============================================================================

/// Format speedup for display
#[must_use]
pub fn format_speedup(speedup: f64) -> String {
    if speedup < 0.1 {
        format!("{:.2}x", speedup)
    } else {
        format!("{:.1}x", speedup)
    }
}

/// Generate bar for TUI display (minimum 1 char so all rows show a bar)
#[must_use]
pub fn generate_bar(ratio: f64, width: usize) -> String {
    let bar_len = ((ratio * width as f64) as usize).clamp(1, width);
    "\u{2588}".repeat(bar_len)
}

/// Bar color (single color for all)
#[must_use]
pub fn bar_color() -> &'static str {
    "\x1b[32m" // green
}

/// Print results to stdout
pub fn print_stdout(results: &Results) {
    println!("=== Scalar vs SIMD vs GPU ===");
    println!("Iterations: {}\n", results.iterations);

    for op in &results.operations {
        println!("--- {} ---", op.name);
        println!("{}\n", op.description);
        println!("{:<10} {:>12} {:>10}", "Backend", "Time (us)", "Speedup");
        println!("{}", "-".repeat(35));

        for b in &op.backends {
            if b.available {
                println!(
                    "{:<10} {:>12.2} {:>10}",
                    b.name,
                    b.duration_us,
                    format_speedup(b.speedup)
                );
            } else {
                println!("{:<10} {:>12} {:>10}", b.name, "---", "---");
            }
        }
        println!();
    }
}

/// Compute log-scale bar ratio (0.0 to 1.0)
#[must_use]
pub fn log_scale_ratio(speedup: f64, min_speedup: f64, max_speedup: f64) -> f64 {
    if max_speedup <= min_speedup || speedup <= 0.0 {
        return 0.0;
    }
    let log_min = min_speedup.max(1e-6).ln();
    let log_max = max_speedup.ln();
    let log_val = speedup.max(1e-6).ln();

    if (log_max - log_min).abs() < 1e-9 {
        return 1.0;
    }
    ((log_val - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
}

/// Render TUI bar chart (bar length = log-scale duration, sorted by speed)
pub fn render_tui(results: &Results) {
    println!("\n\x1b[1m=== Scalar vs SIMD vs GPU ===\x1b[0m");
    println!(
        "\x1b[90mIterations: {} | Bar = log(time), sorted fastest→slowest\x1b[0m\n",
        results.iterations
    );

    for op in &results.operations {
        println!("\x1b[1;4m{}\x1b[0m", op.name);
        println!("\x1b[90m{}\x1b[0m\n", op.description);

        // Sort by duration (fastest first)
        let mut sorted: Vec<_> = op.backends.iter().filter(|b| b.available).collect();
        sorted.sort_by(|a, b| a.duration_us.partial_cmp(&b.duration_us).unwrap());

        let min_duration = sorted.first().map(|b| b.duration_us).unwrap_or(1.0);
        let max_duration = sorted.last().map(|b| b.duration_us).unwrap_or(1.0);
        let color = bar_color();

        for b in &sorted {
            let ratio = log_scale_ratio(b.duration_us, min_duration, max_duration);
            let bar = generate_bar(ratio, 25);
            println!(
                "{:<10} {}{:<25}\x1b[0m {:>10.2} us  ({})",
                b.name,
                color,
                bar,
                b.duration_us,
                format_speedup(b.speedup)
            );
        }

        // Show unavailable backends at the end
        for b in op.backends.iter().filter(|b| !b.available) {
            println!(
                "{:<10} \x1b[90m{:<25} {:>10} {:>10}\x1b[0m",
                b.name, "---", "---", "N/A"
            );
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_scalar_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot_scalar(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_scalar_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert!((dot_scalar(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_scalar_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0_f32; 4];
        matmul_scalar(&a, &b, &mut c, 2, 2, 2);
        // [1,2] * [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_simd_basic() {
        let a = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0_f32, 5.0, 6.0]);
        assert!((dot_simd(&a, &b) - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_simd_2x2() {
        let a = Matrix::from_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]).unwrap();
        let c = matmul_simd(&a, &b);
        let data = c.as_slice();
        assert!((data[0] - 19.0).abs() < 1e-4);
        assert!((data[1] - 22.0).abs() < 1e-4);
    }

    #[test]
    fn test_bench_result_new() {
        let r = BenchResult::new("test", 100.0, 2.5);
        assert_eq!(r.name, "test");
        assert!(r.available);
        assert!((r.speedup - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_bench_result_unavailable() {
        let r = BenchResult::unavailable("gpu");
        assert!(!r.available);
    }

    #[test]
    fn test_op_results_get() {
        let backends = vec![
            BenchResult::new("Scalar", 10.0, 1.0),
            BenchResult::new("SIMD", 5.0, 2.0),
        ];
        let op = OpResults::new("test", "desc".to_string(), backends);
        assert!(op.get("Scalar").is_some());
        assert!(op.get("GPU").is_none());
    }

    #[test]
    fn test_format_speedup() {
        assert_eq!(format_speedup(10.5), "10.5x");
        assert_eq!(format_speedup(0.05), "0.05x");
    }

    #[test]
    fn test_generate_bar() {
        assert_eq!(generate_bar(1.0, 10), "\u{2588}".repeat(10));
        assert_eq!(generate_bar(0.5, 10), "\u{2588}".repeat(5));
        assert_eq!(generate_bar(0.0, 10), "\u{2588}".repeat(1)); // min 1
        assert_eq!(generate_bar(1.5, 10), "\u{2588}".repeat(10)); // capped
    }

    #[test]
    fn test_bar_color() {
        assert_eq!(bar_color(), "\x1b[32m"); // green
    }

    #[test]
    fn test_gpu_available_no_panic() {
        let _ = gpu_available();
    }

    #[test]
    fn test_try_gpu_device_no_panic() {
        let _ = try_gpu_device();
    }

    #[test]
    fn test_run_completes() {
        let results = run(1000, 64, 5);
        assert_eq!(results.operations.len(), 2);
    }

    #[test]
    fn test_print_stdout_no_panic() {
        let results = run(100, 32, 2);
        print_stdout(&results);
    }

    #[test]
    fn test_render_tui_no_panic() {
        let results = run(100, 32, 2);
        render_tui(&results);
    }

    #[test]
    fn test_bench_result_equality() {
        let a = BenchResult::new("test", 10.0, 2.0);
        let b = BenchResult::new("test", 10.0, 2.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_results_new() {
        let ops = vec![OpResults::new("op", "desc".to_string(), vec![])];
        let results = Results::new(ops, 100);
        assert_eq!(results.iterations, 100);
        assert_eq!(results.operations.len(), 1);
    }

    #[test]
    fn test_dot_gpu_with_device() {
        if let Some(device) = try_gpu_device() {
            let a = vec![1.0_f32, 2.0, 3.0];
            let b = vec![4.0_f32, 5.0, 6.0];
            if let Some(result) = dot_gpu(&a, &b, &device) {
                assert!((result - 32.0).abs() < 1.0);
            }
        }
    }

    #[test]
    fn test_matmul_gpu_with_device() {
        if let Some(device) = try_gpu_device() {
            let a = vec![1.0_f32, 2.0, 3.0, 4.0];
            let b = vec![5.0_f32, 6.0, 7.0, 8.0];
            let mut c = vec![0.0_f32; 4];
            if matmul_gpu(&a, &b, &mut c, 2, 2, 2, &device) {
                assert!((c[0] - 19.0).abs() < 1.0);
            }
        }
    }

    #[test]
    fn test_op_results_new() {
        let backends = vec![BenchResult::new("test", 10.0, 1.0)];
        let op = OpResults::new("myop", "description".to_string(), backends);
        assert_eq!(op.name, "myop");
        assert_eq!(op.description, "description");
        assert_eq!(op.backends.len(), 1);
    }

    #[test]
    fn test_log_scale_ratio() {
        // Max speedup gets ratio 1.0
        assert!((log_scale_ratio(10.0, 1.0, 10.0) - 1.0).abs() < 1e-6);
        // Min speedup gets ratio 0.0
        assert!((log_scale_ratio(1.0, 1.0, 10.0) - 0.0).abs() < 1e-6);
        // Middle value on log scale
        let mid = log_scale_ratio(3.16, 1.0, 10.0); // sqrt(10) ≈ 3.16
        assert!(mid > 0.4 && mid < 0.6);
    }

    #[test]
    fn test_log_scale_ratio_edge_cases() {
        // Zero speedup
        assert_eq!(log_scale_ratio(0.0, 1.0, 10.0), 0.0);
        // Equal min/max
        assert_eq!(log_scale_ratio(5.0, 5.0, 5.0), 0.0);
        // Very small speedup
        assert!(log_scale_ratio(0.001, 0.001, 10.0) >= 0.0);
    }

    #[test]
    fn test_render_tui_with_unavailable() {
        let ops = vec![OpResults::new(
            "test",
            "desc".to_string(),
            vec![
                BenchResult::new("Scalar", 100.0, 1.0),
                BenchResult::unavailable("GPU"),
            ],
        )];
        let results = Results::new(ops, 10);
        render_tui(&results);
    }

    #[test]
    fn test_print_stdout_with_unavailable() {
        let ops = vec![OpResults::new(
            "test",
            "desc".to_string(),
            vec![
                BenchResult::new("Scalar", 100.0, 1.0),
                BenchResult::unavailable("GPU"),
            ],
        )];
        let results = Results::new(ops, 10);
        print_stdout(&results);
    }

    #[test]
    fn test_format_speedup_small() {
        assert_eq!(format_speedup(0.01), "0.01x");
        assert_eq!(format_speedup(0.001), "0.00x");
    }

    #[test]
    fn test_render_tui_zero_max() {
        let ops = vec![OpResults::new(
            "test",
            "desc".to_string(),
            vec![BenchResult::new("Scalar", 0.0, 1.0)],
        )];
        let results = Results::new(ops, 10);
        render_tui(&results);
    }
}
