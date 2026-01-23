use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Mutex;
use std::fmt;

/// The lifecycle steps of a token or training step.
/// This enum strictly defines the observability model (ITP-SPEC-001).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceStep {
    /// Tokenization and preprocessing
    Tokenize,
    /// Embedding lookup
    Embed,
    /// Forward pass through Transformer layers
    Forward,
    /// Backward pass (Gradient computation)
    Backward,
    /// Optimization step (Weight update)
    Optimizer,
    /// Data transfer (CPU <-> GPU)
    DataTransfer,
    /// Specific Kernel: Matrix Multiplication
    KernelMatmul,
    /// Specific Kernel: Attention
    KernelAttention,
    /// Overhead: Driver/Launch latency
    Overhead,
}

impl fmt::Display for TraceStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A measurement of a single step.
#[derive(Debug, Clone)]
pub struct TraceMeasurement {
    pub step: TraceStep,
    pub start: Instant,
    pub end: Instant,
    pub metadata: String,
}

impl TraceMeasurement {
    pub fn duration(&self) -> Duration {
        self.end.duration_since(self.start)
    }
}

/// The global tracer (Singleton-style for easy injection).
/// In a real system, this might be thread-local or passed via context.
pub struct InferenceTracer {
    measurements: Mutex<Vec<TraceMeasurement>>,
    active_spans: Mutex<HashMap<TraceStep, Instant>>,
}

impl InferenceTracer {
    /// Create a new tracer.
    pub fn new() -> Self {
        Self {
            measurements: Mutex::new(Vec::new()),
            active_spans: Mutex::new(HashMap::new()),
        }
    }

    /// Start measuring a step.
    pub fn start(&self, step: TraceStep) {
        let mut spans = self.active_spans.lock().unwrap();
        spans.insert(step, Instant::now());
    }

    /// Stop measuring a step and record it.
    pub fn end(&self, step: TraceStep, metadata: impl Into<String>) {
        let mut spans = self.active_spans.lock().unwrap();
        if let Some(start) = spans.remove(&step) {
            let end = Instant::now();
            let mut measurements = self.measurements.lock().unwrap();
            measurements.push(TraceMeasurement {
                step,
                start,
                end,
                metadata: metadata.into(),
            });
        }
    }

    /// Run a closure within a measured span.
    pub fn span<F, R>(&self, step: TraceStep, metadata: impl Into<String>, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.start(step);
        let result = f();
        self.end(step, metadata);
        result
    }

    /// Generate a report of the measurements.
    pub fn report(&self) -> String {
        let measurements = self.measurements.lock().unwrap();
        if measurements.is_empty() {
            return "No measurements recorded.".to_string();
        }

        let mut totals: HashMap<TraceStep, Duration> = HashMap::new();
        let mut counts: HashMap<TraceStep, usize> = HashMap::new();
        let total_time = measurements.last().unwrap().end - measurements.first().unwrap().start;

        for m in measurements.iter() {
            *totals.entry(m.step).or_default() += m.duration();
            *counts.entry(m.step).or_default() += 1;
        }

        let mut output = String::from("\n=== Inference Trace Report (ITP-SPEC-001) ===\n");
        output.push_str(&format!("Total Wall Time: {:.2?}\n", total_time));
        output.push_str("---------------------------------------------\n");
        output.push_str(&format!("{:<20} | {:<10} | {:<15} | {:<8}\n", "Step", "Count", "Total Duration", "% Time"));
        output.push_str("---------------------------------------------\
");

        // Sort by duration descending
        let mut sorted_steps: Vec<_> = totals.keys().collect();
        sorted_steps.sort_by(|a, b| totals[b].cmp(&totals[a]));

        for step in sorted_steps {
            let duration = totals[step];
            let count = counts[step];
            let percentage = (duration.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
            output.push_str(&format!(
                "{:<20} | {:<10} | {:<15.2?} | {:>7.2}%\n",
                step.to_string(),
                count,
                duration,
                percentage
            ));
        }
        output.push_str("---------------------------------------------\n");
        
        // Critical Rationalism Check
        if let Some(forward_time) = totals.get(&TraceStep::Forward) {
             if let Some(matmul_time) = totals.get(&TraceStep::KernelMatmul) {
                 let overhead = *forward_time - *matmul_time;
                 let overhead_pct = (overhead.as_secs_f64() / forward_time.as_secs_f64()) * 100.0;
                 output.push_str(&format!("\n[Dr. Popper Analysis]\n"));
                 output.push_str(&format!("Forward Pass:     {:.2?}\n", forward_time));
                 output.push_str(&format!("Compute (Matmul): {:.2?}\n", matmul_time));
                 output.push_str(&format!("Overhead (Muda):  {:.2?} ({:.2}%)\n", overhead, overhead_pct));
                 
                 if overhead_pct > 50.0 {
                     output.push_str("\n🔴 FALSIFICATION: Overhead dominates computation. 'Kernel Launch' hypothesis CORROBORATED.\n");
                 } else {
                     output.push_str("\n🟢 RESULT: Compute dominates overhead. 'Kernel Launch' hypothesis WEAKENED.\n");
                 }
             }
        }

        output
    }
}

// Global static instance for easy access across the crate
use std::sync::LazyLock;
pub static TRACER: LazyLock<InferenceTracer> = LazyLock::new(|| InferenceTracer::new());
