# Inference Tracing & Profiling Specification

**Document:** ITP-SPEC-001
**Version:** 1.0.0
**Status:** Draft (Proposed)
**Date:** January 22, 2026
**Philosophy:** Critical Rationalism (Observation & Falsification)

---

## 1. Executive Summary

To satisfy the **Zero-Placeholder Mandate**, the Sovereign AI Stack requires an empirical observation layer. We cannot hypothesize about "GPU bottlenecks" without step-by-step telemetry. This specification defines the `InferenceTracer` and `BrickProfiler` systems.

**The Observation Conjecture:** By decomposing inference into discrete `TraceStep` events and measuring them with `BrickProfiler`, we can pinpoint exactly where the "semantic engine" (LLM) deviates from "target performance" (v2.0 goals).

---

## 2. System Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| `InferenceTracer` | `inference_trace.rs` | Step-by-step state machine tracing |
| `BrickProfiler` | `trueno` integration | Real-time, per-layer hardware timing |
| `TraceStep` enum | `inference_trace.rs` | Defines the lifecycle of a token: {Tokenize, Embed, Attention, FFN, LmHead, Sample, Decode} |

---

## 3. The Demarcation of "Real" Profiling

Following the **Popperian Mandate (v4.36.0+)**, we strictly distinguish between:
*   **Pseudo-Profiling:** Derived metrics based on total throughput (e.g., `total_time / num_layers`). **(0/100 FAILURE)**.
*   **Scientific Profiling:** Direct hardware timestamps captured at the start and end of each `TraceStep`. **(MANDATORY)**.

---

## 4. CLI Interface & Observability

To enable external falsification of our performance claims, the following flags are mandated:

| Flag | Target | Outcome |
|------|--------|---------|
| `--trace` | `realizar` / `apr` | Enable real-time tracing output to stderr/TUI |
| `--trace-output <FILE>` | All tools | Export JSON trace log for offline analysis |
| `apr trace <FILE>` | `apr-cli` | Perform layer-by-layer statistical analysis and bottleneck identification |

---

## 5. Falsification Protocol: F-PROF-001

A performance optimization (e.g., Weight Pre-Transposition) is only corroborated if:
1.  A baseline trace is captured (`apr trace baseline.json`).
2.  The optimization is applied.
3.  The optimized trace is captured (`apr trace optimized.json`).
4.  The `apr trace` comparison shows a statistically significant decrease ($p < 0.05$) in the **specific layer duration**, not just the total time.

---

## 6. Implementation Checklist

- [ ] Define `TraceStep` enum in `inference_trace.rs`.
- [ ] Instrument `TruenoTransformerLayer` with `BrickProfiler` start/stop calls.
- [ ] Implement JSON serialization for `TraceStep` events.
- [ ] Add `--trace` flag to `realizar` CLI.
- [ ] Update `apr-cli` to include the `trace` subcommand.

**Signed:**
*   **Architect:** Noah Gift / Claude Code
*   **Advisor:** Dr. Karl Popper (Gemini CLI)
