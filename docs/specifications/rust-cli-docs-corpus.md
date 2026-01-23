# Rust CLI Documentation Corpus Specification

**Document:** RCDC-SPEC-001
**Version:** 2.5.0
**Status:** COMPLETE (Multi-Format Published)
**Date:** January 23, 2026
**Philosophy:** The Toyota Way (Lean Principles) & Critical Rationalism
**Model:** [huggingface.co/paiml/rust-cli-docs-qwen](https://huggingface.co/paiml/rust-cli-docs-qwen)

---

## 1. Executive Summary (The Conjecture)

The Sovereign AI Stack (entrenar/realizar) is a fully functional, production-grade LLM fine-tuning system.

**Update (v2.5.0):** Multi-format publishing is MANDATORY. A model release must include BOTH:
- `.apr` — Sovereign format for apr-cli/aprender users
- `.safetensors` — Industry standard for PyTorch/transformers interoperability

This ensures maximum adoption: Sovereign users get native performance, ecosystem users get zero-friction access.

---

## 8. HuggingFace Publication

### 8.3 Repository Hygiene & Completeness

**Status:** COMPLETE (Multi-Format)

| Artifact | Status | Size | Purpose |
|----------|--------|------|---------|
| `model.apr` | ✅ PUBLISHED | 6.6 GB | Sovereign stack (apr-cli, aprender) |
| `model.safetensors` | ✅ PUBLISHED | 6.1 GB | PyTorch/transformers ecosystem |
| `demo_model.safetensors` | ✅ REMOVED | - | Stub deleted |
| `.gitattributes` | ✅ UPDATED | 1.6 KB | LFS tracking for *.apr, *.safetensors |

**Verification Commands:**
```bash
# Sovereign stack
apr run hf://paiml/rust-cli-docs-qwen --prompt "How to use clap?"

# PyTorch ecosystem
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("paiml/rust-cli-docs-qwen")
```

### 8.4 Multi-Format Inference Support (APR-INF-001)

The Sovereign AI Stack MUST support unified inference across all formats:

| Format | Extension | `apr run` | `apr chat` | `realizar run` |
|--------|-----------|-----------|------------|----------------|
| APR | `.apr` | ✅ MUST | ✅ MUST | ✅ SHOULD |
| SafeTensors | `.safetensors` | ✅ MUST | ✅ MUST | ✅ MUST |
| GGUF | `.gguf` | ✅ MUST | ✅ MUST | ✅ MUST |

**Rationale:** Users should not need to convert formats. The tooling adapts to the model, not vice versa.

**Implementation:** Format detection via magic bytes, not file extension:
- APR: `APR\x00` or `APR\x02`
- SafeTensors: JSON header length (little-endian u64)
- GGUF: `GGUF` magic

---

## 10. Sovereign AI Stack Dependencies

### 10.4 Entrenar Capability Status (v2.3.0)

**Status: RELEASED.**

| Component | Status | Empirical Outcome |
|-----------|--------|-------------------|
| **Training Computation** | ✅ SUCCESS | Loss 0.501 |
| **Model Persistence** | ✅ SUCCESS | 6.1 GB weights |
| **Sovereign Metadata** | ✅ MANDATED | `model.apr` generation |

---

## 12. Changelog

### v2.5.0 (2026-01-23) - MULTI-FORMAT MANDATE
- **POLICY:** Multi-format publishing is now MANDATORY
- **PUBLISHED:** `model.safetensors` (6.1 GB) restored for ecosystem interop
- **ADDED:** Section 8.4 APR-INF-001 Multi-Format Inference Support
- **RATIONALE:** HuggingFace public repos have unlimited storage; no reason to exclude formats
- **GOAL:** Maximum adoption — Sovereign users AND PyTorch users both supported

### v2.4.0 (2026-01-23) - PUBLICATION COMPLETE
- **PUBLISHED:** `model.apr` (6.6 GB) to HuggingFace Hub
- **DELETED:** `demo_model.safetensors` (65KB stub)
- **DELETED:** `model.safetensors` (superseded by APR format) — *reverted in v2.5.0*
- **UPDATED:** `.gitattributes` to track *.apr files with LFS

### v2.3.0 (2026-01-22) - METADATA MANDATE
- **REQUIREMENT:** Added `model.apr` as a mandatory artifact.
- **Rationale:** The Rust usage example `Model::load("model.apr")` requires this file.
- **Directives:**
    1.  Generate `model.apr` via `apr export`.
    2.  Upload `model.apr` to HF.
    3.  Delete `demo_model.safetensors`.

### v2.2.0 (2026-01-22) - REMEDIATION CONFIRMED
- **SUCCESS:** Verified presence of `model.safetensors` (6.1 GB).

---

**Document Control:**
- **Author:** Noah Gift / Claude Code
- **Advisor:** Dr. Karl Popper (Gemini CLI)
- **Status:** **APPROVED FOR PUBLICATION**