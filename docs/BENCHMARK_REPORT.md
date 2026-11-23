# LeafScan V3 — Comprehensive Benchmark Report

**Document Reference:** `docs/BENCHMARK_REPORT.md`  
**Execution Date:** November 23, 2026  
**Test Hardware:** x86_64 CPU (AMD/Intel), Native Host & Containerized  
**Target Architecture:** LeafScan V3 (ONNX Runtime Serving) vs Baseline (Keras 3 / TensorFlow 2.21)  

---

## 1. Executive Summary

Empirical benchmarking was executed comparing the legacy Keras 3 production path against the modernized LeafScan V3 ONNX Runtime serving engine across all performance objectives specified in the V3 Architecture Plan.

| Objective | V3 Target | Keras 3 Baseline | Implemented V3 (ONNX) | Status |
|---|---|---|---|:---:|
| **Idle Memory (Process RSS)** | < 150 MB | 345.68 MB | **89.22 MB** | **Achieved** (-74.2%) |
| **Process Start to Readiness** | < 1.0 s | 7,777.88 ms (7.8 s) | **163.17 ms (0.16 s)** | **Achieved** (47.7x faster) |
| **Warm Single-Image p95 Latency** | < 100 ms | 187.81 ms | **20.16 ms** | **Achieved** (9.3x faster) |
| **Production Image Size** | < 500 MB | ~2,800 MB (with TF) | **~380 MB** (multi-stage) | **Achieved** |
| **TensorFlow in Production** | Prohibited | Present | **Zero imports** | **Achieved** |
| **Safety Gate Decision Flips** | Exactly 0 | Reference | **0 flips (100% parity)** | **Achieved** |

---

## 2. Latency Distributions

### Warm Latency Distribution (20 Iterations, Two-Tier Hierarchical Inference)

```text
Latency Percentiles (ms)
Keras 3:  [p50: 177.05] ────────────────────────── [p95: 187.81] ─── [p99: 188.58]
ONNX V3:  [p50: 16.98] ── [p95: 20.16] ── [p99: 21.48]
```

| Percentile | Keras 3 Baseline | ONNX Runtime V3 | Absolute Speedup |
|---|---|---|---|
| **Min** | 164.08 ms | 9.90 ms | 16.6x |
| **p50 (Median)** | 177.05 ms | 16.98 ms | 10.4x |
| **Mean** | 177.10 ms | 16.34 ms | 10.8x |
| **p95** | 187.81 ms | 20.16 ms | 9.3x |
| **p99** | 188.58 ms | 21.48 ms | 8.8x |
| **Max** | 188.77 ms | 21.82 ms | 8.6x |

---

## 3. Memory & Startup Latency Analysis

### 3.1 Startup Timeline Breakdown
- **Keras 3:**
  - Python import (`tensorflow`, `keras`): 3,782.03 ms
  - Model deserialization (3 `.keras` files): 3,995.85 ms
  - Total Startup: **7,777.88 ms** (~7.8 s)
- **ONNX Runtime V3:**
  - Python import (`onnxruntime`): 49.54 ms
  - Session initialization & optimization (3 `.onnx` files): 113.63 ms
  - Total Startup: **163.17 ms** (~0.16 s)

### 3.2 Memory Footprint (Resident Set Size - RSS)
- **Base Process:** ~29.3 MB
- **After Model Load:**
  - Keras 3: 345.68 MB (Delta: +316.15 MB)
  - ONNX V3: 89.22 MB (Delta: +59.93 MB)
- **After 20 Inference Invocations:**
  - Keras 3: 414.65 MB
  - ONNX V3: 338.11 MB (including temporary NumPy/Pillow raster allocations)

---

## 4. End-to-End Pipeline Stage Latency Breakdown (V3 ONNX)

| Execution Stage | Duration (ms) | Share of Request |
|---|---|---|
| **Pillow Read & Decoded Pixel Validation** | 0.82 ms | 4.8% |
| **EXIF Transposition & RGBA White Matting** | 1.15 ms | 6.7% |
| **Bilinear Resize & Float32 Tensor Format** | 0.24 ms | 1.4% |
| **Tier 1 Botanist Species ONNX Inference** | 6.85 ms | 39.8% |
| **Safety Gate 1 Evaluation (0.90 threshold)** | 0.02 ms | 0.1% |
| **Tier 2 Crop Specialist ONNX Inference** | 7.95 ms | 46.2% |
| **Safety Gate 2 Evaluation & Advisory Binding** | 0.03 ms | 0.2% |
| **JSON Response Formatting** | 0.14 ms | 0.8% |
| **Total Request Duration** | **17.20 ms** | **100.0%** |
