# LeafScan V3 — Testing & Quality Assurance Report

**Document Reference:** `docs/TESTING_AND_VALIDATION_REPORT.md`  
**Execution Date:** November 23, 2026  
**Test Harness:** Pytest 8.3.5  
**Total Tests:** 55 Passed, 0 Failed, 0 Flips  
**Test Duration:** 9.81 seconds  

---

## 1. Executive Summary

A comprehensive automated testing harness was implemented to guarantee numerical parity, deterministic safety gate authority, defensive intake sanitization, API backwards compatibility, and container readiness.

All 55 automated tests passed across the complete test matrix.

---

## 2. Test Suite Breakdown by Architectural Category

| Test Suite Module | Test Count | Passing | Focus Area |
|---|:---:|:---:|---|
| `tests/test_v3_serving.py` | 9 | 9 (100%) | Model bundle verification, SHA-256 integrity, golden parity, readiness probe, defensive intake limits, ECE calibration, conformal sets, feedback idempotency, OOD rejection. |
| `tests/test_api.py` | 14 | 14 (100%) | FastAPI endpoints (`/health`, `/v2/diagnose`, static mounts), defensive upload size limits (10MB), format filtering, mock injections, error masking. |
| `tests/test_v2_inference.py`| 7 | 7 (100%) | Keras 3 safe deserialization, parameter reconciliation (CBAM), preprocessing modes (RGB, RGBA, CMYK, L), gate boundaries (89.99% vs 90.00%), advisory safety. |
| `tests/test_data_pipeline.py`| 13 | 13 (100%) | Manifest loading, dataset batching, corrupt image handling, augmentation integration, zero leakage split validation. |
| `tests/test_losses.py` | 12 | 12 (100%) | SparseCategoricalFocalLoss numerical stability, downweighting easy examples, class weight calculations, gradient flow. |
| **Total Automated Suite** | **55** | **55 (100%)**| **Complete System Verification** |

---

## 3. Golden Evaluation Set & Parity Results

A controlled golden evaluation set of 49 curated specimens was assembled from `test_images/sample_leaf.JPG` and `audit/test_specimen_manifest.csv` spanning:
- All 6 species/condition combinations (Potato: Early Blight, Late Blight, Healthy; Tomato: Early Blight, Late Blight, Healthy).
- Confidence boundary cases (borderline species confidence and borderline disease confidence).
- Persisted to: `tests/fixtures/golden_test_dataset.json`.

### Parity Audit Summary:
- **Argmax Class Parity:** 49 / 49 (100.0%)
- **Safety Gate 1 Decision Parity:** 49 / 49 (100.0%)
- **Safety Gate 2 Decision Parity:** 49 / 49 (100.0%)
- **Late Blight Recall Parity:** 100.0% (Zero missed detections compared to Keras)
- **Unexplained Gate Flips:** **0**
- **Maximum Absolute Probability Difference:** $< 1.2 \times 10^{-5}$ (within tolerance $1 \times 10^{-4}$)

---

## 4. Safety Boundary Tests

1. **Species Confidence Gate Boundaries (Safety Gate 1):**
   - 89.99% confidence: **Abstained** (`safety_status = "species_uncertain"`, Tier 2 never executed).
   - 90.00% confidence: **Passed** (Routes to specialist).
   - 94.41% confidence: **Passed**.
2. **Late Blight High-Consequence Gate Boundaries (Safety Gate 2):**
   - Late Blight at 89.99% confidence: **Escalated** (`safety_status = "late_blight_uncertain"`, systemic suppression warnings bound).
   - Late Blight at 90.00% confidence: **Confirmed** (`safety_status = "confident"`).
   - Late Blight at 95.82% confidence: **Confirmed**.
3. **General Disease Gate Boundaries:**
   - Early Blight at 69.99% confidence: **Uncertain** (`safety_status = "disease_uncertain"`).
   - Early Blight at 70.00% confidence: **Confirmed** (`safety_status = "confident"`).

---

## 5. Security & Defensive Intake Tests

- **Payload Size Enforcement:** Tested with 10 MB + 1 byte upload -> Rejected immediately with `HTTP 413`.
- **Empty File Handling:** Upload of 0 bytes -> Rejected immediately with `HTTP 400`.
- **Corrupted Image Handling:** Corrupted byte streams -> Rejected with `HTTP 400` without stack trace disclosure.
- **Decompression Bomb Protection:** Tested with raster dimensions exceeding 40 Megapixels -> Rejected with `HTTP 400` (`image_too_large_pixels`).
- **EXIF Rotation:** Smartphone photos with orientation tags 3, 6, and 8 transposed into upright orientation.
- **White Alpha Matting:** RGBA images with transparent pixels composited over solid white, preventing transparent black artifacts.
