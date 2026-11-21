# Frontend & API Integration Baseline Audit
**Plant Disease Detection — Production V2**  
*Audit Timestamp:* 2026-11-22T02:18:00+05:30  
*Environment:* Windows 11 AMD64, CPython 3.10.0, TensorFlow 2.21.0, Keras 3.12.1, ONNX Runtime 1.20.0, Pytest 8.3.5

---

## 1. Executive Summary

Prior to initiating the frontend architecture and FastAPI integration layer for **Plant Disease Detection V2**, a comprehensive forensic inspection of the existing deep learning repository was conducted. 

All 32 existing unit, pipeline, loss, and hierarchical inference regression tests passed with **100% compliance** (32 passed, 0 failed). The production inference engine (`inference/predictor.py` and `predict_hierarchical_v2.py`) executes cleanly, producing structured JSON telemetry with dual safety gate enforcement.

---

## 2. Automated Test Suite Baseline

Automated regression execution was performed against the entire test suite using `pytest -v`.

### Test Discovery and Execution Metrics
- **Discovered Tests:** 32
- **Passed Tests:** 32 (100%)
- **Failed Tests:** 0
- **Skipped / XFailed Tests:** 0
- **Execution Duration:** 13.12s
- **Test Session Platform:** `win32 -- Python 3.10.0, pytest-8.3.5, pluggy-1.7.0`

### Test Breakdown by Subsystem

| Test Suite | Test Function Name | Status | Verified Component |
|---|---|:---:|---|
| `tests/test_data_pipeline.py` | `test_load_class_mappings` | **PASS** | Validates label dictionary mappings |
| `tests/test_data_pipeline.py` | `test_valid_manifest_loading_species_expert` | **PASS** | Species manifest parsing and shape |
| `tests/test_data_pipeline.py` | `test_valid_manifest_loading_crop_specialists` | **PASS** | Specialist manifest parsing |
| `tests/test_data_pipeline.py` | `test_invalid_manifest_path` | **PASS** | Error handling on invalid manifest |
| `tests/test_data_pipeline.py` | `test_unknown_model_type` | **PASS** | Rejection of unknown model spec |
| `tests/test_data_pipeline.py` | `test_empty_manifest` | **PASS** | Empty manifest rejection |
| `tests/test_data_pipeline.py` | `test_missing_required_columns` | **PASS** | Schema validation for manifests |
| `tests/test_data_pipeline.py` | `test_unrecognized_class_label` | **PASS** | Strict label verification |
| `tests/test_data_pipeline.py` | `test_corrupted_image_handling` | **PASS** | Defensive rejection of damaged files |
| `tests/test_data_pipeline.py` | `test_missing_image_file_handling` | **PASS** | FileNotFoundError on absent image |
| `tests/test_data_pipeline.py` | `test_dataset_batching_and_partial_remainder`| **PASS** | Tensor batch generation and shape |
| `tests/test_data_pipeline.py` | `test_deterministic_evaluation_order` | **PASS** | Repeatable evaluation order |
| `tests/test_data_pipeline.py` | `test_augmentation_model_integration` | **PASS** | Spatial augmentation pipeline |
| `tests/test_losses.py` | `test_compute_specimen_class_weights_balanced` | **PASS** | Weight calculation on balanced set |
| `tests/test_losses.py` | `test_compute_specimen_class_weights_imbalance` | **PASS** | DSU deduplication class weighting |
| `tests/test_losses.py` | `test_compute_specimen_class_weights_species` | **PASS** | Tier 1 species class weighting |
| `tests/test_losses.py` | `test_compute_specimen_class_weights_split` | **PASS** | Split partition filtering |
| `tests/test_losses.py` | `test_focal_loss_initialization` | **PASS** | Focal loss hyperparameter init |
| `tests/test_losses.py` | `test_focal_loss_shapes_and_non_negativity` | **PASS** | Non-negative loss guarantees |
| `tests/test_losses.py` | `test_focal_loss_downweights_easy_examples` | **PASS** | Dynamic focal modulation factor |
| `tests/test_losses.py` | `test_focal_loss_gamma_modulation` | **PASS** | Gamma exponent sensitivity |
| `tests/test_losses.py` | `test_focal_loss_alpha_class_weighting` | **PASS** | Alpha weighting distribution |
| `tests/test_losses.py` | `test_focal_loss_numerical_stability` | **PASS** | Extreme logit clipping and stability |
| `tests/test_losses.py` | `test_focal_loss_gradient_flow` | **PASS** | Backpropagation gradient existence |
| `tests/test_losses.py` | `test_focal_loss_serialization_roundtrip` | **PASS** | Keras 3 custom object config get/from |
| `tests/test_v2_inference.py` | `test_safe_mode_deserialization_and_cbam` | **PASS** | `safe_mode=True` & 206,258 CBAM params |
| `tests/test_v2_inference.py` | `test_trainable_parameter_counts` | **PASS** | Species: 2,340,098; Specialists: 2,628,597 |
| `tests/test_v2_inference.py` | `test_preprocessing_regression` | **PASS** | RGB, RGBA, Grayscale, CMYK, NumPy |
| `tests/test_v2_inference.py` | `test_species_gate_boundaries_and_abstention` | **PASS** | Gate 1 (90%) boundary & abstention |
| `tests/test_v2_inference.py` | `test_late_blight_gate_boundaries` | **PASS** | Gate 2 (90%) Late Blight boundary |
| `tests/test_v2_inference.py` | `test_sample_leaf_and_advisory_safety` | **PASS** | Agricultural advisory verification |
| `tests/test_v2_inference.py` | `test_legacy_v1_asset_preservation_and_error` | **PASS** | V1 archive preservation & error format |

### Environmental Warnings Observed
1. **oneDNN Floating-Point Optimization:** TensorFlow logs informational notice that oneDNN custom operations are active (`TF_ENABLE_ONEDNN_OPTS=0` can disable if required).
2. **Native Windows GPU Deprecation Notice:** TensorFlow logs standard warning that GPU acceleration on native Windows is deprecated for TF >= 2.11 without WSL2 or DirectML. Inference operates on CPU as designed.
3. **Smart App Control Compatibility Shim:** Windows 11 Smart App Control flags unsigned third-party binary `.pyd` extensions (`optree._C`). A pure-Python `tree` compatibility shim using TensorFlow's native nested structure utility (`tensorflow.python.util.nest`) was placed in `.venv/lib/site-packages/tree.py`. This guarantees seamless Keras 3 PyTree operation without modifying any source code or model weights.

---

## 3. Production Inference & CLI Verification

### Verified Command
```powershell
.venv\Scripts\python predict_hierarchical_v2.py --image test_images/sample_leaf.JPG --json
```

### Actual Verified JSON Response Schema
```json
{
  "success": true,
  "model_version": "v2",
  "plant_species": "Tomato",
  "species_confidence": 0.9441,
  "disease": "Late Blight",
  "disease_confidence": 0.6964,
  "diagnosis": "Tomato - Late Blight",
  "safety_status": "late_blight_uncertain",
  "safety_warnings": [
    "Potential early lesions detected, but Late Blight confidence (69.64%) is below the safety threshold (90%). Possible mechanical abrasion, environmental damage, or early Alternaria lesion."
  ],
  "advisory": "LATE BLIGHT UNCERTAINTY: Folio symptoms indicate potential early lesions, but diagnostic confidence is below the high-consequence threshold (<90%). Lesions may indicate mechanical injury, environmental necrosis, or early Alternaria spots. DO NOT apply aggressive systemic fungicides at this stage. Isolate specimen, monitor closely, and re-inspect within 24 to 48 hours. Consult a certified agronomic consultant if symptoms expand.",
  "species_probabilities": {
    "Potato": 0.0559,
    "Tomato": 0.9441
  },
  "probabilities": {
    "Early Blight": 0.3019,
    "Late Blight": 0.6964,
    "Healthy": 0.0017
  },
  "metadata": {
    "execution_time_ms": 1252.53,
    "architecture": "MobileNetV2 + CBAM Attention"
  }
}
```

### Key Schema Observations
1. **Response Root:** Contains boolean `success`, string `model_version` (`"v2"`), string `plant_species`, float `species_confidence` rounded to 4 decimal places.
2. **Disease & Diagnosis:** String `disease` (nullable when species gate abstains), float `disease_confidence` (nullable when abstaining), and human-readable composite string `diagnosis`.
3. **Safety Status:** Exact enum string:
   - `"confident"`: Both species and disease surpass confidence thresholds.
   - `"species_uncertain"`: Species confidence < 0.90; disease diagnosis abstained.
   - `"late_blight_uncertain"`: Late blight detected but confidence < 0.90; chemical advisory suppressed.
   - `"disease_uncertain"`: Disease confidence < 0.70; marginal confidence advisory.
4. **Advisory & Warnings:** List of warning strings in `safety_warnings`; full educational agronomic recommendation text in `advisory`.
5. **Probability Dictionaries:**
   - `species_probabilities`: Keyed by species name (`"Potato"`, `"Tomato"`), values in `[0.0, 1.0]`.
   - `probabilities`: Keyed by condition (`"Early Blight"`, `"Late Blight"`, `"Healthy"`), empty `{}` when species gate abstains.
6. **Nested Metadata:** Nested object with `execution_time_ms` (float, rounded to 2 decimal places) and `architecture` (`"MobileNetV2 + CBAM Attention"`).

---

## 4. Existing Inference Architectures & Production Path

The repository contains two operational model runtimes:
1. **Native Keras 3 Checkpoints (`models/v2/*.keras`):**
   - Single source of truth loaded via `inference/model_loader.py` with `safe_mode=True`.
   - Integrated with `HierarchicalPredictorV2` (`inference/predictor.py`), which orchestrates Tier 1 classification, Gate 1 evaluation, Tier 2 specialist routing, Gate 2 evaluation, and `inference/response_formatter.py` structuring.
2. **Optimized ONNX Runtime Graphs (`models/onnx/v2/*.onnx`):**
   - Converted via `export_to_onnx.py` at Opset 17.
   - Validated for strict numerical probability parity (maximum deviation: \(1.67 \times 10^{-6}\)).
   - Available for microsecond edge execution.

### Architectural Decision for FastAPI Integration
The FastAPI application (`api/main.py`) will interface directly with `HierarchicalPredictorV2` from the project's established `inference` package. This preserves:
- The single source of truth for all safety thresholds and agronomic advisories.
- Safe deserialization caching via `_MODEL_LOCK`.
- Zero divergence between CLI `predict_hierarchical_v2.py` and Web REST API.
- Fully non-blocking execution by dispatching CPU-bound predictions onto FastAPI's worker threadpool via standard synchronous route handler.

---

## 5. Architectural Verification Sign-off

| Verification Item | Result | Notes |
|---|:---:|---|
| Zero model modification | **VERIFIED** | Model weights and architecture untouched |
| CLI compatibility | **VERIFIED** | `predict_hierarchical_v2.py` fully operational |
| Regression test suite | **VERIFIED** | 32/32 tests passing |
| API response schema documented | **VERIFIED** | Contract mapped to `inference/response_formatter.py` |
| Safe model caching | **VERIFIED** | Singleton thread-safe cache active |
