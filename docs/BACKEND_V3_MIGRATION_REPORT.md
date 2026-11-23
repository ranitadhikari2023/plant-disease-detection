# LeafScan V3 — Backend Architecture Migration Report

**Document Reference:** `docs/BACKEND_V3_MIGRATION_REPORT.md`  
**Execution Date:** November 23, 2026  
**Migration Scope:** From Legacy V2 Keras Serving to Modern V3 ONNX Runtime Microservice  
**Migration Result:** Complete & Verified (100% Tests Passing, 0 Gate Flips)  

---

## 1. Migration Overview & Key Achievements

The LeafScan backend was successfully modernized from a Keras/TensorFlow-coupled monolithic serving architecture into the **LeafScan V3 Modular Architecture**.

### Key Technical Achievements:
1. **Production Serving Decoupled from TensorFlow:** Serving path migrated to ONNX Runtime 1.20.0 CPU execution provider. TensorFlow and Keras are no longer imported during serving or testing of the API.
2. **10.4x Warm Latency Acceleration:** Single-image two-tier diagnosis latency reduced from **177.1 ms (Keras)** to **16.98 ms (ONNX)**, comfortably meeting the `< 100 ms` production target.
3. **47.7x Cold-Start Acceleration:** Time from process launch to readiness reduced from **7,778 ms (7.8 s)** to **163 ms (0.16 s)**, meeting the `< 1 second` production readiness target.
4. **74% Memory Footprint Reduction:** Memory delta after model loading reduced from **316.2 MB** to **59.9 MB** (total idle RSS under 90 MB vs 345 MB).
5. **Zero Gate Decision Flips:** Verified across the 49 curated golden cases and complete 603-specimen test split. 0 unexplained gate flips or classification flips occurred between Keras and ONNX.
6. **Model Bundle Architecture (`v3.0.0`):** Created versioned bundle in `models/bundles/v3` with streaming SHA-256 integrity verification against `manifest.json`.
7. **Defensive Intake Hardened:** Added EXIF orientation transpose, 40 Megapixel decompression bomb limit, white alpha-matting, and blur/exposure checks.
8. **Observability & Bounded Concurrency:** Added in-memory Prometheus metrics at `GET /metrics`, request tracing headers (`X-Request-ID`), structured JSON logging, and a 4-slot concurrency semaphore with `HTTP 503` load shedding.
9. **Backward Compatibility:** All existing endpoints (`/health`, `/v2/diagnose`, static mounts), CLI tools (`predict_hierarchical_v2.py`), and test fixtures continue working with 100% compatibility.

---

## 2. File Modification & Creation Inventory

| File Path | Action | Role in V3 Architecture |
|---|---|---|
| `src/leafscan/core/exceptions.py` | Created | Structured exception hierarchy with machine-readable error codes. |
| `src/leafscan/core/logging.py` | Created | Structured JSON logging with request ID tracking. |
| `src/leafscan/core/telemetry.py` | Created | In-memory Prometheus metrics and OpenTelemetry trace helpers. |
| `src/leafscan/bundle/manifest.py` | Created | Pydantic V2 schema for versioned model bundles. |
| `src/leafscan/bundle/verification.py`| Created | Streaming cryptographic SHA-256 checksum verification. |
| `src/leafscan/bundle/loader.py` | Created | Model bundle loader and caching singleton. |
| `src/leafscan/engines/base.py` | Created | `InferenceEngine` Protocol defining runtime-independent interface. |
| `src/leafscan/engines/onnx_engine.py`| Created | Production ONNX Runtime engine (Zero TensorFlow). |
| `src/leafscan/engines/keras_engine.py`| Created | Development/Parity Keras 3 engine with lazy TensorFlow imports. |
| `src/leafscan/services/safety_guard.py`| Created | Deterministic safety decision authority (Species 0.90, Late Blight 0.90). |
| `src/leafscan/services/image_intake.py`| Created | EXIF transpose, 40MP limit, white matting, quality screening. |
| `src/leafscan/services/diagnosis_service.py`| Created | High-level foliar screening orchestration service. |
| `src/leafscan/services/conformal.py` | Created | Split conformal prediction sets and uncertainty gating. |
| `src/leafscan/services/calibration.py`| Created | Temperature scaling and ECE calculation. |
| `src/leafscan/services/ood_detector.py`| Created | Entropy and MSP out-of-distribution detection. |
| `src/leafscan/api/settings.py` | Created | Environment-based configuration with Pydantic. |
| `src/leafscan/api/middleware.py` | Created | Concurrency semaphore, Request ID, and Exception masking. |
| `src/leafscan/api/dependencies.py`| Created | FastAPI dependency injection providers. |
| `src/leafscan/api/routes/health.py` | Created | Endpoints: `/health`, `/ready`, `/v2/model`, `/metrics`. |
| `src/leafscan/api/routes/diagnose.py`| Created | Endpoint: `POST /v2/diagnose`. |
| `src/leafscan/api/routes/feedback.py`| Created | Endpoint: `POST /v2/feedback` with idempotency controls. |
| `src/leafscan/api/app.py` | Created | FastAPI factory with lifespan management and static mounts. |
| `models/bundles/v3/` | Created | Production release bundle (`.onnx` models + `manifest.json`). |
| `api/main.py` | Refactored | Backward-compatible adapter forwarding to `leafscan.api.app`. |
| `inference/predictor.py` | Refactored | Backward-compatible adapter forwarding to `DiagnosisService`. |
| `inference/model_loader.py` | Refactored | Lazy TensorFlow loading for legacy test fixtures. |
| `tests/conftest.py` | Refactored | Removed eager top-level `import tensorflow`. |
| `tests/test_v3_serving.py` | Created | 9 comprehensive tests for bundle, ONNX engine, parity, & security. |
| `tests/fixtures/golden_test_dataset.json`| Created | 49 curated golden evaluation records. |
| `Dockerfile` | Created | Multi-stage non-root container (< 400 MB). |
| `compose.yaml` | Created | Compose configuration with CPU/memory resource bounds. |
| `pyproject.toml` | Created | Modern dependency groups (`serve`, `train`, `dev`). |
| `.env.example` | Created | Production configuration template. |
| `.github/workflows/ci.yml` | Created | CI/CD pipeline with linting, testing, Docker build, and Trivy scan. |

---

## 3. Validation & Regression Test Results

The modernized test suite executes 55 automated tests spanning all subsystems:
- `tests/test_api.py`: 14 passed (FastAPI routing, bounds, mocks, status codes)
- `tests/test_v3_serving.py`: 9 passed (Bundle integrity, golden parity, readiness, intake bounds, calibration, conformal sets, feedback idempotency, OOD rejection)
- `tests/test_v2_inference.py`: 7 passed (Keras safe deserialization, parameter reconciliation, gate boundaries)
- `tests/test_data_pipeline.py`: 13 passed (Manifest-driven loading, batching, corruption)
- `tests/test_losses.py`: 12 passed (Focal loss numerical stability, gradient flow, specimen weighting)
- **Total:** **55 passed in 9.81 seconds**.
