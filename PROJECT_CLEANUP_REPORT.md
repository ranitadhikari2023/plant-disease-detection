# Project Cleanup & Maintenance Engineering Report

**Project Name:** `plant-disease-detection`  
**Target Repository:** `https://github.com/ranitadhikari2023/plant-disease-detection.git`  
**Execution Timestamp:** November 24, 2026  
**Status:** Successfully Executed, Tested, Synchronized & Pushed  

---

## 1. Initial Project Structure

Prior to the maintenance operation, the project contained a mixture of established V2 training and inference assets, alongside newly implemented LeafScan V3 production components, two web client workspaces (`web/` and `leafscan-web/`), and various temporary build and cache artifacts:

```text
plant-disease-detection/
├── .git/
├── .github/workflows/ci.yml           [untracked]
├── .idea/                             [IDE files, git-ignored]
├── .pytest_cache/                     [Python test cache]
├── .venv/ & wsl_venv/                 [Virtual environments, git-ignored]
├── api/ (main.py, __init__.py, __pycache__/)
├── archive/ (frontend-v1/, v1/)
├── audit/ (manifests, plots, scripts)
├── dataset/ (raw images without README)
├── docs/ (V2 & V3 architecture, benchmarks, audits)
├── edge/ (OFFLINE_PWA_ARCHITECTURE.md) [untracked]
├── frontend/ (compiled static assets)
├── inference/ (predictor, model_loader, preprocessing, __pycache__/)
├── leafscan-web/                      [untracked Next.js 15 PWA with .next/, coverage/, tsconfig.tsbuildinfo]
├── mlops/ (MLOPS_PIPELINE.md)         [untracked]
├── models/ (v2/ keras checkpoints, onnx/v2/, bundles/v3/ [untracked])
├── reports/ (Phase 1 & 2 reports)
├── scripts/ (evaluate_golden_baseline.py) [untracked]
├── src/leafscan/                      [untracked V3 production serving engine]
├── test_images/ (sample_leaf.JPG)
├── tests/ (conftest.py, test_api.py, test_data_pipeline.py, test_losses.py, test_v2_inference.py, test_v3_serving.py [untracked])
├── training/ (training scripts, logs, models.py, __pycache__/)
├── web/ (React 18 SPA with node_modules/, coverage/)
├── test_c.pyd                         [588KB compiled C-extension build artifact in root]
├── Dockerfile & compose.yaml          [untracked]
├── pyproject.toml & .env.example      [untracked]
└── Root Python/shell scripts (predict_hierarchical_v2.py, export_to_onnx.py, etc.)
```

---

## 2. Final Project Structure

After comprehensive cleanup, cache elimination, dependency review, and documentation updates, the repository presents a clean, organized, production-grade layout:

```text
plant-disease-detection/
├── .github/
│   └── workflows/
│       └── ci.yml                            # GitHub Actions DevSecOps workflow
├── api/
│   ├── __init__.py                           # API package marker
│   └── main.py                               # FastAPI backward-compatible service & static mount
├── archive/                                  # Deprecated legacy prototypes
│   ├── frontend-v1/                          # Legacy Vanilla HTML5/CSS/JS frontend
│   └── v1/                                   # Legacy V1 Keras checkpoints & prototype scripts
├── audit/                                    # Dataset forensics, manifests, and charts
│   ├── artifacts/                            # Specimen distribution & training curves
│   ├── audit_summary.json                    # Audit metrics summary
│   ├── dataset_manifest.csv                  # Master manifest of 11,988 images
│   ├── specimen_manifest.csv                 # 3,971 biological leaf specimens
│   ├── test_specimen_manifest.csv            # 603 held-out test specimens (0.0% leakage)
│   ├── train_specimen_manifest.csv           # 2,776 training specimens
│   └── validation_specimen_manifest.csv      # 592 validation specimens
├── dataset/
│   └── README.md                             # Dataset documentation & specimen mapping
├── docs/                                     # Comprehensive technical documentation & audits
│   ├── analysis/                             # Backend architecture analysis
│   ├── benchmarks/                           # Baseline benchmark reports
│   ├── API_CONTRACT.md                       # Full REST API specification with invariant rules
│   ├── BACKEND_V3_ARCHITECTURE.md            # LeafScan V3 production backend architecture
│   ├── BACKEND_V3_IMPLEMENTATION_AUDIT.md    # Code audit and security verification
│   ├── BACKEND_V3_MIGRATION_REPORT.md        # Migration report from V2 Keras to V3 ONNX
│   ├── BENCHMARK_REPORT.md                   # Latency, memory, and throughput benchmarks
│   ├── DEPLOYMENT_GUIDE.md                   # Production deployment manual
│   ├── MODEL_BUNDLE_SPECIFICATION.md         # Bundle format and cryptographic verification
│   ├── ROLLBACK_GUIDE.md                     # Operational rollback procedures
│   └── SECURITY_AND_THREAT_MODEL.md          # Threat model and DevSecOps mitigations
├── edge/
│   └── OFFLINE_PWA_ARCHITECTURE.md           # Field offline caching & synchronization design
├── frontend/                                 # Vite production build (served by FastAPI at /)
│   ├── assets/                               # Compiled hashed JS, CSS, and SVG assets
│   ├── css/tokens.css                        # Botanical design tokens
│   └── index.html                            # Compiled SPA entry point
├── inference/                                # Backward-compatible V2 inference facade
│   ├── inference_config.json                 # Thresholds (90%/70%) and agronomic advisories
│   ├── model_loader.py                       # Lazy TensorFlow loader for test fixtures
│   ├── predictor.py                          # HierarchicalPredictorV2 adapter to V3 engine
│   ├── preprocessing.py                      # Defensive image preprocessing & resizing
│   └── response_formatter.py                 # Standardized JSON response formatting
├── leafscan-web/                             # Botanical Field Console (Next.js 15 PWA)
│   ├── app/                                  # App Router routes (diagnose, result, history, model)
│   ├── src/                                  # Features, stores (Zustand), API client, components
│   ├── tests/                                # 42 Vitest unit, component, and contract tests
│   ├── Dockerfile                            # Standalone container build for frontend
│   └── package.json                          # Next.js 15, React 19, Tailwind CSS v4, Serwist
├── mlops/
│   └── MLOPS_PIPELINE.md                     # Model tracking, CI/CD, and deployment lifecycle
├── models/
│   ├── bundles/v3/                           # Production immutable model bundle
│   │   ├── manifest.json                     # Cryptographic SHA-256 specification
│   │   ├── potato_expert.onnx                # Tier 2 Potato Specialist with CBAM (10.4 MB)
│   │   ├── species_expert.onnx               # Tier 1 Botanist Species Expert (9.2 MB)
│   │   └── tomato_expert.onnx                # Tier 2 Tomato Specialist with CBAM (10.4 MB)
│   ├── onnx/v2/                              # Exported ONNX graphs from export_to_onnx.py
│   └── v2/                                   # Trained Keras 3 checkpoints (.keras format)
├── reports/                                  # Engineering milestone reports
│   ├── END_TO_END_PROJECT_ANALYSIS_REPORT.md
│   ├── PHASE_1_ENVIRONMENT_AND_DEPLOYMENT_HARDENING.md
│   └── PHASE_2_PROJECT_CLEANUP_AND_ORGANIZATION.md
├── scripts/
│   └── evaluate_golden_baseline.py           # Evaluates baseline and curates golden dataset
├── src/leafscan/                             # LeafScan V3 Modular Production Engine
│   ├── api/                                  # FastAPI app factory, routes, middleware, settings
│   ├── bundle/                               # Bundle manifest schemas, loader, and SHA-256 verifier
│   ├── core/                                 # Structured exceptions, logging, Prometheus metrics
│   ├── engines/                              # InferenceEngine Protocol, OnnxEngine, KerasEngine
│   └── services/                             # ImageIntake, SafetyGuard, OODDetector, Conformal
├── tests/                                    # Pytest test suite (55 tests) & test fixtures
│   ├── fixtures/
│   │   └── golden_test_dataset.json          # 49 curated golden evaluation records
│   ├── conftest.py                           # Pytest fixtures and mock predictors
│   ├── test_api.py                           # 14 tests: REST API endpoints and static routing
│   ├── test_data_pipeline.py                 # 13 tests: Data loading, augmentation, error handling
│   ├── test_losses.py                        # 12 tests: Focal loss math and class weighting
│   ├── test_v2_inference.py                  # 7 tests: V2 inference, CBAM, gate boundaries
│   └── test_v3_serving.py                    # 9 tests: V3 bundle integrity, parity, security
├── test_images/
│   └── sample_leaf.JPG                       # Verified test specimen image
├── training/                                 # Training pipelines and evaluation scripts
│   ├── logs/                                 # Training convergence CSV logs
│   ├── augmentation.py                       # Online image augmentation layers
│   ├── class_mappings.json                   # Class label index mappings
│   ├── data_pipeline.py                      # Manifest-driven tf.data pipeline
│   ├── evaluate_v2_system.py                 # Full 603-specimen test evaluation harness
│   ├── losses.py                             # SparseCategoricalFocalLoss implementation
│   ├── models.py                             # MobileNetV2 + CBAM custom layer construction
│   ├── run_preflight_and_smoke_tests.py      # Architecture and environment verification
│   ├── train_utils.py                        # Two-phase transfer learning loop
│   ├── training_config.json                  # Training hyperparameters
│   └── verify_v2_artifacts.py                # Artifact verification harness
├── web/                                      # React 18 / Vite 5 SPA source
│   ├── src/                                  # Modular components, hooks, API client, routes
│   ├── package.json                          # Dependencies and scripts (67 Vitest tests)
│   └── vite.config.ts                        # Bundler configuration (builds to frontend/)
├── .env.example                              # Environment configuration template
├── .gitignore                                # Comprehensive Git ignore rules
├── compose.override.yaml                     # Full-stack Compose override (Web + API)
├── compose.yaml                              # Production Docker Compose specification
├── Dockerfile                                # Multi-stage, non-root production container (<400MB)
├── export_to_onnx.py                         # ONNX export and parity validation script
├── predict_hierarchical_v2.py                # Command-line inference entry point
├── pyproject.toml                            # Standard Python project manifest (Hatchling)
├── pytest.ini                                # Pytest configuration and markers
├── requirements.txt                          # Production Python dependencies
├── requirements-dev.txt                      # Development and testing Python dependencies
└── setup_wsl_env.sh                          # Automated WSL environment setup script
```

---

## 3. Files Deleted

| File Path | Size | Reason for Deletion |
|---|---|---|
| `test_c.pyd` | 588,800 bytes | Temporary compiled C-extension build artifact in root; unreferenced across the codebase. |
| `leafscan-web/tsconfig.tsbuildinfo` | ~55,000 bytes | Ephemeral TypeScript compiler build cache file. |
| Multiple `*.pyc` files | ~75,000 bytes | Compiled Python bytecode cache files across `api/`, `inference/`, `training/`, `tests/`, and `src/leafscan/`. |

---

## 4. Directories Deleted

| Directory Path | Approximate Contents | Reason for Deletion |
|---|---|---|
| `leafscan-web/.next/` | ~25 MB / ~300 files | Next.js generated build cache and static packs; should not be committed to Git. |
| `leafscan-web/coverage/` | ~35 files | Temporary Vitest code coverage output directory. |
| `web/coverage/` | ~35 files | Temporary Vitest code coverage output directory. |
| `.pytest_cache/` | ~10 files | Pytest internal step cache. |
| `api/__pycache__/` | 4 `.pyc` files | Python bytecode cache. |
| `inference/__pycache__/` | 6 `.pyc` files | Python bytecode cache. |
| `training/__pycache__/` | 6 `.pyc` files | Python bytecode cache. |
| `tests/__pycache__/` | 6 `.pyc` files | Python bytecode cache. |
| `src/leafscan/**/__pycache__/` | 24 `.pyc` files | Python bytecode cache across V3 submodules. |

---

## 5. Files Retained and Justification

- **Trained Model Weights (`models/v2/*.keras`, `archive/v1/models/*.keras`):** Critical trained neural network weights for MobileNetV2 Botanist and CBAM specialists. Retraining would alter production inference behavior.
- **Production ONNX Model Bundle (`models/bundles/v3/`):** Contains `manifest.json`, `species_expert.onnx`, `potato_expert.onnx`, and `tomato_expert.onnx`. Essential for the zero-TensorFlow serving runtime, container builds, and CI verification.
- **Dataset Audit Manifests (`audit/*.csv`):** Master and specimen manifests defining the 3,971 biological leaf clusters and clean training/validation/test splits with 0.0% data leakage.
- **Source Code (`src/leafscan/`, `api/`, `inference/`, `training/`, `web/`, `leafscan-web/`):** Complete production backend, backward-compatible facades, and modern frontend clients.
- **Test Image (`test_images/sample_leaf.JPG`):** Essential sample specimen utilized in unit tests, integration smoke tests, and CLI demonstrations.

---

## 6. Files Modified or Relocated

| File Path | Nature of Modification |
|---|---|
| `.gitignore` | Added rules to ignore `.next/`, `*.tsbuildinfo`, `dist/`, `.storybook-static/`, and `*.pyd`. Unignored `!models/bundles/v3/*.onnx` to ensure the immutable production bundle is tracked. |
| `README.md` | Complete rewrite into a 15-section, technically accurate, professional project guide covering V3 ONNX serving, model bundles, safety gates, Docker deployment, and testing. |
| `api/main.py` | Refactored to delegate to `src/leafscan` while preserving exact V2 endpoint schemas; updated deprecated Starlette `HTTP_413_REQUEST_ENTITY_TOO_LARGE` to non-deprecated status. |
| `src/leafscan/api/routes/diagnose.py` | Updated `HTTP_413_REQUEST_ENTITY_TOO_LARGE` to non-deprecated `getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)`. |
| `leafscan-web/vitest.config.ts` | Configured `exclude: ["**/node_modules/**", "**/dist/**", "tests/e2e/**", "**/*.spec.ts"]` so Vitest runs unit and contract tests without attempting to run Playwright specs. |
| `dataset/README.md` | Created comprehensive documentation for the dataset directory, explaining Solanaceous crops, classes, and group manifests. |
| `docs/API_CONTRACT.md` | Updated with V3 REST endpoints (`/health`, `/ready`, `/v2/model`, `/metrics`, `/v2/diagnose`, `/v2/feedback`), invariant contract rules, and error codes. |
| `inference/model_loader.py` & `predictor.py` | Updated adapters to lazily import TensorFlow and seamlessly forward to `DiagnosisService`. |
| `pytest.ini` & `tests/conftest.py` | Added `security` marker and lazy model cache cleanup to avoid eager TensorFlow initialization. |

---

## 7. Dependency and Configuration Changes

- **Git Configuration (`.gitignore`):**
  - Added `.next/`, `*.tsbuildinfo`, `dist/`, `.storybook-static/`, `*.pyd`.
  - Added exception `!models/bundles/v3/*.onnx` to guarantee deployment model bundles are versioned with their cryptographic manifest.
- **Python Project Manifest (`pyproject.toml`):**
  - Validated standard PEP 621 packaging with `hatchling` build backend and optional dependency groups: `serve` (zero-TF ONNX), `train` (TensorFlow/Keras), and `dev`.
- **Environment Template (`.env.example`):**
  - Validated safe template containing zero hardcoded secrets or credentials.

---

## 8. README Improvements

The project documentation was thoroughly updated to include all 15 required sections:
1. **Project Overview:** Clear agricultural context, economic impact of *Alternaria solani* and *Phytophthora infestans*, and the two-tier architectural solution.
2. **Key Features:** Full inventory of both V2 and V3 capabilities, CBAM attention, deterministic safety gates, and defensive intake.
3. **System Architecture:** Detailed Mermaid diagram mapping Transport, Service, Engine, and Model Bundle layers.
4. **Project Directory Structure:** Accurate, comprehensive directory tree matching the cleaned codebase.
5. **Technologies Used:** Pinned versions and roles for Python 3.10/3.12, ONNX Runtime 1.20, FastAPI 0.143, Docker, React 18, and Next.js 15.
6. **Machine Learning Model:** Two-tier cascade, CBAM submodule parameter counts (206,258 params), model bundle specifications, supported species, and empirical test metrics.
7. **Installation and Setup:** Step-by-step instructions for Windows PowerShell, Linux/WSL, and Docker.
8. **Environment Configuration:** Complete table of `LEAFSCAN_*` configuration settings with safe defaults.
9. **Running the Application:** Production FastAPI web mode, CLI mode, and frontend dev server instructions.
10. **Docker Deployment:** Standalone container build and full-stack Docker Compose instructions.
11. **API Documentation:** Comprehensive documentation of `/health`, `/ready`, `/v2/model`, `/metrics`, `/v2/diagnose`, and `/v2/feedback`.
12. **Testing:** Complete test harness table and execution commands for Pytest, Vitest (`web/`), and Vitest (`leafscan-web/`).
13. **Troubleshooting:** Solutions for PowerShell execution policy, port conflicts, model checksum mismatches, and image format rejections.
14. **Future Improvements:** Realistic roadmap items (INT8 edge quantization, multilingual advisories).
15. **License and Project Information:** Project metadata, author contact, and GitHub repository links.

---

## 9. Tests Executed and Actual Results

| Test Harness | Scope / Module | Tests Executed | Passed | Failed | Execution Time |
|---|---|:---:|:---:|:---:|---:|
| **Pytest** | `tests/test_api.py` (FastAPI REST endpoints) | 14 | 14 | 0 | 0.44s |
| **Pytest** | `tests/test_data_pipeline.py` (tf.data pipeline) | 13 | 13 | 0 | 2.10s |
| **Pytest** | `tests/test_losses.py` (Focal loss & class weights) | 12 | 12 | 0 | 1.85s |
| **Pytest** | `tests/test_v2_inference.py` (Keras CBAM & gates) | 7 | 7 | 0 | 3.20s |
| **Pytest** | `tests/test_v3_serving.py` (Bundle, ONNX, parity, security) | 9 | 9 | 0 | 2.68s |
| **Vitest (`web/`)** | React 18 SPA components, hooks, API, utils | 67 | 67 | 0 | 16.42s |
| **Vitest (`leafscan-web/`)** | Next.js 15 PWA unit, component, contract tests | 42 | 42 | 0 | 4.08s |
| **CLI Smoke Test** | `predict_hierarchical_v2.py --json` | 1 | 1 | 0 | 0.45s |
| **Total Automated Tests** | **Full System Verification** | **165** | **165** | **0** | **100% Passing** |

---

## 10. Docker Verification Results

- **Dockerfile Analysis:** Multi-stage build (`python:3.12-slim`), unprivileged user `leafscan` (UID 10001), isolated virtual environment with `uv`, non-root execution, readiness health check via `GET /ready`, entrypoint `uvicorn leafscan.api.app:app`. Image size budget `< 400 MB`.
- **Docker Compose:** `compose.yaml` (API service with health checks, tmpfs `/tmp`, memory/CPU reservations) and `compose.override.yaml` (LeafScan Web PWA on port 3000) verified for valid YAML schema syntax and correct context mappings.

---

## 11. Issues Discovered and Fixes Applied

1. **Issue:** `leafscan-web` Vitest suite failed with `Playwright Test did not expect test.describe() to be called here` when Vitest picked up `tests/e2e/e2e.spec.ts`.  
   **Fix:** Updated `leafscan-web/vitest.config.ts` to exclude `tests/e2e/**` and `**/*.spec.ts`. Vitest now runs the 42 unit, component, and contract tests cleanly.
2. **Issue:** Starlette deprecation warning `StarletteDeprecationWarning: 'HTTP_413_REQUEST_ENTITY_TOO_LARGE' is deprecated. Use 'HTTP_413_CONTENT_TOO_LARGE' instead`.  
   **Fix:** Updated `api/main.py` and `src/leafscan/api/routes/diagnose.py` to use `getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)`.
3. **Issue:** Next.js build cache (`.next/`) and TypeScript compiler cache (`tsconfig.tsbuildinfo`) were untracked and accumulating disk space.  
   **Fix:** Removed the cache directories and added `.next/` and `*.tsbuildinfo` to `.gitignore`.
4. **Issue:** An orphaned 588 KB compiled binary (`test_c.pyd`) was residing in the project root.  
   **Fix:** Confirmed zero references across all project code and safely removed it.
5. **Issue:** Missing documentation in `dataset/`.  
   **Fix:** Created `dataset/README.md` explaining species, pathological classes, and referencing the canonical manifests in `audit/`.

---

## 12. Git Commit Hash and Commit Message

- **Commit Hash:** `3b965e0` (Full: `3b965e043975cc1203a89d13da606e672b1267e6`)
- **Branch:** `main`
- **Commit Message:**
  ```text
  feat(v3): clean project structure, integrate LeafScan V3 production engine, and update documentation
  ```
- **Files Committed:** 129 files changed, 22,262 insertions(+), 1,170 deletions(-)

---

## 13. GitHub Push Status

- **Remote Target:** `https://github.com/ranitadhikari2023/plant-disease-detection.git`
- **Pushed Reference:** `refs/heads/main`
- **Push Result:**
  ```text
  To https://github.com/ranitadhikari2023/plant-disease-detection.git
     849b343..3b965e0  main -> main
  ```
- **Remote Verification:** `git ls-remote origin refs/heads/main` confirmed:
  ```text
  3b965e043975cc1203a89d13da606e672b1267e6	refs/heads/main
  ```
- **Working Tree State:** Clean (`nothing to commit, working tree clean`).

---

## 14. Remaining Recommendations

1. **Docker Desktop Execution:** If deploying locally via Docker, ensure the Docker Desktop service is running before executing `docker compose up`.
2. **Edge Quantization (Optional Future Work):** To achieve sub-5ms inference on constrained micro-controllers or Raspberry Pi agricultural edge sensors, consider adding an INT8 quantization export pass to `export_to_onnx.py`.
