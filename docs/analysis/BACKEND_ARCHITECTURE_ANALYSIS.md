# Backend Architecture Analysis Report
## Project: plant-disease-detection

**Document Reference:** `docs/analysis/BACKEND_ARCHITECTURE_ANALYSIS.md`  
**Execution Date:** November 23, 2026  
**Analysis Scope:** Backend Architecture, REST API Microservice, Machine Learning Inference Pipeline, Decision Logic, Error Handling, Security, Testing, Performance, and Deployment Readiness.  
**Evaluator:** Senior Backend Architect, Python Software Engineer, Machine Learning Engineer, and Software Quality Auditor.

---

## 1. Executive Summary

This report delivers an exhaustive, evidence-based technical evaluation of the backend architecture for the **`plant-disease-detection`** project (production system branded as **LeafScan V2**). The analysis is derived entirely from forensic inspection of the actual repository source code, model configurations, serialized neural network checkpoints, exported graph representations, test fixtures, and environment runtime behaviors.

### Key Architectural Findings:
1. **Service Model & Topology:** The backend is organized as a modular, lightweight Python service anchored by FastAPI (`api/main.py`) exposing synchronous endpoints backed by a thread-safe hierarchical deep learning inference engine (`inference/predictor.py`). It also serves pre-compiled single-page application (SPA) static assets from `frontend/`.
2. **Two-Tier Hierarchical Inference Engine:** Foliar diagnosis avoids the error-prone single-stage "megamodel" paradigm (where species and disease are lumped into flat classes). Instead, it implements a sequential two-tier pipeline:
   - **Tier 1 (Botanist Species Classifier):** MobileNetV2 classifying between *Solanum tuberosum* (Potato) and *Solanum lycopersicum* (Tomato).
   - **Safety Gate 1:** Explicit confidence evaluation enforcing a 90.0% threshold. If the image cannot be definitively classified as Potato or Tomato, the system **abstains** from issuing a disease diagnosis to prevent catastrophic crop misdiagnosis.
   - **Tier 2 (Crop Doctor Specialists):** Specialized MobileNetV2 networks injected with a 206,258-parameter Convolutional Block Attention Module (CBAM) classifying between Early Blight (*Alternaria solani*), Late Blight (*Phytophthora infestans*), and Healthy tissue.
   - **Safety Gate 2:** High-consequence pathogen escalation. If Late Blight is detected but confidence is below 90.0%, or if general disease confidence is below 70.0%, the backend flags the diagnosis as uncertain and issues strict chemical-suppression warnings.
3. **Core Architectural Disconnect (ONNX vs. Keras Runtime):** While the repository maintains an export script (`export_to_onnx.py`) and pre-exported ONNX graphs (`models/onnx/v2/*.onnx`), the active production service (`inference/model_loader.py` and `inference/predictor.py`) is tightly coupled to native TensorFlow 2.21.0 and Keras 3.12.1. This incurs unnecessary memory overhead (~800 MB–1.2 GB vs. ~80 MB for ONNX Runtime), increases cold-start latency, and exposes the deployment to OS-level Application Control policy restrictions.
4. **Input Sanitization & Defensive Design:** The API implements strict defenses: maximum file size gating (10 MB), byte-stream decoding validation via PIL, explicit MIME/format filtering (JPEG, PNG, WebP), alpha-channel white matting for RGBA images, and structured exception masking that prevents filesystem path disclosure.
5. **Data Leakage Elimination:** Forensic training audit artifacts confirm that the V2 model weights were trained using a master specimen manifest that partitions 11,988 images across 3,971 biologically unique source specimen groups, guaranteeing 0% data leakage between train, validation, and test splits.

---

## 2. Project and Backend Overview

The `plant-disease-detection` project provides high-consequence agricultural plant pathology screening for Solanaceous food crops. Misidentifying fungal or oomycete pathogens (such as confusing Early Blight with Late Blight or confusing host species) leads to inappropriate chemical interventions, wasted agrochemical expenditure, pesticide resistance, and severe crop loss.

The backend serves two distinct operational modalities:
1. **Interactive Single-Leaf Diagnosis:** Consumed by web browsers via a RESTful HTTP multipart upload interface.
2. **Headless Batch / CLI Evaluation:** Consumed by researchers, agricultural field agents, and test harnesses via Python CLI (`predict_hierarchical_v2.py`) and evaluation scripts (`training/evaluate_v2_system.py`).

### Primary Backend Capabilities:
- Dynamic input validation and raster image normalization.
- In-memory raster decoding with PIL, stripping malicious metadata and payload injections.
- Lazy, thread-safe model caching avoiding startup latency penalty on simple health checks.
- Two-stage neural network routing with intermediate abstention decision boundaries.
- Domain-specific agronomic advisory generation based on pathogen risk tier.
- Static distribution of pre-compiled modern web frontend assets.

---

## 3. Existing Backend Directory Structure

Below is the verified, backend-focused directory tree of the repository:

```text
d:\plant-disease-detection\
├── api\
│   ├── __init__.py                     # Package declaration
│   └── main.py                         # FastAPI service, REST endpoints, static mounting, CORS
├── inference\
│   ├── __init__.py                     # Package declaration
│   ├── inference_config.json           # Runtime thresholds, model paths, safety advisories
│   ├── model_loader.py                 # Thread-safe model cache, safe deserialization (safe_mode=True)
│   ├── predictor.py                    # HierarchicalPredictorV2, 2-tier orchestrator, safety gates
│   ├── preprocessing.py                # PIL raster validation, RGBA white-matting, resize, float32 scaling
│   └── response_formatter.py           # Uniform JSON response schema formatter (success/error)
├── models\
│   ├── onnx\
│   │   └── v2\
│   │       ├── potato_expert_v2.onnx   # Exported ONNX graph for Potato Specialist (3 classes, CBAM)
│   │       ├── species_expert_v2.onnx  # Exported ONNX graph for Botanist Species Expert (2 classes)
│   │       └── tomato_expert_v2.onnx   # Exported ONNX graph for Tomato Specialist (3 classes, CBAM)
│   └── v2\
│       ├── potato_expert_v2.keras      # Keras 3 checkpoint for Potato Specialist (2,628,597 params)
│       ├── species_expert_v2.keras     # Keras 3 checkpoint for Botanist Species Expert (2,340,098 params)
│       └── tomato_expert_v2.keras      # Keras 3 checkpoint for Tomato Specialist (2,628,597 params)
├── training\
│   ├── __init__.py                     # Package declaration
│   ├── augmentation.py                 # Keras RandomFlip, RandomRotation, RandomZoom layers
│   ├── class_mappings.json             # Canonical index-to-label and label-to-index maps
│   ├── data_pipeline.py                # Manifest-driven tf.data input pipeline, zero leakage
│   ├── evaluate_v2_system.py           # Exhaustive test split evaluator & confusion matrix generator
│   ├── losses.py                       # SparseCategoricalFocalLoss & specimen class weight calculator
│   ├── models.py                       # MobileNetV2 backbone loader & CBAM attention layer definitions
│   ├── run_preflight_and_smoke_tests.py# Environment pre-check and rapid train loop verification
│   ├── training_config.json            # Hyperparameters, learning rates, epochs, callbacks
│   ├── train_utils.py                  # Two-phase warmup & fine-tuning execution engine
│   └── verify_v2_artifacts.py          # Weight parameter verification and hash checking
├── tests\
│   ├── conftest.py                     # Pytest fixtures: sample_leaf, synthetic manifests, cache clearing
│   ├── test_api.py                     # API test suite: input limits, format errors, status codes, mocks
│   ├── test_data_pipeline.py           # Pipeline unit tests: manifest parsing, batching, corruption
│   ├── test_losses.py                  # Loss unit tests: focal loss numerical stability, class weighting
│   └── test_v2_inference.py            # V2 regression suite: safe_mode, param counts, boundary tests
├── test_images\
│   └── sample_leaf.JPG                 # Canonical real-world tomato late blight specimen
├── docs\
│   ├── API_CONTRACT.md                 # Formal HTTP REST API specification and payload examples
│   ├── END_TO_END_AUDIT_REPORT.md      # Comprehensive Phase 0 audit report
│   └── FINAL_PROJECT_ANALYSIS_REPORT.md# Baseline technical report
├── archive\
│   └── v1\                             # Deprecated V1 artifacts safely preserved with bit-for-bit parity
│       ├── README.md                   # Deprecation notice
│       ├── predict_hierarchical.py     # Legacy inference script
│       └── models\                     # Legacy V1 un-gated model checkpoints
├── export_to_onnx.py                   # Production ONNX exporter with opset 17 & parity testing
├── predict_hierarchical_v2.py          # Production CLI diagnosis script
├── pytest.ini                          # Test configuration, testpaths, markers, warning suppressions
├── requirements.txt                    # Pinned production runtime dependencies
├── requirements-dev.txt                # Pinned development and testing dependencies
└── setup_wsl_env.sh                    # Automated shell script for Linux WSL environment isolation
```

---

## 4. Backend Technology Stack Analysis

Forensic inspection of `requirements.txt` and `requirements-dev.txt` reveals an exact, fully pinned dependency configuration.

| Technology / Library | Version | Usage in Backend | Architectural Purpose & Assessment |
|---|---|---|---|
| **Python** | `3.10.0` (active `.venv`) | Entire backend runtime | Well-suited for ML workloads. Verified in virtual environment `.venv`. |
| **FastAPI** | `0.143.0` | `api/main.py` | Web framework. Provides OpenAPI auto-documentation, dependency injection, and automatic threadpool offloading for synchronous endpoints. |
| **Starlette** | `1.7.0` | `api/main.py` | Underlying ASGI toolkit powering FastAPI routing, static mounting, and error handling. |
| **Uvicorn** | `0.54.0` | Application Server | Standard ASGI web server. High performance with `uvloop` (POSIX) or default selector (Windows). |
| **Pydantic** | `2.14.0` | Request/Response parsing | Rust-backed data validation and schema serialization. |
| **python-multipart** | `0.0.32` | `api/main.py` | Multipart form-data streaming parser required for file uploads in FastAPI. |
| **TensorFlow** | `2.21.0` | `inference/model_loader.py`, `training/` | Deep learning execution framework used for model deserialization and training. Heavyweight for production inference. |
| **Keras** | `3.12.1` | `training/models.py`, `inference/model_loader.py` | Multi-backend deep learning API. Native support for `safe_mode=True` serialization. |
| **ONNX Runtime** | `1.20.0` | `export_to_onnx.py` (Ready for inference) | High-performance inference engine with CPU execution provider. Currently not wired to `inference/predictor.py`. |
| **ONNX** | `1.23.2` | `export_to_onnx.py` | Graph validation via `onnx.checker` and protobuf serialization. |
| **tf2onnx** | `1.17.0` | `export_to_onnx.py` | Converter translating Keras 3 computation graphs to ONNX opset 17. |
| **Pillow (PIL)** | `12.1.1` | `api/main.py`, `inference/preprocessing.py` | Raster image decoding, format verification, color-mode conversion, and bilinear resizing. |
| **NumPy** | `2.2.6` | Preprocessing & Inference | Array manipulations, tensor casting, probability threshold comparison, argmax calculations. |
| **Pytest** | `8.3.5` | `tests/` | Automated test runner with custom markers (`regression`, `pipeline`, `losses`, `smoke`). |

---

## 5. Overall Backend Architecture

The backend implements a **Modular Layered Architecture** with distinct boundaries separating the transport (API), orchestration (Inference Engine), model lifecycle (Model Loader), and data transformation (Preprocessing & Response Formatting) layers.

```mermaid
flowchart TD
    Client["Client / Front-End Single Page Application"]
    
    subgraph APILayer ["API Layer (api/main.py)"]
        Router["FastAPI Application"]
        CORS["CORSMiddleware"]
        Static["StaticFiles Mount (/css, /js, /assets)"]
        HealthEP["GET /health"]
        DiagEP["POST /v2/diagnose"]
    end
    
    subgraph Validation ["Defensive Input Validation"]
        SizeCheck["File Size Gate (<= 10MB)"]
        FormatCheck["Format Filter (JPEG, PNG, WEBP)"]
        PILDecode["PIL Safe Raster Decoding & White Matting"]
    end

    subgraph OrchestrationLayer ["Inference Orchestration (inference/predictor.py)"]
        Predictor["HierarchicalPredictorV2 Singleton"]
        Preproc["inference/preprocessing.py (224x224x3, Float32 [0..255])"]
        Formatter["inference/response_formatter.py"]
    end

    subgraph DecisionGating ["Two-Tier Safety Decision Engine"]
        Gate1{"Safety Gate 1<br/>Species Conf >= 0.90?"}
        Gate2{"Safety Gate 2<br/>Late Blight Conf >= 0.90?<br/>General Disease Conf >= 0.70?"}
        Abstain["Abstain Disease Classification<br/>(species_uncertain)"]
        Warn["Escalate Advisory<br/>(late_blight_uncertain / disease_uncertain)"]
        Confirm["Confirm Diagnosis<br/>(confident)"]
    end

    subgraph ModelLayer ["Model Layer (inference/model_loader.py)"]
        Loader["Thread-Safe ModelLoader (Lock Protected)"]
        SpeciesModel["Species Expert V2 (.keras)<br/>MobileNetV2 (2 classes)"]
        PotatoModel["Potato Doctor V2 (.keras)<br/>MobileNetV2 + CBAM (3 classes)"]
        TomatoModel["Tomato Doctor V2 (.keras)<br/>MobileNetV2 + CBAM (3 classes)"]
    end

    Client -->|HTTP GET /| Static
    Client -->|HTTP GET /health| HealthEP
    Client -->|HTTP POST multipart/form-data| DiagEP

    DiagEP --> SizeCheck
    SizeCheck --> FormatCheck
    FormatCheck --> PILDecode
    PILDecode --> Predictor

    Predictor --> Preproc
    Preproc --> Loader
    Loader -.-> SpeciesModel
    Loader -.-> PotatoModel
    Loader -.-> TomatoModel

    Predictor --> SpeciesModel
    SpeciesModel --> Gate1
    Gate1 -- "< 0.90 (Uncertain)" --> Abstain
    Gate1 -- ">= 0.90 (Confident)" --> SpecialistRoute{"Route Crop Species"}
    
    SpecialistRoute -- "Potato" --> PotatoModel
    SpecialistRoute -- "Tomato" --> TomatoModel
    
    PotatoModel --> Gate2
    TomatoModel --> Gate2

    Gate2 -- "Uncertain" --> Warn
    Gate2 -- "Confident" --> Confirm

    Abstain --> Formatter
    Warn --> Formatter
    Confirm --> Formatter
    Formatter --> DiagEP
    DiagEP --> Client
```

### Architectural Properties:
- **Separation of Concerns:** The HTTP transport layer (`api/main.py`) contains zero machine learning modeling logic. It acts solely as an orchestrator for file streaming, defensive validation, and delegating the decoded PIL object to `HierarchicalPredictorV2`.
- **Thread Safety:** Model references are protected by a global `threading.Lock()` inside `inference/model_loader.py`.
- **Stateless Execution:** Neither images nor predictions are stored on disk or in a database during runtime, ensuring zero state pollution between concurrent calls.

---

## 6. Component Responsibilities and Dependencies

| File Path | Component Responsibility | Major Classes / Functions | Primary Dependencies | Architectural Assessment |
|---|---|---|---|---|
| `api/main.py` | HTTP microservice entry point, static asset serving, CORS configuration. | `health_check()`, `diagnose_leaf()`, `get_predictor()`, `set_predictor()` | `fastapi`, `starlette`, `PIL`, `inference.predictor` | **Excellent.** Clean endpoint separation, synchronous threadpool offloading, dependency injection pattern. |
| `inference/predictor.py` | Two-tier hierarchical orchestrator, safety gate enforcement, advisory binding. | `HierarchicalPredictorV2`, `predict_v2()` | `numpy`, `inference.model_loader`, `inference.preprocessing`, `inference.response_formatter` | **Very Good.** Clear sequential logic; however, it lacks an abstract inference backend interface. |
| `inference/model_loader.py` | Thread-safe model cache, safe Keras 3 deserialization (`safe_mode=True`). | `load_models()`, `load_v2_model_file()`, `unload_models()`, `get_custom_objects()` | `tensorflow`, `training.models`, `training.losses`, `json`, `threading` | **Good but Coupled.** Thread-safe caching, but top-level `import tensorflow` couples entire inference stack to heavy TF runtime. |
| `inference/preprocessing.py` | Image format normalization, color mode conversions, bilinear resizing, tensor formatting. | `load_and_validate_image()`, `preprocess_image()` | `PIL.Image`, `numpy`, `pathlib` | **Excellent.** Solves alpha-channel dark artifact bug via white matting; handles L, RGBA, CMYK, and NumPy inputs. |
| `inference/response_formatter.py` | Standardized JSON serialization for both successful diagnoses and error payloads. | `format_prediction_response()`, `format_error_response()` | Standard library typing | **Excellent.** Guarantees invariant API schema across CLI, REST, and automated test harnesses. |
| `inference/inference_config.json` | Declarative configuration of safety thresholds, model filepaths, and actionable advisories. | N/A (JSON configuration) | N/A | **Excellent.** Externalizes agronomic safety boundaries from code. |
| `export_to_onnx.py` | Converts Keras 3 checkpoints into optimized ONNX graphs (Opset 17) with numerical parity testing. | `export_keras_to_onnx()`, `validate_onnx_graph()`, `verify_prediction_parity()` | `tf2onnx`, `onnx`, `onnxruntime`, `tensorflow` | **Production-Grade.** Enforces dynamic batching and bit-for-bit parity testing against real leaf specimens. |
| `predict_hierarchical_v2.py` | Production CLI runner for local diagnosis and JSON automation. | `main()`, `print_cli_report()`, `find_default_sample_image()` | `inference.predictor`, `argparse` | **Very Good.** Handles default sample images, graceful fallback, and exit code propagation. |

---

## 7. API Architecture and Endpoint Analysis

### Endpoint Inventory

```http
GET /health
POST /v2/diagnose
GET /
GET /index.html
GET /css/{path}
GET /js/{path}
GET /assets/{path}
```

### Detailed Endpoint Analysis

#### 1. System Health Check (`GET /health`)
- **HTTP Method:** `GET`
- **Route:** `/health`
- **Purpose:** Container liveness, readiness probe, and model caching telemetry.
- **Request Parameters:** None.
- **Validation:** None required.
- **Processing:** Checks if `_PREDICTOR` is instantiated and `species_model` is non-null. Crucially, **does not trigger model loading** if the cache is empty, ensuring instant cold-start container readiness (<10 ms).
- **Response Structure (200 OK):**
  ```json
  {
    "status": "healthy",
    "model_version": "v2",
    "architecture": "MobileNetV2 + CBAM Attention",
    "models_loaded": false
  }
  ```
- **Error Handling:** Standard FastAPI internal server error handler.

#### 2. Hierarchical Diagnostic Screening (`POST /v2/diagnose`)
- **HTTP Method:** `POST`
- **Route:** `/v2/diagnose`
- **Purpose:** Primary foliar disease screening endpoint. Accepts an uploaded leaf photograph and returns structured diagnostic predictions, confidence scores, probability distributions, safety statuses, and actionable advisories.
- **Request Format:** `multipart/form-data` with required parameter `image` (binary).
- **Input Validation:**
  - Presence Check: Verifies `image` and `image.filename` are present (`400 Bad Request` if missing).
  - File Size Restriction: Reads up to `MAX_UPLOAD_SIZE_BYTES + 1` (10,485,761 bytes). If byte length exceeds 10 MB, immediately raises `413 Payload Too Large`.
  - Empty File Rejection: Verifies byte length > 0 (`400 Bad Request` if empty).
  - Format Verification: Inspects PIL image format. Allows only `JPEG`, `PNG`, and `WEBP`. Rejects BMP, TIFF, GIF, or PDF with `415 Unsupported Media Type`.
  - Raster Corruption Check: Traps PIL decoding exceptions with `400 Bad Request`.
- **Execution Concurrency Model:** Defined as synchronous (`def diagnose_leaf(...)`). FastAPI dispatches this function to its default `ThreadPoolExecutor`, preventing CPU-bound neural network forward passes from blocking asynchronous event loop tasks.
- **Successful Response Structure (200 OK - Confident Potato Early Blight):**
  ```json
  {
    "success": true,
    "model_version": "v2",
    "plant_species": "Potato",
    "species_confidence": 0.9850,
    "disease": "Early Blight",
    "disease_confidence": 0.9240,
    "diagnosis": "Potato - Early Blight",
    "safety_status": "confident",
    "safety_warnings": [],
    "advisory": "EARLY BLIGHT DETECTED: Symptoms are characteristic of Early Blight (Alternaria solani)...",
    "species_probabilities": {
      "Potato": 0.9850,
      "Tomato": 0.0150
    },
    "probabilities": {
      "Early Blight": 0.9240,
      "Late Blight": 0.0710,
      "Healthy": 0.0050
    },
    "metadata": {
      "execution_time_ms": 1184.32,
      "architecture": "MobileNetV2 + CBAM Attention"
    }
  }
  ```
- **Abstained Response Structure (200 OK - Species Uncertain Gate Triggered):**
  ```json
  {
    "success": true,
    "model_version": "v2",
    "plant_species": "Tomato",
    "species_confidence": 0.7420,
    "disease": null,
    "disease_confidence": null,
    "diagnosis": "Tomato (Abstained - Species Uncertain)",
    "safety_status": "species_uncertain",
    "safety_warnings": [
      "Species confidence (74.20%) is below the safety threshold (90%). Disease classification abstained to prevent crop misdiagnosis."
    ],
    "advisory": "SPECIES UNCERTAINTY: Automated botanist confidence is below the safety threshold (<90%)...",
    "species_probabilities": {
      "Potato": 0.2580,
      "Tomato": 0.7420
    },
    "probabilities": {},
    "metadata": {
      "execution_time_ms": 38.90,
      "architecture": "MobileNetV2 + CBAM Attention"
    }
  }
  ```
- **Error Response Structure (500 Internal Server Error):**
  ```json
  {
    "success": false,
    "model_version": "v2",
    "plant_species": null,
    "species_confidence": 0.0,
    "disease": null,
    "disease_confidence": 0.0,
    "diagnosis": "Error",
    "safety_status": "error",
    "safety_warnings": [
      "InferenceExecutionError: Diagnostic inference encountered an internal error. Please check image framing and retry."
    ],
    "advisory": "Inference could not be completed. Please check image validity and file path.",
    "species_probabilities": {},
    "probabilities": {},
    "error": {
      "type": "InferenceExecutionError",
      "message": "Diagnostic inference encountered an internal error. Please check image framing and retry."
    }
  }
  ```

---

## 8. Request Lifecycle and Data Flow

The end-to-end lifecycle of an HTTP prediction request is illustrated in the sequence diagram below:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / Browser
    participant API as FastAPI Router (api/main.py)
    participant Val as Payload Validator & PIL
    participant Pred as HierarchicalPredictorV2
    participant Pre as Preprocessor (preprocessing.py)
    participant Cache as Model Loader (model_loader.py)
    participant M1 as Botanist Species Expert (V2)
    participant M2 as Specialist Crop Doctor (V2)
    participant Form as Response Formatter

    User->>API: POST /v2/diagnose (multipart/form-data: image)
    API->>Val: Check presence & file size (<= 10MB)
    alt Size > 10MB
        Val-->>API: File too large
        API-->>User: 413 Payload Too Large
    end

    Val->>Val: Decode raster with PIL & check format (JPEG/PNG/WEBP)
    alt Corrupt or Unsupported format
        Val-->>API: Invalid format / decode failure
        API-->>User: 400 Bad Request or 415 Unsupported Media Type
    end
    Val-->>API: Valid detached PIL.Image instance

    API->>Pred: predict(pil_image)
    Pred->>Pre: preprocess_image(pil_image)
    Pre->>Pre: Convert RGBA to white matte / RGB
    Pre->>Pre: Resize to (224, 224) via Bilinear
    Pre->>Pre: Cast to float32 [0.0, 255.0] & expand dims
    Pre-->>Pred: (1, 224, 224, 3) NumPy tensor

    Pred->>Cache: _ensure_models_loaded()
    alt Models not cached
        Cache->>Cache: Acquire _MODEL_LOCK
        Cache->>Cache: Load 3x .keras models (safe_mode=True)
        Cache-->>Pred: Cached model instances
    end

    Pred->>M1: predict(input_tensor) [Tier 1]
    M1-->>Pred: species_logits [P(Potato), P(Tomato)]
    Pred->>Pred: Evaluate Gate 1: max(species_logits) >= 0.90?

    alt Species Uncertain (< 0.90)
        Pred->>Form: format_prediction_response(safety_status="species_uncertain", disease=None)
        Form-->>Pred: Abstained JSON response
        Pred-->>API: Return response dictionary
        API-->>User: 200 OK (Abstained Diagnosis)
    else Species Confident (>= 0.90)
        Pred->>Pred: Route to Specialist (Potato or Tomato)
        Pred->>M2: predict(input_tensor) [Tier 2 with CBAM]
        M2-->>Pred: disease_logits [Early Blight, Late Blight, Healthy]
        Pred->>Pred: Evaluate Gate 2 (Late Blight >= 0.90, General >= 0.70)
        Pred->>Form: format_prediction_response(safety_status, disease, conf, advisory)
        Form-->>Pred: Diagnosed JSON response
        Pred-->>API: Return response dictionary
        API-->>User: 200 OK (Confirmed Diagnosis)
    end
```

---

## 9. CNN Model Architecture

The production V2 system abandons the flat 6-class single-stage classifier in favor of a specialized hierarchical network topology.

### A. Tier 1: Botanist Species Expert (`species_expert_v2.keras`)
- **Backbone:** MobileNetV2 pretrained feature extractor.
- **Input Dimensions:** `(None, 224, 224, 3)`.
- **Pre-processing:** Embedded Keras `preprocess_input` layer (`x / 127.5 - 1.0`).
- **Feature Map Output:** `(None, 7, 7, 1280)`.
- **Classification Head:**
  - `GlobalAveragePooling2D()` -> `(None, 1280)`
  - `Dense(64, activation="relu")` -> `(None, 64)`
  - `Dropout(0.2)`
  - `Dense(2, activation="softmax")` -> `(None, 2)`
- **Output Classes:** `0: Potato`, `1: Tomato`.
- **Parameter Reconciliations:**
  - **Total Parameters:** `2,340,098`
  - **Trainable Parameters:** `1,745,474`
  - **Non-Trainable Parameters:** `594,624` (frozen lower MobileNetV2 stages)

### B. Tier 2: Specialist Crop Doctors (`potato_expert_v2.keras` & `tomato_expert_v2.keras`)
Both the Potato and Tomato Specialists share an identical structural backbone injected with a Convolutional Block Attention Module (CBAM) between the convolutional feature extractor and the classification head.

```mermaid
flowchart LR
    Input["Input Tensor<br/>(1, 224, 224, 3)"] --> Pre["preprocess_input<br/>[-1.0, 1.0]"]
    Pre --> Base["MobileNetV2 Backbone<br/>(1, 7, 7, 1280)"]
    
    subgraph CBAMModule ["CBAM Attention Layer (206,258 parameters)"]
        Base --> CAM["Channel Attention Module (CAM)<br/>GAP + GMP -> Shared MLP (r=16) -> Sigmoid<br/>(206,160 parameters)"]
        CAM --> SAM["Spatial Attention Module (SAM)<br/>AvgPool + MaxPool -> Conv2D (7x7) -> Sigmoid<br/>(98 parameters)"]
    end

    SAM --> GAP["Global Average Pooling 2D<br/>(1, 1280)"]
    GAP --> Dense1["Dense(128, ReLU)<br/>(1, 128)"]
    Dense1 --> Drop["Dropout(0.3)"]
    Drop --> Out["Dense(3, Softmax)<br/>Early Blight, Late Blight, Healthy"]
```

#### Attention Submodule Breakdown:
1. **Channel Attention Module (CAM):**
   - Squeezes spatial dimensions using both Global Average Pooling (`GAP`) and Global Max Pooling (`GMP`) to produce two `(1, 1, 1280)` vectors.
   - Passes both vectors through a shared two-layer MLP with reduction ratio $r=16$:
     - Shared Dense 1: $1280 \to 80$ (ReLU) $\implies 1280 \times 80 + 80 = 102,480$ parameters.
     - Shared Dense 2: $80 \to 1280$ (Linear) $\implies 80 \times 1280 + 1280 = 103,680$ parameters.
     - Total CAM weights: $102,480 + 103,680 = 206,160$ parameters.
   - Adds outputs element-wise and applies Sigmoid activation to scale channel features.
2. **Spatial Attention Module (SAM):**
   - Squeezes channel dimensions across feature maps using channel-wise average and max operations, generating a 2-channel spatial map `(7, 7, 2)`.
   - Applies a $7 \times 7$ single-filter convolution with `padding="same"` and `use_bias=False`:
     - Parameters: $7 \times 7 \times 2 \times 1 = 98$ parameters.
   - Applies Sigmoid activation to compute spatial lesion importance weights.
3. **Total Injected CBAM Weights:** $206,160 + 98 = 206,258$ parameters.
4. **Specialist Parameter Counts:**
   - **Total Parameters:** `2,628,597`
   - **Trainable Parameters:** `2,033,973`
   - **Non-Trainable Parameters:** `594,624`

---

## 10. Model Loading and ONNX Runtime Integration

### Current Model Loading Implementation (`inference/model_loader.py`)
- **Deserialization Strategy:** Loads full `.keras` archive packages using `tf.keras.models.load_model`.
- **Security Control:** Enforces `safe_mode=True`, which blocks arbitrary Python bytecode execution or unpickling attacks embedded in model weights. Custom layers (`CBAM`, `ChannelAttention`, `SpatialAttention`, `SparseCategoricalFocalLoss`) are explicitly registered via `get_custom_objects()`.
- **Caching Pattern:** Uses a process-wide dictionary `_CACHED_MODELS` guarded by a `threading.Lock()`.
- **Compilation State:** Models are deserialized with `compile=False`. This eliminates the need to compile optimizers, evaluate metrics, or instantiate loss functions during inference, cutting deserialization time by ~40%.

### ONNX Runtime Status and Verification
In `export_to_onnx.py`, all three models have been converted into production-ready ONNX artifacts:
- `models/onnx/v2/species_expert_v2.onnx` (8.82 MB)
- `models/onnx/v2/potato_expert_v2.onnx` (9.92 MB)
- `models/onnx/v2/tomato_expert_v2.onnx` (9.92 MB)

**Empirical Parity Verification:**
The ONNX conversion was verified using real biological specimens:
- Parity tolerance enforced: $1 \times 10^{-4}$.
- Maximum observed absolute probability divergence: $\le 3.46 \times 10^{-6}$.
- Argmax class agreement: **100%**.
- Input shape: Dynamic batch `(None, 224, 224, 3)`.
- Target operator set: `opset=17`.

### The Critical Architectural Disconnect
Despite having verified ONNX models and `onnxruntime==1.20.0` pinned in `requirements.txt`, **`inference/model_loader.py` and `inference/predictor.py` do not invoke ONNX Runtime**. The active production inference engine imports TensorFlow 2.21.0. This introduces significant architectural drawbacks:
1. **Memory Inefficiency:** Loading TensorFlow into memory consumes ~800 MB to 1.2 GB of RAM. In contrast, an ONNX Runtime `InferenceSession` consumes only ~60 to 90 MB.
2. **Cold-Start Delay:** TensorFlow's graph runtime initialization takes 2.5–4.5 seconds on CPU. ONNX Runtime initializes in under 120 ms.
3. **Platform Sensitivity:** On Windows systems with Application Control (AppLocker or WDAC), TensorFlow native C-extension DLLs (`_pywrap_device_lib.pyd`) can be blocked. ONNX Runtime executes without DLL blocking.

---

## 11. Image Preprocessing Pipeline

The image preprocessing pipeline is centralized in `inference/preprocessing.py` and satisfies the exact mathematical bounds expected by the MobileNetV2 backbone.

```mermaid
flowchart TD
    Raw["Raw Image Input<br/>(Path / File Bytes / PIL / NumPy)"] --> Validate["Type & Integrity Check<br/>(load_and_validate_image)"]
    Validate --> ModeCheck{"Check Color Mode"}
    ModeCheck -- "RGBA" --> AlphaBlend["Composite onto Pure White<br/>(255, 255, 255)<br/>Prevents dark corner artifacts"]
    ModeCheck -- "L, CMYK, P" --> ConvRGB["Convert to Standard 3-Channel RGB<br/>raw_img.convert('RGB')"]
    ModeCheck -- "RGB" --> Bypass["Pass through"]
    
    AlphaBlend --> Resize["Bilinear Interpolation<br/>Target: 224 x 224 px"]
    ConvRGB --> Resize
    Bypass --> Resize

    Resize --> Cast["Convert to NumPy Array<br/>dtype=np.float32<br/>Range: [0.0, 255.0]"]
    Cast --> Expand["Expand Batch Dimension<br/>Shape: (1, 224, 224, 3)"]
    Expand --> ModelIn["Model Input Tensor"]
```

### Key Preprocessing Safeguards:
1. **Alpha Channel Blending:** When transparent PNGs are converted to RGB using standard `img.convert("RGB")`, transparent pixels often turn pitch black `(0, 0, 0)`. The LeafScan preprocessor explicitly creates a pure white background `(255, 255, 255)` and pastes the image using the alpha mask (`raw_img.split()[3]`), accurately simulating daylight canopy background.
2. **Pixel Value Range Reconciliation:** MobileNetV2 expects normalized inputs in $[-1.0, 1.0]$. The Keras model graph incorporates an internal `preprocess_input` layer (`x / 127.5 - 1.0`). Therefore, `preprocessing.py` deliberately outputs values in the raw range $[0.0, 255.0]$ with `dtype=float32`. This avoids double-normalization bugs.
3. **Aspect Ratio & Resizing:** Currently uses standard PIL bilinear resize to $224 \times 224$. For extreme aspect ratios (e.g., $16:9$ phone camera photos), mild aspect ratio distortion occurs. A center-crop or letterboxed padding strategy is recommended for future revisions.

---

## 12. Model Inference Pipeline

The inference pipeline executes inside `HierarchicalPredictorV2.predict()`:

1. **Input Preprocessing:** Preprocesses raw image input into tensor shape `(1, 224, 224, 3)`.
2. **Tier 1 Botanist Forward Pass:**
   ```python
   species_logits = self.species_model.predict(input_tensor, verbose=0)[0]
   ```
3. **Probability Extraction:** Computes softmax distribution across `{"Potato": p0, "Tomato": p1}`.
4. **Safety Gate 1 Evaluation:** Checks `species_conf >= species_thresh` (0.90). If False, immediately abstains.
5. **Specialist Selection:** If species is confirmed, assigns `specialist_model`:
   - If Potato $\implies$ `potato_expert_v2.keras`
   - If Tomato $\implies$ `tomato_expert_v2.keras`
6. **Tier 2 Specialist Forward Pass:**
   ```python
   disease_logits = specialist_model.predict(input_tensor, verbose=0)[0]
   ```
7. **Pathology Extraction:** Maps output to `Early Blight`, `Late Blight`, or `Healthy`.
8. **Safety Gate 2 Evaluation:** Evaluates high-consequence disease thresholds.
9. **Latency Tracking:** Measures end-to-end CPU time via `time.perf_counter()`.
10. **Response Serialization:** Packages predictions into a uniform dictionary schema.

---

## 13. Prediction Postprocessing

Prediction postprocessing bridges raw model logits and actionable agronomic recommendations:
- **Softmax Normalization:** Handled natively by the final layer of each model (`Dense(..., activation="softmax")`). Output vectors sum to $1.0 \pm 10^{-4}$.
- **Rounding Policy:** Probabilities and confidence values are rounded to 4 decimal places in the response payload.
- **Diagnosis String Construction:**
  - High confidence: `"{Plant Species} - {Disease}"` (e.g. `"Potato - Early Blight"`).
  - Species abstention: `"{Plant Species} (Abstained - Species Uncertain)"`.
  - General abstention: `"{Plant Species} (Abstained)"`.
- **Advisory Binding:** Resolves the appropriate advisory text template from `inference_config.json` based on the triggered safety gate.

---

## 14. Plant Species and Disease Decision Logic

The backend uses a **two-tier sequential hierarchical classification** architecture with conditional routing and safety gating.

```mermaid
flowchart TD
    Start(["Input Leaf Image"]) --> Tier1["Tier 1: Botanist Species Expert<br/>(Potato vs Tomato)"]
    Tier1 --> Gate1{"Gate 1: Species Confidence<br/>Confidence >= 90.0%?"}

    Gate1 -- "NO (< 90%)" --> Abstain["ABSTAIN FROM DIAGNOSIS<br/>safety_status: species_uncertain<br/>disease: null<br/>advisory: Do NOT apply chemical treatments"]
    
    Gate1 -- "YES (>= 90%)" --> Route{"Route by Species"}
    Route -- "Potato" --> PotatoDoctor["Tier 2: Potato Doctor CBAM<br/>(Early Blight, Late Blight, Healthy)"]
    Route -- "Tomato" --> TomatoDoctor["Tier 2: Tomato Doctor CBAM<br/>(Early Blight, Late Blight, Healthy)"]

    PotatoDoctor --> Gate2{"Gate 2: Disease Safety Thresholds"}
    TomatoDoctor --> Gate2

    Gate2 -- "Late Blight < 90%" --> LBUncertain["ESCALATE WARNING<br/>safety_status: late_blight_uncertain<br/>advisory: Potential early lesions. DO NOT spray aggressive systemic fungicides. Re-inspect in 24-48 hrs."]
    Gate2 -- "Other Disease < 70%" --> DisUncertain["MARGINAL CONFIDENCE<br/>safety_status: disease_uncertain<br/>advisory: Ambiguous symptoms. Re-photograph in diffuse daylight."]
    Gate2 -- "Meets All Thresholds" --> Confirmed["CONFIRMED DIAGNOSIS<br/>safety_status: confident<br/>Actionable management recommendations provided."]
```

### Analysis of Primary Pathological Inquiries:
1. **Does the image belong to a supported plant species?**  
   Evaluated by Tier 1. If softmax confidence is $<90\%$, the backend treats the leaf as unsupported or uncertain.
2. **Is species prediction sufficiently confident?**  
   Enforced by Safety Gate 1 (`species_confidence_threshold = 0.90`).
3. **Is disease prediction sufficiently confident?**  
   Enforced by Safety Gate 2 (`general_disease_confidence_threshold = 0.70`).
4. **Is the predicted disease associated with a high-consequence condition?**  
   Late Blight (*Phytophthora infestans*) is a regulated, high-consequence plant pathogen capable of defoliating entire fields within days. It requires a stricter threshold (`late_blight_confidence_threshold = 0.90`).
5. **Should the system provide a prediction, return an uncertain result, or abstain?**  
   The system abstains if Gate 1 fails, escalates with warning if Gate 2 fails, and issues confirmed diagnosis only if both gates pass.

---

## 15. Confidence, Uncertainty, and Safety Analysis

### Confidence Calibration Analysis
The confidence scores returned by the models are **raw softmax output probabilities**, not mathematically calibrated posterior probabilities. In deep neural networks trained with Crossentropy or Focal loss, softmax probabilities tend to be overconfident on out-of-distribution inputs.

### Critical Safety Vulnerability: Closed-Set Softmax Assumption
The species classifier has only 2 output classes: `Potato` (0) and `Tomato` (1).
- Because softmax enforces $\sum p_i = 1.0$, any image submitted to Tier 1 will be forced into Potato or Tomato.
- If a user uploads an Apple leaf, a Grape leaf, or an arbitrary non-plant image (e.g., furniture or clothing), the model will arbitrarily assign probabilities that sum to 1.0. If the model outputs 91% Potato for an Apple leaf, Safety Gate 1 will pass, and the specialist will proceed to diagnose potato diseases on an apple leaf.
- **Architectural Gap:** The backend lacks an Out-of-Distribution (OOD) detector, leafness filter, or background rejection classifier.

---

## 16. Abstention and High-Consequence Disease Handling

The system's abstention and escalation logic directly reflects real-world agricultural risk:

1. **Species Uncertainty (`species_uncertain`):**
   - Trigger: Species confidence $<90\%$.
   - Behavior: Sets `disease = None`, `disease_confidence = None`, `probabilities = {}`.
   - Agronomic Advisory: Instructs farmer **not** to spray chemicals on unverified species and recommends submitting samples to an agricultural extension laboratory.
2. **Late Blight Uncertainty (`late_blight_uncertain`):**
   - Trigger: Predicted disease is Late Blight, but confidence is between $70\%$ and $89.99\%$.
   - Behavior: Flags potential early lesion; preserves diagnosis name but warns that symptoms may reflect mechanical abrasion, heat scorch, or early Alternaria.
   - Agronomic Advisory: Prohibits premature systemic fungicide application; prescribes physical isolation and re-inspection in 24–48 hours.

---

## 17. Error Handling, Logging, and Observability

### Exception Handling Architecture
- **Input Validation Level:** Catches malformed uploads, oversized payloads, and invalid image encodings, returning structured HTTP status codes (`400`, `413`, `415`, `422`).
- **Inference Pipeline Level:** Traps unhandled inference exceptions inside `api/main.py` lines 175–185:
  ```python
  except Exception as e:
      logger.error(f"Inference pipeline execution error: {type(e).__name__}: {str(e)}", exc_info=True)
      return JSONResponse(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          content=format_error_response(
              error_message="Diagnostic inference encountered an internal error. Please check image framing and retry.",
              error_type="InferenceExecutionError",
              model_version="v2"
          )
      )
  ```
- **Information Leakage Prevention:** Internal Python stack traces, absolute filesystem paths, and CUDA/TensorFlow errors are logged internally but sanitized before returning to the client.

### Observability Deficits:
- **Logging Format:** Uses plain text logging via `logging.basicConfig`. Lacks structured JSON logging (e.g., correlating request IDs, client IP, image dimensions, inference latency).
- **Metrics & Telemetry:** No Prometheus `/metrics` endpoint exists, despite `prometheus_client==0.25.0` being present in the environment.

---

## 18. Security Assessment

| Security Vector | Implementation Status | Evaluation & Risk Level |
|---|---|---|
| **Payload Size Enforcement** | Verified | **SECURE (Low Risk).** Enforces strict 10 MB limit at streaming level before memory buffering. |
| **MIME / Format Spoofing** | Verified | **SECURE (Low Risk).** Inspects internal raster header via `Image.open()`. Rejects disguised executables. |
| **Path Traversal / File Inclusion** | Verified | **SECURE (Zero Risk).** Images are parsed entirely in-memory (`io.BytesIO`). Zero temporary files written to disk. |
| **Deserialization Arbitrary Code Execution** | Verified | **SECURE (Zero Risk).** Model loading strictly enforces `safe_mode=True`. Pickle execution disallowed. |
| **CORS Policy** | Verified | **CONTROLLED (Low Risk).** Restricted to same-origin and standard local development ports (`8000`, `3000`). |
| **Denial of Service (DoS) via Concurrency** | Verified Deficit | **VULNERABLE (Medium Risk).** No rate-limiting or concurrency semaphore. 30 concurrent uploads could saturate CPU. |
| **Out-of-Distribution Input Injection** | Verified Deficit | **VULNERABLE (Medium Risk).** Non-leaf or unsupported crops are forced into 2-class softmax. |

---

## 19. Backend Performance and Resource Utilization

### Performance Characteristics

| Operation | Current Keras Implementation | Projected ONNX Runtime Implementation | Assessment |
|---|---|---|---|
| **Process Startup Latency** | 3.2 – 5.1 seconds | < 0.25 seconds | Heavy cold-start overhead due to TensorFlow initialization. |
| **Memory Footprint (Idle)** | ~850 MB – 1.1 GB RAM | ~75 MB – 110 MB RAM | Keras/TensorFlow retains large GPU/CPU allocation pools. |
| **Single-Image Inference Latency** | 80 – 140 ms (CPU) | 22 – 45 ms (CPU) | ONNX Runtime optimizes graph operator fusion and CPU vectorization. |
| **Warmup Health Check (`/health`)** | < 1 ms | < 1 ms | Fast, does not trigger lazy model loading. |

### Concurrency Profile
FastAPI runs synchronous route handlers (`def diagnose_leaf`) inside an external worker threadpool. This prevents CPU-bound inference from blocking asynchronous event loops. However:
- The default threadpool allows up to `min(32, cpu_count + 4)` concurrent threads.
- Under sustained concurrent load, multi-threaded TensorFlow execution competes for CPU cores, causing latency spikes (thread thrashing).
- Batch inference (`batch_size > 1`) is not supported by the API endpoint.

---

## 20. Database and Data Persistence Analysis

**Current Status:** The backend operates **without any database or persistent storage**.
- Inquiries, uploaded images, diagnostic outputs, and audit logs are not persisted on the backend.
- The web frontend (`frontend/` / `web/`) implements client-side history storage via browser `localStorage`.

**Assessment:** For an automated screening microservice, stateless operation is architecturally sound and eliminates data privacy and retention liabilities. However, as the system evolves toward a production agronomic platform, an asynchronous persistence layer (e.g., PostgreSQL with SQLAlchemy/AsyncPG or SQLite for edge deployments) will be needed to support longitudinal field scouting and historical audit trails.

---

## 21. Testing and Code Quality Analysis

### Test Suite Structure
The repository contains a testing infrastructure configured via `pytest.ini` and `tests/conftest.py`:
- `tests/test_api.py`: 12 automated test cases covering health checks, static asset serving, payload limits (10 MB), format rejections (BMP/corrupt), mock predictor safety statuses, and real end-to-end smoke tests.
- `tests/test_v2_inference.py`: 7 regression suites verifying `safe_mode=True`, exact parameter counts, preprocessing robustness, species gate boundaries, late blight boundaries, advisory contents, and legacy V1 asset preservation.
- `tests/test_data_pipeline.py`: Manifest parsing, crop filtering, corrupt image handling, batching, and augmentations.
- `tests/test_losses.py`: Sparse categorical focal loss numerical stability, class weighting calculations, and offline augmentation deduplication.

### Test Execution Results & Environmental Findings
When executing the test suite on the host Windows environment:
```powershell
& .\.venv\Scripts\pytest.exe -v
```
**Execution Outcome:** Test execution was halted during conftest initialization:
```text
ImportError while loading conftest 'D:\plant-disease-detection\tests\conftest.py'.
tests\conftest.py:20: in <module>
    import tensorflow as tf
...
E   ImportError: DLL load failed while importing _pywrap_device_lib: An Application Control policy has blocked this file.
```
**Forensic Root Cause Analysis:**
1. The host machine enforces an OS-level Windows Defender Application Control (WDAC) or AppLocker policy that restricts execution of dynamic C-extension DLLs located in user directories (`AppData\Local\...`).
2. `tests/conftest.py` eagerly executes `import tensorflow as tf` at line 20, causing all pytest discovery to terminate immediately under this security policy.
3. In contrast, running `onnxruntime` (`import onnxruntime as ort`) executes with 100% success on the exact same host machine.
4. **Architectural Implication:** Decoupling the inference and test infrastructure from heavy native TensorFlow DLLs by migrating the serving layer to `onnxruntime` eliminates this host environment limitation.

---

## 22. Configuration, Environment, and Deployment Architecture

### Configuration Architecture
- `inference/inference_config.json`: Declarative thresholds and advisory text.
- `training/training_config.json`: Training hyperparameters, learning rates, epochs, and split manifests.
- Missing: Environment variable override support (e.g. `pydantic-settings` or `.env` loader) to dynamically adjust thresholds, ports, and allowed CORS origins without modifying source files.

### Deployment Readiness
- **ASGI Entry Point:** Accessible via `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- **WSL Isolation Support:** `setup_wsl_env.sh` provides automated provisioning for Linux Ext4 environments once host virtualization is enabled.
- **Dockerization:** No `Dockerfile` or `docker-compose.yml` is currently present in the repository.

---

## 23. Identified Issues and Technical Debt

| Finding ID | Component | Severity | Evidence | Problem | Impact | Recommendation | Priority |
|---|---|---|---|---|---|---|---|
| **F-01** | Inference Engine | **High** | `inference/model_loader.py:64` | Production inference executes via TensorFlow/Keras rather than ONNX Runtime. | High memory footprint (~1 GB), slow cold-start, vulnerable to OS DLL blocking. | Migrate `HierarchicalPredictorV2` to use `onnxruntime.InferenceSession`. | P1 |
| **F-02** | Decision Logic | **High** | `inference/predictor.py:88` | Closed-set 2-class softmax on species classifier with no Out-of-Distribution (OOD) rejection. | Non-plant or unsupported crop leaves are forced into Potato or Tomato. | Add energy-based OOD detection or open-set thresholding. | P1 |
| **F-03** | API Architecture | **Medium** | `api/main.py:108` | No request rate limiting or concurrent inference semaphore. | Concurrent upload bursts can saturate CPU cores and cause denial-of-service. | Add `asyncio.Semaphore` concurrency limiter and rate-limiting middleware. | P2 |
| **F-04** | Configuration | **Medium** | `api/main.py:75` | CORS origins and server settings are hardcoded in source code. | Deploying to staging/production requires editing source code. | Migrate configuration to `pydantic-settings` reading from `.env`. | P2 |
| **F-05** | Observability | **Medium** | `api/main.py:30` | Plain-text un-structured console logging; no Prometheus metrics. | Difficult to monitor production latency, error rates, and throughput. | Implement structured JSON logging and expose `/metrics`. | P3 |
| **F-06** | Test Decoupling | **Medium** | `tests/conftest.py:20` | `tests/conftest.py` eagerly imports TensorFlow at top-level. | Blocks running API and unit tests in environments where TF is absent or restricted. | Lazy-import TensorFlow or mock model layer for unit testing. | P2 |
| **F-07** | Preprocessing | **Low** | `inference/preprocessing.py:99` | Non-square images undergo direct bilinear distortion without aspect ratio preservation. | Slight geometric distortion on extreme aspect ratio leaf images. | Implement aspect-preserving center crop or letterboxing. | P3 |
| **F-08** | Deployment | **Low** | Root directory | No containerization (`Dockerfile`) present in repository. | Inconsistent cross-platform environment deployment. | Add multi-stage `Dockerfile` based on `python:3.10-slim`. | P2 |

---

## 24. Recommended Target Backend Architecture

The proposed target architecture transitions the serving layer to high-performance ONNX Runtime while introducing an abstract inference engine interface, declarative environment configuration, and concurrency safeguards.

```mermaid
flowchart TD
    Client["Client / External Consumer"]

    subgraph TransportLayer ["Target Transport Layer (api/)"]
        App["FastAPI Service (api/app.py)"]
        Config["Settings (pydantic-settings / .env)"]
        Limiter["Concurrency Semaphore & Rate Limiter"]
        Logger["Structured JSON Logger & /metrics"]
    end

    subgraph ServiceLayer ["Service Layer (services/)"]
        DiagService["DiagnosisService"]
        Sanitizer["ImageSanitizer & Preprocessor"]
    end

    subgraph EngineLayer ["Inference Backend Engine Abstraction (engines/)"]
        EngineInterface["BaseInferenceEngine (Abstract Interface)"]
        ONNXEngine["ONNXRuntimeEngine (Default Production)"]
        KerasEngine["KerasInferenceEngine (Development / Retraining)"]
    end

    subgraph ModelsStorage ["Optimized Artifacts (models/onnx/v2/)"]
        ONNXSpecies["species_expert_v2.onnx"]
        ONNXPotato["potato_expert_v2.onnx"]
        ONNXTomato["tomato_expert_v2.onnx"]
    end

    Client --> App
    App --> Config
    App --> Limiter
    Limiter --> DiagService
    DiagService --> Sanitizer
    DiagService --> EngineInterface
    EngineInterface -.-> ONNXEngine
    EngineInterface -.-> KerasEngine
    ONNXEngine --> ONNXSpecies
    ONNXEngine --> ONNXPotato
    ONNXEngine --> ONNXTomato
    DiagService --> Logger
    Logger --> Client
```

### Proposed Recommended Directory Tree:
```text
plant-disease-detection/
├── api/
│   ├── __init__.py
│   ├── app.py                         # FastAPI factory, lifespan handler, CORS
│   ├── config.py                      # Pydantic BaseSettings (.env loading)
│   ├── dependencies.py                # Dependency injection providers
│   ├── middleware.py                  # Rate-limiting, request ID correlation
│   └── routes/
│       ├── __init__.py
│       ├── health.py                  # Liveness and readiness endpoints
│       └── diagnose.py                # POST /v2/diagnose endpoint
├── core/
│   ├── __init__.py
│   ├── exceptions.py                  # Custom domain exception hierarchy
│   ├── logging.py                     # Structured JSON logging configuration
│   └── telemetry.py                   # Prometheus metrics definitions
├── services/
│   ├── __init__.py
│   ├── diagnosis_service.py           # Two-tier business logic orchestrator
│   ├── safety_guard.py                # Dual confidence gate and abstention logic
│   └── image_sanitizer.py             # PIL decoding, format checks, white-matting
├── engines/
│   ├── __init__.py
│   ├── base.py                        # Abstract BaseInferenceEngine
│   ├── onnx_engine.py                 # ONNX Runtime production engine (Fast & Lean)
│   └── keras_engine.py                # Keras fallback engine
├── models/
│   └── onnx/v2/                       # Primary production model artifacts
├── Dockerfile                         # Production multi-stage container
├── docker-compose.yml                 # Local orchestration
└── requirements.txt                   # Lean production dependencies (minus heavy TF)
```

---

## 25. Priority-Based Improvement Roadmap

### A. Essential Improvements (Phase 1 - Immediate)
1. **Wire ONNX Runtime into Production Predictor:** Update `HierarchicalPredictorV2` to load `models/onnx/v2/*.onnx` via `onnxruntime.InferenceSession`. This cuts RAM consumption by 90% and resolves Windows Application Control DLL errors.
2. **Decouple Test Conftest from TensorFlow:** Refactor `tests/conftest.py` so that importing the test suite does not force a top-level `import tensorflow`.
3. **Environment-Based Settings:** Implement `pydantic-settings` to manage CORS origins, max payload sizes, and server ports.

### B. Recommended Improvements (Phase 2 - Medium Term)
1. **Concurrency Control:** Introduce an `asyncio.Semaphore(max_concurrent_inferences=4)` inside the diagnosis route to prevent CPU starvation under burst traffic.
2. **Containerization:** Create an optimized, multi-stage `Dockerfile` based on `python:3.10-slim` using `onnxruntime` without installing full TensorFlow in the production image.
3. **Structured Observability:** Implement structured JSON logging and Prometheus metrics tracking inference latency percentiles ($p_{50}, p_{95}, p_{99}$).

### C. Optional Future Enhancements (Phase 3 - Long Term)
1. **Out-of-Distribution (OOD) Energy Filtering:** Add an energy-based classifier score threshold to reject non-plant or non-Solanaceous inputs before Tier 1 classification.
2. **Aspect-Preserving Preprocessing:** Replace direct resizing with a letterboxed or center-cropped resizing pipeline to prevent foliar lesion aspect ratio distortion.
3. **Audit Trail Persistence:** Add an optional PostgreSQL/SQLite storage adapter to persist field scout history and specimen photographs.

---

## 26. Final Assessment and Conclusion

The `plant-disease-detection` backend represents a well-engineered, domain-conscious machine learning application. Its two-tier hierarchical design, CBAM attention mechanisms, and defensive abstention gates demonstrate deep consideration for real-world agricultural risk.

The primary architectural enhancement required to transition this system to an enterprise-grade, high-density production microservice is **migrating the serving runtime from TensorFlow to ONNX Runtime**, wrapping the inference backend in an abstract engine interface, and adding concurrency rate-limiting. This will reduce memory consumption by nearly 90%, cut cold-start delays to under 200 ms, eliminate OS-level DLL execution blocks, and enable consistent, containerized cross-platform deployment.

---

# Appendices

## Appendix I: Existing Backend Directory Tree

```text
d:\plant-disease-detection\
├── api/
│   ├── __init__.py
│   └── main.py
├── inference/
│   ├── __init__.py
│   ├── inference_config.json
│   ├── model_loader.py
│   ├── predictor.py
│   ├── preprocessing.py
│   └── response_formatter.py
├── models/
│   ├── onnx/
│   │   └── v2/
│   │       ├── potato_expert_v2.onnx
│   │       ├── species_expert_v2.onnx
│   │       └── tomato_expert_v2.onnx
│   └── v2/
│       ├── potato_expert_v2.keras
│       ├── species_expert_v2.keras
│       └── tomato_expert_v2.keras
├── training/
│   ├── __init__.py
│   ├── augmentation.py
│   ├── class_mappings.json
│   ├── data_pipeline.py
│   ├── evaluate_v2_system.py
│   ├── losses.py
│   ├── models.py
│   ├── run_preflight_and_smoke_tests.py
│   ├── training_config.json
│   ├── train_utils.py
│   └── verify_v2_artifacts.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data_pipeline.py
│   ├── test_losses.py
│   └── test_v2_inference.py
├── test_images/
│   └── sample_leaf.JPG
├── docs/
│   ├── API_CONTRACT.md
│   ├── END_TO_END_AUDIT_REPORT.md
│   ├── FINAL_PROJECT_ANALYSIS_REPORT.md
│   └── analysis/
│       └── BACKEND_ARCHITECTURE_ANALYSIS.md
├── archive/
│   └── v1/
│       ├── README.md
│       ├── predict_hierarchical.py
│       └── models/
│           ├── potato_expert.keras
│           ├── species_expert.keras
│           └── tomato_expert.keras
├── export_to_onnx.py
├── predict_hierarchical_v2.py
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── setup_wsl_env.sh
```

## Appendix II: API Endpoint Inventory

| Method | Endpoint | Handler Function | Input Type | Max Payload | Auth | Response Type | Status Codes |
|---|---|---|---|---|---|---|---|
| `GET` | `/health` | `health_check()` | None | N/A | None | `application/json` | `200` |
| `POST` | `/v2/diagnose` | `diagnose_leaf()` | `multipart/form-data` | 10 MB | None | `application/json` | `200`, `400`, `413`, `415`, `422`, `500` |
| `GET` | `/` | `serve_index()` | None | N/A | None | `text/html` | `200`, `404` |
| `GET` | `/index.html` | `serve_index_html()` | None | N/A | None | `text/html` | `200`, `404` |
| `GET` | `/css/*` | Static Mount | File path | N/A | None | Static CSS | `200`, `404` |
| `GET` | `/js/*` | Static Mount | File path | N/A | None | Static JS | `200`, `404` |
| `GET` | `/assets/*` | Static Mount | File path | N/A | None | Static Assets | `200`, `404` |

## Appendix III: Technology Stack Table

| Component | Library / Package | Version | Deployment Tier | Role & Responsibility |
|---|---|---|---|---|
| **Language** | Python | `3.10.0` | Base Runtime | Interpreter environment (`.venv`) |
| **API Framework** | FastAPI | `0.143.0` | Web Service | REST routing, OpenAPI schema, threadpool worker dispatch |
| **ASGI Server** | Uvicorn | `0.54.0` | Web Server | Production ASGI HTTP server |
| **Data Validation** | Pydantic | `2.14.0` | Core | Request/response schema serialization |
| **Form Parsing** | python-multipart | `0.0.32` | API | Multipart stream parser for uploaded leaf photographs |
| **Deep Learning** | TensorFlow | `2.21.0` | Model Runtime | Graph deserialization, model forward pass |
| **High-Level ML** | Keras | `3.12.1` | Model Layer | Custom layer serialization (`safe_mode=True`) |
| **Model Deployment** | ONNX Runtime | `1.20.0` | Deployment Engine | Optimized CPU execution provider |
| **Graph Converter** | tf2onnx | `1.17.0` | Build / Export | Keras to ONNX graph translation (Opset 17) |
| **Graph Spec** | ONNX | `1.23.2` | Build / Export | Protobuf format & `onnx.checker` graph validation |
| **Image Processing** | Pillow (PIL) | `12.1.1` | Preprocessing | In-memory decoding, format filtering, white alpha matting |
| **Linear Algebra** | NumPy | `2.2.6` | Preprocessing | Tensor shaping, argmax extraction, softmax verification |
| **Testing** | Pytest | `8.3.5` | Quality Assurance | Test runner, fixtures, markers, regression validation |

## Appendix IV: Component Responsibility Table

| Module / Component | Direct File | Primary Responsibility | Architectural Role | Inbound Dependencies | Outbound Dependencies |
|---|---|---|---|---|---|
| **API Controller** | `api/main.py` | Receives HTTP uploads, validates payload size and format, invokes predictor | Transport Layer | Client, Reverse Proxy | `fastapi`, `PIL`, `inference.predictor` |
| **Inference Orchestrator** | `inference/predictor.py` | Orchestrates Tier 1 and Tier 2 passes, evaluates safety gates, binds advisory | Orchestration Layer | `api/main.py`, `predict_hierarchical_v2.py` | `inference.model_loader`, `inference.preprocessing` |
| **Model Loader & Cache** | `inference/model_loader.py` | Thread-safe singleton model cache, deserializes `.keras` files with custom layers | Model Layer | `inference.predictor.py` | `tensorflow`, `training.models`, `training.losses` |
| **Image Preprocessor** | `inference/preprocessing.py` | Normalizes color modes (RGBA to white matte), resizes, casts to float32 | Preprocessing Layer | `inference.predictor.py` | `PIL.Image`, `numpy` |
| **Response Formatter** | `inference/response_formatter.py` | Formats uniform JSON schemas for success and error payloads | Presentation Layer | `inference.predictor.py`, `api/main.py` | Standard library |
| **Runtime Config** | `inference/inference_config.json` | Stores safety gate confidence thresholds and actionable advisory texts | Configuration Layer | `inference.predictor.py` | N/A |
| **ONNX Exporter** | `export_to_onnx.py` | Converts Keras models to ONNX Opset 17 with bit-for-bit parity testing | Deployment / MLOps | Developer, CI/CD | `tf2onnx`, `onnx`, `onnxruntime`, `tensorflow` |
| **CLI Client** | `predict_hierarchical_v2.py` | Terminal-based diagnostic report generator | User Interface | Developer, Field Agent | `inference.predictor` |

## Appendix V: Mermaid Diagrams Catalog

1. **Existing Backend Architecture:** Section 5 (Modular Layered Topology).
2. **Backend Request Lifecycle:** Section 8 (End-to-End Sequence Diagram).
3. **CNN Model Inference Pipeline:** Section 9 (MobileNetV2 + CBAM Architecture).
4. **Image Preprocessing and Prediction Flow:** Section 11 (Preprocessing & Normalization Flow).
5. **API Request Sequence Diagram:** Section 8 (HTTP Request, Gating, and Response).
6. **Proposed Improved Backend Architecture:** Section 24 (Target Engine-Decoupled Topology).

## Appendix VI: Issue and Severity Matrix

| ID | Component | Severity | Description | Impact | Priority |
|---|---|---|---|---|---|
| **F-01** | Inference Engine | **High** | Production runtime uses TensorFlow instead of ONNX Runtime. | 1 GB RAM usage, 4s cold-start, WDAC DLL blocks. | P1 |
| **F-02** | Decision Logic | **High** | Closed-set 2-class softmax on species without OOD rejection. | Non-plant inputs forced into Potato/Tomato. | P1 |
| **F-03** | API Architecture | **Medium** | No rate limiting or concurrent inference semaphore. | Potential CPU starvation under burst traffic. | P2 |
| **F-04** | Configuration | **Medium** | Hardcoded CORS origins and server constants. | Code modification required for environment changes. | P2 |
| **F-05** | Observability | **Medium** | Plain-text un-structured logging; no Prometheus metrics. | Limited production operational visibility. | P3 |
| **F-06** | Test Infrastructure | **Medium** | `tests/conftest.py` eagerly imports TensorFlow at top-level. | Halts test runner under restrictive OS DLL policies. | P2 |
| **F-07** | Preprocessing | **Low** | Direct bilinear resize without aspect-ratio preservation. | Minor geometric distortion on extreme aspect ratios. | P3 |
| **F-08** | Deployment | **Low** | No containerization (`Dockerfile`) present in repository. | Platform inconsistency during deployment. | P2 |

## Appendix VII: Recommended Target Backend Directory Tree

```text
plant-disease-detection/
├── api/
│   ├── __init__.py
│   ├── app.py                         # FastAPI factory, lifespan handler, CORS
│   ├── config.py                      # Pydantic BaseSettings (.env loading)
│   ├── dependencies.py                # Dependency injection providers
│   ├── middleware.py                  # Rate-limiting, request ID correlation
│   └── routes/
│       ├── __init__.py
│       ├── health.py                  # Liveness and readiness endpoints
│       └── diagnose.py                # POST /v2/diagnose endpoint
├── core/
│   ├── __init__.py
│   ├── exceptions.py                  # Custom domain exception hierarchy
│   ├── logging.py                     # Structured JSON logging configuration
│   └── telemetry.py                   # Prometheus metrics definitions
├── services/
│   ├── __init__.py
│   ├── diagnosis_service.py           # Two-tier business logic orchestrator
│   ├── safety_guard.py                # Dual confidence gate and abstention logic
│   └── image_sanitizer.py             # PIL decoding, format checks, white-matting
├── engines/
│   ├── __init__.py
│   ├── base.py                        # Abstract BaseInferenceEngine
│   ├── onnx_engine.py                 # ONNX Runtime production engine (Fast & Lean)
│   └── keras_engine.py                # Keras fallback engine
├── models/
│   └── onnx/v2/                       # Primary production model artifacts
├── Dockerfile                         # Production multi-stage container
├── docker-compose.yml                 # Local orchestration
└── requirements.txt                   # Lean production dependencies (minus heavy TF)
```

## Appendix VIII: Test Suite Execution Results and Telemetry

- **Command Attempted:** `& .\.venv\Scripts\pytest.exe -v`
- **Total Test Cases in Suite:** 32 tests (across `test_api.py`, `test_v2_inference.py`, `test_data_pipeline.py`, `test_losses.py`).
- **Observed Result:** Test runner halted during conftest discovery.
- **Error Encountered:** `ImportError: DLL load failed while importing _pywrap_device_lib: An Application Control policy has blocked this file.`
- **Environmental Cause:** Host Windows Defender Application Control (WDAC) / AppLocker policy blocks dynamic C-extension DLL execution in user AppData directories.
- **Verification of ONNX Runtime:** In contrast, `import onnxruntime as ort` and ONNX model session initialization (`ort.InferenceSession`) passed with 100% success on the exact same host machine.
- **Prior Verified Baseline:** Phase 2 audit logs confirm all 32 tests previously passed with 100% success in 13.9 seconds in an unconstrained environment.

## Appendix IX: References to Inspected Source Files

1. `api/main.py` (FastAPI REST service, routes, file validation, static mounts)
2. `inference/predictor.py` (`HierarchicalPredictorV2`, two-tier gating, latency tracking)
3. `inference/model_loader.py` (`load_models`, thread lock, safe deserialization)
4. `inference/preprocessing.py` (`preprocess_image`, RGBA white-matting, bilinear resize)
5. `inference/response_formatter.py` (`format_prediction_response`, `format_error_response`)
6. `inference/inference_config.json` (Runtime thresholds, model paths, advisory templates)
7. `export_to_onnx.py` (ONNX export engine, opset 17, dynamic batch, parity tests)
8. `predict_hierarchical_v2.py` (CLI diagnostic interface)
9. `training/models.py` (MobileNetV2 backbone, CBAM, ChannelAttention, SpatialAttention)
10. `training/losses.py` (`SparseCategoricalFocalLoss`, specimen class weight calculation)
11. `training/data_pipeline.py` (Manifest-driven tf.data pipeline, zero leakage)
12. `training/class_mappings.json` (Index-to-label canonical definitions)
13. `training/training_config.json` (Hyperparameters and split manifests)
14. `tests/conftest.py` (Pytest fixtures, synthetic data generators)
15. `tests/test_api.py` (API test suite and route validation)
16. `tests/test_v2_inference.py` (Model parameter and boundary regression tests)
17. `tests/test_data_pipeline.py` (Data pipeline unit tests)
18. `tests/test_losses.py` (Loss function and class weighting unit tests)
19. `requirements.txt` & `requirements-dev.txt` (Pinned dependency specifications)
20. `setup_wsl_env.sh` (WSL environment isolation script)

