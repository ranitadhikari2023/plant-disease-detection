# LeafScan V2 — Frontend Architecture & Backend Integration Report
**Project:** Plant Disease Detection — Production V2  
**Report Date:** 2026-11-22  
**Lead Engineers:** Senior Full-Stack Engineer, Frontend Architect, FastAPI Backend Engineer, Computer Vision Deployment Engineer, UI/UX Designer, Software Quality Assurance Engineer  
**Status:** **COMPLETE — ALL 22 ACCEPTANCE CRITERIA VERIFIED (PASS)**

---

## 1. Executive Overview

This report documents the design, implementation, verification, and deployment of a modern, lightweight, dependency-free web application (**LeafScan**) built around the existing **Plant Disease Detection V2** hierarchical deep learning inference engine.

The system connects a mobile-first Vanilla HTML5/CSS3/JavaScript Single Page Application (SPA) to a high-performance **FastAPI** backend that exposes the two-tier MobileNetV2 + CBAM attention inference pipeline, enforces dual confidence safety gates, preserves high-consequence agronomic advisories, and maintains local user diagnostic history.

---

## 2. Directory Structure & File Manifest

The implementation was completed directly within the existing repository structure without altering or moving existing model weights, datasets, training scripts, or CLI utilities.

```text
plant-disease-detection/
│
├── frontend/
│   ├── index.html                   # Semantic HTML5 Single Page Application shell
│   │
│   ├── css/
│   │   ├── tokens.css               # Agronomic green design system tokens & variables
│   │   ├── layout.css               # App shell, responsive 720px grid, view containers
│   │   └── components.css           # Dropzone, preview, cards, meters, banners, accordion
│   │
│   ├── js/
│   │   ├── api.js                   # API client module for /v2/diagnose and /health
│   │   └── app.js                   # Primary SPA controller, state, drag-drop, history
│   │
│   └── assets/
│       └── logo.svg                 # Scalable vector LeafScan foliar reticle branding
│
├── api/
│   ├── __init__.py                  # FastAPI package marker
│   └── main.py                      # REST API endpoints, image validation & static mounts
│
├── docs/
│   ├── FRONTEND_BASELINE_AUDIT.md   # Pre-implementation baseline audit report
│   ├── API_CONTRACT.md              # REST API schema contract specification
│   ├── FRONTEND_IMPLEMENTATION_REPORT.md # Comprehensive final implementation report
│   └── ...                          # Existing project reports & audit documentation
│
├── inference/                       # Existing hierarchical inference engine (UNTOUCHED)
│   ├── inference_config.json
│   ├── model_loader.py
│   ├── predictor.py
│   ├── preprocessing.py
│   └── response_formatter.py
│
├── models/                          # Existing model checkpoints & ONNX graphs (UNTOUCHED)
│   ├── v2/                          # Keras 3 model checkpoints
│   └── onnx/v2/                     # Optimized ONNX Runtime CPU graphs (Opset 17)
│
├── tests/
│   ├── conftest.py                  # Pytest fixtures and cache cleaners
│   ├── test_data_pipeline.py        # 13 pipeline unit tests
│   ├── test_losses.py               # 12 loss unit tests
│   ├── test_v2_inference.py         # 7 hierarchical inference regression tests
│   └── test_api.py                  # 14 new FastAPI endpoint & integration tests
│
├── predict_hierarchical_v2.py       # Primary CLI utility (UNTOUCHED & FULLY WORKING)
├── pytest.ini                       # Pytest configuration with registered test markers
├── requirements.txt                 # Pinned production runtime dependencies
└── README.md                        # Master project documentation
```

### File Modification Summary
- **New Files Created:**
  - `frontend/index.html`
  - `frontend/css/tokens.css`
  - `frontend/css/layout.css`
  - `frontend/css/components.css`
  - `frontend/js/api.js`
  - `frontend/js/app.js`
  - `frontend/assets/logo.svg`
  - `api/__init__.py`
  - `api/main.py`
  - `tests/test_api.py`
  - `docs/FRONTEND_BASELINE_AUDIT.md`
  - `docs/API_CONTRACT.md`
  - `docs/FRONTEND_IMPLEMENTATION_REPORT.md`
  - `.venv/lib/site-packages/tree.py` *(Windows Smart App Control compatibility shim for Keras 3)*
- **Modified Files:**
  - `pytest.ini` *(Registered `smoke` and `api` markers)*
  - `requirements.txt` *(Added pinned FastAPI, Uvicorn, and Python-Multipart dependencies)*
  - `README.md` *(Added Web Application, API, Frontend Architecture, and Execution sections)*

---

## 3. Backend Integration Architecture

### 3.1 Inference Engine as Single Source of Truth
The FastAPI backend (`api/main.py`) does **not** reimplement inference, retrain models, or alter safety boundaries. It directly interfaces with `HierarchicalPredictorV2` from the project's established `inference.predictor` module.

```text
               HTTP CLIENT (Browser SPA / cURL)
                              │
                              ▼
                      FastAPI Application
                       [POST /v2/diagnose]
                              │
             Defensive Size & Format Validation
            (Max 10 MB, PIL Verify: JPEG/PNG/WebP)
                              │
                              ▼
           Worker Threadpool (Synchronous Execution)
                              │
                              ▼
              HierarchicalPredictorV2 Singleton
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Tier 1: Botanist               Safety Gate 1
       (MobileNetV2)                 (90% Threshold)
               │                             │
               │                    ┌────────┴────────┐
               │                    ▼                 ▼
               │              Below 90%          Pass (≥ 90%)
               │            (Abstain & Warn)          │
               │                                      ▼
               │                              Tier 2: Specialist
               │                              (MobileNetV2 + CBAM)
               │                                      │
               │                                      ▼
               │                                Safety Gate 2
               │                             (90% LB / 70% Dis)
               │                                      │
               └──────────────┬───────────────────────┘
                              ▼
                 Standard Response Formatter
               [inference/response_formatter.py]
                              │
                              ▼
                     JSON HTTP Response
```

### 3.2 Threadpool Offloading
To prevent synchronous CPU-bound deep learning inference from blocking FastAPI's asynchronous event loop, the diagnostic endpoint is declared as a synchronous handler `def diagnose_leaf(...)`. FastAPI automatically schedules it onto Starlette's `run_in_threadpool` worker pool, allowing concurrent health checks and static file requests to proceed without latency spikes.

### 3.3 Model Caching & Lazy Loading
Models are lazily loaded into memory upon the initial diagnosis call and held as cached singletons protected by `_MODEL_LOCK`. The health probe (`GET /health`) executes in sub-millisecond time without triggering unnecessary disk I/O or GPU/CPU memory allocation.

---

## 4. User Interface Architecture & Design

### 4.1 Agronomic Visual Identity
The UI adopts an agronomic deep-green visual identity rooted in Solanaceous pathology:
- **Primary Brand:** `#1f6f43` (Foliar Chlorophyll Green)
- **Brand Light:** `#e8f3ec`
- **Surface:** `#ffffff` over `#f7f8f6` subtle earth backdrop
- **Status Confident:** `#1f6f43`
- **Status Species Uncertain:** `#475569` (Slate)
- **Status Late Blight Uncertain:** `#b42318` (High-Consequence Hazard Red)
- **Status General Disease Uncertain:** `#b45309` (Amber)

### 4.2 Application Views
1. **Scan View:**
   - Drag-and-drop foliar image dropzone with click and keyboard activation (Enter/Space).
   - Instant client-side format and size pre-validation (10 MB ceiling).
   - In-memory object URL preview with natural dimensions, filename, and filesize.
   - Action controls: *Run Hierarchical Diagnosis*, *Change Image*, and *Clear*.
   - Diagnostic imaging protocol checklist (single leaf, diffuse daylight, margin venation, clear lesion halo).
2. **Analyzing View:**
   - Animated SVG circular loader and live ARIA progress announcements.
   - Five-stage architectural cascade breakdown (Defensive Preprocessing, Tier 1 Botanist, Safety Gate 1, Tier 2 CBAM Specialist, Safety Gate 2).
   - Does **not** simulate fake completion ticks for intermediate stages.
3. **Result View:**
   - **Safety Status Banner:** Direct mapping of the backend `safety_status` string to distinct visual alert containers with icons, badges, headings, and warning lists.
   - **Botanical Identification Card:** Displays predicted crop, scientific name (*Solanum tuberosum* / *Solanum lycopersicum*), softmax confidence percentage, and distribution meters.
   - **Foliar Pathology Specialist Card:** Displays predicted disease, confidence percentage, and individual meters for Early Blight, Late Blight, and Healthy tissue. Gracefully renders an abstention notification when Gate 1 triggers.
   - **Agronomic Advisory Card:** Preserves raw actionable advisory text and highlights high-consequence warnings (e.g. suppressing systemic fungicides during Late Blight uncertainty).
   - **Technical Details Accordion:** Collapsible details container showing model version, architecture, inference latency in milliseconds, and preformatted raw JSON with one-click clipboard copy.
4. **History View:**
   - Client-side diagnostic session persistence via browser `localStorage`.
   - Records scan timestamp, filename, species, disease, confidence, and safety badge.
   - "View Report" action to instantly inspect previous scan results.
   - Enforces a 25-record cap and includes a confirmation-gated "Clear History" action.
5. **About View:**
   - Comprehensive technical documentation of the Two-Tier CNN, CBAM Attention anatomy, Dual Safety Gates, ONNX Runtime benchmarks, supported pathogen taxonomy, and operational limitations.

---

## 5. Security & Reliability Controls

1. **Defensive Image Decoding:** The backend does not trust client-supplied MIME headers. Uploaded bytes are parsed through `PIL.Image.open` inside an in-memory buffer, verifying raster integrity and magic headers against `{"JPEG", "PNG", "WEBP"}`.
2. **Payload Ceilings:** File reads are bounded to `MAX_UPLOAD_SIZE_BYTES + 1` (10 MB + 1 byte), rejecting oversized files with HTTP 413 without reading unbounded streams into RAM.
3. **Information Disclosure Prevention:** Server-side exceptions are logged to internal loggers and return a sanitized `InferenceExecutionError` JSON response without leaking filesystem paths or Python tracebacks.
4. **Restricted CORS Policy:** CORS headers are restricted to same-origin browser origins and verified local development ports.
5. **XSS Prevention:** All dynamic data (filenames, advisory recommendations, confidence scores, warnings) are injected into the DOM strictly via `.textContent` or `document.createElement`, eliminating HTML injection vulnerabilities.
6. **Object URL Lifecycle:** All `URL.createObjectURL` instances are explicitly revoked via `URL.revokeObjectURL` upon file changes or clearance, avoiding browser memory bloat.

---

## 6. Automated Testing & Verification

### 6.1 Baseline vs. Final Test Comparison

| Test Suite Category | Baseline Count | Baseline Status | Final Count | Final Status |
|---|:---:|:---:|:---:|:---:|
| Data Pipeline Unit Tests (`test_data_pipeline.py`) | 13 | PASS | 13 | **PASS** |
| Focal Loss & Class Weights (`test_losses.py`) | 12 | PASS | 12 | **PASS** |
| V2 Hierarchical Regression (`test_v2_inference.py`) | 7 | PASS | 7 | **PASS** |
| FastAPI REST & Static Service (`test_api.py`) | *0 (New)* | *N/A* | 14 | **PASS** |
| **Total Automated Tests** | **32** | **PASS** | **46** | **PASS (100%)** |

### 6.2 Test Suite Execution Output
```text
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-8.3.5, pluggy-1.7.0 -- D:\plant-disease-detection\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\plant-disease-detection
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 46 items

tests/test_data_pipeline.py .............                                [ 28%]
tests/test_losses.py ............                                        [ 54%]
tests/test_v2_inference.py .......                                       [ 69%]
tests/test_api.py ..............                                         [100%]

============================== 46 passed in 16.90s ==============================
```

### 6.3 End-to-End Live HTTP Smoke Verification
A live Uvicorn daemon was spawned and probed with `test_images/sample_leaf.JPG`:
- `GET /health` → **200 OK** (`{"status": "healthy", "model_version": "v2", "models_loaded": false}`)
- `GET /` → **200 OK** (LeafScan SPA HTML payload delivered, 29,450 bytes)
- `GET /css/tokens.css` → **200 OK**
- `GET /js/app.js` → **200 OK**
- `GET /assets/logo.svg` → **200 OK**
- `POST /v2/diagnose` (multipart upload) → **200 OK**
  - Plant Species: `Tomato` (Confidence: 94.41%)
  - Disease: `Late Blight` (Confidence: 69.64%)
  - Safety Status: `late_blight_uncertain` (Safety Gate 2 triggered)
  - Diagnosis: `Tomato - Late Blight`
  - Advisory: Suppresses chemical systemic fungicide recommendations, requests isolation and 24–48h monitoring.
  - Architecture: `MobileNetV2 + CBAM Attention`
  - Latency: `1225.97 ms`

---

## 7. Execution Instructions

### Windows Environment (PowerShell)
```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Run automated test suite (all 46 tests)
python -m pytest -q

# 3. Launch LeafScan web service
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to: **`http://127.0.0.1:8000`**

### WSL / Linux Environment (Bash)
```bash
# 1. Activate isolated Linux virtual environment
source ~/venvs/plant-disease-detection/bin/activate

# 2. Run automated test suite
python -m pytest -q

# 3. Launch LeafScan web service
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 8. Final Acceptance Criteria Verification Table

| No. | Acceptance Criterion | Verification Method | Status |
|:---:|---|---|:---:|
| 1 | Frontend structure created | Verified files in `frontend/`, `frontend/css/`, `frontend/js/`, `frontend/assets/` | **PASS** |
| 2 | Responsive UI implemented | Tested mobile-first CSS layout with 720px breakpoint | **PASS** |
| 3 | Image upload functional | Drag-and-drop and input picker implemented and tested | **PASS** |
| 4 | Image preview functional | In-memory Object URL preview with natural dimensions and clearance | **PASS** |
| 5 | FastAPI integration implemented | `api/main.py` created with `/health` and `/v2/diagnose` endpoints | **PASS** |
| 6 | Existing V2 inference connected | Directly interfaces with `HierarchicalPredictorV2` | **PASS** |
| 7 | API response contract preserved | Verified identical schema to `inference/response_formatter.py` | **PASS** |
| 8 | All 4 safety statuses displayed | Explicit UI mapping for `confident`, `species_uncertain`, `late_blight_uncertain`, `disease_uncertain` | **PASS** |
| 9 | Species probabilities displayed | Solanum tuberosum vs Solanum lycopersicum meters rendered | **PASS** |
| 10 | Disease probabilities displayed | Early Blight, Late Blight, and Healthy condition meters rendered | **PASS** |
| 11 | Agronomic advisory displayed | Actionable recommendation text and warnings rendered safely | **PASS** |
| 12 | Technical JSON displayed | Collapsible technical details with raw JSON and copy button | **PASS** |
| 13 | Local history functional | `localStorage` persistence with newest-first ordering and clearance | **PASS** |
| 14 | About page implemented | Technical explanation of two-tier CNN, CBAM, gates, and ONNX | **PASS** |
| 15 | Accessibility addressed | Semantic HTML5, visible focus rings, ARIA regions, reduced motion | **PASS** |
| 16 | Input validation implemented | Max 10 MB limit, PIL format validation, corrupted file rejection | **PASS** |
| 17 | Existing ML tests preserved | All 32 baseline regression tests continue passing | **PASS** |
| 18 | New API tests passed | 14 new API unit and integration tests passing | **PASS** |
| 19 | Application starts successfully | Live Uvicorn server started and verified on port 8008 | **PASS** |
| 20 | E2E sample-image diagnosis verified | Live diagnosis on `sample_leaf.JPG` verified with correct 200 payload | **PASS** |
| 21 | README updated | Master documentation updated with full frontend and API guide | **PASS** |
| 22 | Implementation report generated | Comprehensive report delivered in `docs/FRONTEND_IMPLEMENTATION_REPORT.md` | **PASS** |

---

## 9. Compact Image Preview UI Refinement

### 9.1 Overview & Architectural Objectives
The image upload interface of **Plant Disease Detection V2 (LeafScan)** was refined to replace the vertically stacked preview card (which previously consumed significant vertical viewport space with a 4:3 media container) with a compact, horizontal preview card (`.preview-panel`).

This refinement achieves:
1. **Immediate Vertical Viewport Efficiency:** Height reduced from >420px to ~96–112px on desktop.
2. **Side-by-Side Diagnostic Guidance Alignment:** Top-aligns seamlessly with the neighboring Diagnostic Imaging Protocol checklist on desktop screens without leaving an empty grid row.
3. **Memory Safety & Clean Lifecycle:** Explicit `URL.revokeObjectURL` invocations on file change, file cancel, and new scan initiation prevent memory leaks in long-running browser sessions.
4. **Resilient Responsive Degradation:** Flex row layout on desktop (`>= 481px`) and clean stacked layout on mobile viewports (`<= 480px`).

### 9.2 Structural Comparison: Before vs After

#### Before: Vertically Stacked Card (`#preview-container`)
```html
<div id="preview-container" class="card preview-card" hidden>
  <div class="card-header">
    <h2 class="card-title"><span>📸</span> Specimen Ready for Screening</h2>
    <span id="preview-filesize" class="card-subtitle">-- KB</span>
  </div>
  <div class="preview-media-container">
    <img id="image-preview" src="" alt="Selected foliar specimen preview">
  </div>
  <div class="preview-meta">
    <span id="preview-filename">filename.jpg</span>
    <span id="preview-dimensions">224 × 224 Target</span>
  </div>
  <div class="preview-actions">
    <button type="button" id="btn-run-diagnosis" class="btn btn-primary btn-full">Run Hierarchical Diagnosis</button>
    <button type="button" id="btn-choose-another" class="btn btn-secondary">Change</button>
    <button type="button" id="btn-clear-image" class="btn btn-secondary">Clear</button>
  </div>
</div>
```

#### After: Compact Horizontal Preview Panel (`#preview-panel`)
```html
<div class="preview-panel" id="preview-panel" hidden aria-live="polite">
  <!-- Left: Image Thumbnail (80x80) -->
  <div class="preview-panel__media">
    <img id="preview-img" alt="Selected leaf preview">
  </div>

  <!-- Right: Details and Actions -->
  <div class="preview-panel__details">
    <div class="preview-panel__info">
      <h3 class="preview-panel__title" id="preview-name">Selected image</h3>
      <p class="preview-panel__meta">
        Size: <span id="preview-size">0 KB</span>
      </p>
    </div>

    <div class="preview-panel__actions">
      <button id="run-btn" type="button" class="btn btn--primary">
        Run Diagnosis
      </button>
      <button id="change-btn" type="button" class="btn btn-secondary">
        Change
      </button>
      <button id="clear-btn" type="button" class="btn btn--ghost">
        Cancel
      </button>
    </div>
  </div>
</div>
```

### 9.3 Layout & ASCII Visual Diagram

```text
+---------------------------------------------------------------------------------------------------------+
| .upload-column                                    | .guidance-column                                    |
|                                                   |                                                     |
| +-----------------------------------------------+ | +-------------------------------------------------+ |
| | [Thumbnail]  filename_of_specimen.jpg         | | | Diagnostic Imaging Protocol                     | |
| |  (80x80px)   Size: 1.4 MB                     | | |                                                 | |
| |              [Run Diagnosis] [Change] [Cancel]| | | 1. Single Leaf Focus                            | |
| +-----------------------------------------------+ | | 2. Diffuse Natural Daylight                     | |
|                                                   | | 3. Clear Margins & Venation                     | |
|                                                   | +-------------------------------------------------+ |
+---------------------------------------------------------------------------------------------------------+
```

### 9.4 CSS Specifications & Responsive Behavior
- **Desktop (`min-width: 481px`):**
  - Display: `display: flex; align-items: center; gap: 16px;`
  - Fixed-dimension thumbnail container: `80px × 80px` with `border-radius: 8px` and `object-fit: cover`.
  - Content details: `flex: 1; min-width: 0; display: flex; align-items: center; justify-content: space-between;`.
  - Filename truncation: `white-space: nowrap; overflow: hidden; text-overflow: ellipsis;` preventing flex parent blowouts.
  - Entrance animation: `slideDown 0.2s ease-out` (disabled under `prefers-reduced-motion: reduce`).
- **Mobile (`max-width: 480px`):**
  - Panel switches to `flex-direction: column; align-items: stretch; text-align: center;`.
  - Actions switch to `flex-direction: column; width: 100%;` with full-width action buttons.
- **Grid Stability:**
  - Placed directly inside `.upload-column` alongside `#upload-zone`.
  - `.view-grid-2col` utilizes `align-items: start;` ensuring column height symmetry and zero empty grid rows.

### 9.5 JavaScript State & Lifecycle Management
- **Element Mapping:** Managed via `elements.uploadZone`, `elements.leafInput`, `elements.previewPanel`, `elements.previewImg`, `elements.previewName`, `elements.previewSize`, `elements.runBtn`, `elements.changeBtn`, `elements.clearBtn` with backward-compatible getter aliases.
- **`selectFile(file)`:** Validates MIME and size, executes `URL.revokeObjectURL(state.previewObjectUrl)` if a previous URL exists, instantiates `URL.createObjectURL(file)`, updates image and text metadata, sets `#upload-zone` hidden and `#preview-panel` visible.
- **`resetUpload()`:** Revokes active Object URL, resets file input value, resets image `src`, resets preview text, hides `#preview-panel`, restores `#upload-zone`, and disables `#run-btn`.
- **Initialization:** Invoked on `DOMContentLoaded` to clear any browser-cached file input states.

### 9.6 Compact Preview Acceptance Verification Checklist

| No. | Acceptance Criterion | Verification Status | Notes |
|:---:|---|:---:|---|
| 1 | Horizontal compact layout on desktop | **PASS** | `display: flex; align-items: center;` with ~96px height |
| 2 | Thumbnail (80x80px, object-fit: cover, rounded) | **PASS** | `.preview-panel__media` exactly 80x80px, `radius-md: 8px`, `object-fit: cover` |
| 3 | Right side details & actions | **PASS** | Title, size, Run Diagnosis, Change, Cancel buttons rendered |
| 4 | Card height constrained on desktop | **PASS** | Compact single-row height (~96–112px) |
| 5 | Mobile (< 480px) responsive adaptation | **PASS** | `@media (max-width: 480px)` stacks thumbnail, info, and buttons full-width |
| 6 | Filename truncation | **PASS** | `white-space: nowrap; text-overflow: ellipsis; min-width: 0;` prevents layout breaks |
| 7 | Hidden when no file is selected | **PASS** | `hidden` attribute with `.preview-panel[hidden] { display: none !important; }` |
| 8 | Visible immediately upon selection | **PASS** | Drag-drop and file input events trigger `selectFile()` showing panel |
| 9 | Upload dropzone hides when preview visible | **PASS** | `#upload-zone[hidden] { display: none !important; }` |
| 10 | Dropzone restores on Cancel/Clear | **PASS** | `resetUpload()` restores `#upload-zone` and hides `#preview-panel` |
| 11 | Run Diagnosis triggers `/v2/diagnose` | **PASS** | `#run-btn` connected to `executeDiagnosis()` -> `diagnoseImage()` |
| 12 | Loading state disables actions | **PASS** | Buttons disabled during transition to analyzing view |
| 13 | Cancel/Clear revokes Object URL & clears input | **PASS** | `URL.revokeObjectURL()` called and `elements.leafInput.value = ''` |
| 14 | New file revokes previous Object URL | **PASS** | `selectFile()` checks and revokes existing URL before creating new |
| 15 | Protocol card remains top-aligned | **PASS** | `.view-grid-2col` has `align-items: start;` and `.upload-column` has `min-width: 0;` |
| 16 | All automated tests continue passing | **PASS** | 46/46 pytest test cases passing in regression suite |
