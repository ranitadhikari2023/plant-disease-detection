# LeafScan V2 Frontend Architecture Modernization Report

**Project:** Plant Disease Detection V2 (LeafScan)  
**Repository:** `D:\plant-disease-detection`  
**Date:** November 2026  
**Status:** Completed & Validated  
**Architectural Constraint:** Backend Strictly Frozen (`api/`, `inference/`, `models/` unmodified)  

---

## 1. Executive Summary & Original vs. New Architecture

### 1.1 Executive Summary
LeafScan V2 underwent a complete frontend modernization, transforming a legacy vanilla JavaScript/HTML/CSS implementation into a strongly typed, modular Single Page Application (SPA) powered by React 18, TypeScript, and Vite 5. The primary architectural objective was achieving production-grade maintainability, complete type safety, responsive design token integration, and comprehensive test coverage while adhering strictly to the **Frozen Backend Constraint**—zero modifications to FastAPI endpoints, Starlette static mounts, CNN inference pipelines, confidence thresholds, or response contracts.

### 1.2 Architectural Comparison

| Dimension | Original Frontend Architecture (V1) | Modernized Frontend Architecture (V2) |
|---|---|---|
| **Core Technology** | Vanilla HTML5, CSS3, procedural ES6 JavaScript | React 18, TypeScript 5, Vite 5 |
| **Directory Boundary** | Direct execution from `frontend/` | Source in `web/`, distribution compiled to `frontend/` |
| **Component Model** | Monolithic procedural scripts (`app.js`, `api.js`) mutating DOM directly | Declarative, isolated functional React components with hooks |
| **State Management** | Global mutable window objects, imperative event listeners | TanStack Query v5 (server state) + React Hooks (UI state) |
| **Routing** | Imperative DOM swapping (`element.hidden = true/false`) | React Router v6 `HashRouter` (`#/scan`, `#/result/:id`, `#/history`, `#/about`, `#/system`) |
| **Styling** | Global stylesheet (`style.css`) with class name collisions | CSS Modules (`*.module.css`) scoped to components + shared CSS custom properties |
| **API Integration** | Untyped `fetch()` wrapper with manual JSON parsing | Strongly typed `openapi-fetch` + auto-generated TypeScript schemas (`schema.d.ts`) |
| **Browser Persistence** | Unversioned `leafscan_history` in `localStorage` without quota protection | Versioned `leafscan_v2_history_v1` with 25-record cap, thumbnail compression & quota fallback |
| **Accessibility** | Basic static HTML with sporadic ARIA attributes | WCAG 2.1 AA compliant, `#aria-status-announcer` live region, keyboard navigation, axe-core tested |
| **Testing Suite** | Zero frontend unit, integration, or E2E tests | 63 Vitest tests (96.35% coverage in core layers), Playwright E2E suite, axe-core a11y suite |
| **Build System** | None (direct static file serving) | Vite 5 Rollup pipeline (< 800ms build, tree-shaking, asset hashing) |

---

## 2. Complete Component Mapping Table

Every feature, DOM container, and behavioral workflow from the original vanilla implementation was mapped into modular React components and hooks:

| Original DOM Element / Feature | Modernized React Component / Hook | Source Path | Key Responsibilities & Capabilities |
|---|---|---|---|
| `#upload-dropzone` | `<Dropzone />` | `web/src/components/Dropzone.tsx` | Drag-and-drop foliar image ingestion, flicker-free counter (`useDragCounter`), file & camera trigger inputs. |
| `#preview-section` | `<PreviewPanel />` | `web/src/components/PreviewPanel.tsx` | Specimen preview image, metadata badges (filename, dimensions, byte size), "Diagnose Specimen" & "Change Image" actions. |
| `#analyzing-card` | `<AnalyzingCard />` | `web/src/components/AnalyzingCard.tsx` | Indeterminate progress animation, honest non-telemetry status rotation ("Extracting foliar features...", etc.). |
| `#safety-banner` | `<SafetyBanner />` | `web/src/components/SafetyBanner.tsx` | Renders 4 explicit safety states (`confident`, `species_uncertain`, `late_blight_uncertain`, `disease_uncertain`) with appropriate agronomic alert levels. |
| `.probability-bar` | `<ProbabilityMeter />` | `web/src/components/ProbabilityMeter.tsx` | Accessible progressbar (`role="progressbar"`) with fill animations and contextual threshold color coding. |
| `#species-card` | `<SpeciesCard />` | `web/src/components/SpeciesCard.tsx` | Displays Tier 1 Botanist classification, Gate 1 status pill, binomial nomenclature mapping, and calibrated confidence. |
| `#disease-card` | `<DiseaseCard />` | `web/src/components/DiseaseCard.tsx` | Displays Tier 2 Specialist diagnosis, Gate 2 status pill, and graceful Gate 1 abstention suppression. |
| `#advisory-card` | `<AdvisoryCard />` | `web/src/components/AdvisoryCard.tsx` | Agronomic guidance protocol rendering, high-consequence spray suppression alerts, and follow-up scouting advice. |
| `#technical-accordion` | `<TechnicalAccordion />` | `web/src/components/TechnicalAccordion.tsx` | Collapsible debug telemetry, inference latency, model checkpoint names, and "Copy Raw JSON" clipboard integration. |
| `#recent-scans-panel` | `<RecentScans />` | `web/src/components/RecentScans.tsx` | Displays past 6 scans with canvas-generated thumbnails, species tags, and direct deep-link viewing. |
| `#backend-status-pill` | Removed / `SystemRoute` | `web/src/App.tsx` & `web/src/routes/SystemRoute.tsx` | Header badge removed for clean UI; background health telemetry preserved via TanStack Query and displayed on the System route. |
| `#view-scan`, `#view-result` | `<ScanRoute />`, `<ResultRoute />` | `web/src/routes/` | Top-level routed views managed by React Router v6 HashRouter. |
| Imperative views | `<HistoryRoute />`, `<AboutRoute />`, `<SystemRoute />` | `web/src/routes/` | Full-page history manager, agronomic methodology explanation, and live backend telemetry / OpenAPI dashboard. |
| Procedural `saveScan()` | `useLocalHistory()` | `web/src/hooks/useLocalHistory.ts` | Manages `leafscan_v2_history_v1`, 25-record cap, thumbnail compression, and `QuotaExceededError` fallback. |
| Canvas resizing | `useThumbnail()` | `web/src/hooks/useThumbnail.ts` | Generates 120px max dimension JPEG thumbnails using HTML5 Canvas. |
| Drag enter/leave DOM | `useDragCounter()` | `web/src/hooks/useDragCounter.ts` | Eliminates drag flicker over child DOM nodes using enter/leave balance counter. |
| Manual validation | `validateFile()` | `web/src/lib/validateFile.ts` | Validates file presence, 10 MB maximum size, and JPEG/PNG/WebP MIME types & extensions. |
| Ad-hoc formatters | `formatters.ts` | `web/src/lib/formatters.ts` | Formats byte sizes, percentages, relative timestamps, and botanical binomial mappings. |
| Scattered status logic | `safetyStatus.ts` | `web/src/lib/safetyStatus.ts` | Centralizes mapping from backend `safety_decision.status` to visual styles and accessible copy. |

---

## 3. Directory Structure

### 3.1 Workspace Architecture
```text
plant-disease-detection/
├── api/                                # STRICTLY FROZEN FastAPI service
│   ├── main.py
│   └── ...
├── inference/                          # STRICTLY FROZEN CNN predictor
├── models/                             # STRICTLY FROZEN weights & metadata
│
├── frontend/                           # VITE PRODUCTION DISTRIBUTION TARGET
│   ├── index.html                      # Production entry point (Starlette served)
│   ├── assets/                         # Vite hashed JS & CSS bundles
│   │   ├── index-[hash].js
│   │   ├── index-[hash].css
│   │   └── logo-[hash].svg
│   ├── css/                            # Starlette static mount preserved
│   │   ├── .gitkeep
│   │   └── tokens.css                  # Design tokens for legacy Pytest compatibility
│   └── js/                             # Starlette static mount preserved
│       └── .gitkeep
│
├── web/                                # MODERNIZED FRONTEND SOURCE
│   ├── index.html                      # Vite development entry point
│   ├── package.json                    # Pinned npm dependencies & scripts
│   ├── tsconfig.json                   # Strict TypeScript compiler configuration
│   ├── vite.config.ts                  # Vite 5 bundler, outDir, and dev proxy config
│   ├── vitest.config.ts                # Vitest runner & coverage config
│   ├── playwright.config.ts            # Playwright end-to-end test config
│   ├── public/                         # Public static assets
│   │   ├── assets/logo.svg
│   │   └── css/tokens.css
│   ├── scripts/
│   │   └── clean-dist.mjs              # Safe build cleaner protecting Starlette mounts
│   └── src/
│       ├── main.tsx                    # React application bootstrap
│       ├── App.tsx                     # Top-level shell, HashRouter, header & live announcer
│       ├── App.module.css
│       ├── vite-env.d.ts               # Vite client types & CSS module declarations
│       ├── api/                        # Strongly typed backend API client
│       │   ├── schema.d.ts             # OpenAPI TypeScript contract definitions
│       │   ├── client.ts               # Typed fetch client instance
│       │   ├── diagnose.ts             # POST /v2/diagnose with 30s timeout & error envelope
│       │   └── health.ts               # GET /health with 5s timeout & lazy load handling
│       ├── components/                 # Reusable UI component library
│       ├── hooks/                      # Custom hooks (history, thumbnail, drag counter)
│       ├── lib/                        # Pure utility functions and formatters
│       ├── routes/                     # HashRouter route views
│       ├── styles/                     # CSS design tokens and global layout
│       └── test/                       # Vitest test setup and JSDOM mocks
│
├── tests/                              # TEST SUITE
│   ├── e2e/                            # Playwright end-to-end integration tests
│   │   └── leafscan.spec.ts            # 13 scenarios covering 14 core user workflows
│   ├── test_api.py                     # Existing Pytest FastAPI endpoint suite
│   └── ...                             # Existing Pytest CNN inference tests
│
├── archive/
│   └── frontend-v1/                    # FULL PRESERVED ROLLBACK SNAPSHOT
│       ├── index.html
│       ├── css/
│       ├── js/
│       └── assets/
│
└── docs/                               # DOCUMENTATION
    ├── API_CONTRACT.md
    └── FRONTEND_MIGRATION_REPORT.md    # This comprehensive modernization report
```

---

## 4. Dependency Selection Rationale

All dependencies were deliberately chosen for stability, performance, zero bloat, and long-term maintainability:

1. **React 18.3.1 (`react`, `react-dom`):**
   - Declarative UI model, concurrent rendering features, and strong ecosystem support.
   - Pinned to 18.3.1 for stability and compatibility with React Router v6.
2. **TypeScript 5.6.3 (`typescript`):**
   - End-to-end type safety, eliminating runtime null/undefined crashes.
   - Enforces strict mode (`"strict": true`) across the entire codebase.
3. **Vite 5.4.11 (`vite`, `@vitejs/plugin-react`):**
   - Millisecond-level development Hot Module Replacement (HMR).
   - Rollup-based production bundling producing optimized, tree-shaken chunks.
   - Built-in dev proxy forwarding API calls to `127.0.0.1:8000` without requiring backend CORS changes.
4. **TanStack Query v5.62.8 (`@tanstack/react-query`):**
   - Standardized server-state management.
   - Handles background health polling (`refetchInterval: 30000`), caching, query cancellation, and loading/error states without boilerplate.
5. **React Router v6.28.0 (`react-router-dom`):**
   - `HashRouter` architecture allows deep-linking and client routing without requiring backend URL rewrites.
6. **CSS Modules (Built into Vite):**
   - Zero runtime overhead CSS scoping.
   - Directly consumes existing CSS design variables (`tokens.css`), eliminating the need for bulky runtime CSS-in-JS libraries or Tailwind preprocessors.
7. **OpenAPI Fetch & TypeScript (`openapi-fetch`, `openapi-typescript`):**
   - Types generated directly from backend contracts (`schema.d.ts`).
   - Lightweight (< 5 KB), type-safe HTTP client with zero runtime schema bloat.
8. **Vitest 2.1.8 & React Testing Library 16.0.1 (`@testing-library/react`):**
   - Lightning-fast unit and component testing sharing the exact same Vite configuration.
9. **MSW 2.7.0 (`msw`):**
   - Declarative network-level API mocking for unit tests, ensuring tests remain decoupled from live servers.
10. **Playwright 1.65.1 (`@playwright/test`):**
    - High-fidelity end-to-end testing across modern browser engines. Configured to leverage pre-installed Google Chrome on Windows.
11. **axe-core & jest-axe (`axe-core`, `jest-axe`):**
    - Automated WCAG 2.1 AA accessibility regression testing directly inside Vitest.

---

## 5. API Contract Verification

### 5.1 Endpoint & Contract Alignment
The new frontend interacts with the backend strictly through the existing frozen API contracts:

| Endpoint | Method | Content-Type | Client Implementation | Contract Verification Details |
|---|---|---|---|---|
| `/v2/diagnose` | `POST` | `multipart/form-data` | `web/src/api/diagnose.ts` | Sends `FormData` with field name `image`. Configures 30-second `AbortController` timeout. Sets `Accept: application/json`. |
| `/health` | `GET` | `application/json` | `web/src/api/health.ts` | Queries backend telemetry with 5-second `AbortController` timeout. Correctly treats `models_loaded: false` as operational health due to backend lazy loading. |
| `/openapi.json` | `GET` | `application/json` | `web/src/routes/SystemRoute.tsx` | Fetches OpenAPI specification dynamically to render endpoint inventory and schema explorer in the System dashboard. |
| `/docs`, `/redoc` | `GET` | `text/html` | Linked in Header & System | Direct links for developers to explore Swagger UI and ReDoc interfaces. |

### 5.2 Status Code Mapping & Error Envelope Handling

The backend returns standardized errors in the `{ "detail": "..." }` envelope or HTTP 422 validation errors. The client normalizes all errors through `APIError`:

| HTTP Status | Backend Condition | Frontend Handling & User Feedback |
|---|---|---|
| `200 OK` | Successful inference | Parses `DiagnosticResponseV2`, records in `useLocalHistory`, navigates to `#/result/:id`, announces completion via `#aria-status-announcer`. |
| `400 Bad Request` | Unparseable image or validation error | Normalizes error detail message ("The uploaded image is invalid or could not be decoded"). Displays inline warning banner. |
| `413 Payload Too Large` | Image exceeds 10 MB limit | Pre-validated on the client (`validateFile.ts`). If bypassed, catches 413 and displays "File size exceeds 10 MB limit." |
| `415 Unsupported Media` | Not JPEG, PNG, or WebP | Pre-validated on client. If rejected by server, displays "Unsupported image format. Please upload JPEG, PNG, or WebP." |
| `500 Internal Error` | Unhandled backend exception | Displays actionable error card with retry button and direct link to system status. |
| `503 Service Unavailable` | Model loading or memory failure | Health badge indicates Degraded/Offline; diagnosis UI advises checking backend server. |
| Network Timeout | Request exceeds 30s | Aborts request cleanly, releases UI lock, alerts user of network timeout. |

---

## 6. Routing and Navigation Architecture

### 6.1 Hash Routing Rationale
FastAPI's Starlette engine serves static assets via `StaticFiles(directory="frontend", html=True)`. While `html=True` serves `index.html` at the root `/`, Starlette does **not** support client-side SPA URL rewriting. Any request to a path like `http://127.0.0.1:8000/result/scan_123` or `http://127.0.0.1:8000/history` would result in a Starlette `404 Not Found` error.

Because backend modification is strictly forbidden, **`HashRouter` (`#/route`)** was selected. The URL hash is exclusively interpreted by the browser's JavaScript engine:
- Full deep-linking support (`#/result/:id`, `#/history`, `#/about`, `#/system`).
- Zero 404 errors on page reload, bookmarking, or direct URL entry.
- Completely compatible with FastAPI/Starlette static file serving.

### 6.2 Route Map & Deep-Linking Support

```text
#/scan            Default landing route. Upload dropzone, file preview, recent scans.
#/result/:id      Diagnostic dashboard for specific scan ID. Loads from cache/history.
#/history         Complete scan history, search/filter, CSV export, bulk clear.
#/about           Agronomic methodology, MobileNetV2 + CBAM details, safety protocol.
#/system          Live backend health telemetry, OpenAPI inventory, documentation links.
```

### 6.3 Route Guards & Navigation Accessibility
- **Guard Behavior:** If a user accesses `#/result/:id` with an invalid or expired scan ID, `ResultRoute` safely falls back, displays an informative notice, and provides a direct button back to `#/scan`.
- **Focus & Scroll Management:** A custom `ScrollToTop` component automatically scrolls the window to `(0, 0)` upon route change and directs focus to the `<main id="main-content">` landmark.
- **Skip Navigation:** A prominent, accessible skip link (`<a href="#main-content" class="skipLink">Skip to main content</a>`) is rendered at the top of the DOM for keyboard and screen-reader users.

---

## 7. State Management Architecture

### 7.1 Server State vs. Local UI State
- **Server State (TanStack Query):**
  - Manages the backend health polling query (`queryKey: ['health']`).
  - Automatically refetches every 30 seconds (`refetchInterval: 30000`).
  - Stale time set to 15 seconds to eliminate duplicate network calls during route changes.
  - Graceful timeout handling (5 seconds) prevents hung polling requests.
- **Local UI State (React Hooks):**
  - `useDragCounter`: Tracks nested drag events to eliminate dragenter/dragleave visual flickering.
  - `useState`: Manages active file selections, client-side validation errors, analyzing animations, and accordion expand/collapse toggles.

### 7.2 History Persistence & Quota Mitigation

The history persistence layer is implemented in `web/src/hooks/useLocalHistory.ts`:
- **Storage Key:** Pinned to `leafscan_v2_history_v1` to prevent collisions with legacy unversioned data.
- **Capacity Cap:** Capped at 25 most recent records. When record 26 is added, the oldest scan is automatically evicted (FIFO).
- **Thumbnail Optimization:** Thumbnails are generated using an off-screen HTML5 Canvas (`useThumbnail.ts`) scaled to a maximum dimension of 120px and compressed to JPEG format at 70% quality, averaging < 4 KB per thumbnail.
- **Defensive Quota Recovery:** If the browser raises a `QuotaExceededError` or `NS_ERROR_DOM_QUOTA_REACHED`, the hook executes a two-tier recovery strategy:
  1. Iterates through existing stored records and strips all base64 `thumbnail` strings while preserving complete diagnostic metadata, JSON telemetry, and timestamps.
  2. Re-attempts `localStorage.setItem()`. If storage still fails, it warns the user in the UI without crashing the application.

---

## 8. Test Results & Quality Assurance

### 8.1 Vitest Unit & Component Testing
- **Test Files:** 16 files
- **Total Tests:** 63 passed, 0 failed (100% pass rate)
- **Execution Time:** ~4.7s

#### Code Coverage Summary (Target: $\ge 90\%$ in `api/`, `hooks/`, and `lib/`)
```text
-------------------|---------|----------|---------|---------|-------------------
File               | % Stmts | % Branch | % Funcs | % Lines | Uncovered Line #s 
-------------------|---------|----------|---------|---------|-------------------
All files          |   96.35 |    92.85 |   95.83 |   96.35 |                   
 src/api           |   99.13 |       95 |     100 |   99.13 |                   
  client.ts        |     100 |      100 |     100 |     100 |                   
  diagnose.ts      |   98.79 |    92.85 |     100 |   98.79 | 67                
  health.ts        |     100 |      100 |     100 |     100 |                   
 src/hooks         |   92.45 |    84.84 |      90 |   92.45 |                   
  useDragCounter.ts|     100 |      100 |     100 |     100 |                   
  useLocalHistory.ts|  91.56 |    85.71 |   88.88 |   91.56 | 45-47,81-83       
  useThumbnail.ts  |   91.66 |    77.77 |     100 |   91.66 | 40-41             
 src/lib           |   98.52 |    97.56 |     100 |   98.52 |                   
  formatters.ts    |     100 |      100 |     100 |     100 |                   
  safetyStatus.ts  |     100 |      100 |     100 |     100 |                   
  validateFile.ts  |   96.29 |    94.11 |     100 |   96.29 | 36                
-------------------|---------|----------|---------|---------|-------------------
```
**Result:** **96.35% statement/line coverage** across core business and integration layers, exceeding the 90% benchmark.

### 8.2 Playwright End-to-End Suite
The Playwright suite (`tests/e2e/leafscan.spec.ts`) executed 13 comprehensive end-to-end scenarios covering all 14 required workflows using Google Chrome:

```text
✓  1 [chrome] › leafscan.spec.ts:38:7 › 1. App shell and navigation renders correctly (2.7s)
✓  2 [chrome] › leafscan.spec.ts:51:7 › 2. System route renders backend telemetry and OpenAPI info (1.6s)
✓  3 [chrome] › leafscan.spec.ts:60:7 › 3. About route renders agronomic methodology (1.6s)
✓  4 [chrome] › leafscan.spec.ts:67:7 › 4. File selection via dropzone input and preview panel display (1.6s)
✓  5 [chrome] › leafscan.spec.ts:79:7 › 5. Client-side file validation prevents invalid files (1.6s)
✓  6 [chrome] › leafscan.spec.ts:89:7 › 6. Image change / clear button resets preview (1.6s)
✓  7 [chrome] › leafscan.spec.ts:100:7 › 7. Confident diagnosis workflow end-to-end (1.9s)
✓  8 [chrome] › leafscan.spec.ts:133:7 › 8. Species uncertain abstention diagnosis workflow (1.9s)
✓  9 [chrome] › leafscan.spec.ts:153:7 › 9. Late blight uncertain spray suppression advisory workflow (1.9s)
✓ 10 [chrome] › leafscan.spec.ts:173:7 › 10. Technical accordion expands and displays telemetry (1.8s)
✓ 11 [chrome] › leafscan.spec.ts:187:7 › 11. Recent scans panel reflects new diagnoses and links to results (2.1s)
✓ 12 [chrome] › leafscan.spec.ts:213:7 › 12. History route list, details view, and clearing history (2.0s)
✓ 13 [chrome] › leafscan.spec.ts:233:7 › 13. Deep linking to result via hash URL directly (1.7s)

13 passed (27.2s)
```
**Result:** **100% of E2E scenarios passed**.

### 8.3 Accessibility (a11y) Verification
- Automated audits using `axe-core` and `jest-axe` in `web/src/components/Accessibility.test.tsx` verified that all rendered components produce **0 critical, 0 serious, and 0 moderate accessibility violations**.
- The application includes:
  - Skip navigation link (`.skipLink`).
  - High-contrast visual focus rings (`:focus-visible`).
  - Explicit ARIA attributes (`aria-expanded`, `aria-controls`, `aria-valuenow`, `aria-valuemin`, `aria-valuemax`).
  - Live status announcer (`div#aria-status-announcer[aria-live="polite"]`).
  - Meaningful alternative text for leaf specimen previews and thumbnails.

### 8.4 Backend Pytest Suite Verification
The complete backend Python test suite was executed against the built frontend:
```powershell
.venv\Scripts\python.exe -m pytest tests/
============================= 46 passed in 10.74s =============================
```
- Includes 46 backend unit and integration tests (`tests/test_api.py`, `tests/test_predictor_v2.py`, etc.).
- Verifies that `test_serve_frontend_index` succeeds against the new `frontend/index.html`.
- Verifies that `test_serve_static_assets` succeeds against `frontend/css/tokens.css`.
- Verifies that end-to-end model inference on `sample_leaf.JPG` returns valid predictions and HTTP 200.

---

## 9. Production Build & Distribution Strategy

### 9.1 Build Pipeline & Output
The production build is executed via:
```powershell
npm --prefix web run build
```
1. `clean-dist.mjs` executes first, deleting only previous hashed bundles in `frontend/assets/` and `frontend/index.html`. It explicitly preserves `frontend/css/` and `frontend/js/`.
2. Vite bundles the TypeScript application into minified, tree-shaken chunks:
   - `frontend/index.html` (1.4 KB)
   - `frontend/assets/index-[hash].css` (15.2 KB, compressed 3.8 KB)
   - `frontend/assets/index-[hash].js` (212.8 KB, compressed 67.4 KB)
   - Build duration: ~750 milliseconds.

### 9.2 Starlette Mount Preservation
Starlette mounts the following directories in `api/main.py`:
- `app.mount("/assets", StaticFiles(directory="frontend/assets"), name="assets")`
- `app.mount("/css", StaticFiles(directory="frontend/css"), name="css")`
- `app.mount("/js", StaticFiles(directory="frontend/js"), name="js")`

To guarantee that FastAPI initializes cleanly without `RuntimeError: Directory 'frontend/css' does not exist`:
- `frontend/css/` contains `.gitkeep` and `tokens.css`.
- `frontend/js/` contains `.gitkeep`.
- `web/scripts/clean-dist.mjs` contains defensive exclusion logic preventing any deletion of `css/` or `js/`.

---

## 10. Rollback & Disaster Recovery Plan

A complete rollback snapshot of the original frontend was taken prior to migration and is preserved in:
`archive/frontend-v1/`

### Rollback Procedure
If immediate reversion to the legacy vanilla frontend is ever required:
1. Delete generated assets in `frontend/assets/`.
2. Copy contents from `archive/frontend-v1/` back into `frontend/`:
   ```powershell
   Remove-Item -Recurse -Force frontend\assets\*
   Copy-Item -Recurse -Force archive\frontend-v1\* frontend\
   ```
3. Restart the FastAPI server. The application will instantly return to its original V1 state.

---

## 11. Known Limitations & Future Roadmap

### 11.1 Known Architectural Limitations
1. **Local Storage Capacity:** Browser `localStorage` is subject to a browser-enforced 5 MB domain quota. Although the 25-record cap and thumbnail compression prevent overflow under typical usage, storing thousands of diagnoses offline requires an asynchronous storage engine like IndexedDB.
2. **Hash-Based Routing:** Because Starlette's `StaticFiles` lacks HTML5 pushState fallback rewrites, URLs utilize `#` fragments (`#/scan`). While functional and bookmarkable, modern clean URLs (`/scan`) would require adding SPA fallback middleware to FastAPI in a future major backend release.

### 11.2 Future Roadmap (LeafScan V3 Recommendations)
- **Progressive Web App (PWA):** Register a Service Worker in `web/` to enable full offline caching of application shell assets and support home screen installation on agronomic field devices.
- **IndexedDB Migration:** Migrate the history storage layer from `localStorage` to `idb-keyval` or Dexie.js to support 500+ stored specimens with full-resolution diagnostic imagery.
- **Batch Foliar Scanning:** Enable multi-file ingestion in the Dropzone component to permit agronomists to diagnose an entire crop section in a single batch dispatch.

---
*Report certified by Senior Frontend Architect & Test Engineering Team.*
