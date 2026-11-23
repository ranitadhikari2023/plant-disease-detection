# LeafScan V3 Security & Threat Model

**Project:** `plant-disease-detection`  
**System Architecture:** LeafScan V3  
**Classification:** Agricultural Critical Infrastructure (Diagnostic Screening)  
**Security Status:** Production Hardened  
**Date:** November 23, 2026  
**DevSecOps Specialist:** Google DeepMind & Antigravity Engineering  

---

## 1. Executive Security Architecture

LeafScan V3 provides foliar disease screening for Solanaceous food staples (Potato and Tomato). Because agricultural diagnostics directly inform decisions regarding chemical pesticide applications and crop quarantine, systemic integrity, defensive intake sanitization, and deterministic decision authority are non-negotiable requirements.

### Primary Security Objectives:
1. **Zero Path and Internal Detail Disclosure:** Eliminate filesystem paths, stack traces, and internal configuration from API client responses.
2. **Defensive Resource Bounding:** Protect serving infrastructure from decompression bombs, oversized uploads, and denial-of-service via concurrent inference exhaustion.
3. **Cryptographic Model Integrity:** Prevent unauthorized model substitution through SHA-256 manifest verification.
4. **Minimal Attack Surface:** Exclude heavyweight deep learning training frameworks (TensorFlow, Keras, CUDA) from production containers, reducing CVE exposure by over 70%.
5. **Non-Root Container Hardening:** Execute inside containers with an unprivileged non-root user (UID 10001) and drop all Linux capabilities.

---

## 2. STRIDE Threat Analysis & Defensive Controls

```mermaid
flowchart TD
    subgraph Threats ["STRIDE Threat Categories"]
        S["Spoofing"]
        T["Tampering"]
        R["Repudiation"]
        I["Information Disclosure"]
        D["Denial of Service"]
        E["Elevation of Privilege"]
    end

    subgraph Defenses ["LeafScan V3 Defensive Controls"]
        D_S["Hashed API Key Verification & Trusted-Proxy IP Extraction"]
        D_T["SHA-256 Bundle Manifest Verification & Read-Only Artifacts"]
        D_R["Structured JSON Logging & X-Request-ID Header Propagation"]
        D_I["Strict Exception Masking & Zero Image Persistence"]
        D_D["10MB Payload Gate, 40MP Decompression Guard, Semaphore Concurrency"]
        D_E["Non-Root Container (UID 10001) & No New Privileges"]
    end

    S --> D_S
    T --> D_T
    R --> D_R
    I --> D_I
    D --> D_D
    E --> D_E
```

| STRIDE Category | Specific Threat Vector | Risk Level | LeafScan V3 Defensive Control |
|---|---|---|---|
| **Spoofing** | Unauthorized integrators impersonating registered clients; spoofed `X-Forwarded-For` headers. | Medium | Optional hashed API key validation (`X-API-Key`) comparing against SHA-256 digests. Trusted proxy validation rules documented for reverse proxies. |
| **Tampering** | Malicious alteration of neural network weights or class mappings to manipulate diagnoses. | Critical | Model Bundle Architecture (`models/bundles/v3`): startup verification computes streaming SHA-256 checksums of all ONNX files and halts process on mismatch (`ChecksumMismatchError`). |
| **Repudiation** | Client or operator disputing an advisory or diagnosis issued by the system. | Medium | Unique request ID (`X-Request-ID`) generated per request and propagated across logs, metrics, and response metadata. Deterministic safety guard rules recorded. |
| **Information Disclosure** | Stack traces, internal filesystem paths, or Python environment variables leaked via error payloads. | High | Centralized `ExceptionMaskingMiddleware` intercepts all exceptions and emits a sanitized JSON error envelope `{success: false, error: {type, message, code}}`. Internal traces logged to server stdout only. |
| **Denial of Service** | Decompression bombs (e.g. 100MP image compressed into 1MB); resource exhaustion via concurrent inference. | High | 1. Maximum upload stream byte check (`10 MB`).<br/>2. Pillow `MAX_IMAGE_PIXELS = 40_000_000` (40 Megapixels) rejecting decompression bombs.<br/>3. `BoundedConcurrencyMiddleware` sheds excess requests with `HTTP 503` and `Retry-After: 1`.<br/>4. In-memory sliding window IP rate limiter (`60 req/min`). |
| **Elevation of Privilege** | Container breakout or local privilege escalation via vulnerable dependencies. | Critical | Container runs as unprivileged user `leafscan:leafscan` (UID 10001), `no-new-privileges:true`, zero package managers in runtime stage, and read-only root compatibility. |

---

## 3. Defense-in-Depth Implementation Details

### 3.1 Defensive Image Intake Pipeline (`ImageIntake`)
1. **Byte Length Guard:** Before decoding, the byte stream is validated against `MAX_UPLOAD_SIZE_BYTES` (10 MB). Empty uploads (0 bytes) are rejected immediately.
2. **Format Whitelisting:** Enforces explicit format filtering via Pillow inspection: only `JPEG`, `PNG`, and `WEBP` are accepted. Unknown or executable extensions are rejected with `HTTP 415`.
3. **Decoded Dimension Guard:** Image dimensions `width * height` are inspected immediately upon header decode. If pixels exceed 40 Megapixels (approx. 40,000,000 pixels), `PixelLimitExceededError` is raised before allocating raster buffers.
4. **EXIF Normalization:** `ImageOps.exif_transpose` corrects mobile camera rotation tags, ensuring orientational alignment without introducing black margins.
5. **Alpha-Matting Sanitization:** RGBA images with transparency channels are composited over an opaque white background `(255, 255, 255)`. This eliminates transparent black artifacts and prevents steganographic attacks hidden in unused alpha channels.
6. **In-Memory Volatility:** Images are processed strictly in RAM as detached PIL copies. No temporary files are written to the host filesystem.

### 3.2 Bounded Concurrency & Load Shedding
Because deep learning inference on CPU involves multi-threaded matrix multiplication, admitting unbounded concurrent requests causes thread thrashing, memory ballooning, and server starvation.
- **Asyncio Semaphore:** Configured to `max_concurrent_inferences` (default: 4 concurrent calls per container).
- **Predictable Load Shedding:** When all semaphore slots are active, incoming requests are shed immediately with `HTTP 503 Service Unavailable`, a `Retry-After: 1` header, and error code `overloaded`.
- **IP Rate Limiting:** A sliding-window token bucket enforces a maximum of 60 requests per minute per IP, rejecting rapid flooding with `HTTP 429 Too Many Requests`.

### 3.3 Container Supply Chain & Vulnerability Elimination
- **Zero TensorFlow in Serving Image:** Removing TensorFlow 2.21 from the production image eliminates 350+ transitive C/C++ libraries, CUDA bindings, and reduces the image attack surface from >2.5 GB to < 400 MB.
- **Non-Root Execution:**
  ```dockerfile
  RUN groupadd -g 10001 leafscan && useradd -u 10001 -g leafscan -s /usr/sbin/nologin -M leafscan
  USER leafscan:leafscan
  ```
- **Automated CI Security Scans:**
  - `pip-audit`: Scans pinned production packages against the Python Packaging Advisory Database.
  - `trivy`: Scans OS and library packages for Critical and High CVEs.
  - `syft`: Generates machine-readable Software Bill of Materials (SPDX-JSON).

---

## 4. Model Safety Authority & Deterministic Isolation

No machine learning model, LLM, or external agent is permitted to override the deterministic `SafetyGuard`.
1. **Safety Gate 1:** If species confidence is below 0.9000, the system abstains from issuing a disease diagnosis (`disease = null`, `safety_status = "species_uncertain"`).
2. **Safety Gate 2:** If Late Blight is detected but confidence is below 0.9000, diagnosis is flagged as `late_blight_uncertain`, and systemic fungicide suppression warnings are deterministically bound to the payload.
3. **Threshold Floor:** Confidence boundaries are externalized in `manifest.json` and cryptographically locked.
