# LeafScan V3 — REST API Contract Specification

**Document Version:** 3.0.0  
**Status:** Active Production Contract  
**Protocol:** HTTP/1.1, HTTP/2  
**Content Types:** `application/json`, `multipart/form-data`, `text/plain`  

---

## 1. Overview & Compatibility Guarantee

The LeafScan V3 Web API exposes high-throughput, low-latency REST endpoints for foliar disease screening in Solanaceous crops (*Solanum tuberosum* - Potato, and *Solanum lycopersicum* - Tomato).

### Invariant Contract Rules
1. **100% Invariant Response Fields:** All response fields existing in V2 (`success`, `model_version`, `plant_species`, `species_confidence`, `disease`, `disease_confidence`, `diagnosis`, `safety_status`, `safety_warnings`, `advisory`, `species_probabilities`, `probabilities`, `metadata`) are preserved with exact key names, types, and rounding policies.
2. **Additive Metadata Only:** Additive response telemetry (`request_id`, `model_bundle`) is placed within the `metadata` dictionary or HTTP headers, never modifying existing key paths.
3. **Deterministic Error Envelope:** Error responses always follow the standard `{success: false, model_version: "v2", ... error: {type, message, code}}` schema.

---

## 2. Authentication & Headers

| Header | Type | Description |
|---|---|---|
| `X-Request-ID` | String (Optional) | Client-supplied tracing identifier. If omitted, the server generates a unique ID (`req_<12 hex>`) and returns it in the response header. |
| `X-API-Key` | String (Optional) | Required only when `LEAFSCAN_API_KEY_AUTH_ENABLED=true`. Authenticates integrator via SHA-256 hash comparison. |

---

## 3. Endpoint Inventory

```http
GET  /health
GET  /ready
GET  /v2/model
GET  /metrics
POST /v2/diagnose
GET  /
GET  /index.html
GET  /css/{path}
GET  /js/{path}
GET  /assets/{path}
```

---

## 4. Endpoint Specifications

### 4.1 System Liveness Check (`GET /health`)
Lightweight liveness probe for basic process reachability. Does not execute model deserialization.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_version": "v2",
  "architecture": "MobileNetV2 + CBAM Attention",
  "models_loaded": true
}
```

---

### 4.2 System Readiness Probe (`GET /ready`)
Readiness probe verifying that model sessions are warmed up, cryptographic hashes match the manifest, and the inference engine is operational.

**Response (200 OK):**
```json
{
  "status": "ready",
  "bundle_version": "v3.0.0",
  "engine": "OnnxEngine",
  "execution_provider": "CPUExecutionProvider",
  "models_verified": [
    "species_expert",
    "potato_expert",
    "tomato_expert"
  ]
}
```

**Response (503 Service Unavailable):** Returned if the model bundle failed verification or sessions are uninitialized.

---

### 4.3 Model Bundle Information (`GET /v2/model`)
Returns read-only operational telemetry, input specifications, safety gate thresholds, and evaluation metrics for the active release bundle.

**Response (200 OK):**
```json
{
  "bundle_name": "LeafScan Hierarchical Solanaceous Diagnosis Bundle",
  "bundle_version": "v3.0.0",
  "runtime": "onnxruntime",
  "opset": 17,
  "disclaimer": "This deep learning model provides automated screening assistance...",
  "models": {
    "species_expert": {
      "filename": "species_expert.onnx",
      "architecture": "MobileNetV2",
      "sha256": "eb3dd0cc145620406cdf37f3f1f7581634a67a5675cf82b04f85183305afa912",
      "classes": ["Potato", "Tomato"],
      "input_node": "input_tensor",
      "output_node": "species_output"
    },
    "potato_expert": {
      "filename": "potato_expert.onnx",
      "architecture": "MobileNetV2 + CBAM Attention",
      "sha256": "5c4d6b39fcf760c9b052c832aa37948c1ca297692f9c17de5c63b7eb208d15ef",
      "classes": ["Early Blight", "Late Blight", "Healthy"],
      "input_node": "input_tensor",
      "output_node": "disease_output"
    },
    "tomato_expert": {
      "filename": "tomato_expert.onnx",
      "architecture": "MobileNetV2 + CBAM Attention",
      "sha256": "68a08551ca9b3e25d2d9569d67350ffeda30bdfb89666e7b1313b6016ca91d45",
      "classes": ["Early Blight", "Late Blight", "Healthy"],
      "input_node": "input_tensor",
      "output_node": "disease_output"
    }
  },
  "safety_gates": {
    "species_confidence_threshold": 0.90,
    "late_blight_confidence_threshold": 0.90,
    "general_disease_confidence_threshold": 0.70
  },
  "input_spec": {
    "image_size": [224, 224],
    "channels": 3,
    "color_mode": "RGB",
    "dtype": "float32",
    "rescaling_range": [0.0, 255.0],
    "max_decoded_pixels": 40000000,
    "interpolation": "bilinear",
    "alpha_matting_background": [255, 255, 255]
  },
  "engine_info": {
    "name": "OnnxEngine",
    "provider": "CPUExecutionProvider",
    "intra_op_threads": 2,
    "inter_op_threads": 1
  },
  "metrics": {
    "species_expert": {"accuracy": 0.9801, "macro_f1": 0.9767},
    "potato_expert": {"accuracy": 0.9784, "macro_f1": 0.9673},
    "tomato_expert": {"accuracy": 0.8995, "macro_f1": 0.8870}
  }
}
```

---

### 4.4 Prometheus Metrics (`GET /metrics`)
Exposes live operational metrics formatted for Prometheus scrapers and Grafana dashboards.

**Response (200 OK, `text/plain`):**
```text
# HELP leafscan_requests_total Total HTTP requests handled
# TYPE leafscan_requests_total counter
leafscan_requests_total{endpoint="/v2/diagnose",status="200"} 42
# HELP leafscan_model_load_duration_ms Model initialization duration in milliseconds
# TYPE leafscan_model_load_duration_ms gauge
leafscan_model_load_duration_ms{bundle="v3.0.0"} 163.17
# HELP leafscan_inference_latency_ms Inference latency summary in milliseconds
# TYPE leafscan_inference_latency_ms summary
leafscan_inference_latency_ms{quantile="0.5"} 16.98
leafscan_inference_latency_ms{quantile="0.95"} 20.16
leafscan_inference_latency_ms{quantile="0.99"} 21.48
```

---

### 4.5 Hierarchical Foliar Diagnosis (`POST /v2/diagnose`)
Consumes a leaf photograph, normalizes EXIF orientation, applies white alpha-matting, executes two-tier ONNX screening, enforces safety gates, and returns diagnostic results.

**Request:**
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Field Name:** `image` (binary file)
- **Limits:** Byte size `<= 10 MB`, format in `{"JPEG", "PNG", "WEBP"}`, decoded pixels `<= 40 MP`.

#### Successful Response: Confirmed Diagnosis (200 OK)
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
    "execution_time_ms": 17.20,
    "architecture": "MobileNetV2 + CBAM Attention",
    "model_bundle": "v3.0.0",
    "request_id": "req_8a7d1f2c"
  }
}
```

#### Successful Response: Species Uncertain (200 OK, Abstained)
```json
{
  "success": true,
  "model_version": "v2",
  "plant_species": "Tomato",
  "species_confidence": 0.7245,
  "disease": null,
  "disease_confidence": null,
  "diagnosis": "Tomato (Abstained - Species Uncertain)",
  "safety_status": "species_uncertain",
  "safety_warnings": [
    "Species confidence (72.45%) is below the safety threshold (90%). Disease classification abstained to prevent crop misdiagnosis."
  ],
  "advisory": "SPECIES UNCERTAINTY: Automated botanist confidence is below the safety threshold (<90%). Specimen disease diagnosis has been abstained to avoid crop misdiagnosis. Provide a clear, well-lit photograph showing leaf margins and venation, or submit a sample to an accredited agricultural extension laboratory. Do NOT apply chemical treatments to unverified plant species.",
  "species_probabilities": {
    "Potato": 0.2755,
    "Tomato": 0.7245
  },
  "probabilities": {},
  "metadata": {
    "execution_time_ms": 9.45,
    "architecture": "MobileNetV2 + CBAM Attention",
    "model_bundle": "v3.0.0",
    "request_id": "req_f82e3b1a"
  }
}
```

#### Successful Response: Late Blight Uncertainty (200 OK, High-Consequence Warning)
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
    "execution_time_ms": 18.52,
    "architecture": "MobileNetV2 + CBAM Attention",
    "model_bundle": "v3.0.0",
    "request_id": "req_a1c2e3f4"
  }
}
```

---

## 5. Machine-Readable Error Codes & HTTP Status

| HTTP Status | Machine Error Code | Trigger Condition |
|---|---|---|
| `400 Bad Request` | `invalid_image` | Missing file, empty upload (0 bytes), or corrupt image raster bytes. |
| `400 Bad Request` | `image_too_large_pixels` | Decoded image exceeds 40,000,000 pixels (decompression bomb protection). |
| `413 Content Too Large` | `file_too_large` | Upload stream exceeds 10 MB payload limit. |
| `415 Unsupported Media Type` | `unsupported_format` | Format is not JPEG, PNG, or WebP. |
| `422 Unprocessable Content` | `poor_image_quality` | Image fails blur or lighting quality checks (when enabled). |
| `429 Too Many Requests` | `rate_limit_exceeded` | Client IP exceeds configured requests per minute (default: 60/min). Contains `Retry-After`. |
| `503 Service Unavailable` | `overloaded` | Server inference concurrency is saturated. Contains `Retry-After: 1`. |
| `503 Service Unavailable` | `model_unavailable` | Model bundle is corrupt, unverified, or uninitialized. |
| `500 Internal Server Error` | `inference_error` | Unhandled internal exception masked safely without path exposure. |
