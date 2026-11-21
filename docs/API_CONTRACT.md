# LeafScan V2 — API Contract Specification
**Plant Disease Detection — Production V2**  
*Document Version:* 2.0.0  
*Base Path:* `/`  
*Protocol:* HTTP/1.1, HTTP/2  
*Content Type:* JSON (`application/json`), Multipart Form Data (`multipart/form-data`)

---

## 1. Overview

The LeafScan V2 Web API provides high-throughput, low-latency REST endpoints for high-consequence agricultural plant pathology screening in Solanaceous crops (*Solanum tuberosum* and *Solanum lycopersicum*). It connects directly to the production two-tier hierarchical deep learning engine (`HierarchicalPredictorV2`) featuring MobileNetV2, Convolutional Block Attention Module (CBAM), and dual confidence safety gates.

---

## 2. Authentication & Authorization

All endpoints in the local diagnostic service run without API key authentication in the default deployment. CORS is configured for same-origin browser access and local development interfaces (`http://127.0.0.1:8000`, `http://localhost:8000`, `http://127.0.0.1:3000`, `http://localhost:3000`).

---

## 3. Endpoints

### 3.1 System Health Check

```http
GET /health
```

#### Purpose
Lightweight probe for container liveness, readiness, and monitoring. Reports operational status and model caching telemetry without triggering heavy model deserialization if not yet cached.

#### Request Headers
| Header | Value | Description |
|---|---|---|
| `Accept` | `application/json` | Preferred response format |

#### Response (200 OK)
```json
{
  "status": "healthy",
  "model_version": "v2",
  "architecture": "MobileNetV2 + CBAM Attention",
  "models_loaded": false
}
```

#### Response Field Definitions
| Field | Type | Description |
|---|---|---|
| `status` | string | Operational state (`"healthy"`) |
| `model_version` | string | Target model suite iteration (`"v2"`) |
| `architecture` | string | Underlying network backbone specification |
| `models_loaded` | boolean | `true` if deep learning model weights are actively resident in memory |

---

### 3.2 Hierarchical Diagnostic Screening

```http
POST /v2/diagnose
```

#### Purpose
Accepts an uploaded foliar leaf photograph, executes defensive validation and resizing to \(224 \times 224 \times 3\), passes the image through Tier 1 Botanist Species classification, evaluates Safety Gate 1, routes to the appropriate Tier 2 CBAM Specialist, evaluates Safety Gate 2, and returns actionable agronomic decision-support telemetry.

#### Request Format
- **Content-Type:** `multipart/form-data`
- **Maximum Payload Size:** 10 MB (10,485,760 bytes)
- **Supported Image Formats:** JPEG (`image/jpeg`), PNG (`image/png`), WebP (`image/webp`)

#### Form Parameters
| Field Name | Type | Required | Description |
|---|---|:---:|---|
| `image` | binary file | **YES** | Raw image file of the leaf specimen |

---

#### Successful Diagnostic Response (200 OK)

##### Case A: High-Confidence Diagnosis (`confident`)
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
  "advisory": "EARLY BLIGHT DETECTED: Symptoms are characteristic of Early Blight (Alternaria solani) with high diagnostic confidence. Recommended management options: prune lower senescent leaves to minimize soil-splash inoculation; ensure proper plant spacing for canopy drying. If chemical control is warranted, consult certified agricultural extension guidelines for locally approved preventive fungicides.",
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

##### Case B: Species Uncertain Abstention (`species_uncertain` — Gate 1 Triggered)
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
  "advisory": "SPECIES UNCERTAINTY: Automated botanist confidence is below the safety threshold (<90%). Specimen disease diagnosis has been abstained to avoid crop misdiagnosis. Provide a clear, well-lit photograph showing leaf margins and venation, or submit a sample to an accredited agricultural extension laboratory. Do NOT apply chemical treatments to unverified plant species.",
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

##### Case C: Late Blight Uncertainty Escalation (`late_blight_uncertain` — Gate 2 Triggered)
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
    "execution_time_ms": 1225.97,
    "architecture": "MobileNetV2 + CBAM Attention"
  }
}
```

---

## 4. Response Field Schema & Nullability

| Field | Type | Nullable | Description |
|---|---|:---:|---|
| `success` | boolean | No | `true` if inference completed successfully |
| `model_version` | string | No | Deep learning architecture release identifier (`"v2"`) |
| `plant_species` | string | Yes | Predicted botanical species (`"Potato"` or `"Tomato"`) |
| `species_confidence`| float | No | Softmax probability for identified species \([0.0, 1.0]\) |
| `disease` | string | **Yes** | Identified pathology condition. **`null` if Gate 1 abstains.** |
| `disease_confidence`| float | **Yes** | Specialist softmax confidence. **`null` if Gate 1 abstains.** |
| `diagnosis` | string | No | Human-readable composite diagnosis label |
| `safety_status` | string | No | One of: `"confident"`, `"species_uncertain"`, `"late_blight_uncertain"`, `"disease_uncertain"`, `"error"` |
| `safety_warnings` | string[] | No | Array of active diagnostic and safety alerts |
| `advisory` | string | No | Actionable agronomic recommendation and risk mitigation instructions |
| `species_probabilities` | object | No | Key-value pairs of species softmax probabilities |
| `probabilities` | object | No | Key-value pairs of disease softmax probabilities. **Empty `{}` if Gate 1 abstains.** |
| `metadata` | object | No | Nested execution telemetry |
| `metadata.execution_time_ms` | float | Yes | CPU inference latency in milliseconds |
| `metadata.architecture` | string | No | Model architecture summary string |

---

## 5. HTTP Error Codes & Schemas

| HTTP Status | Error Type | Trigger Condition |
|---|---|---|
| `400 Bad Request` | Client Error | Empty upload payload, unreadable stream, or damaged raster image |
| `413 Payload Too Large` | Client Error | Uploaded file size exceeds the 10 MB maximum limit |
| `415 Unsupported Media Type` | Client Error | Image is not JPEG, PNG, or WebP (e.g. BMP, TIFF, GIF, PDF) |
| `422 Unprocessable Entity` | Validation Error | Missing required multipart form field `image` |
| `500 Internal Server Error` | Server Error | Unhandled neural network or inference pipeline execution exception |

#### Standard Validation Error Body (`400`, `413`, `415`, `422`)
```json
{
  "detail": "Unsupported image format 'BMP'. Allowed formats: JPEG, PNG, WEBP."
}
```

#### Structured Server Error Body (`500`)
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

## 6. Implementation Code Examples

### cURL
```bash
# Health check
curl -X GET http://127.0.0.1:8000/health

# Diagnosis screening
curl -X POST http://127.0.0.1:8000/v2/diagnose \
     -F "image=@test_images/sample_leaf.JPG"
```

### JavaScript Fetch
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

const response = await fetch('/v2/diagnose', {
  method: 'POST',
  body: formData
});

const report = await response.json();
console.log(`Diagnosis: ${report.diagnosis} (${report.safety_status})`);
```

### Python Requests
```python
import requests

url = "http://127.0.0.1:8000/v2/diagnose"
with open("test_images/sample_leaf.JPG", "rb") as f:
    response = requests.post(url, files={"image": ("sample_leaf.JPG", f, "image/jpeg")})

data = response.json()
print("Species:", data["plant_species"], f"({data['species_confidence']*100:.1f}%)")
print("Disease:", data["disease"])
print("Advisory:", data["advisory"])
```
