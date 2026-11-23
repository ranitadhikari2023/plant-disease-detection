# LeafScan Model Bundle Specification (V3)

**Specification Version:** 3.0.0  
**Status:** Active Production Specification  
**Default Location:** `models/bundles/v3/`  

---

## 1. Specification Overview

In LeafScan V3, a model release is an **atomic, immutable logical unit**. Neural network graph artifacts must never be decoupled from their corresponding input tensor dimensions, class label mappings, cryptographic checksums, safety thresholds, or calibration parameters.

A valid LeafScan Model Bundle consists of a dedicated directory containing:
1. `species_expert.onnx` — Botanist species classifier (2 classes: Potato, Tomato).
2. `potato_expert.onnx` — Potato specialist with CBAM attention (3 classes).
3. `tomato_expert.onnx` — Tomato specialist with CBAM attention (3 classes).
4. `manifest.json` — Cryptographic specification and metadata manifest.

---

## 2. Directory Structure

```text
models/bundles/v3/
├── species_expert.onnx     # 9,247,293 bytes (~8.82 MB)
├── potato_expert.onnx      # 10,406,402 bytes (~9.92 MB)
├── tomato_expert.onnx      # 10,406,602 bytes (~9.92 MB)
└── manifest.json           # JSON Schema v3.0.0
```

---

## 3. Cryptographic Manifest Schema (`manifest.json`)

The manifest is defined by Pydantic V2 models in `src/leafscan/bundle/manifest.py`.

```json
{
  "bundle_version": "v3.0.0",
  "created_at": "2026-11-23T20:30:00Z",
  "name": "LeafScan Hierarchical Solanaceous Diagnosis Bundle",
  "description": "Production ONNX models with CBAM attention and two-tier safety gating for Potato and Tomato pathology.",
  "runtime": "onnxruntime",
  "opset": 17,
  "disclaimer": "This deep learning model provides automated screening assistance for educational and decision-support purposes only. It does not replace professional on-site plant pathology diagnosis. Always consult certified agricultural extension officers and strictly follow locally registered pesticide label instructions prior to chemical applications.",
  "models": {
    "species_expert": {
      "filename": "species_expert.onnx",
      "sha256": "eb3dd0cc145620406cdf37f3f1f7581634a67a5675cf82b04f85183305afa912",
      "architecture": "MobileNetV2",
      "num_classes": 2,
      "input_name": "input_tensor",
      "input_shape": [null, 224, 224, 3],
      "output_name": "species_output",
      "output_shape": [null, 2],
      "index_to_label": {
        "0": "Potato",
        "1": "Tomato"
      },
      "label_to_index": {
        "Potato": 0,
        "Tomato": 1
      }
    },
    "potato_expert": {
      "filename": "potato_expert.onnx",
      "sha256": "5c4d6b39fcf760c9b052c832aa37948c1ca297692f9c17de5c63b7eb208d15ef",
      "architecture": "MobileNetV2 + CBAM Attention",
      "num_classes": 3,
      "input_name": "input_tensor",
      "input_shape": [null, 224, 224, 3],
      "output_name": "disease_output",
      "output_shape": [null, 3],
      "index_to_label": {
        "0": "Early Blight",
        "1": "Late Blight",
        "2": "Healthy"
      },
      "label_to_index": {
        "Early Blight": 0,
        "Late Blight": 1,
        "Healthy": 2
      }
    },
    "tomato_expert": {
      "filename": "tomato_expert.onnx",
      "sha256": "68a08551ca9b3e25d2d9569d67350ffeda30bdfb89666e7b1313b6016ca91d45",
      "architecture": "MobileNetV2 + CBAM Attention",
      "num_classes": 3,
      "input_name": "input_tensor",
      "input_shape": [null, 224, 224, 3],
      "output_name": "disease_output",
      "output_shape": [null, 3],
      "index_to_label": {
        "0": "Early Blight",
        "1": "Late Blight",
        "2": "Healthy"
      },
      "label_to_index": {
        "Early Blight": 0,
        "Late Blight": 1,
        "Healthy": 2
      }
    }
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
  "safety_gates": {
    "species_confidence_threshold": 0.90,
    "late_blight_confidence_threshold": 0.90,
    "general_disease_confidence_threshold": 0.70
  },
  "advisories": {
    "species_uncertain": "SPECIES UNCERTAINTY: Automated botanist confidence is below the safety threshold...",
    "late_blight_uncertain": "LATE BLIGHT UNCERTAINTY: Folio symptoms indicate potential early lesions...",
    "late_blight_confirmed": "LATE BLIGHT DETECTED: Symptoms are characteristic of Late Blight...",
    "early_blight_confirmed": "EARLY BLIGHT DETECTED: Symptoms are characteristic of Early Blight...",
    "healthy_confirmed": "NO FOLIAR BLIGHT DETECTED: The specimen displays healthy foliar tissue...",
    "general_disease_uncertain": "MARGINAL DIAGNOSTIC CONFIDENCE: Classification confidence is below 70%...",
    "out_of_scope": "OUT OF SCOPE SPECIMEN: The input leaf could not be reliably verified as Potato or Tomato..."
  },
  "calibration": {
    "species_temperature": 0.80,
    "potato_temperature": 1.00,
    "tomato_temperature": 1.10
  },
  "metrics": {
    "species_expert": {"accuracy": 0.9801, "macro_f1": 0.9767},
    "potato_expert": {"accuracy": 0.9784, "macro_f1": 0.9673},
    "tomato_expert": {"accuracy": 0.8995, "macro_f1": 0.8870}
  }
}
```

---

## 4. Verification & Loading Rules

1. **Mandatory Hash Verification:**
   Before an ONNX session can be created, `src/leafscan/bundle/verification.py` computes the streaming SHA-256 digest of each `.onnx` file. If a file is missing or its hash does not match `manifest.json`, the loader raises `ChecksumMismatchError`.
2. **Fail-Safe Startup:**
   Under no circumstances may an alternative unverified model be loaded silently. If verification fails, the application marks `app.state.bundle = None` and the readiness endpoint (`GET /ready`) returns `HTTP 503`.
3. **Immutability:**
   Model bundle files are mounted read-only (`ro`) in production containers.
