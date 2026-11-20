# Legacy V1 Architecture Archive (April 2026)

## 1. Overview and Archival Purpose

This directory contains the archived legacy **V1** deep learning assets for the Plant Disease Detection project. 
These files have been preserved for historical provenance, auditability, and regression benchmarking.

> [!WARNING]
> **DEPRECATION NOTICE:**  
> The V1 models and inference pipeline are **deprecated and unsupported for production use**.  
> All active operational workflows and microservices must target the **V2 Hierarchical Architecture** via `predict_hierarchical_v2.py` and model checkpoints located in `models/v2/`.

---

## 2. Archived Assets Inventory

| Asset Name | Original Location | Description | SHA-256 Checksum |
|---|---|---|---|
| `predict_hierarchical.py` | `predict_hierarchical.py` | Legacy CLI inference script with single-gate thresholding | `d46025969eac798378da050092a6395ffb967125158c9d48c14c2e9c7e4eb1a1` |
| `species_expert.keras` | `models/species_expert.keras` | Legacy Species classifier (Potato vs. Tomato) | `4afbb0581c149641d84433467ae601928a156f67e389452a63917ea5d7b935b5` |
| `potato_expert.keras` | `models/potato_expert.keras` | Legacy Potato disease specialist (No attention mechanism) | `d6823bca076a729c0a052e73635d1ef2c787d8c9de071f3b081ea3ca1bcf4e60` |
| `tomato_expert.keras` | `models/tomato_expert.keras` | Legacy Tomato disease specialist (No attention mechanism) | `bf053fe0ae9d7ec20d3a8a7571d8530a7678c03b28a46565bffcc222dfae7036` |

---

## 3. Why V1 was Superseded by V2

The technical audit conducted in November 2026 revealed several critical vulnerabilities in the V1 pipeline:

1. **Catastrophic Data Leakage in Evaluation:**  
   V1 evaluation claimed 98.6% accuracy based on naive random image splitting. Forensic hashing revealed that 66.88% of the dataset consisted of synthetic augmentations and duplicate files. Under random splitting, 86.3% of biological source specimens crossed train/validation boundaries, invalidating reported performance metrics.
2. **Absence of Attention Mechanisms:**  
   V1 specialists used standard MobileNetV2 without spatial or channel feature refinement, leading to poor morphological discrimination between early Alternaria lesions and Late Blight.
3. **Fragile Deserialization Hook:**  
   V1 models suffered from a Keras 3 `quantization_config` deserialization bug requiring runtime monkey-patching of `keras.layers.Dense.from_config`.
4. **Single-Gate Fragility:**  
   V1 used a flat threshold without botanical abstention or specialized Late Blight safety gates.
5. **Pre-processing Vulnerability:**  
   V1 did not handle non-RGB image inputs (e.g. RGBA with transparency, CMYK, or Grayscale).

---

## 4. Production V2 Architecture Comparison

| Feature | Legacy V1 (Archived) | Production V2 (Active) |
|---|---|---|
| **Location** | `archive/v1/` | `models/v2/` & `inference/` |
| **Backbone** | MobileNetV2 | MobileNetV2 + CBAM Attention |
| **Attention Mechanism** | None | 206,258-param Channel + Spatial CBAM |
| **Dataset Splitting** | Naive random split (86.3% leakage) | Leak-free DSU biological specimen group split (0.0% leakage) |
| **Evaluation Base** | Leaked test set | 603 strictly held-out biological specimens |
| **Safety Gates** | Flat 90% threshold | Dual Gate: Species Abstention (<90%) & Late Blight Warning (<90%) |
| **Preprocessing** | Basic PIL RGB conversion | Alpha compositing on white, CMYK, Grayscale, NumPy |
| **Deployment** | Python script only | Python CLI + Optimized ONNX graphs (`models/onnx/v2/`) |

---

## 5. Migration Instructions

To run production inference on leaf images, execute:

```bash
python predict_hierarchical_v2.py --image test_images/sample_leaf.JPG --json
```

For ONNX microservice export:

```bash
python export_to_onnx.py
```
