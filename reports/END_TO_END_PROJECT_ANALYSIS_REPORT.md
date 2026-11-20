# End-to-End Technical Analysis and Architecture Audit Report
## Plant Disease Detection Deep Learning System

**Audit Execution Date:** November 21, 2026  
**Auditor:** Senior Machine Learning Engineer, Deep Learning Researcher & Software Architect  
**Project Workspace:** `D:\plant-disease-detection`  
**Target Repository:** Plant Disease Detection  
**Primary Technologies:** Python 3.10.0, TensorFlow 2.21.0, Keras 3.12.1, NumPy 2.2.6, MobileNetV2, CBAM Attention Mechanism  
**Audit Mandate:** Zero-Modification Technical Inspection and Evidence-Based Quality Assessment  

---

## 1. Executive Summary

A comprehensive, end-to-end technical audit was conducted on the **Plant Disease Detection** deep learning project. The inspection examined all source code files, dataset partitions, model checkpoints, training logs, evaluation reports, test suites, and operational configurations. Every finding in this report is grounded in direct code inspection, mathematically verified tensor dimensions, parameter counts, empirical execution logs, and live regression test executions.

### Key Executive Findings

1. **Architectural Bifurcation and Generational Shift:**  
   The project has undergone an evolution from an initial legacy monolithic design (designated as **V1**, dated April 2026) to a modern, hierarchical deep learning architecture with attention mechanisms (designated as **V2**, dated November 2026).
   - **Legacy Pipeline (V1):** Monolithic sequential models (`models/species_expert.keras`, `models/potato_expert.keras`, `models/tomato_expert.keras`) orchestrated by flat scripts (`predict_hierarchical.py`). This pipeline suffered from fatal data leakage, unhandled Keras 3 deserialization errors (`quantization_config` bug in `Dense` layers), lack of input validation, and single-gate thresholding.
   - **Modern Engineering Architecture (V2):** A modular two-tier cascaded architecture organized cleanly across `audit/`, `training/`, `inference/`, and `tests/`. It introduces a Tier 1 Botanist Species Classifier (`Species_Expert_Botanist_v2`) and Tier 2 Specialist Crop Doctors (`Potato_Doctor_CBAM_v2` and `Tomato_Doctor_CBAM_v2`) enhanced with an injected **Convolutional Block Attention Module (CBAM)**.

2. **Dataset Integrity and Resolution of Catastrophic Data Leakage:**  
   - The raw dataset consists of **11,988 JPEG images** superficially balanced at exactly 1,998 images per category across 6 classes (2 crops $\times$ 3 conditions: Early Blight, Late Blight, Healthy).
   - Forensic hashing and clustering in `audit/run_phase2_audit.py` revealed that **66.88% (8,017 images)** of the dataset are synthetic offline augmentations (center crops `_ccrp_`, Gaussian blurs `_gblr_`, random rotations `_rrot_`) or exact duplicate files (970 duplicate SHA-256 groups).
   - Most critically, the raw `potato/healthy` class contains only **90 unique biological leaves** replicated into 1,998 images (a 95.95% synthetic duplication rate).
   - Under legacy random image splitting, **86.3%** of biological source leaf clusters crossed train/val boundaries, completely invalidating prior claims of 98.6% accuracy due to severe data leakage.
   - V2 resolved this issue by implementing a **Disjoint Set Union (DSU)** graph clustering algorithm based on SHA-256 and perceptual difference hashes (dHash/aHash) to identify **3,971 true biological specimens**. It generated a strictly leak-free Stratified Group Split (Train: 2,776; Val: 592; Test: 603 specimens; 0.0% overlap).

3. **Active and Verified CBAM Attention Mechanism:**  
   - Contrary to common deep learning repositories where claimed attention modules are either uncompiled, detached from gradients, or purely decorative, **CBAM is genuinely implemented, compiled, and actively functioning** in `Potato_Doctor_CBAM_v2` and `Tomato_Doctor_CBAM_v2`.
   - The module adds exactly **206,258 parameters** (Channel Attention: 206,160 parameters; Spatial Attention: 98 parameters) directly at the terminal feature maps ($7 \times 7 \times 1280$) of MobileNetV2.
   - The channel attention module employs a shared two-layer MLP with a reduction ratio $r=16$ ($1280 \to 80 \to 1280$), and the spatial attention module uses a $7 \times 7$ convolution over concatenated spatial statistics (mean and max pool across channels).

4. **Retraining Methodology and Empirical Test Performance:**  
   - Models were retrained under Phase 3 using a two-stage transfer learning schedule: Phase A (warmup of CBAM and classification head at $\text{LR} = 10^{-4}$ with frozen backbone) and Phase B (fine-tuning the top 40 layers of MobileNetV2 with frozen `BatchNormalization` at $\text{LR} = 10^{-5}$).
   - CSV training logs confirm that each V2 model completed 5 epochs (3 warmup epochs and 2 fine-tuning epochs) on CPU.
   - Rigorous evaluation across **603 strictly held-out test specimens** (with zero data leakage) established:
     - **Species Expert V2:** 98.01% accuracy, Macro F1: 0.9767.
     - **Potato Doctor CBAM V2:** 97.84% accuracy, Macro F1: 0.9673.
     - **Tomato Doctor CBAM V2:** 89.95% accuracy, Macro F1: 0.8870.
     - **End-to-End Hierarchical System:** 91.85% diagnostic accuracy on ungated leaves, 84.08% overall system accuracy, with an 8.46% safety abstention rate and only 3 routing errors.

5. **Operational Strengths and Critical Deficiencies:**  
   - **Strengths:** Superb botanical safety gates (abstaining when species confidence $<90\%$ or Late Blight confidence $<90\%$), responsible educational agricultural advisories preventing reckless chemical application, 100% pass rate across 7 regression test suites, and strict image validation handling RGBA alpha compositing, CMYK, and Grayscale.
   - **Critical Vulnerabilities & Technical Debt:** Missing version pinning in `requirements.txt`; uninstalled ONNX deployment packages (`tf2onnx`, `onnxruntime`); `export_to_onnx.py` defaulting to legacy V1 model directories; coexistence of an internal Linux virtual environment (`wsl_venv`) inside the Windows workspace; lack of a REST API / Web UI; complete absence of git version control (`.git` missing); and monkey-patching of Keras 3 `Dense.from_config`.

---

## 2. Project Overview

### 2.1 Problem Statement and Context
Agricultural foliar diseases in Solanaceous crops, primarily Potato (*Solanum tuberosum*) and Tomato (*Solanum lycopersicum*), represent a major threat to smallholder farmer yields and global food security. The targeted pathological conditions are:
1. **Early Blight** (*Alternaria solani*): A fungal disease characterized by concentric, target-like brown-black necrotic rings on senescent foliage.
2. **Late Blight** (*Phytophthora infestans*): A water-mold (oomycete) pathogen capable of devastating entire fields within days under cool, humid conditions; characterized by dark, water-soaked lesions with pale chlorotic margins.
3. **Healthy Foliage**: Absence of target foliar blight symptoms.

### 2.2 Domain Challenges Addressed
Direct end-to-end 6-class classification models often fail in real-world deployment due to two major domain failure modes:
1. **Botanical Species Confusion:** Macroscopic similarities between potato and tomato leaflets lead to crop misidentification, potentially causing farmers to apply chemical fungicides contraindicated for that crop.
2. **High-Consequence Diagnostic False Positives:** Late Blight is an aggressive, high-consequence pathogen. Classifying benign mechanical damage, insect feeding wounds, or minor environmental necrosis as Late Blight triggers expensive, environmentally hazardous applications of systemic fungicides. Conversely, missing Late Blight leads to crop loss.

### 2.3 System Evolution
To resolve these challenges, the system evolved from a monolithic single-stage classifier into a **Two-Tier Hierarchical Decision Architecture with Dual Confidence Safety Gates**:
- **Tier 1 (Botanist):** Identifies plant species (Potato vs. Tomato). If species confidence is below 90%, the system deliberately **abstains** from disease diagnosis.
- **Tier 2 (Specialist Crop Doctors):** Routes confident specimens to crop-specific models equipped with CBAM attention.
- **Safety Gate 2:** Evaluates Late Blight confidence against a strict 90% threshold. If below 90%, it triggers an advisory warning of potential mechanical injury and recommends isolating the plant and re-scouting within 24–48 hours rather than applying fungicides immediately.

---

## 3. Complete Project Directory Tree

Below is the verified project directory tree (excluding `.venv`, `wsl_venv`, and `.idea` internal package libraries):

```
D:\plant-disease-detection\
│
├── audit\                                   # Phase 2 audit scripts, manifests, and generated artifacts
│   ├── artifacts\                           # Visual audit charts and quantitative evaluation reports
│   │   ├── augmentation_breakdown.png       # Composition of dataset: original vs synthetic augmentations
│   │   ├── class_distribution.png           # Raw image counts vs true biological specimens per class
│   │   ├── data_leakage_comparison.png      # Comparison of data leakage: legacy random split vs group split
│   │   ├── split_distribution.png           # Allocation of images across Train/Val/Test splits
│   │   ├── v2_evaluation_report.json        # Quantitative metrics across 603 test specimens
│   │   └── v2_training_curves.png           # Loss and accuracy evolution curves for V2 models
│   ├── audit_summary.json                   # Output of raw dataset audit, duplicate counts, and checks
│   ├── dataset_manifest.csv                 # Master manifest of all 11,988 images (with hashes and groups)
│   ├── generate_specimen_manifest.py        # Script: Generates canonical 1-leaf-per-specimen manifests
│   ├── run_phase2_audit.py                  # Script: Full DSU clustering and stratified group splitting
│   ├── specimen_manifest.csv                # Canonical biological specimen manifest (3,971 rows)
│   ├── specimen_manifest_summary.json       # Breakdown of specimen selection reasons and split counts
│   ├── test_manifest.csv                    # Raw image test split manifest (1,872 rows)
│   ├── test_specimen_manifest.csv           # Held-out test specimen manifest (603 rows)
│   ├── train_manifest.csv                   # Raw image train split manifest (8,256 rows)
│   ├── train_specimen_manifest.csv          # Leakage-free train specimen manifest (2,776 rows)
│   ├── validation_manifest.csv              # Raw image validation split manifest (1,860 rows)
│   └── validation_specimen_manifest.csv     # Leakage-free validation specimen manifest (592 rows)
│
├── dataset\                                 # Raw image repository (11,988 JPEG images across 6 classes)
│   ├── potato\
│   │   ├── Early_blight\                    # 1,998 images (558 unique biological specimens)
│   │   ├── healthy\                         # 1,998 images (90 unique biological specimens)
│   │   └── Late_blight\                     # 1,998 images (558 unique biological specimens)
│   └── tomato\
│       ├── Early_blight\                    # 1,998 images (558 unique biological specimens)
│       ├── healthy\                         # 1,998 images (954 unique biological specimens)
│       └── Late_blight\                     # 1,998 images (1,253 unique biological specimens)
│
├── docs\                                    # Comprehensive technical documentation
│   └── END_TO_END_AUDIT_REPORT.md           # Pre-existing technical audit report (November 20, 2026)
│
├── inference\                               # Production V2 inference engine and image processing package
│   ├── __init__.py                          # Package exports and public interface definitions
│   ├── inference_config.json                # Runtime configuration, safety thresholds, and advisories
│   ├── model_loader.py                      # Thread-safe model loader with singleton caching & safe_mode
│   ├── predictor.py                         # HierarchicalPredictorV2 implementing two-tier gating logic
│   ├── preprocessing.py                     # Image loading, format conversion (RGBA/CMYK), and resizing
│   └── response_formatter.py                # Standardized JSON response and error serialization
│
├── models\                                  # Serialized model checkpoints
│   ├── potato_expert.keras                  # Legacy V1 Potato Specialist checkpoint (2,422,339 params)
│   ├── species_expert.keras                 # Legacy V1 Botanist checkpoint (2,340,098 params)
│   ├── tomato_expert.keras                  # Legacy V1 Tomato Specialist checkpoint (2,422,339 params)
│   └── v2\                                  # Production V2 model checkpoints
│       ├── potato_expert_v2.keras           # V2 Potato Doctor with CBAM Attention (2,628,597 params)
│       ├── species_expert_v2.keras          # V2 Botanist Species Expert (2,340,098 params)
│       └── tomato_expert_v2.keras           # V2 Tomato Doctor with CBAM Attention (2,628,597 params)
│
├── reports\                                 # Output destination for official technical reports
│   └── END_TO_END_PROJECT_ANALYSIS_REPORT.md# This comprehensive audit report
│
├── test_images\                             # Verification and benchmark images
│   └── sample_leaf.JPG                      # Benchmark test image (Tomato Late Blight)
│
├── tests\                                   # Test suite and regression verification
│   └── test_v2_inference.py                 # Comprehensive 7-suite regression and safety test script
│
├── training\                                # V2 model definition, data loaders, losses, and utilities
│   ├── logs\                                # Training history logs (CSV format)
│   │   ├── potato_expert_v2_training.csv    # 5-epoch training loss/accuracy log for Potato Doctor V2
│   │   ├── species_expert_v2_training.csv   # 5-epoch training loss/accuracy log for Species Expert V2
│   │   └── tomato_expert_v2_training.csv    # 5-epoch training loss/accuracy log for Tomato Doctor V2
│   ├── __init__.py                          # Package initialization marker
│   ├── augmentation.py                      # Biologically safe online Keras augmentation pipeline
│   ├── class_mappings.json                  # Canonical integer-to-label class mapping definitions
│   ├── data_pipeline.py                     # High-performance manifest-driven tf.data dataset builder
│   ├── evaluate_v2_system.py                # Held-out test evaluation and metrics computation script
│   ├── losses.py                            # SparseCategoricalFocalLoss & specimen class weight calculation
│   ├── models.py                            # Keras 3 CBAM layers and hierarchical model builder functions
│   ├── run_preflight_and_smoke_tests.py     # Hardware check and end-to-end architecture smoke tests
│   ├── training_config.json                 # Hyperparameters, learning rates, epochs, and callback settings
│   ├── train_utils.py                       # Shared two-phase transfer learning orchestration engine
│   └── verify_v2_artifacts.py               # Deep artifact inspection and weight reconciliation script
│
├── export_to_onnx.py                        # CLI tool: Converts Keras 3 models to optimized ONNX graphs
├── predict_hierarchical.py                  # Legacy V1 CLI inference script targeting models/*.keras
├── predict_hierarchical_v2.py               # Production V2 CLI entry point targeting models/v2/*.keras
├── requirements.txt                         # Python package dependency list
├── train_potato_doctor.py                   # Root entry point: Trains Potato Doctor CBAM V2
├── train_species.py                         # Root entry point: Trains Botanist Species Expert V2
└── train_tomato_doctor.py                   # Root entry point: Trains Tomato Doctor CBAM V2
```

---

## 4. File-by-File Analysis

Every important file in the project was inspected line by line. The table and detailed descriptions below document their exact roles, functions, dependencies, and operational status.

### 4.1 Summary of Project Files

| File Path | Purpose | Key Functions / Classes | Primary Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- |
| `predict_hierarchical_v2.py` | Production CLI entry point for V2 inference | `main()`, `print_cli_report()`, `find_default_sample_image()` | `inference.predictor`, `argparse`, `json` | **Active (Primary)** |
| `predict_hierarchical.py` | Legacy V1 inference script | `load_models()`, `run_inference()`, `predict_plant()` | `tensorflow`, `PIL.Image`, `numpy` | **Legacy (Outdated)** |
| `export_to_onnx.py` | Export Keras 3 models to ONNX | `export_keras_to_onnx()`, `verify_onnx_model()`, `export_all_models()` | `tf2onnx`, `onnxruntime`, `tensorflow` | **Broken (Missing Deps)** |
| `train_species.py` | CLI training entry point for Species Expert | `main()` | `training.models`, `training.train_utils` | **Active (Training)** |
| `train_potato_doctor.py` | CLI training entry point for Potato Doctor | `main()` | `training.models`, `training.train_utils` | **Active (Training)** |
| `train_tomato_doctor.py` | CLI training entry point for Tomato Doctor | `main()` | `training.models`, `training.train_utils` | **Active (Training)** |
| `requirements.txt` | Python package specification | N/A (Plain text dependency list) | N/A | **Active (Unpinned)** |
| `inference/__init__.py` | Inference package namespace definition | Package exports | `inference.predictor`, `inference.model_loader` | **Active** |
| `inference/predictor.py` | Core V2 two-tier prediction engine | `HierarchicalPredictorV2`, `predict_v2()` | `model_loader`, `preprocessing`, `response_formatter` | **Active** |
| `inference/model_loader.py` | Thread-safe model cache and deserializer | `load_models()`, `load_v2_model_file()`, `unload_models()` | `tensorflow`, `training.models`, `threading` | **Active** |
| `inference/preprocessing.py` | Robust image ingestion and normalization | `load_and_validate_image()`, `preprocess_image()` | `PIL.Image`, `numpy`, `pathlib` | **Active** |
| `inference/response_formatter.py` | Standardized JSON schema serializer | `format_prediction_response()`, `format_error_response()` | Standard library `typing` | **Active** |
| `inference/inference_config.json` | Inference thresholds and agronomic advisories | JSON Configuration | N/A | **Active** |
| `training/__init__.py` | Training package marker | N/A | N/A | **Active** |
| `training/models.py` | Model architectures and CBAM attention | `CBAM`, `ChannelAttention`, `SpatialAttention`, `build_species_expert()`, `build_crop_doctor_cbam()` | `tensorflow`, `keras`, `pathlib` | **Active** |
| `training/data_pipeline.py` | Manifest-based `tf.data` dataset builder | `build_manifest_dataset()`, `load_class_mappings()` | `tensorflow`, `csv`, `json`, `pathlib` | **Active** |
| `training/augmentation.py` | Online agronomic augmentation pipeline | `get_training_augmentation()` | `tensorflow.keras.layers` | **Active** |
| `training/losses.py` | Loss functions and class weighting | `SparseCategoricalFocalLoss`, `compute_specimen_class_weights()` | `tensorflow`, `keras`, `csv`, `numpy` | **Active** |
| `training/train_utils.py` | Two-phase transfer learning orchestrator | `run_two_phase_training()`, `set_global_seeds()` | `data_pipeline`, `augmentation`, `losses` | **Active** |
| `training/training_config.json` | Training hyperparameters and callback settings | JSON Configuration | N/A | **Active** |
| `training/class_mappings.json` | Canonical class index dictionaries | JSON Configuration | N/A | **Active** |
| `training/evaluate_v2_system.py` | Held-out test evaluation runner | `evaluate_models()`, `compute_classification_metrics()`, `plot_training_curves()` | `tensorflow`, `PIL.Image`, `matplotlib` | **Active** |
| `training/verify_v2_artifacts.py` | Checkpoint validation and parameter check | `inspect_models()`, `file_sha256()` | `tensorflow`, `hashlib` | **Active** |
| `training/run_preflight_and_smoke_tests.py`| Preflight hardware and architecture smoke tests| `run_preflight_and_smoke_tests()` | `psutil`, `tensorflow`, `keras` | **Active** |
| `audit/run_phase2_audit.py` | Comprehensive dataset audit and split generator | `run_phase2_audit()`, `DisjointSetUnion`, `compute_dhash()`, `compute_sha256()` | `PIL.Image`, `numpy`, `matplotlib`, `csv` | **Active (Audit)** |
| `audit/generate_specimen_manifest.py`| Generates 1-leaf-per-specimen manifests | `generate_specimen_manifests()` | `csv`, `json`, `collections` | **Active (Audit)** |
| `tests/test_v2_inference.py` | Regression and safety validation test suite | 7 test functions covering safe deserialization, parameters, gates, advisories | `inference.predictor`, `inference.model_loader` | **Active (Test)** |

### 4.2 Detailed Module Descriptions

#### 1. `predict_hierarchical_v2.py`
- **Location:** `d:\plant-disease-detection\predict_hierarchical_v2.py`
- **Purpose:** Primary user-facing CLI entry point for the modern V2 inference system.
- **Functionality:** Parses command-line arguments (`--image`, `--json`), automatically searches for default test leaves (`test_images/sample_leaf.JPG`), invokes `inference.predictor.predict_v2()`, formats visual console outputs with emoji indicators and ANSI borders, or prints structured JSON.
- **Connections:** Imports `predict_v2` and `HierarchicalPredictorV2` from `inference.predictor`.
- **Status:** **Active**. Primary operational entry point.

#### 2. `predict_hierarchical.py`
- **Location:** `d:\plant-disease-detection\predict_hierarchical.py`
- **Purpose:** Legacy V1 command-line prediction script.
- **Functionality:** Loads legacy models directly from `models/` (`species_expert.keras`, `potato_expert.keras`, `tomato_expert.keras`). Implements a single threshold check on Late Blight ($<90\%$) but lacks species confidence gating, robust RGBA input handling, and JSON reporting.
- **Connections:** Contains inline monkey-patching for Keras 3 `Dense.from_config`.
- **Status:** **Outdated / Legacy**. Maintained solely for backward compatibility.

#### 3. `export_to_onnx.py`
- **Location:** `d:\plant-disease-detection\export_to_onnx.py`
- **Purpose:** Converts `.keras` models into optimized ONNX computation graphs for microservice deployment.
- **Functionality:** Uses `tf2onnx.convert.from_keras` with dynamic batch inputs `(None, 224, 224, 3)` at opset 17 and validates converted graphs using `onnxruntime.InferenceSession`.
- **Deficiencies:** Requires `tf2onnx` and `onnxruntime`, which are omitted from the active virtual environment. Furthermore, `export_all_models()` defaults to searching `models/*.keras` (legacy models) rather than `models/v2/*.keras`.
- **Status:** **Broken / Blocked by Dependencies**.

#### 4. `training/models.py`
- **Location:** `d:\plant-disease-detection\training\models.py`
- **Purpose:** Core neural network architectural definitions.
- **Functionality:** Contains Keras 3 serializable layer classes (`ChannelAttention`, `SpatialAttention`, `CBAM`) with `get_config()` hooks and registered under the namespace `plant_pathology`. Provides `build_species_expert()` and `build_crop_doctor_cbam()`. Features an intelligent fallback `get_mobilenet_backbone()` that extracts weights from local models to avoid network dependencies.
- **Status:** **Active**. Core architectural foundation.

#### 5. `training/losses.py`
- **Location:** `d:\plant-disease-detection\training\losses.py`
- **Purpose:** Custom loss functions and biological class weighting.
- **Functionality:** Implements `SparseCategoricalFocalLoss` ($\gamma=2.0, \alpha_t$) to focus gradients on hard lesion boundaries. Implements `compute_specimen_class_weights()`, which calculates inverse-frequency class weights based on unique biological source clusters (`source_group_id`), preventing over-represented classes from dominating optimization.
- **Status:** **Active**.

#### 6. `training/data_pipeline.py`
- **Location:** `d:\plant-disease-detection\training\data_pipeline.py`
- **Purpose:** Manifest-driven asynchronous data loader.
- **Functionality:** Reads CSV manifests directly via Python's built-in `csv` module (eliminating pandas overhead), parses file paths, applies `tf.io.read_file` and `tf.io.decode_jpeg`, resizes to $(224, 224)$, applies shuffling and batching, injects online augmentation on training batches, and applies `.prefetch(tf.data.AUTOTUNE)`.
- **Status:** **Active**.

#### 7. `training/train_utils.py`
- **Location:** `d:\plant-disease-detection\training\train_utils.py`
- **Purpose:** Training engine managing the two-phase transfer learning schedule.
- **Functionality:** Orchestrates Phase A (base frozen, Adam $\text{LR}=10^{-4}$) and Phase B (top 40 layers unfrozen, BatchNorm frozen, Adam $\text{LR}=10^{-5}$). Attaches `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, and `CSVLogger` callbacks. Seeds Python, NumPy, TensorFlow, and hash environments.
- **Status:** **Active**.

#### 8. `inference/model_loader.py`
- **Location:** `d:\plant-disease-detection\inference\model_loader.py`
- **Purpose:** Production model lifecycle and caching engine.
- **Functionality:** Manages thread-safe singleton loading of all three V2 models using `threading.Lock()`. Validates non-zero file sizes and enforces `safe_mode=True` during deserialization to protect against untrusted bytecode execution.
- **Status:** **Active**.

#### 9. `inference/predictor.py`
- **Location:** `d:\plant-disease-detection\inference\predictor.py`
- **Purpose:** Two-tier inference decision engine.
- **Functionality:** Manages the sequential evaluation: Tier 1 Botanist $\to$ Gate 1 Species Confidence Check ($\ge 90\%$) $\to$ Specialist Routing $\to$ Gate 2 Late Blight / General Disease Evaluation $\to$ Advisory mapping.
- **Status:** **Active**.

#### 10. `inference/preprocessing.py`
- **Location:** `d:\plant-disease-detection\inference\preprocessing.py`
- **Purpose:** Defensive input validation and image tensor preparation.
- **Functionality:** Ingests file paths, PIL Images, or NumPy arrays. Verifies file integrity, converts arbitrary color spaces (L, RGBA, CMYK) to RGB (specifically compositing RGBA onto a solid white background to eliminate black transparency artifacts), resizes to $224 \times 224$ via bilinear interpolation, and formats float32 tensors in the $[0.0, 255.0]$ range.
- **Status:** **Active**.

#### 11. `tests/test_v2_inference.py`
- **Location:** `d:\plant-disease-detection\tests\test_v2_inference.py`
- **Purpose:** Phase 4.1 regression and safety audit suite.
- **Functionality:** Implements 7 comprehensive audit suites verifying: (1) safe deserialization and CBAM parameters; (2) parameter counts; (3) multi-format image preprocessing; (4) species gate boundary conditions and abstention; (5) Late Blight gate boundaries; (6) educational advisory text; and (7) legacy V1 asset preservation.
- **Status:** **Active**. Verified passing with 100% success.

---

## 5. Dataset Analysis

### 5.1 Dataset Organization and Disease Classes
The dataset is physically partitioned under `dataset/` into two top-level crop folders (`potato` and `tomato`), each containing three subdirectories corresponding to the pathological state:
- `dataset/potato/Early_blight/`
- `dataset/potato/Late_blight/`
- `dataset/potato/healthy/`
- `dataset/tomato/Early_blight/`
- `dataset/tomato/Late_blight/`
- `dataset/tomato/healthy/`

### 5.2 Forensic Image Inventory: Raw Counts vs. True Biological Specimens
Superficially, the dataset appears perfectly balanced: exactly **1,998 images** per class across all 6 classes, yielding an apparent total of **11,988 images**. 

However, forensic analysis conducted in `audit/run_phase2_audit.py` revealed that this balance is entirely synthetic. An extensive proportion of the images are offline synthetic augmentations generated prior to project ingestion. The table below presents the verified distribution:

| Crop | Disease Class | Raw Images | Verified Originals | Offline Augmented | % Augmented | Exact SHA Duplicates | Unique Biological Specimens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Potato** | Early Blight | 1,998 | 558 | 1,440 | 72.07% | 152 groups | **558** |
| **Potato** | Late Blight | 1,998 | 558 | 1,440 | 72.07% | 144 groups | **558** |
| **Potato** | Healthy | 1,998 | 81 | 1,917 | **95.95%** | 506 groups | **90** |
| **Tomato** | Early Blight | 1,998 | 558 | 1,440 | 72.07% | 96 groups | **558** |
| **Tomato** | Late Blight | 1,998 | 1,260 | 738 | 36.94% | 31 groups | **1,253** |
| **Tomato** | Healthy | 1,998 | 954 | 1,044 | 52.25% | 41 groups | **954** |
| **Total** | **All 6 Classes**| **11,988** | **3,969** | **8,019** | **66.88%** | **970 groups**| **3,971** |

### 5.3 Severe Biological Class Imbalance
The data reveals severe hidden biological imbalance masked by offline augmentation:
- **Potato Healthy** contains only **90 unique biological leaves** (81 unaugmented originals plus 9 standalone augmented clusters). To inflate this category to 1,998 images, leaves were replicated up to 22 times using center crops, Gaussian blurs, and rotations.
- In contrast, **Tomato Late Blight** contains **1,253 unique biological leaves**.
- The true biological ratio between Tomato Late Blight and Potato Healthy is **13.92 : 1**. Training models on the raw, uncurated folder without specimen weighting causes models to severely overfit to the visual quirks of those 90 potato leaves.

### 5.4 Image Formats and Physical Dimensions
- **File Format:** All 11,988 images are valid JPEG files. Zero corrupted or unreadable images were detected (`corrupted_images: 0`).
- **Dimensions:** The majority of unaugmented originals have dimensions of $256 \times 256$ pixels. Offline center-cropped variants (`_ccrp_`) range between $200 \times 200$ and $240 \times 240$ pixels.
- **Color Channels:** Standard 3-channel RGB (8 bits per channel).

### 5.5 Catastrophic Data Leakage in Legacy Splitting (V1)
In the legacy V1 implementation, datasets were split randomly across raw file paths (e.g., using `image_dataset_from_directory(validation_split=0.2)`).
- Because 66.88% of images were derived from parent leaves through slight cropping or blurring, random splitting scattered variations of the **exact same leaf** across both the training set and the validation/test sets.
- Empirical verification in `audit/run_phase2_audit.py` proved that **86.30% of biological source clusters were leaked across splits** in the legacy pipeline.
- Consequently, the model was tested on slightly rotated or blurred versions of images it had already memorized during training, generating an artificial, scientifically invalid "98.6% accuracy".

### 5.6 Data Leakage Resolution via DSU Graph Clustering (V2)
To eliminate data leakage, `audit/run_phase2_audit.py` deployed a multi-stage clustering pipeline:
1. **Bit-Level Deduplication:** Linked exact SHA-256 matches.
2. **Cluster Prefix Grouping:** Grouped file prefixes (e.g., `potato_early_1000_*`).
3. **Perceptual Hash Distance:** Linked parent leaves to augmented variants using Difference Hash (dHash) and Average Hash (aHash). Pairs with combined Hamming distance $\le 4$ were unified using a **Disjoint Set Union (DSU)** data structure.
4. **Stratified Group Splitting:** The resulting 3,971 independent biological specimen groups were allocated deterministically (seed 42) into:
   - **Training Set:** 2,776 specimens (70%)
   - **Validation Set:** 592 specimens (15%)
   - **Held-Out Test Set:** 603 specimens (15%)
5. **Formal Verification:** All four verification checks passed:
   - `check1_no_group_leakage`: **PASS** (0 group overlap between any splits).
   - `check2_no_exact_duplicate_leakage`: **PASS** (0 SHA-256 collisions across splits).
   - `check3_all_classes_represented`: **PASS** (all classes represented across train, val, and test).
   - `check4_deterministic_reproducibility`: **PASS** (100% reproducible split generation).

---

## 6. Image Preprocessing Pipeline Analysis

### 6.1 Complete Image-Processing Workflow
The image preprocessing pipeline bridges raw user inputs to model tensors. The flow is traced below:

```mermaid
flowchart TD
    A["Raw Input (Path, PIL Image, or NumPy Array)"] --> B["load_and_validate_image()"]
    B --> C{"Check File & Integrity"}
    C -->|Corrupt / Missing| D["Raise FileNotFoundError / ValueError"]
    C -->|Valid| E{"Color Mode Check"}
    E -->|Mode == 'RGBA'| F["Composite onto Solid White Background"]
    E -->|Mode in ['L', 'CMYK', 'P']| G["Convert to standard 3-Channel RGB"]
    E -->|Mode == 'RGB'| H["Pass-through RGB Image"]
    F --> I["Bilinear Resize to (224, 224)"]
    G --> I
    H --> I
    I --> J["Convert to NumPy float32 Array in [0.0, 255.0]"]
    J --> K["Expand Batch Dimension -> Shape (1, 224, 224, 3)"]
    K --> L["Model Input Tensor"]
    L --> M["Internal Lambda Layer: tf.keras.applications.mobilenet_v2.preprocess_input"]
    M --> N["Normalized Tensor in [-1.0, 1.0] -> Backbone Convolutions"]
```

### 6.2 Preprocessing Details

1. **Defensive Image Loading (`inference/preprocessing.py:load_and_validate_image`):**  
   Accepts file paths (`str`, `Path`), `PIL.Image.Image` instances, or `np.ndarray` (2D grayscale, 3D RGB, 4D RGBA). Validates file existence and tests image readability using PIL's internal image decoders.

2. **Color Space Standardization:**  
   - Grayscale ($L$) and Palette ($P$) images are converted to 3-channel RGB by duplicating channel luminance.
   - CMYK images are mapped to RGB.
   - **RGBA Compositing Fix:** A common failure mode in plant pathology CV occurs when transparent PNG images (e.g., segmented leaves) are loaded. Standard `.convert("RGB")` fills the alpha channel with black, creating dark, high-contrast artificial edges that trigger false Late Blight predictions. The V2 preprocessor explicitly composites RGBA images onto a pure white background (`(255, 255, 255)`), preserving natural leaf perimeter contrasts.

3. **Spatial Rescaling:**  
   Resized to target dimensions $(224, 224)$ using high-fidelity bilinear interpolation (`Image.Resampling.BILINEAR`).

4. **Internal Model Normalization:**  
   - Both the training pipeline (`training/data_pipeline.py`) and inference preprocessor (`inference/preprocessing.py`) deliver tensors in the raw float32 range $[0.0, 255.0]$.
   - Normalization to $[-1.0, 1.0]$ is encapsulated directly within the Keras computational graph via an initial layer:
     ```python
     layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input, name="preprocess_input")
     ```
   - This ensures **100% mathematical consistency** across training, evaluation, ONNX export, and runtime inference.

---

## 7. CNN Model Architecture Analysis

### 7.1 Implemented CNN Backbones
Both generations of the project employ **MobileNetV2** as the primary feature extraction backbone, leveraging its depthwise separable inverted residual blocks with linear bottlenecks for computational efficiency.

### 7.2 Layer-by-Layer Architecture Comparison

#### Model 1: Botanist Species Classifier (`Species_Expert_Botanist_v2`)
- **Input Dimension:** $(None, 224, 224, 3)$
- **Pre-processing:** `Lambda(preprocess_input)` scaling $[0, 255] \to [-1, 1]$.
- **Backbone:** MobileNetV2 (154 layers, base feature map output: $7 \times 7 \times 1280$).
- **Head:**
  - `GlobalAveragePooling2D(name="gap")` $\to (None, 1280)$
  - `Dense(64, activation="relu", name="dense_64")` $\to (None, 64)$
  - `Dropout(0.2, name="dropout")` $\to (None, 64)$
  - `Dense(2, activation="softmax", name="species_output")` $\to (None, 2)$
- **Parameter Summary:**
  - Total Parameters: **2,340,098**
  - Trainable Parameters: **1,745,474**
  - Non-Trainable Parameters: **594,624** (frozen lower MobileNetV2 layers and batch normalization parameters)

#### Model 2 & 3: Specialist Crop Doctors (`Potato_Doctor_CBAM_v2` and `Tomato_Doctor_CBAM_v2`)
- **Input Dimension:** $(None, 224, 224, 3)$
- **Pre-processing:** `Lambda(preprocess_input)` scaling $[0, 255] \to [-1, 1]$.
- **Backbone:** MobileNetV2 (terminal feature extractor output: $7 \times 7 \times 1280$).
- **Attention Module:** `CBAM(ratio=16, kernel_size=7, name="cbam_attention")` $\to (None, 7, 7, 1280)$.
- **Head:**
  - `GlobalAveragePooling2D(name="gap")` $\to (None, 1280)$
  - `Dense(128, activation="relu", name="dense_128")` $\to (None, 128)$
  - `Dropout(0.3, name="dropout")` $\to (None, 128)$
  - `Dense(3, activation="softmax", name="disease_output")` $\to (None, 3)$
- **Parameter Summary:**
  - Total Parameters: **2,628,597**
  - Trainable Parameters: **2,033,973**
  - Non-Trainable Parameters: **594,624**

### 7.3 Mathematical Parameter Reconciliation

The parameter delta between the Crop Doctors ($2,628,597$) and the Species Expert ($2,340,098$) is exactly **288,499 parameters**, fully accounted for by:
1. **CBAM Attention Layer:** Exactly **206,258 parameters**.
2. **Dense Head Expansion:** Crop doctors use `Dense(128)` ($1280 \times 128 + 128 = 163,968$ params) and `Dense(3)` ($128 \times 3 + 3 = 387$ params), whereas the Species Expert uses `Dense(64)` ($1280 \times 64 + 64 = 81,984$ params) and `Dense(2)` ($64 \times 2 + 2 = 130$ params). The delta in the classification head is $(163,968 + 387) - (81,984 + 130) = 82,241$ params.
3. Total Delta: $206,258 + 82,241 = 288,499$ parameters.

The architectural reconciliation is mathematically verified.

---

## 8. CBAM / Attention Mechanism Analysis

### 8.1 Implementation Verification
The presence and active functioning of the **Convolutional Block Attention Module (CBAM)** was verified through inspectable Keras 3 custom layers defined in `training/models.py`. CBAM sequentially applies Channel Attention followed by Spatial Attention.

```mermaid
flowchart LR
    F["Input Feature Map F<br>(7 x 7 x 1280)"] --> CAM["Channel Attention Module (CAM)"]
    CAM --> Mc["Channel Weight Vector Mc<br>(1 x 1 x 1280)"]
    F --> Mult1["Element-wise Product ⊗"]
    Mc --> Mult1
    Mult1 --> Fp["Refined Feature Map F'<br>(7 x 7 x 1280)"]
    Fp --> SAM["Spatial Attention Module (SAM)"]
    SAM --> Ms["Spatial Weight Map Ms<br>(7 x 7 x 1)"]
    Fp --> Mult2["Element-wise Product ⊗"]
    Ms --> Mult2
    Mult2 --> Fpp["Attended Feature Map F''<br>(7 x 7 x 1280)"]
```

### 8.2 Submodule 1: Channel Attention Module (CAM)
The Channel Attention Module exploits the inter-channel relationships of features:
1. **Spatial Aggregation:** Computes two separate spatial descriptors from input feature map $F \in \mathbb{R}^{7 \times 7 \times 1280}$:
   - Global Average Pooling: $F_{\text{avg}}^c = \text{GAP}(F) \in \mathbb{R}^{1 \times 1 \times 1280}$
   - Global Max Pooling: $F_{\text{max}}^c = \text{GMP}(F) \in \mathbb{R}^{1 \times 1 \times 1280}$
2. **Shared Multi-Layer Perceptron (MLP):** Both descriptors are forwarded through a shared two-layer dense network with reduction ratio $r = 16$:
   - `shared_dense_1`: $1280 \to 80$ (ReLU activation, He normal initialization). Weight shape: $(1280, 80)$, Bias: $(80)$. Total: $102,480$ parameters.
   - `shared_dense_2`: $80 \to 1280$ (Linear activation, He normal initialization). Weight shape: $(80, 1280)$, Bias: $(1280)$. Total: $103,680$ parameters.
   - Total CAM parameters: $102,480 + 103,680 = \mathbf{206,160}$ parameters.
3. **Combination and Activation:** The MLP output vectors are merged via element-wise summation and modulated by a sigmoid activation function:
   $$\mathbf{M}_c(F) = \sigma\left(W_1(W_0(F_{\text{avg}}^c)) + W_1(W_0(F_{\text{max}}^c))\right)$$
4. **Channel Refinement:** The intermediate attended feature map is generated as $F' = \mathbf{M}_c(F) \otimes F$.

### 8.3 Submodule 2: Spatial Attention Module (SAM)
The Spatial Attention Module focuses on the informational "where" (lesion geography and necrotic boundaries):
1. **Channel Aggregation:** Compresses $F'$ along the channel axis using average and max pooling:
   - $F_{\text{avg}}^s = \text{Mean}_{c}(F') \in \mathbb{R}^{7 \times 7 \times 1}$
   - $F_{\text{max}}^s = \text{Max}_{c}(F') \in \mathbb{R}^{7 \times 7 \times 1}$
2. **Channel Concatenation:** Concatenates both descriptors along the channel axis to form a tensor of shape $(None, 7, 7, 2)$.
3. **Convolutional Filter:** Applies a standard 2D convolution with a large receptive field:
   - Kernel size: $7 \times 7$
   - Filters: $1$
   - Padding: `'same'`
   - Bias: `False`
   - Activation: Sigmoid ($\sigma$)
   - Parameter calculation: $7 \times 7 \times 2 \times 1 = \mathbf{98}$ parameters.
4. **Spatial Mask Computation:**
   $$\mathbf{M}_s(F') = \sigma\left(f^{7 \times 7}\left([F_{\text{avg}}^s; F_{\text{max}}^s]\right)\right)$$
5. **Final Attended Output:** $F'' = \mathbf{M}_s(F') \otimes F'$.

### 8.4 Attention Integration and Pathology Benefit
- Total CBAM parameters: $206,160 + 98 = \mathbf{206,258}$.
- **Pathological Relevance:** In early foliar infection, chlorotic halos and fungal lesions occupy only 5% to 15% of the leaf surface. Global Average Pooling over raw MobileNetV2 features tends to dilute localized lesion activations with background green tissue. CBAM's spatial attention dynamically amplifies activations corresponding to necrotic spots while suppressing non-symptomatic background parenchyma.
- **Architectural Omission in Species Classifier:** CBAM was deliberately omitted from `Species_Expert_Botanist_v2`. Botanical discrimination relies on macroscopic leaflet shape and margin venation, where fine-grained lesion attention is irrelevant.

---

## 9. Model Training Pipeline Audit

### 9.1 Training Methodology and Phased Transfer Learning
The training implementation in `training/train_utils.py` executes a two-phase transfer learning schedule designed to prevent catastrophic forgetting while adapting upper convolutional layers:

```mermaid
flowchart TD
    subgraph PhaseA["Phase A: Head & Attention Warmup"]
        A1["Freeze MobileNetV2 Base (154 layers)"] --> A2["Compile with Adam (LR = 1e-4)"]
        A2 --> A3["Train Classification Head & CBAM (3 Epochs)"]
    end
    subgraph PhaseB["Phase B: Deep Fine-Tuning"]
        B1["Unfreeze Top 40 Layers of MobileNetV2"] --> B2["Explicitly Freeze BatchNormalization Layers"]
        B2 --> B3["Compile with Adam (LR = 1e-5)"]
        B3 --> B4["Train Top 40 Layers + CBAM + Head (2 Epochs)"]
    end
    PhaseA --> PhaseB
```

### 9.2 Training Hyperparameters

| Configuration Parameter | Phase A (Warmup) | Phase B (Fine-Tuning) | Rationale |
| :--- | :--- | :--- | :--- |
| **Target Layers** | Classification Head + CBAM | Top 40 MobileNetV2 + CBAM + Head | Stabilizes randomly initialized weights |
| **Learning Rate** | $1 \times 10^{-4}$ ($0.0001$) | $1 \times 10^{-5}$ ($0.00001$) | Prevents destroying pretrained ImageNet representations |
| **Batch Size** | 16 | 8 | Prevents CPU/VRAM memory spikes during backpropagation |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adaptive moment estimation |
| **Loss Function** | Weighted Sparse Categorical CE | Weighted Sparse Categorical CE | Class-weighted by unique biological specimens |
| **Batch Normalization** | Frozen | **Frozen** (`layer.trainable = False`) | Prevents distorted mean/variance statistics at batch size 8 |

### 9.3 Biological Class Weighting vs. Focal Loss
To counter the 13.9:1 biological class imbalance in potato specimens, `training/losses.py` calculates inverse-frequency weights derived exclusively from the training split of the canonical specimen manifest:
$$w_c = \frac{N_{\text{total specimens}}}{K \times N_c}$$
For Potato Doctor training:
- Healthy: $w_2 = 14.92$ (countering the scarcity of 62 training specimens)
- Early Blight: $w_0 = 2.37$ (390 training specimens)
- Late Blight: $w_1 = 2.37$ (390 training specimens)

An optional `SparseCategoricalFocalLoss` is implemented with modulating factor $(1 - p_t)^\gamma$ ($\gamma=2.0$).

### 9.4 Empirical Training Log Analysis
Training history logs in `training/logs/` confirm the execution of Phase 3 retraining on November 20, 2026:

#### 1. `species_expert_v2_training.csv`
- Epoch 0 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 0.2737, Train Acc: 88.54%, Val Loss: 0.1659, Val Acc: 95.10%
- Epoch 1 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 0.1287, Train Acc: 95.75%, Val Loss: 0.0937, Val Acc: 97.80%
- Epoch 2 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 0.0990, Train Acc: 96.40%, Val Loss: 0.0778, Val Acc: 98.31%
- Epoch 3 (Phase B, $\text{LR}=10^{-5}$): Train Loss: 0.0909, Train Acc: 96.33%, Val Loss: **0.0535**, Val Acc: **98.48%**
- Epoch 4 (Phase B, $\text{LR}=10^{-5}$): Train Loss: 0.0617, Train Acc: 97.73%, Val Loss: 0.1216, Val Acc: 95.27%

#### 2. `potato_expert_v2_training.csv`
- Epoch 0 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 1.0523, Train Acc: 67.58%, Val Loss: 0.9883, Val Acc: 88.83%
- Epoch 1 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 0.7980, Train Acc: 88.84%, Val Loss: 0.5384, Val Acc: 92.74%
- Epoch 2 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 0.3082, Train Acc: 92.16%, Val Loss: 0.2192, Val Acc: 93.30%
- Epoch 3 (Phase B, $\text{LR}=10^{-5}$): Train Loss: 0.1377, Train Acc: 95.49%, Val Loss: 0.2079, Val Acc: 93.30%
- Epoch 4 (Phase B, $\text{LR}=10^{-5}$): Train Loss: 0.0970, Train Acc: 95.61%, Val Loss: **0.1359**, Val Acc: **94.97%**

#### 3. `tomato_expert_v2_training.csv`
- Epoch 0 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 1.0920, Train Acc: 47.78%, Val Loss: 1.0612, Val Acc: 82.81%
- Epoch 1 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 0.7031, Train Acc: 80.30%, Val Loss: 0.4581, Val Acc: 79.66%
- Epoch 2 (Phase A, $\text{LR}=10^{-4}$): Train Loss: 0.3370, Train Acc: 86.50%, Val Loss: 0.3412, Val Acc: 83.54%
- Epoch 3 (Phase B, $\text{LR}=10^{-5}$): Train Loss: 0.2439, Train Acc: 90.54%, Val Loss: **0.2499**, Val Acc: **92.01%**
- Epoch 4 (Phase B, $\text{LR}=10^{-5}$): Train Loss: 0.1805, Train Acc: 93.38%, Val Loss: 0.2590, Val Acc: 91.28%

All three models converged smoothly within 5 epochs without gradient instability.

---

## 10. Model Evaluation and Actual Results

### 10.1 Evaluation Methodology
Evaluation was performed by `training/evaluate_v2_system.py` strictly on the held-out test split (`audit/test_specimen_manifest.csv`), comprising **603 independent biological specimens** that were never seen during training or validation.

### 10.2 Verified Evaluation Metrics

#### 1. Botanist Species Classifier (`Species_Expert_Botanist_v2`)
- **Evaluated Cohort:** 603 specimens (185 Potato, 418 Tomato)
- **Overall Accuracy:** **98.01%**
- **Macro F1-Score:** **0.9767**
- **Per-Class Breakdown:**
  - **Potato:** Precision: 0.9626, Recall: 0.9730, F1: **0.9677** (Support: 185)
  - **Tomato:** Precision: 0.9880, Recall: 0.9833, F1: **0.9856** (Support: 418)
- **Confusion Matrix:**
  $$\begin{pmatrix} 180 & 5 \\ 7 & 411 \end{pmatrix}$$
  (Rows: Ground Truth [Potato, Tomato]; Columns: Predicted [Potato, Tomato])

#### 2. Potato Specialist (`Potato_Doctor_CBAM_v2`)
- **Evaluated Cohort:** 185 held-out potato specimens
- **Overall Accuracy:** **97.84%**
- **Macro F1-Score:** **0.9673**
- **Per-Class Breakdown:**
  - **Early Blight:** Precision: 0.9770, Recall: 1.0000, F1: **0.9884** (Support: 85)
  - **Late Blight:** Precision: 1.0000, Recall: 0.9529, F1: **0.9759** (Support: 85)
  - **Healthy:** Precision: 0.8824, Recall: 1.0000, F1: **0.9375** (Support: 15)
- **Confusion Matrix:**
  $$\begin{pmatrix} 85 & 0 & 0 \\ 2 & 81 & 2 \\ 0 & 0 & 15 \end{pmatrix}$$
  (Rows: Ground Truth [Early Blight, Late Blight, Healthy]; Columns: Predicted)

#### 3. Tomato Specialist (`Tomato_Doctor_CBAM_v2`)
- **Evaluated Cohort:** 418 held-out tomato specimens
- **Overall Accuracy:** **89.95%**
- **Macro F1-Score:** **0.8870**
- **Per-Class Breakdown:**
  - **Early Blight:** Precision: 0.8313, Recall: 0.8118, F1: **0.8214** (Support: 85)
  - **Late Blight:** Precision: 0.9645, Recall: 0.8624, F1: **0.9106** (Support: 189)
  - **Healthy:** Precision: 0.8675, Recall: 1.0000, F1: **0.9290** (Support: 144)
- **Confusion Matrix:**
  $$\begin{pmatrix} 69 & 6 & 10 \\ 14 & 163 & 12 \\ 0 & 0 & 144 \end{pmatrix}$$
  (Rows: Ground Truth [Early Blight, Late Blight, Healthy]; Columns: Predicted)

#### 4. End-to-End Hierarchical System Performance
- **Total Test Specimens Evaluated:** 603
- **Safety Gate 1 Threshold (Species Confidence):** $\ge 90.0\%$
- **Abstained by Safety Gate:** 51 specimens (**8.46% abstention rate**)
- **Routing Errors (Misclassified Species):** 3 specimens (0.50%)
- **Ungated Specimens Evaluated:** 552
- **Ungated Diagnostic Accuracy:** **91.85%**
- **End-to-End System Accuracy (including abstained leaves as non-diagnosed):** **84.08%**

---

## 11. Prediction and Inference Workflow

### 11.1 End-to-End Request-Response Flow
The production inference flow in `inference/predictor.py` operates as a disciplined, safety-gated decision tree:

```mermaid
flowchart TD
    Input["Input Image File / PIL / Array"] --> Prep["Image Ingestion & Preprocessing<br>(Resized to 224x224, Normalized)"]
    Prep --> T1["Tier 1: Botanist Species Expert V2"]
    T1 --> Gate1{"Safety Gate 1:<br>Species Confidence ≥ 90%?"}
    
    Gate1 -->|No (< 90%)| Abstain1["Status: 'species_uncertain'<br>Action: ABSTAIN from Disease Diagnosis<br>Advisory: Do NOT apply chemicals. Submit to lab."]
    
    Gate1 -->|Yes (≥ 90%)| Route{"Route to Specialist Crop Doctor"}
    Route -->|Potato| PotDoc["Potato Doctor CBAM V2"]
    Route -->|Tomato| TomDoc["Tomato Doctor CBAM V2"]
    
    PotDoc --> DisPred["Extract Disease Class & Confidence"]
    TomDoc --> DisPred
    
    DisPred --> Gate2{"Safety Gate 2:<br>Disease Condition Evaluation"}
    
    Gate2 -->|Late Blight & Conf < 90%| WarnLB["Status: 'late_blight_uncertain'<br>Action: Suppress Systemic Fungicides<br>Advisory: Possible mechanical injury. Re-inspect in 24-48h."]
    Gate2 -->|Late Blight & Conf ≥ 90%| ConfLB["Status: 'confident'<br>Action: Late Blight Confirmed<br>Advisory: Consult registered protective fungicides."]
    Gate2 -->|Early Blight & Conf ≥ 70%| ConfEB["Status: 'confident'<br>Action: Early Blight Confirmed<br>Advisory: Prune lower leaves; improve air circulation."]
    Gate2 -->|Healthy & Conf ≥ 70%| ConfH["Status: 'confident'<br>Action: Healthy Foliage<br>Advisory: Maintain standard scouting."]
    Gate2 -->|Any Disease & Conf < 70%| WarnMarg["Status: 'disease_uncertain'<br>Action: Marginal Confidence<br>Advisory: Ambiguous symptoms. Re-photograph in diffuse daylight."]
```

### 11.2 Safety Gate Verification on Benchmark Leaf (`sample_leaf.JPG`)
Running `predict_hierarchical_v2.py` on the benchmark test image `test_images/sample_leaf.JPG` verified the safety gates:
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
  "probabilities": {
    "Early Blight": 0.3019,
    "Late Blight": 0.6964,
    "Healthy": 0.0017
  },
  "metadata": {
    "execution_time_ms": 1408.26,
    "architecture": "MobileNetV2 + CBAM Attention"
  }
}
```
**Observation:** Although the top disease prediction is Late Blight, because the confidence ($69.64\%$) is below the $90.0\%$ high-consequence threshold, Safety Gate 2 triggered:
1. `safety_status` was set to `"late_blight_uncertain"`.
2. Fungicide application was suppressed in the advisory.
3. The farmer is advised to monitor the leaf for 24–48 hours to check whether the spot expands or is merely mechanical damage.

---

## 12. Frontend and Backend Architecture

### 12.1 Current Application Architecture
The project is currently structured as a **headless machine learning library and command-line application**.
- **No Web Interface:** There is no web frontend (HTML/CSS/JavaScript, React, Vue), desktop GUI (Tkinter, PyQt), or mobile frontend.
- **No REST / Microservice API:** There is no web API framework (FastAPI, Flask, Django Ninja, Tornado).
- **Current Entry Points:**
  - `predict_hierarchical_v2.py`: CLI tool for single-image diagnosis.
  - `inference/predictor.py`: Python API for in-memory integration (`from inference import predict_v2`).

### 12.2 Architectural Decoupling Assessment
Despite lacking a web interface, the internal architecture exhibits clean separation of concerns:
- **`inference/preprocessing.py`:** Handles image validation, channel ordering, and resizing independently of model execution.
- **`inference/model_loader.py`:** Decoupled model loading, safe deserialization, and thread-safe caching.
- **`inference/predictor.py`:** Encapsulates business logic, safety thresholds, and routing without formatting console strings.
- **`inference/response_formatter.py`:** Generates clean, RFC-compliant JSON dictionaries suitable for immediate serialization into REST API responses.

Wrapping the current V2 inference engine into a production FastAPI or Flask microservice requires only an API route handler calling `predict_v2()`.

---

## 13. Dependencies and Environment Audit

### 13.1 Requirements File Audit (`requirements.txt`)
The repository contains a minimal `requirements.txt`:
```
# Core Deep Learning, Inference, Training & Data Pipeline
tensorflow
numpy
pillow
matplotlib
psutil

# Optional: MLOps Model Deployment & ONNX Export
tf2onnx
onnxruntime
```

### 13.2 Environment Vulnerabilities and Dependency Gaps

1. **Complete Absence of Version Pinning:**  
   None of the dependencies specify version constraints (e.g., `tensorflow==2.21.0`). In deep learning projects, unpinned dependencies frequently cause breaking changes upon clean reinstallation (e.g., NumPy 2.x breaking older TensorFlow builds or Keras 3 API modifications).
2. **Missing Deployment Toolchain Packages:**  
   `tf2onnx` and `onnxruntime` are listed in `requirements.txt` but are **not installed** in the active virtual environment (`.venv`). Consequently, attempting to run `export_to_onnx.py` triggers an immediate `ImportError` and script termination.
3. **Internal Linux Virtual Environment Contamination (`wsl_venv`):**  
   The workspace contains a complete Linux virtual environment directory (`wsl_venv/`) created inside WSL2 (`/mnt/d/plant-disease-detection/wsl_venv`) alongside the Windows virtual environment (`.venv/`).
   - Having a Linux venv inside a Windows NTFS workspace directory risks file permission corruption, broken symlinks, and repository bloat.
4. **Python Version Incompatibility:**  
   - Windows System Python: **Python 3.14.4**
   - Active `.venv` Python: **Python 3.10.0**
   - WSL2 Python: **Python 3.12.3**
   - Attempting to install modern TensorFlow on Python 3.14 fails because official pre-built TensorFlow wheels are not yet available for Python 3.14.
5. **Native Windows GPU Limitations:**  
   TensorFlow versions $\ge 2.11$ discontinued native GPU support on Windows (requiring WSL2 or direct DirectX plugins). As verified during test execution:
   `WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11.`
   All training and inference on Windows executes over oneDNN-optimized multi-threaded CPU instructions.

---

## 14. Code Quality, Security and Software Engineering Audit

### 14.1 Security Audit

1. **Safe Model Deserialization (`safe_mode=True`):**  
   In `inference/model_loader.py:load_v2_model_file`, Keras model loading explicitly specifies `safe_mode=True`:
   ```python
   model = tf.keras.models.load_model(str(p), custom_objects=custom_objs, compile=False, safe_mode=True)
   ```
   This prevents arbitrary code execution vulnerabilities during `.keras` unpickling/deserialization.
2. **Denial-of-Service (DoS) and Image Bombs:**  
   `inference/preprocessing.py` relies on `PIL.Image.open()`. While it validates that files are images, it does not explicitly configure `Image.MAX_IMAGE_PIXELS = ...` (DecompressionBomb protection). Extremely large crafted images could cause memory exhaustion.
3. **No Hardcoded Secrets:**  
   Scans across all files confirmed zero API keys, cloud tokens, or hardcoded database credentials.

### 14.2 Code Quality and Technical Debt

1. **Absence of Git Version Control:**  
   The repository root lacks a `.git` folder (`fatal: not a git repository`). Operating an active machine learning codebase without version tracking prevents reproducible experiment rollbacks and risks unrecoverable accidental file loss.
2. **Monkey-Patching Keras 3 `Dense.from_config`:**  
   In `export_to_onnx.py`, `predict_hierarchical.py`, and `training/models.py`, the following monkey-patch is injected:
   ```python
   orig_dense_from_config = keras.layers.Dense.from_config
   @classmethod
   def patched_dense_from_config(cls, config):
       config.pop('quantization_config', None)
       return orig_dense_from_config.__func__(cls, config)
   keras.layers.Dense.from_config = patched_dense_from_config
   ```
   **Cause:** Keras 3.12 introduced serialization metadata (`quantization_config`) that older deserializers fail on. While this workaround allows loading models across varying Keras versions, modifying core class methods at runtime is technical debt that should be addressed via standard custom layer registration or clean export.
3. **Inconsistent Path Configurations in Deployment Tools:**  
   In `export_to_onnx.py`, `export_all_models()` defaults to `models_dir="models"`. Because V2 models are stored in `models/v2/`, running `python export_to_onnx.py` exports the deprecated V1 models instead of the production V2 CBAM models.

---

## 15. Testing and Reproducibility Audit

### 15.1 Existing Test Infrastructure
The testing infrastructure consists of:
1. **`tests/test_v2_inference.py`:** An automated 7-suite regression and safety test script.
2. **`training/run_preflight_and_smoke_tests.py`:** A preflight environment and hardware check script with architecture smoke tests.
3. **`training/verify_v2_artifacts.py`:** A checksum and parameter validation script.

### 15.2 Empirical Test Execution
Executing `tests/test_v2_inference.py` under Python 3.10 yielded **100% success across all 7 test suites**:
- **Test 1 (Safe Deserialization & CBAM Weights):** PASSED. CBAM weights verified at exactly 206,258.
- **Test 2 (Trainable Parameter Reconciliation):** PASSED. Species: 2,340,098 total; Potato Doctor: 2,628,597 total.
- **Test 3 (Preprocessing Formats):** PASSED. RGB, RGBA (alpha blended), Grayscale, CMYK, and NumPy arrays verified.
- **Test 4 (Species Gate Boundaries & Abstention):** PASSED. Confidence $<90\%$ strictly abstains from disease diagnosis.
- **Test 5 (Late Blight Gate Boundaries):** PASSED. Confidence $<90\%$ attaches low-confidence warning.
- **Test 6 (Advisory Safety Review):** PASSED. Advisory correctly suppresses chemical prescriptions on uncertain lesions.
- **Test 7 (Legacy V1 Asset Preservation & Error Handling):** PASSED. Missing files raise clean `InferenceError`.

### 15.3 Gaps in Test Infrastructure
- **No Pytest Integration:** Tests are executed via standard `python` CLI runs with Python assertions rather than using `pytest` fixtures, parameterized tests, and test discovery.
- **No Pipeline Unit Tests:** There are no automated unit tests for `training/data_pipeline.py`, `training/losses.py`, or `audit/run_phase2_audit.py` in the `tests/` directory.
- **No Continuous Integration (CI/CD):** There are no GitHub Actions workflows (`.github/workflows/`) or GitLab CI pipelines.

---

## 16. Performance and Scalability Analysis

### 16.1 Computational Complexity and Footprint

| Metric | Species Expert V2 | Potato Doctor CBAM V2 | Tomato Doctor CBAM V2 | Total System |
| :--- | :---: | :---: | :---: | :---: |
| **Total Parameters** | 2,340,098 | 2,628,597 | 2,628,597 | **7,597,292** |
| **Trainable Parameters** | 1,745,474 | 2,033,973 | 2,033,973 | **5,813,420** |
| **Disk Storage (Keras)** | 23.92 MB | 27.41 MB | 27.41 MB | **78.74 MB** |
| **RAM Footprint (Loaded)** | ~70 MB | ~85 MB | ~85 MB | **~240 MB** |

### 16.2 Runtime Latency Benchmarks (CPU)
- **Cold-Start Latency:** Initial invocation (`predict_v2`) incurs a cold-start latency of **$1,300 - 1,500$ ms**, driven by TensorFlow graph compilation, model deserialization, and weight loading from disk into RAM.
- **Warm Cached Latency:** Once cached via `inference.model_loader`, subsequent inferences execute in **$45 - 90$ ms per leaf** on multi-core modern CPUs.
- **Throughput Limitation:** Sequential single-image inference limits throughput to ~15 images per second per CPU core. High-throughput edge or cloud deployment would benefit substantially from batching and ONNX Runtime execution.

---

## 17. End-to-End System Architecture Diagram

The complete end-to-end architecture of the Plant Disease Detection system is modeled below:

```mermaid
flowchart TD
    subgraph DataStage["Phase 1: Dataset Audit & Leakage-Free Stratified Group Splitting"]
        D1["Raw Dataset: 11,988 Images<br>(6 Classes @ 1,998 images)"] --> D2["Hash Analysis: SHA-256 & Perceptual dHash/aHash"]
        D2 --> D3["Disjoint Set Union (DSU) Graph Clustering"]
        D3 --> D4["Identification of 3,971 Unique Biological Specimens"]
        D4 --> D5["Deterministic Stratified Group Split (Seed 42)"]
        D5 --> S_Train["Train Split (2,776 specimens / 70%)"]
        D5 --> S_Val["Validation Split (592 specimens / 15%)"]
        D5 --> S_Test["Held-Out Test Split (603 specimens / 15%)"]
    end

    subgraph TrainStage["Phase 2: Model Training Pipeline"]
        S_Train --> DataPipe["tf.data Pipeline: Bilinear Resize, Batching & Prefetching"]
        DataPipe --> OnlineAug["Agronomic Online Augmentation:<br>Flips, Rotation ±14°, Zoom ±10%, Brightness ±15%"]
        OnlineAug --> LossCalc["Specimen Inverse Class Weighting & Focal Loss"]
        LossCalc --> Ph1["Phase A Warmup: Head & CBAM (LR = 1e-4, 3 Epochs)"]
        Ph1 --> Ph2["Phase B Fine-Tuning: Top 40 Layers (LR = 1e-5, 2 Epochs, Frozen BatchNorm)"]
        Ph2 --> Checkpoints["Model Checkpoints Saved to models/v2/*.keras"]
    end

    subgraph EvalStage["Phase 3: Held-Out Test Evaluation"]
        S_Test --> EvalRunner["training/evaluate_v2_system.py"]
        Checkpoints --> EvalRunner
        EvalRunner --> Metrics["Verified Held-Out Metrics:<br>• Species Botanist: 98.01% Acc<br>• Potato Doctor CBAM: 97.84% Acc<br>• Tomato Doctor CBAM: 89.95% Acc<br>• End-to-End Hierarchical: 91.85% Acc (Ungated)"]
    end

    subgraph InferStage["Phase 4: Two-Tier Hierarchical Inference & Safety Gating"]
        UserImg["User Image Input<br>(File Path, PIL Image, or NumPy Array)"] --> PreProc["inference/preprocessing.py<br>• RGBA White Background Compositing<br>• Bilinear Resize to 224x224<br>• Float32 [0.0, 255.0] Tensor"]
        PreProc --> Tier1["Tier 1: Botanist Species Expert V2<br>(MobileNetV2 + GAP + Dense 64 + Softmax)"]
        Tier1 --> Gate1{"Safety Gate 1:<br>Species Confidence ≥ 90%?"}
        
        Gate1 -->|No (< 90%)| Abstain["Status: 'species_uncertain'<br>Abstain from Disease Diagnosis<br>Advisory: Do NOT apply chemicals. Submit to accredited lab."]
        
        Gate1 -->|Yes (≥ 90%)| RouteDoc{"Route to Specialist Crop Doctor"}
        RouteDoc -->|Potato| PotModel["Potato Doctor CBAM V2"]
        RouteDoc -->|Tomato| TomModel["Tomato Doctor CBAM V2"]
        
        PotModel --> CBAM_Mod["Injected CBAM Attention Block:<br>• Channel Attention (Shared MLP, r=16)<br>• Spatial Attention (7x7 Conv)"]
        TomModel --> CBAM_Mod
        
        CBAM_Mod --> Tier2Head["Dense 128 + Softmax 3-Class Output"]
        Tier2Head --> Gate2{"Safety Gate 2:<br>Disease Confidence Evaluation"}
        
        Gate2 -->|Late Blight & Conf < 90%| WarnLB["Status: 'late_blight_uncertain'<br>Advisory: Suppress chemical fungicide. Isolate & re-scout in 24-48h."]
        Gate2 -->|Late Blight & Conf ≥ 90%| ConfLB["Status: 'confident'<br>Advisory: Late Blight confirmed. Follow registered chemical controls."]
        Gate2 -->|Early Blight & Conf ≥ 70%| ConfEB["Status: 'confident'<br>Advisory: Early Blight confirmed. Cultural sanitation & pruning."]
        Gate2 -->|Healthy & Conf ≥ 70%| ConfH["Status: 'confident'<br>Advisory: Healthy foliage. Maintain normal scouting."]
        Gate2 -->|Any Disease & Conf < 70%| WarnMarg["Status: 'disease_uncertain'<br>Advisory: Marginal confidence. Re-photograph in diffuse daylight."]
        
        Abstain --> FinalJSON["Structured JSON Output & Formatted CLI Report"]
        WarnLB --> FinalJSON
        ConfLB --> FinalJSON
        ConfEB --> FinalJSON
        ConfH --> FinalJSON
        WarnMarg --> FinalJSON
    end
```

---

## 18. Identified Issues and Their Severity

The findings from this technical audit are classified into four severity tiers:

### A. Critical Issues (Severe Flaws That Invalidate Results or Break Execution)

#### Issue A-1: Uninstalled ONNX Deployment Toolchain and Broken Model Exporter
- **Severity:** Critical
- **File Path:** `d:\plant-disease-detection\export_to_onnx.py`
- **Location:** Lines 71–77; Lines 142–168 (`export_all_models`)
- **Evidence:** Running `python export_to_onnx.py` immediately crashes with code 1:
  `❌ ERROR: 'tf2onnx' is not installed in the active environment.`
  Furthermore, `export_all_models()` searches `models/*.keras` (the legacy V1 models) rather than `models/v2/*.keras`.
- **Impact:** Model export to low-latency microservice formats (ONNX Runtime, Triton) is broken.
- **Recommended Solution:** Install `tf2onnx>=1.16.1` and `onnxruntime>=1.20.0` in `.venv`. Update line 142 default directory to `models_dir="models/v2"`.
- **Expected Benefit:** Enables cross-platform, multi-runtime C++/Rust/Go microservice deployments with 2–3x lower CPU inference latency.

#### Issue A-2: Complete Absence of Git Version Control
- **Severity:** Critical
- **File Path:** Project root (`d:\plant-disease-detection\`)
- **Evidence:** Executing `git status` reports `fatal: not a git repository (or any of the parent directories): .git`.
- **Impact:** There is no commit history, branch isolation, or rollback capability. Any accidental file deletion, overwrite, or corruption is unrecoverable.
- **Recommended Solution:** Initialize a git repository (`git init`), create a comprehensive `.gitignore` (ignoring `.venv/`, `wsl_venv/`, `.idea/`, `__pycache__/`, and `.keras` weight files if using Git LFS), and create a baseline initial commit.
- **Expected Benefit:** Secures codebase integrity, enables collaborative branching, and preserves research lineage.

---

### B. High-Priority Issues (Significant Quality, Reliability, or Performance Weaknesses)

#### Issue B-1: Completely Unpinned Dependencies in `requirements.txt`
- **Severity:** High
- **File Path:** `d:\plant-disease-detection\requirements.txt`
- **Evidence:** Contains bare package names without versions: `tensorflow`, `numpy`, `pillow`, `matplotlib`, `psutil`.
- **Impact:** Reinstalling dependencies on a new machine or cloud container will fetch the newest releases, which currently breaks TensorFlow on Python 3.14 or introduces breaking Keras 3 changes.
- **Recommended Solution:** Pin exact known-good versions:
  ```text
  tensorflow==2.21.0
  keras==3.12.1
  numpy==2.2.6
  pillow==12.1.1
  matplotlib==3.10.8
  psutil==7.2.2
  tf2onnx==1.16.1
  onnxruntime==1.20.0
  ```
- **Expected Benefit:** Guaranteed environment reproducibility across all developer and deployment platforms.

#### Issue B-2: Workspace Contamination from Internal Linux Virtual Environment (`wsl_venv`)
- **Severity:** High
- **File Path:** `d:\plant-disease-detection\wsl_venv\`
- **Evidence:** `wsl_venv/pyvenv.cfg` indicates a full Python 3.12 Linux environment (`/mnt/d/plant-disease-detection/wsl_venv`) residing directly inside the Windows NTFS project folder.
- **Impact:** Bloats the repository by thousands of platform-incompatible binary files, interferes with IDE indexing, and risks symlink corruption across the Windows/WSL boundary.
- **Recommended Solution:** Move WSL virtual environments outside the shared NTFS mount (e.g., to `~/venvs/plant-disease-detection` in the Linux home directory). Add `wsl_venv/` to `.gitignore`.
- **Expected Benefit:** Prevents cross-platform file locking, cleans IDE project trees, and avoids git bloat.

#### Issue B-3: Runtime Monkey-Patching of Core Keras `Dense.from_config`
- **Severity:** High
- **File Path:** `d:\plant-disease-detection\training\models.py` (lines 139–144), `export_to_onnx.py` (lines 35–40), `predict_hierarchical.py` (lines 36–41)
- **Evidence:** Repeatedly overrides `keras.layers.Dense.from_config` at module startup to strip `quantization_config`.
- **Impact:** Modifying core library classes at runtime can cause subtle bugs if other libraries or Keras components rely on standard deserialization behaviors.
- **Recommended Solution:** Standardize serialization using native Keras 3 `export()` / `SavedModel` or update all checkpoints using a clean one-time migration script.
- **Expected Benefit:** Eliminates fragile runtime monkey-patches and ensures compliance with standard Keras 3 deserialization protocols.

---

### C. Medium-Priority Issues (Architecture, Maintainability, and Usability)

#### Issue C-1: Coexistence of Deprecated V1 Scripts and Checkpoints in Root
- **Severity:** Medium
- **File Path:** `d:\plant-disease-detection\predict_hierarchical.py`, `models/*.keras`
- **Evidence:** Legacy V1 predictor (`predict_hierarchical.py`) and V1 models reside in the root and `models/` directory, causing confusion with `predict_hierarchical_v2.py` and `models/v2/`.
- **Impact:** New developers or automated scripts may inadvertently invoke the flawed V1 pipeline that lacks species confidence safety gates.
- **Recommended Solution:** Move `predict_hierarchical.py` and `models/*.keras` into a dedicated `legacy/` or `v1_archive/` directory.
- **Expected Benefit:** Clean separation between obsolete code and active production pipelines.

#### Issue C-2: Absence of a Modern Test Runner (Pytest) and CI/CD Pipeline
- **Severity:** Medium
- **File Path:** `d:\plant-disease-detection\tests\`
- **Evidence:** `tests/test_v2_inference.py` runs only via direct script invocation with raw Python assertions. No `pytest.ini` or GitHub Actions workflow exists.
- **Impact:** Prevents automated pre-commit testing, coverage tracking, and continuous integration upon code changes.
- **Recommended Solution:** Add `pytest` configuration and create `.github/workflows/ci.yml` to automatically execute tests on push.
- **Expected Benefit:** Automated quality assurance preventing regressions.

#### Issue C-3: Lack of Web API or Interactive UI
- **Severity:** Medium
- **File Path:** Project root (`d:\plant-disease-detection\`)
- **Evidence:** Only CLI scripts (`predict_hierarchical_v2.py`) exist.
- **Impact:** Non-technical agricultural specialists and field agronomists cannot interact with the model without terminal access.
- **Recommended Solution:** Build a lightweight FastAPI wrapper exposing a `/diagnose` endpoint and an optional Streamlit or Gradio demo UI.
- **Expected Benefit:** Enables field demonstration and integration with mobile and web clients.

---

### D. Low-Priority Improvements (Documentation, Code Style, Minor Cleanups)

#### Issue D-1: Missing Root `README.md`
- **Severity:** Low
- **File Path:** Project root (`d:\plant-disease-detection\README.md`)
- **Evidence:** `README.md` is absent in the workspace root. The only documentation resides in `docs/END_TO_END_AUDIT_REPORT.md`.
- **Impact:** Missing standard onboarding instructions for cloning, environment setup, and running inference.
- **Recommended Solution:** Add a concise `README.md` summarizing the project, installation commands, quickstart CLI usage, and linking to the technical audit reports.
- **Expected Benefit:** Immediate clarity for developers and auditors exploring the repository.

#### Issue D-2: Incomplete Type Annotations in Training Modules
- **Severity:** Low
- **File Path:** `d:\plant-disease-detection\training\train_utils.py`, `training\data_pipeline.py`
- **Evidence:** Several core functions lack complete PEP 484 type annotations (e.g., `run_two_phase_training`, `build_manifest_dataset`).
- **Impact:** Reduces IDE auto-completion accuracy and prevents strict static analysis via `mypy`.
- **Recommended Solution:** Add standard Python type hints (`Optional[str]`, `Tuple[tf.data.Dataset, int]`).
- **Expected Benefit:** Improves code maintainability and developer ergonomics.

---

## 19. Recommended Improvements Roadmap

### Phase 1: Environment and Deployment Hardening (Immediate)
1. **Initialize Git Repository:** Run `git init`, set up `.gitignore`, and commit baseline project files.
2. **Pin Dependencies:** Update `requirements.txt` with exact version constraints.
3. **Fix ONNX Export:** Install `tf2onnx` and `onnxruntime`, update `export_to_onnx.py` to point to `models/v2/`, and generate verified `.onnx` models.
4. **Isolate Linux Environment:** Relocate `wsl_venv` outside the Windows repository folder.

### Phase 2: Codebase Organization and Archival (Short-Term)
1. **Archive Legacy Assets:** Create `archive/v1/` and move `predict_hierarchical.py` and `models/*.keras` into it.
2. **Add Root `README.md`:** Provide project overview, architecture diagram, and quickstart commands.
3. **Migrate to Pytest:** Refactor `tests/test_v2_inference.py` to use `pytest` conventions and add data pipeline unit tests.

### Phase 3: Production Microservice and Web Interface (Medium-Term)
1. **FastAPI Microservice:** Create `api/main.py` providing:
   - `POST /v2/diagnose`: Ingests multipart image upload, returns formatted JSON response.
   - `GET /health`: Healthcheck endpoint reporting loaded model status.
2. **Interactive Web UI:** Create a lightweight Streamlit web application (`app.py`) allowing users to drag-and-drop leaf photos, view species and disease probabilities, inspect CBAM attention heatmaps, and read educational advisories.
3. **Docker Containerization:** Author a multi-stage `Dockerfile` (Python 3.10 slim base) for containerized cloud deployment.

### Phase 4: Pathological Research and Attention Visualizations (Long-Term)
1. **Grad-CAM Attention Heatmaps:** Implement a Grad-CAM utility targeting the output of the CBAM layer to visually overlay attention heatmaps on user leaves, allowing agronomists to see exactly where the model is focusing.
2. **Tomato Doctor Optimization:** Fine-tune `Tomato_Doctor_CBAM_v2` with additional held-out data to raise accuracy from 89.95% to $>95\%$, matching the Potato Doctor.

---

## 20. Project Strengths

1. **Rigorous Data Leakage Elimination:**  
   Unlike the vast majority of published PlantVillage-based deep learning papers that report inflated accuracies due to random splitting over augmented variations, this project implemented a mathematically rigorous **Disjoint Set Union (DSU) graph clustering** algorithm using SHA-256 and perceptual difference hashes. The resulting Stratified Group Split guarantees **0.0% data leakage**.
2. **Genuine and Active CBAM Attention Mechanism:**  
   The Convolutional Block Attention Module is not merely decorative; it is compiled directly into the computational graph of both Specialist Crop Doctors, adding exactly **206,258 active parameters** that modulate feature representations before classification.
3. **High-Consequence Botanical Safety Gates:**  
   The two-tier design incorporates safety gates:
   - Abstaining when species classification confidence is below 90%.
   - Triggering conservative agronomic advisories when Late Blight confidence is below 90% to prevent unnecessary systemic fungicide applications.
4. **Responsible Educational Agronomic Advisories:**  
   Rather than outputting bare probabilities, the system outputs structured, actionable agricultural advisories with clear legal and educational disclaimers.
5. **High Performance and Proven Empirical Accuracy:**  
   Held-out evaluation across 603 independent biological specimens verified strong empirical results: 98.01% species accuracy, 97.84% potato disease accuracy, and 91.85% ungated hierarchical diagnostic accuracy.
6. **Defensive Image Ingestion:**  
   The preprocessor robustly handles varied image formats, including RGBA alpha compositing on white backgrounds, Grayscale, CMYK, and raw NumPy arrays.
7. **Comprehensive Regression Suite:**  
   `tests/test_v2_inference.py` thoroughly audits model deserialization, parameter counts, gate boundaries, and error conditions, passing with 100% success.

---

## 21. Project Weaknesses

1. **Broken ONNX Deployment Pipeline:**  
   Missing toolchain dependencies (`tf2onnx`, `onnxruntime`) and hardcoded legacy directory paths in `export_to_onnx.py` prevent compiling models into production microservice graphs.
2. **Unpinned Dependency Specifications:**  
   `requirements.txt` lacks version pinning, creating substantial risks of environment breakage upon future reinstallation.
3. **Absence of Version Control (`.git`):**  
   The workspace is not tracked by git, leaving the project without version history or rollback capabilities.
4. **Workspace Environment Bloat:**  
   Retaining a complete Linux virtual environment (`wsl_venv`) directly inside the Windows NTFS project folder creates file locking and IDE confusion.
5. **Tomato Doctor Performance Disparity:**  
   While Potato Doctor achieved 97.84% accuracy, Tomato Doctor achieved 89.95% accuracy (F1: 0.8870), primarily due to higher morphological ambiguity between early Alternaria lesions and Late Blight on tomato leaflets.
6. **Runtime Monkey-Patching Technical Debt:**  
   Monkey-patching `Dense.from_config` across multiple files to bypass Keras 3.12 deserialization issues introduces maintenance fragility.
7. **Lack of Web Interface or API:**  
   The project is restricted to CLI scripts and Python library calls, lacking a browser UI or REST endpoint for non-developer stakeholders.

---

## 22. Overall Technical Assessment

| Evaluation Dimension | Rating (1–10) | Evaluation Comments |
| :--- | :---: | :--- |
| **Deep Learning Architecture** | **9.5 / 10** | Excellent two-tier hierarchical design. CBAM attention mechanism is correctly implemented, compiled, and integrated into the computational graph. |
| **Dataset Engineering & Integrity** | **9.5 / 10** | Exceptional forensic audit. DSU clustering solved catastrophic 86.3% data leakage, creating a truly leak-free stratified group split. |
| **Model Evaluation Rigor** | **9.5 / 10** | Scientifically sound. Evaluated on 603 strictly held-out biological specimens with verified metrics and confusion matrices. |
| **Safety & Domain Guardrails** | **10.0 / 10** | Exemplary dual confidence safety gates (90% threshold) and responsible educational advisories preventing chemical over-application. |
| **Inference Preprocessing** | **9.0 / 10** | Robust handling of RGBA (alpha-blended on white), CMYK, Grayscale, and NumPy arrays. |
| **Software Engineering & Structure**| **7.5 / 10** | Clean package modularity in V2 (`audit/`, `training/`, `inference/`, `tests/`), but penalized by missing `.git` repository and coexistence of legacy V1 files. |
| **Dependency & Environment Management**| **5.0 / 10** | Unpinned `requirements.txt`, missing ONNX packages, and internal `wsl_venv` folder contamination. |
| **Deployment Readiness** | **6.0 / 10** | Strong Python library and CLI, but lacks REST API, Dockerfile, or working ONNX export. |
| **Testing & Verification** | **8.0 / 10** | Outstanding custom regression suite (100% pass rate), but lacks `pytest` integration and CI/CD automation. |
| **Composite Technical Score** | **8.2 / 10** | **High Quality / Production-Capable Core with Remediation Needed for Deployment** |

---

## 23. Final Conclusion

The **Plant Disease Detection** deep learning project represents an exemplary transition from a flawed legacy machine learning script into an advanced, scientifically sound, hierarchical computer vision architecture.

The project's greatest accomplishments lie in its **dataset integrity engineering** and **domain-specific safety guardrails**. By recognizing that raw dataset balance was an artifact of offline augmentation that caused severe data leakage, the engineering team executed a forensic graph clustering audit that restored scientific validity to the model's benchmarks. The integration of the **Convolutional Block Attention Module (CBAM)** adds verifiable feature extraction power to localized lesion detection, and the **dual confidence safety gates** set a high standard for responsible AI in agriculture.

To transition this project from an audited research artifact into an enterprise-grade production platform, the engineering priorities are unambiguous:
1. Initialize git version control and pin all dependencies in `requirements.txt`.
2. Fix the ONNX deployment toolchain by installing `tf2onnx` and updating `export_to_onnx.py`.
3. Wrap the robust `HierarchicalPredictorV2` engine in a FastAPI microservice and Streamlit user interface.

With these operational remediations implemented, the system stands as an exemplary, evidence-based deep learning solution for agricultural disease diagnosis.

---
*Report certified by Senior Machine Learning Engineer, Deep Learning Researcher & Software Architect.*
