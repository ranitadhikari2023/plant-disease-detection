# Phase 1 — Environment and Deployment Hardening
## Verification, Implementation, and Parity Report

**Project Workspace:** `D:\plant-disease-detection`  
**Execution Date:** November 21, 2026  
**Role:** Senior Machine Learning Engineer, Python Environment Specialist, TensorFlow/Keras Engineer, Model Deployment Engineer  
**Task Phase:** Phase 1 — Environment and Deployment Hardening  
**Target Architecture:** Hierarchical V2 (Botanist Species Expert + Crop Doctors with CBAM Attention)  

---

## 1. Executive Summary and Final Decision

### 1.1 Phase 1 Status: PARTIALLY COMPLETE (Deployment Hardened; WSL Blocked by Host BIOS/OS Virtualization)

All core Python environment hardening, dependency pinning, ONNX exporter remediation, three-model ONNX conversion, ONNX Runtime validation, numerical parity verification, and full V2 regression testing have been **successfully executed and verified with 100% precision**.

The sole pending item is live WSL execution, which is **technically blocked by host Windows machine configuration** (`Virtual Machine Platform` disabled in Windows optional features and/or BIOS hardware virtualization disabled, triggering `HCS_E_SERVICE_NOT_AVAILABLE`). A turnkey automated shell script (`setup_wsl_env.sh`) has been provided to immediately provision and verify `~/venvs/plant-disease-detection` once the host virtualization feature is enabled by the system administrator.

### 1.2 Summary of Acceptance Criteria

| No. | Acceptance Criterion | Status | Evidence / Notes |
|:---:|---|:---:|---|
| **1** | Existing Git repository and baseline commit preserved | **PASS** | Verified clean commit `021e8d8 Initial commit: Plant Disease Detection V2` intact. No new commits created. |
| **2** | Direct dependencies pinned to compatible exact versions | **PASS** | Replaced unpinned entries in `requirements.txt` with exact tested versions. |
| **3** | Dependency resolver reports no conflicts | **PASS** | `python -m pip check` outputs `No broken requirements found.` |
| **4** | TensorFlow and Keras import successfully | **PASS** | `tensorflow==2.21.0`, `keras==3.12.1`, `numpy==2.2.6` import cleanly. |
| **5** | ONNX export dependencies installed | **PASS** | `tf2onnx==1.17.0`, `onnx==1.23.2`, `onnxruntime==1.20.0` installed in Windows `.venv`. |
| **6** | Exporter defaults to `models/v2/` | **PASS** | `export_to_onnx.py` defaults to `models/v2/` and outputs to `models/onnx/v2/`. Legacy fallbacks removed. |
| **7** | Species Expert V2 ONNX exported and validated | **PASS** | `species_expert_v2.onnx` (8.82 MB, opset 17, dynamic batch, 2 classes) verified via `onnx.checker`. |
| **8** | Potato Doctor V2 ONNX exported and validated | **PASS** | `potato_expert_v2.onnx` (9.92 MB, opset 17, dynamic batch, 3 classes, CBAM) verified via `onnx.checker`. |
| **9** | Tomato Doctor V2 ONNX exported and validated | **PASS** | `tomato_expert_v2.onnx` (9.92 MB, opset 17, dynamic batch, 3 classes, CBAM) verified via `onnx.checker`. |
| **10** | ONNX Runtime inference verified for all three models | **PASS** | All 3 models execute in `onnxruntime` with `CPUExecutionProvider` on single and batched inputs. |
| **11** | Keras/ONNX probability parity verified | **PASS** | Parity confirmed across 7 biological specimens and 2 synthetic tensors. Max abs diff: $\le 3.46 \times 10^{-6}$ (Tolerance: $10^{-4}$). Argmax agreement: 100%. |
| **12** | WSL environment recreated outside project directory | **BLOCKED** | Host machine has `Virtual Machine Platform` disabled (`HCS_E_SERVICE_NOT_AVAILABLE`). Turnkey script `setup_wsl_env.sh` provided. |
| **13** | Old project-local WSL environment safely retired | **PARTIALLY COMPLETE** | `wsl_venv` confirmed ignored by `.gitignore` and untracked by Git. Retirement paused until new WSL instance can be launched to avoid data loss. |
| **14** | Existing V2 regression tests pass | **PASS** | `tests/test_v2_inference.py` passed all 7 regression suites with 100% success. |
| **15** | V2 end-to-end inference remains functional | **PASS** | `predict_hierarchical_v2.py --image test_images/sample_leaf.JPG --json` functional, correct gates & advisory. |
| **16** | Original model checkpoints remain unchanged | **PASS** | SHA-256 hashes of all 3 `.keras` checkpoints match baseline down to the exact bit. |
| **17** | Phase 1 documentation generated | **PASS** | Complete report compiled in `reports/PHASE_1_ENVIRONMENT_AND_DEPLOYMENT_HARDENING.md`. |

---

## 2. Initial Problems Identified

Prior to Phase 1 implementation, forensic technical inspection identified the following defects:

1. **Unpinned Dependency Specification:**  
   `requirements.txt` contained unpinned package names (`tensorflow`, `numpy`, `pillow`, `matplotlib`, `psutil`, `tf2onnx`, `onnxruntime`). Resolving these packages dynamically threatened reproducibility and risks breakage upon subsequent reinstallation.
2. **Missing ONNX Toolchain in Windows Runtime:**  
   The active Windows virtual environment (`.venv`, Python 3.10.0) lacked `tf2onnx`, `onnx`, and `onnxruntime`, causing `export_to_onnx.py` to immediately terminate.
3. **Audit Proposed Version Incompatibility (`tf2onnx==1.16.1` vs `tensorflow==2.21.0`):**  
   The audit proposed `tf2onnx==1.16.1`. However, empirical dependency resolution proved that `tf2onnx==1.16.1` pins `protobuf~=3.20`, whereas `tensorflow==2.21.0` mandates `protobuf<8.0.0,>=6.31.1` (installed: `protobuf==7.34.1`). Forcing `tf2onnx==1.16.1` downgraded protobuf and created a fatal dependency conflict. Resolving to `tf2onnx==1.17.0` restored 100% compatibility across TensorFlow, Keras, protobuf, and ONNX.
4. **Leftover Legacy Packages in `.venv`:**  
   `tensorboard 2.10.1` and associated plugins remained in `.venv` from prior experiments, causing `pip check` to report an unresolved protobuf requirement.
5. **Exporter Directory Misdirection in `export_to_onnx.py`:**  
   The existing `export_to_onnx.py` defaulted to searching `models/*.keras` (the legacy V1 checkpoints) rather than `models/v2/`.
6. **Fragile Runtime Monkey-Patching in Exporter:**  
   `export_to_onnx.py` contained an unneeded global monkey-patch of `keras.layers.Dense.from_config`. V2 model deserialization natively supports `safe_mode=True` without monkey-patching.
7. **Workspace Contamination (`wsl_venv`):**  
   A 6.7 GB Linux Python 3.12 virtual environment was residing directly inside `D:\plant-disease-detection\wsl_venv`.

---

## 3. Dependency Pinning and Environment Reproducibility (Task 1)

### 3.1 Exact Version Specifications (`requirements.txt`)

The updated `requirements.txt` has been pinned to exact, verified versions:

```text
# Core Deep Learning, Inference, Training & Data Pipeline
tensorflow==2.21.0
keras==3.12.1
numpy==2.2.6
pillow==12.1.1
matplotlib==3.10.8
psutil==7.2.2
protobuf==7.34.1

# MLOps Model Deployment & ONNX Export
tf2onnx==1.17.0
onnx==1.23.2
onnxruntime==1.20.0
```

### 3.2 Technical Justification for Version Selections

1. **`tensorflow==2.21.0` & `keras==3.12.1`:**  
   Baseline deep learning frameworks used to train the V2 architecture. MobileNetV2 backbone and CBAM submodules deserialize natively.
2. **`numpy==2.2.6`:**  
   NumPy 2.x baseline utilized by TensorFlow 2.21. Preprocessing arrays and tensor conversions operate cleanly.
3. **`protobuf==7.34.1`:**  
   Required by `tensorflow 2.21.0` (`protobuf<8.0.0,>=6.31.1`).
4. **`tf2onnx==1.17.0` (Updated from candidate 1.16.1):**  
   Candidate `1.16.1` required `protobuf~=3.20`, creating an irresolvable conflict with TensorFlow 2.21.0. Version `1.17.0` provides full support for Protobuf 7.x, Keras 3 functional models, and ONNX opset 17.
5. **`onnx==1.23.2` & `onnxruntime==1.20.0`:**  
   Exact pairing verified for high-performance CPU execution and schema validation.

### 3.3 Verification Evidence

**Command executed:**
```powershell
& ".\.venv\Scripts\python.exe" -m pip check
```
**Output:**
```text
No broken requirements found.
```

**Critical imports test:**
```python
import tensorflow as tf
import keras
import numpy as np
import PIL
import matplotlib
import psutil
import tf2onnx
import onnx
import onnxruntime as ort

print('TF:', tf.__version__)           # 2.21.0
print('Keras:', keras.__version__)     # 3.12.1
print('NumPy:', np.__version__)         # 2.2.6
print('PIL:', PIL.__version__)         # 12.1.1
print('Matplotlib:', matplotlib.__version__) # 3.10.8
print('psutil:', psutil.__version__)   # 7.2.2
print('tf2onnx:', tf2onnx.__version__) # 1.17.0
print('onnx:', onnx.__version__)       # 1.23.2
print('onnxruntime:', ort.__version__) # 1.20.0
```
**Result:** `ALL CRITICAL IMPORTS PASSED!` with zero warnings or errors.

---

## 4. Production ONNX Export Engine (Task 2)

### 4.1 Exporter Enhancements in `export_to_onnx.py`

`export_to_onnx.py` was refactored with the following architectural hardening:

1. **V2 Discovery Default:** Defaults to `models/v2/` as the source directory and `models/onnx/v2/` as the destination directory.
2. **Strict Guardrails:** If any of the three required V2 checkpoints is missing, the engine aborts with a clear `FileNotFoundError`. Silent fallback to `models/*.keras` is strictly prohibited.
3. **Safe Model Deserialization:** Utilizes `safe_mode=True` with explicit registration of `CBAM`, `ChannelAttention`, `SpatialAttention`, `SparseCategoricalFocalLoss`, and `preprocess_input`. The fragile `Dense.from_config` monkey-patch was eliminated.
4. **Dynamic Batch Input Signature:** Enforces `tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input_tensor")`.
5. **Preserved Preprocessing:** Retains MobileNetV2 internal `preprocess_input` Lambda layer directly in the computation graph.
6. **ONNX Graph Validation:** Integrates `onnx.checker.check_model()`, shape introspection, dynamic batch dimension confirmation, and output class reconciliation.
7. **Empirical Numerical Parity Engine:** Tests exported ONNX models against Keras forward passes across real held-out biological leaf specimens and multi-batch tensors.
8. **CLI Interface:** Provides robust CLI options (`--models-dir`, `--output-dir`, `--model-path`, `--output-path`, `--opset`, `--tolerance`, `--no-verify`, `--no-parity`).

### 4.2 Three-Model ONNX Conversion Results

All three V2 production models were exported to `models/onnx/v2/`:

```text
models/
├── v2/
│   ├── species_expert_v2.keras
│   ├── potato_expert_v2.keras
│   └── tomato_expert_v2.keras
└── onnx/
    └── v2/
        ├── species_expert_v2.onnx
        ├── potato_expert_v2.onnx
        └── tomato_expert_v2.onnx
```

#### Detailed Model Specifications

| Attribute | Species Expert V2 | Potato Doctor CBAM V2 | Tomato Doctor CBAM V2 |
|---|---|---|---|
| **Keras Source** | `models/v2/species_expert_v2.keras` | `models/v2/potato_expert_v2.keras` | `models/v2/tomato_expert_v2.keras` |
| **ONNX Destination** | `models/onnx/v2/species_expert_v2.onnx` | `models/onnx/v2/potato_expert_v2.onnx` | `models/onnx/v2/tomato_expert_v2.onnx` |
| **Keras Parameters** | 2,340,098 | 2,628,597 | 2,628,597 |
| **CBAM Parameters** | None (Standard Backbone) | 206,258 (CA: 206,160; SA: 98) | 206,258 (CA: 206,160; SA: 98) |
| **Keras File Size** | 23,924,105 bytes | 27,407,925 bytes | 27,407,925 bytes |
| **ONNX File Size** | 9,247,005 bytes (8.82 MB) | 10,406,798 bytes (9.92 MB) | 10,406,710 bytes (9.92 MB) |
| **ONNX SHA-256** | `906fa73f1d71b9b0...` | `815a200235a1b194...` | `c400a727e1f8a1c6...` |
| **Target Opset** | 17 | 17 | 17 |
| **Input Node Name** | `input_tensor` | `input_tensor` | `input_tensor` |
| **Input Tensor Shape** | `['unk__655', 224, 224, 3]` | `['unk__1319', 224, 224, 3]` | `['unk__1984', 224, 224, 3]` |
| **Input Data Type** | `tensor(float)` (float32) | `tensor(float)` (float32) | `tensor(float)` (float32) |
| **Dynamic Batch** | Confirmed (`unk__*`) | Confirmed (`unk__*`) | Confirmed (`unk__*`) |
| **Output Node Name** | `species_output` | `disease_output` | `disease_output` |
| **Output Tensor Shape** | `['unk__656', 2]` | `['unk__1320', 3]` | `['unk__1985', 3]` |
| **onnx.checker Status** | **PASSED** | **PASSED** | **PASSED** |
| **ONNX Runtime Loading** | **PASSED** (CPUExecutionProvider) | **PASSED** (CPUExecutionProvider) | **PASSED** (CPUExecutionProvider) |

---

## 5. Numerical Prediction Parity Verification (Task 2 & Task 4)

Parity testing evaluated whether the exported ONNX computation graphs preserve the exact numerical inference behavior of the original Keras models.

### 5.1 Test Dataset and Inputs

Each model was tested across 9 distinct input cases:
- 7 representative real biological leaf specimens:
  1. `test_images/sample_leaf.JPG`
  2. `dataset/potato/Early_blight/potato_early_123.JPG`
  3. `dataset/potato/healthy/potato_healthy_133.JPG`
  4. `dataset/potato/Late_blight/potato_late_116.JPG`
  5. `dataset/tomato/Early_blight/tomato_early_124.JPG`
  6. `dataset/tomato/healthy/tomato_healthy_1005.JPG`
  7. `dataset/tomato/Late_blight/tomato_late_1003.JPG`
- Synthetic Tensor Batch Size 1 (uniform float32 in $[0.0, 255.0]$)
- Synthetic Tensor Batch Size 4 (uniform float32 in $[0.0, 255.0]$)

### 5.2 Parity Results Summary

| Model | Tested Inputs | Max Absolute Difference | Argmax Agreement | Softmax Sum Check | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Species Expert V2** | 9 inputs (7 leaves + 2 batches) | **$7.75 \times 10^{-7}$** | 100% (9/9) | $\approx 1.0 \pm 10^{-6}$ | **PASS** |
| **Potato Doctor CBAM V2** | 9 inputs (7 leaves + 2 batches) | **$1.73 \times 10^{-6}$** | 100% (9/9) | $\approx 1.0 \pm 10^{-6}$ | **PASS** |
| **Tomato Doctor CBAM V2** | 9 inputs (7 leaves + 2 batches) | **$3.46 \times 10^{-6}$** | 100% (9/9) | $\approx 1.0 \pm 10^{-6}$ | **PASS** |

**Conclusion:** All three models achieved numerical agreement well below the rigorous $10^{-4}$ tolerance threshold (maximum observed discrepancy was only $3.46 \times 10^{-6}$, attributable to minor floating-point operation re-ordering between TensorFlow C++ and ONNX Runtime CPU kernel implementations). Zero NaN or Inf values occurred, and all predicted class indices matched identically.

---

## 6. WSL Environment Migration Audit and Technical Limitation (Task 3)

### 6.1 Current Environment Inspection

- **Observed Path:** `D:\plant-disease-detection\wsl_venv` (`/mnt/d/plant-disease-detection/wsl_venv` in WSL)
- **Interpreter in config:** Python 3.12.3 (`wsl_venv/pyvenv.cfg`)
- **Folder Size:** 6,708 MB (6.7 GB)
- **Installed Linux Packages:** `tensorflow==2.21.0`, `keras==3.13.2`, `numpy==2.4.4`, `pillow==12.2.0`, `matplotlib==3.10.8`, `protobuf==7.34.1`, `fastapi`, `streamlit`, `gradio`.

### 6.2 Host Machine Virtualization Blocker

Attempts to launch WSL2 to provision `~/venvs/plant-disease-detection` were blocked by host Windows OS / BIOS configuration:

**Command executed:**
```powershell
wsl --status
```
**Output:**
```text
Default Distribution: Ubuntu
Default Version: 2
WSL 2 is not supported with your current machine configuration.
Please enable the "Virtual Machine Platform" optional component and ensure virtualization is enabled in the BIOS.
Enable "Virtual Machine Platform" by running: wsl.exe --install --no-distribution
For information please visit https://aka.ms/enablevirtualization
```

**Launch attempt command:**
```powershell
wsl -d Ubuntu bash -c "python3 --version"
```
**Output:**
```text
The operation could not be started because a required feature is not installed.
Error code: Wsl/Service/CreateInstance/CreateVm/HCS/HCS_E_SERVICE_NOT_AVAILABLE
```

**Elevation check command:**
```powershell
wsl.exe --install --no-distribution
```
**Output:**
```text
The requested operation requires elevation.
Error code: Wsl/ERROR_NOT_SUPPORTED
```

### 6.3 Turnkey WSL Migration Automation (`setup_wsl_env.sh`)

To enable seamless recreation the moment the host administrator enables Virtual Machine Platform, the production script `setup_wsl_env.sh` was created in the project root:

```bash
#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="$HOME/venvs/plant-disease-detection"
PROJECT_DIR="/mnt/d/plant-disease-detection"

mkdir -p "$HOME/venvs"
python3.12 -m venv "$TARGET_DIR"
source "$TARGET_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements.txt"
python -m pip check

cd "$PROJECT_DIR"
python training/verify_v2_artifacts.py
python tests/test_v2_inference.py
```

### 6.4 Safety of Existing Assets

- `wsl_venv/` is explicitly listed in `.gitignore` (line 5) and is completely untracked by Git (`git ls-files wsl_venv` returns empty).
- The Windows `.venv` operates completely independently using Python 3.10.0 and was unaffected.
- To prevent permanent data loss of the existing Linux site-packages before the new WSL instance can be verified, `wsl_venv` was preserved intact.

---

## 7. Post-Implementation Regression and Safety Verification (Task 4)

### 7.1 Checkpoint Hash Verification (`training/verify_v2_artifacts.py`)

All three production V2 checkpoints were verified against their pre-implementation cryptographic hashes:

| Checkpoint | File Size (Bytes) | SHA-256 Hash | Status |
|---|---|---|:---:|
| `models/v2/species_expert_v2.keras` | 23,924,105 | `38c1eaef82e4cbe9476ff0bf9128236d0f1a92d0c1ad6ab86387896276f65017` | **MATCH (UNTOUCHED)** |
| `models/v2/potato_expert_v2.keras` | 27,407,925 | `d0111cb8e4561fbbc2aefecfe9ce63bf3a36a1466a9290feb8b7ba535e17b1af` | **MATCH (UNTOUCHED)** |
| `models/v2/tomato_expert_v2.keras` | 27,407,925 | `6b57b2b7d25bd451f88da3d1e82b26d71f110b1ff3f4263813904fffd80dfd14` | **MATCH (UNTOUCHED)** |

### 7.2 Regression Suite Verification (`tests/test_v2_inference.py`)

Execution of the full 7-suite regression suite verified:
- **Test 1 (Safe Deserialization & CBAM):** Deserialized with `safe_mode=True`. CBAM parameters reconciled to exactly 206,258. (**PASSED**)
- **Test 2 (Trainable Parameters):** Species total: 2,340,098 (trainable: 1,745,474). Potato Doctor total: 2,628,597 (trainable: 2,033,973). (**PASSED**)
- **Test 3 (Preprocessing Formats):** RGB, RGBA with white alpha blending, Grayscale, CMYK, and NumPy verified in $[0.0, 255.0]$. (**PASSED**)
- **Test 4 (Species Confidence Gate & Abstention):** 90% boundary condition verified; species uncertainty correctly triggers diagnosis abstention. (**PASSED**)
- **Test 5 (Late Blight Safety Gate):** 90% boundary condition verified; marginal confidence triggers `late_blight_uncertain`. (**PASSED**)
- **Test 6 (Advisory Safety Review):** Chemical prescription suppression confirmed for uncertain diagnoses. (**PASSED**)
- **Test 7 (Asset Preservation & Errors):** V1 legacy files remain preserved; non-existent image triggers `FileNotFoundError`. (**PASSED**)

**Result:** `ALL 7 PHASE 4.1 AUDIT SUITES PASSED WITH 100% SUCCESS!`

### 7.3 Smoke Test Suite (`training/run_preflight_and_smoke_tests.py`)

All 5 preflight hardware and model architecture audits passed with 100% success:
- Audit 1/5: Python 3.10.0, TF 2.21.0, Keras 3.12.1, 15.68 GB RAM.
- Audit 2/5: Data pipeline batching ($4 \times 224 \times 224 \times 3$).
- Audit 3/5: Model construction & CBAM tensor wiring.
- Audit 4/5: Forward pass & gradient step execution.
- Audit 5/5: Checkpoint serialization and reload.

### 7.4 End-to-End Inference Verification (`predict_hierarchical_v2.py`)

Execution on `test_images/sample_leaf.JPG`:
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
    "execution_time_ms": 1222.36,
    "architecture": "MobileNetV2 + CBAM Attention"
  }
}
```

---

## 8. Git Working-Tree Status

**Command executed:**
```powershell
git status
```
**Output:**
```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   export_to_onnx.py
	modified:   requirements.txt

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	reports/PHASE_1_ENVIRONMENT_AND_DEPLOYMENT_HARDENING.md
	setup_wsl_env.sh

no changes added to commit (use "git add" and/or "git commit -a")
```

All existing commits and branches remain unmodified in strict adherence to Git rules.

---

## 9. Required Next Actions for Complete WSL Activation

To transition WSL isolation from **BLOCKED** to **COMPLETE**:
1. Run PowerShell as Administrator on the host machine and enable Virtual Machine Platform:
   ```powershell
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```
2. Ensure hardware virtualization (Intel VT-x or AMD-V) is enabled in machine BIOS/UEFI settings.
3. Reboot the host machine.
4. Launch WSL and execute the provided turnkey automation script:
   ```bash
   bash /mnt/d/plant-disease-detection/setup_wsl_env.sh
   ```
5. Remove the obsolete NTFS folder once verified:
   ```powershell
   Remove-Item -Recurse -Force D:\plant-disease-detection\wsl_venv
   ```
