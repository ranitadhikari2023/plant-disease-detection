# Plant Pathology Dataset — Solanaceous Crops

This directory contains raw agricultural foliar imagery utilized for training and evaluating the hierarchical plant pathology detection models.

> **Note:** Raw image files within this directory are intentionally excluded from Git version control via `.gitignore` to maintain a lightweight repository footprint. Full specimen splits, deduplication manifests, and group IDs are versioned in [`audit/`](../audit/).

---

## 1. Crop Species & Diagnostic Classes

The dataset comprises high-resolution field and laboratory specimens across two primary Solanaceous crops:

### Potato (*Solanum tuberosum*)
- **Early Blight** (*Alternaria solani*) — Characteristic concentric rings and target-like lesions.
- **Late Blight** (*Phytophthora infestans*) — High-consequence water-soaked necrotic lesions.
- **Healthy** — Asymptomatic vegetative foliage.

### Tomato (*Solanum lycopersicum*)
- **Early Blight** (*Alternaria solani*) — Brown-black necrotic spots with chlorotic halos.
- **Late Blight** (*Phytophthora infestans*) — Rapidly expanding pale-green to brown lesions.
- **Healthy** — Intact asymptomatic foliar structures.

---

## 2. Directory Layout

```text
dataset/
├── potato/
│   ├── Early_blight/
│   ├── Late_blight/
│   └── healthy/
└── tomato/
    ├── Early_blight/
    ├── Late_blight/
    └── healthy/
```

---

## 3. Data Integrity & Specimen Manifests

To prevent data leakage caused by duplicated or augmented field images appearing across training and evaluation splits, canonical specimen grouping manifests are maintained in the [`audit/`](../audit/) directory:

- `audit/specimen_manifest.csv` — Master catalog of all distinct specimen groups.
- `audit/train_specimen_manifest.csv` — Clean training split (specimen-grouped).
- `audit/validation_specimen_manifest.csv` — Validation split for model tuning.
- `audit/test_specimen_manifest.csv` — Held-out test split for golden benchmarking.
