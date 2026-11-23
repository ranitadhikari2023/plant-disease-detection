# LeafScan V3 — Emergency Rollback Guide

**Document Reference:** `docs/ROLLBACK_GUIDE.md`  
**Target:** Emergency Incident Remediation & Model Rollback  
**Version:** 3.0.0  

---

## 1. Rollback Scenarios

LeafScan V3 features a completely stateless architecture with versioned model bundles. Because user uploads and predictions are never persisted to a stateful database by default, rollbacks are immediate, low-risk, and require zero database migrations.

---

## 2. Model Bundle Rollback (Level 1: Fastest)

If a newly deployed model bundle exhibits unexpected false positives, safety gate shifts, or higher latency:

### Procedure:
1. Revert the `LEAFSCAN_MODEL_BUNDLE_PATH` environment variable to the previous verified bundle directory:
   ```bash
   # In docker-compose.yaml or Kubernetes deployment:
   LEAFSCAN_MODEL_BUNDLE_PATH=/app/models/bundles/v2_fallback
   ```
2. Restart the application workers:
   ```bash
   docker compose restart api
   ```
3. Verify readiness:
   ```bash
   curl -f http://localhost:8000/ready
   ```
   Confirm that the active `bundle_version` matches the intended rollback target.

---

## 3. Container Image Rollback (Level 2: Standard)

If a regression occurs in the web service, middleware, or dependencies:

### Procedure:
1. Retag or deploy the previous stable container image:
   ```bash
   docker pull ghcr.io/your-org/leafscan:v2.0.0  # or previous release tag
   docker tag ghcr.io/your-org/leafscan:v2.0.0 leafscan:active
   docker compose up -d
   ```
2. Run smoke tests:
   ```bash
   python predict_hierarchical_v2.py --image test_images/sample_leaf.JPG --json
   ```

---

## 4. Emergency Keras Fallback (Level 3: Development / Debugging)

If an unforeseen edge-case requires falling back from ONNX Runtime to the reference Keras 3 engine during an emergency investigation:

In Python code or test harnesses:
```python
from inference.predictor import HierarchicalPredictorV2
# Force Keras 3 execution
predictor = HierarchicalPredictorV2(use_keras_fallback=True)
result = predictor.predict("test_images/sample_leaf.JPG")
```

The backward-compatibility adapter in `inference/predictor.py` preserves both paths.
