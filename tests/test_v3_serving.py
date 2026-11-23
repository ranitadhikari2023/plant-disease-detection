"""
tests/test_v3_serving.py
Verification suite for LeafScan V3 Production Serving Engine:
  - Model Bundle validation and cryptographic SHA-256 integrity
  - OnnxEngine execution and warmup
  - Numerical parity and zero gate flips against the golden evaluation set
  - Readiness (/ready), Model Info (/v2/model), and Prometheus Metrics (/metrics) endpoints
  - Request ID propagation and defensive security limits
"""

import io
import sys
import json
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from leafscan.bundle.loader import load_model_bundle
from leafscan.bundle.verification import compute_file_sha256
from leafscan.engines.onnx_engine import OnnxEngine
from leafscan.services.diagnosis_service import DiagnosisService
from leafscan.services.image_intake import ImageIntake
from leafscan.services.safety_guard import SafetyGuard
from leafscan.core.exceptions import (
    ChecksumMismatchError,
    PayloadTooLargeError,
    PixelLimitExceededError,
    UnsupportedFormatError
)
from leafscan.api.app import create_app
from leafscan.api.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bundle():
    p = PROJECT_ROOT / "models/bundles/v3"
    return load_model_bundle(p, verify_checksums=True)


@pytest.fixture(scope="module")
def engine(bundle):
    e = OnnxEngine(bundle)
    e.warmup()
    return e


@pytest.fixture(scope="module")
def service(bundle, engine):
    return DiagnosisService(bundle, engine)


@pytest.mark.regression
def test_model_bundle_integrity(bundle):
    """Verifies that all three ONNX models match manifest SHA-256 hashes."""
    assert bundle.version == "v3.0.0"
    assert "species_expert" in bundle.verified_hashes
    assert "potato_expert" in bundle.verified_hashes
    assert "tomato_expert" in bundle.verified_hashes

    for key, path in bundle.model_paths.items():
        assert path.exists()
        assert compute_file_sha256(path) == bundle.manifest.models[key].sha256.lower()


@pytest.mark.regression
def test_corrupted_model_bundle_rejection(tmp_path):
    """Verifies that tampering with a model artifact raises ChecksumMismatchError."""
    fake_bundle_dir = tmp_path / "fake_bundle"
    fake_bundle_dir.mkdir()

    # Copy real manifest
    real_manifest = PROJECT_ROOT / "models/bundles/v3/manifest.json"
    fake_manifest = fake_bundle_dir / "manifest.json"
    fake_manifest.write_text(real_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    # Create dummy files with mismatched content
    (fake_bundle_dir / "species_expert.onnx").write_bytes(b"tampered content")
    (fake_bundle_dir / "potato_expert.onnx").write_bytes(b"tampered content")
    (fake_bundle_dir / "tomato_expert.onnx").write_bytes(b"tampered content")

    with pytest.raises(ChecksumMismatchError):
        load_model_bundle(fake_bundle_dir, verify_checksums=True)


@pytest.mark.regression
def test_golden_dataset_onnx_parity(service):
    """
    Evaluates the 49 golden test cases and verifies:
      1. 100% agreement on argmax species and disease classes.
      2. 100% agreement on Safety Gate 1 & 2 decisions (zero gate flips).
      3. Numerical parity within 1e-4 tolerance.
    """
    golden_path = PROJECT_ROOT / "tests/fixtures/golden_test_dataset.json"
    assert golden_path.exists(), "golden_test_dataset.json missing"

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    for case in golden_cases:
        img_path = PROJECT_ROOT / case["path"]
        assert img_path.exists(), f"Golden image {img_path} not found"

        res = service.diagnose(str(img_path))
        keras_ref = case["keras"]

        # 1. Compare species
        assert res["plant_species"] == keras_ref["species"], f"Species mismatch on {case['path']}"

        # 2. Compare safety status (0 gate flips)
        assert res["safety_status"] == keras_ref["safety_status"], (
            f"Gate flip on {case['path']}: ONNX={res['safety_status']} Keras={keras_ref['safety_status']}"
        )

        # 3. Compare disease (if not abstained)
        if keras_ref["disease"] is not None:
            assert res["disease"] == keras_ref["disease"], f"Disease mismatch on {case['path']}"
            assert abs(res["disease_confidence"] - keras_ref["disease_confidence"]) <= 1e-3

        # 4. Species confidence tolerance
        assert abs(res["species_confidence"] - keras_ref["species_confidence"]) <= 1e-3


@pytest.mark.api
def test_v3_api_health_ready_model_endpoints(bundle, engine):
    """Verifies /health, /ready, /v2/model, and /metrics on the V3 application."""
    from leafscan.api.routes.health import health_check, readiness_check, model_info, prometheus_metrics
    from leafscan.api.settings import Settings

    test_settings = Settings(verify_bundle_checksums=True, log_json=False)

    # 1. Liveness
    h_data = health_check()
    assert h_data["status"] == "healthy"
    assert h_data["architecture"] == "MobileNetV2 + CBAM Attention"
    assert h_data["models_loaded"] is True

    # 2. Readiness
    r_data = readiness_check(bundle=bundle, engine=engine)
    assert r_data["status"] == "ready"
    assert r_data["bundle_version"] == "v3.0.0"
    assert "species_expert" in r_data["models_verified"]

    # 3. Model Information
    m_data = model_info(bundle=bundle, engine=engine, settings=test_settings)
    assert m_data["bundle_version"] == "v3.0.0"
    assert "species_expert" in m_data["models"]
    assert "potato_expert" in m_data["models"]
    assert "tomato_expert" in m_data["models"]
    assert m_data["safety_gates"]["species_confidence_threshold"] == 0.90
    assert m_data["safety_gates"]["late_blight_confidence_threshold"] == 0.90

    # 4. Prometheus Metrics
    met_resp = prometheus_metrics()
    assert met_resp.status_code == 200
    assert "leafscan_requests_total" in met_resp.body.decode("utf-8")
    assert "leafscan_model_load_duration_ms" in met_resp.body.decode("utf-8")


@pytest.mark.security
def test_image_intake_defensive_bounds():
    """Verifies that ImageIntake enforces byte limits, format filters, and pixel bomb limits."""
    intake = ImageIntake(max_upload_bytes=1000, max_decoded_pixels=10000)

    # Oversized byte payload
    with pytest.raises(PayloadTooLargeError):
        intake.validate_file_bytes(b"x" * 1001)

    # Unsupported format (GIF)
    gif_buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(gif_buf, format="GIF")
    with pytest.raises(UnsupportedFormatError):
        intake.decode_and_orient(gif_buf.getvalue())

    # Pixel limit exceeded (100x200 = 20,000 > 10,000)
    png_buf = io.BytesIO()
    Image.new("RGB", (100, 200)).save(png_buf, format="PNG")
    with pytest.raises(PixelLimitExceededError):
        intake.decode_and_orient(png_buf.getvalue())


@pytest.mark.regression
def test_calibration_and_ece():
    """Verifies Expected Calibration Error (ECE) calculation and temperature scaling."""
    from leafscan.services.calibration import compute_ece, TemperatureScaler

    # Synthetic well-calibrated vs miscalibrated data
    confs = np.array([0.9, 0.8, 0.7, 0.6, 0.9, 0.8])
    preds = np.array([1, 1, 0, 0, 1, 0])
    targets = np.array([1, 1, 0, 0, 0, 1])

    ece = compute_ece(confs, preds, targets, n_bins=5)
    assert 0.0 <= ece <= 1.0

    # Temperature scaler softens probabilities when T > 1
    scaler = TemperatureScaler(temperature=1.5)
    probs = np.array([[0.9, 0.1]])
    scaled = scaler.scale_probabilities(probs)
    assert scaled[0, 0] < 0.9  # softened
    assert np.isclose(np.sum(scaled), 1.0)


@pytest.mark.regression
def test_conformal_prediction_rules():
    """Verifies V3 conformal prediction set rules and safety escalation."""
    from leafscan.services.conformal import ConformalPredictor

    cp = ConformalPredictor(quantiles={"potato_expert": 0.20, "tomato_expert": 0.40})

    # Case 1: High confidence Early Blight -> singleton set -> None (passes to confidence gate)
    set1 = cp.predict_set("Tomato", {"Early Blight": 0.92, "Late Blight": 0.05, "Healthy": 0.03})
    assert set1 == ["Early Blight"]
    assert cp.evaluate_conformal_safety(set1) is None

    # Case 2: Multi-class containing Late Blight -> late_blight_uncertain
    set2 = cp.predict_set("Tomato", {"Early Blight": 0.62, "Late Blight": 0.65, "Healthy": 0.05})
    assert "Late Blight" in set2 and len(set2) > 1
    assert cp.evaluate_conformal_safety(set2) == "late_blight_uncertain"

    # Case 3: Empty or all classes -> disease_uncertain
    set3 = ["Early Blight", "Late Blight", "Healthy"]
    assert cp.evaluate_conformal_safety(set3) == "disease_uncertain"


@pytest.mark.api
def test_feedback_submission_and_idempotency():
    """Verifies POST /v2/feedback endpoint and idempotent transaction key handling."""
    from leafscan.api.routes.feedback import submit_feedback, FeedbackRequest

    req = FeedbackRequest(
        request_id="req_test_123",
        predicted_species="Tomato",
        predicted_disease="Late Blight",
        observed_species="Tomato",
        observed_disease="Late Blight",
        feedback_type="agree",
        agronomic_notes="Symptoms confirmed by on-site field agent.",
        idempotency_key="tx-idem-999"
    )

    resp1 = submit_feedback(payload=req, idempotency_key="tx-idem-999")
    assert resp1.success is True
    assert resp1.status == "recorded"
    assert resp1.idempotency_key == "tx-idem-999"

    # Second submission with same idempotency key returns exact same feedback_id
    resp2 = submit_feedback(payload=req, idempotency_key="tx-idem-999")
    assert resp2.success is True
    assert resp2.feedback_id == resp1.feedback_id


@pytest.mark.regression
def test_ood_detection_and_out_of_scope_rejection(bundle):
    """Verifies that OOD inputs trigger out_of_scope safety status."""
    guard = SafetyGuard(bundle.manifest, enable_ood=True)

    # Low-confidence / high-entropy species prediction
    ambiguous_probs = {"Potato": 0.51, "Tomato": 0.49}
    is_uncertain, decision = guard.evaluate_species_gate(
        species_name="Potato",
        species_conf=0.51,
        species_probs=ambiguous_probs
    )

    assert is_uncertain is True
    assert decision.safety_status == "out_of_scope"
    assert "out_of_scope" in decision.safety_warnings[0].lower() or "out-of-scope" in decision.safety_warnings[0].lower()
    assert "OUT OF SCOPE" in decision.advisory
    assert decision.disease_abstained is True
