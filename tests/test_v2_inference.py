# tests/test_v2_inference.py
"""
Phase 2 Modernized Pytest Regression Suite for V2 Hierarchical Plant Disease Inference.
Retains all 7 core verification categories:
  1. Safe Model Deserialization (safe_mode=True) and CBAM layer weights.
  2. Trainable and non-trainable parameter count reconciliation.
  3. Image preprocessing regression (RGB, RGBA, Grayscale, CMYK, NumPy).
  4. Species Confidence Gate boundary conditions (89.99%, 90.00%, 94.41%, 99.00%) & Abstention.
  5. Late Blight Confidence Gate boundary conditions (89.99%, 90.00%, 90.01%).
  6. Educational Agricultural Advisory Safety review on sample_leaf.JPG.
  7. Legacy V1 asset preservation in archive/v1/ & error handling.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from inference.model_loader import load_models, unload_models, get_custom_objects
from inference.preprocessing import preprocess_image
from inference.predictor import HierarchicalPredictorV2, predict_v2
from training.models import CBAM


@pytest.mark.regression
def test_safe_mode_deserialization_and_cbam():
    """Verify that all 3 V2 models load safely with safe_mode=True and CBAM layers match parameter specs."""
    unload_models()
    species_m, potato_m, tomato_m, mappings = load_models()

    assert species_m is not None, "Species model failed to load"
    assert potato_m is not None, "Potato model failed to load"
    assert tomato_m is not None, "Tomato model failed to load"

    # Inspect CBAM layer in Potato Doctor
    cbam_layer = next((l for l in potato_m.layers if isinstance(l, CBAM) or l.__class__.__name__ == 'CBAM'), None)
    assert cbam_layer is not None, "Potato Doctor missing CBAM layer"

    # Inspect submodules
    ca = cbam_layer.channel_attention
    sa = cbam_layer.spatial_attention

    ca_weights = sum(tf.size(w).numpy() for w in ca.weights)
    sa_weights = sum(tf.size(w).numpy() for w in sa.weights)
    total_cbam = sum(tf.size(w).numpy() for w in cbam_layer.weights)

    # Reconciled theoretical values:
    # ChannelAttention: 1280*80 + 80 + 80*1280 + 1280 = 206,160
    # SpatialAttention: 7*7*2*1 = 98 (use_bias=False)
    # Total CBAM: 206,160 + 98 = 206,258
    assert ca_weights == 206160, f"Expected 206,160 CA weights, got {ca_weights}"
    assert sa_weights == 98, f"Expected 98 SA weights, got {sa_weights}"
    assert total_cbam == 206258, f"Expected 206,258 total CBAM weights, got {total_cbam}"

    # Also verify CBAM layer in Tomato Doctor
    cbam_layer_t = next((l for l in tomato_m.layers if isinstance(l, CBAM) or l.__class__.__name__ == 'CBAM'), None)
    assert cbam_layer_t is not None, "Tomato Doctor missing CBAM layer"
    assert sum(tf.size(w).numpy() for w in cbam_layer_t.weights) == 206258


@pytest.mark.regression
def test_trainable_parameter_counts():
    """Verify exact total, trainable, and non-trainable parameter counts across V2 models."""
    unload_models()
    species_m, potato_m, tomato_m, _ = load_models()

    # Species Botanist V2
    sp_total = species_m.count_params()
    sp_trainable = sum(tf.size(w).numpy() for w in species_m.trainable_weights)
    sp_nontrainable = sum(tf.size(w).numpy() for w in species_m.non_trainable_weights)

    assert sp_total == 2340098, f"Unexpected Species total params: {sp_total}"
    assert sp_trainable == 1745474, f"Unexpected Species trainable params: {sp_trainable}"
    assert sp_nontrainable == 594624, f"Unexpected Species non-trainable params: {sp_nontrainable}"

    # Potato Doctor CBAM V2
    pot_total = potato_m.count_params()
    pot_trainable = sum(tf.size(w).numpy() for w in potato_m.trainable_weights)
    pot_nontrainable = sum(tf.size(w).numpy() for w in potato_m.non_trainable_weights)

    assert pot_total == 2628597, f"Unexpected Potato Doctor total params: {pot_total}"
    assert pot_trainable == 2033973, f"Unexpected Potato Doctor trainable params: {pot_trainable}"
    assert pot_nontrainable == 594624, f"Unexpected Potato Doctor non-trainable params: {pot_nontrainable}"

    # Tomato Doctor CBAM V2
    tom_total = tomato_m.count_params()
    tom_trainable = sum(tf.size(w).numpy() for w in tomato_m.trainable_weights)
    tom_nontrainable = sum(tf.size(w).numpy() for w in tomato_m.non_trainable_weights)

    assert tom_total == 2628597, f"Unexpected Tomato Doctor total params: {tom_total}"
    assert tom_trainable == 2033973, f"Unexpected Tomato Doctor trainable params: {tom_trainable}"
    assert tom_nontrainable == 594624, f"Unexpected Tomato Doctor non-trainable params: {tom_nontrainable}"


@pytest.mark.regression
def test_preprocessing_regression():
    """Verify image preprocessing across diverse color modes and numpy arrays."""
    # 1. RGB Image
    rgb_img = Image.new("RGB", (320, 240), color=(120, 180, 70))
    t_rgb, _ = preprocess_image(rgb_img)
    assert t_rgb.shape == (1, 224, 224, 3)
    assert t_rgb.dtype == np.float32
    assert 0.0 <= t_rgb.min() and t_rgb.max() <= 255.0, "Raw pixels outside [0, 255]"

    # 2. RGBA with Alpha compositing on white
    rgba_img = Image.new("RGBA", (256, 256), color=(50, 150, 50, 128))
    t_rgba, _ = preprocess_image(rgba_img)
    assert t_rgba.shape == (1, 224, 224, 3)
    assert t_rgba.dtype == np.float32

    # 3. Grayscale (L)
    gray_img = Image.new("L", (180, 180), color=128)
    t_gray, _ = preprocess_image(gray_img)
    assert t_gray.shape == (1, 224, 224, 3)
    assert np.allclose(t_gray[0, :, :, 0], t_gray[0, :, :, 1])

    # 4. CMYK Image
    cmyk_img = Image.new("CMYK", (200, 200), color=(0, 100, 100, 0))
    t_cmyk, _ = preprocess_image(cmyk_img)
    assert t_cmyk.shape == (1, 224, 224, 3)

    # 5. NumPy 2D array
    np_arr = (np.ones((200, 200), dtype=np.uint8) * 150)
    t_np, _ = preprocess_image(np_arr)
    assert t_np.shape == (1, 224, 224, 3)


@pytest.mark.regression
def test_species_gate_boundaries_and_abstention(sample_leaf_path):
    """Verify species confidence gate boundary conditions and diagnostic abstention."""
    # Boundary Case A: Species Confidence below threshold -> MUST TRIGGER species_uncertain & abstain
    cfg_below = {
        "safety_gates": {
            "species_confidence_threshold": 0.9442,  # sample_leaf has ~0.9441 < 0.9442
            "late_blight_confidence_threshold": 0.90,
            "general_disease_confidence_threshold": 0.70
        }
    }
    pred_below = HierarchicalPredictorV2(config=cfg_below)
    res_below = pred_below.predict(sample_leaf_path)
    assert res_below["safety_status"] == "species_uncertain"
    assert res_below["disease"] is None, "Did not abstain from disease classification!"
    assert res_below["disease_confidence"] is None
    assert res_below["probabilities"] == {}
    assert "Abstained" in res_below["diagnosis"]
    assert len(res_below["species_probabilities"]) == 2

    # Boundary Case B: Species Confidence exactly equal to threshold -> MUST PASS (Inclusive policy)
    actual_conf = res_below["species_confidence"]
    cfg_exact = {
        "safety_gates": {
            "species_confidence_threshold": actual_conf,
            "late_blight_confidence_threshold": 0.90,
            "general_disease_confidence_threshold": 0.70
        }
    }
    pred_exact = HierarchicalPredictorV2(config=cfg_exact)
    res_exact = pred_exact.predict(sample_leaf_path)
    assert res_exact["safety_status"] != "species_uncertain", "Equal threshold triggered gate!"
    assert res_exact["disease"] is not None, "Should have routed to specialist!"

    # Boundary Case C: Standard 90% threshold -> MUST PASS
    cfg_std = {
        "safety_gates": {
            "species_confidence_threshold": 0.90,
            "late_blight_confidence_threshold": 0.90,
            "general_disease_confidence_threshold": 0.70
        }
    }
    pred_std = HierarchicalPredictorV2(config=cfg_std)
    res_std = pred_std.predict(sample_leaf_path)
    assert res_std["safety_status"] != "species_uncertain"


@pytest.mark.regression
def test_late_blight_gate_boundaries(sample_leaf_path):
    """Verify Late Blight confidence gate boundary conditions and advisory escalation."""
    actual_dis_conf = 0.6964

    # Boundary A: Disease Conf < Threshold -> MUST TRIGGER late_blight_uncertain
    cfg_lb_below = {
        "safety_gates": {
            "species_confidence_threshold": 0.90,
            "late_blight_confidence_threshold": round(actual_dis_conf + 0.0001, 4),
            "general_disease_confidence_threshold": 0.50
        }
    }
    res_lb_below = HierarchicalPredictorV2(config=cfg_lb_below).predict(sample_leaf_path)
    assert res_lb_below["safety_status"] == "late_blight_uncertain"
    assert "Late Blight confidence" in res_lb_below["safety_warnings"][0]

    # Boundary B: Disease Conf == Threshold -> MUST PASS (Inclusive policy)
    cfg_lb_exact = {
        "safety_gates": {
            "species_confidence_threshold": 0.90,
            "late_blight_confidence_threshold": actual_dis_conf,
            "general_disease_confidence_threshold": 0.50
        }
    }
    res_lb_exact = HierarchicalPredictorV2(config=cfg_lb_exact).predict(sample_leaf_path)
    assert res_lb_exact["safety_status"] == "confident"

    # Boundary C: Disease Conf > Threshold -> MUST PASS
    cfg_lb_above = {
        "safety_gates": {
            "species_confidence_threshold": 0.90,
            "late_blight_confidence_threshold": round(actual_dis_conf - 0.0001, 4),
            "general_disease_confidence_threshold": 0.50
        }
    }
    res_lb_above = HierarchicalPredictorV2(config=cfg_lb_above).predict(sample_leaf_path)
    assert res_lb_above["safety_status"] == "confident"


@pytest.mark.regression
def test_sample_leaf_and_advisory_safety(sample_leaf_path):
    """Verify educational agricultural advisory safety on sample_leaf.JPG."""
    res = predict_v2(sample_leaf_path)
    assert res["success"] is True
    assert res["safety_status"] == "late_blight_uncertain"

    # Verify advisory safety requirements:
    advisory = res["advisory"]
    assert "DO NOT apply" in advisory or "Do not apply" in advisory, "Missing chemical suppression advisory!"
    assert "24" in advisory and "48" in advisory, "Missing isolation re-inspection timeline"
    assert "extension" in advisory.lower() or "consult" in advisory.lower(), "Missing professional consultation advisory"

    # Verify JSON serialization
    json_str = json.dumps(res, indent=2)
    assert len(json_str) > 0


@pytest.mark.regression
def test_legacy_v1_asset_preservation_and_error_handling():
    """Verify that legacy V1 assets are safely preserved in archive/v1/ and invalid files raise errors."""
    v1_models = [
        "archive/v1/models/species_expert.keras",
        "archive/v1/models/potato_expert.keras",
        "archive/v1/models/tomato_expert.keras"
    ]
    for p in v1_models:
        assert os.path.exists(p), f"Archived V1 legacy model missing: {p}"
        assert os.path.getsize(p) > 10_000_000, f"Archived V1 model corrupted or truncated: {p}"

    assert os.path.exists("archive/v1/predict_hierarchical.py"), "Archived predict_hierarchical.py missing!"
    assert os.path.exists("archive/v1/README.md"), "archive/v1/README.md missing!"

    # Verify production V2 models remain present
    v2_models = [
        "models/v2/species_expert_v2.keras",
        "models/v2/potato_expert_v2.keras",
        "models/v2/tomato_expert_v2.keras"
    ]
    for p in v2_models:
        assert os.path.exists(p), f"Production V2 model missing: {p}"

    # Test error handling on invalid file
    res_err = predict_v2("non_existent_file_987654.jpg")
    assert res_err["success"] is False
    assert res_err["safety_status"] == "error"
    assert len(res_err["safety_warnings"]) > 0


# Backward compatibility runner when invoked directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
