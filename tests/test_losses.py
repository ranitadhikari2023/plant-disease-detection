# tests/test_losses.py
"""
Phase 2 Unit Tests for Loss Functions and Specimen Class Weighting (training/losses.py).
Validates SparseCategoricalFocalLoss numerical stability, modulation behavior,
gradient flow, serialization, and specimen-level class weighting calculations.
"""

import csv
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from training.losses import SparseCategoricalFocalLoss, compute_specimen_class_weights


# ==============================================================================
# 1. Specimen Class Weighting Tests
# ==============================================================================

@pytest.mark.losses
def test_compute_specimen_class_weights_balanced(tmp_path):
    """Verify that perfectly balanced specimen counts produce equal weights of 1.0."""
    manifest_file = tmp_path / "balanced_manifest.csv"
    headers = ["source_group_id", "crop", "disease_class", "assigned_split"]

    rows = []
    # 10 specimens for each of the 3 potato classes
    for i in range(10):
        rows.append({"source_group_id": f"EB_{i}", "crop": "potato", "disease_class": "Early_blight", "assigned_split": "train"})
        rows.append({"source_group_id": f"LB_{i}", "crop": "potato", "disease_class": "Late_blight", "assigned_split": "train"})
        rows.append({"source_group_id": f"H_{i}", "crop": "potato", "disease_class": "healthy", "assigned_split": "train"})

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    weights, counts = compute_specimen_class_weights(str(manifest_file), crop_filter="potato", model_type="potato_expert")

    assert counts == {"Early_blight": 10, "Late_blight": 10, "healthy": 10}
    # Formula: 30 / (3 * 10) = 1.0
    for idx in [0, 1, 2]:
        assert np.isclose(weights[idx], 1.0)


@pytest.mark.losses
def test_compute_specimen_class_weights_imbalance_and_duplicate_deduplication(tmp_path):
    """
    Verify that:
      1. Repeated rows with the same source_group_id (offline augmentations) are deduplicated.
      2. Rare classes receive proportionally higher weights following w_c = N_total / (K * N_c).
    """
    manifest_file = tmp_path / "imbalanced_manifest.csv"
    headers = ["source_group_id", "crop", "disease_class", "assigned_split"]

    rows = []
    # 20 distinct Early Blight specimens (each appears 1 time)
    for i in range(20):
        rows.append({"source_group_id": f"EB_{i}", "crop": "potato", "disease_class": "Early_blight", "assigned_split": "train"})

    # 10 distinct Late Blight specimens (each appears 1 time)
    for i in range(10):
        rows.append({"source_group_id": f"LB_{i}", "crop": "potato", "disease_class": "Late_blight", "assigned_split": "train"})

    # 2 distinct Healthy specimens, but heavily duplicated (5 duplicate images per specimen = 10 rows)
    for i in range(2):
        for dup in range(5):
            rows.append({"source_group_id": f"H_{i}", "crop": "potato", "disease_class": "healthy", "assigned_split": "train"})

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    weights, counts = compute_specimen_class_weights(str(manifest_file), crop_filter="potato", model_type="potato_expert")

    # True biological counts: 20 Early Blight, 10 Late Blight, 2 Healthy (NOT 10 Healthy!)
    assert counts["Early_blight"] == 20
    assert counts["Late_blight"] == 10
    assert counts["healthy"] == 2

    total_specimens = 32
    num_classes = 3

    # Expected:
    # Early_blight (idx 0): 32 / (3 * 20) = 32 / 60 ~ 0.5333
    # Late_blight  (idx 1): 32 / (3 * 10) = 32 / 30 ~ 1.0667
    # healthy      (idx 2): 32 / (3 * 2)  = 32 / 6  ~ 5.3333
    assert np.isclose(weights[0], total_specimens / (num_classes * 20))
    assert np.isclose(weights[1], total_specimens / (num_classes * 10))
    assert np.isclose(weights[2], total_specimens / (num_classes * 2))


@pytest.mark.losses
def test_compute_specimen_class_weights_species_expert(tmp_path):
    """Verify class weight computation for Botanist Species Expert (2 classes: potato, tomato)."""
    manifest_file = tmp_path / "species_manifest.csv"
    headers = ["source_group_id", "crop", "disease_class", "assigned_split"]

    rows = []
    # 15 potato specimens, 5 tomato specimens
    for i in range(15):
        rows.append({"source_group_id": f"P_{i}", "crop": "potato", "disease_class": "healthy", "assigned_split": "train"})
    for i in range(5):
        rows.append({"source_group_id": f"T_{i}", "crop": "tomato", "disease_class": "healthy", "assigned_split": "train"})

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    weights, counts = compute_specimen_class_weights(str(manifest_file), model_type="species_expert")

    assert counts == {"potato": 15, "tomato": 5}
    # Total: 20, Classes: 2
    # Potato (idx 0): 20 / (2 * 15) = 20 / 30 = 2/3 ~ 0.6667
    # Tomato (idx 1): 20 / (2 * 5)  = 20 / 10 = 2.0
    assert np.isclose(weights[0], 20.0 / (2 * 15))
    assert np.isclose(weights[1], 20.0 / (2 * 5))


@pytest.mark.losses
def test_compute_specimen_class_weights_split_filtering(tmp_path):
    """Verify that validation and test split rows are strictly ignored in weight calculation."""
    manifest_file = tmp_path / "split_manifest.csv"
    headers = ["source_group_id", "crop", "disease_class", "assigned_split"]

    rows = [
        {"source_group_id": "P_01", "crop": "potato", "disease_class": "healthy", "assigned_split": "train"},
        {"source_group_id": "P_02", "crop": "potato", "disease_class": "healthy", "assigned_split": "train"},
        {"source_group_id": "P_03", "crop": "potato", "disease_class": "healthy", "assigned_split": "val"},
        {"source_group_id": "P_04", "crop": "potato", "disease_class": "healthy", "assigned_split": "test"},
        {"source_group_id": "T_01", "crop": "tomato", "disease_class": "healthy", "assigned_split": "train"},
    ]

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    _, counts = compute_specimen_class_weights(str(manifest_file), model_type="species_expert")
    assert counts["potato"] == 2  # Only train rows counted
    assert counts["tomato"] == 1


# ==============================================================================
# 2. SparseCategoricalFocalLoss Tests
# ==============================================================================

@pytest.mark.losses
def test_focal_loss_initialization():
    """Verify focal loss initialization parameters and defaults."""
    loss_fn = SparseCategoricalFocalLoss()
    assert loss_fn.gamma == 2.0
    assert loss_fn.alpha is None
    assert loss_fn.from_logits is False
    assert loss_fn.name == "sparse_focal_loss"

    loss_custom = SparseCategoricalFocalLoss(gamma=3.5, alpha=[0.5, 1.0, 2.0], from_logits=True)
    assert loss_custom.gamma == 3.5
    assert loss_custom.alpha == [0.5, 1.0, 2.0]
    assert loss_custom.from_logits is True


@pytest.mark.losses
def test_focal_loss_shapes_and_non_negativity():
    """Verify loss calculation produces non-negative, finite scalar for 1D and 2D target shapes."""
    loss_fn = SparseCategoricalFocalLoss(gamma=2.0)

    # Targets as 1D vector (B,)
    y_true_1d = tf.constant([0, 1, 2], dtype=tf.int32)
    y_pred = tf.constant([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7]
    ], dtype=tf.float32)

    val_1d = loss_fn(y_true_1d, y_pred)
    assert np.isfinite(val_1d.numpy())
    assert val_1d.numpy() >= 0.0

    # Targets as 2D column vector (B, 1)
    y_true_2d = tf.constant([[0], [1], [2]], dtype=tf.int32)
    val_2d = loss_fn(y_true_2d, y_pred)
    assert np.isclose(val_1d.numpy(), val_2d.numpy())


@pytest.mark.losses
def test_focal_loss_downweights_easy_examples():
    """Verify that confident correct predictions receive exponentially smaller loss than uncertain ones."""
    loss_fn = SparseCategoricalFocalLoss(gamma=2.0)

    # Easy / confident prediction: p_t = 0.99
    y_easy_true = tf.constant([0], dtype=tf.int32)
    y_easy_pred = tf.constant([[0.99, 0.01]], dtype=tf.float32)
    loss_easy = loss_fn(y_easy_true, y_easy_pred).numpy()

    # Hard / ambiguous prediction: p_t = 0.51
    y_hard_true = tf.constant([0], dtype=tf.int32)
    y_hard_pred = tf.constant([[0.51, 0.49]], dtype=tf.float32)
    loss_hard = loss_fn(y_hard_true, y_hard_pred).numpy()

    # Hard loss must be significantly higher than easy loss
    assert loss_hard > loss_easy * 100.0


@pytest.mark.losses
def test_focal_loss_gamma_modulation():
    """Verify that increasing gamma increases down-weighting of easy examples."""
    loss_gamma_0 = SparseCategoricalFocalLoss(gamma=0.0)  # Standard cross-entropy
    loss_gamma_2 = SparseCategoricalFocalLoss(gamma=2.0)  # Moderate focal
    loss_gamma_4 = SparseCategoricalFocalLoss(gamma=4.0)  # Aggressive focal

    y_true = tf.constant([0], dtype=tf.int32)
    y_pred = tf.constant([[0.9, 0.1]], dtype=tf.float32)  # Modestly easy example

    l0 = loss_gamma_0(y_true, y_pred).numpy()
    l2 = loss_gamma_2(y_true, y_pred).numpy()
    l4 = loss_gamma_4(y_true, y_pred).numpy()

    assert l0 > l2 > l4


@pytest.mark.losses
def test_focal_loss_alpha_class_weighting():
    """Verify that class-specific alpha weighting scales loss for specified classes."""
    # Alpha giving class 1 double the weight of class 0
    alpha = [1.0, 2.0]
    loss_fn = SparseCategoricalFocalLoss(gamma=2.0, alpha=alpha)

    # Class 0 prediction with p_t = 0.6
    y0_true = tf.constant([0], dtype=tf.int32)
    y0_pred = tf.constant([[0.6, 0.4]], dtype=tf.float32)
    l0 = loss_fn(y0_true, y0_pred).numpy()

    # Class 1 prediction with same p_t = 0.6
    y1_true = tf.constant([1], dtype=tf.int32)
    y1_pred = tf.constant([[0.4, 0.6]], dtype=tf.float32)
    l1 = loss_fn(y1_true, y1_pred).numpy()

    # l1 should be exactly 2 * l0
    assert np.isclose(l1, 2.0 * l0, rtol=1e-4)


@pytest.mark.losses
def test_focal_loss_numerical_stability():
    """Verify numerical stability with extreme probabilities close to 0.0 and 1.0."""
    loss_fn = SparseCategoricalFocalLoss(gamma=2.0)

    # Extreme probabilities
    y_true = tf.constant([0, 1], dtype=tf.int32)
    y_pred = tf.constant([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=tf.float32)

    val = loss_fn(y_true, y_pred).numpy()
    assert np.isfinite(val)
    assert not np.isnan(val)
    assert not np.isinf(val)


@pytest.mark.losses
def test_focal_loss_gradient_flow():
    """Verify that focal loss supports automatic differentiation and computes finite non-zero gradients."""
    loss_fn = SparseCategoricalFocalLoss(gamma=2.0, from_logits=True)

    logits = tf.Variable([[2.0, 0.5, -1.0], [-0.5, 3.0, 0.0]], dtype=tf.float32)
    y_true = tf.constant([0, 1], dtype=tf.int32)

    with tf.GradientTape() as tape:
        loss = loss_fn(y_true, logits)

    grads = tape.gradient(loss, logits)
    assert grads is not None
    assert grads.shape == logits.shape
    assert np.all(np.isfinite(grads.numpy()))
    assert np.any(grads.numpy() != 0.0)


@pytest.mark.losses
def test_focal_loss_serialization_roundtrip():
    """Verify that SparseCategoricalFocalLoss get_config() supports exact deserialization."""
    loss_orig = SparseCategoricalFocalLoss(gamma=3.2, alpha=[0.25, 0.75], from_logits=True, name="custom_focal")
    config = loss_orig.get_config()

    loss_reconstructed = SparseCategoricalFocalLoss.from_config(config)
    assert loss_reconstructed.gamma == 3.2
    assert loss_reconstructed.alpha == [0.25, 0.75]
    assert loss_reconstructed.from_logits is True
    assert loss_reconstructed.name == "custom_focal"
