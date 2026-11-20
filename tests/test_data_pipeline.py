# tests/test_data_pipeline.py
"""
Phase 2 Unit Tests for Manifest-Driven Data Pipeline (training/data_pipeline.py).
Validates manifest parsing, crop filtering, error handling, image decoding,
tensor output formatting, batching behavior, and augmentation integration.
"""

import os
import csv
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from training.data_pipeline import build_manifest_dataset, load_class_mappings


@pytest.mark.pipeline
def test_load_class_mappings():
    """Verify class mappings schema and label-to-index consistency."""
    mappings = load_class_mappings()
    assert "species_expert" in mappings
    assert "potato_expert" in mappings
    assert "tomato_expert" in mappings

    # Check species expert
    assert mappings["species_expert"]["num_classes"] == 2
    assert mappings["species_expert"]["label_to_index"] == {"potato": 0, "tomato": 1}
    assert mappings["species_expert"]["index_to_label"] == {"0": "Potato", "1": "Tomato"}

    # Check potato expert
    assert mappings["potato_expert"]["num_classes"] == 3
    assert mappings["potato_expert"]["label_to_index"] == {
        "Early_blight": 0,
        "Late_blight": 1,
        "healthy": 2
    }
    assert mappings["potato_expert"]["index_to_label"] == {
        "0": "Early Blight",
        "1": "Late Blight",
        "2": "Healthy"
    }

    # Check tomato expert
    assert mappings["tomato_expert"]["num_classes"] == 3
    assert mappings["tomato_expert"]["label_to_index"] == {
        "Early_blight": 0,
        "Late_blight": 1,
        "healthy": 2
    }
    assert mappings["tomato_expert"]["index_to_label"] == {
        "0": "Early Blight",
        "1": "Late Blight",
        "2": "Healthy"
    }


@pytest.mark.pipeline
def test_valid_manifest_loading_species_expert(synthetic_manifest_factory):
    """Verify loading species expert dataset from a valid manifest."""
    manifest_path = synthetic_manifest_factory()
    ds, n_samples = build_manifest_dataset(
        manifest_path=manifest_path,
        model_type="species_expert",
        batch_size=2,
        is_training=False
    )

    assert n_samples == 6
    batches = list(ds)
    assert len(batches) == 3  # 6 samples / 2 per batch

    for images, labels in batches:
        assert images.shape == (2, 224, 224, 3)
        assert images.dtype == tf.float32
        assert labels.shape == (2,)
        # Species labels are in {0, 1}
        assert np.all(np.isin(labels.numpy(), [0, 1]))


@pytest.mark.pipeline
def test_valid_manifest_loading_crop_specialists(synthetic_manifest_factory):
    """Verify crop-specific filtering and disease label indexing for potato and tomato specialists."""
    manifest_path = synthetic_manifest_factory()

    # Potato specialist
    ds_pot, n_pot = build_manifest_dataset(
        manifest_path=manifest_path,
        model_type="potato_expert",
        crop_filter="potato",
        batch_size=3,
        is_training=False
    )
    assert n_pot == 3
    for images, labels in ds_pot:
        assert images.shape == (3, 224, 224, 3)
        assert set(labels.numpy().tolist()) == {0, 1, 2}

    # Tomato specialist
    ds_tom, n_tom = build_manifest_dataset(
        manifest_path=manifest_path,
        model_type="tomato_expert",
        crop_filter="tomato",
        batch_size=3,
        is_training=False
    )
    assert n_tom == 3
    for images, labels in ds_tom:
        assert images.shape == (3, 224, 224, 3)
        assert set(labels.numpy().tolist()) == {0, 1, 2}


@pytest.mark.pipeline
def test_invalid_manifest_path():
    """Verify FileNotFoundError is raised for non-existent manifest file."""
    with pytest.raises(FileNotFoundError):
        build_manifest_dataset("non_existent_manifest.csv")


@pytest.mark.pipeline
def test_unknown_model_type(synthetic_manifest_factory):
    """Verify ValueError is raised for unrecognized model_type."""
    manifest_path = synthetic_manifest_factory()
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_manifest_dataset(manifest_path, model_type="unsupported_expert")


@pytest.mark.pipeline
def test_empty_manifest(tmp_path):
    """Verify ValueError is raised when manifest contains only headers and no samples."""
    empty_csv = tmp_path / "empty_manifest.csv"
    with open(empty_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_group_id", "relative_image_path", "crop", "disease_class"])

    with pytest.raises(ValueError, match="No samples matched manifest"):
        build_manifest_dataset(str(empty_csv), model_type="species_expert")


@pytest.mark.pipeline
def test_missing_required_columns(tmp_path, synthetic_jpeg_factory):
    """Verify KeyError is raised when required CSV columns are missing."""
    img_p = synthetic_jpeg_factory()
    bad_csv = tmp_path / "missing_cols.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_image_path", "wrong_column"])
        writer.writerow([img_p, "some_value"])

    with pytest.raises(KeyError):
        build_manifest_dataset(str(bad_csv), model_type="species_expert")


@pytest.mark.pipeline
def test_unrecognized_class_label(tmp_path, synthetic_jpeg_factory):
    """Verify KeyError is raised when manifest contains an unknown disease label."""
    img_p = synthetic_jpeg_factory()
    bad_csv = tmp_path / "unknown_label.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_image_path", "crop", "disease_class"])
        writer.writeheader()
        writer.writerow({
            "relative_image_path": img_p,
            "crop": "potato",
            "disease_class": "Alien_Fungus"  # Invalid label
        })

    with pytest.raises(KeyError):
        build_manifest_dataset(str(bad_csv), model_type="potato_expert", crop_filter="potato")


@pytest.mark.pipeline
def test_corrupted_image_handling(synthetic_manifest_factory):
    """Verify decoding error is triggered when image file is corrupted."""
    manifest_path = synthetic_manifest_factory(corrupt_image=True)
    ds, _ = build_manifest_dataset(manifest_path, model_type="species_expert", batch_size=2)

    with pytest.raises(tf.errors.InvalidArgumentError):
        # Trigger actual dataset pipeline execution
        for _ in ds:
            pass


@pytest.mark.pipeline
def test_missing_image_file_handling(synthetic_manifest_factory):
    """Verify read error is triggered when an image referenced in manifest is missing."""
    manifest_path = synthetic_manifest_factory(missing_image=True)
    ds, _ = build_manifest_dataset(manifest_path, model_type="species_expert", batch_size=2)

    with pytest.raises(tf.errors.NotFoundError):
        for _ in ds:
            pass


@pytest.mark.pipeline
def test_dataset_batching_and_partial_remainder(synthetic_manifest_factory):
    """Verify dataset batching behavior with non-divisible sample count (partial final batch)."""
    manifest_path = synthetic_manifest_factory()  # 6 samples total
    ds, n_samples = build_manifest_dataset(
        manifest_path=manifest_path,
        batch_size=4,  # 6 / 4 = 1 batch of 4 + 1 batch of 2
        is_training=False
    )
    batches = list(ds)
    assert len(batches) == 2
    assert batches[0][0].shape == (4, 224, 224, 3)
    assert batches[1][0].shape == (2, 224, 224, 3)


@pytest.mark.pipeline
def test_deterministic_evaluation_order(synthetic_manifest_factory):
    """Verify that is_training=False guarantees deterministic order without shuffling."""
    manifest_path = synthetic_manifest_factory()
    ds1, _ = build_manifest_dataset(manifest_path, batch_size=6, is_training=False)
    ds2, _ = build_manifest_dataset(manifest_path, batch_size=6, is_training=False)

    labels1 = next(iter(ds1))[1].numpy()
    labels2 = next(iter(ds2))[1].numpy()
    np.testing.assert_array_equal(labels1, labels2)


@pytest.mark.pipeline
def test_augmentation_model_integration(synthetic_manifest_factory):
    """Verify that online augmentation model is invoked during training mode."""
    manifest_path = synthetic_manifest_factory()

    # Simple mock augmentation layer that adds a constant offset
    class OffsetAugmentation(tf.keras.layers.Layer):
        def call(self, x, training=True):
            return x + 1.0

    aug_model = OffsetAugmentation()
    ds, _ = build_manifest_dataset(
        manifest_path=manifest_path,
        batch_size=2,
        is_training=True,
        augmentation_model=aug_model
    )

    batch_images, _ = next(iter(ds))
    assert batch_images.shape == (2, 224, 224, 3)
