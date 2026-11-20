# training/run_preflight_and_smoke_tests.py
"""
Preflight hardware inspection and smoke test suite for Phase 3 models.
Verifies data loading, batching, CBAM architecture, forward pass, gradient step, and serialization.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
import numpy as np

from training.data_pipeline import build_manifest_dataset
from training.augmentation import get_training_augmentation
from training.models import (
    build_species_expert,
    build_crop_doctor_cbam,
    CBAM,
    ChannelAttention,
    SpatialAttention
)
from training.losses import compute_specimen_class_weights, SparseCategoricalFocalLoss

def run_preflight_and_smoke_tests():
    print("=" * 80)
    print("PHASE 3: PREFLIGHT HARDWARE AUDIT & ARCHITECTURE SMOKE TESTS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Environment & Hardware Audit
    # --------------------------------------------------------------------------
    print("\n[Audit 1/5] Hardware and Environment Inspection...")
    py_ver = sys.version.split()[0]
    tf_ver = tf.__version__
    keras_ver = keras.__version__
    gpus = tf.config.list_physical_devices('GPU')
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)

    print(f"   Python Version:     {py_ver}")
    print(f"   TensorFlow Version: {tf_ver}")
    print(f"   Keras Version:      {keras_ver}")
    print(f"   System RAM:         {ram_gb:.2f} GB total ({ram_avail_gb:.2f} GB available)")
    print(f"   GPUs Detected (TF): {len(gpus)}")

    if not gpus:
        print("   ⚠️ GPU Note: TensorFlow native GPU acceleration is not available under Windows for TF >= 2.11.")
        print("      Execution will run under multi-threaded CPU instructions (oneDNN optimized).")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Smoke Tests
    # --------------------------------------------------------------------------
    print("\n[Audit 2/5] Data Pipeline Verification...")
    test_manifest = "audit/train_specimen_manifest.csv"
    aug_model = get_training_augmentation(seed=42)

    configs = [
        ("species_expert", None, 2),
        ("potato_expert", "potato", 3),
        ("tomato_expert", "tomato", 3)
    ]

    for model_type, crop_filter, expected_classes in configs:
        ds, n_samples = build_manifest_dataset(
            manifest_path=test_manifest,
            model_type=model_type,
            crop_filter=crop_filter,
            batch_size=4,
            is_training=True,
            augmentation_model=aug_model
        )
        for images, labels in ds.take(1):
            assert images.shape == (4, 224, 224, 3), f"Bad image batch shape: {images.shape}"
            assert labels.shape == (4,), f"Bad label batch shape: {labels.shape}"
            assert np.all(labels.numpy() < expected_classes), f"Label exceeds expected classes: {labels.numpy()}"
            print(f"   ✅ {model_type:15} pipeline: {n_samples:4d} samples -> Batch Shape: {images.shape}, Labels: {labels.numpy()}")

    # --------------------------------------------------------------------------
    # 3. Model Architecture & CBAM Verification
    # --------------------------------------------------------------------------
    print("\n[Audit 3/5] Model Construction & CBAM Verification...")
    
    # Species
    m_species, _ = build_species_expert()
    print(f"   ✅ Species Expert V2: {m_species.count_params():,} total params, Output: {m_species.output_shape}")

    # Potato Doctor
    m_potato, _ = build_crop_doctor_cbam("Potato_Doctor_CBAM_v2", num_classes=3)
    cbam_layer = m_potato.get_layer("cbam_attention")
    assert isinstance(cbam_layer, CBAM), "CBAM layer missing in Potato Doctor!"
    print(f"   ✅ Potato Doctor V2:  {m_potato.count_params():,} total params (CBAM: {cbam_layer.count_params():,} params), Output: {m_potato.output_shape}")

    # Tomato Doctor
    m_tomato, _ = build_crop_doctor_cbam("Tomato_Doctor_CBAM_v2", num_classes=3)
    cbam_layer_t = m_tomato.get_layer("cbam_attention")
    assert isinstance(cbam_layer_t, CBAM), "CBAM layer missing in Tomato Doctor!"
    print(f"   ✅ Tomato Doctor V2:  {m_tomato.count_params():,} total params (CBAM: {cbam_layer_t.count_params():,} params), Output: {m_tomato.output_shape}")

    # --------------------------------------------------------------------------
    # 4. Forward Pass & Training Step Smoke Test
    # --------------------------------------------------------------------------
    print("\n[Audit 4/5] Forward Pass & Gradient Optimization Step...")
    dummy_x = tf.random.uniform((4, 224, 224, 3), 0.0, 255.0)
    dummy_y = tf.constant([0, 1, 2, 0], dtype=tf.int32)

    m_potato.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy")
    loss_before = m_potato.evaluate(dummy_x, dummy_y, verbose=0)
    history = m_potato.train_on_batch(dummy_x, dummy_y)
    print(f"   ✅ Potato Doctor Step: Loss before = {loss_before:.4f}, Batch Train Loss = {history[0] if isinstance(history, list) else history:.4f}")

    # --------------------------------------------------------------------------
    # 5. Serialization and Deserialization Verification
    # --------------------------------------------------------------------------
    print("\n[Audit 5/5] Model Serialization & Reload Verification...")
    Path("models/v2").mkdir(parents=True, exist_ok=True)
    test_save_path = "models/v2/smoke_test_model.keras"

    m_potato.save(test_save_path)
    print(f"   Saved test checkpoint to: {test_save_path}")

    # Reload in-memory
    m_reloaded = tf.keras.models.load_model(
        test_save_path,
        custom_objects={
            'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input,
            'CBAM': CBAM,
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention
        },
        compile=False
    )
    assert m_reloaded.count_params() == m_potato.count_params()
    # Clean up test artifact
    if Path(test_save_path).exists():
        Path(test_save_path).unlink()

    print(f"   ✅ Successfully reloaded model! Total params verified: {m_reloaded.count_params():,}")

    print("\n" + "=" * 80)
    print("🎉 ALL PREFLIGHT AUDITS AND SMOKE TESTS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    run_preflight_and_smoke_tests()
