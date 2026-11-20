# training/train_utils.py
"""
Shared two-phase training orchestrator for hierarchical plant disease models.
Manages learning rate schedules, callbacks, class weighting, and model serialization.
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

from training.data_pipeline import build_manifest_dataset
from training.augmentation import get_training_augmentation
from training.losses import compute_specimen_class_weights, SparseCategoricalFocalLoss

CONFIG_PATH = Path("training/training_config.json")

def load_training_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def set_global_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_two_phase_training(
    model,
    base_model,
    model_type,
    crop_filter=None,
    model_save_path="models/v2/model_v2.keras",
    log_csv_path="training/logs/model_training.csv",
    epochs_phase1=10,
    epochs_phase2=10,
    batch_size_phase1=16,
    batch_size_phase2=8,
    use_focal_loss=False,
    unfreeze_top_n=40,
    smoke_test=False
):
    """
    Executes Phase A (Warmup Head & Attention) and Phase B (Deep Fine-Tuning).
    """
    cfg = load_training_config()
    set_global_seeds(cfg.get("random_seed", 42))

    train_manifest = cfg["manifests"]["train_specimen"]
    val_manifest = cfg["manifests"]["val_specimen"]

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Specimen-derived Class Weights
    class_weights, specimen_counts = compute_specimen_class_weights(
        train_manifest, crop_filter=crop_filter, model_type=model_type
    )
    print(f"\n📊 Specimen Counts in Training Split: {specimen_counts}")
    print(f"⚖️ Computed Specimen Class Weights:   {class_weights}")

    # 2. Augmentation & Data Loaders
    aug_model = get_training_augmentation(seed=cfg.get("random_seed", 42))

    train_ds_p1, n_train = build_manifest_dataset(
        train_manifest, model_type=model_type, crop_filter=crop_filter,
        batch_size=batch_size_phase1, is_training=True, augmentation_model=aug_model
    )
    val_ds_p1, n_val = build_manifest_dataset(
        val_manifest, model_type=model_type, crop_filter=crop_filter,
        batch_size=batch_size_phase1, is_training=False
    )

    print(f"📦 Data Pipeline Ready: {n_train} training specimens, {n_val} validation specimens.")

    # 3. Loss Configuration
    if use_focal_loss:
        # Alpha vector from normalized class weights
        weights_arr = np.array([class_weights[i] for i in range(len(class_weights))], dtype=np.float32)
        alpha = weights_arr / weights_arr.sum()
        loss_fn = SparseCategoricalFocalLoss(gamma=2.0, alpha=alpha)
        train_class_weights = None # Alpha already embedded in focal loss
        print(f"🎯 Loss Function: Alpha-Weighted Sparse Categorical Focal Loss (alpha={alpha.round(3).tolist()})")
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        train_class_weights = class_weights
        print("🎯 Loss Function: Class-Weighted Sparse Categorical Crossentropy")

    # If smoke test, reduce epochs
    if smoke_test:
        epochs_phase1 = 1
        epochs_phase2 = 1
        print("⚡ SMOKE TEST MODE: 1 epoch per phase.")

    # --------------------------------------------------------------------------
    # PHASE A: Warmup Classification Head & CBAM (Base Frozen)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE A: Training Classification Head & CBAM (Base MobileNetV2 Frozen)")
    print("=" * 70)

    base_model.trainable = False
    lr_phase1 = cfg["phase1"]["learning_rate"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase1),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["callbacks"]["early_stopping"]["patience"],
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg["callbacks"]["reduce_lr_on_plateau"]["factor"],
            patience=cfg["callbacks"]["reduce_lr_on_plateau"]["patience"], verbose=1
        ),
        tf.keras.callbacks.CSVLogger(log_csv_path, separator=",", append=False)
    ]

    history_p1 = model.fit(
        train_ds_p1,
        validation_data=val_ds_p1,
        epochs=epochs_phase1,
        class_weight=train_class_weights,
        callbacks=callbacks_phase1
    )

    # --------------------------------------------------------------------------
    # PHASE B: Deep Fine-Tuning (Unfreeze Top N Layers)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PHASE B: Fine-Tuning Top {unfreeze_top_n} Layers of MobileNetV2")
    print("=" * 70)

    # Re-batch to protect VRAM ceiling
    train_ds_p2, _ = build_manifest_dataset(
        train_manifest, model_type=model_type, crop_filter=crop_filter,
        batch_size=batch_size_phase2, is_training=True, augmentation_model=aug_model
    )
    val_ds_p2, _ = build_manifest_dataset(
        val_manifest, model_type=model_type, crop_filter=crop_filter,
        batch_size=batch_size_phase2, is_training=False
    )

    base_model.trainable = True
    # Freeze lower layers, unfreeze top N layers
    for layer in base_model.layers[:-unfreeze_top_n]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_top_n:]:
        # Freeze BatchNormalization to stabilize fine-tuning statistics
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    active_tensors = sum(len(l.trainable_weights) for l in base_model.layers if l.trainable)
    print(f"🔓 Base Layers Active: {active_tensors} trainable weight tensors in MobileNetV2.")

    lr_phase2 = cfg["phase2"]["learning_rate"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase2),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["callbacks"]["early_stopping"]["patience"],
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg["callbacks"]["reduce_lr_on_plateau"]["factor"],
            patience=cfg["callbacks"]["reduce_lr_on_plateau"]["patience"], verbose=1
        ),
        tf.keras.callbacks.CSVLogger(log_csv_path, separator=",", append=True)
    ]

    history_p2 = model.fit(
        train_ds_p2,
        validation_data=val_ds_p2,
        initial_epoch=len(history_p1.history.get("loss", [])),
        epochs=len(history_p1.history.get("loss", [])) + epochs_phase2,
        class_weight=train_class_weights,
        callbacks=callbacks_phase2
    )

    print(f"\n🎉 Training complete! Best V2 model checkpoint saved to: {model_save_path}")
    print(f"📜 Training log saved to: {log_csv_path}")
    return history_p1, history_p2
