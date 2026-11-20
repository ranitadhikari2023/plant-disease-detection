# training/data_pipeline.py
"""
Manifest-driven tf.data data loader for hierarchical plant disease classification.
Ensures zero data leakage, explicit class indexing, and optimized prefetching.
Uses standard Python csv module (zero dependency on pandas).
"""

import csv
import json
from pathlib import Path
import tensorflow as tf

CLASS_MAPPINGS_PATH = Path("training/class_mappings.json")

def load_class_mappings():
    with open(CLASS_MAPPINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_manifest_dataset(
    manifest_path,
    model_type="species_expert",
    crop_filter=None,
    batch_size=16,
    is_training=False,
    shuffle_buffer=None,
    seed=42,
    augmentation_model=None
):
    """
    Constructs a high-performance tf.data.Dataset from a CSV manifest.

    Args:
        manifest_path (str or Path): Path to the manifest CSV.
        model_type (str): 'species_expert', 'potato_expert', or 'tomato_expert'.
        crop_filter (str, optional): 'potato', 'tomato', or None.
        batch_size (int): Batch size.
        is_training (bool): If True, applies shuffling and online augmentation.
        shuffle_buffer (int, optional): Buffer size for shuffling.
        seed (int): Random seed for reproducibility.
        augmentation_model (tf.keras.Model, optional): Online augmentation layers.

    Returns:
        tf.data.Dataset: Yields batches of (image_batch, label_batch).
    """
    mappings = load_class_mappings()
    if model_type not in mappings:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected one of: {list(mappings.keys())}")

    mapping_info = mappings[model_type]
    label_to_index = mapping_info["label_to_index"]

    filepaths = []
    labels = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if crop_filter and r["crop"] != crop_filter:
                continue
            
            filepaths.append(r["relative_image_path"])
            if model_type == "species_expert":
                labels.append(label_to_index[r["crop"]])
            else:
                labels.append(label_to_index[r["disease_class"]])

    num_samples = len(filepaths)
    if num_samples == 0:
        raise ValueError(f"No samples matched manifest {manifest_path} with crop_filter={crop_filter}")

    # Base dataset of filepaths and labels
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        buf_size = shuffle_buffer or min(num_samples, 2048)
        ds = ds.shuffle(buffer_size=buf_size, seed=seed, reshuffle_each_iteration=True)

    def _parse_image_and_label(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (224, 224), method="bilinear")
        img = tf.cast(img, tf.float32) # In range [0.0, 255.0]
        return img, label

    ds = ds.map(_parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Batching
    ds = ds.batch(batch_size, drop_remainder=False)

    # Apply online augmentation on batches if training
    if is_training and augmentation_model is not None:
        def _augment_batch(images, batch_labels):
            augmented_images = augmentation_model(images, training=True)
            return augmented_images, batch_labels

        ds = ds.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds, num_samples
