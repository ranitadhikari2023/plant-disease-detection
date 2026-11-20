# training/augmentation.py
"""
Agronomic Online Data Augmentation for Plant Leaf Disease Classification.
Applies biologically realistic transformations strictly during training.
"""

import tensorflow as tf
from tensorflow.keras import layers

def get_training_augmentation(image_size=(224, 224), seed=42):
    """
    Constructs a biologically safe online data augmentation pipeline.
    
    Transformations:
      - Random horizontal and vertical flips: Simulates detached leaf presentation.
      - Rotation (+/- 14.4 deg): Simulates natural leaf angle under camera lens.
      - Zoom (+/- 10%): Simulates distance variation without losing leaf margins.
      - Brightness (+/- 15%): Simulates field illumination variances (sun vs cloud).
      - Contrast (+/- 15%): Simulates camera exposure variations.
      
    Excluded (Harmful to Pathology):
      - Extreme shear (distorts concentric lesion rings).
      - Color channel inversion / hue scrambling (destroys chlorosis / necrosis cues).
      - Coarse cutout (masks small fungal lesions).
    """
    augmentation = tf.keras.Sequential([
        layers.Input(shape=(image_size[0], image_size[1], 3)),
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomFlip("vertical", seed=seed + 1),
        layers.RandomRotation(factor=0.04, seed=seed + 2, fill_mode="reflect"),
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=seed + 3, fill_mode="reflect"),
        layers.RandomBrightness(factor=0.15, value_range=(0.0, 255.0), seed=seed + 4),
        layers.RandomContrast(factor=0.15, seed=seed + 5),
    ], name="agronomic_online_augmentation")
    
    return augmentation
