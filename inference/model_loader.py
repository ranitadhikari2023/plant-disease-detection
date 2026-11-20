# inference/model_loader.py
"""
Thread-safe Model Loader and Cache for V2 Hierarchical Deep Learning System.
Loads Botanist Species Expert V2, Potato Doctor CBAM V2, and Tomato Doctor CBAM V2 once,
caching references for high-throughput, low-latency production inference.
"""

import os
import sys
import json
import threading
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from training.models import CBAM, ChannelAttention, SpatialAttention
from training.losses import SparseCategoricalFocalLoss

# Global model cache and thread lock
_MODEL_LOCK = threading.Lock()
_CACHED_MODELS = {
    "species": None,
    "potato": None,
    "tomato": None,
    "mappings": None
}

def get_custom_objects():
    """Returns registration dictionary for Keras 3 deserialization."""
    return {
        'CBAM': CBAM,
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss,
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }

def load_class_mappings(mappings_path):
    """Loads explicit index-to-label class mappings."""
    p = Path(mappings_path)
    if not p.exists():
        raise FileNotFoundError(f"Class mappings file not found: {mappings_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_v2_model_file(model_path, custom_objs):
    """
    Loads a single .keras model artifact with safety validation.
    Enforces safe_mode=True to disallow arbitrary code execution during deserialization.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Required V2 model artifact missing: '{model_path}'. "
            f"Please verify model checkpoints exist in models/v2/."
        )
    if p.stat().st_size == 0:
        raise ValueError(f"Model file '{model_path}' is empty or corrupted (0 bytes).")

    try:
        model = tf.keras.models.load_model(
            str(p),
            custom_objects=custom_objs,
            compile=False,
            safe_mode=True
        )
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to safely deserialize V2 model '{model_path}': {type(e).__name__}: {str(e)}"
        ) from e

def load_models(config=None, force_reload=False):
    """
    Retrieves or initializes the cached singleton V2 deep learning models.

    Args:
        config (dict, optional): Configuration dictionary. Defaults to inference/inference_config.json.
        force_reload (bool): If True, reloads models even if already cached.

    Returns:
        tuple: (species_model, potato_model, tomato_model, class_mappings)
    """
    global _CACHED_MODELS

    with _MODEL_LOCK:
        if not force_reload and all(_CACHED_MODELS.values()):
            return (
                _CACHED_MODELS["species"],
                _CACHED_MODELS["potato"],
                _CACHED_MODELS["tomato"],
                _CACHED_MODELS["mappings"]
            )

        if config is None:
            cfg_path = Path("inference/inference_config.json")
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                config = {
                    "models": {
                        "species_model_path": "models/v2/species_expert_v2.keras",
                        "potato_model_path": "models/v2/potato_expert_v2.keras",
                        "tomato_model_path": "models/v2/tomato_expert_v2.keras",
                        "class_mappings_path": "training/class_mappings.json"
                    }
                }

        model_cfg = config.get("models", {})
        species_path = model_cfg.get("species_model_path", "models/v2/species_expert_v2.keras")
        potato_path = model_cfg.get("potato_model_path", "models/v2/potato_expert_v2.keras")
        tomato_path = model_cfg.get("tomato_model_path", "models/v2/tomato_expert_v2.keras")
        mappings_path = model_cfg.get("class_mappings_path", "training/class_mappings.json")

        custom_objs = get_custom_objects()

        print("📦 [ModelLoader] Initializing V2 Hierarchical Medical Team Models...")
        species_model = load_v2_model_file(species_path, custom_objs)
        potato_model = load_v2_model_file(potato_path, custom_objs)
        tomato_model = load_v2_model_file(tomato_path, custom_objs)
        class_mappings = load_class_mappings(mappings_path)

        _CACHED_MODELS["species"] = species_model
        _CACHED_MODELS["potato"] = potato_model
        _CACHED_MODELS["tomato"] = tomato_model
        _CACHED_MODELS["mappings"] = class_mappings

        print(f"  ✓ Cached Species Expert V2 ({species_model.count_params():,} params)")
        print(f"  ✓ Cached Potato Doctor CBAM V2 ({potato_model.count_params():,} params)")
        print(f"  ✓ Cached Tomato Doctor CBAM V2 ({tomato_model.count_params():,} params)")

        return species_model, potato_model, tomato_model, class_mappings

def unload_models():
    """Clears cached models from memory for clean worker shutdown."""
    global _CACHED_MODELS
    with _MODEL_LOCK:
        _CACHED_MODELS = {
            "species": None,
            "potato": None,
            "tomato": None,
            "mappings": None
        }
        print("🧹 [ModelLoader] V2 Model cache cleared.")
