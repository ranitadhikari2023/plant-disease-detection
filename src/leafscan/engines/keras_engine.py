"""
src/leafscan/engines/keras_engine.py
Keras 3 / TensorFlow Inference Engine for Training, Parity & Development.
TensorFlow is imported lazily to keep the production serving path 100% decoupled.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

from leafscan.bundle.loader import ModelBundle
from leafscan.engines.base import InferenceEngine, EngineInfo
from leafscan.core.exceptions import InferenceEngineError
from leafscan.core.logging import logger


class KerasEngine:
    """Development and parity testing engine using Keras 3 with lazy TensorFlow import."""

    def __init__(self, bundle: Optional[ModelBundle] = None, keras_models_dir: str = "models/v2"):
        self.bundle = bundle
        self.keras_models_dir = Path(keras_models_dir)
        self.models: Dict[str, Any] = {}
        self._tf = None
        self._initialize_models()

    def _get_tf(self):
        if self._tf is None:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf
            self._tf = tf
        return self._tf

    def _initialize_models(self):
        tf = self._get_tf()
        from training.models import CBAM, ChannelAttention, SpatialAttention
        from training.losses import SparseCategoricalFocalLoss

        custom_objs = {
            'CBAM': CBAM,
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention,
            'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss,
            'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
        }

        mapping = {
            "species_expert": self.keras_models_dir / "species_expert_v2.keras",
            "potato_expert": self.keras_models_dir / "potato_expert_v2.keras",
            "tomato_expert": self.keras_models_dir / "tomato_expert_v2.keras"
        }

        for key, path in mapping.items():
            if not path.exists():
                raise FileNotFoundError(f"Keras model artifact not found: {path}")
            model = tf.keras.models.load_model(str(path), custom_objects=custom_objs, compile=False, safe_mode=True)
            self.models[key] = model

    def predict(self, model_key: str, batch: np.ndarray) -> np.ndarray:
        if model_key not in self.models:
            raise InferenceEngineError(f"Model '{model_key}' not loaded in KerasEngine.")

        model = self.models[model_key]
        try:
            outputs = model(batch, training=False).numpy()
            return outputs
        except Exception as e:
            raise InferenceEngineError(f"Keras inference failure on '{model_key}': {e}") from e

    def warmup(self) -> None:
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        for key in self.models:
            self.predict(key, dummy)
        logger.info("Keras Engine warmed up.")

    @property
    def info(self) -> EngineInfo:
        tf = self._get_tf()
        return EngineInfo(
            engine_name="KerasEngine",
            runtime_version=tf.__version__,
            execution_provider="TensorFlow-CPU",
            intra_op_threads=0,
            inter_op_threads=0,
            models_loaded={k: str(p.name) for k, p in self.models.items()}
        )
