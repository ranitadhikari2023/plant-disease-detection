"""
src/leafscan/engines/base.py
Runtime-independent inference interface and abstractions.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EngineInfo:
    engine_name: str
    runtime_version: str
    execution_provider: str
    intra_op_threads: int
    inter_op_threads: int
    models_loaded: dict[str, Any]


@dataclass(frozen=True)
class ModelPrediction:
    probabilities: np.ndarray
    predicted_index: int
    confidence: float


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol defining the runtime-independent inference contract."""

    def predict(self, model_key: str, batch: np.ndarray) -> np.ndarray:
        """
        Executes forward inference on a model given an input batch tensor.

        Args:
            model_key: Identifier of model ('species_expert', 'potato_expert', 'tomato_expert')
            batch: np.ndarray of shape (N, 224, 224, 3) and dtype float32

        Returns:
            np.ndarray of shape (N, num_classes) containing normalized probabilities.
        """
        ...

    def warmup(self) -> None:
        """Executes dummy inference through all sessions to prime JIT and runtime caches."""
        ...

    @property
    def info(self) -> EngineInfo:
        """Returns runtime and session metadata."""
        ...
