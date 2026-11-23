"""
src/leafscan/engines/onnx_engine.py
Production ONNX Runtime Inference Engine.
Zero TensorFlow dependency. High throughput, low-latency CPU serving.
"""

import numpy as np
import onnxruntime as ort

from leafscan.bundle.loader import ModelBundle
from leafscan.core.exceptions import InferenceEngineError
from leafscan.core.logging import logger
from leafscan.engines.base import EngineInfo


class OnnxEngine:
    """Production ONNX Runtime execution engine."""

    def __init__(
        self,
        bundle: ModelBundle,
        execution_provider: str = "CPUExecutionProvider",
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
    ):
        self.bundle = bundle
        self.execution_provider = execution_provider
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads

        self.sessions: dict[str, ort.InferenceSession] = {}
        self.input_names: dict[str, str] = {}
        self.output_names: dict[str, str] = {}

        self._initialize_sessions()

    def _initialize_sessions(self):
        """Initializes and caches ONNX inference sessions for each model in bundle."""
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if self.intra_op_threads > 0:
            so.intra_op_num_threads = self.intra_op_threads
        if self.inter_op_threads > 0:
            so.inter_op_num_threads = self.inter_op_threads

        # Optimize thread pool configuration for CPU serving
        providers = [self.execution_provider]
        if self.execution_provider not in ort.get_available_providers():
            logger.warning(f"Provider {self.execution_provider} not available. Falling back to CPUExecutionProvider.")
            providers = ["CPUExecutionProvider"]

        for key, _spec in self.bundle.manifest.models.items():
            model_path = self.bundle.model_paths[key]
            logger.info(f"Initializing ONNX session for {key} from {model_path.name}")
            try:
                session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
                self.sessions[key] = session
                self.input_names[key] = session.get_inputs()[0].name
                self.output_names[key] = session.get_outputs()[0].name
            except Exception as e:
                raise InferenceEngineError(
                    f"Failed to initialize ONNX session for '{key}': {e}",
                    details={"model_key": key, "path": str(model_path)},
                ) from e

    def predict(self, model_key: str, batch: np.ndarray) -> np.ndarray:
        """
        Executes forward inference on a model given an input batch tensor.

        Args:
            model_key: Identifier of model ('species_expert', 'potato_expert', 'tomato_expert')
            batch: np.ndarray of shape (N, 224, 224, 3) and dtype float32

        Returns:
            np.ndarray of shape (N, num_classes) with normalized probabilities.
        """
        if model_key not in self.sessions:
            raise InferenceEngineError(f"Model '{model_key}' not loaded in engine.")

        if not isinstance(batch, np.ndarray):
            batch = np.asarray(batch, dtype=np.float32)

        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)

        if batch.ndim != 4:
            raise InferenceEngineError(f"Expected 4D tensor (N, H, W, C), got {batch.ndim}D tensor.")

        session = self.sessions[model_key]
        in_name = self.input_names[model_key]
        out_name = self.output_names[model_key]

        try:
            ort_outputs = session.run([out_name], {in_name: batch})
            probs = ort_outputs[0]
            if not np.all(np.isfinite(probs)):
                raise InferenceEngineError("Model produced NaN or Infinite probabilities.")
            return probs
        except Exception as e:
            if isinstance(e, InferenceEngineError):
                raise
            raise InferenceEngineError(f"ONNX inference failure on '{model_key}': {e}") from e

    def warmup(self) -> None:
        """Executes a single dummy forward pass through all sessions to warm up JIT and thread pools."""
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        for key in self.sessions:
            self.predict(key, dummy)
        logger.info("ONNX Engine warmed up successfully.")

    @property
    def info(self) -> EngineInfo:
        return EngineInfo(
            engine_name="OnnxEngine",
            runtime_version=ort.__version__,
            execution_provider=self.execution_provider,
            intra_op_threads=self.intra_op_threads,
            inter_op_threads=self.inter_op_threads,
            models_loaded={k: str(p.name) for k, p in self.bundle.model_paths.items()},
        )
