# inference/predictor.py
"""
Hierarchical Inference Predictor for LeafScan (V2/V3 Compatibility Adapter).
Provides backward-compatible interface backed by the modernized V3 ONNX engine,
preserving full legacy parameter contracts, threshold overrides, and response schemas.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image

# Import modernized V3 core
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from leafscan.bundle.loader import load_model_bundle, ModelBundle
from leafscan.engines.onnx_engine import OnnxEngine
from leafscan.services.image_intake import ImageIntake
from leafscan.services.safety_guard import SafetyGuard, SafetyGates
from leafscan.services.diagnosis_service import DiagnosisService
from leafscan.core.logging import logger

CONFIG_FILE = Path("inference/inference_config.json")


def load_inference_config():
    """Loads runtime inference and safety-gate configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "safety_gates": {
            "species_confidence_threshold": 0.90,
            "late_blight_confidence_threshold": 0.90,
            "general_disease_confidence_threshold": 0.70
        }
    }


class HierarchicalPredictorV2:
    """
    Hierarchical Predictor with dual safety gating.
    Uses the production ONNX Runtime V3 engine under the hood for low latency and zero TF serving overhead,
    while retaining complete backward compatibility with all V2 tests, fixtures, and configs.
    """

    def __init__(self, config=None, use_keras_fallback=False):
        self.config = config or load_inference_config()
        self.use_keras_fallback = use_keras_fallback

        self.bundle: Optional[ModelBundle] = None
        self.engine = None
        self.diagnosis_service: Optional[DiagnosisService] = None

        self.species_model = None
        self.potato_model = None
        self.tomato_model = None
        self.class_mappings = None

        self._ensure_models_loaded()

    def _ensure_models_loaded(self):
        if self.use_keras_fallback:
            from inference.model_loader import load_models
            self.species_model, self.potato_model, self.tomato_model, self.class_mappings = load_models(self.config)
            return

        # Modern ONNX path
        bundle_path = PROJECT_ROOT / "models/bundles/v3"
        if not bundle_path.exists():
            bundle_path = Path("models/bundles/v3")

        self.bundle = load_model_bundle(bundle_path, verify_checksums=False)
        self.engine = OnnxEngine(self.bundle)

        # Expose non-None references for legacy health check (is_loaded checks)
        self.species_model = self.engine.sessions.get("species_expert")
        self.potato_model = self.engine.sessions.get("potato_expert")
        self.tomato_model = self.engine.sessions.get("tomato_expert")

        # Apply any custom config overrides (e.g. from test boundary fixtures)
        manifest = self.bundle.manifest
        if self.config and "safety_gates" in self.config:
            custom_gates = self.config["safety_gates"]
            manifest.safety_gates = SafetyGates(
                species_confidence_threshold=float(custom_gates.get("species_confidence_threshold", 0.90)),
                late_blight_confidence_threshold=float(custom_gates.get("late_blight_confidence_threshold", 0.90)),
                general_disease_confidence_threshold=float(custom_gates.get("general_disease_confidence_threshold", 0.70))
            )

        intake = ImageIntake()
        guard = SafetyGuard(manifest)
        self.diagnosis_service = DiagnosisService(
            bundle=self.bundle,
            engine=self.engine,
            image_intake=intake,
            safety_guard=guard
        )

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Executes end-to-end hierarchical diagnostic inference.
        """
        if self.use_keras_fallback:
            # Fall back to original Keras path if explicitly requested
            from inference.preprocessing import preprocess_image
            from inference.response_formatter import format_prediction_response, format_error_response
            start_time = time.perf_counter()
            try:
                input_tensor, _ = preprocess_image(image_input)
            except Exception as e:
                return format_error_response(error_message=str(e), error_type=type(e).__name__, model_version="v2")

            sp_logits = self.species_model.predict(input_tensor, verbose=0)[0]
            sp_map = self.class_mappings["species_expert"]["index_to_label"]
            species_probs = {sp_map[str(i)]: float(sp_logits[i]) for i in range(len(sp_logits))}
            sp_idx = int(np.argmax(sp_logits))
            plant_species = sp_map[str(sp_idx)]
            species_conf = float(sp_logits[sp_idx])

            gates = self.config.get("safety_gates", {})
            species_thresh = float(gates.get("species_confidence_threshold", 0.90))
            late_blight_thresh = float(gates.get("late_blight_confidence_threshold", 0.90))
            general_thresh = float(gates.get("general_disease_confidence_threshold", 0.70))
            advisories = self.config.get("advisories", {})

            if round(species_conf, 4) < round(species_thresh, 4):
                return format_prediction_response(
                    plant_species=plant_species,
                    species_confidence=species_conf,
                    species_probabilities=species_probs,
                    disease=None,
                    disease_confidence=None,
                    disease_probabilities={},
                    safety_status="species_uncertain",
                    safety_warnings=[f"Species confidence ({species_conf*100:.2f}%) is below the safety threshold ({species_thresh*100:.0f}%). Disease classification abstained to prevent crop misdiagnosis."],
                    advisory=advisories.get("species_uncertain", ""),
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                    model_version="v2"
                )

            spec_model = self.potato_model if plant_species == "Potato" else self.tomato_model
            spec_key = "potato_expert" if plant_species == "Potato" else "tomato_expert"
            dis_map = self.class_mappings[spec_key]["index_to_label"]
            dis_logits = spec_model.predict(input_tensor, verbose=0)[0]
            dis_probs = {dis_map[str(i)]: float(dis_logits[i]) for i in range(len(dis_logits))}
            dis_idx = int(np.argmax(dis_logits))
            dis_name = dis_map[str(dis_idx)]
            dis_conf = float(dis_logits[dis_idx])

            if dis_name == "Late Blight":
                if round(dis_conf, 4) < round(late_blight_thresh, 4):
                    status_val = "late_blight_uncertain"
                    adv_key = "late_blight_uncertain"
                    warn = [f"Potential early lesions detected, but Late Blight confidence ({dis_conf*100:.2f}%) is below the safety threshold ({late_blight_thresh*100:.0f}%). Possible mechanical abrasion, environmental damage, or early Alternaria lesion."]
                else:
                    status_val = "confident"
                    adv_key = "late_blight_confirmed"
                    warn = []
            elif round(dis_conf, 4) < round(general_thresh, 4):
                status_val = "disease_uncertain"
                adv_key = "general_disease_uncertain"
                warn = [f"Diagnostic confidence ({dis_conf*100:.2f}%) is below the general safety threshold ({general_thresh*100:.0f}%)."]
            else:
                status_val = "confident"
                warn = []
                adv_key = "early_blight_confirmed" if dis_name == "Early Blight" else "healthy_confirmed"

            return format_prediction_response(
                plant_species=plant_species,
                species_confidence=species_conf,
                species_probabilities=species_probs,
                disease=dis_name,
                disease_confidence=dis_conf,
                disease_probabilities=dis_probs,
                safety_status=status_val,
                safety_warnings=warn,
                advisory=advisories.get(adv_key, ""),
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                model_version="v2"
            )

        # Standard V3 ONNX path
        try:
            return self.diagnosis_service.diagnose(image_input)
        except Exception as e:
            from inference.response_formatter import format_error_response
            return format_error_response(error_message=str(e), error_type=type(e).__name__, model_version="v2")


_GLOBAL_PREDICTOR = None

def predict_v2(image_input, config=None):
    """Module-level inference function utilizing a persistent global HierarchicalPredictorV2."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or config is not None:
        _GLOBAL_PREDICTOR = HierarchicalPredictorV2(config=config)
    return _GLOBAL_PREDICTOR.predict(image_input)
