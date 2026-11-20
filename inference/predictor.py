# inference/predictor.py
"""
V2 Hierarchical Inference Engine.
Orchestrates two-tier diagnosis:
  Tier 1: Botanist Species Expert V2 (Potato vs Tomato)
  Safety Gate 1: Species Confidence Evaluation (Threshold: 90%)
  Tier 2: Specialist Crop Doctor with CBAM Attention (Early Blight vs Late Blight vs Healthy)
  Safety Gate 2: High-Consequence Late Blight Confidence Evaluation (Threshold: 90%)
"""

import time
import json
from pathlib import Path
import numpy as np

from inference.model_loader import load_models
from inference.preprocessing import preprocess_image
from inference.response_formatter import format_prediction_response, format_error_response

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
    Production Hierarchical Predictor utilizing V2 CNN architectures with CBAM attention.
    Maintains cached model singletons across multiple inference calls.
    """

    def __init__(self, config=None):
        self.config = config or load_inference_config()
        self.species_model = None
        self.potato_model = None
        self.tomato_model = None
        self.class_mappings = None

        # Load models eagerly upon instantiation
        self._ensure_models_loaded()

    def _ensure_models_loaded(self):
        if self.species_model is None:
            self.species_model, self.potato_model, self.tomato_model, self.class_mappings = load_models(self.config)

    def predict(self, image_input):
        """
        Executes end-to-end hierarchical diagnostic inference.

        Args:
            image_input (str, Path, PIL.Image.Image, np.ndarray): Raw input image.

        Returns:
            dict: Structured JSON-serializable diagnostic dictionary.
        """
        start_time = time.perf_counter()

        # 1. Image Preprocessing & Validation
        try:
            input_tensor, _ = preprocess_image(image_input)
        except Exception as e:
            return format_error_response(
                error_message=str(e),
                error_type=type(e).__name__,
                model_version="v2"
            )

        self._ensure_models_loaded()

        # 2. Extract Thresholds & Advisories
        gates = self.config.get("safety_gates", {})
        species_thresh = float(gates.get("species_confidence_threshold", 0.90))
        late_blight_thresh = float(gates.get("late_blight_confidence_threshold", 0.90))
        general_thresh = float(gates.get("general_disease_confidence_threshold", 0.70))

        advisories = self.config.get("advisories", {})

        # 3. Tier 1: Botanist Species Classification
        species_logits = self.species_model.predict(input_tensor, verbose=0)[0]
        sp_map = self.class_mappings["species_expert"]["index_to_label"]
        species_probs = {sp_map[str(i)]: float(species_logits[i]) for i in range(len(species_logits))}

        species_idx = int(np.argmax(species_logits))
        plant_species = sp_map[str(species_idx)]
        species_conf = float(species_logits[species_idx])

        # 4. Safety Gate 1: Species Confidence Evaluation (Inclusive threshold policy: >= thresh passes)
        species_is_uncertain = (round(species_conf, 4) < round(species_thresh, 4))
        safety_warnings = []

        if species_is_uncertain:
            # Safe behavior: Abstain from definitive crop-specific disease classification
            safety_status = "species_uncertain"
            safety_warnings.append(
                f"Species confidence ({species_conf * 100:.2f}%) is below the safety threshold ({species_thresh * 100:.0f}%). "
                f"Disease classification abstained to prevent crop misdiagnosis."
            )
            advisory_text = advisories.get("species_uncertain", "")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return format_prediction_response(
                plant_species=plant_species,
                species_confidence=species_conf,
                species_probabilities=species_probs,
                disease=None,
                disease_confidence=None,
                disease_probabilities={},
                safety_status=safety_status,
                safety_warnings=safety_warnings,
                advisory=advisory_text,
                execution_time_ms=elapsed_ms,
                model_version="v2"
            )

        # 5. Tier 2: Specialist Routing & Disease Classification (Only executed when Species Confident)
        specialist_model = self.potato_model if plant_species == "Potato" else self.tomato_model
        specialist_cfg_key = "potato_expert" if plant_species == "Potato" else "tomato_expert"
        dis_map = self.class_mappings[specialist_cfg_key]["index_to_label"]

        disease_logits = specialist_model.predict(input_tensor, verbose=0)[0]
        disease_probs = {dis_map[str(i)]: float(disease_logits[i]) for i in range(len(disease_logits))}

        disease_idx = int(np.argmax(disease_logits))
        disease_name = dis_map[str(disease_idx)]
        disease_conf = float(disease_logits[disease_idx])

        # 6. Safety Gate 2: Disease Diagnostic Confidence Evaluation
        safety_status = "confident"
        advisory_key = None

        if disease_name == "Late Blight":
            if round(disease_conf, 4) < round(late_blight_thresh, 4):
                safety_status = "late_blight_uncertain"
                safety_warnings.append(
                    f"Potential early lesions detected, but Late Blight confidence ({disease_conf * 100:.2f}%) is below the safety threshold ({late_blight_thresh * 100:.0f}%). "
                    f"Possible mechanical abrasion, environmental damage, or early Alternaria lesion."
                )
                advisory_key = "late_blight_uncertain"
            else:
                safety_status = "confident"
                advisory_key = "late_blight_confirmed"

        elif round(disease_conf, 4) < round(general_thresh, 4):
            safety_status = "disease_uncertain"
            safety_warnings.append(
                f"Diagnostic confidence ({disease_conf * 100:.2f}%) is below the general safety threshold ({general_thresh * 100:.0f}%)."
            )
            advisory_key = "general_disease_uncertain"

        else:
            safety_status = "confident"
            if disease_name == "Early Blight":
                advisory_key = "early_blight_confirmed"
            elif disease_name == "Healthy":
                advisory_key = "healthy_confirmed"

        advisory_text = advisories.get(advisory_key, "")
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return format_prediction_response(
            plant_species=plant_species,
            species_confidence=species_conf,
            species_probabilities=species_probs,
            disease=disease_name,
            disease_confidence=disease_conf,
            disease_probabilities=disease_probs,
            safety_status=safety_status,
            safety_warnings=safety_warnings,
            advisory=advisory_text,
            execution_time_ms=elapsed_ms,
            model_version="v2"
        )

# Module-level convenience function
_GLOBAL_PREDICTOR = None

def predict_v2(image_input, config=None):
    """
    Module-level inference function utilizing a persistent global HierarchicalPredictorV2.
    """
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = HierarchicalPredictorV2(config=config)
    return _GLOBAL_PREDICTOR.predict(image_input)
