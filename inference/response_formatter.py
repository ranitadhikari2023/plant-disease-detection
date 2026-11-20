# inference/response_formatter.py
"""
Standardized JSON-serializable Response Formatter for V2 Hierarchical Inference.
Ensures uniform schema across console CLI, REST APIs, and automated test harnesses.
"""

from typing import Dict, Any, List, Optional

def format_prediction_response(
    plant_species: str,
    species_confidence: float,
    species_probabilities: Dict[str, float],
    disease: Optional[str] = None,
    disease_confidence: Optional[float] = None,
    disease_probabilities: Optional[Dict[str, float]] = None,
    safety_status: str = "confident",
    safety_warnings: Optional[List[str]] = None,
    advisory: str = "",
    execution_time_ms: Optional[float] = None,
    model_version: str = "v2"
) -> Dict[str, Any]:
    """
    Constructs a structured successful prediction dictionary.
    """
    if safety_warnings is None:
        safety_warnings = []

    if disease:
        diagnosis_str = f"{plant_species} - {disease}"
    elif safety_status == "species_uncertain":
        diagnosis_str = f"{plant_species} (Abstained - Species Uncertain)"
    else:
        diagnosis_str = f"{plant_species} (Abstained)"

    response = {
        "success": True,
        "model_version": model_version,
        "plant_species": plant_species,
        "species_confidence": round(float(species_confidence), 4),
        "disease": disease,
        "disease_confidence": round(float(disease_confidence), 4) if disease_confidence is not None else None,
        "diagnosis": diagnosis_str,
        "safety_status": safety_status,
        "safety_warnings": safety_warnings,
        "advisory": advisory,
        "species_probabilities": {k: round(float(v), 4) for k, v in species_probabilities.items()},
        "probabilities": {k: round(float(v), 4) for k, v in disease_probabilities.items()} if disease_probabilities else {},
        "metadata": {
            "execution_time_ms": round(float(execution_time_ms), 2) if execution_time_ms is not None else None,
            "architecture": "MobileNetV2 + CBAM Attention"
        }
    }
    return response

def format_error_response(
    error_message: str,
    error_type: str = "InferenceError",
    model_version: str = "v2"
) -> Dict[str, Any]:
    """
    Constructs a structured error response dictionary.
    """
    return {
        "success": False,
        "model_version": model_version,
        "plant_species": None,
        "species_confidence": 0.0,
        "disease": None,
        "disease_confidence": 0.0,
        "diagnosis": "Error",
        "safety_status": "error",
        "safety_warnings": [f"{error_type}: {error_message}"],
        "advisory": "Inference could not be completed. Please check image validity and file path.",
        "species_probabilities": {},
        "probabilities": {},
        "error": {
            "type": error_type,
            "message": error_message
        }
    }
