"""
src/leafscan/services/conformal.py
Split Conformal Prediction for Specialist Crop Pathology Diagnoses.
Guarantees finite-sample coverage at 1 - alpha significance level.
"""


class ConformalPredictor:
    """
    Implements split conformal prediction set generation and V3 uncertainty gating.
    Calibrated against held-out specimen-grouped validation split.
    """

    def __init__(self, quantiles: dict[str, float] | None = None):
        # 90% finite-sample coverage thresholds computed from 592 held-out validation specimens
        self.quantiles = quantiles or {
            "potato_expert": 0.1969,  # q_hat at alpha=0.10
            "tomato_expert": 0.4335,  # q_hat at alpha=0.10
        }

    def predict_set(self, crop: str, disease_probs: dict[str, float]) -> list[str]:
        """
        Generates conformal prediction set C(x) = { y : P(y) >= 1 - q_hat }.
        """
        model_key = "potato_expert" if crop.lower() == "potato" else "tomato_expert"
        q_hat = self.quantiles.get(model_key, 0.40)
        threshold = 1.0 - q_hat

        included = [disease_name for disease_name, prob in disease_probs.items() if prob >= threshold]

        # In case none exceeded threshold, include argmax
        if not included and disease_probs:
            argmax_disease = max(disease_probs.items(), key=lambda x: x[1])[0]
            included = [argmax_disease]

        return sorted(included)

    def evaluate_conformal_safety(self, prediction_set: list[str]) -> str | None:
        """
        Evaluates LeafScan V3 Conformal Decision Rules:
          - If prediction set contains Late Blight AND another class -> 'late_blight_uncertain'
          - If prediction set is empty or contains every class (all 3) -> 'disease_uncertain'
          - Otherwise -> None (defers to standard safety thresholds)
        """
        if not prediction_set or len(prediction_set) >= 3:
            return "disease_uncertain"

        if "Late Blight" in prediction_set and len(prediction_set) > 1:
            return "late_blight_uncertain"

        return None
