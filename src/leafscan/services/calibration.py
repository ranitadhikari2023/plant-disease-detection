"""
src/leafscan/services/calibration.py
Temperature scaling and Expected Calibration Error (ECE) measurement for LeafScan V3.
"""

import numpy as np


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, ground_truth: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across uniform confidence bins.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(predictions[in_bin] == ground_truth[in_bin])
            conf = np.mean(confidences[in_bin])
            ece += np.abs(acc - conf) * prop

    return float(ece)


class TemperatureScaler:
    """Post-hoc probability calibration via temperature scaling."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.01, float(temperature))

    def scale_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies temperature scaling to softmax probabilities.
        Converts to pseudo-logits: z = log(p + eps), scales by 1/T, re-applies softmax.
        """
        if abs(self.temperature - 1.0) < 1e-4:
            return probs

        eps = 1e-12
        pseudo_logits = np.log(np.clip(probs, eps, 1.0)) / self.temperature
        # Subtract max for numerical stability
        exp_logits = np.exp(pseudo_logits - np.max(pseudo_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
