"""
src/leafscan/services/ood_detector.py
Out-of-Distribution (OOD) Rejection Architecture for LeafScan V3.
Detects non-solanaceous crops, non-leaf photographs, and high-entropy anomalies.
"""

import numpy as np


class OODDetector:
    """
    Evaluates prediction entropy and maximum softmax confidence to detect OOD inputs.
    Prevents closed-set classification from hallucinating confident species for unsupported crops.
    """

    def __init__(self, min_species_confidence: float = 0.65, max_normalized_entropy: float = 0.88):
        self.min_species_confidence = min_species_confidence
        self.max_normalized_entropy = max_normalized_entropy

    def evaluate_species_ood(self, species_probs: dict[str, float]) -> tuple[bool, str | None]:
        """
        Evaluates species probabilities for OOD characteristics.

        Returns:
            (is_ood: bool, reason: Optional[str])
        """
        probs = np.array(list(species_probs.values()), dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs)

        # 1. Max probability check
        max_p = float(np.max(probs))
        if max_p < self.min_species_confidence:
            return (
                True,
                f"Maximum species confidence ({max_p * 100:.1f}%) is below OOD rejection threshold ({self.min_species_confidence * 100:.0f}%).",
            )

        # 2. Normalized Shannon Entropy check for C=2 classes (H / log2(2) = H)
        entropy = -float(np.sum(probs * np.log2(probs)))
        if entropy > self.max_normalized_entropy:
            return (
                True,
                f"Species entropy ({entropy:.3f}) indicates severe diagnostic ambiguity (threshold: {self.max_normalized_entropy:.2f}).",
            )

        return False, None
