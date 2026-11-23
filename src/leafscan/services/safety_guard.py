"""
src/leafscan/services/safety_guard.py
Deterministic Safety Guard & Agricultural Advisory Policy Engine.
Enforces non-negotiable multi-tier confidence boundaries, OOD rejection,
conformal prediction uncertainty rules, and agricultural advisory bindings.
"""

from dataclasses import dataclass

from leafscan.bundle.manifest import BundleManifest, SafetyGates
from leafscan.services.conformal import ConformalPredictor
from leafscan.services.ood_detector import OODDetector


@dataclass(frozen=True)
class SafetyDecision:
    safety_status: str
    safety_warnings: list[str]
    advisory: str
    disease_abstained: bool
    prediction_set: list[str] | None = None


class SafetyGuard:
    """
    Deterministic Safety Decision Engine.
    No machine learning model or external service may override these rules.
    """

    def __init__(self, manifest: BundleManifest, enable_conformal: bool = True, enable_ood: bool = True):
        self.manifest = manifest
        self.gates: SafetyGates = manifest.safety_gates
        self.advisories: dict[str, str] = manifest.advisories
        self.conformal_predictor = ConformalPredictor() if enable_conformal else None
        self.ood_detector = OODDetector() if enable_ood else None

    def evaluate_species_gate(
        self, species_name: str, species_conf: float, species_probs: dict[str, float] | None = None
    ) -> tuple[bool, SafetyDecision | None]:
        """
        Safety Gate 1: Evaluates OOD indicators and species classifier confidence (0.90 threshold).
        """
        # 1. OOD Screening
        if self.ood_detector and species_probs:
            is_ood, reason = self.ood_detector.evaluate_species_ood(species_probs)
            if is_ood and reason:
                return True, self.create_out_of_scope_decision(reason)

        # 2. Safety Gate 1 Threshold (0.90)
        threshold = self.gates.species_confidence_threshold
        if round(species_conf, 4) < round(threshold, 4):
            warnings = [
                f"Species confidence ({species_conf * 100:.2f}%) is below the safety threshold ({threshold * 100:.0f}%). "
                f"Disease classification abstained to prevent crop misdiagnosis."
            ]
            advisory = self.advisories.get("species_uncertain", "")
            decision = SafetyDecision(
                safety_status="species_uncertain", safety_warnings=warnings, advisory=advisory, disease_abstained=True
            )
            return True, decision

        return False, None

    def evaluate_disease_gate(
        self,
        disease_name: str,
        disease_conf: float,
        crop: str | None = None,
        disease_probs: dict[str, float] | None = None,
    ) -> SafetyDecision:
        """
        Safety Gate 2: Evaluates disease classification confidence & conformal prediction sets.
        High-consequence Late Blight has strict 0.90 threshold; general diseases require 0.70.
        """
        late_blight_thresh = self.gates.late_blight_confidence_threshold
        general_thresh = self.gates.general_disease_confidence_threshold

        warnings = []
        advisory_key = ""

        # Generate conformal prediction set if available
        prediction_set = None
        conformal_status = None
        if self.conformal_predictor and crop and disease_probs:
            prediction_set = self.conformal_predictor.predict_set(crop, disease_probs)
            conformal_status = self.conformal_predictor.evaluate_conformal_safety(prediction_set)

        # Base deterministic gate evaluation
        if disease_name == "Late Blight":
            if round(disease_conf, 4) < round(late_blight_thresh, 4) or conformal_status == "late_blight_uncertain":
                safety_status = "late_blight_uncertain"
                warnings.append(
                    f"Potential early lesions detected, but Late Blight confidence ({disease_conf * 100:.2f}%) is below the safety threshold ({late_blight_thresh * 100:.0f}%). "
                    f"Possible mechanical abrasion, environmental damage, or early Alternaria lesion."
                )
                advisory_key = "late_blight_uncertain"
            else:
                safety_status = "confident"
                advisory_key = "late_blight_confirmed"

        elif round(disease_conf, 4) < round(general_thresh, 4) or conformal_status == "disease_uncertain":
            safety_status = "disease_uncertain"
            warnings.append(
                f"Diagnostic confidence ({disease_conf * 100:.2f}%) is below the general safety threshold ({general_thresh * 100:.0f}%)."
            )
            advisory_key = "general_disease_uncertain"

        else:
            safety_status = "confident"
            if disease_name == "Early Blight":
                advisory_key = "early_blight_confirmed"
            elif disease_name == "Healthy":
                advisory_key = "healthy_confirmed"

        advisory_text = self.advisories.get(advisory_key, "")
        return SafetyDecision(
            safety_status=safety_status,
            safety_warnings=warnings,
            advisory=advisory_text,
            disease_abstained=False,
            prediction_set=prediction_set,
        )

    def create_out_of_scope_decision(self, reason: str) -> SafetyDecision:
        """Creates an out_of_scope decision for unsupported plant species or non-leaf inputs."""
        advisory = self.advisories.get(
            "out_of_scope",
            "OUT OF SCOPE SPECIMEN: The input leaf could not be reliably verified as Potato (Solanum tuberosum) or Tomato (Solanum lycopersicum), or the image is out-of-distribution. Consult a local agricultural extension officer or certified agronomist for non-Solanaceous pathology.",
        )
        return SafetyDecision(
            safety_status="out_of_scope",
            safety_warnings=[f"Out-of-scope input detected: {reason}"],
            advisory=advisory,
            disease_abstained=True,
        )
