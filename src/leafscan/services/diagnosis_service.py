"""
src/leafscan/services/diagnosis_service.py
High-level diagnosis orchestration service.
Coordinates Image Intake, Inference Engine, Safety Guard, and canonical JSON response formatting.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from leafscan.bundle.loader import ModelBundle
from leafscan.core.telemetry import metrics
from leafscan.engines.base import InferenceEngine
from leafscan.services.image_intake import ImageIntake
from leafscan.services.safety_guard import SafetyDecision, SafetyGuard


class DiagnosisService:
    """Orchestrates end-to-end foliar diagnosis."""

    def __init__(
        self,
        bundle: ModelBundle,
        engine: InferenceEngine,
        image_intake: ImageIntake | None = None,
        safety_guard: SafetyGuard | None = None,
    ):
        self.bundle = bundle
        self.engine = engine
        self.image_intake = image_intake or ImageIntake()
        self.safety_guard = safety_guard or SafetyGuard(bundle.manifest)
        self.manifest = bundle.manifest

    def diagnose(
        self, image_source: bytes | str | Path | Image.Image | np.ndarray, request_id: str | None = None
    ) -> dict[str, Any]:
        """
        Executes end-to-end two-tier foliar diagnosis.

        1. Ingests, sanitizes, and preprocesses leaf photograph.
        2. Tier 1: Botanist Species Expert (Potato vs Tomato).
        3. Safety Gate 1: 90% species confidence check (abstains if uncertain).
        4. Tier 2: CBAM Crop Specialist (Early Blight, Late Blight, Healthy).
        5. Safety Gate 2: 90% Late Blight escalation & 70% general disease threshold.
        6. Constructs canonical JSON response preserving complete contract.
        """
        t0 = time.perf_counter()

        # Step 1: Image intake & preprocessing
        tensor, _ = self.image_intake.process(image_source)

        # Step 2: Tier 1 - Species Expert
        sp_spec = self.manifest.models["species_expert"]
        species_logits = self.engine.predict("species_expert", tensor)[0]
        species_probs = {sp_spec.index_to_label[str(i)]: float(species_logits[i]) for i in range(len(species_logits))}

        species_idx = int(np.argmax(species_logits))
        plant_species = sp_spec.index_to_label[str(species_idx)]
        species_conf = float(species_logits[species_idx])

        # Step 3: Safety Gate 1 Evaluation
        is_uncertain, gate1_decision = self.safety_guard.evaluate_species_gate(
            plant_species, species_conf, species_probs=species_probs
        )

        if is_uncertain and gate1_decision:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics.record_latency(elapsed_ms)
            metrics.record_safety_status(gate1_decision.safety_status)

            return self._build_response(
                plant_species=plant_species,
                species_confidence=species_conf,
                species_probabilities=species_probs,
                disease=None,
                disease_confidence=None,
                disease_probabilities={},
                safety_decision=gate1_decision,
                elapsed_ms=elapsed_ms,
                request_id=request_id,
            )

        # Step 4: Tier 2 - Crop Specialist
        specialist_key = "potato_expert" if plant_species == "Potato" else "tomato_expert"
        dis_spec = self.manifest.models[specialist_key]

        disease_logits = self.engine.predict(specialist_key, tensor)[0]
        disease_probs = {dis_spec.index_to_label[str(i)]: float(disease_logits[i]) for i in range(len(disease_logits))}

        disease_idx = int(np.argmax(disease_logits))
        disease_name = dis_spec.index_to_label[str(disease_idx)]
        disease_conf = float(disease_logits[disease_idx])

        # Step 5: Safety Gate 2 Evaluation
        gate2_decision = self.safety_guard.evaluate_disease_gate(
            disease_name, disease_conf, crop=plant_species, disease_probs=disease_probs
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics.record_latency(elapsed_ms)
        metrics.record_safety_status(gate2_decision.safety_status)

        return self._build_response(
            plant_species=plant_species,
            species_confidence=species_conf,
            species_probabilities=species_probs,
            disease=disease_name,
            disease_confidence=disease_conf,
            disease_probabilities=disease_probs,
            safety_decision=gate2_decision,
            elapsed_ms=elapsed_ms,
            request_id=request_id,
        )

    def _build_response(
        self,
        plant_species: str,
        species_confidence: float,
        species_probabilities: dict[str, float],
        disease: str | None,
        disease_confidence: float | None,
        disease_probabilities: dict[str, float],
        safety_decision: SafetyDecision,
        elapsed_ms: float,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Builds exact backward-compatible JSON schema."""
        if disease:
            diagnosis_str = f"{plant_species} - {disease}"
        elif safety_decision.safety_status == "species_uncertain":
            diagnosis_str = f"{plant_species} (Abstained - Species Uncertain)"
        else:
            diagnosis_str = f"{plant_species} (Abstained)"

        meta: dict[str, Any] = {
            "execution_time_ms": round(float(elapsed_ms), 2),
            "architecture": "MobileNetV2 + CBAM Attention",
            "model_bundle": self.manifest.bundle_version,
        }
        if safety_decision.prediction_set is not None:
            meta["prediction_set"] = safety_decision.prediction_set
        if request_id:
            meta["request_id"] = request_id

        return {
            "success": True,
            "model_version": "v2",  # preserved for contract compatibility
            "plant_species": plant_species,
            "species_confidence": round(float(species_confidence), 4),
            "disease": disease,
            "disease_confidence": round(float(disease_confidence), 4) if disease_confidence is not None else None,
            "diagnosis": diagnosis_str,
            "safety_status": safety_decision.safety_status,
            "safety_warnings": safety_decision.safety_warnings,
            "advisory": safety_decision.advisory,
            "species_probabilities": {k: round(float(v), 4) for k, v in species_probabilities.items()},
            "probabilities": {k: round(float(v), 4) for k, v in disease_probabilities.items()}
            if disease_probabilities
            else {},
            "metadata": meta,
        }
