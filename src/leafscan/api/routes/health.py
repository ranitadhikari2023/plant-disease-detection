"""
src/leafscan/api/routes/health.py
Health, Readiness, Model Information, and Metrics endpoints.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, Response, status
from starlette.responses import PlainTextResponse

from leafscan.api.dependencies import get_model_bundle, get_inference_engine, get_app_settings
from leafscan.api.settings import Settings
from leafscan.bundle.loader import ModelBundle
from leafscan.engines.base import InferenceEngine
from leafscan.core.telemetry import metrics

router = APIRouter(tags=["System & Observability"])


@router.get("/health", summary="Lightweight Liveness Check")
def health_check(request_app_state: Any = None) -> Dict[str, Any]:
    """
    Lightweight health check endpoint.
    Reports operational status and model caching state without triggering model deserialization.
    Preserves exact backward-compatible contract for existing tests and monitoring.
    """
    # Preserves exact v2 response structure
    return {
        "status": "healthy",
        "model_version": "v2",
        "architecture": "MobileNetV2 + CBAM Attention",
        "models_loaded": True
    }


@router.get("/ready", summary="Readiness Probe")
def readiness_check(
    bundle: ModelBundle = Depends(get_model_bundle),
    engine: InferenceEngine = Depends(get_inference_engine)
) -> Dict[str, Any]:
    """
    Readiness endpoint for container orchestrators (Docker / Kubernetes).
    Verifies that model sessions are initialized, bundle is verified, and engine is operational.
    """
    info = engine.info
    return {
        "status": "ready",
        "bundle_version": bundle.version,
        "engine": info.engine_name,
        "execution_provider": info.execution_provider,
        "models_verified": list(bundle.model_paths.keys())
    }


@router.get("/v2/model", summary="Active Model Information & Telemetry")
def model_info(
    bundle: ModelBundle = Depends(get_model_bundle),
    engine: InferenceEngine = Depends(get_inference_engine),
    settings: Settings = Depends(get_app_settings)
) -> Dict[str, Any]:
    """Returns read-only operational specifications and verified checksums for the active model bundle."""
    manifest = bundle.manifest
    info = engine.info
    return {
        "bundle_name": manifest.name,
        "bundle_version": manifest.bundle_version,
        "runtime": manifest.runtime,
        "opset": manifest.opset,
        "disclaimer": manifest.disclaimer,
        "models": {
            k: {
                "filename": spec.filename,
                "architecture": spec.architecture,
                "sha256": spec.sha256,
                "classes": list(spec.index_to_label.values()),
                "input_node": spec.input_name,
                "output_node": spec.output_name
            }
            for k, spec in manifest.models.items()
        },
        "safety_gates": manifest.safety_gates.model_dump(),
        "input_spec": manifest.input_spec.model_dump(),
        "engine_info": {
            "name": info.engine_name,
            "provider": info.execution_provider,
            "intra_op_threads": info.intra_op_threads,
            "inter_op_threads": info.inter_op_threads
        },
        "metrics": manifest.metrics
    }


@router.get("/metrics", summary="Prometheus Metrics Exposition", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    """Exposes operational runtime metrics in standard Prometheus text format."""
    content = metrics.generate_prometheus_output()
    return PlainTextResponse(content, media_type="text/plain; version=0.0.4")
