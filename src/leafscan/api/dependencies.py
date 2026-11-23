"""
src/leafscan/api/dependencies.py
FastAPI Dependency Injection providers for LeafScan V3.
"""

from typing import Optional
from fastapi import Request, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader
import hashlib

from leafscan.api.settings import Settings, get_settings
from leafscan.bundle.loader import ModelBundle
from leafscan.engines.base import InferenceEngine
from leafscan.services.diagnosis_service import DiagnosisService
from leafscan.services.image_intake import ImageIntake

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_app_settings() -> Settings:
    return get_settings()


def get_model_bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded or application is starting up."
        )
    return bundle


def get_inference_engine(request: Request) -> InferenceEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized or application is starting up."
        )
    return engine


def get_diagnosis_service(request: Request) -> DiagnosisService:
    service = getattr(request.app.state, "diagnosis_service", None)
    if service is None:
        # Construct dynamically from state
        bundle = get_model_bundle(request)
        engine = get_inference_engine(request)
        settings = get_app_settings()
        intake = ImageIntake(
            max_upload_bytes=settings.max_upload_size_bytes,
            max_decoded_pixels=settings.max_decoded_pixels,
            enable_quality_checks=settings.enable_quality_checks,
            blur_threshold=settings.blur_threshold
        )
        service = DiagnosisService(bundle=bundle, engine=engine, image_intake=intake)
        request.app.state.diagnosis_service = service
    return service


def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> Optional[str]:
    """Validates hashed API key when integrator authentication is enabled."""
    settings = get_app_settings()
    if not settings.api_key_auth_enabled:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required API key in header 'X-API-Key'."
        )

    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    if key_hash not in settings.authorized_api_key_hashes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized API key."
        )

    return api_key
