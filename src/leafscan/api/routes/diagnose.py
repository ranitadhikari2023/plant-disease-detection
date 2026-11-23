"""
src/leafscan/api/routes/diagnose.py
Diagnostic inference route for LeafScan V3.
Preserves complete backward compatibility with existing web frontend and API consumers.
"""

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from leafscan.api.dependencies import get_diagnosis_service, verify_api_key
from leafscan.core.exceptions import ImageIntakeError, UnsupportedFormatError
from leafscan.core.logging import logger
from leafscan.services.diagnosis_service import DiagnosisService

router = APIRouter(tags=["Diagnosis"])


@router.post("/v2/diagnose", summary="Execute Hierarchical Foliar Diagnosis")
def diagnose_leaf(
    request: Request,
    image: UploadFile = File(...),
    service: DiagnosisService = Depends(get_diagnosis_service),
    api_key: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """
    Receives an uploaded leaf photograph and executes two-tier hierarchical screening:
      1. Defensive payload size and image format verification.
      2. Tier 1: Botanist Species Classification (Potato vs Tomato).
      3. Safety Gate 1: 90% species confidence evaluation (abstains on uncertainty).
      4. Tier 2: CBAM Crop Specialist (Early Blight, Late Blight, Healthy).
      5. Safety Gate 2: 90% Late Blight escalation & 70% general disease threshold.
    """
    # 1. Validate file presence
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No image file provided in upload field 'image'."
        )

    # 2. Defensive read up to max_upload_size_bytes + 1
    max_bytes = service.image_intake.max_upload_bytes
    try:
        file_bytes = image.file.read(max_bytes + 1)
    except Exception as e:
        logger.error(f"Failed to read uploaded stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Could not read uploaded file content."
        ) from e

    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"File exceeds maximum size limit of {max_bytes // (1024 * 1024)} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    req_id = getattr(request.state, "request_id", None)

    # 3. Execute Diagnosis Service
    try:
        result = service.diagnose(file_bytes, request_id=req_id)
        return result
    except UnsupportedFormatError as ufe:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=ufe.message) from ufe
    except ImageIntakeError as iie:
        raise HTTPException(status_code=iie.status_code, detail=iie.message) from iie
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Inference pipeline execution error: {type(exc).__name__}: {exc!s}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "model_version": "v2",
                "plant_species": None,
                "species_confidence": 0.0,
                "disease": None,
                "disease_confidence": 0.0,
                "diagnosis": "Error",
                "safety_status": "error",
                "safety_warnings": [
                    "InferenceExecutionError: Diagnostic inference encountered an internal error. Please check image framing and retry."
                ],
                "advisory": "Diagnostic inference encountered an internal error. Please check image framing and retry.",
                "species_probabilities": {},
                "probabilities": {},
                "error": {
                    "type": "InferenceExecutionError",
                    "message": "Diagnostic inference encountered an internal error. Please check image framing and retry.",
                },
            },
        )
