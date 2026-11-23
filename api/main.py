"""
api/main.py
FastAPI Web Integration Service for Plant Disease Detection (LeafScan V3 Production Engine).
Exposes the two-tier hierarchical inference service over HTTP REST endpoints,
serves the dependency-free frontend Single Page Application, and preserves full backward compatibility.
"""

import io
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add repository root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predictor import HierarchicalPredictorV2
from inference.response_formatter import format_error_response
from leafscan.api.app import create_app
from leafscan.api.settings import get_settings

logger = logging.getLogger("LeafScanAPI")

# Maximum permitted payload size: 10 Megabytes
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

# Singleton predictor instance for backward-compatible test injection
_PREDICTOR: Optional[HierarchicalPredictorV2] = None


def get_predictor() -> HierarchicalPredictorV2:
    """
    Returns the cached singleton HierarchicalPredictorV2 instance.
    Uses the production ONNX V3 engine under the hood.
    """
    global _PREDICTOR
    if _PREDICTOR is None:
        logger.info("Initializing cached HierarchicalPredictorV2 singleton...")
        _PREDICTOR = HierarchicalPredictorV2()
    return _PREDICTOR


def set_predictor(predictor: Optional[HierarchicalPredictorV2]):
    """Allows test fixtures to inject a mock predictor for fast deterministic testing."""
    global _PREDICTOR
    _PREDICTOR = predictor


# Instantiate application from LeafScan V3 factory
app = create_app()


def health_check() -> Dict[str, Any]:
    """Lightweight health check endpoint preserving exact V2 contract."""
    predictor = _PREDICTOR
    is_loaded = predictor is not None and getattr(predictor, "species_model", None) is not None
    return {
        "status": "healthy",
        "model_version": "v2",
        "architecture": "MobileNetV2 + CBAM Attention",
        "models_loaded": is_loaded
    }


def diagnose_leaf(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Synchronous entry point preserving exact V2 endpoint interface.
    Delegates to injected mock predictor (if set by test fixtures) or production V3 engine.
    """
    # 1. Validate file presence
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in upload field 'image'."
        )

    # 2. Enforce upload size limits defensively
    try:
        file_bytes = image.file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    except Exception as e:
        logger.error(f"Failed to read uploaded stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content."
        )

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"File exceeds maximum size limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    # 3. Defensive image decoding and format verification via PIL
    try:
        byte_stream = io.BytesIO(file_bytes)
        with Image.open(byte_stream) as raw_img:
            img_format = (raw_img.format or "").upper()
            if img_format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported image format '{img_format}'. Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}."
                )
            pil_image = raw_img.copy()
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Corrupted or invalid image uploaded: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image payload. Unable to decode image raster."
        )

    # 4. Execute Inference
    try:
        predictor = get_predictor()
        result = predictor.predict(pil_image)
        return result
    except Exception as e:
        logger.error(f"Inference pipeline execution error: {type(e).__name__}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(
                error_message="Diagnostic inference encountered an internal error. Please check image framing and retry.",
                error_type="InferenceExecutionError",
                model_version="v2"
            )
        )


def serve_index():
    """Serves index.html for root path."""
    frontend_dir = PROJECT_ROOT / "frontend"
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_file))
