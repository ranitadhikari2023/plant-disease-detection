"""
api/main.py
FastAPI Web Integration Service for Plant Disease Detection V2 (LeafScan).
Exposes the hierarchical two-tier inference engine over HTTP REST endpoints
and serves the dependency-free frontend Single Page Application.
"""

import io
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add repository root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.predictor import HierarchicalPredictorV2
from inference.response_formatter import format_error_response

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LeafScanAPI")

# Maximum permitted payload size: 10 Megabytes
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

# Singleton predictor instance
_PREDICTOR: Optional[HierarchicalPredictorV2] = None


def get_predictor() -> HierarchicalPredictorV2:
    """
    Returns the cached singleton HierarchicalPredictorV2 instance.
    Loads models on-demand upon the first diagnosis request to ensure
    lightweight, instantaneous startup for health checks.
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


# FastAPI Application Factory
app = FastAPI(
    title="LeafScan V2 — Plant Disease Detection API",
    description="Hierarchical Two-Tier Foliar Screening with CBAM Attention for Solanaceous Crops",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration (Restricted to same-origin and standard local development ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", summary="Lightweight System Health Check")
def health_check() -> Dict[str, Any]:
    """
    Lightweight health check endpoint.
    Reports operational status and model caching telemetry without triggering
    heavy deep learning model deserialization.
    """
    is_loaded = _PREDICTOR is not None and _PREDICTOR.species_model is not None
    return {
        "status": "healthy",
        "model_version": "v2",
        "architecture": "MobileNetV2 + CBAM Attention",
        "models_loaded": is_loaded
    }


@app.post("/v2/diagnose", summary="Execute Hierarchical Foliar Diagnosis")
def diagnose_leaf(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Receives an uploaded leaf photograph and executes two-tier hierarchical screening:
      1. Defensive payload size and image format verification.
      2. Tier 1: MobileNetV2 Botanist Species Classification (Potato vs Tomato).
      3. Safety Gate 1: 90% species confidence evaluation (abstains on uncertainty).
      4. Tier 2: CBAM Crop Specialist (Early Blight, Late Blight, Healthy).
      5. Safety Gate 2: 90% Late Blight escalation & 70% general disease threshold.
    
    Synchronous definition ensures CPU-intensive neural network inference executes
    safely on FastAPI's worker threadpool without stalling the asynchronous event loop.
    """
    # 1. Validate file presence
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in upload field 'image'."
        )

    # 2. Enforce upload size limits defensively (read up to MAX_UPLOAD_SIZE_BYTES + 1)
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
            status_code=status.HTTP_413_CONTENT_TOO_LARGE if hasattr(status, "HTTP_413_CONTENT_TOO_LARGE") else 413,
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
            # Create in-memory copy detached from file handle
            pil_image = raw_img.copy()
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Corrupted or invalid image uploaded: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image payload. Unable to decode image raster."
        )

    # 4. Execute Hierarchical V2 Inference
    try:
        predictor = get_predictor()
        result = predictor.predict(pil_image)
        return result
    except Exception as e:
        logger.error(f"Inference pipeline execution error: {type(e).__name__}: {str(e)}", exc_info=True)
        # Avoid leaking internal filesystem paths or stack traces in responses
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(
                error_message="Diagnostic inference encountered an internal error. Please check image framing and retry.",
                error_type="InferenceExecutionError",
                model_version="v2"
            )
        )


# -----------------------------------------------------------------------------
# Static Asset Serving & SPA Routing
# -----------------------------------------------------------------------------

FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    assets_dir = FRONTEND_DIR / "assets"

    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_class=FileResponse, summary="Serve LeafScan SPA")
    def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(str(index_file))

    @app.get("/index.html", response_class=FileResponse, include_in_schema=False)
    def serve_index_html():
        return serve_index()
