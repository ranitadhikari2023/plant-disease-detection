"""
src/leafscan/api/app.py
FastAPI Application Factory with lifespan management for LeafScan V3.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from leafscan.api.middleware import BoundedConcurrencyMiddleware, RequestIdMiddleware
from leafscan.api.routes import diagnose, feedback, health
from leafscan.api.settings import Settings, get_settings
from leafscan.bundle.loader import load_model_bundle
from leafscan.core.logging import logger, setup_logging
from leafscan.core.telemetry import metrics
from leafscan.engines.onnx_engine import OnnxEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: initializes verified model bundle and warms up ONNX engine."""
    settings = getattr(app.state, "settings", None) or get_settings()
    setup_logging(level=settings.log_level, json_format=settings.log_json)

    logger.info("Initializing LeafScan V3 production runtime...")
    t_start = time.perf_counter()

    try:
        # Step 1: Load and verify model bundle
        bundle_path = Path(settings.model_bundle_path)
        if not bundle_path.is_absolute():
            bundle_path = PROJECT_ROOT / bundle_path

        bundle = load_model_bundle(bundle_path, verify_checksums=settings.verify_bundle_checksums)
        app.state.bundle = bundle

        # Step 2: Initialize ONNX inference engine
        engine = OnnxEngine(
            bundle=bundle,
            execution_provider=settings.execution_provider,
            intra_op_threads=settings.intra_op_threads,
            inter_op_threads=settings.inter_op_threads,
        )
        app.state.engine = engine

        # Step 3: Warmup JIT and session threadpools
        engine.warmup()

        init_duration = (time.perf_counter() - t_start) * 1000.0
        metrics.model_load_time_ms = init_duration
        metrics.bundle_version = bundle.version
        logger.info(f"LeafScan V3 ready. Initialized in {init_duration:.1f} ms.")
    except Exception as e:
        logger.critical(f"Fatal error during model bundle initialization: {e}", exc_info=True)
        # Store None so readiness probe reports 503 safely
        app.state.bundle = None
        app.state.engine = None

    yield

    logger.info("LeafScan V3 shutting down cleanly.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory for LeafScan V3 REST API."""
    cfg = settings or get_settings()

    app = FastAPI(
        title=cfg.app_title,
        description="Hierarchical Two-Tier Foliar Screening with CBAM Attention for Solanaceous Crops (LeafScan V3 Production Engine)",
        version=cfg.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = cfg

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Concurrency Control Middleware (load shedding on peak concurrency)
    app.add_middleware(
        BoundedConcurrencyMiddleware, max_concurrent=cfg.max_concurrent_inferences, target_path="/v2/diagnose"
    )

    # 3. Request ID Propagation Middleware
    app.add_middleware(RequestIdMiddleware)

    # 4. Include REST Routes
    app.include_router(health.router)
    app.include_router(diagnose.router)
    app.include_router(feedback.router)

    # 5. Static Assets & Frontend SPA Mounting
    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.exists():
        css_dir = frontend_dir / "css"
        js_dir = frontend_dir / "js"
        assets_dir = frontend_dir / "assets"

        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", response_class=FileResponse, summary="Serve LeafScan SPA", tags=["Frontend"])
        def serve_index():
            index_file = frontend_dir / "index.html"
            if not index_file.exists():
                raise HTTPException(status_code=404, detail="Frontend index.html not found.")
            return FileResponse(str(index_file))

        @app.get("/index.html", response_class=FileResponse, include_in_schema=False)
        def serve_index_html():
            return serve_index()

    return app


# Module-level instance for ASGI servers
app = create_app()
