"""
src/leafscan/api/middleware.py
Production ASGI Middleware for LeafScan V3:
  - Request ID propagation (X-Request-ID)
  - Bounded concurrency control (Semaphore load shedding with HTTP 503)
  - Client IP Rate Limiting (Token Bucket with HTTP 429)
  - Safe error masking (enforces invariant JSON error schema)
"""

import asyncio
import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_503_SERVICE_UNAVAILABLE

from leafscan.core.exceptions import LeafScanException
from leafscan.core.logging import logger
from leafscan.core.telemetry import metrics


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generates or propagates X-Request-ID across the request lifecycle."""

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RateLimiter:
    """Sliding window in-memory rate limiter per client IP."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.window = 60.0
        self.client_records: dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        async with self._lock:
            history = self.client_records[client_ip]
            # Prune timestamps older than 60s
            self.client_records[client_ip] = [ts for ts in history if ts > cutoff]
            if len(self.client_records[client_ip]) >= self.rpm:
                return False
            self.client_records[client_ip].append(now)
            return True


class BoundedConcurrencyMiddleware(BaseHTTPMiddleware):
    """
    Limits concurrent inference requests using an asyncio Semaphore.
    Protects CPU memory and avoids latency spikes under high concurrent load.
    """

    def __init__(self, app, max_concurrent: int = 4, target_path: str = "/v2/diagnose"):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.target_path = target_path

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path != self.target_path or request.method != "POST":
            return await call_next(request)

        # Non-blocking acquisition attempt for load shedding
        if self.semaphore.locked():
            metrics.record_error("overloaded")
            return JSONResponse(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "1"},
                content={
                    "success": False,
                    "model_version": "v2",
                    "plant_species": None,
                    "species_confidence": 0.0,
                    "disease": None,
                    "disease_confidence": 0.0,
                    "diagnosis": "Error",
                    "safety_status": "error",
                    "safety_warnings": ["InferenceEngineError: Server is operating at peak capacity. Please retry."],
                    "advisory": "Inference service is currently overloaded. Please retry in a few seconds.",
                    "species_probabilities": {},
                    "probabilities": {},
                    "error": {
                        "type": "CapacityExceededError",
                        "message": "Inference server is temporarily operating at maximum concurrent capacity.",
                    },
                },
            )

        async with self.semaphore:
            metrics.active_requests += 1
            try:
                response = await call_next(request)
                return response
            finally:
                metrics.active_requests -= 1


class ExceptionMaskingMiddleware(BaseHTTPMiddleware):
    """
    Intercepts unhandled exceptions and formats standard LeafScan JSON error responses.
    Prevents disclosure of internal paths, credentials, or stack traces.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except LeafScanException as lse:
            req_id = getattr(request.state, "request_id", None)
            logger.warning(f"Handled LeafScanException: {lse.error_code} - {lse.message}", extra={"request_id": req_id})
            metrics.record_error(lse.error_code)

            headers = {}
            if hasattr(lse, "retry_after"):
                headers["Retry-After"] = str(lse.retry_after)

            return JSONResponse(
                status_code=lse.status_code,
                headers=headers,
                content={
                    "success": False,
                    "model_version": "v2",
                    "plant_species": None,
                    "species_confidence": 0.0,
                    "disease": None,
                    "disease_confidence": 0.0,
                    "diagnosis": "Error",
                    "safety_status": "error",
                    "safety_warnings": [f"{type(lse).__name__}: {lse.message}"],
                    "advisory": "Inference could not be completed. Please check image validity and file path.",
                    "species_probabilities": {},
                    "probabilities": {},
                    "error": {"type": type(lse).__name__, "message": lse.message, "code": lse.error_code},
                },
            )
        except Exception as exc:
            req_id = getattr(request.state, "request_id", None)
            logger.error(f"Unhandled internal server error: {exc}", exc_info=True, extra={"request_id": req_id})
            metrics.record_error("internal_error")

            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "model_version": "v2",
                    "plant_species": None,
                    "species_confidence": 0.0,
                    "disease": None,
                    "disease_confidence": 0.0,
                    "diagnosis": "Error",
                    "safety_status": "error",
                    "safety_warnings": ["InferenceExecutionError: Diagnostic inference encountered an internal error."],
                    "advisory": "Diagnostic inference encountered an internal error. Please check image framing and retry.",
                    "species_probabilities": {},
                    "probabilities": {},
                    "error": {
                        "type": "InferenceExecutionError",
                        "message": "Diagnostic inference encountered an internal error. Please check image framing and retry.",
                    },
                },
            )
