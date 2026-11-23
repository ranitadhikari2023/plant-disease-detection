"""
src/leafscan/core/exceptions.py
Structured exception hierarchy for LeafScan V3.
"""

from typing import Optional, Dict, Any


class LeafScanException(Exception):
    """Base exception for all LeafScan errors."""
    def __init__(self, message: str, error_code: str = "internal_error", status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ModelBundleError(LeafScanException):
    """Raised when a model bundle is missing, invalid, or corrupted."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="model_unavailable", status_code=503, details=details)


class ChecksumMismatchError(ModelBundleError):
    """Raised when SHA-256 integrity verification fails for a model artifact."""
    def __init__(self, filename: str, expected: str, actual: str):
        message = f"Model artifact '{filename}' checksum mismatch. Integrity verification failed."
        super().__init__(message, details={"filename": filename, "expected": expected, "actual": actual})


class ImageIntakeError(LeafScanException):
    """Base error for image validation or decoding failures."""
    def __init__(self, message: str, error_code: str = "invalid_image", status_code: int = 400, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code, status_code=status_code, details=details)


class PayloadTooLargeError(ImageIntakeError):
    """Raised when image upload byte size exceeds limit."""
    def __init__(self, max_bytes: int, actual_bytes: int):
        message = f"File exceeds maximum size limit of {max_bytes // (1024 * 1024)} MB."
        super().__init__(message, error_code="file_too_large", status_code=413, details={"max_bytes": max_bytes, "actual_bytes": actual_bytes})


class UnsupportedFormatError(ImageIntakeError):
    """Raised when image format is not in allowed list."""
    def __init__(self, fmt: str, allowed: set):
        message = f"Unsupported image format '{fmt}'. Allowed formats: {', '.join(sorted(allowed))}."
        super().__init__(message, error_code="unsupported_format", status_code=415, details={"format": fmt, "allowed": list(allowed)})


class PixelLimitExceededError(ImageIntakeError):
    """Raised when decoded image pixel dimensions exceed safety threshold (decompression bomb guard)."""
    def __init__(self, pixels: int, max_pixels: int):
        message = f"Image dimensions ({pixels:,} pixels) exceed safety limit of {max_pixels:,} pixels."
        super().__init__(message, error_code="image_too_large_pixels", status_code=400, details={"pixels": pixels, "max_pixels": max_pixels})


class PoorImageQualityError(ImageIntakeError):
    """Raised when image quality screening fails (e.g. excessive blur, extreme underexposure)."""
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Image quality insufficient for pathology diagnosis: {reason}."
        super().__init__(message, error_code="poor_image_quality", status_code=422, details=details)


class InferenceEngineError(LeafScanException):
    """Raised when runtime neural network inference encounters a failure."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="inference_error", status_code=500, details=details)


class ConcurrencyLimitExceededError(LeafScanException):
    """Raised when inference capacity is saturated (load shedding)."""
    def __init__(self, retry_after: int = 1):
        super().__init__(
            "Inference service is operating at peak capacity. Please retry shortly.",
            error_code="overloaded",
            status_code=503,
            details={"retry_after": retry_after}
        )
        self.retry_after = retry_after


class RateLimitExceededError(LeafScanException):
    """Raised when client exceeds rate limit."""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            "Rate limit exceeded. Please throttle your requests.",
            error_code="rate_limit_exceeded",
            status_code=429,
            details={"retry_after": retry_after}
        )
        self.retry_after = retry_after
