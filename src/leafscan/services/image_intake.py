"""
src/leafscan/services/image_intake.py
Defensive Image Ingestion, Validation, Sanitization, and Preprocessing.
Enforces EXIF transposition, pixel limits, white matting, quality screening, and float32 scaling.
"""

import io
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageOps

from leafscan.core.exceptions import (
    ImageIntakeError,
    PayloadTooLargeError,
    UnsupportedFormatError,
    PixelLimitExceededError,
    PoorImageQualityError
)

# Set Pillow decompression bomb maximum
Image.MAX_IMAGE_PIXELS = 40_000_000


class ImageIntake:
    """Production-hardened image ingestion and normalization pipeline."""

    def __init__(
        self,
        max_upload_bytes: int = 10 * 1024 * 1024,  # 10 MB
        allowed_formats: Optional[set] = None,
        max_decoded_pixels: int = 40_000_000,      # 40 Megapixels
        target_size: Tuple[int, int] = (224, 224),
        enable_quality_checks: bool = False,
        blur_threshold: float = 30.0
    ):
        self.max_upload_bytes = max_upload_bytes
        self.allowed_formats = allowed_formats or {"JPEG", "PNG", "WEBP"}
        self.max_decoded_pixels = max_decoded_pixels
        self.target_size = target_size
        self.enable_quality_checks = enable_quality_checks
        self.blur_threshold = blur_threshold

    def validate_file_bytes(self, raw_bytes: bytes) -> None:
        """Validates payload byte size and non-emptiness."""
        if len(raw_bytes) == 0:
            raise ImageIntakeError("Uploaded file is empty (0 bytes).", error_code="invalid_image", status_code=400)
        if len(raw_bytes) > self.max_upload_bytes:
            raise PayloadTooLargeError(self.max_upload_bytes, len(raw_bytes))

    def decode_and_orient(self, source: Union[bytes, str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Decodes raw bytes or converts existing representations into a clean PIL Image.
        Normalizes EXIF orientation and enforces pixel dimension limits.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise ImageIntakeError(f"Image file does not exist: {p}", error_code="invalid_image", status_code=404)
            try:
                with open(p, "rb") as f:
                    raw_bytes = f.read()
                return self.decode_and_orient(raw_bytes)
            except Exception as e:
                raise ImageIntakeError(f"Unable to read image from '{p}': {e}", error_code="invalid_image", status_code=400) from e

        if isinstance(source, np.ndarray):
            # Convert numpy array to PIL Image
            if source.ndim == 2:
                img = Image.fromarray(source.astype(np.uint8), mode="L")
            elif source.ndim == 3:
                channels = source.shape[2]
                if channels == 1:
                    img = Image.fromarray(source[:, :, 0].astype(np.uint8), mode="L")
                elif channels == 3:
                    img = Image.fromarray(source.astype(np.uint8), mode="RGB")
                elif channels == 4:
                    img = Image.fromarray(source.astype(np.uint8), mode="RGBA")
                else:
                    raise ImageIntakeError(f"Unsupported numpy array channel count: {channels}", status_code=400)
            else:
                raise ImageIntakeError(f"Unsupported numpy array dimensions: {source.ndim}D", status_code=400)
            return img

        if isinstance(source, Image.Image):
            raw_img = source.copy()
        elif isinstance(source, (bytes, bytearray)):
            self.validate_file_bytes(source)
            try:
                byte_stream = io.BytesIO(source)
                with Image.open(byte_stream) as opened_img:
                    img_fmt = (opened_img.format or "").upper()
                    if img_fmt not in self.allowed_formats:
                        raise UnsupportedFormatError(img_fmt, self.allowed_formats)
                    # Check decoded pixel dimensions before loading full buffer
                    w, h = opened_img.size
                    pixels = w * h
                    if pixels > self.max_decoded_pixels:
                        raise PixelLimitExceededError(pixels, self.max_decoded_pixels)
                    raw_img = opened_img.copy()
            except ImageIntakeError:
                raise
            except Image.DecompressionBombError:
                raise PixelLimitExceededError(0, self.max_decoded_pixels)
            except Exception as e:
                raise ImageIntakeError("Invalid or corrupted image payload. Unable to decode image raster.", error_code="invalid_image", status_code=400) from e
        else:
            raise ImageIntakeError(f"Unsupported source type: {type(source).__name__}", error_code="invalid_image", status_code=400)

        # Normalize EXIF orientation if present
        try:
            raw_img = ImageOps.exif_transpose(raw_img)
        except Exception:
            pass  # Fall back to un-transposed image if EXIF is malformed

        # Check total pixel count on decoded image
        pixels = raw_img.width * raw_img.height
        if pixels > self.max_decoded_pixels:
            raise PixelLimitExceededError(pixels, self.max_decoded_pixels)

        return raw_img

    def check_quality(self, pil_rgb: Image.Image) -> None:
        """Lightweight image quality heuristic checks (blur, under/over-exposure)."""
        if not self.enable_quality_checks:
            return

        gray = pil_rgb.convert("L")
        arr = np.array(gray, dtype=np.float32)
        mean_val = float(np.mean(arr))

        if mean_val < 15.0:
            raise PoorImageQualityError("Image is extremely dark / underexposed", details={"mean_brightness": mean_val})
        if mean_val > 245.0:
            raise PoorImageQualityError("Image is completely washed out / overexposed", details={"mean_brightness": mean_val})

        # Fast 3x3 Laplacian variance blur check
        # [-1, -1, -1], [-1, 8, -1], [-1, -1, -1]
        lap = (
            arr[:-2, :-2] * -1 + arr[:-2, 1:-1] * -1 + arr[:-2, 2:] * -1 +
            arr[1:-1, :-2] * -1 + arr[1:-1, 1:-1] * 8 + arr[1:-1, 2:] * -1 +
            arr[2:, :-2] * -1 + arr[2:, 1:-1] * -1 + arr[2:, 2:] * -1
        )
        lap_var = float(np.var(lap))
        if lap_var < self.blur_threshold:
            raise PoorImageQualityError("Image is severely blurred", details={"laplacian_variance": lap_var, "threshold": self.blur_threshold})

    def process(self, source: Union[bytes, str, Path, Image.Image, np.ndarray]) -> Tuple[np.ndarray, Image.Image]:
        """
        Executes full intake and normalization:
          1. Decodes and transposes EXIF.
          2. Composites RGBA over white background (eliminates black alpha artifacts).
          3. Resizes to (224, 224) via bilinear interpolation.
          4. Casts to (1, 224, 224, 3) float32 in range [0.0, 255.0].
        """
        raw_img = self.decode_and_orient(source)

        # Alpha compositing & mode conversion
        if raw_img.mode != "RGB":
            if raw_img.mode == "RGBA":
                background = Image.new("RGB", raw_img.size, (255, 255, 255))
                background.paste(raw_img, mask=raw_img.split()[3])
                rgb_img = background
            else:
                rgb_img = raw_img.convert("RGB")
        else:
            rgb_img = raw_img

        self.check_quality(rgb_img)

        # High quality resize
        resized_img = rgb_img.resize(self.target_size, Image.Resampling.BILINEAR)

        # Convert to numpy float32 array in [0.0, 255.0]
        img_array = np.array(resized_img, dtype=np.float32)
        batch_tensor = np.expand_dims(img_array, axis=0)

        return batch_tensor, resized_img
