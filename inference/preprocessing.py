# inference/preprocessing.py
"""
Robust Image Preprocessing and Validation for V2 Hierarchical Inference.
Enforces RGB 3-channel layout, handles arbitrary color formats (Grayscale, RGBA, CMYK),
and generates properly scaled tensor inputs for V2 MobileNetV2-CBAM backbones.
"""

from pathlib import Path
from PIL import Image
import numpy as np

TARGET_SIZE = (224, 224)

def load_and_validate_image(image_input):
    """
    Validates and loads an image into a PIL Image instance.

    Args:
        image_input (str, Path, PIL.Image.Image, or np.ndarray): Image source.

    Returns:
        PIL.Image.Image: Loaded PIL Image.

    Raises:
        FileNotFoundError: If a supplied file path does not exist.
        ValueError: If image format is invalid or corrupted.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_input}")
        if not p.is_file():
            raise ValueError(f"Path is not a regular file: {image_input}")
        try:
            with Image.open(str(p)) as raw_img:
                img = raw_img.copy()
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable image file '{image_input}': {str(e)}") from e

    elif isinstance(image_input, Image.Image):
        img = image_input.copy()

    elif isinstance(image_input, np.ndarray):
        # Handle numpy array input
        if image_input.ndim == 2:
            img = Image.fromarray(image_input.astype(np.uint8), mode="L")
        elif image_input.ndim == 3:
            if image_input.shape[2] == 1:
                img = Image.fromarray(image_input[:, :, 0].astype(np.uint8), mode="L")
            elif image_input.shape[2] == 3:
                img = Image.fromarray(image_input.astype(np.uint8), mode="RGB")
            elif image_input.shape[2] == 4:
                img = Image.fromarray(image_input.astype(np.uint8), mode="RGBA")
            else:
                raise ValueError(f"Unsupported numpy array channel count: {image_input.shape[2]}")
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {image_input.ndim}D (expected 2D or 3D)")

    else:
        raise TypeError(
            f"Unsupported input type '{type(image_input).__name__}'. "
            f"Provide a file path (str/Path), PIL.Image.Image, or numpy.ndarray."
        )

    return img

def preprocess_image(image_input, target_size=TARGET_SIZE):
    """
    Complete preprocessing pipeline for V2 hierarchical inference:
      1. Loads and validates image input.
      2. Converts color mode (L, RGBA, CMYK, P) to standard 3-channel RGB.
      3. Resizes to target dimension (default 224x224) via bilinear interpolation.
      4. Casts to float32 numpy array with batch dimension (1, 224, 224, 3) in [0.0, 255.0].

    Args:
        image_input (str, Path, PIL.Image.Image, or np.ndarray): Raw input image.
        target_size (tuple): Width and height tuple (default: 224, 224).

    Returns:
        tuple: (batch_array, pil_rgb_image)
            - batch_array: np.ndarray of shape (1, 224, 224, 3), dtype float32.
            - pil_rgb_image: PIL Image in RGB format resized to target_size.
    """
    raw_img = load_and_validate_image(image_input)

    # Convert to RGB (handles Grayscale, RGBA with alpha blending, Palette, CMYK)
    if raw_img.mode != "RGB":
        if raw_img.mode == "RGBA":
            # Composite onto pure white background to prevent dark transparent artifacts
            background = Image.new("RGB", raw_img.size, (255, 255, 255))
            background.paste(raw_img, mask=raw_img.split()[3])
            rgb_img = background
        else:
            rgb_img = raw_img.convert("RGB")
    else:
        rgb_img = raw_img

    # High-quality resize
    resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to numpy array in [0.0, 255.0] range expected by internal Rescaling layer
    img_array = np.array(resized_img, dtype=np.float32)

    # Expand to batch dimension
    batch_array = np.expand_dims(img_array, axis=0)

    return batch_array, resized_img
