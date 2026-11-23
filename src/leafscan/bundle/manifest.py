"""
src/leafscan/bundle/manifest.py
Pydantic V2 schema for versioned LeafScan Model Bundles.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    filename: str
    sha256: str
    architecture: str
    num_classes: int
    input_name: str = "input_tensor"
    input_shape: List[Optional[int]] = Field(default_factory=lambda: [None, 224, 224, 3])
    output_name: str
    output_shape: List[Optional[int]]
    index_to_label: Dict[str, str]
    label_to_index: Dict[str, int]


class InputSpec(BaseModel):
    image_size: List[int] = Field(default_factory=lambda: [224, 224])
    channels: int = 3
    color_mode: str = "RGB"
    dtype: str = "float32"
    rescaling_range: List[float] = Field(default_factory=lambda: [0.0, 255.0])
    max_decoded_pixels: int = 40000000
    interpolation: str = "bilinear"
    alpha_matting_background: List[int] = Field(default_factory=lambda: [255, 255, 255])


class SafetyGates(BaseModel):
    species_confidence_threshold: float = 0.90
    late_blight_confidence_threshold: float = 0.90
    general_disease_confidence_threshold: float = 0.70


class BundleManifest(BaseModel):
    bundle_version: str
    created_at: str
    name: str
    description: str
    runtime: str = "onnxruntime"
    opset: int = 17
    disclaimer: str
    models: Dict[str, ModelSpec]
    input_spec: InputSpec = Field(default_factory=InputSpec)
    safety_gates: SafetyGates = Field(default_factory=SafetyGates)
    advisories: Dict[str, str]
    calibration: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, Any]] = None
