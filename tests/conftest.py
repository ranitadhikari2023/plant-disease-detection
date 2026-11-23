# tests/conftest.py
"""
Shared Pytest fixtures for V2 Hierarchical Plant Disease testing.
Provides reusable specimen image paths, clean model caching fixtures,
and synthetic dataset generators for pipeline and loss testing.
"""

import csv
import os

# Ensure root directory and src directory are on Python path
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@pytest.fixture(autouse=True)
def clean_model_cache():
    """Ensures model singleton cache is cleared before and after each test without eager TF imports."""
    try:
        from inference.model_loader import unload_models

        unload_models()
    except (ImportError, Exception):
        pass
    yield
    try:
        from inference.model_loader import unload_models

        unload_models()
    except (ImportError, Exception):
        pass


@pytest.fixture
def sample_leaf_path():
    """Provides the path to the canonical sample leaf image."""
    p = Path("test_images/sample_leaf.JPG")
    if not p.exists():
        p = Path("test_images/sample_leaf.jpg")
    assert p.exists(), f"Sample leaf image not found at {p}"
    return str(p)


@pytest.fixture
def synthetic_jpeg_factory(tmp_path):
    """
    Factory fixture to generate temporary valid JPEG images with custom dimensions and colors.
    """

    def _create_jpeg(filename="leaf.jpg", size=(256, 256), color=(60, 140, 50), mode="RGB"):
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / filename

        img = Image.new(mode, size, color)
        if mode == "RGBA":
            # Save as PNG first or convert for JPEG
            img.convert("RGB").save(str(img_path), format="JPEG")
        else:
            img.save(str(img_path), format="JPEG")
        return str(img_path)

    return _create_jpeg


@pytest.fixture
def synthetic_manifest_factory(tmp_path, synthetic_jpeg_factory):
    """
    Factory fixture to create temporary CSV manifests with synthetic images for pipeline testing.
    """

    def _create_manifest(
        rows=None,
        filename="test_manifest.csv",
        corrupt_image=False,
        missing_image=False,
        include_headers=True,
        headers=None,
    ):
        manifest_file = tmp_path / filename

        if headers is None:
            headers = [
                "source_group_id",
                "relative_image_path",
                "crop",
                "disease_class",
                "original_or_augmented",
                "assigned_split",
            ]

        if rows is None:
            # Default 6 canonical specimens (2 crops x 3 conditions)
            rows = []
            specs = [
                ("POTATO_GRP_01", "potato", "Early_blight", "p_eb.jpg"),
                ("POTATO_GRP_02", "potato", "Late_blight", "p_lb.jpg"),
                ("POTATO_GRP_03", "potato", "healthy", "p_h.jpg"),
                ("TOMATO_GRP_01", "tomato", "Early_blight", "t_eb.jpg"),
                ("TOMATO_GRP_02", "tomato", "Late_blight", "t_lb.jpg"),
                ("TOMATO_GRP_03", "tomato", "healthy", "t_h.jpg"),
            ]
            for gid, crop, dis, fname in specs:
                if missing_image:
                    img_path = str(tmp_path / "non_existent.jpg")
                elif corrupt_image:
                    bad_file = tmp_path / "images" / fname
                    bad_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(bad_file, "wb") as f:
                        f.write(b"not a valid jpeg content")
                    img_path = str(bad_file)
                else:
                    img_path = synthetic_jpeg_factory(filename=fname)

                rows.append(
                    {
                        "source_group_id": gid,
                        "relative_image_path": img_path,
                        "crop": crop,
                        "disease_class": dis,
                        "original_or_augmented": "original",
                        "assigned_split": "train",
                    }
                )

        with open(manifest_file, "w", newline="", encoding="utf-8") as f:
            if include_headers:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in headers})
            else:
                for r in rows:
                    f.write(",".join(str(r.get(k, "")) for k in headers) + "\n")

        return str(manifest_file)

    return _create_manifest
