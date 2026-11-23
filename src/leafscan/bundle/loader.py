"""
src/leafscan/bundle/loader.py
Model bundle loading, verification, and metadata encapsulation.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from leafscan.bundle.manifest import BundleManifest
from leafscan.bundle.verification import verify_bundle_integrity
from leafscan.core.exceptions import ModelBundleError
from leafscan.core.logging import logger


class ModelBundle:
    """Encapsulates a verified, immutable LeafScan model bundle."""

    def __init__(self, bundle_dir: Path, manifest: BundleManifest, verified_hashes: Dict[str, str]):
        self.bundle_dir = bundle_dir
        self.manifest = manifest
        self.verified_hashes = verified_hashes
        self.version = manifest.bundle_version

    @property
    def model_paths(self) -> Dict[str, Path]:
        return {key: self.bundle_dir / spec.filename for key, spec in self.manifest.models.items()}

    def get_model_spec(self, key: str):
        if key not in self.manifest.models:
            raise KeyError(f"Model key '{key}' not in bundle manifest. Available: {list(self.manifest.models.keys())}")
        return self.manifest.models[key]


def load_model_bundle(bundle_dir: Path | str, verify_checksums: bool = True) -> ModelBundle:
    """
    Loads and validates a Model Bundle from directory.
    Reads manifest.json, parses schema, and optionally verifies SHA-256 checksums.
    """
    p = Path(bundle_dir).resolve()
    if not p.exists() or not p.is_dir():
        raise ModelBundleError(f"Model bundle directory does not exist: {p}")

    manifest_path = p / "manifest.json"
    if not manifest_path.exists():
        raise ModelBundleError(f"manifest.json missing in model bundle directory: {p}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = BundleManifest.model_validate(data)
    except Exception as e:
        raise ModelBundleError(f"Failed to parse bundle manifest '{manifest_path}': {e}") from e

    verified_hashes = {}
    if verify_checksums:
        verified_hashes = verify_bundle_integrity(p, manifest)

    logger.info(f"Loaded LeafScan Model Bundle '{manifest.name}' version {manifest.bundle_version}")
    return ModelBundle(p, manifest, verified_hashes)
