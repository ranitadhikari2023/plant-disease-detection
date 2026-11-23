"""
src/leafscan/bundle/verification.py
Cryptographic verification for LeafScan Model Bundles.
"""

import hashlib
from pathlib import Path
from typing import Dict

from leafscan.bundle.manifest import BundleManifest
from leafscan.core.exceptions import ChecksumMismatchError, ModelBundleError
from leafscan.core.logging import logger


def compute_file_sha256(path: Path) -> str:
    """Computes hexadecimal SHA-256 digest of a local file in streaming blocks."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_bundle_integrity(bundle_dir: Path, manifest: BundleManifest) -> Dict[str, str]:
    """
    Verifies that all model artifacts referenced in the manifest exist on disk
    and match their cryptographic SHA-256 checksums exactly.

    Raises:
        ModelBundleError: If an expected model artifact is missing or unreadable.
        ChecksumMismatchError: If the computed SHA-256 does not match the manifest.
    """
    verified_hashes = {}

    for key, spec in manifest.models.items():
        model_path = bundle_dir / spec.filename
        if not model_path.exists():
            raise ModelBundleError(
                f"Missing model artifact '{spec.filename}' in bundle directory '{bundle_dir}'.",
                details={"model_key": key, "expected_path": str(model_path)}
            )

        actual_hash = compute_file_sha256(model_path)
        expected_hash = spec.sha256.lower().strip()

        if actual_hash != expected_hash:
            logger.error(
                f"SHA-256 verification failed for {spec.filename}: expected {expected_hash}, got {actual_hash}"
            )
            raise ChecksumMismatchError(spec.filename, expected=expected_hash, actual=actual_hash)

        verified_hashes[key] = actual_hash
        logger.info(f"Verified artifact integrity: {spec.filename} [SHA-256: {actual_hash[:12]}...]")

    return verified_hashes
