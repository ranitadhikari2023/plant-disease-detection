# predict_hierarchical_v2.py
"""
Production Entry Point for Tier 1 & Tier 2 Hierarchical Plant Disease Inference (V2).
Powered by MobileNetV2 with CBAM Spatial & Channel Attention and Dual Confidence Safety Gates.

Usage:
    python predict_hierarchical_v2.py --image path/to/leaf.jpg
    python predict_hierarchical_v2.py --image path/to/leaf.jpg --json
    python predict_hierarchical_v2.py  (defaults to test_images/sample_leaf.JPG)
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.predictor import predict_v2, HierarchicalPredictorV2

def find_default_sample_image():
    """Locates sample_leaf.JPG / sample_leaf.jpg in test_images/ across platforms."""
    test_dir = Path("test_images")
    if not test_dir.exists():
        return None
    for ext in ["JPG", "jpg", "jpeg", "PNG", "png"]:
        candidate = test_dir / f"sample_leaf.{ext}"
        if candidate.exists():
            return candidate
    # Any image in test_images/
    for p in test_dir.iterdir():
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return p
    return None

def print_cli_report(res: dict):
    """Prints a structured agricultural diagnostic report to stdout."""
    print("\n" + "=" * 65)
    print("🌿 V2 HIERARCHICAL PLANT PATHOLOGY DIAGNOSIS (CBAM ATTENTION)")
    print("=" * 65)

    if not res.get("success", False):
        print("❌ INFERENCE FAILED:")
        for w in res.get("safety_warnings", []):
            print(f"  • {w}")
        print(f"\nADVISORY: {res.get('advisory', '')}")
        print("=" * 65 + "\n")
        return

    # Species Botanist Output
    sp = res["plant_species"]
    sp_conf = res["species_confidence"] * 100.0
    print(f"🌱 BOTANIST SPECIES: {sp.upper()} (Confidence: {sp_conf:.2f}%)")
    sp_probs_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in res.get("species_probabilities", {}).items()])
    print(f"   Distribution: [{sp_probs_str}]")

    # Safety Gate 1 Warning
    if res["safety_status"] == "species_uncertain":
        print("\n⚠️  SAFETY GATE 1 TRIGGERED: SPECIES UNCERTAIN (<90%)")
        for w in res["safety_warnings"]:
            print(f"   • {w}")

    print("-" * 65)

    # Specialist Doctor Output
    dis = res["disease"]
    if dis is None:
        print("🔬 SPECIALIST DIAGNOSIS: ABSTAINED (Species uncertain - no diagnosis issued)")
    else:
        dis_conf = res["disease_confidence"] * 100.0 if res["disease_confidence"] is not None else 0.0
        print(f"🔬 SPECIALIST DIAGNOSIS: {dis} (Confidence: {dis_conf:.2f}%)")
        dis_probs_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in res.get("probabilities", {}).items()])
        print(f"   Distribution: [{dis_probs_str}]")

    # Safety Gate 2 Warning
    if res["safety_status"] == "late_blight_uncertain":
        print("\n⚠️  SAFETY GATE 2 TRIGGERED: LATE BLIGHT UNCERTAINTY (<90%)")
        for w in res["safety_warnings"]:
            print(f"   • {w}")
    elif res["safety_status"] == "disease_uncertain":
        print("\n⚠️  SAFETY GATE TRIGGERED: MARGINAL CONFIDENCE (<70%)")
        for w in res["safety_warnings"]:
            print(f"   • {w}")

    print("=" * 65)
    print(f"📋 ACTIONABLE ADVISORY:\n{res.get('advisory', '')}")
    if res.get("metadata", {}).get("execution_time_ms") is not None:
        print(f"\n⏱️  Latency: {res['metadata']['execution_time_ms']:.1f} ms | Model: V2 MobileNetV2 + CBAM")
    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="V2 Hierarchical Plant Disease Diagnosis (MobileNetV2 + CBAM)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input leaf image")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    image_path = args.image
    if image_path is None:
        default_img = find_default_sample_image()
        if default_img is None:
            print("Error: No image specified and default sample leaf not found in test_images/.", file=sys.stderr)
            print("Usage: python predict_hierarchical_v2.py --image path/to/leaf.jpg", file=sys.stderr)
            sys.exit(1)
        image_path = str(default_img)
        if not args.json:
            print(f"🔎 Using default sample leaf: {image_path}")

    result = predict_v2(image_path)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_cli_report(result)

    if not result.get("success", False):
        sys.exit(1)

if __name__ == "__main__":
    main()
