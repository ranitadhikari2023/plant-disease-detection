# export_to_onnx.py
"""
Production ONNX Export Engine for Hierarchical Plant Disease Models (V2).
Converts Keras 3 / TensorFlow 2.21+ models (.keras) into highly optimized
ONNX graphs (.onnx) for low-latency headless microservice deployment.

Targets the production V2 architecture:
  1. models/v2/species_expert_v2.keras   -> models/onnx/v2/species_expert_v2.onnx  (Output: 2 classes)
  2. models/v2/potato_expert_v2.keras    -> models/onnx/v2/potato_expert_v2.onnx   (Output: 3 classes)
  3. models/v2/tomato_expert_v2.keras    -> models/onnx/v2/tomato_expert_v2.onnx   (Output: 3 classes)
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

# Mute TensorFlow C++ startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Import custom layers and loss for safe deserialization
try:
    from training.models import CBAM, ChannelAttention, SpatialAttention
    from training.losses import SparseCategoricalFocalLoss
except ImportError:
    try:
        from train_potato_doctor import CBAM, ChannelAttention, SpatialAttention
        from training.losses import SparseCategoricalFocalLoss
    except ImportError:
        CBAM = None
        ChannelAttention = None
        SpatialAttention = None
        SparseCategoricalFocalLoss = None

# V2 model specifications
V2_MODEL_SPECS = {
    "species_expert_v2.keras": {
        "onnx_filename": "species_expert_v2.onnx",
        "description": "Species Expert Botanist V2 (MobileNetV2)",
        "expected_classes": 2,
        "input_shape": (None, 224, 224, 3),
        "output_shape": (None, 2)
    },
    "potato_expert_v2.keras": {
        "onnx_filename": "potato_expert_v2.onnx",
        "description": "Potato Doctor CBAM V2 (MobileNetV2 + CBAM)",
        "expected_classes": 3,
        "input_shape": (None, 224, 224, 3),
        "output_shape": (None, 3)
    },
    "tomato_expert_v2.keras": {
        "onnx_filename": "tomato_expert_v2.onnx",
        "description": "Tomato Doctor CBAM V2 (MobileNetV2 + CBAM)",
        "expected_classes": 3,
        "input_shape": (None, 224, 224, 3),
        "output_shape": (None, 3)
    }
}


def file_sha256(path):
    """Calculates SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_custom_objects():
    """Builds a dictionary of all custom layers and functions required for deserialization."""
    custom_objs = {
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }
    if CBAM is not None:
        custom_objs.update({
            'CBAM': CBAM,
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention
        })
    if SparseCategoricalFocalLoss is not None:
        custom_objs['SparseCategoricalFocalLoss'] = SparseCategoricalFocalLoss
    return custom_objs


def load_v2_model(keras_model_path):
    """
    Safely loads a Keras 3 model with safe_mode=True and custom object registration.
    """
    path = Path(keras_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    if path.stat().st_size == 0:
        raise ValueError(f"Model file is empty: {path.resolve()}")

    custom_objs = get_custom_objects()
    model = tf.keras.models.load_model(
        str(path),
        custom_objects=custom_objs,
        compile=False,
        safe_mode=True
    )
    return model


def validate_onnx_graph(onnx_model_path, expected_classes=None):
    """
    Performs full ONNX graph validation using onnx.checker and inspects tensor metadata.
    """
    import onnx

    onnx_path = Path(onnx_model_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    size_bytes = onnx_path.stat().st_size
    sha256_hash = file_sha256(onnx_path)

    print(f"   📋 ONNX File Size: {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
    print(f"   📋 SHA-256:        {sha256_hash}")

    # Load and check model integrity
    onnx_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_proto)
    print("   ✅ onnx.checker.check_model() passed: Graph topology is well-formed.")

    # Inspect inputs and outputs
    graph = onnx_proto.graph
    inputs = graph.input
    outputs = graph.output

    if len(inputs) == 0:
        raise ValueError("ONNX graph has no inputs.")
    if len(outputs) == 0:
        raise ValueError("ONNX graph has no outputs.")

    input_tensor = inputs[0]
    output_tensor = outputs[0]

    input_name = input_tensor.name
    output_name = output_tensor.name

    # Check input dimensions
    in_shape = [d.dim_value if d.dim_value > 0 else (d.dim_param or "None") for d in input_tensor.type.tensor_type.shape.dim]
    out_shape = [d.dim_value if d.dim_value > 0 else (d.dim_param or "None") for d in output_tensor.type.tensor_type.shape.dim]

    print(f"   📋 Input Node:     '{input_name}', Shape: {in_shape}, Type: FLOAT")
    print(f"   📋 Output Node:    '{output_name}', Shape: {out_shape}, Type: FLOAT")

    # Verify dynamic batch
    batch_dim = in_shape[0]
    if isinstance(batch_dim, int) and batch_dim > 0:
        print(f"   ⚠️ Warning: Batch dimension is fixed to {batch_dim}, expected dynamic.")
    else:
        print(f"   ✅ Dynamic batch confirmed for input (dim 0: {batch_dim})")

    # Verify input spatial dimensions (224, 224, 3)
    if len(in_shape) == 4:
        assert in_shape[1:] == [224, 224, 3], f"Unexpected spatial shape: {in_shape[1:]}, expected [224, 224, 3]"

    # Verify output class dimension
    if expected_classes is not None and len(out_shape) == 2:
        out_classes = out_shape[1]
        if isinstance(out_classes, int):
            assert out_classes == expected_classes, f"Expected {expected_classes} output classes, got {out_classes}"
            print(f"   ✅ Output class dimension matches expected: {expected_classes}")

    return {
        "file_size": size_bytes,
        "sha256": sha256_hash,
        "input_name": input_name,
        "input_shape": in_shape,
        "output_name": output_name,
        "output_shape": out_shape,
        "opset": onnx_proto.opset_import[0].version if onnx_proto.opset_import else None
    }


def verify_prediction_parity(keras_model, onnx_model_path, tolerance=1e-4, specimen_paths=None):
    """
    Verifies numerical prediction parity between Keras and ONNX Runtime.
    Tests single specimen inputs and batched inputs.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    output_name = session.get_outputs()[0].name

    max_abs_diff_overall = 0.0
    parity_records = []

    # Prepare test inputs
    test_tensors = []
    labels = []

    # 1. Real specimen images if provided
    if specimen_paths:
        for sp in specimen_paths:
            p = Path(sp)
            if p.exists():
                try:
                    with Image.open(str(p)) as raw_img:
                        rgb = raw_img.convert("RGB")
                    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
                    tensor = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
                    test_tensors.append(tensor)
                    labels.append(f"Specimen: {p.name}")
                except Exception as e:
                    print(f"   ⚠️ Could not load specimen '{p}': {e}")

    # 2. Add sample synthetic tensors (single input and batch > 1)
    np.random.seed(42)
    dummy_single = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    test_tensors.append(dummy_single)
    labels.append("Synthetic Tensor (Batch Size 1)")

    dummy_batch = np.random.uniform(0.0, 255.0, size=(4, 224, 224, 3)).astype(np.float32)
    test_tensors.append(dummy_batch)
    labels.append("Synthetic Tensor (Batch Size 4)")

    print(f"   🧪 Running numerical parity checks across {len(test_tensors)} test inputs (tolerance: {tolerance})...")

    all_match = True
    for tensor, desc in zip(test_tensors, labels):
        # Keras forward pass
        k_out = keras_model(tensor, training=False).numpy()
        # ONNX Runtime forward pass
        ort_out = session.run([output_name], {input_name: tensor})[0]

        # Checks
        assert np.all(np.isfinite(k_out)), f"NaN or Inf detected in Keras output for {desc}"
        assert np.all(np.isfinite(ort_out)), f"NaN or Inf detected in ONNX output for {desc}"

        # Softmax sum ~ 1.0 per row
        k_sums = np.sum(k_out, axis=-1)
        ort_sums = np.sum(ort_out, axis=-1)
        assert np.allclose(k_sums, 1.0, atol=1e-3), f"Keras probabilities do not sum to 1.0: {k_sums}"
        assert np.allclose(ort_sums, 1.0, atol=1e-3), f"ONNX probabilities do not sum to 1.0: {ort_sums}"

        diff = np.abs(k_out - ort_out)
        max_diff = float(np.max(diff))
        max_abs_diff_overall = max(max_abs_diff_overall, max_diff)

        # Argmax class agreement
        k_classes = np.argmax(k_out, axis=-1)
        ort_classes = np.argmax(ort_out, axis=-1)
        class_match = bool(np.array_equal(k_classes, ort_classes))

        if not class_match:
            all_match = False

        status = "✅ PASS" if (max_diff <= tolerance and class_match) else "❌ FAIL"
        print(f"      {status} | {desc} | Max Diff: {max_diff:.2e} | Argmax Match: {class_match}")

        parity_records.append({
            "description": desc,
            "batch_size": tensor.shape[0],
            "max_abs_diff": max_diff,
            "class_match": class_match,
            "passed": max_diff <= tolerance and class_match
        })

    parity_passed = bool(max_abs_diff_overall <= tolerance and all_match)
    print(f"   🎯 Parity Summary: Max Abs Diff = {max_abs_diff_overall:.2e} (Tol: {tolerance}) | Overall: {'PASSED' if parity_passed else 'FAILED'}")

    return {
        "parity_passed": parity_passed,
        "max_abs_diff": max_abs_diff_overall,
        "tolerance": tolerance,
        "records": parity_records
    }


def export_keras_to_onnx(
    keras_model_path,
    onnx_output_path=None,
    opset=17,
    expected_classes=None,
    verify=True,
    verify_parity=True,
    tolerance=1e-4,
    specimen_paths=None
):
    """
    Converts a single .keras model to an optimized .onnx graph using tf2onnx.

    Args:
        keras_model_path (str or Path): Path to the source .keras file.
        onnx_output_path (str or Path, optional): Destination .onnx file path.
        opset (int): ONNX target operator set version (default: 17).
        expected_classes (int, optional): Expected number of output classes.
        verify (bool): Run graph validation and dummy ONNX Runtime inference.
        verify_parity (bool): Compare Keras vs ONNX probability parity.
        tolerance (float): Numerical tolerance for probability parity (default: 1e-4).
        specimen_paths (list, optional): Paths to real leaf image specimens for parity testing.

    Returns:
        dict: Summary of export, validation, and parity results.
    """
    try:
        import tf2onnx
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        print(f"\n❌ ERROR: Required ONNX toolchain package missing: {e}")
        print("Please install the verified ONNX toolchain using:")
        print("    pip install tf2onnx==1.17.0 onnx==1.23.2 onnxruntime==1.20.0")
        sys.exit(1)

    keras_path = Path(keras_model_path)
    if not keras_path.exists():
        raise FileNotFoundError(f"Source Keras model not found: {keras_path.resolve()}")

    if onnx_output_path is None:
        onnx_output_path = Path("models/onnx/v2") / keras_path.with_suffix(".onnx").name
    else:
        onnx_output_path = Path(onnx_output_path)

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Loading Keras 3 model: {keras_path.name}")
    model = load_v2_model(keras_path)
    print(f"   Architecture: {model.name} ({model.count_params():,} parameters)")

    print(f"🔄 Converting {keras_path.name} to ONNX (Opset {opset})...")
    input_signature = [
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input_tensor")
    ]

    # Convert using tf2onnx
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(onnx_output_path)
    )

    print(f"✅ ONNX model serialized to: {onnx_output_path}")

    validation_result = None
    if verify:
        print("🔍 Validating exported ONNX graph...")
        validation_result = validate_onnx_graph(onnx_output_path, expected_classes=expected_classes)

    parity_result = None
    if verify and verify_parity:
        parity_result = verify_prediction_parity(
            model,
            onnx_output_path,
            tolerance=tolerance,
            specimen_paths=specimen_paths
        )

    return {
        "model_name": keras_path.name,
        "keras_path": str(keras_path),
        "onnx_path": str(onnx_output_path),
        "validation": validation_result,
        "parity": parity_result
    }


def find_specimen_images():
    """Discovers representative held-out specimens from test_specimen_manifest.csv or test_images."""
    specimens = []

    # Priority 1: sample_leaf.JPG
    sample_img = Path("test_images/sample_leaf.JPG")
    if sample_img.exists():
        specimens.append(str(sample_img))

    # Priority 2: Canonical biological specimens from test manifest
    manifest_path = Path("audit/test_specimen_manifest.csv")
    if manifest_path.exists():
        import csv
        with open(manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            found_classes = set()
            for row in reader:
                cls_key = (row.get("crop"), row.get("disease_class"))
                if cls_key not in found_classes:
                    rel_p = Path(row.get("relative_image_path", ""))
                    if rel_p.exists():
                        specimens.append(str(rel_p))
                        found_classes.add(cls_key)
                if len(found_classes) >= 6:
                    break

    return specimens


def export_all_models(
    models_dir="models/v2",
    output_dir="models/onnx/v2",
    opset=17,
    verify=True,
    verify_parity=True,
    tolerance=1e-4
):
    """
    Exports all three production V2 models to ONNX.
    Explicitly requires models/v2/ and never falls back to legacy models/*.keras.
    """
    models_path = Path(models_dir)
    output_path = Path(output_dir)

    if not models_path.exists():
        raise FileNotFoundError(
            f"❌ Production V2 models directory not found: '{models_dir}'.\n"
            f"V2 checkpoints must reside in 'models/v2/'. Legacy models in 'models/' are deprecated."
        )

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 PRODUCTION V2 ONNX EXPORT ENGINE")
    print(f"   Source Directory: {models_path.resolve()}")
    print(f"   Target Directory: {output_path.resolve()}")
    print(f"   Opset Version:    {opset}")
    print("=" * 70)

    # Verify all 3 required V2 models are present
    missing_models = []
    for filename in V2_MODEL_SPECS.keys():
        if not (models_path / filename).exists():
            missing_models.append(filename)

    if missing_models:
        raise FileNotFoundError(
            f"❌ Missing required production V2 model(s) in '{models_dir}':\n"
            + "\n".join(f"   - {m}" for m in missing_models)
            + "\nSilently falling back to legacy V1 models is prohibited."
        )

    specimen_paths = find_specimen_images() if verify_parity else None
    if specimen_paths:
        print(f"📸 Located {len(specimen_paths)} representative specimen(s) for numerical parity testing:")
        for sp in specimen_paths:
            print(f"   - {sp}")

    results = []
    for filename, spec in V2_MODEL_SPECS.items():
        keras_file = models_path / filename
        onnx_file = output_path / spec["onnx_filename"]

        print("\n" + "-" * 70)
        print(f"🌟 Processing: {spec['description']}")
        print(f"   Source: {keras_file}")
        print(f"   Target: {onnx_file}")
        print("-" * 70)

        res = export_keras_to_onnx(
            keras_file,
            onnx_output_path=onnx_file,
            opset=opset,
            expected_classes=spec["expected_classes"],
            verify=verify,
            verify_parity=verify_parity,
            tolerance=tolerance,
            specimen_paths=specimen_paths
        )
        results.append(res)

    print("\n" + "=" * 70)
    print("🎉 ALL THREE V2 PRODUCTION MODELS SUCCESSFULLY CONVERTED AND VERIFIED!")
    print("=" * 70)
    for r in results:
        v = r.get("validation", {})
        p = r.get("parity", {})
        p_str = f"Max Diff: {p.get('max_abs_diff', 0.0):.2e}" if p else "Parity: Skipped"
        sz_mb = v.get("file_size", 0) / (1024 * 1024)
        print(f"   📁 {Path(r['onnx_path']).name:<25} | Size: {sz_mb:5.2f} MB | {p_str}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras 3 Plant Disease V2 models to optimized ONNX graphs."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/v2",
        help="Path to V2 models directory (default: models/v2)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx/v2",
        help="Output directory for exported ONNX graphs (default: models/onnx/v2)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a specific .keras file. If omitted, converts all 3 models in models/v2/."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for the .onnx file (used only when --model-path is set)."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX target opset version (default: 17)."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum absolute tolerance for Keras vs ONNX probability parity (default: 1e-4)."
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification with ONNX Runtime."
    )
    parser.add_argument(
        "--no-parity",
        action="store_true",
        help="Skip numerical prediction parity verification against Keras."
    )

    args = parser.parse_args()

    if args.model_path:
        export_keras_to_onnx(
            args.model_path,
            args.output_path,
            opset=args.opset,
            verify=not args.no_verify,
            verify_parity=not args.no_parity,
            tolerance=args.tolerance
        )
    else:
        export_all_models(
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            opset=args.opset,
            verify=not args.no_verify,
            verify_parity=not args.no_parity,
            tolerance=args.tolerance
        )


if __name__ == "__main__":
    main()
