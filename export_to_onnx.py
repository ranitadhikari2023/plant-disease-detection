# export_to_onnx.py
"""
Production ONNX Export Engine for Hierarchical Plant Disease Models.
Converts Keras 3 / TensorFlow 2.21+ models (.keras) into highly optimized
ONNX graphs (.onnx) for low-latency headless microservice deployment.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Mute TensorFlow C++ startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Import custom layers and loss for deserialization
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

# Backward compatibility hook for Dense deserialization in Keras 3.12
import keras
orig_dense_from_config = keras.layers.Dense.from_config
@classmethod
def patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return orig_dense_from_config.__func__(cls, config)
keras.layers.Dense.from_config = patched_dense_from_config


def get_custom_objects():
    """Builds a dictionary of all custom layers and functions required for deserialization."""
    custom_objs = {
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }
    if CBAM is not None:
        custom_objs.update({
            'CBAM': CBAM,
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention,
            'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss
        })
    return custom_objs


def export_keras_to_onnx(keras_model_path, onnx_output_path=None, opset=17, verify=True):
    """
    Converts a single .keras model to an optimized .onnx graph using tf2onnx.

    Args:
        keras_model_path (str or Path): Path to the source .keras file.
        onnx_output_path (str or Path, optional): Destination .onnx file path.
        opset (int): ONNX target operator set version (default: 17).
        verify (bool): Run dummy inference test via ONNX Runtime if installed.

    Returns:
        str: Absolute path to the exported ONNX model.
    """
    try:
        import tf2onnx
    except ImportError:
        print("\n❌ ERROR: 'tf2onnx' is not installed in the active environment.")
        print("Please install the ONNX toolchain using:")
        print("    pip install tf2onnx onnx onnxruntime")
        sys.exit(1)

    keras_path = Path(keras_model_path)
    if not keras_path.exists():
        raise FileNotFoundError(f"Model file not found: {keras_path}")

    if onnx_output_path is None:
        onnx_output_path = keras_path.with_suffix(".onnx")
    else:
        onnx_output_path = Path(onnx_output_path)

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Loading Keras 3 model: {keras_path.name}")
    custom_objs = get_custom_objects()
    model = tf.keras.models.load_model(str(keras_path), custom_objects=custom_objs, compile=False)

    print(f"🔄 Converting {keras_path.name} to ONNX (Opset {opset})...")
    # Define input specification with dynamic batch size
    input_signature = [
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input_tensor")
    ]

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(onnx_output_path)
    )

    print(f"✅ Exported successfully to: {onnx_output_path}")

    if verify:
        verify_onnx_model(onnx_output_path)

    return str(onnx_output_path)


def verify_onnx_model(onnx_model_path):
    """
    Validates the exported ONNX model using ONNX Runtime with a dummy tensor.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️ Warning: 'onnxruntime' is not installed. Skipping model inference verification.")
        print("   Run 'pip install onnxruntime' to enable automated verification.")
        return

    print("🧪 Verifying ONNX graph with ONNX Runtime...")
    session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    output_name = session.get_outputs()[0].name

    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    raw_outputs = session.run([output_name], {input_name: dummy_input})
    predictions = raw_outputs[0]

    print(f"   Input Node:  '{input_name}', Shape: {input_info.shape}")
    print(f"   Output Node: '{output_name}', Output Shape: {predictions.shape}")
    print(f"   Sample Output Probabilities: {np.round(predictions[0], 4)}")
    print("✅ Model verification complete: Graph is valid and executable.")


def export_all_models(models_dir="models", opset=17, verify=True):
    """
    Discovers and exports all .keras models located in the target directory.
    """
    models_path = Path(models_dir)
    keras_files = list(models_path.glob("*.keras"))

    if not keras_files:
        print(f"❌ No .keras models found in directory '{models_dir}'.")
        return

    print(f"🚀 Found {len(keras_files)} model(s) to export in '{models_dir}':")
    for kf in keras_files:
        print(f"   - {kf.name}")

    exported = []
    for kf in keras_files:
        out_onnx = kf.with_suffix(".onnx")
        exported_path = export_keras_to_onnx(kf, out_onnx, opset=opset, verify=verify)
        exported.append(exported_path)

    print("\n" + "=" * 60)
    print(f"🎉 All {len(exported)} models successfully converted to ONNX!")
    for ep in exported:
        print(f"   📁 {ep}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Keras 3 plant pathology models to optimized ONNX graphs."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a specific .keras file. If omitted, converts all models in models/."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for the .onnx file (used only when --model_path is set)."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX target opset version (default: 17)."
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip verification with ONNX Runtime."
    )

    args = parser.parse_args()

    if args.model_path:
        export_keras_to_onnx(
            args.model_path,
            args.output_path,
            opset=args.opset,
            verify=not args.no_verify
        )
    else:
        export_all_models(
            models_dir="models",
            opset=args.opset,
            verify=not args.no_verify
        )
