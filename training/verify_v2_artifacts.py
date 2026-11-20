# training/verify_v2_artifacts.py
"""
Deep inspection and validation of saved V2 model artifacts in a clean process.
Verifies file integrity, parameter counts, tensor dimensions, and CBAM attention submodules.
"""

import os
import sys
import hashlib

sys.path.insert(0, os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from training.models import CBAM, ChannelAttention, SpatialAttention

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def inspect_models():
    custom_objs = {
        'CBAM': CBAM,
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }

    models_info = [
        ("Species Expert V2", "models/v2/species_expert_v2.keras", 2, False),
        ("Potato Doctor CBAM V2", "models/v2/potato_expert_v2.keras", 3, True),
        ("Tomato Doctor CBAM V2", "models/v2/tomato_expert_v2.keras", 3, True)
    ]

    results = {}

    for name, path, expected_classes, expect_cbam in models_info:
        print("=" * 70)
        print(f"🔍 INSPECTING: {name}")
        print(f"Path: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file missing: {path}")

        size = os.path.getsize(path)
        sha = file_sha256(path)
        print(f"File Size:    {size:,} bytes")
        print(f"SHA-256:      {sha}")

        # Test deserialization with compile=False
        model = tf.keras.models.load_model(path, custom_objects=custom_objs, compile=False, safe_mode=False)

        input_shape = model.input_shape
        output_shape = model.output_shape
        total_params = model.count_params()
        trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
        non_trainable_params = sum(tf.size(w).numpy() for w in model.non_trainable_weights)

        print(f"Model Name:   {model.name}")
        print(f"Input Shape:  {input_shape}")
        print(f"Output Shape: {output_shape}")
        print(f"Params Total: {total_params:,} (Trainable: {trainable_params:,}, Non-trainable: {non_trainable_params:,})")

        # Layer checks
        layer_names = [l.name for l in model.layers]
        layer_classes = [l.__class__.__name__ for l in model.layers]

        has_cbam = any(c == 'CBAM' for c in layer_classes)
        cbam_details = None

        if expect_cbam:
            cbam_layer = next((l for l in model.layers if l.__class__.__name__ == 'CBAM'), None)
            if cbam_layer is None:
                raise ValueError(f"Expected CBAM layer in {name}, but none found!")
            
            # Inspect CBAM internal components
            ch_att = getattr(cbam_layer, 'channel_attention', None)
            sp_att = getattr(cbam_layer, 'spatial_attention', None)
            cbam_weights = sum(tf.size(w).numpy() for w in cbam_layer.weights)
            
            cbam_details = {
                "layer_name": cbam_layer.name,
                "channel_attention_present": ch_att is not None,
                "spatial_attention_present": sp_att is not None,
                "cbam_parameter_count": int(cbam_weights)
            }
            print(f"✅ CBAM Attention Verified: {cbam_layer.name} ({cbam_weights:,} params)")
            print(f"   ├─ ChannelAttention: {ch_att.__class__.__name__}")
            print(f"   └─ SpatialAttention: {sp_att.__class__.__name__}")
        else:
            print("ℹ️ Standard Backbone (No CBAM expected)")

        # Verify output dimension matches expected classes
        assert output_shape[-1] == expected_classes, f"Expected {expected_classes} classes, got {output_shape[-1]}"

        # Test dummy forward pass
        dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        out = model(dummy_input, training=False)
        assert out.shape == (1, expected_classes), f"Unexpected output shape: {out.shape}"
        assert abs(tf.reduce_sum(out).numpy() - 1.0) < 1e-4, f"Softmax sum != 1.0: {tf.reduce_sum(out).numpy()}"
        print(f"✅ Dummy Forward Pass Success: Output shape {out.shape}, sum = {tf.reduce_sum(out).numpy():.4f}")

        results[name] = {
            "path": path,
            "size_bytes": size,
            "sha256": sha,
            "model_name": model.name,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "non_trainable_params": int(non_trainable_params),
            "has_cbam": has_cbam,
            "cbam_details": cbam_details
        }

    print("\n" + "=" * 70)
    print("ALL THREE V2 MODELS SUCCESSFULLY VERIFIED IN CLEAN PROCESS!")
    print("=" * 70)
    return results

if __name__ == "__main__":
    inspect_models()
