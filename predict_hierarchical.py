# predict_hierarchical.py
import os
from pathlib import Path
from PIL import Image
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# --- 1. Constants & Vocabularies ---
SPECIES_LABELS = ['Potato', 'Tomato']
DISEASE_LABELS = ['Early Blight', 'Late Blight', 'Healthy']
THRESHOLD = 90.0

# Global model references (cached singletons)
MODEL_SPECIES = None
MODEL_POTATO = None
MODEL_TOMATO = None


def load_models(models_dir="models"):
    """
    Loads and caches the hierarchical deep learning models.
    Can be invoked eagerly at server startup or lazily upon first inference.
    """
    global MODEL_SPECIES, MODEL_POTATO, MODEL_TOMATO

    if MODEL_SPECIES is not None and MODEL_POTATO is not None and MODEL_TOMATO is not None:
        return MODEL_SPECIES, MODEL_POTATO, MODEL_TOMATO

    custom_objs = {
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }
    # Backward compatibility hook for Dense deserialization in Keras 3.12
    import keras
    orig_dense_from_config = keras.layers.Dense.from_config
    @classmethod
    def patched_dense_from_config(cls, config):
        config.pop('quantization_config', None)
        return orig_dense_from_config.__func__(cls, config)
    keras.layers.Dense.from_config = patched_dense_from_config

    try:
        from training.models import CBAM, ChannelAttention, SpatialAttention
        from training.losses import SparseCategoricalFocalLoss
        custom_objs.update({
            'CBAM': CBAM,
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention,
            'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss
        })
    except ImportError:
        try:
            from train_potato_doctor import CBAM, ChannelAttention, SpatialAttention
            custom_objs.update({
                'CBAM': CBAM,
                'ChannelAttention': ChannelAttention,
                'SpatialAttention': SpatialAttention
            })
        except ImportError:
            pass

    base_dir = Path(models_dir)
    species_path = base_dir / "species_expert.keras"
    potato_path = base_dir / "potato_expert.keras"
    tomato_path = base_dir / "tomato_expert.keras"

    if not (species_path.exists() and potato_path.exists() and tomato_path.exists()):
        raise FileNotFoundError(
            f"Missing required model weights in directory '{models_dir}'. "
            f"Ensure species_expert.keras, potato_expert.keras, and tomato_expert.keras exist."
        )

    print("Loading Medical Team Models...")
    MODEL_SPECIES = tf.keras.models.load_model(str(species_path), custom_objects=custom_objs)
    MODEL_POTATO = tf.keras.models.load_model(str(potato_path), custom_objects=custom_objs)
    MODEL_TOMATO = tf.keras.models.load_model(str(tomato_path), custom_objects=custom_objs)

    return MODEL_SPECIES, MODEL_POTATO, MODEL_TOMATO


def run_inference(image_input):
    """
    Executes the two-tier hierarchical inference pipeline.

    Args:
        image_input (str, Path, or PIL.Image.Image): File path or PIL Image object.

    Returns:
        dict: Structured diagnostic results including predictions, confidences,
              safety gate flags, warnings, and suggested agricultural actions.
    """
    model_species, model_potato, model_tomato = load_models()

    # Preprocessing with RGB enforcement
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("Unsupported image input type. Provide a file path or PIL.Image.Image instance.")

    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # --- STEP 1: Botanist (Species Classification) ---
    species_preds = model_species.predict(img_array, verbose=0)
    species_idx = int(np.argmax(species_preds))
    species_name = SPECIES_LABELS[species_idx]
    species_conf = float(species_preds[0][species_idx] * 100)

    # --- STEP 2: Specialist Routing ---
    specialist_model = model_potato if species_name == 'Potato' else model_tomato
    disease_preds = specialist_model.predict(img_array, verbose=0)
    disease_idx = int(np.argmax(disease_preds))
    disease_name = DISEASE_LABELS[disease_idx]
    disease_conf = float(disease_preds[0][disease_idx] * 100)

    # --- STEP 3: Confidence Safety Gate Evaluation ---
    is_low_confidence = (disease_name == 'Late Blight' and disease_conf < THRESHOLD)
    warning = None
    if is_low_confidence:
        warning = "Potential early lesions detected, but confidence is low. Possible mechanical damage (scratches)."

    # Action Logic
    if "Blight" in disease_name and disease_conf >= THRESHOLD:
        action = "Apply appropriate fungicide immediately."
    elif "Blight" in disease_name and disease_conf < THRESHOLD:
        action = "Monitor closely. Check for physical damage and do not apply chemicals yet."
    else:
        action = "Continue normal irrigation and monitoring."

    return {
        "species": species_name,
        "species_confidence": round(species_conf, 2),
        "disease": disease_name,
        "disease_confidence": round(disease_conf, 2),
        "diagnosis": f"{species_name} - {disease_name}",
        "is_low_confidence": is_low_confidence,
        "warning": warning,
        "action": action
    }


def predict_plant(img_path):
    """
    CLI wrapper around run_inference to provide formatted console output.
    """
    result = run_inference(img_path)

    print(f"\n BOTANIST: I am {result['species_confidence']:.2f}% sure this is a {result['species']}.")
    print("=" * 50)

    if result["is_low_confidence"]:
        print(f" DIAGNOSIS: {result['diagnosis']}")
        print(f" Confidence Score: {result['disease_confidence']:.2f}%")
        print("-" * 50)
        print(f" WARNING: {result['warning']}")
        print(" SUGGESTION: Check for physical damage (scratches) or re-scan in 24 hours.")
    else:
        print(f" FINAL DIAGNOSIS: {result['diagnosis']}")
        print(f" Confidence Score: {result['disease_confidence']:.2f}%")

    print("=" * 50)
    print(f" ACTION: {result['action']}")
    return result


if __name__ == "__main__":
    # Case-insensitive path search for WSL2 and Windows cross-compatibility
    sample_candidate = Path("test_images") / "sample_leaf.jpg"
    target_image = None

    if sample_candidate.exists():
        target_image = sample_candidate
    else:
        # Fallback search for .JPG or alternate casing
        for p in Path("test_images").glob("*"):
            if p.name.lower() == "sample_leaf.jpg":
                target_image = p
                break

    if target_image and target_image.exists():
        print(f"Running CLI inference on: {target_image}")
        predict_plant(target_image)
    else:
        print("\nPlease place an image named 'sample_leaf.jpg' or 'sample_leaf.JPG' in test_images/ to test.")
