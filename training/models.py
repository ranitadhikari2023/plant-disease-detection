# training/models.py
"""
Hierarchical CNN Model Architectures for Plant Pathology.
Implements MobileNetV2 Transfer Learning with Injected CBAM Attention.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
import keras

# ==============================================================================
# 1. Keras 3 Serializable CBAM Module
# ==============================================================================
@keras.saving.register_keras_serializable(package="plant_pathology")
class ChannelAttention(layers.Layer):
    """
    Channel Attention Module (CAM).
    Compresses spatial dimensions via GAP and GMP, passes through a shared MLP,
    and scales channels with a sigmoid activation function.
    """
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduction = max(1, channels // self.ratio)
        self.shared_dense_one = layers.Dense(
            reduction,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros',
            name="shared_dense_1"
        )
        self.shared_dense_two = layers.Dense(
            channels,
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros',
            name="shared_dense_2"
        )
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True, name="gap")
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True, name="gmp")
        super().build(input_shape)

    def call(self, inputs):
        avg_out = self.shared_dense_two(self.shared_dense_one(self.avg_pool(inputs)))
        max_out = self.shared_dense_two(self.shared_dense_one(self.max_pool(inputs)))
        scale = tf.keras.activations.sigmoid(avg_out + max_out)
        return inputs * scale

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


@keras.saving.register_keras_serializable(package="plant_pathology")
class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module (SAM).
    Compresses channels via average and max pooling, concatenates,
    and applies a 7x7 convolution with sigmoid to focus on lesion geography.
    """
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            use_bias=False,
            name="conv_spatial"
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = layers.concatenate([avg_pool, max_pool], axis=-1)
        scale = self.conv(concat)
        return inputs * scale

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


@keras.saving.register_keras_serializable(package="plant_pathology")
class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module.
    Sequentially combines Channel Attention and Spatial Attention.
    """
    def __init__(self, ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(ratio=ratio, name="channel_attention")
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size, name="spatial_attention")

    def build(self, input_shape):
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio,
            'kernel_size': self.kernel_size
        })
        return config


# ==============================================================================
# 2. Backbone Loading Utility
# ==============================================================================
def get_mobilenet_backbone(image_size=(224, 224)):
    """
    Retrieves MobileNetV2 pretrained feature extractor.
    First attempts local extraction from models/species_expert.keras to prevent network dependency.
    Falls back to tf.keras.applications.MobileNetV2 if available.
    """
    local_ref = Path("models/species_expert.keras")
    if local_ref.exists():
        # Backward-compat hook for Dense deserialization if needed
        orig_dense_from_config = keras.layers.Dense.from_config
        @classmethod
        def patched_dense_from_config(cls, config):
            config.pop('quantization_config', None)
            return orig_dense_from_config.__func__(cls, config)
        keras.layers.Dense.from_config = patched_dense_from_config

        m_ref = tf.keras.models.load_model(
            str(local_ref),
            custom_objects={'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input},
            compile=False
        )
        base = m_ref.get_layer("mobilenetv2_1.00_224")
        # Return a cloned or isolated functional instance with fresh weights
        return base
    else:
        return tf.keras.applications.MobileNetV2(
            input_shape=(image_size[0], image_size[1], 3),
            include_top=False,
            weights="imagenet"
        )


# ==============================================================================
# 3. Model Builders
# ==============================================================================
def build_species_expert(image_size=(224, 224)):
    """
    Tier 1: Botanist Species Classifier (Potato vs Tomato).
    Input (224, 224, 3) -> Preprocess -> MobileNetV2 -> GAP -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(2, Softmax).
    """
    base_model = get_mobilenet_backbone(image_size)
    base_model.trainable = False

    inputs = layers.Input(shape=(image_size[0], image_size[1], 3), name="input_tensor")
    x = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="preprocess_input"
    )(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(2, activation="softmax", name="species_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Species_Expert_Botanist_v2")
    return model, base_model


def build_crop_doctor_cbam(model_name="Potato_Doctor_CBAM_v2", num_classes=3, image_size=(224, 224)):
    """
    Tier 2: Specialist Disease Classifier with CBAM Attention.
    Input (224, 224, 3) -> Preprocess -> MobileNetV2 -> CBAM -> GAP -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(3, Softmax).
    """
    base_model = get_mobilenet_backbone(image_size)
    base_model.trainable = False

    inputs = layers.Input(shape=(image_size[0], image_size[1], 3), name="input_tensor")
    x = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="preprocess_input"
    )(inputs)

    # Base feature maps: (None, 7, 7, 1280)
    features = base_model(x, training=False)

    # Injected CBAM Attention
    attended_features = CBAM(ratio=16, kernel_size=7, name="cbam_attention")(features)

    # Head projection
    x = layers.GlobalAveragePooling2D(name="gap")(attended_features)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model, base_model
