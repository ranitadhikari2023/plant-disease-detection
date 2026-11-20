# training/losses.py
"""
Loss functions and specimen-level class weighting for hierarchical plant pathology.
Addresses severe biological class imbalance (e.g. Potato Healthy specimen minority).
Uses standard Python csv module (zero dependency on pandas).
"""

import csv
from collections import defaultdict
import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="plant_pathology")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for multi-class classification with sparse integer targets.
    Down-weights easily classified leaves (p_t close to 1) and focuses gradients
    on hard ambiguous lesions.
    
    Formula:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, from_logits=False, name="sparse_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])
        num_classes = tf.shape(y_pred)[-1]

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        ce = -tf.math.log(p_t)

        if self.alpha is not None:
            alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
            alpha_t = tf.gather(alpha_tensor, y_true)
            loss = alpha_t * modulating_factor * ce
        else:
            loss = modulating_factor * ce

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        })
        return config


def compute_specimen_class_weights(manifest_path, crop_filter=None, model_type="potato_expert"):
    """
    Computes balanced class weights derived strictly from unique biological specimens
    (source_group_id), rather than raw offline-augmented image counts.

    Formula:
        weight[c] = total_specimens / (num_classes * specimens[c])

    Returns:
        tuple: (class_weight_dict, specimen_counts_dict)
    """
    class_groups = defaultdict(set)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if crop_filter and r["crop"] != crop_filter:
                continue
            if "assigned_split" in r and r["assigned_split"] != "train":
                continue

            if model_type == "species_expert":
                key = r["crop"]
            else:
                key = r["disease_class"]
            class_groups[key].add(r["source_group_id"])

    if model_type == "species_expert":
        label_map = {"potato": 0, "tomato": 1}
    else:
        label_map = {"Early_blight": 0, "Late_blight": 1, "healthy": 2}

    total_specimens = sum(len(gset) for gset in class_groups.values())
    num_classes = len(label_map)

    class_weights = {}
    specimen_counts = {}

    for name, idx in label_map.items():
        cnt = len(class_groups[name]) if name in class_groups else 1
        specimen_counts[name] = cnt
        class_weights[idx] = float(total_specimens / (num_classes * cnt))

    return class_weights, specimen_counts
