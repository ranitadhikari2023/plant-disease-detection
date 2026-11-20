# training/evaluate_v2_system.py
"""
Comprehensive Held-Out Test Evaluation & Hierarchical Pipeline Validation for V2 Models.
Evaluates:
  1. Species Expert V2 (Potato vs Tomato) on all test specimens.
  2. Potato Doctor CBAM V2 on held-out potato specimens.
  3. Tomato Doctor CBAM V2 on held-out tomato specimens.
  4. End-to-End Hierarchical Pipeline (Gating, Routing, Diagnosis) across all test specimens.
Generates performance metrics, confusion matrices, and training curves.
"""

import os
import sys
import csv
import json
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from training.models import CBAM, ChannelAttention, SpatialAttention

def load_v2_models():
    print("📦 Loading V2 Models with Custom Deserializers...")
    custom_objs = {
        'CBAM': CBAM,
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }
    
    species_path = "models/v2/species_expert_v2.keras"
    potato_path = "models/v2/potato_expert_v2.keras"
    tomato_path = "models/v2/tomato_expert_v2.keras"
    
    species_model = tf.keras.models.load_model(species_path, custom_objects=custom_objs, compile=False, safe_mode=False)
    potato_model = tf.keras.models.load_model(potato_path, custom_objects=custom_objs, compile=False, safe_mode=False)
    tomato_model = tf.keras.models.load_model(tomato_path, custom_objects=custom_objs, compile=False, safe_mode=False)
    
    print(f"  ✓ Loaded Species Expert: {species_path}")
    print(f"  ✓ Loaded Potato Doctor CBAM: {potato_path}")
    print(f"  ✓ Loaded Tomato Doctor CBAM: {tomato_path}")
    return species_model, potato_model, tomato_model

def load_test_manifest(manifest_path="audit/test_specimen_manifest.csv"):
    records = []
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    print(f"📋 Loaded {len(records)} test specimens from {manifest_path}")
    return records

def preprocess_image(img_path):
    # Load and resize image to (224, 224) RGB
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

def compute_classification_metrics(y_true, y_pred, class_names):
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    total = len(y_true)
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    
    per_class = {}
    f1s = []
    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        
        per_class[cname] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support
        }
    
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

def evaluate_models():
    species_model, potato_model, tomato_model = load_v2_models()
    test_records = load_test_manifest()
    
    with open("training/class_mappings.json", "r") as f:
        mappings = json.load(f)
    
    species_names = ["potato", "tomato"]
    disease_names = ["Early_blight", "Late_blight", "healthy"]
    
    # 1. Evaluate Species Expert
    print("\n" + "="*60)
    print("1. EVALUATING BOTANIST SPECIES EXPERT V2 ON ALL TEST SPECIMENS")
    print("="*60)
    
    species_true = []
    species_pred = []
    species_confs = []
    
    # Pre-load batches for efficiency
    batch_size = 32
    all_imgs = []
    for r in test_records:
        img_arr = np.array(Image.open(r["relative_image_path"]).convert("RGB").resize((224, 224)), dtype=np.float32)
        all_imgs.append(img_arr)
    all_imgs = np.array(all_imgs)
    
    sp_map = mappings["species_expert"]["label_to_index"]
    pot_map = mappings["potato_expert"]["label_to_index"]
    tom_map = mappings["tomato_expert"]["label_to_index"]

    species_logits = species_model.predict(all_imgs, batch_size=batch_size, verbose=0)
    for idx, r in enumerate(test_records):
        t_label = sp_map[r["crop"]]
        probs = species_logits[idx]
        p_label = int(np.argmax(probs))
        conf = float(probs[p_label])
        
        species_true.append(t_label)
        species_pred.append(p_label)
        species_confs.append(conf)
        
    species_metrics = compute_classification_metrics(species_true, species_pred, species_names)
    print(f"Accuracy: {species_metrics['accuracy'] * 100:.2f}% | Macro F1: {species_metrics['macro_f1']:.4f}")
    print(f"Per-class breakdown: {json.dumps(species_metrics['per_class'], indent=2)}")
    print(f"Confusion Matrix (Rows: True, Cols: Pred [Potato, Tomato]):\n{np.array(species_metrics['confusion_matrix'])}")
    
    # 2. Evaluate Potato Doctor CBAM on Potato Test Specimens
    print("\n" + "="*60)
    print("2. EVALUATING POTATO DOCTOR CBAM V2 ON HELD-OUT POTATO SPECIMENS")
    print("="*60)
    potato_indices = [i for i, r in enumerate(test_records) if r["crop"] == "potato"]
    potato_imgs = all_imgs[potato_indices]
    potato_records = [test_records[i] for i in potato_indices]
    
    potato_logits = potato_model.predict(potato_imgs, batch_size=batch_size, verbose=0)
    potato_true = [pot_map[r["disease_class"]] for r in potato_records]
    potato_pred = [int(np.argmax(probs)) for probs in potato_logits]
    
    potato_metrics = compute_classification_metrics(potato_true, potato_pred, disease_names)
    print(f"Accuracy: {potato_metrics['accuracy'] * 100:.2f}% | Macro F1: {potato_metrics['macro_f1']:.4f}")
    print(f"Per-class breakdown: {json.dumps(potato_metrics['per_class'], indent=2)}")
    print(f"Confusion Matrix (Rows: True, Cols: Pred [Early, Late, Healthy]):\n{np.array(potato_metrics['confusion_matrix'])}")
    
    # 3. Evaluate Tomato Doctor CBAM on Tomato Test Specimens
    print("\n" + "="*60)
    print("3. EVALUATING TOMATO DOCTOR CBAM V2 ON HELD-OUT TOMATO SPECIMENS")
    print("="*60)
    tomato_indices = [i for i, r in enumerate(test_records) if r["crop"] == "tomato"]
    tomato_imgs = all_imgs[tomato_indices]
    tomato_records = [test_records[i] for i in tomato_indices]
    
    tomato_logits = tomato_model.predict(tomato_imgs, batch_size=batch_size, verbose=0)
    tomato_true = [tom_map[r["disease_class"]] for r in tomato_records]
    tomato_pred = [int(np.argmax(probs)) for probs in tomato_logits]
    
    tomato_metrics = compute_classification_metrics(tomato_true, tomato_pred, disease_names)
    print(f"Accuracy: {tomato_metrics['accuracy'] * 100:.2f}% | Macro F1: {tomato_metrics['macro_f1']:.4f}")
    print(f"Per-class breakdown: {json.dumps(tomato_metrics['per_class'], indent=2)}")
    print(f"Confusion Matrix (Rows: True, Cols: Pred [Early, Late, Healthy]):\n{np.array(tomato_metrics['confusion_matrix'])}")

    # 4. End-to-End Hierarchical Pipeline Evaluation
    print("\n" + "="*60)
    print("4. END-TO-END HIERARCHICAL PIPELINE VALIDATION (CONFIDENCE GATE = 0.90)")
    print("="*60)
    
    CONFIDENCE_GATE = 0.90
    total_eval = len(test_records)
    gated_count = 0
    routing_errors = 0
    end_to_end_correct = 0
    ungated_correct = 0
    
    for idx, r in enumerate(test_records):
        true_crop = r["crop"]
        true_disease = r["disease_class"]
        
        # Step 1: Botanist
        sp_probs = species_logits[idx]
        sp_class_idx = int(np.argmax(sp_probs))
        sp_conf = float(sp_probs[sp_class_idx])
        pred_crop = species_names[sp_class_idx]
        
        if sp_conf < CONFIDENCE_GATE:
            gated_count += 1
            continue
            
        if pred_crop != true_crop:
            routing_errors += 1
            continue
            
        # Step 2: Route to Specialist
        if pred_crop == "potato":
            spec_probs = potato_model.predict(all_imgs[idx:idx+1], verbose=0)[0]
        else:
            spec_probs = tomato_model.predict(all_imgs[idx:idx+1], verbose=0)[0]
            
        dis_class_idx = int(np.argmax(spec_probs))
        pred_disease = disease_names[dis_class_idx]
        
        if pred_crop == true_crop and pred_disease == true_disease:
            end_to_end_correct += 1
            ungated_correct += 1
    
    ungated_total = total_eval - gated_count
    ungated_accuracy = (ungated_correct / ungated_total) if ungated_total > 0 else 0.0
    overall_system_accuracy = (end_to_end_correct / total_eval)
    
    pipeline_results = {
        "total_test_specimens": total_eval,
        "confidence_gate_threshold": CONFIDENCE_GATE,
        "gated_abstained_count": gated_count,
        "gated_abstention_rate_pct": round((gated_count / total_eval) * 100, 2),
        "routing_errors": routing_errors,
        "ungated_specimens_evaluated": ungated_total,
        "ungated_accuracy_pct": round(ungated_accuracy * 100, 2),
        "end_to_end_system_accuracy_pct": round(overall_system_accuracy * 100, 2)
    }
    
    print(f"Total Held-Out Specimens:       {total_eval}")
    print(f"Safety Gated (<90% Species):    {gated_count} ({pipeline_results['gated_abstention_rate_pct']}%)")
    print(f"Routing Errors (Wrong Crop):    {routing_errors}")
    print(f"Ungated Diagnostics Accuracy:  {pipeline_results['ungated_accuracy_pct']}%")
    print(f"End-to-End System Accuracy:    {pipeline_results['end_to_end_system_accuracy_pct']}%")
    
    # Save combined report
    full_report = {
        "species_expert_metrics": species_metrics,
        "potato_doctor_cbam_metrics": potato_metrics,
        "tomato_doctor_cbam_metrics": tomato_metrics,
        "hierarchical_pipeline_metrics": pipeline_results
    }
    
    os.makedirs("audit/artifacts", exist_ok=True)
    with open("audit/artifacts/v2_evaluation_report.json", "w") as f:
        json.dump(full_report, f, indent=2)
    print("\n✅ Saved evaluation report to: audit/artifacts/v2_evaluation_report.json")
    
    return full_report

def plot_training_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    os.makedirs("audit/artifacts", exist_ok=True)
    log_files = {
        "Species Expert V2": "training/logs/species_expert_v2_training.csv",
        "Potato Doctor CBAM V2": "training/logs/potato_expert_v2_training.csv",
        "Tomato Doctor CBAM V2": "training/logs/tomato_expert_v2_training.csv"
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("V2 Models Training & Validation Performance Curves", fontsize=16, fontweight='bold')
    
    for row_idx, (name, path) in enumerate(log_files.items()):
        if not os.path.exists(path):
            continue
        
        epochs = []
        loss, val_loss = [], []
        acc, val_acc = [], []
        
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                epochs.append(int(r["epoch"]) + 1)
                loss.append(float(r["loss"]))
                val_loss.append(float(r["val_loss"]))
                acc.append(float(r["accuracy"]))
                val_acc.append(float(r["val_accuracy"]))
        
        # Loss Plot
        ax_loss = axes[row_idx, 0]
        ax_loss.plot(epochs, loss, 'o-', color='#1f77b4', label='Train Loss', linewidth=2)
        ax_loss.plot(epochs, val_loss, 's--', color='#d62728', label='Val Loss', linewidth=2)
        ax_loss.set_title(f"{name} - Loss Evolution", fontweight='bold')
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle=':', alpha=0.6)
        ax_loss.legend(loc='upper right')
        
        # Accuracy Plot
        ax_acc = axes[row_idx, 1]
        ax_acc.plot(epochs, acc, 'o-', color='#2ca02c', label='Train Accuracy', linewidth=2)
        ax_acc.plot(epochs, val_acc, 's--', color='#ff7f0e', label='Val Accuracy', linewidth=2)
        ax_acc.set_title(f"{name} - Accuracy Evolution", fontweight='bold')
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim([0.4, 1.02])
        ax_acc.grid(True, linestyle=':', alpha=0.6)
        ax_acc.legend(loc='lower right')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    curve_path = "audit/artifacts/v2_training_curves.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"📊 Training curves saved to: {curve_path}")

if __name__ == "__main__":
    evaluate_models()
    plot_training_curves()
