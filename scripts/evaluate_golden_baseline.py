# scripts/evaluate_golden_baseline.py
import os
import sys
import json
import csv
import time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from inference.predictor import HierarchicalPredictorV2
from inference.preprocessing import preprocess_image
import onnxruntime as ort

def main():
    print("Evaluating baseline over test manifest and selecting curated golden set...")
    with open("audit/test_specimen_manifest.csv", "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    predictor = HierarchicalPredictorV2()

    # Load ONNX sessions
    sess_sp = ort.InferenceSession("models/onnx/v2/species_expert_v2.onnx", providers=["CPUExecutionProvider"])
    sess_pot = ort.InferenceSession("models/onnx/v2/potato_expert_v2.onnx", providers=["CPUExecutionProvider"])
    sess_tom = ort.InferenceSession("models/onnx/v2/tomato_expert_v2.onnx", providers=["CPUExecutionProvider"])

    with open("training/class_mappings.json", "r") as f:
        class_maps = json.load(f)

    # Curate specimens across all 6 classes:
    class_buckets = {
        ("potato", "Early_blight"): [],
        ("potato", "Late_blight"): [],
        ("potato", "healthy"): [],
        ("tomato", "Early_blight"): [],
        ("tomato", "Late_blight"): [],
        ("tomato", "healthy"): []
    }

    for row in manifest_rows:
        k = (row["crop"], row["disease_class"])
        if k in class_buckets:
            class_buckets[k].append(row)

    golden_records = []
    # 8 from each of the 6 classes -> 48
    for k, rows in class_buckets.items():
        step = max(1, len(rows) // 8)
        chosen = rows[::step][:8]
        for r in chosen:
            golden_records.append(r)

    print(f"Selected {len(golden_records)} specimens from manifest")

    results = []

    def run_single(img_path, ground_truth=None):
        tensor, _ = preprocess_image(img_path)
        # Keras run
        k_res = predictor.predict(img_path)

        # ONNX run
        sp_logits = sess_sp.run(["species_output"], {"input_tensor": tensor})[0][0]
        sp_idx = int(np.argmax(sp_logits))
        sp_name = class_maps["species_expert"]["index_to_label"][str(sp_idx)]
        sp_conf = float(sp_logits[sp_idx])

        if round(sp_conf, 4) < 0.90:
            onnx_status = "species_uncertain"
            onnx_disease = None
            onnx_disease_conf = None
        else:
            sess_doc = sess_pot if sp_name == "Potato" else sess_tom
            map_key = "potato_expert" if sp_name == "Potato" else "tomato_expert"
            dis_logits = sess_doc.run(["disease_output"], {"input_tensor": tensor})[0][0]
            dis_idx = int(np.argmax(dis_logits))
            dis_name = class_maps[map_key]["index_to_label"][str(dis_idx)]
            dis_conf = float(dis_logits[dis_idx])
            onnx_disease = dis_name
            onnx_disease_conf = dis_conf

            if dis_name == "Late Blight":
                onnx_status = "late_blight_uncertain" if round(dis_conf, 4) < 0.90 else "confident"
            elif round(dis_conf, 4) < 0.70:
                onnx_status = "disease_uncertain"
            else:
                onnx_status = "confident"

        return {
            "path": img_path,
            "ground_truth": ground_truth,
            "keras": {
                "species": k_res["plant_species"],
                "species_confidence": k_res["species_confidence"],
                "disease": k_res["disease"],
                "disease_confidence": k_res["disease_confidence"],
                "safety_status": k_res["safety_status"],
                "species_probabilities": k_res["species_probabilities"],
                "probabilities": k_res["probabilities"]
            },
            "onnx": {
                "species": sp_name,
                "species_confidence": round(sp_conf, 4),
                "disease": onnx_disease,
                "disease_confidence": round(onnx_disease_conf, 4) if onnx_disease_conf else None,
                "safety_status": onnx_status
            }
        }

    # Add sample_leaf.JPG
    results.append(run_single("test_images/sample_leaf.JPG", {"crop": "tomato", "disease_class": "Late_blight", "source_group_id": "SAMPLE_LEAF"}))

    for r in golden_records:
        res = run_single(r["relative_image_path"], {
            "crop": r["crop"],
            "disease_class": r["disease_class"],
            "source_group_id": r.get("source_group_id", "")
        })
        results.append(res)

    print(f"Total golden cases evaluated: {len(results)}")

    flips = 0
    for idx, r in enumerate(results):
        k = r["keras"]
        o = r["onnx"]
        if k["species"] != o["species"] or k["disease"] != o["disease"] or k["safety_status"] != o["safety_status"]:
            print(f"Flip at {r['path']}: Keras={k['safety_status']} ONNX={o['safety_status']}")
            flips += 1

    print(f"Gate/prediction flips between Keras and ONNX across golden set: {flips}")

    os.makedirs("tests/fixtures", exist_ok=True)
    with open("tests/fixtures/golden_test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved tests/fixtures/golden_test_dataset.json")

if __name__ == "__main__":
    main()
