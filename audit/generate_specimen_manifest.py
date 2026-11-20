# audit/generate_specimen_manifest.py
"""
Generates the Canonical Specimen-Level Manifest (one representative per biological source leaf).
Prevents offline-augmented variations from biasing gradient updates or acting as pseudoreplicates.
"""

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

DATASET_MANIFEST_PATH = Path("audit/dataset_manifest.csv")
AUDIT_ROOT = Path("audit")
ARTIFACTS_ROOT = AUDIT_ROOT / "artifacts"

AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

def generate_specimen_manifests():
    print("=" * 80)
    print("GENERATING CANONICAL SPECIMEN-LEVEL MANIFESTS (1 IMAGE PER SOURCE LEAF)")
    print("=" * 80)

    with open(DATASET_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Total rows in master dataset manifest: {len(all_rows)}")

    # Group by source_group_id
    grouped_records = defaultdict(list)
    for r in all_rows:
        grouped_records[r["source_group_id"]].append(r)

    total_groups = len(grouped_records)
    print(f"Total unique biological source leaf groups: {total_groups}")

    specimens = []
    reason_counts = Counter()
    class_group_counts = Counter()
    split_group_counts = Counter()
    class_split_distribution = defaultdict(Counter)

    for gid in sorted(grouped_records.keys()):
        recs = grouped_records[gid]
        crop = recs[0]["crop"]
        disease = recs[0]["disease_class"]
        split = recs[0]["assigned_split"]

        class_group_counts[(crop, disease)] += 1
        split_group_counts[split] += 1
        class_split_distribution[(crop, disease)][split] += 1

        # 1. Look for original unaugmented images
        origs = [r for r in recs if r["original_or_augmented"] == "original"]
        if origs:
            origs.sort(key=lambda x: x["relative_image_path"])
            selected = origs[0]
            if len(origs) == 1:
                reason = "canonical_original_specimen"
            else:
                reason = "canonical_original_specimen_merged_pair_rep"
        else:
            # Standalone cluster with omitted parent (prefer gblr_1: 256x256, mild blur, no crop)
            gblrs = [r for r in recs if "_gblr_1" in r["relative_image_path"]]
            if gblrs:
                selected = gblrs[0]
                reason = "standalone_cluster_full_res_gblr1_rep"
            else:
                recs_sorted = sorted(recs, key=lambda x: (-int(x["image_width"]), x["relative_image_path"]))
                selected = recs_sorted[0]
                reason = "standalone_cluster_max_res_fallback_rep"

        reason_counts[reason] += 1

        specimen_row = {
            "source_group_id": gid,
            "relative_image_path": selected["relative_image_path"],
            "crop": selected["crop"],
            "disease_class": selected["disease_class"],
            "original_or_augmented": selected["original_or_augmented"],
            "image_width": selected["image_width"],
            "image_height": selected["image_height"],
            "image_format": selected["image_format"],
            "sha256_hash": selected["sha256_hash"],
            "perceptual_hash": selected["perceptual_hash"],
            "assigned_split": selected["assigned_split"],
            "selection_reason": reason,
            "total_variations_in_group": len(recs)
        }
        specimens.append(specimen_row)

    print(f"\nSelection Breakdown ({len(specimens)} specimens total):")
    for r, c in reason_counts.items():
        print(f"  - {r:45}: {c}")

    headers = [
        "source_group_id",
        "relative_image_path",
        "crop",
        "disease_class",
        "original_or_augmented",
        "image_width",
        "image_height",
        "image_format",
        "sha256_hash",
        "perceptual_hash",
        "assigned_split",
        "selection_reason",
        "total_variations_in_group"
    ]

    def export_csv(filepath, rows):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    # Master specimen manifest
    specimen_manifest_path = AUDIT_ROOT / "specimen_manifest.csv"
    export_csv(specimen_manifest_path, specimens)

    # Split-specific specimen manifests
    train_specimens = [s for s in specimens if s["assigned_split"] == "train"]
    val_specimens = [s for s in specimens if s["assigned_split"] == "val"]
    test_specimens = [s for s in specimens if s["assigned_split"] == "test"]

    export_csv(AUDIT_ROOT / "train_specimen_manifest.csv", train_specimens)
    export_csv(AUDIT_ROOT / "validation_specimen_manifest.csv", val_specimens)
    export_csv(AUDIT_ROOT / "test_specimen_manifest.csv", test_specimens)

    # Summary JSON
    summary_data = {
        "total_specimens": len(specimens),
        "split_counts": {
            "train": len(train_specimens),
            "val": len(val_specimens),
            "test": len(test_specimens)
        },
        "selection_reasons": dict(reason_counts),
        "class_distribution": {f"{c}_{d}": count for (c, d), count in class_group_counts.items()},
        "class_split_distribution": {
            f"{c}_{d}": dict(splits) for (c, d), splits in class_split_distribution.items()
        }
    }

    summary_json_path = AUDIT_ROOT / "specimen_manifest_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✅ Specimen manifests successfully exported:")
    print(f"   - {specimen_manifest_path} ({len(specimens)} records)")
    print(f"   - {AUDIT_ROOT / 'train_specimen_manifest.csv'} ({len(train_specimens)} records)")
    print(f"   - {AUDIT_ROOT / 'validation_specimen_manifest.csv'} ({len(val_specimens)} records)")
    print(f"   - {AUDIT_ROOT / 'test_specimen_manifest.csv'} ({len(test_specimens)} records)")
    print(f"   - Summary: {summary_json_path}")

if __name__ == "__main__":
    generate_specimen_manifests()
