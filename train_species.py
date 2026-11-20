# train_species.py
"""
Tier 1: Botanist Species Classifier (Potato vs Tomato).
Trains on the Canonical Specimen Manifest with zero data leakage.
"""

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from training.models import build_species_expert
from training.train_utils import run_two_phase_training

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 Botanist Species Classifier (Potato vs Tomato)")
    parser.add_argument("--epochs_phase1", type=int, default=10, help="Epochs for Phase A head warmup")
    parser.add_argument("--epochs_phase2", type=int, default=10, help="Epochs for Phase B fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for Phase A")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch lightweight smoke test")
    parser.add_argument("--output_path", type=str, default="models/v2/species_expert_v2.keras")
    parser.add_argument("--log_path", type=str, default="training/logs/species_expert_v2_training.csv")
    args = parser.parse_args()

    print("\n🌱 INITIALIZING BOTANIST SPECIES EXPERT V2...")
    model, base_model = build_species_expert()

    run_two_phase_training(
        model=model,
        base_model=base_model,
        model_type="species_expert",
        crop_filter=None,
        model_save_path=args.output_path,
        log_csv_path=args.log_path,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size_phase1=args.batch_size,
        batch_size_phase2=max(4, args.batch_size // 2),
        use_focal_loss=False,
        smoke_test=args.smoke_test
    )

if __name__ == "__main__":
    main()
