# train_potato_doctor.py
"""
Tier 2: Potato Specialist (Early Blight vs Late Blight vs Healthy).
Trains MobileNetV2 with injected CBAM attention and specimen class balancing.
"""

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from training.models import build_crop_doctor_cbam, CBAM, ChannelAttention, SpatialAttention
from training.train_utils import run_two_phase_training

def main():
    parser = argparse.ArgumentParser(description="Train Tier 2 Potato Specialist with CBAM Attention")
    parser.add_argument("--epochs_phase1", type=int, default=10, help="Epochs for Phase A head + CBAM warmup")
    parser.add_argument("--epochs_phase2", type=int, default=10, help="Epochs for Phase B fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for Phase A")
    parser.add_argument("--focal_loss", action="store_true", help="Use Alpha-Weighted Focal Loss instead of Crossentropy")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch lightweight smoke test")
    parser.add_argument("--output_path", type=str, default="models/v2/potato_expert_v2.keras")
    parser.add_argument("--log_path", type=str, default="training/logs/potato_expert_v2_training.csv")
    args = parser.parse_args()

    print("\n🥔 INITIALIZING POTATO DOCTOR CBAM V2...")
    model, base_model = build_crop_doctor_cbam(
        model_name="Potato_Doctor_CBAM_v2",
        num_classes=3
    )

    run_two_phase_training(
        model=model,
        base_model=base_model,
        model_type="potato_expert",
        crop_filter="potato",
        model_save_path=args.output_path,
        log_csv_path=args.log_path,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size_phase1=args.batch_size,
        batch_size_phase2=max(4, args.batch_size // 2),
        use_focal_loss=args.focal_loss,
        smoke_test=args.smoke_test
    )

if __name__ == "__main__":
    main()
