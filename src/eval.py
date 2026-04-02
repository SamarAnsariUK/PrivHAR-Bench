"""
PrivHAR-Bench Evaluation Toolkit.

Accepts model predictions and computes all benchmark metrics:
  - Per-class and overall Top-1 accuracy
  - Cross-tier accuracy drop (delta_acc)
  - ROI-SSIM and ROI-PSNR
  - Face Detection Failure Rate
  - Composite Privacy-Utility (PU) score

Usage:
    python -m src.eval --predictions_dir results/ --dataset_dir PrivHAR-Bench_v1.0.0/

The predictions directory should contain one CSV per tier:
    Original.csv, Tier1_Blur.csv, etc.
Each CSV has columns: video_id, predicted_label
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_predictions(csv_path):
    """Load predictions from a CSV file. Returns dict {video_id: predicted_label}."""
    preds = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[row["video_id"]] = row["predicted_label"]
    return preds


def compute_accuracy(predictions, annotations, split="test"):
    """
    Compute per-class and overall Top-1 accuracy.

    Args:
        predictions: Dict {video_id: predicted_label}.
        annotations: List of annotation dicts from annotations.json.
        split: Which split to evaluate ("test" or "train").

    Returns:
        Dict with per_class accuracies and overall accuracy.
    """
    test_anns = [a for a in annotations if a["split"] == split]

    class_correct = {}
    class_total = {}

    for ann in test_anns:
        vid_id = ann["video_id"]
        true_label = ann["class"]

        if vid_id not in predictions:
            continue

        pred_label = predictions[vid_id]

        if true_label not in class_total:
            class_total[true_label] = 0
            class_correct[true_label] = 0

        class_total[true_label] += 1
        if pred_label == true_label:
            class_correct[true_label] += 1

    per_class = {}
    for cls in sorted(class_total.keys()):
        per_class[cls] = round(
            100.0 * class_correct[cls] / max(class_total[cls], 1), 2
        )

    total_correct = sum(class_correct.values())
    total_count = sum(class_total.values())
    overall = round(100.0 * total_correct / max(total_count, 1), 2)

    return {"per_class": per_class, "overall": overall}


def compute_delta_acc(original_acc, tier_acc):
    """Compute accuracy drop relative to Original tier."""
    return round(original_acc - tier_acc, 2)


def compute_pu_score(tier_acc, original_acc, tier_ssim):
    """
    Compute composite Privacy-Utility score.

    PU = (AccTier / AccOriginal) * (1 - SSIM_Tier)
    """
    if original_acc == 0:
        return 0.0
    return round((tier_acc / original_acc) * (1.0 - tier_ssim), 4)


def main():
    parser = argparse.ArgumentParser(description="PrivHAR-Bench Evaluation Toolkit")
    parser.add_argument(
        "--predictions_dir", required=True,
        help="Directory containing per-tier prediction CSV files",
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="PrivHAR-Bench dataset root directory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file for results (default: stdout)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    pred_dir = Path(args.predictions_dir)

    # Load annotations
    ann_path = dataset_dir / "annotations.json"
    if not ann_path.exists():
        print(f"ERROR: annotations.json not found at {ann_path}", file=sys.stderr)
        sys.exit(1)

    with open(ann_path) as f:
        annotations = json.load(f)

    # Load privacy metrics if available
    metrics_path = dataset_dir / "privacy_metrics.json"
    privacy_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            privacy_metrics = json.load(f)

    # Process each tier's predictions
    results = {}
    original_acc = None

    for csv_file in sorted(pred_dir.glob("*.csv")):
        tier_name = csv_file.stem
        predictions = load_predictions(csv_file)
        acc_result = compute_accuracy(predictions, annotations)

        tier_result = {
            "top1_accuracy": acc_result["overall"],
            "per_class_accuracy": acc_result["per_class"],
        }

        if tier_name == "Original":
            original_acc = acc_result["overall"]
        elif original_acc is not None:
            tier_result["delta_acc"] = compute_delta_acc(original_acc, acc_result["overall"])

        # Add privacy metrics and PU score if available
        if tier_name in privacy_metrics and original_acc is not None:
            pm = privacy_metrics[tier_name]
            tier_result["roi_ssim"] = pm["ssim_mean"]
            tier_result["roi_psnr"] = pm["psnr_mean"]
            tier_result["pu_score"] = compute_pu_score(
                acc_result["overall"], original_acc, pm["ssim_mean"]
            )

        results[tier_name] = tier_result

    # Output
    output = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
