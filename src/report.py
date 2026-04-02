"""
Summary report generation for PrivHAR-Bench.

Aggregates results from all pipeline phases and produces a human-readable
text report plus structured JSON output.
"""

import json
import time
from pathlib import Path

import numpy as np

from .utils import get_class_names, has_valid_json, log

TIER_DIRS = [
    "Original", "Tier1_Blur", "Tier2_Edge",
    "Tier3_AES_B4", "Tier3_AES_B8", "Tier3_AES_B16",
    "Tier3_AES_B4_NoBG", "Tier3_AES_B8_NoBG", "Tier3_AES_B16_NoBG",
]


def generate_report(cfg):
    """
    Phase 7: Generate a summary report of the complete pipeline run.

    Args:
        cfg: Loaded YAML configuration dict.
    """
    log("=== PHASE 7: SUMMARY REPORT ===")

    output_dir = Path(cfg["output_dir"])
    classes = get_class_names(cfg)
    report = []

    report.append("=" * 70)
    report.append("PRIVHAR-BENCH DATASET GENERATION REPORT")
    report.append("=" * 70)

    # Annotations summary
    ann_path = output_dir / "annotations.json"
    if ann_path.exists():
        ann = json.load(open(ann_path))
        n_train = sum(1 for a in ann if a["split"] == "train")
        n_test = sum(1 for a in ann if a["split"] == "test")
        report.append(f"\nDataset: {len(ann)} videos, {len(classes)} classes")
        report.append(f"Split: {n_train} train / {n_test} test (group-based)")

        det_rates = [a["detection_rate"] for a in ann]
        report.append(
            f"Detection rate: mean={np.mean(det_rates):.1%}, "
            f"min={np.min(det_rates):.1%}"
        )

        report.append("\nPer-class detection rates (median):")
        for cls in classes:
            rates = [a["detection_rate"] for a in ann if a["class"] == cls]
            if rates:
                report.append(f"  {cls:25s} {np.median(rates):.1%}  (n={len(rates)})")
            else:
                report.append(f"  {cls:25s} NO DATA")

    # Privacy metrics
    met_path = output_dir / "privacy_metrics.json"
    if met_path.exists() and has_valid_json(met_path, min_keys=1):
        met = json.load(open(met_path))
        report.append("\nPrivacy Metrics:")
        report.append(f"  {'Tier':<20} {'SSIM':>8} {'PSNR':>10}")
        report.append(f"  {'-' * 40}")
        for tier, vals in met.items():
            report.append(
                f"  {tier:<20} {vals['ssim_mean']:>8.4f} "
                f"{vals['psnr_mean']:>8.2f} dB"
            )
    else:
        report.append("\nPrivacy Metrics: NOT COMPUTED")

    # ArcFace results
    af_path = output_dir / "arcface_results.json"
    if af_path.exists() and has_valid_json(af_path, min_keys=1):
        af = json.load(open(af_path))
        if "summary" in af and af["summary"]:
            report.append("\nArcFace Block-Size Experiment:")
            report.append(f"  {'B':>5} {'Mean Sim':>10} {'<0.2':>8} {'N':>5}")
            for bs, s in sorted(af["summary"].items(), key=lambda x: int(x[0])):
                report.append(
                    f"  {bs:>5} {s['mean']:>10.3f} "
                    f"{s['pct_below_0.2']:>7.0f}% {s['n']:>5}"
                )
    else:
        report.append("\nArcFace: NOT COMPUTED")

    # Baseline training results
    tr_path = output_dir / "baseline_results" / "all_results.json"
    if tr_path.exists() and has_valid_json(tr_path, min_keys=1):
        tr = json.load(open(tr_path))
        if tr.get("config_a"):
            report.append("\nBaseline R3D-18 Results:")
            report.append("\n  Config A (train & eval same tier):")
            report.append(f"  {'Tier':<25} {'Acc %':>8}")
            report.append(f"  {'-' * 35}")
            for tier, acc in sorted(tr.get("config_a", {}).items()):
                report.append(f"  {tier:<25} {acc:>8.1f}")
        if tr.get("config_b"):
            report.append("\n  Config B (train Original, eval each tier):")
            report.append(f"  {'Tier':<25} {'Acc %':>8}")
            report.append(f"  {'-' * 35}")
            for tier, acc in sorted(tr.get("config_b", {}).items()):
                report.append(f"  {tier:<25} {acc:>8.1f}")
    else:
        report.append("\nBaseline Training: NOT COMPLETED")

    # Disk usage
    total_size = 0
    for td in TIER_DIRS:
        td_path = output_dir / td
        if td_path.exists():
            size = sum(f.stat().st_size for f in td_path.rglob("*") if f.is_file())
            total_size += size
    report.append(f"\nDisk usage: {total_size / (1024 ** 3):.1f} GB")

    report.append(f"\n{'=' * 70}")
    report.append(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"{'=' * 70}")

    report_text = "\n".join(report)
    print(report_text)

    with open(output_dir / "REPORT.txt", "w") as f:
        f.write(report_text)
    log(f"  Report saved to {output_dir}/REPORT.txt")
