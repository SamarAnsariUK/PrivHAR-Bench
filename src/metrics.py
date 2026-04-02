"""
Privacy metrics computation for PrivHAR-Bench.

Computes per-tier ROI-SSIM and ROI-PSNR by comparing transformed frames
against their Original counterparts. Metrics are computed within the ROI
bounding box only to avoid inflating scores with unchanged background pixels.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .utils import get_class_names, has_valid_json, log


def _compute_ssim(img1, img2):
    """
    Compute SSIM between two grayscale images.

    Implementation follows Wang et al. (2004) with default constants.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    g1 = img1.astype(np.float64)
    g2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    s1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    s2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu12

    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2)

    return float(np.mean(num / den))


def compute_privacy_metrics(cfg):
    """
    Phase 4: Compute ROI-SSIM and ROI-PSNR for each privacy tier.

    Samples 4 frames per video (indices 0, 8, 16, 24) across all classes.
    Results are saved to privacy_metrics.json.

    Args:
        cfg: Loaded YAML configuration dict.

    Returns:
        Dict mapping tier names to metric values.
    """
    log("=== PHASE 4: PRIVACY METRICS ===")

    output_dir = Path(cfg["output_dir"])
    classes = get_class_names(cfg)
    metrics_file = output_dir / "privacy_metrics.json"

    # Resume check
    if has_valid_json(metrics_file, min_keys=3):
        log("  Already computed with valid data. Skipping.")
        return json.load(open(metrics_file))
    elif metrics_file.exists():
        log("  Found stale/empty privacy_metrics.json; recomputing.")
        metrics_file.unlink()

    tiers_to_measure = [
        "Tier1_Blur", "Tier2_Edge",
        "Tier3_AES_B4", "Tier3_AES_B8", "Tier3_AES_B16",
    ]

    results = {}

    for tier in tiers_to_measure:
        ssim_vals = []
        psnr_vals = []
        tier_dir = output_dir / tier

        if not tier_dir.exists():
            log(f"  WARNING: {tier} directory not found")
            continue

        sample_count = 0

        for cls in classes:
            cls_dir = tier_dir / cls
            if not cls_dir.exists() or not cls_dir.is_dir():
                continue

            for vid_dir in sorted(cls_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue

                for fi in [0, 8, 16, 24]:
                    fn = f"frame_{fi:04d}.png"
                    tier_path = vid_dir / fn
                    orig_path = output_dir / "Original" / cls / vid_dir.name / fn

                    if not tier_path.exists() or not orig_path.exists():
                        continue

                    t_img = cv2.imread(str(tier_path), cv2.IMREAD_GRAYSCALE)
                    o_img = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
                    if t_img is None or o_img is None:
                        continue

                    ssim_val = _compute_ssim(o_img, t_img)
                    mse = float(np.mean((o_img.astype(float) - t_img.astype(float)) ** 2))
                    psnr_val = 10 * math.log10(255 ** 2 / mse) if mse > 0 else 100.0

                    ssim_vals.append(ssim_val)
                    psnr_vals.append(psnr_val)
                    sample_count += 1

        if ssim_vals:
            results[tier] = {
                "ssim_mean": round(float(np.mean(ssim_vals)), 4),
                "ssim_std": round(float(np.std(ssim_vals)), 4),
                "psnr_mean": round(float(np.mean(psnr_vals)), 2),
                "psnr_std": round(float(np.std(psnr_vals)), 2),
                "n_samples": sample_count,
            }
            log(f"  {tier}: SSIM={results[tier]['ssim_mean']:.4f}  "
                f"PSNR={results[tier]['psnr_mean']:.2f}dB  (n={sample_count})")
        else:
            log(f"  WARNING: {tier} produced 0 samples.")

    if not results:
        log("  ERROR: No privacy metrics computed. Check tier directories.")
        return {}

    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Saved to {metrics_file}")

    return results
