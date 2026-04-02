"""
ArcFace block-size experiment for PrivHAR-Bench.

Evaluates the effect of block scrambling on face identity by:
  1. Sampling frames with detectable faces from Original videos.
  2. Applying block scrambling at multiple block sizes.
  3. Attempting face detection on each scrambled frame.
  4. Computing cosine similarity between original and scrambled embeddings.

Requires: insightface, onnxruntime (or onnxruntime-gpu)
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .tier3_scramble import apply_scramble
from .utils import get_class_names, has_valid_json, log


def run_arcface_experiment(cfg):
    """
    Phase 5: ArcFace face detection and identity verification experiment.

    Args:
        cfg: Loaded YAML configuration dict.

    Returns:
        Dict with per-face results and summary statistics, or None if
        insightface is not installed.
    """
    log("=== PHASE 5: ARCFACE BLOCK-SIZE EXPERIMENT ===")

    output_dir = Path(cfg["output_dir"])
    arcface_file = output_dir / "arcface_results.json"

    # Resume check
    if has_valid_json(arcface_file, min_keys=2):
        log("  Already computed with valid data. Skipping.")
        return json.load(open(arcface_file))
    elif arcface_file.exists():
        log("  Found stale/empty arcface_results.json; recomputing.")
        arcface_file.unlink()

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        log("  insightface not installed. Skipping.")
        log("  Install: pip install insightface onnxruntime-gpu")
        return None

    arcface_cfg = cfg.get("arcface", {})
    model_name = arcface_cfg.get("model", "buffalo_l")
    max_faces = arcface_cfg.get("max_faces", 500)
    test_blocks = arcface_cfg.get("test_block_sizes", [2, 4, 8, 16, 32])

    source_dir = Path(cfg["source_dir"])
    det_dir = Path(cfg["detections_dir"])
    classes = get_class_names(cfg)
    key_hex = cfg["tiers"]["scramble"]["key_hex"]

    app = FaceAnalysis(
        name=model_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(160, 160))

    results = []
    count = 0

    for cls in classes:
        if count >= max_faces:
            break

        cls_det = det_dir / cls
        if not cls_det.exists():
            continue

        for vdir in sorted(cls_det.iterdir()):
            if count >= max_faces:
                break
            if not vdir.is_dir():
                continue

            jpath = vdir / "detections.json"
            if not jpath.exists():
                continue

            with open(jpath) as f:
                d = json.load(f)

            vid_stem = vdir.name
            bframes = [fd for fd in d["frames"] if fd.get("bbox")]
            if not bframes:
                continue

            # Sample the middle detected frame
            fd = bframes[len(bframes) // 2]

            avi = source_dir / cls / f"{vid_stem}.avi"
            if not avi.exists():
                continue

            cap = cv2.VideoCapture(str(avi))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fd["frame_idx"])
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue

            faces_orig = app.get(frame)
            if not faces_orig:
                continue

            emb_orig = max(faces_orig, key=lambda f: f.det_score).embedding

            row = {"video_id": vid_stem, "class": cls, "similarities": {}}
            for bs in test_blocks:
                sc = apply_scramble(frame, fd["bbox"], bs, key_hex, vid_stem, fd["frame_idx"])
                faces_sc = app.get(sc)

                if not faces_sc:
                    row["similarities"][str(bs)] = None
                    continue

                emb_sc = max(faces_sc, key=lambda f: f.det_score).embedding
                dot = float(np.dot(emb_orig.flatten(), emb_sc.flatten()))
                norm = float(np.linalg.norm(emb_orig) * np.linalg.norm(emb_sc))
                sim = dot / norm if norm > 1e-10 else 0.0
                row["similarities"][str(bs)] = round(sim, 6)

            results.append(row)
            count += 1
            if count % 50 == 0:
                log(f"  Processed {count} faces...")

    # Compute summary statistics
    summary = {}
    for bs in test_blocks:
        vals = [
            r["similarities"][str(bs)]
            for r in results
            if r["similarities"].get(str(bs)) is not None
        ]
        if vals:
            summary[str(bs)] = {
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "pct_below_0.2": round(float(np.mean(np.array(vals) < 0.2)) * 100, 1),
                "n": len(vals),
            }
            log(f"  B={bs}: mean={summary[str(bs)]['mean']:.3f}  "
                f"<0.2={summary[str(bs)]['pct_below_0.2']:.0f}%  n={len(vals)}")

    out_data = {"results": results, "summary": summary}
    with open(arcface_file, "w") as f:
        json.dump(out_data, f, indent=2)

    # Generate plot if matplotlib is available
    _save_plot(output_dir, results, test_blocks)

    return out_data


def _save_plot(output_dir, results, test_blocks):
    """Save the ArcFace cosine similarity boxplot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not results:
        return

    data = []
    labels = []
    for bs in test_blocks:
        vals = [
            r["similarities"][str(bs)]
            for r in results
            if r["similarities"].get(str(bs)) is not None
        ]
        data.append(vals)
        labels.append(f"B={bs}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.axhline(y=0.2, color="red", linestyle="--", label="Threshold=0.2")
    ax.set_ylabel("ArcFace Cosine Similarity")
    ax.set_xlabel("Block Size")
    ax.set_title("Identity Verification vs Block Scrambling")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "arcface_block_plot.png"), dpi=200)
    plt.close()
    log(f"  Plot saved: {output_dir}/arcface_block_plot.png")
