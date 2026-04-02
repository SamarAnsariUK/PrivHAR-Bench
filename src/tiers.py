"""
Tier generation for PrivHAR-Bench.

Transforms raw source videos into multi-tier lossless frame sequences
through a deterministic, reproducible process. For each source video,
generates 9 parallel variants: Original, Blur, Edge, AES B4/B8/B16,
and NoBG variants of each AES tier.

All tiers share the same per-frame ROI bounding box and mask, ensuring
spatial consistency across variants.
"""

import json
import os
from pathlib import Path

import cv2

from .tier1_blur import apply_blur
from .tier2_edge import apply_edge
from .tier3_scramble import apply_scramble
from .nobg import apply_nobg
from .utils import get_class_names, log, save_png, scale_bbox

TIER_DIRS = [
    "Original", "Tier1_Blur", "Tier2_Edge",
    "Tier3_AES_B4", "Tier3_AES_B8", "Tier3_AES_B16",
    "Tier3_AES_B4_NoBG", "Tier3_AES_B8_NoBG", "Tier3_AES_B16_NoBG",
]


def generate_tiers(cfg):
    """
    Phase 2: Generate all privacy tiers for every source video.

    Reads detection cache, loads source video frames, applies all
    transformations, and exports as lossless PNG. Resume-safe: skips
    videos whose Original frames already exist.

    Args:
        cfg: Loaded YAML configuration dict.
    """
    log("=== PHASE 2: TIER GENERATION ===")

    output_dir = Path(cfg["output_dir"])
    source_dir = Path(cfg["source_dir"])
    det_dir = Path(cfg["detections_dir"])
    resolution = tuple(cfg["resolution"])
    clip_length = cfg["clip_length"]
    classes = get_class_names(cfg)

    # Tier parameters
    blur_sigma = cfg["tiers"]["blur"]["sigma"]
    edge_low = cfg["tiers"]["edge"]["threshold_low"]
    edge_high = cfg["tiers"]["edge"]["threshold_high"]
    block_sizes = cfg["tiers"]["scramble"]["block_sizes"]
    key_hex = cfg["tiers"]["scramble"]["key_hex"]

    # Create output directories
    for td in TIER_DIRS:
        os.makedirs(output_dir / td, exist_ok=True)

    done = 0
    skipped = 0

    for cls in classes:
        cls_det = det_dir / cls
        if not cls_det.exists():
            log(f"  WARNING: No detections for {cls}")
            continue

        for vdir in sorted(cls_det.iterdir()):
            if not vdir.is_dir():
                continue

            jpath = vdir / "detections.json"
            if not jpath.exists():
                continue

            vid_stem = vdir.name

            # Skip if already generated (check Original canary frame)
            orig_check = output_dir / "Original" / cls / vid_stem / "frame_0000.png"
            if orig_check.exists():
                skipped += 1
                continue

            with open(jpath) as f:
                dets = json.load(f)

            src_avi = source_dir / cls / f"{vid_stem}.avi"
            if not src_avi.exists():
                continue

            # Read all frames
            cap = cv2.VideoCapture(str(src_avi))
            if not cap.isOpened():
                continue

            all_frames = []
            while True:
                ret, fr = cap.read()
                if not ret:
                    break
                all_frames.append(fr)
            cap.release()

            if not all_frames:
                continue

            # Center crop to clip_length
            if len(all_frames) > clip_length:
                s = (len(all_frames) - clip_length) // 2
                all_frames = all_frames[s:s + clip_length]
                start_idx = s
            else:
                start_idx = 0

            # Build frame index -> bbox mapping
            bbox_map = {}
            for fd in dets["frames"]:
                bbox_map[fd["frame_idx"]] = fd.get("bbox")

            # Generate all tiers for each frame
            for li, frame in enumerate(all_frames):
                gi = start_idx + li
                osh = frame.shape
                frame_r = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
                raw_bb = bbox_map.get(gi)
                bb = scale_bbox(raw_bb, osh, resolution) if raw_bb else None
                fn = f"frame_{li:04d}.png"

                # Original
                save_png(frame_r, output_dir / "Original" / cls / vid_stem / fn)

                # Tier 1: Blur
                save_png(
                    apply_blur(frame_r, bb, blur_sigma),
                    output_dir / "Tier1_Blur" / cls / vid_stem / fn,
                )

                # Tier 2: Edge
                save_png(
                    apply_edge(frame_r, bb, edge_low, edge_high),
                    output_dir / "Tier2_Edge" / cls / vid_stem / fn,
                )

                # Tier 3: AES block scrambling at each block size + NoBG variants
                for bs in block_sizes:
                    sc = apply_scramble(frame_r, bb, bs, key_hex, vid_stem, li)
                    save_png(sc, output_dir / f"Tier3_AES_B{bs}" / cls / vid_stem / fn)
                    save_png(
                        apply_nobg(sc, bb),
                        output_dir / f"Tier3_AES_B{bs}_NoBG" / cls / vid_stem / fn,
                    )

            done += 1
            if done % 100 == 0:
                log(f"  Generated tiers for {done} videos...")

    log(f"  Phase 2 complete. Generated: {done}, skipped (existing): {skipped}")
