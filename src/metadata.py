"""
Metadata generation for PrivHAR-Bench.

Produces:
  - annotations.json: per-video metadata (class, split, detection rate, etc.)
  - train_split.txt / test_split.txt: fixed lists of video IDs
  - Estimated_Poses/: per-video pose keypoints extracted from detection cache
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .utils import get_class_names, log, parse_group


def generate_metadata(cfg):
    """
    Phase 3: Generate annotations, splits, and estimated pose files.

    Args:
        cfg: Loaded YAML configuration dict.
    """
    log("=== PHASE 3: METADATA ===")

    output_dir = Path(cfg["output_dir"])
    det_dir = Path(cfg["detections_dir"])
    classes = get_class_names(cfg)
    clip_length = cfg["clip_length"]
    group_regex = cfg["splits"]["group_regex"]
    train_groups = cfg["splits"]["train_groups"]

    annotations = []
    vid_counter = 0

    for cls in classes:
        cls_det = det_dir / cls
        if not cls_det.exists():
            continue

        for vdir in sorted(cls_det.iterdir()):
            if not vdir.is_dir():
                continue

            jpath = vdir / "detections.json"
            if not jpath.exists():
                continue

            with open(jpath) as f:
                d = json.load(f)

            vid_stem = vdir.name
            grp = parse_group(vid_stem, group_regex)
            split = "train" if grp in train_groups else "test"

            bboxes = [fd["bbox"] for fd in d["frames"] if fd.get("bbox")]
            if bboxes:
                mean_bb = [
                    round(float(np.mean([b[i] for b in bboxes])), 1) for i in range(4)
                ]
            else:
                mean_bb = None

            vid_counter += 1
            annotations.append({
                "video_id": f"{vid_counter:05d}",
                "source_file": d["video_file"],
                "class": cls,
                "group": grp,
                "split": split,
                "source_fps": d.get("fps", 25),
                "total_frames": d["total_frames"],
                "clip_frames": clip_length,
                "detection_rate": d["detection_rate"],
                "roi_bbox_mean": mean_bb,
            })

    # Write annotations
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    # Write split files
    train_ids = [
        a["source_file"].replace(".avi", "")
        for a in annotations if a["split"] == "train"
    ]
    test_ids = [
        a["source_file"].replace(".avi", "")
        for a in annotations if a["split"] == "test"
    ]

    with open(output_dir / "train_split.txt", "w") as f:
        f.write("\n".join(train_ids))
    with open(output_dir / "test_split.txt", "w") as f:
        f.write("\n".join(test_ids))

    # Save estimated poses
    _save_poses(cfg, det_dir, output_dir, classes, clip_length)

    log(f"  annotations.json: {len(annotations)} entries")
    log(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Log per-split class distribution
    for split_name, ids in [("Train", train_ids), ("Test", test_ids)]:
        by_cls = defaultdict(int)
        for vid in ids:
            for c in classes:
                if vid.startswith(f"v_{c}_"):
                    by_cls[c] += 1
                    break
        log(f"  {split_name}: " + ", ".join(f"{c}:{n}" for c, n in sorted(by_cls.items())))

    return annotations


def _save_poses(cfg, det_dir, output_dir, classes, clip_length):
    """Extract and save per-video pose keypoints from detection cache."""
    poses_dir = output_dir / "Estimated_Poses"
    pose_count = 0

    for cls in classes:
        cls_det = det_dir / cls
        if not cls_det.exists():
            continue

        for vdir in sorted(cls_det.iterdir()):
            if not vdir.is_dir():
                continue

            jpath = vdir / "detections.json"
            if not jpath.exists():
                continue

            with open(jpath) as f:
                d = json.load(f)

            vid_stem = vdir.name
            total = d["total_frames"]
            s = (total - clip_length) // 2 if total > clip_length else 0

            clip_frames = []
            for fd in d["frames"]:
                fi = fd["frame_idx"]
                if fi < s or fi >= s + clip_length:
                    continue
                clip_frames.append({
                    "local_idx": fi - s,
                    "keypoints": fd.get("keypoints"),
                    "bbox": fd.get("bbox"),
                    "confidence": fd.get("confidence"),
                })

            pose_out = poses_dir / cls / vid_stem
            pose_out.mkdir(parents=True, exist_ok=True)
            with open(pose_out / "pose.json", "w") as f:
                json.dump({"frames": clip_frames}, f)
            pose_count += 1

    log(f"  Estimated poses: {pose_count} videos")
