"""
Person detection and pose estimation using YOLOv8n-Pose.

For each video frame, outputs:
  - Bounding box [x1, y1, x2, y2] for the largest detected person
  - Detection confidence
  - 17 COCO-format keypoints with per-joint confidence

When multiple persons are detected, the detection with the largest
bounding box area is selected as the primary subject. Detections
with confidence below the configured threshold are discarded.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .utils import get_class_names, log


def run_detection(cfg):
    """
    Phase 1: Run YOLOv8-Pose on all source videos and cache detections.

    Detections are saved as per-video JSON files in the detections directory.
    Already-detected videos are skipped (resume-safe).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log("  ultralytics not installed. Skipping detection.")
        log("  Install: pip install ultralytics")
        return

    log("=== PHASE 1: DETECTION ===")

    source_dir = Path(cfg["source_dir"])
    det_dir = Path(cfg["detections_dir"])
    weights = cfg["weights_path"]
    conf_thresh = cfg["confidence_threshold"]
    classes = get_class_names(cfg)

    model = YOLO(weights)
    total_new = 0

    for cls in classes:
        cls_dir = source_dir / cls
        if not cls_dir.exists():
            log(f"  WARNING: Class directory missing: {cls_dir}")
            continue

        videos = sorted(cls_dir.glob("*.avi"))
        for vpath in videos:
            det_json = det_dir / cls / vpath.stem / "detections.json"
            if det_json.exists():
                continue

            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames_data = []
            fidx = 0
            det_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                r = results[0]

                fd = {
                    "frame_idx": fidx,
                    "detected": False,
                    "bbox": None,
                    "confidence": None,
                    "keypoints": None,
                }

                if len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    mask = confs >= conf_thresh

                    if mask.any():
                        boxes_f = boxes[mask]
                        confs_f = confs[mask]
                        areas = (boxes_f[:, 2] - boxes_f[:, 0]) * (boxes_f[:, 3] - boxes_f[:, 1])
                        bi = np.argmax(areas)

                        fd["detected"] = True
                        fd["bbox"] = boxes_f[bi].tolist()
                        fd["confidence"] = float(confs_f[bi])
                        det_count += 1

                        if r.keypoints is not None:
                            kps = r.keypoints.data.cpu().numpy()[mask]
                            if len(kps) > bi:
                                fd["keypoints"] = kps[bi].tolist()

                frames_data.append(fd)
                fidx += 1

            cap.release()

            meta = {
                "video_file": vpath.name,
                "class": cls,
                "fps": fps,
                "width": w_orig,
                "height": h_orig,
                "total_frames": total_frames,
                "frames_with_detection": det_count,
                "detection_rate": round(det_count / max(total_frames, 1), 3),
                "frames": frames_data,
            }

            out_dir = det_dir / cls / vpath.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "detections.json", "w") as f:
                json.dump(meta, f)

            total_new += 1
            if total_new % 50 == 0:
                log(f"  Detected {total_new} new videos...")

    log(f"  Phase 1 complete. {total_new} new detections.")
