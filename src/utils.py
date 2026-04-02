"""
Shared utilities for the PrivHAR-Bench pipeline.

Provides: config loading, determinism setup, bounding box helpers,
file I/O, logging, and group parsing.
"""

import hashlib
import json
import os
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(config_path):
    """Load and validate a YAML configuration file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    required = ["dataset_name", "source_dir", "output_dir", "classes", "tiers", "splits"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: '{key}'")

    if not Path(cfg["source_dir"]).exists():
        raise FileNotFoundError(
            f"source_dir does not exist: {cfg['source_dir']}\n"
            f"Edit {config_path} and set the correct path."
        )

    return cfg


def setup_determinism(seed):
    """
    Set all random seeds and enable deterministic execution.
    Must be called before any other imports that initialize random state.
    """
    import random
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    warnings.filterwarnings("ignore")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return 'cuda' if available, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Logging ──

_log_path = None


def init_logging(output_dir):
    """Set the log file path. Call once at startup."""
    global _log_path
    os.makedirs(output_dir, exist_ok=True)
    _log_path = Path(output_dir) / "pipeline.log"


def log(msg):
    """Print a timestamped message and append to the log file."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_path is not None:
        try:
            with open(_log_path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ── Bounding box helpers ──

def clamp_bbox(bbox, shape):
    """Clamp a [x1, y1, x2, y2] bounding box to frame dimensions."""
    h, w = shape[:2]
    return (
        max(0, int(bbox[0])),
        max(0, int(bbox[1])),
        min(w, int(bbox[2])),
        min(h, int(bbox[3])),
    )


def scale_bbox(bbox, orig_shape, target_resolution):
    """Scale a bounding box from orig_shape to target_resolution (w, h)."""
    oh, ow = orig_shape[:2]
    tw, th = target_resolution
    return [
        bbox[0] * tw / ow,
        bbox[1] * th / oh,
        bbox[2] * tw / ow,
        bbox[3] * th / oh,
    ]


# ── File I/O ──

def save_png(img, path):
    """Save an image as lossless PNG with minimal compression."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def has_valid_json(path, min_keys=1):
    """Check if a JSON file exists and contains meaningful content."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return len(data) >= min_keys
        if isinstance(data, list):
            return len(data) >= min_keys
        return False
    except (json.JSONDecodeError, OSError):
        return False


# ── Group parsing ──

def parse_group(video_stem, group_regex):
    """Extract the group/subject identifier from a video filename."""
    m = re.search(group_regex, video_stem)
    return int(m.group(1)) if m else -1


# ── Weight verification ──

def verify_weights(weights_path, expected_hash):
    """Verify YOLOv8 weight file by SHA-256 hash. Fatal error on mismatch."""
    if expected_hash is None:
        return
    sha = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected_hash:
        raise RuntimeError(
            f"Weight hash mismatch.\n"
            f"  Expected: {expected_hash}\n"
            f"  Got:      {actual}\n"
            f"  File:     {weights_path}"
        )


def get_class_names(cfg):
    """Extract the list of class names from config."""
    return [c["name"] for c in cfg["classes"]]
