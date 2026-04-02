"""
NoBG (No Background) variant generation.

Zeroes all pixels outside the detected ROI bounding box, leaving only the
transformed human region visible. Used as a context bias control: any model
evaluated on NoBG variants must derive its classification signal exclusively
from the transformed ROI, not from environmental cues.
"""

import numpy as np

from .utils import clamp_bbox


def apply_nobg(frame, bbox):
    """
    Remove background by zeroing all pixels outside the ROI.

    Args:
        frame: Input BGR frame (H, W, 3). Typically a tier-transformed frame.
        bbox: Bounding box [x1, y1, x2, y2] or None.

    Returns:
        Frame with only the ROI region preserved; all other pixels are black.
    """
    out = np.zeros_like(frame)
    if bbox is None:
        return out

    x1, y1, x2, y2 = clamp_bbox(bbox, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return out

    out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out
