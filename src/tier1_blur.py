"""
Tier 1: Spatial Obfuscation (Gaussian Blur).

Applies a Gaussian blur with configurable sigma to all pixels within the
detected ROI. Preserves gross spatial structure (body outline, posture,
limb position) while destroying fine-grained features (facial landmarks,
skin texture, clothing detail). Pixels outside the ROI are unchanged.
"""

import cv2
import numpy as np

from .utils import clamp_bbox


def apply_blur(frame, bbox, sigma):
    """
    Apply Gaussian blur to the ROI region.

    Args:
        frame: Input BGR frame (H, W, 3).
        bbox: Bounding box [x1, y1, x2, y2] or None.
        sigma: Gaussian kernel standard deviation.

    Returns:
        Blurred frame (copy; input is not modified).
    """
    if bbox is None:
        return frame.copy()

    out = frame.copy()
    x1, y1, x2, y2 = clamp_bbox(bbox, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return out

    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1

    out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), sigma)
    return out
