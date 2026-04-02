"""
Tier 2: Structural Abstraction (Canny Edge Extraction).

Applies Canny edge detection to the ROI. All non-edge pixels within the ROI
are set to black. All pixels outside the ROI are also set to black. This
removes all texture and color information, retaining only the structural
contours of the human body.
"""

import cv2
import numpy as np

from .utils import clamp_bbox


def apply_edge(frame, bbox, threshold_low, threshold_high):
    """
    Apply Canny edge detection to the ROI.

    Args:
        frame: Input BGR frame (H, W, 3).
        bbox: Bounding box [x1, y1, x2, y2] or None.
        threshold_low: Canny lower threshold.
        threshold_high: Canny upper threshold.

    Returns:
        Edge frame: white edges on black background within ROI,
        black everywhere else.
    """
    out = np.zeros_like(frame)
    if bbox is None:
        return out

    x1, y1, x2, y2 = clamp_bbox(bbox, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return out

    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold_low, threshold_high)
    out[y1:y2, x1:x2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return out
