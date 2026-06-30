"""Shared geometric primitives. The lowest layer: no plant, safety or domain.

Both the SafetyArbiter (keep-out) and the domain detectors (dodge clearance)
reason about distances between keypoints and strike lines. Keeping the primitive
here, in core, lets every layer depend on it without creating a safety <-> domain
coupling.
"""

from __future__ import annotations

import numpy as np

from robot_twin.core.types import FloatArray


def point_to_segment_distance(p0: FloatArray, p1: FloatArray, points: FloatArray) -> FloatArray:
    """Shortest distances from each of ``points`` to the segment ``p0``-``p1``.

    Vectorised over ``points`` (shape ``(K, 3)``) because callers check many
    keypoints in one shot on the hot path. Returns shape ``(K,)``.

    Args:
        p0: Segment start, shape ``(3,)``.
        p1: Segment end, shape ``(3,)``.
        points: Query points, shape ``(K, 3)``.

    Returns:
        Distance from each query point to the closest point on the segment.
    """
    seg = p1 - p0  # (3,)
    seg_len2 = float(seg @ seg)
    if seg_len2 == 0.0:
        # Degenerate segment: distance to the single point p0.
        return np.linalg.norm(points - p0, axis=1)
    # Project each point onto the infinite line, clamp the parameter to [0, 1]
    # so the closest point stays on the finite segment.
    t = np.clip((points - p0) @ seg / seg_len2, 0.0, 1.0)  # (K,)
    projection = p0 + t[:, None] * seg  # (K, 3)
    return np.linalg.norm(points - projection, axis=1)
