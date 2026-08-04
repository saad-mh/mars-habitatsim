"""Belief-uncertainty-driven "ghost mask": a translucent circle drawn on a
frame at the belief's projected pixel, radius proportional to uncertainty.
Pure numpy, mirrors navdp/navdp/extensions/ghost_geometry.py's shape
(body-frame bearing/range -> pixel -> filled-circle mask) but built against
sam_vla.core.goal_geometry.intrinsics_from_hfov and BeliefGoalTracker's
[forward, left] body-frame convention, so it composes directly with
.bearing()/.distance() instead of needing a separate world-frame projection
step. Ported as a pattern, not an import -- navdp/ stays read-only per
CLAUDE.md.

Non-destructive and policy-blind: callers draw this onto a copy of the frame
handed to a VLM/visualization, never onto anything a policy consumes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from sam_vla.core.goal_geometry import intrinsics_from_hfov

GHOST_COLOR = np.array([0, 255, 0], dtype=np.float32)


def uncertainty_to_radius_px(
    uncertainty: float, min_px: float, max_px: float, scale: float
) -> float:
    """Maps a scalar uncertainty to an on-screen circle radius in pixels,
    clamped to [min_px, max_px] -- same clamp pattern kb_teleop_vl.py already
    uses for obstacle overlays (OVERLAY_MIN_PIXEL_RADIUS/OVERLAY_MAX_PIXEL_RADIUS).
    """
    return float(
        np.clip(float(uncertainty) * float(scale), float(min_px), float(max_px))
    )


def project_body_point_to_pixel(
    forward: float, left: float, hfov_deg: float, h: int, w: int
) -> Optional[Tuple[float, float]]:
    """Body-frame [forward, left] point -> pixel (u, v), or None if behind the
    camera or outside the frame's horizontal extent.

    BeliefGoalTracker's belief state is 2D (no elevation component), so v is
    fixed at the frame's vertical center (cy) -- the belief only ever carries
    bearing/range, not height.
    """
    if forward <= 1e-3:
        return None
    intr = intrinsics_from_hfov(h, w, hfov_deg)
    u = intr["cx"] - float(left) * intr["fx"] / float(forward)
    if u < 0.0 or u > float(w - 1):
        return None
    return u, intr["cy"]


def draw_ghost_mask(
    rgb: np.ndarray,
    u: float,
    v: float,
    radius_px: float,
    color: np.ndarray = GHOST_COLOR,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blends a translucent filled circle onto rgb (uint8 HWC) at pixel
    (u, v) with the given radius. Non-destructive -- returns a copy, modeled
    on kb_teleop_vl.py's overlay_obstacles()."""
    rgb = np.asarray(rgb)
    annotated = rgb.astype(np.float32)
    yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
    annotated[mask] = annotated[mask] * (1.0 - alpha) + np.asarray(
        color, dtype=np.float32
    ) * alpha
    return annotated.astype(np.uint8)
