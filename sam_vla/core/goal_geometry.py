"""Pinhole backprojection of the (image-space, first-frame-only) goal bbox into a
static world-frame goal position, so rollouts can log a per-frame distance-to-goal.

GoalSpec only ever carries a normalized bbox in the frame it was resolved from —
there is no separate goal actor in the sim to query a world position from. This
module anchors one by backprojecting the bbox center through the depth map of
that same frame, using the pinhole model and the yaw convention pose_integrator
already commits to (heading = (cos_yaw, sin_yaw) in the x-z plane).
"""

from __future__ import annotations

import math

import numpy as np

from sam_vla.core.types import GoalSpec, Observation, Pose

GoalPosition = tuple[float, float, float]

# Semantic ids painted onto goal/obstacle mask meshes (see disc_mesh + the
# env's register_object_mask), so the semantic sensor can tell them apart.
MESH_GOAL_ID = 1
MESH_OBST_ID = 2


def bbox_to_world(
    obs: Observation, bbox_norm: tuple[float, float, float, float], hfov_deg: float
) -> GoalPosition | None:
    """Backproject a normalized bbox into a world-frame (x, y, z) point.

    Bearing/elevation come from the bbox's center pixel; range comes from the
    MEDIAN depth over the bbox's interior, mirroring rollout_navdp_policy's
    bbox_to_body robustness -- a single center pixel can land on a depth
    discontinuity (a rock's silhouette edge, a gap between the object and the
    background behind it) and seed a badly wrong range.

    Returns None if depth is unavailable, or no pixel in the bbox has valid
    depth (e.g. the bbox is entirely a sky/void hit).
    """
    if obs.depth is None:
        return None

    height, width = obs.depth.shape[:2]
    x0, y0, x1, y1 = bbox_norm
    fx0, fx1 = x0 * width, x1 * width
    fy0, fy1 = y0 * height, y1 * height
    ix0, ix1 = sorted((min(max(int(fx0), 0), width - 1), min(max(int(fx1), 0), width - 1)))
    iy0, iy1 = sorted((min(max(int(fy0), 0), height - 1), min(max(int(fy1), 0), height - 1)))

    patch = np.asarray(obs.depth)[iy0:iy1 + 1, ix0:ix1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid))

    # Bearing/elevation from the bbox's continuous center (not the rounded patch
    # bounds above) -- mirrors bbox_to_body's `u = 0.5 * (x1 + x2)`.
    px = 0.5 * (fx0 + fx1)
    py = 0.5 * (fy0 + fy1)

    # Depth sensor reports z-depth (perpendicular to the image plane), which is
    # exactly what a pinhole model expects as z_cam.
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    x_cam = (px - width / 2.0) * depth_m / fx
    y_cam = (py - height / 2.0) * depth_m / fy
    z_cam = depth_m

    pose = obs.pose
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    world_x = pose.x + z_cam * cos_yaw + x_cam * sin_yaw
    world_z = pose.z + z_cam * sin_yaw - x_cam * cos_yaw
    world_y = pose.y - y_cam

    return (world_x, world_y, world_z)


def backproject_goal_position(
    obs: Observation, goal_spec: GoalSpec, hfov_deg: float
) -> GoalPosition | None:
    """Backproject the goal bbox into a world-frame (x, y, z) point."""
    return bbox_to_world(obs, goal_spec.goal_bbox_norm, hfov_deg)


def disc_mesh(
    center: GoalPosition, radius: float, segments: int = 16, lift: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Flat circular fan of triangles in the x-z plane, centered at `center`
    and lifted slightly above it so the render-only mesh doesn't z-fight
    whatever surface it's marking. Output shape (verts, faces) mirrors
    rollout_navdp_policy's depth_patch_mesh, so it can be handed to the same
    _save_obj / register_semantic_mesh path to become a goal/obstacle mask.
    """
    cx, cy, cz = center
    cy = float(cy) + float(lift)
    angles = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False)
    ring = [(cx + radius * math.cos(a), cy, cz + radius * math.sin(a)) for a in angles]
    verts = np.asarray([(cx, cy, cz)] + ring, dtype=np.float64)
    n = len(ring)
    faces = np.asarray([(0, i + 1, (i + 1) % n + 1) for i in range(n)], dtype=np.int64)
    return verts, faces


def distance_to_goal(pose: Pose, goal_position: GoalPosition) -> float:
    gx, gy, gz = goal_position
    return math.sqrt((pose.x - gx) ** 2 + (pose.y - gy) ** 2 + (pose.z - gz) ** 2)


def goal_pixel_center(
    goal_bbox_norm: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int]:
    """Pixel coords of the (first-frame-resolved) goal bbox's center, scaled to
    the given frame's resolution."""
    x0, y0, x1, y1 = goal_bbox_norm
    return (int(0.5 * (x0 + x1) * width), int(0.5 * (y0 + y1) * height))


if __name__ == "__main__":
    import numpy as np

    depth = np.full((4, 4), 5.0, dtype=np.float32)
    obs = Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        frame_idx=0,
    )
    goal_spec = GoalSpec(
        goal_bbox_norm=(0.4, 0.4, 0.6, 0.6),
        obstacle_bboxes_norm=[],
        instruction_text="Navigate to the rock target.",
    )

    goal_position = backproject_goal_position(obs, goal_spec, hfov_deg=90.0)
    print("goal_position:", goal_position)
    assert goal_position is not None

    dist = distance_to_goal(obs.pose, goal_position)
    print("distance_to_goal:", dist)
    assert math.isclose(dist, 5.0, rel_tol=1e-3)
    print("OK")
