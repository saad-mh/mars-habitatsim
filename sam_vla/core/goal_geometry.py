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

from sam_vla.core.types import GoalSpec, Observation, Pose

GoalPosition = tuple[float, float, float]


def backproject_goal_position(
    obs: Observation, goal_spec: GoalSpec, hfov_deg: float
) -> GoalPosition | None:
    """Backproject the goal bbox center into a world-frame (x, y, z) point.

    Returns None if depth is unavailable, or the sampled depth at the goal
    pixel is invalid (e.g. a sky/void hit).
    """
    if obs.depth is None:
        return None

    height, width = obs.depth.shape[:2]
    x0, y0, x1, y1 = goal_spec.goal_bbox_norm
    px = min(max(int((x0 + x1) / 2.0 * width), 0), width - 1)
    py = min(max(int((y0 + y1) / 2.0 * height), 0), height - 1)

    depth_m = float(obs.depth[py, px])
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        return None

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


def distance_to_goal(pose: Pose, goal_position: GoalPosition) -> float:
    gx, gy, gz = goal_position
    return math.sqrt((pose.x - gx) ** 2 + (pose.y - gy) ** 2 + (pose.z - gz) ** 2)


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
