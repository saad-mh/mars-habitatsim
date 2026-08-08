"""Pure SE(2) helpers for turning a world-frame (x, z) point into the
body-frame [forward, left] point BeliefGoalTracker/NavdpUpstreamPolicy want,
and for picking random/home world points to drive to.

Convention (must match sam_vla.core.pose_integrator.integrate_mars and
sam_vla.core.goal_geometry.bbox_to_world, both of which this module derives
from independently rather than importing -- these are three-line pure
functions, not worth adding a cross-module dependency for): world heading at
yaw is (cos(yaw), sin(yaw)) in the (x, z) plane; "left" is the
counter-clockwise perpendicular of heading, i.e. (-sin(yaw), cos(yaw)).
Solving bbox_to_world's own (world_x, world_z) = pose + fwd*heading +
right*rightward for (fwd, right) given a world point yields the formulas
below (right = -left). Cross-checked numerically in this module's __main__.
"""

from __future__ import annotations

import math
from typing import Optional

from sam_vla.core.types import Pose


def body_frame_goal(pose: Pose, world_xz: tuple[float, float]) -> tuple[float, float]:
    """World-frame (x, z) point -> body-frame (forward, left) from `pose`."""
    gx, gz = world_xz
    dx = gx - pose.x
    dz = gz - pose.z
    cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
    forward = dx * cos_yaw + dz * sin_yaw
    left = dz * cos_yaw - dx * sin_yaw
    # print(f"body pose -> {forward}, {left} (from world {gx}, {gz})")
    return forward, left


def clamp_to_yard(x: float, z: float, limit: float) -> tuple[float, float]:
    return (max(-limit, min(limit, x)), max(-limit, min(limit, z)))


def random_ahead_point(
    pose: Pose,
    rng,
    bearing_range_deg: float = 60.0,
    dist_range: tuple[float, float] = (4.0, 8.0),
    world_limit: Optional[float] = None,
) -> tuple[float, float]:
    """A random world (x, z) point `dist_range` meters ahead of `pose`, within
    +/-bearing_range_deg of its current heading -- mirrors the "random goal"
    preset a human tester would want to poke the policy with, without needing
    any detection/VLM in the loop. Clamped into [-world_limit, world_limit]
    on both axes if given."""
    bearing = math.radians(float(rng.uniform(-bearing_range_deg, bearing_range_deg)))
    dist = float(rng.uniform(*dist_range))
    theta = pose.yaw + bearing
    gx = pose.x + dist * math.cos(theta)
    gz = pose.z + dist * math.sin(theta)
    if world_limit is not None:
        gx, gz = clamp_to_yard(gx, gz, world_limit)
    return gx, gz


if __name__ == "__main__":
    import numpy as np

    # Cross-check against goal_geometry.bbox_to_world's own forward derivation:
    # picking a (fwd, right) pair, building the world point its formula would
    # produce, then confirming body_frame_goal recovers the same (fwd, left=-right).
    for yaw_deg in (0.0, 37.0, 90.0, -125.0):
        yaw = math.radians(yaw_deg)
        pose = Pose(x=1.5, y=0.0, z=-2.5, yaw=yaw)
        fwd_in, right_in = 4.0, 1.2
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        world_x = pose.x + fwd_in * cos_yaw + right_in * sin_yaw
        world_z = pose.z + fwd_in * sin_yaw - right_in * cos_yaw
        fwd_out, left_out = body_frame_goal(pose, (world_x, world_z))
        assert math.isclose(fwd_out, fwd_in, abs_tol=1e-6), (yaw_deg, fwd_out, fwd_in)
        assert math.isclose(left_out, -right_in, abs_tol=1e-6), (
            yaw_deg,
            left_out,
            -right_in,
        )
    print("OK: body_frame_goal matches goal_geometry.bbox_to_world's convention")

    rng = np.random.default_rng(0)
    p = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    pt = random_ahead_point(p, rng, world_limit=20.0)
    print("random_ahead_point:", pt)
