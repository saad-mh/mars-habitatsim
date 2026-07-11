"""Low-level habitat_sim helpers: sensor specs, agent pose, RGB-D extraction,
and render-only semantic meshes (used to paint goal/obstacle masks that the
semantic sensor picks up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import quaternion
import habitat_sim

from sam_vla.core.types import Pose


def make_sensor(uuid: str, sensor_type, height: int, width: int, hfov_deg: float):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [int(height), int(width)]
    spec.position = [0.0, 0.0, 0.0]
    spec.hfov = float(hfov_deg)
    return spec


def set_agent_pose(agent, x: float, y: float, z: float, yaw: float) -> None:
    state = agent.get_state()
    state.position = np.asarray([x, y, z], dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
    agent.set_state(state)


def rgb_depth(obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(obs["rgb"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    depth = np.asarray(obs["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return rgb.astype(np.uint8), depth.astype(np.float32)


def save_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def distance_to_goal(pose: Pose, goal_position: Tuple[float, float, float]) -> float:
    """Planar (x-z) distance from `pose` to `goal_position`, ignoring elevation --
    mirrors rollout_navdp_policy's `goal_dist_now` (terrain height makes the y
    component noisy/uninformative for a ground rover)."""
    gx, _gy, gz = goal_position
    return float(np.linalg.norm(np.asarray([pose.x - gx, pose.z - gz], dtype=np.float32)))


def register_semantic_mesh(sim, mesh_path: str, semantic_id: int):
    """Add a render-only (kinematic, non-collidable) mesh carrying a semantic
    id, so the semantic sensor renders it as a distinct mask."""
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    template = otm.create_new_template(mesh_path)
    template.render_asset_handle = mesh_path
    template.collision_asset_handle = mesh_path
    template.is_collidable = False
    template_id = otm.register_template(template, f"sem_{semantic_id}_{Path(mesh_path).name}")
    obj = rom.add_object_by_template_handle(otm.get_template_handle_by_id(template_id))
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.collidable = False
    obj.semantic_id = int(semantic_id)
    return obj
