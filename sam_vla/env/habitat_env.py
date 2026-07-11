from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import habitat_sim
import quaternion
from habitat_sim.agent import AgentConfiguration

from sam_vla.core.lifecycle import ServiceRegistry
from sam_vla.core.types import Observation, Pose

from rollout_navdp_policy import (
    SIZE_X,
    SIZE_Y,
    SIZE_Z,
    SceneMappedTerrain,
    TerrainHeight,
    make_sensor,
    rgb_depth,
    set_agent_pose,
)

RGB_HEIGHT = 480
RGB_WIDTH = 640
HFOV_DEG = 90.0
DEPTH_MAX_RANGE_M = 10.0

# Mirrors rollout_navdp_policy.py's --clearance / --pose-terrain-radius defaults.
SPAWN_CLEARANCE_M = 1.4
SPAWN_TERRAIN_RADIUS_M = 0.8


class MarsHabitatEnv:
    def __init__(
        self,
        scene_path: str,
        heightmap_path: str,
        services: list = None,
        start_x: float = 0.0,
        start_z: float = 8.0,
        start_yaw: float = 0.0,
        randomize_spawn: bool = False,
        spawn_clearance: float = SPAWN_CLEARANCE_M,
        spawn_terrain_radius: float = SPAWN_TERRAIN_RADIUS_M,
    ):
        self._scene_path = Path(scene_path)
        self._heightmap_path = Path(heightmap_path)
        self._registry = ServiceRegistry()
        for service in services or []:
            self._registry.register(service)
        self._sim = None
        self._agent = None
        self._terrain = None
        self._start_x = start_x
        self._start_z = start_z
        self._start_yaw = start_yaw
        self._randomize_spawn = randomize_spawn
        self._spawn_clearance = spawn_clearance
        self._spawn_terrain_radius = spawn_terrain_radius

    def __enter__(self) -> "MarsHabitatEnv":
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self._scene_path.expanduser().resolve())
        sim_cfg.enable_physics = False

        rgb_spec = make_sensor("rgb", habitat_sim.SensorType.COLOR, RGB_HEIGHT, RGB_WIDTH, HFOV_DEG)
        depth_spec = make_sensor("depth", habitat_sim.SensorType.DEPTH, RGB_HEIGHT, RGB_WIDTH, HFOV_DEG)
        # rollout_navdp_policy.make_sensor doesn't set a far clip; cap depth range here
        # to match the old pipeline's 10m sensor spec.
        depth_spec.far = DEPTH_MAX_RANGE_M

        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self._agent = self._sim.initialize_agent(0)

        raw_terrain = TerrainHeight(
            mode="heightmap",
            heightmap=self._heightmap_path.expanduser().resolve(),
            obj=None,
            flat_y=0.0,
            size_x=SIZE_X,
            size_z=SIZE_Z,
            size_y=SIZE_Y,
            flip_x=False,
            flip_z=True,
            swap_xz=False,
        )
        self._terrain = SceneMappedTerrain(raw_terrain, flip_x=False, flip_z=True, swap_xz=False)

        self._registry.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._registry.stop_all()
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        return False

    def get_observation(self, frame_idx: int) -> Observation:
        obs = self._sim.get_sensor_observations()
        rgb, depth = rgb_depth(obs)
        state = self._agent.get_state()
        x, _y, z = (float(v) for v in state.position)
        yaw = float(quaternion.as_rotation_vector(state.rotation)[1])
        pose = Pose(x=x, y=float(state.position[1]), z=z, yaw=yaw)
        return Observation(rgb=rgb, depth=depth, pose=pose, frame_idx=frame_idx)

    def step(self, pose: Pose) -> None:
        y = self._terrain(pose.x, pose.z)
        set_agent_pose(self._agent, pose.x, y, pose.z, pose.yaw)


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent.parent.parent
    scene = HERE / "assets" / "marsyard2022.glb"
    heightmap = HERE / "marsyard2022_terrain_hm.png"

    with MarsHabitatEnv(str(scene), str(heightmap), services=[]) as env:
        obs = env.get_observation(0)
        print(f"rgb shape={obs.rgb.shape} dtype={obs.rgb.dtype}")
        print(f"pose={obs.pose}")
