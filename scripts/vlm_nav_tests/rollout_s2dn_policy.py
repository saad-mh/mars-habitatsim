from __future__ import annotations

import argparse
import io
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import habitat_sim
import numpy as np
import quaternion
import requests
from habitat_sim.agent import AgentConfiguration
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566
MESH_OBSTACLE_ID = 2


@dataclass(frozen=True)
class NavDPS2DiffOutput:
    trajectory: np.ndarray
    all_trajectories: np.ndarray
    all_values: np.ndarray
    selected_index: int
    fallback_stop: bool
    escape_turn: bool
    valid_obstacle_points: int
    selected_circulation_sign: float
    selected_barrier_energy: float
    selected_circulation_energy: float
    minimum_clearance: np.ndarray
    selected_minimum_clearance: float
    mean_guidance_noise_correction: float
    final_guidance_noise_correction: float
    maximum_guidance_noise_correction: float


class NavDPS2DiffClient:
    def __init__(self, server_url: str, timeout: float = 180.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)

    def reset(
        self,
        intrinsic: np.ndarray,
        *,
        stop_threshold: float = -3.0,
        batch_size: int = 1,
    ) -> str:
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must have shape [3,3], got {intrinsic.shape}")
        response = requests.post(
            f"{self.server_url}/navigator_reset",
            json={
                "intrinsic": intrinsic.tolist(),
                "stop_threshold": float(stop_threshold),
                "batch_size": int(batch_size),
            },
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        return str(response.json().get("algo", ""))

    def plan(
        self,
        *,
        goal_xy: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        obstacle_pixels: np.ndarray,
    ) -> NavDPS2DiffOutput:
        goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        if goal_xy.shape != (2,):
            raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")

        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
        rgb = rgb[..., :3]

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"depth/rgb shape mismatch: {depth.shape} vs {rgb.shape[:2]}"
            )

        pixels = np.asarray(obstacle_pixels)
        if pixels.size == 0:
            pixels = np.zeros((0, 2), dtype=np.int32)
        else:
            pixels = pixels.reshape(-1, 2)
            if not np.all(np.isfinite(pixels)):
                raise ValueError("obstacle pixels must be finite")
            if not np.allclose(pixels, np.round(pixels)):
                raise ValueError("obstacle pixels must be integer [u,v] coordinates")
            pixels = np.round(pixels).astype(np.int32)

        rgb_bytes = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(rgb_bytes, format="JPEG", quality=95)
        depth_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
        depth_bytes = io.BytesIO()
        Image.fromarray(depth_u16).save(depth_bytes, format="PNG")

        response = requests.post(
            f"{self.server_url}/pointgoal_step",
            files={
                "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
                "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
            },
            data={
                "goal_data": json.dumps(
                    {
                        "goal_x": [float(goal_xy[0])],
                        "goal_y": [float(goal_xy[1])],
                        "obstacle_pixels": [pixels.tolist()],
                    }
                )
            },
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        diagnostics = payload["s2diff"]
        trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
        all_trajectories = np.asarray(payload["all_trajectory"], dtype=np.float32)
        all_values = np.asarray(payload["all_values"], dtype=np.float32)

        return NavDPS2DiffOutput(
            trajectory=trajectory[0],
            all_trajectories=all_trajectories[0],
            all_values=all_values[0],
            selected_index=int(diagnostics["selected_index"][0]),
            fallback_stop=bool(diagnostics["fallback_stop"][0]),
            escape_turn=bool(diagnostics["escape_turn"][0]),
            valid_obstacle_points=int(diagnostics["valid_obstacle_points"][0]),
            selected_circulation_sign=float(
                diagnostics["selected_circulation_sign"][0]
            ),
            selected_barrier_energy=float(diagnostics["selected_barrier_energy"][0]),
            selected_circulation_energy=float(
                diagnostics["selected_circulation_energy"][0]
            ),
            minimum_clearance=np.asarray(
                diagnostics["minimum_clearance"][0], dtype=np.float32
            ),
            selected_minimum_clearance=float(
                diagnostics["selected_minimum_clearance"][0]
            ),
            mean_guidance_noise_correction=float(
                diagnostics["mean_guidance_noise_correction"][0]
            ),
            final_guidance_noise_correction=float(
                diagnostics["final_guidance_noise_correction"][0]
            ),
            maximum_guidance_noise_correction=float(
                diagnostics["maximum_guidance_noise_correction"][0]
            ),
        )

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(str(payload["error"]))
        response.raise_for_status()


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def wait_for_server(
    process: subprocess.Popen[Any], host: str, port: int, timeout: float
) -> None:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"NavDP/S2Diff server exited with code {process.returncode}"
            )
        if port_is_open(host, port):
            return
        time.sleep(1.0)
    raise TimeoutError(f"NavDP server did not open port {port} within {timeout}s")


def stop_server(process: Optional[subprocess.Popen[Any]]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start_server(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
    if not args.start_server:
        return None
    if port_is_open(args.server_host, args.server_port):
        raise RuntimeError(
            f"port {args.server_port} is already in use; use --no-start-server "
            "to connect to an existing guided server"
        )

    navdp_root = Path(args.navdp_root).expanduser().resolve()
    checkpoint = Path(args.navdp_checkpoint).expanduser().resolve()
    server_dir = navdp_root / "baselines" / "navdp"
    server_file = server_dir / "navdp_s2diff_server.py"
    if not server_file.is_file():
        raise FileNotFoundError(f"guided server not found: {server_file}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint}")

    command = [
        str(args.navdp_python),
        str(server_file),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(args.navdp_device),
        "--planner-mode",
        str(args.planner_mode),
        "--seed",
        str(args.seed),
        "--port",
        str(args.server_port),
        "--candidates",
        str(args.candidates),
        "--particles",
        str(args.particles),
        "--particle-std",
        str(args.particle_std),
        "--guidance-strength",
        str(args.guidance_strength),
        "--temperature",
        str(args.temperature),
        "--safe-distance",
        str(args.safe_distance),
        "--hard-collision-distance",
        str(args.hard_collision_distance),
        "--safety-weight",
        str(args.safety_weight),
        "--barrier-weight",
        str(args.barrier_weight),
        "--barrier-rate",
        str(args.barrier_rate),
        "--circulation-weight",
        str(args.circulation_weight),
        "--circulation-activation-distance",
        str(args.circulation_activation_distance),
        "--circulation-activation-sharpness",
        str(args.circulation_activation_sharpness),
        "--minimum-circulation-progress",
        str(args.minimum_circulation_progress),
        "--blocking-alignment-threshold",
        str(args.blocking_alignment_threshold),
        "--circulation-switch-weight",
        str(args.circulation_switch_weight),
        "--escape-lateral-target",
        str(args.escape_lateral_target),
        "--minimum-obstacle-depth",
        str(args.minimum_obstacle_depth),
        "--maximum-obstacle-depth",
        str(args.maximum_obstacle_depth),
        "--maximum-obstacle-pixels",
        str(args.maximum_obstacle_pixels),
    ]
    command.append("--remove-critic" if args.remove_critic else "--no-remove-critic")
    print("[server]", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(server_dir))
    wait_for_server(process, args.server_host, args.server_port, args.server_timeout)
    return process


def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
    height, width = grid.shape
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = px - x0
    ty = py - y0
    top = float(grid[y0, x0]) * (1.0 - tx) + float(grid[y0, x1]) * tx
    bottom = float(grid[y1, x0]) * (1.0 - tx) + float(grid[y1, x1]) * tx
    return top * (1.0 - ty) + bottom * ty


class TerrainHeight:
    def __init__(
        self,
        *,
        mode: str,
        heightmap: Optional[Path],
        obj: Optional[Path],
        flat_y: float,
        size_x: float,
        size_z: float,
        size_y: float,
        flip_x: bool,
        flip_z: bool,
        swap_xz: bool,
    ):
        if mode == "auto":
            mode = (
                "heightmap"
                if heightmap and heightmap.exists()
                else ("obj" if obj and obj.exists() else "flat")
            )
        self.mode = mode
        self.flat_y = float(flat_y)
        self.size_x = float(size_x)
        self.size_z = float(size_z)
        self.size_y = float(size_y)
        self.flip_x = bool(flip_x)
        self.flip_z = bool(flip_z)
        self.swap_xz = bool(swap_xz)
        self.height: Optional[np.ndarray] = None
        self.obj_xs: Optional[np.ndarray] = None
        self.obj_zs: Optional[np.ndarray] = None
        self.obj_h: Optional[np.ndarray] = None

        if mode == "heightmap":
            if heightmap is None or not heightmap.exists():
                raise FileNotFoundError(f"heightmap not found: {heightmap}")
            array = np.asarray(Image.open(heightmap))
            if array.ndim == 3:
                array = array[..., 0]
            array = array.astype(np.float32)
            array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
            self.height = array * self.size_y - float(np.mean(array * self.size_y))
        elif mode == "obj":
            if obj is None or not obj.exists():
                raise FileNotFoundError(f"terrain OBJ not found: {obj}")
            vertices = []
            with obj.open("r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if line.startswith("v "):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.append(tuple(float(value) for value in parts[1:4]))
            if not vertices:
                raise RuntimeError(f"no vertices found in {obj}")
            array = np.asarray(vertices, dtype=np.float32)
            xs = np.unique(array[:, 0])
            zs = np.unique(array[:, 1])
            grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
            x_index = {float(value): index for index, value in enumerate(xs.tolist())}
            z_index = {float(value): index for index, value in enumerate(zs.tolist())}
            for x, z, height in array:
                grid[z_index[float(z)], x_index[float(x)]] = height
            self.obj_xs = xs
            self.obj_zs = zs
            self.obj_h = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
        elif mode != "flat":
            raise ValueError(f"unknown terrain mode: {mode}")

    def _map(self, x: float, z: float) -> tuple[float, float]:
        if self.swap_xz:
            x, z = z, x
        u = (x + self.size_x / 2.0) / self.size_x
        v = (z + self.size_z / 2.0) / self.size_z
        if self.flip_x:
            u = 1.0 - u
        if self.flip_z:
            v = 1.0 - v
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

    def __call__(self, x: float, z: float) -> float:
        if self.mode == "flat":
            return self.flat_y
        if self.mode == "heightmap":
            assert self.height is not None
            u, v = self._map(x, z)
            return bilinear_grid(
                self.height,
                u * (self.height.shape[1] - 1),
                v * (self.height.shape[0] - 1),
            )
        assert (
            self.obj_xs is not None
            and self.obj_zs is not None
            and self.obj_h is not None
        )
        xx = float(np.clip(x, self.obj_xs[0], self.obj_xs[-1]))
        zz = float(np.clip(z, self.obj_zs[0], self.obj_zs[-1]))
        column = int(
            np.clip(np.searchsorted(self.obj_xs, xx) - 1, 0, len(self.obj_xs) - 2)
        )
        row = int(
            np.clip(np.searchsorted(self.obj_zs, zz) - 1, 0, len(self.obj_zs) - 2)
        )
        x0, x1 = float(self.obj_xs[column]), float(self.obj_xs[column + 1])
        z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
        tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
        tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
        top = (
            float(self.obj_h[row, column]) * (1.0 - tx)
            + float(self.obj_h[row, column + 1]) * tx
        )
        bottom = (
            float(self.obj_h[row + 1, column]) * (1.0 - tx)
            + float(self.obj_h[row + 1, column + 1]) * tx
        )
        return top * (1.0 - tz) + bottom * tz

    def local_height_max(
        self, x: float, z: float, radius: float, samples: int = 5
    ) -> float:
        if radius <= 1e-6:
            return float(self(x, z))
        values = [
            float(self(x + dx, z + dz))
            for dx in np.linspace(-radius, radius, samples)
            for dz in np.linspace(-radius, radius, samples)
            if dx * dx + dz * dz <= radius * radius + 1e-8
        ]
        return max(values) if values else float(self(x, z))


def make_sensor(
    uuid: str, sensor_type: Any, height: int, width: int, hfov_deg: float
) -> habitat_sim.CameraSensorSpec:
    specification = habitat_sim.CameraSensorSpec()
    specification.uuid = uuid
    specification.sensor_type = sensor_type
    specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    specification.resolution = [int(height), int(width)]
    specification.position = [0.0, 0.0, 0.0]
    specification.hfov = float(hfov_deg)
    return specification


def make_simulator(
    scene: Path,
    height: int,
    width: int,
    hfov_deg: float,
    *,
    with_semantic: bool,
):
    simulator_configuration = habitat_sim.SimulatorConfiguration()
    simulator_configuration.scene_id = str(scene.expanduser().resolve())
    simulator_configuration.enable_physics = False
    sensors = [
        make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
        make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
    ]
    if with_semantic:
        sensors.append(
            make_sensor(
                "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
            )
        )
    agent_configuration = AgentConfiguration()
    agent_configuration.sensor_specifications = sensors
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_configuration, [agent_configuration])
    )


def set_agent_pose(agent: Any, position: np.ndarray, yaw: float) -> None:
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
    agent.set_state(state)


def rgb_depth(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(observation["rgb"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    depth = np.asarray(observation["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return rgb.astype(np.uint8), depth.astype(np.float32)


def semantic_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    semantic = np.asarray(observation["semantic"])
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    return semantic.astype(np.int32)


def pixel_to_world(
    u: float,
    v: float,
    depth: float,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
) -> np.ndarray:
    right = (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
    up = -(v - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
    forward_vector = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)])
    right_vector = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)])
    return (
        np.asarray(position, dtype=np.float64)
        + depth * forward_vector
        + right * right_vector
        + up * np.asarray([0.0, 1.0, 0.0])
    )


def depth_patch_mesh(
    u_center: float,
    v_center: float,
    half_size: int,
    stride: int,
    depth: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    *,
    lift: float,
    maximum_depth_jump: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    columns = list(
        range(
            max(0, int(u_center - half_size)),
            min(width, int(u_center + half_size) + 1),
            max(int(stride), 1),
        )
    )
    rows = list(
        range(
            max(0, int(v_center - half_size)),
            min(height, int(v_center + half_size) + 1),
            max(int(stride), 1),
        )
    )
    indices = -np.ones((len(rows), len(columns)), dtype=np.int64)
    depths = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
    vertices: list[tuple[float, float, float]] = []
    for row_index, v in enumerate(rows):
        for column_index, u in enumerate(columns):
            metric_depth = float(depth[v, u])
            if not np.isfinite(metric_depth) or metric_depth <= 0.1:
                continue
            indices[row_index, column_index] = len(vertices)
            depths[row_index, column_index] = metric_depth
            point = pixel_to_world(
                u, v, metric_depth, position, yaw, intrinsic
            ) + float(lift) * np.asarray([0.0, 1.0, 0.0])
            vertices.append(tuple(float(value) for value in point))

    faces: list[tuple[int, int, int]] = []
    for row_index in range(len(rows) - 1):
        for column_index in range(len(columns) - 1):
            a = int(indices[row_index, column_index])
            b = int(indices[row_index, column_index + 1])
            c = int(indices[row_index + 1, column_index])
            d = int(indices[row_index + 1, column_index + 1])
            if min(a, b, c, d) < 0:
                continue
            cell_depths = (
                depths[row_index, column_index],
                depths[row_index, column_index + 1],
                depths[row_index + 1, column_index],
                depths[row_index + 1, column_index + 1],
            )
            if max(cell_depths) - min(cell_depths) > maximum_depth_jump:
                continue
            faces.append((a, c, d))
            faces.append((a, d, b))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as file:
        for x, y, z in vertices:
            file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            file.write(f"f {a + 1} {b + 1} {c + 1}\n")


def register_semantic_mesh(simulator: Any, mesh_path: Path, semantic_id: int) -> Any:
    template_manager = simulator.get_object_template_manager()
    object_manager = simulator.get_rigid_object_manager()
    template = template_manager.create_new_template(str(mesh_path))
    template.render_asset_handle = str(mesh_path)
    template.collision_asset_handle = str(mesh_path)
    template.is_collidable = False
    template_id = template_manager.register_template(
        template, f"s2diff_obstacle_{semantic_id}_{os.path.basename(mesh_path)}"
    )
    object_handle = template_manager.get_template_handle_by_id(template_id)
    obstacle = object_manager.add_object_by_template_handle(object_handle)
    obstacle.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obstacle.collidable = False
    obstacle.semantic_id = int(semantic_id)
    return obstacle


def parse_uv_fraction(
    specification: str, width: int, height: int
) -> tuple[float, float]:
    u_fraction, v_fraction = (float(value) for value in str(specification).split(","))
    if not (0.0 <= u_fraction <= 1.0 and 0.0 <= v_fraction <= 1.0):
        raise ValueError(f"mesh pixel fraction must be in [0,1], got {specification!r}")
    return u_fraction * width, v_fraction * height


def place_obstacle_meshes(
    simulator: Any,
    depth: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    uv_specifications: Sequence[str],
    output_directory: Path,
    *,
    mesh_half_pixels: int,
    mesh_lift: float,
) -> tuple[list[Any], list[np.ndarray]]:
    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    height, width = depth.shape
    objects: list[Any] = []
    centroids: list[np.ndarray] = []
    for index, specification in enumerate(uv_specifications):
        u, v = parse_uv_fraction(specification, width, height)
        vertices, faces = depth_patch_mesh(
            u,
            v,
            mesh_half_pixels,
            2,
            depth,
            position,
            yaw,
            intrinsic,
            lift=mesh_lift,
        )
        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError(
                f"obstacle mesh {index} at {specification!r} has no valid depth surface"
            )
        mesh_path = mesh_directory / f"obstacle_{index}.obj"
        save_obj(mesh_path, vertices, faces)
        semantic_id = MESH_OBSTACLE_ID + index
        objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
        centroid = vertices.mean(axis=0).astype(np.float32)
        centroids.append(centroid)
        print(
            f"[mesh] obstacle={index} semantic_id={semantic_id} "
            f"pixels={specification} vertices={len(vertices)} "
            f"world={centroid.tolist()}",
            flush=True,
        )
    return objects, centroids


def camera_coordinates(
    point: np.ndarray, position: np.ndarray, yaw: float
) -> tuple[float, float, float]:
    delta = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    forward = forward_x * float(delta[0]) + forward_z * float(delta[2])
    left = left_x * float(delta[0]) + left_z * float(delta[2])
    return -left, float(delta[1]), forward


def camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(float(hfov_deg))
    focal = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    return np.asarray(
        [
            [focal, 0.0, (width - 1) * 0.5],
            [0.0, focal, (height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


def project_world_mask(
    point: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    radius: int,
) -> tuple[np.ndarray, float]:
    right, up, forward = camera_coordinates(point, position, yaw)
    if forward <= 0.05:
        return np.zeros((height, width), dtype=np.uint8), forward
    u = float(intrinsic[0, 2] + intrinsic[0, 0] * right / forward)
    v = float(intrinsic[1, 2] - intrinsic[1, 1] * up / forward)
    if not (radius <= u < width - radius and radius <= v < height - radius):
        return np.zeros((height, width), dtype=np.uint8), forward
    return circle_mask(height, width, u, v, radius), forward


def depth_obstacle_mask(
    depth: np.ndarray, threshold: float, minimum_y_fraction: float
) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > 0.05) & (depth < float(threshold))
    mask[: int(depth.shape[0] * minimum_y_fraction)] = False
    return mask.astype(np.uint8)


def pixels_from_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
    v, u = np.nonzero(np.asarray(mask) > 0)
    if u.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pixels = np.stack((u, v), axis=-1).astype(np.int32)
    if maximum > 0 and len(pixels) > maximum:
        indices = np.linspace(0, len(pixels) - 1, maximum).astype(np.int64)
        pixels = pixels[indices]
    return pixels


def waypoint_action(
    trajectory: np.ndarray,
    *,
    lookahead_index: int,
    maximum_forward_speed: float,
    maximum_yaw_rate: float,
    yaw_gain: float,
) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
        return np.zeros(3, dtype=np.float32)
    if np.max(np.linalg.norm(trajectory[:, :2], axis=-1)) < 1e-5:
        return np.zeros(3, dtype=np.float32)
    index = int(np.clip(lookahead_index, 0, trajectory.shape[0] - 1))
    forward, left = float(trajectory[index, 0]), float(trajectory[index, 1])
    bearing = math.atan2(left, max(forward, 1e-4))
    velocity = maximum_forward_speed * max(0.0, math.cos(bearing))
    yaw_rate = float(np.clip(yaw_gain * bearing, -maximum_yaw_rate, maximum_yaw_rate))
    return np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)


def integrate_mars(
    position: np.ndarray, yaw: float, action: np.ndarray, dt: float
) -> tuple[np.ndarray, float]:
    forward_velocity, lateral_velocity, yaw_rate = [float(value) for value in action]
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    output = np.asarray(position, dtype=np.float32).copy()
    output[0] += (forward_x * forward_velocity + left_x * lateral_velocity) * dt
    output[2] += (forward_z * forward_velocity + left_z * lateral_velocity) * dt
    return output, yaw + yaw_rate * dt


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def overlay_frame(
    rgb: np.ndarray, goal_mask: np.ndarray, obstacle_mask: np.ndarray, text: str
) -> Image.Image:
    output = np.asarray(rgb, dtype=np.uint8).copy()
    output[goal_mask > 0] = (
        0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
    ).astype(np.uint8)
    output[obstacle_mask > 0] = (
        0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
    ).astype(np.uint8)
    image = Image.fromarray(output)
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
    draw.text((10, 9), text, fill=(255, 255, 255))
    return image


def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=float(fps)) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="One-file released NavDP + in-denoising S2Diff Mars rollout"
    )
    argument_parser.add_argument("--navdp-root", required=True)
    argument_parser.add_argument("--navdp-checkpoint", required=True)
    argument_parser.add_argument("--navdp-python", default=sys.executable)
    argument_parser.add_argument("--navdp-device", default="cuda:0")
    argument_parser.add_argument(
        "--planner-mode", choices=["pure-navdp", "s2diff"], default="s2diff"
    )
    argument_parser.add_argument(
        "--remove-critic", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--seed", type=int, default=7)
    argument_parser.add_argument(
        "--start-server", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--server-host", default="127.0.0.1")
    argument_parser.add_argument("--server-port", type=int, default=8888)
    argument_parser.add_argument("--server-timeout", type=float, default=180.0)
    argument_parser.add_argument("--candidates", type=int, default=16)
    argument_parser.add_argument("--particles", type=int, default=8)
    argument_parser.add_argument("--particle-std", type=float, default=0.22)
    argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
    argument_parser.add_argument("--temperature", type=float, default=0.35)
    argument_parser.add_argument("--safe-distance", type=float, default=0.42)
    argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
    argument_parser.add_argument("--safety-weight", type=float, default=35.0)
    argument_parser.add_argument("--barrier-weight", type=float, default=25.0)
    argument_parser.add_argument("--barrier-rate", type=float, default=0.15)
    argument_parser.add_argument("--circulation-weight", type=float, default=18.0)
    argument_parser.add_argument(
        "--circulation-activation-distance", type=float, default=1.50
    )
    argument_parser.add_argument(
        "--circulation-activation-sharpness", type=float, default=0.20
    )
    argument_parser.add_argument(
        "--minimum-circulation-progress", type=float, default=0.025
    )
    argument_parser.add_argument(
        "--blocking-alignment-threshold", type=float, default=0.25
    )
    argument_parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
    argument_parser.add_argument("--escape-lateral-target", type=float, default=0.35)
    argument_parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
    argument_parser.add_argument("--maximum-obstacle-depth", type=float, default=5.0)
    argument_parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)

    argument_parser.add_argument("--scene", required=True)
    argument_parser.add_argument("--terrain-obj", default=None)
    argument_parser.add_argument("--heightmap", default=None)
    argument_parser.add_argument(
        "--terrain-height-mode",
        choices=["auto", "heightmap", "obj", "flat"],
        default="auto",
    )
    argument_parser.add_argument("--flat-y", type=float, default=0.0)
    argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
    argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
    argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
    argument_parser.add_argument("--flip-heightmap-x", action="store_true")
    argument_parser.add_argument(
        "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--swap-heightmap-xz", action="store_true")
    argument_parser.add_argument("--clearance", type=float, default=1.4)
    argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)

    argument_parser.add_argument("--height", type=int, default=720)
    argument_parser.add_argument("--width", type=int, default=720)
    argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
    argument_parser.add_argument("--hz", type=float, default=10.0)
    argument_parser.add_argument("--max-steps", type=int, default=300)
    argument_parser.add_argument("--stop-distance", type=float, default=1.0)
    argument_parser.add_argument("--start-x", type=float, default=0.0)
    argument_parser.add_argument("--start-z", type=float, default=8.0)
    argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    argument_parser.add_argument("--goal-x", type=float, required=True)
    argument_parser.add_argument("--goal-z", type=float, required=True)
    argument_parser.add_argument("--goal-y", type=float, default=None)
    argument_parser.add_argument("--goal-height", type=float, default=1.2)
    argument_parser.add_argument("--goal-radius", type=int, default=18)

    argument_parser.add_argument(
        "--obstacle-mode", choices=["none", "depth", "mesh", "ghost"], default="none"
    )
    argument_parser.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
    argument_parser.add_argument("--obstacle-min-y-fraction", type=float, default=0.45)
    argument_parser.add_argument("--ghost-obstacle-x", type=float, default=None)
    argument_parser.add_argument("--ghost-obstacle-z", type=float, default=None)
    argument_parser.add_argument("--ghost-obstacle-y", type=float, default=None)
    argument_parser.add_argument("--ghost-obstacle-height", type=float, default=0.45)
    argument_parser.add_argument("--ghost-obstacle-radius", type=int, default=24)
    argument_parser.add_argument(
        "--obstacle-mesh-uv",
        nargs="+",
        default=[],
        help=(
            "Actual rendered obstacle mesh locations as image fractions u,v. "
            "Example: --obstacle-mesh-uv 0.50,0.72 0.30,0.68"
        ),
    )
    argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
    argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)

    argument_parser.add_argument("--lookahead-index", type=int, default=4)
    argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
    argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
    argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
    argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
    argument_parser.add_argument("--save-every", type=int, default=1)
    argument_parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    np.random.seed(args.seed)
    if args.obstacle_mode == "ghost" and (
        args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
    ):
        raise ValueError(
            "ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z"
        )
    if args.obstacle_mode == "mesh" and not args.obstacle_mesh_uv:
        raise ValueError("mesh mode requires --obstacle-mesh-uv u,v [u,v ...]")

    server_process: Optional[subprocess.Popen[Any]] = None
    simulator = None
    try:
        server_process = start_server(args)
        server_url = f"http://{args.server_host}:{args.server_port}"
        client = NavDPS2DiffClient(server_url)
        algorithm = client.reset(
            camera_intrinsic(args.height, args.width, args.hfov_deg),
            batch_size=1,
            stop_threshold=-3.0,
        )
        supported_algorithms = {
            "navdp-s2diff-pixels",
            "navdp-hlc-s2diff",
            "navdp-hlc-s2diff-no-critic",
            "navdp-pure-critic",
        }
        if algorithm not in supported_algorithms:
            raise RuntimeError(f"unexpected planner response: {algorithm!r}")

        terrain = TerrainHeight(
            mode=args.terrain_height_mode,
            heightmap=(
                Path(args.heightmap).expanduser().resolve() if args.heightmap else None
            ),
            obj=(
                Path(args.terrain_obj).expanduser().resolve()
                if args.terrain_obj
                else None
            ),
            flat_y=args.flat_y,
            size_x=args.size_x,
            size_z=args.size_z,
            size_y=args.size_y,
            flip_x=args.flip_heightmap_x,
            flip_z=args.flip_heightmap_z,
            swap_xz=args.swap_heightmap_xz,
        )
        output_directory = Path(args.output).expanduser().resolve()
        frame_directory = output_directory / "frames"
        frame_directory.mkdir(parents=True, exist_ok=True)

        simulator = make_simulator(
            Path(args.scene),
            args.height,
            args.width,
            args.hfov_deg,
            with_semantic=args.obstacle_mode == "mesh",
        )
        agent = simulator.initialize_agent(0)
        intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
        x, z = float(args.start_x), float(args.start_z)
        yaw = math.radians(float(args.start_yaw_deg))
        dt = 1.0 / float(args.hz)

        goal_y = args.goal_y
        if goal_y is None:
            goal_y = (
                terrain.local_height_max(args.goal_x, args.goal_z, 0.8)
                + args.goal_height
            )
        goal = np.asarray([args.goal_x, goal_y, args.goal_z], dtype=np.float32)

        ghost = None
        if args.obstacle_mode == "ghost":
            ghost_y = args.ghost_obstacle_y
            if ghost_y is None:
                ghost_y = (
                    terrain.local_height_max(
                        args.ghost_obstacle_x,
                        args.ghost_obstacle_z,
                        args.pose_terrain_radius,
                    )
                    + args.ghost_obstacle_height
                )
            ghost = np.asarray(
                [args.ghost_obstacle_x, ghost_y, args.ghost_obstacle_z],
                dtype=np.float32,
            )

        mesh_objects: list[Any] = []
        mesh_centroids: list[np.ndarray] = []
        mesh_placed = False

        rows: dict[str, list[Any]] = {
            key: []
            for key in (
                "rgb",
                "depth",
                "goal_mask",
                "obstacle_mask",
                "pose",
                "action_3d",
                "point_goal",
                "selected_trajectory",
                "all_trajectories",
                "all_values",
                "selected_index",
                "fallback_stop",
                "escape_turn",
                "valid_obstacle_points",
                "selected_circulation_sign",
                "selected_barrier_energy",
                "selected_circulation_energy",
                "planning_time_seconds",
                "selected_minimum_clearance",
                "mean_guidance_noise_correction",
                "final_guidance_noise_correction",
                "maximum_guidance_noise_correction",
                "goal_distance",
            )
        }
        video_frames: list[Image.Image] = []
        success = False

        for step in range(int(args.max_steps)):
            y = (
                terrain.local_height_max(x, z, args.pose_terrain_radius)
                + args.clearance
            )
            position = np.asarray([x, y, z], dtype=np.float32)
            set_agent_pose(agent, position, yaw)
            observation = simulator.get_sensor_observations()
            rgb, depth = rgb_depth(observation)

            if args.obstacle_mode == "mesh" and not mesh_placed:
                mesh_objects, mesh_centroids = place_obstacle_meshes(
                    simulator,
                    depth,
                    position,
                    yaw,
                    intrinsic,
                    args.obstacle_mesh_uv,
                    output_directory,
                    mesh_half_pixels=args.mesh_half_pixels,
                    mesh_lift=args.mesh_obstacle_lift,
                )
                mesh_placed = True
                observation = simulator.get_sensor_observations()
                rgb, depth = rgb_depth(observation)

            goal_right, _goal_up, goal_forward = camera_coordinates(goal, position, yaw)
            point_goal = np.asarray(
                [max(goal_forward, 0.0), -goal_right], dtype=np.float32
            )
            goal_mask, _ = project_world_mask(
                goal,
                position,
                yaw,
                intrinsic,
                args.height,
                args.width,
                args.goal_radius,
            )

            guidance_depth = depth.copy()
            if args.obstacle_mode == "depth":
                obstacle_mask = depth_obstacle_mask(
                    depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
                )
            elif args.obstacle_mode == "mesh":
                semantic = semantic_from_observation(observation)
                semantic_ids = list(
                    range(
                        MESH_OBSTACLE_ID,
                        MESH_OBSTACLE_ID + len(mesh_objects),
                    )
                )
                obstacle_mask = np.isin(semantic, semantic_ids).astype(np.uint8)
                # The depth image was re-rendered after mesh placement, so
                # guidance_depth already contains the real obstacle depth.
            elif args.obstacle_mode == "ghost":
                assert ghost is not None
                obstacle_mask, obstacle_forward = project_world_mask(
                    ghost,
                    position,
                    yaw,
                    intrinsic,
                    args.height,
                    args.width,
                    args.ghost_obstacle_radius,
                )
                if obstacle_forward > 0.05:
                    guidance_depth[obstacle_mask > 0] = obstacle_forward
            else:
                obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)

            # Replace this mask-to-pixels line with your own detector's [u,v]
            # array if obstacle pixels already come directly from your system.
            obstacle_pixels = pixels_from_mask(
                obstacle_mask, args.maximum_obstacle_pixels
            )
            planning_start = time.perf_counter()
            result = client.plan(
                goal_xy=point_goal,
                rgb=rgb,
                depth=guidance_depth,
                obstacle_pixels=obstacle_pixels,
            )
            planning_time = time.perf_counter() - planning_start
            action = (
                np.zeros(3, dtype=np.float32)
                if result.fallback_stop
                else waypoint_action(
                    result.trajectory,
                    lookahead_index=args.lookahead_index,
                    maximum_forward_speed=args.maximum_forward_speed,
                    maximum_yaw_rate=args.maximum_yaw_rate,
                    yaw_gain=args.yaw_gain,
                )
            )

            next_position, next_yaw = integrate_mars(position, yaw, action, dt)
            x = float(
                np.clip(
                    next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
                )
            )
            z = float(
                np.clip(
                    next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
                )
            )
            yaw = wrap_angle(next_yaw)
            goal_distance = float(np.linalg.norm(goal[[0, 2]] - np.asarray([x, z])))
            rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
            pose = np.asarray(
                [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w],
                dtype=np.float32,
            )

            rows["rgb"].append(rgb)
            rows["depth"].append(depth)
            rows["goal_mask"].append(goal_mask)
            rows["obstacle_mask"].append(obstacle_mask)
            rows["pose"].append(pose)
            rows["action_3d"].append(action)
            rows["point_goal"].append(point_goal)
            rows["selected_trajectory"].append(result.trajectory)
            rows["all_trajectories"].append(result.all_trajectories)
            rows["all_values"].append(result.all_values)
            rows["selected_index"].append(result.selected_index)
            rows["fallback_stop"].append(result.fallback_stop)
            rows["escape_turn"].append(result.escape_turn)
            rows["valid_obstacle_points"].append(result.valid_obstacle_points)
            rows["selected_circulation_sign"].append(result.selected_circulation_sign)
            rows["selected_barrier_energy"].append(result.selected_barrier_energy)
            rows["selected_circulation_energy"].append(
                result.selected_circulation_energy
            )
            rows["planning_time_seconds"].append(planning_time)
            rows["selected_minimum_clearance"].append(result.selected_minimum_clearance)
            rows["mean_guidance_noise_correction"].append(
                result.mean_guidance_noise_correction
            )
            rows["final_guidance_noise_correction"].append(
                result.final_guidance_noise_correction
            )
            rows["maximum_guidance_noise_correction"].append(
                result.maximum_guidance_noise_correction
            )
            rows["goal_distance"].append(goal_distance)

            if step % max(int(args.save_every), 1) == 0:
                label = (
                    f"t={step} goal={goal_distance:.2f}m pixels={len(obstacle_pixels)} "
                    f"clear={result.selected_minimum_clearance:.2f}m "
                    f"mode={result.selected_circulation_sign:+.0f} "
                    f"escape={int(result.escape_turn)} "
                    f"guide_rms={result.mean_guidance_noise_correction:.4f} "
                    f"v={action[0]:.2f} w={action[2]:.2f}"
                )
                frame = overlay_frame(rgb, goal_mask, obstacle_mask, label)
                frame.save(frame_directory / f"frame_{step:04d}.png")
                video_frames.append(frame)

            print(
                f"step={step:04d} goal={goal_distance:.2f}m "
                f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
                f"selected={result.selected_index} fallback={result.fallback_stop} "
                f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
                f"clear={result.selected_minimum_clearance:.3f}m "
                f"barrier={result.selected_barrier_energy:.5f} "
                f"circ={result.selected_circulation_energy:.5f} "
                f"latency={planning_time * 1000.0:.1f}ms "
                f"guide_rms={result.mean_guidance_noise_correction:.6f} "
                f"action={action.tolist()}",
                flush=True,
            )
            if goal_distance <= args.stop_distance:
                success = True
                break

        if not rows["rgb"]:
            raise RuntimeError("rollout produced no frames")
        rollout_path = output_directory / "rollout.npz"
        np.savez_compressed(
            rollout_path,
            **{
                key: (
                    np.stack(values)
                    if isinstance(values[0], np.ndarray)
                    else np.asarray(values)
                )
                for key, values in rows.items()
            },
            goal_position=goal,
            obstacle_position=(
                mesh_centroids[0]
                if mesh_centroids
                else (
                    ghost
                    if ghost is not None
                    else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
                )
            ),
            obstacle_positions=(
                np.stack(mesh_centroids)
                if mesh_centroids
                else np.zeros((0, 3), dtype=np.float32)
            ),
            success=np.asarray(success),
            hz=np.asarray(args.hz, dtype=np.float32),
        )
        with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "success": success,
                    "frames": len(rows["rgb"]),
                    "final_goal_distance": float(rows["goal_distance"][-1]),
                    "planner": "released_navdp_s2diff_pixels",
                    "controller": "direct_waypoint_no_optimizer",
                    "uses_velocity_chunk": False,
                    "obstacle_mode": args.obstacle_mode,
                    "mesh_obstacle_count": len(mesh_centroids),
                    "rollout": str(rollout_path),
                },
                file,
                indent=2,
            )
        if args.save_video and video_frames:
            save_video(
                video_frames,
                output_directory / "rollout.mp4",
                fps=max(args.hz / max(args.save_every, 1), 1.0),
            )
        print(f"Saved rollout: {rollout_path}", flush=True)
        print(f"Success: {success}", flush=True)
    finally:
        if simulator is not None:
            simulator.close()
        stop_server(server_process)


if __name__ == "__main__":
    main()
