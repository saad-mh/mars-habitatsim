# # # # # """Legacy self-contained NavDP/S2DiT rollout adapter -- the most complete of
# # # # # the scripts/vlm_nav_tests/rollout_navdp*.py iterations: adds a trained
# # # # # --belief-adapter (replacing the plain P-controller) and an async background
# # # # # thread for logging so per-tick work never blocks on disk I/O. Superseded by
# # # # # sam_vla/run_navdp_rollout.py; kept for reference and for flags not yet
# # # # # ported over.

# # # # # Usage:
# # # # #     python scripts/vlm_nav_tests/rollout_navdp_policy.py \
# # # # #       --navdp-root ./navdp --ckpt ./navdp/ckpt_last.pt \
# # # # #       --scene assets/marsyard2022.glb --terrain-obj assets/marsyard2022.obj \
# # # # #       --scene-height-flip-z --start-x 10 --start-z 12 \
# # # # #       --goal-mesh-uv 0.5,0.52 --obstacle-mesh-uv 0.5,0.80 \
# # # # #       --belief-adapter ./navdp/belief_adapter.pt \
# # # # #       --cbf --cbf-mode cone --cbf-hard-gate --save-video --out mars_belief_demo1
# # # # # """

# # # # # from __future__ import annotations

# # # # # import argparse
# # # # # import json
# # # # # import math
# # # # # import os
# # # # # import queue
# # # # # import re
# # # # # import sys
# # # # # import threading
# # # # # import time
# # # # # from datetime import datetime, timezone
# # # # # from pathlib import Path
# # # # # from types import SimpleNamespace
# # # # # from typing import Dict, Optional, Sequence, Tuple

# # # # # import numpy as np
# # # # # import torch
# # # # # from PIL import Image, ImageDraw

# # # # # import habitat_sim
# # # # # from habitat_sim.agent import AgentConfiguration
# # # # # import quaternion

# # # # # HERE = Path(__file__).resolve().parent
# # # # # DEFAULT_SCENE = HERE / "marsyard2022_tri.glb"
# # # # # DEFAULT_OBJ = HERE / "marsyard2022.obj"

# # # # # SIZE_X = 50.0
# # # # # SIZE_Z = 50.0
# # # # # SIZE_Y = 4.820803273566


# # # # # class TerrainHeight:
# # # # #     def __init__(
# # # # #         self,
# # # # #         *,
# # # # #         mode: str,
# # # # #         heightmap: Optional[Path],
# # # # #         obj: Optional[Path],
# # # # #         flat_y: float,
# # # # #         size_x: float,
# # # # #         size_z: float,
# # # # #         size_y: float,
# # # # #         flip_x: bool,
# # # # #         flip_z: bool,
# # # # #         swap_xz: bool,
# # # # #     ):
# # # # #         self.mode = mode
# # # # #         self.flat_y = float(flat_y)
# # # # #         self.size_x = float(size_x)
# # # # #         self.size_z = float(size_z)
# # # # #         self.size_y = float(size_y)
# # # # #         self.flip_x = bool(flip_x)
# # # # #         self.flip_z = bool(flip_z)
# # # # #         self.swap_xz = bool(swap_xz)
# # # # #         self.height = None
# # # # #         self.hm_h = 0
# # # # #         self.hm_w = 0
# # # # #         self.obj_xs = None
# # # # #         self.obj_zs = None
# # # # #         self.obj_h = None

# # # # #         if mode == "auto":
# # # # #             if heightmap is not None and heightmap.exists():
# # # # #                 mode = "heightmap"
# # # # #             elif obj is not None and obj.exists():
# # # # #                 mode = "obj"
# # # # #             else:
# # # # #                 mode = "flat"
# # # # #         self.mode = mode

# # # # #         if self.mode == "heightmap":
# # # # #             if heightmap is None or not heightmap.exists():
# # # # #                 raise FileNotFoundError(f"heightmap not found: {heightmap}")
# # # # #             self._load_heightmap(heightmap)
# # # # #         elif self.mode == "obj":
# # # # #             if obj is None or not obj.exists():
# # # # #                 raise FileNotFoundError(f"OBJ terrain not found: {obj}")
# # # # #             self._load_obj_grid(obj)
# # # # #         elif self.mode == "flat":
# # # # #             pass
# # # # #         else:
# # # # #             raise ValueError(f"unknown terrain height mode: {self.mode}")

# # # # #     def _load_heightmap(self, path: Path) -> None:
# # # # #         arr = np.asarray(Image.open(path))
# # # # #         if arr.ndim == 3:
# # # # #             arr = arr[:, :, 0]
# # # # #         arr = arr.astype(np.float32)
# # # # #         arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8)
# # # # #         y = arr * self.size_y
# # # # #         y = y - float(np.mean(y))
# # # # #         self.height = y.astype(np.float32)
# # # # #         self.hm_h, self.hm_w = self.height.shape

# # # # #     def _load_obj_grid(self, path: Path) -> None:
# # # # #         verts = []
# # # # #         with path.open("r", encoding="utf-8", errors="ignore") as f:
# # # # #             for line in f:
# # # # #                 if not line.startswith("v "):
# # # # #                     continue
# # # # #                 parts = line.split()
# # # # #                 if len(parts) < 4:
# # # # #                     continue
# # # # #                 try:
# # # # #                     # hm2obj.py wrote OBJ as v x row_axis height.  Blender/Habitat
# # # # #                     # turns this into x/z ground plane with y-up height.
# # # # #                     verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
# # # # #                 except ValueError:
# # # # #                     continue
# # # # #         if not verts:
# # # # #             raise RuntimeError(f"no OBJ vertices found in {path}")
# # # # #         arr = np.asarray(verts, dtype=np.float32)
# # # # #         xs = np.unique(arr[:, 0])
# # # # #         zs = np.unique(arr[:, 1])
# # # # #         xs.sort()
# # # # #         zs.sort()
# # # # #         grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
# # # # #         x_to_i = {float(x): i for i, x in enumerate(xs.tolist())}
# # # # #         z_to_i = {float(z): i for i, z in enumerate(zs.tolist())}
# # # # #         for x, z, h in arr:
# # # # #             grid[z_to_i[float(z)], x_to_i[float(x)]] = h
# # # # #         if np.isnan(grid).any():
# # # # #             fill = float(np.nanmean(grid))
# # # # #             grid = np.nan_to_num(grid, nan=fill)
# # # # #         self.obj_xs = xs.astype(np.float32)
# # # # #         self.obj_zs = zs.astype(np.float32)
# # # # #         self.obj_h = grid.astype(np.float32)

# # # # #     def __call__(self, x: float, z: float) -> float:
# # # # #         if self.mode == "flat":
# # # # #             return self.flat_y
# # # # #         if self.mode == "heightmap":
# # # # #             return self._sample_heightmap(x, z)
# # # # #         return self._sample_obj(x, z)

# # # # #     def _map_xz(self, x: float, z: float) -> Tuple[float, float]:
# # # # #         if self.swap_xz:
# # # # #             x, z = z, x
# # # # #         u = (x + self.size_x / 2.0) / self.size_x
# # # # #         v = (z + self.size_z / 2.0) / self.size_z
# # # # #         if self.flip_x:
# # # # #             u = 1.0 - u
# # # # #         if self.flip_z:
# # # # #             v = 1.0 - v
# # # # #         return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

# # # # #     def _sample_heightmap(self, x: float, z: float) -> float:
# # # # #         assert self.height is not None
# # # # #         u, v = self._map_xz(x, z)
# # # # #         px = u * (self.hm_w - 1)
# # # # #         py = v * (self.hm_h - 1)
# # # # #         return bilinear_grid(self.height, px, py)

# # # # #     def _sample_obj(self, x: float, z: float) -> float:
# # # # #         assert (
# # # # #             self.obj_xs is not None
# # # # #             and self.obj_zs is not None
# # # # #             and self.obj_h is not None
# # # # #         )
# # # # #         xx = float(np.clip(x, float(self.obj_xs[0]), float(self.obj_xs[-1])))
# # # # #         zz = float(np.clip(z, float(self.obj_zs[0]), float(self.obj_zs[-1])))
# # # # #         col = np.searchsorted(self.obj_xs, xx) - 1
# # # # #         row = np.searchsorted(self.obj_zs, zz) - 1
# # # # #         col = int(np.clip(col, 0, len(self.obj_xs) - 2))
# # # # #         row = int(np.clip(row, 0, len(self.obj_zs) - 2))
# # # # #         x0, x1 = float(self.obj_xs[col]), float(self.obj_xs[col + 1])
# # # # #         z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
# # # # #         tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
# # # # #         tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
# # # # #         h00 = float(self.obj_h[row, col])
# # # # #         h10 = float(self.obj_h[row, col + 1])
# # # # #         h01 = float(self.obj_h[row + 1, col])
# # # # #         h11 = float(self.obj_h[row + 1, col + 1])
# # # # #         h0 = h00 * (1.0 - tx) + h10 * tx
# # # # #         h1 = h01 * (1.0 - tx) + h11 * tx
# # # # #         return float(h0 * (1.0 - tz) + h1 * tz)


# # # # # class SceneMappedTerrain:
# # # # #     def __init__(self, base, *, flip_x: bool, flip_z: bool, swap_xz: bool):
# # # # #         self.base = base
# # # # #         self.mode = getattr(base, "mode", "unknown")
# # # # #         self.flip_x = bool(flip_x)
# # # # #         self.flip_z = bool(flip_z)
# # # # #         self.swap_xz = bool(swap_xz)

# # # # #     def _map(self, x: float, z: float) -> Tuple[float, float]:
# # # # #         xx = float(x)
# # # # #         zz = float(z)
# # # # #         if self.swap_xz:
# # # # #             xx, zz = zz, xx
# # # # #         if self.flip_x:
# # # # #             xx = -xx
# # # # #         if self.flip_z:
# # # # #             zz = -zz
# # # # #         return xx, zz

# # # # #     def __call__(self, x: float, z: float) -> float:
# # # # #         xx, zz = self._map(x, z)
# # # # #         return float(self.base(xx, zz))

# # # # #     def local_height_max(
# # # # #         self, x: float, z: float, radius: float, samples: int = 5
# # # # #     ) -> float:
# # # # #         radius = max(float(radius), 0.0)
# # # # #         samples = max(int(samples), 1)
# # # # #         if radius <= 1e-6 or samples == 1:
# # # # #             return float(self(x, z))
# # # # #         vals = []
# # # # #         for dx in np.linspace(-radius, radius, samples):
# # # # #             for dz in np.linspace(-radius, radius, samples):
# # # # #                 if dx * dx + dz * dz <= radius * radius + 1e-8:
# # # # #                     vals.append(float(self(float(x) + float(dx), float(z) + float(dz))))
# # # # #         return float(max(vals)) if vals else float(self(x, z))


# # # # # def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
# # # # #     h, w = grid.shape
# # # # #     x0 = int(np.floor(px))
# # # # #     y0 = int(np.floor(py))
# # # # #     x1 = min(x0 + 1, w - 1)
# # # # #     y1 = min(y0 + 1, h - 1)
# # # # #     dx = float(px - x0)
# # # # #     dy = float(py - y0)
# # # # #     h00 = float(grid[y0, x0])
# # # # #     h10 = float(grid[y0, x1])
# # # # #     h01 = float(grid[y1, x0])
# # # # #     h11 = float(grid[y1, x1])
# # # # #     h0 = h00 * (1.0 - dx) + h10 * dx
# # # # #     h1 = h01 * (1.0 - dx) + h11 * dx
# # # # #     return float(h0 * (1.0 - dy) + h1 * dy)


# # # # # def add_navdp_to_path(navdp_root: Path) -> None:
# # # # #     root = navdp_root.expanduser().resolve()
# # # # #     scripts = root / "scripts"
# # # # #     for p in (root, scripts):
# # # # #         if str(p) not in sys.path:
# # # # #             sys.path.insert(0, str(p))


# # # # # def resolve_navdp_root(raw: Optional[str]) -> Path:
# # # # #     candidates = []
# # # # #     if raw:
# # # # #         candidates.append(Path(raw))
# # # # #     env = os.environ.get("NAVDP_ROOT")
# # # # #     if env:
# # # # #         candidates.append(Path(env))
# # # # #     candidates.extend(
# # # # #         [
# # # # #             HERE.parent / "navdp_sam",
# # # # #             HERE.parent / "New code",
# # # # #             HERE.parent / "ICRA2027" / "New code",
# # # # #         ]
# # # # #     )
# # # # #     for c in candidates:
# # # # #         c = c.expanduser().resolve()
# # # # #         if (c / "model_s2_dit.py").exists() and (
# # # # #             c / "scripts" / "rollout_habitat_policy.py"
# # # # #         ).exists():
# # # # #             return c
# # # # #     raise FileNotFoundError(
# # # # #         "Could not find NavDP repo. Pass --navdp-root /path/to/navdp_sam "
# # # # #         "or set NAVDP_ROOT."
# # # # #     )


# # # # # def make_sensor(uuid: str, sensor_type, height: int, width: int, hfov_deg: float):
# # # # #     spec = habitat_sim.CameraSensorSpec()
# # # # #     spec.uuid = uuid
# # # # #     spec.sensor_type = sensor_type
# # # # #     spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
# # # # #     spec.resolution = [int(height), int(width)]
# # # # #     spec.position = [0.0, 0.0, 0.0]
# # # # #     spec.hfov = float(hfov_deg)
# # # # #     return spec


# # # # # def make_sim(
# # # # #     scene: Path, height: int, width: int, hfov_deg: float, with_semantic: bool = False
# # # # # ):
# # # # #     sim_cfg = habitat_sim.SimulatorConfiguration()
# # # # #     sim_cfg.scene_id = str(scene.expanduser().resolve())
# # # # #     sim_cfg.enable_physics = False
# # # # #     specs = [
# # # # #         make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
# # # # #         make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
# # # # #     ]
# # # # #     if with_semantic:  # only added for --goal-mesh-uv; non-mesh runs are unchanged
# # # # #         specs.append(
# # # # #             make_sensor(
# # # # #                 "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
# # # # #             )
# # # # #         )
# # # # #     agent_cfg = AgentConfiguration()
# # # # #     agent_cfg.sensor_specifications = specs
# # # # #     return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


# # # # # def yaw_quat_xyzw(yaw: float) -> np.ndarray:
# # # # #     h = 0.5 * float(yaw)
# # # # #     return np.asarray([0.0, math.sin(h), 0.0, math.cos(h)], dtype=np.float32)


# # # # # def set_agent_pose(agent, x: float, y: float, z: float, yaw: float) -> None:
# # # # #     state = agent.get_state()
# # # # #     state.position = np.asarray([x, y, z], dtype=np.float32)
# # # # #     state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
# # # # #     agent.set_state(state)


# # # # # def rgb_depth(obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
# # # # #     rgb = np.asarray(obs["rgb"])
# # # # #     if rgb.ndim == 3 and rgb.shape[-1] == 4:
# # # # #         rgb = rgb[:, :, :3]
# # # # #     depth = np.asarray(obs["depth"], dtype=np.float32)
# # # # #     if depth.ndim == 3:
# # # # #         depth = depth[..., 0]
# # # # #     return rgb.astype(np.uint8), depth.astype(np.float32)


# # # # # def camera_coords(
# # # # #     point: np.ndarray, position: np.ndarray, yaw: float
# # # # # ) -> Tuple[float, float, float]:
# # # # #     d = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
# # # # #     fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
# # # # #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# # # # #     forward = float(fwd_x * d[0] + fwd_z * d[2])
# # # # #     left = float(left_x * d[0] + left_z * d[2])
# # # # #     right = -left
# # # # #     up = float(d[1])
# # # # #     return right, up, forward


# # # # # def intrinsics_from_hfov(height: int, width: int, hfov_deg: float) -> Dict[str, float]:
# # # # #     hfov = math.radians(float(hfov_deg))
# # # # #     fx = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
# # # # #     fy = fx
# # # # #     return {"fx": fx, "fy": fy, "cx": (width - 1) * 0.5, "cy": (height - 1) * 0.5}


# # # # # # --- rendered-mask goal: place a semantic mesh from a pixel, render its mask, belief from mask ---
# # # # # MESH_GOAL_ID = 1
# # # # # MESH_OBST_ID = 2


# # # # # def semantic_from_obs(obs) -> np.ndarray:
# # # # #     s = np.asarray(obs["semantic"])
# # # # #     if s.ndim == 3:
# # # # #         s = s[..., 0]
# # # # #     return s.astype(np.int32)


# # # # # def pixel_to_world(u, v, d, position, yaw, intr):
# # # # #     """Unproject pixel (u=col, v=row) + planar depth d -> world point (mars camera conventions)."""
# # # # #     right = (u - intr["cx"]) * d / intr["fx"]
# # # # #     up = -(v - intr["cy"]) * d / intr["fy"]
# # # # #     fx_vec = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
# # # # #     rt_vec = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
# # # # #     return (
# # # # #         np.asarray(position, np.float64)
# # # # #         + d * fx_vec
# # # # #         + right * rt_vec
# # # # #         + up * np.array([0.0, 1.0, 0.0])
# # # # #     )


# # # # # def mask_to_body(mask, depth_img, height, width, hfov_deg, fallback_range, min_px=1):
# # # # #     """Body-frame goal point [forward, left] from a rendered mask centroid + depth (belief from mask)."""
# # # # #     ys, xs = np.where(np.asarray(mask) > 0)
# # # # #     if xs.size < min_px:
# # # # #         return None
# # # # #     return pixel_to_body(
# # # # #         float(xs.mean()),
# # # # #         float(ys.mean()),
# # # # #         depth_img,
# # # # #         height,
# # # # #         width,
# # # # #         hfov_deg,
# # # # #         fallback_range,
# # # # #     )


# # # # # def belief_feat(belief, r_scale=10.0):
# # # # #     """[forward,left] -> [cos(bearing), sin(bearing), range/scale]  (matches train_belief_adapter)."""
# # # # #     f, l = float(belief[0]), float(belief[1])
# # # # #     bearing = math.atan2(l, f)
# # # # #     return np.array(
# # # # #         [math.cos(bearing), math.sin(bearing), min(math.hypot(f, l) / r_scale, 1.0)],
# # # # #         np.float32,
# # # # #     )


# # # # # def depth_patch_mesh(
# # # # #     u0, v0, half, stride, depth, position, yaw, intr, lift=0.03, jump=0.4
# # # # # ):
# # # # #     """A surface-following patch: back-project a pixel window through depth so verts sit on the
# # # # #     surface the camera sees (no floating). Skips cells that bridge a depth discontinuity.
# # # # #     """
# # # # #     H, W = depth.shape
# # # # #     us = list(range(max(0, int(u0 - half)), min(W, int(u0 + half) + 1), stride))
# # # # #     vs = list(range(max(0, int(v0 - half)), min(H, int(v0 + half) + 1), stride))
# # # # #     idx = -np.ones((len(vs), len(us)), int)
# # # # #     dep = np.full((len(vs), len(us)), np.nan)
# # # # #     verts = []
# # # # #     for j, vv in enumerate(vs):
# # # # #         for i, uu in enumerate(us):
# # # # #             dd = float(depth[vv, uu])
# # # # #             if not np.isfinite(dd) or dd <= 0.1:
# # # # #                 continue
# # # # #             idx[j, i] = len(verts)
# # # # #             dep[j, i] = dd
# # # # #             verts.append(
# # # # #                 tuple(
# # # # #                     pixel_to_world(uu, vv, dd, position, yaw, intr)
# # # # #                     + lift * np.array([0.0, 1.0, 0.0])
# # # # #                 )
# # # # #             )
# # # # #     faces = []
# # # # #     for j in range(len(vs) - 1):
# # # # #         for i in range(len(us) - 1):
# # # # #             a, b, c, e = idx[j, i], idx[j, i + 1], idx[j + 1, i], idx[j + 1, i + 1]
# # # # #             if min(a, b, c, e) < 0:
# # # # #                 continue
# # # # #             q = dep[j, i], dep[j, i + 1], dep[j + 1, i], dep[j + 1, i + 1]
# # # # #             if max(q) - min(q) > jump:
# # # # #                 continue
# # # # #             faces.append((a, c, e))
# # # # #             faces.append((a, e, b))
# # # # #     return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


# # # # # def _save_obj(path, verts, faces):
# # # # #     with open(path, "w") as f:
# # # # #         for x, y, z in verts:
# # # # #             f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
# # # # #         for a, b, c in faces:
# # # # #             f.write(f"f {a + 1} {b + 1} {c + 1}\n")


# # # # # def register_semantic_mesh(sim, mesh_path, semantic_id):
# # # # #     """Add a render-only (kinematic, non-collidable) mesh carrying a semantic id."""
# # # # #     otm = sim.get_object_template_manager()
# # # # #     rom = sim.get_rigid_object_manager()
# # # # #     t = otm.create_new_template(mesh_path)
# # # # #     t.render_asset_handle = mesh_path
# # # # #     t.collision_asset_handle = mesh_path
# # # # #     t.is_collidable = False
# # # # #     tid = otm.register_template(t, f"sem_{semantic_id}_{os.path.basename(mesh_path)}")
# # # # #     obj = rom.add_object_by_template_handle(otm.get_template_handle_by_id(tid))
# # # # #     obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
# # # # #     obj.collidable = False
# # # # #     obj.semantic_id = int(semantic_id)
# # # # #     return obj


# # # # # def _parse_uv(s, W, H):
# # # # #     fu, fv = (float(t) for t in str(s).split(","))
# # # # #     return fu * W, fv * H


# # # # # def place_mesh_goal_obstacle(sim, depth, position, yaw, intr, args, out_dir):
# # # # #     """Place a goal (and optional raised obstacle) mesh from first-frame pixels; return world centroids."""
# # # # #     md = Path(out_dir) / "meshes"
# # # # #     md.mkdir(parents=True, exist_ok=True)
# # # # #     H, W = depth.shape[:2]
# # # # #     gu, gv = _parse_uv(args.goal_mesh_uv, W, H)
# # # # #     gvv, gff = depth_patch_mesh(
# # # # #         gu, gv, int(args.mesh_half_px), 2, depth, position, yaw, intr
# # # # #     )
# # # # #     goal_world = None
# # # # #     if len(gvv):
# # # # #         gp = str(md / "goal.obj")
# # # # #         _save_obj(gp, gvv, gff)
# # # # #         register_semantic_mesh(sim, gp, MESH_GOAL_ID)
# # # # #         goal_world = gvv.mean(axis=0)
# # # # #         print(
# # # # #             f"[MASK] goal mesh: {len(gvv)} verts at pixel ({gu:.0f},{gv:.0f}) -> "
# # # # #             f"world=({goal_world[0]:.2f},{goal_world[1]:.2f},{goal_world[2]:.2f})",
# # # # #             flush=True,
# # # # #         )
# # # # #     else:
# # # # #         print(
# # # # #             f"[MASK] WARN goal pixel ({gu:.0f},{gv:.0f}) had no valid depth (sky?)",
# # # # #             flush=True,
# # # # #         )
# # # # #     obst_world = None
# # # # #     if args.obstacle_mesh_uv:
# # # # #         ou, ov = _parse_uv(args.obstacle_mesh_uv, W, H)
# # # # #         ovv, off = depth_patch_mesh(
# # # # #             ou,
# # # # #             ov,
# # # # #             int(args.mesh_half_px),
# # # # #             2,
# # # # #             depth,
# # # # #             position,
# # # # #             yaw,
# # # # #             intr,
# # # # #             lift=float(args.mesh_obstacle_lift),
# # # # #         )
# # # # #         if len(ovv):
# # # # #             op = str(md / "obstacle.obj")
# # # # #             _save_obj(op, ovv, off)
# # # # #             register_semantic_mesh(sim, op, MESH_OBST_ID)
# # # # #             obst_world = ovv.mean(axis=0)
# # # # #             print(
# # # # #                 f"[MASK] obstacle mesh: {len(ovv)} verts at pixel ({ou:.0f},{ov:.0f})",
# # # # #                 flush=True,
# # # # #             )
# # # # #     return goal_world, obst_world


# # # # # def draw_circle_mask(
# # # # #     height: int, width: int, u: float, v: float, radius: int
# # # # # ) -> np.ndarray:
# # # # #     yy, xx = np.ogrid[:height, :width]
# # # # #     mask = (xx - float(u)) ** 2 + (yy - float(v)) ** 2 <= float(radius) ** 2
# # # # #     return mask.astype(np.uint8)


# # # # # def project_goal_mask(
# # # # #     *,
# # # # #     goal: np.ndarray,
# # # # #     position: np.ndarray,
# # # # #     yaw: float,
# # # # #     height: int,
# # # # #     width: int,
# # # # #     hfov_deg: float,
# # # # #     radius: int,
# # # # #     clamp_to_edge: bool,
# # # # # ) -> Tuple[np.ndarray, Dict[str, float]]:
# # # # #     intr = intrinsics_from_hfov(height, width, hfov_deg)
# # # # #     right, up, forward = camera_coords(goal, position, yaw)
# # # # #     visible = forward > 0.05
# # # # #     if not visible:
# # # # #         return np.zeros((height, width), dtype=np.uint8), {
# # # # #             "visible": 0.0,
# # # # #             "u": -1.0,
# # # # #             "v": -1.0,
# # # # #             "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
# # # # #             "bearing": float(
# # # # #                 math.atan2(right, forward if abs(forward) > 1e-6 else 1e-6)
# # # # #             ),
# # # # #         }
# # # # #     u = intr["cx"] + intr["fx"] * right / max(forward, 1e-6)
# # # # #     v = intr["cy"] - intr["fy"] * up / max(forward, 1e-6)
# # # # #     in_frame = radius <= u < width - radius and radius <= v < height - radius
# # # # #     if not in_frame and clamp_to_edge:
# # # # #         u = float(np.clip(u, radius, width - radius - 1))
# # # # #         v = float(np.clip(v, radius, height - radius - 1))
# # # # #         in_frame = True
# # # # #     if not in_frame:
# # # # #         return np.zeros((height, width), dtype=np.uint8), {
# # # # #             "visible": 0.0,
# # # # #             "u": float(u),
# # # # #             "v": float(v),
# # # # #             "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
# # # # #             "bearing": float(math.atan2(right, forward)),
# # # # #         }
# # # # #     mask = draw_circle_mask(height, width, u, v, radius)
# # # # #     return mask, {
# # # # #         "visible": 1.0,
# # # # #         "u": float(u),
# # # # #         "v": float(v),
# # # # #         "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
# # # # #         "bearing": float(math.atan2(right, forward)),
# # # # #     }


# # # # # def obstacle_point_from_world(
# # # # #     obstacle: np.ndarray, position: np.ndarray, yaw: float
# # # # # ) -> Optional[np.ndarray]:
# # # # #     right, _up, forward = camera_coords(obstacle, position, yaw)
# # # # #     if forward <= 0.05:
# # # # #         return None
# # # # #     # CBF helpers use robot-frame [x_forward, y_left].
# # # # #     return np.asarray([forward, -right], dtype=np.float32)


# # # # # def project_body_point_mask(bg, height, width, hfov_deg, radius, clamp_to_edge):
# # # # #     """Render a filled-circle goal mask from a BODY-frame point bg=[forward, left]. Used to draw
# # # # #     the belief-tracked goal (bg is a propagated estimate, not the known goal). Mirrors
# # # # #     project_goal_mask's projection."""
# # # # #     intr = intrinsics_from_hfov(height, width, hfov_deg)
# # # # #     forward, left = float(bg[0]), float(bg[1])
# # # # #     right = -left
# # # # #     info = {
# # # # #         "visible": 0.0,
# # # # #         "u": -1.0,
# # # # #         "v": -1.0,
# # # # #         "range": float(math.hypot(forward, left)),
# # # # #         "bearing": float(math.atan2(left, forward)),
# # # # #     }
# # # # #     if forward <= 0.05:
# # # # #         return np.zeros((height, width), dtype=np.uint8), info
# # # # #     u = intr["cx"] + intr["fx"] * right / max(forward, 1e-6)
# # # # #     v = intr["cy"]
# # # # #     in_frame = radius <= u < width - radius and radius <= v < height - radius
# # # # #     if not in_frame and clamp_to_edge:
# # # # #         u = float(np.clip(u, radius, width - radius - 1))
# # # # #         v = float(np.clip(v, radius, height - radius - 1))
# # # # #         in_frame = True
# # # # #     if not in_frame:
# # # # #         info.update({"u": float(u), "v": float(v)})
# # # # #         return np.zeros((height, width), dtype=np.uint8), info
# # # # #     info.update({"visible": 1.0, "u": float(u), "v": float(v)})
# # # # #     return draw_circle_mask(height, width, u, v, radius), info


# # # # # def pixel_to_body(u, v, depth_img, height, width, hfov_deg, fallback_range):
# # # # #     """Unproject an image pixel (u=col, v=row) to a body-frame point [forward, left] using the
# # # # #     depth at that pixel (or a fallback range if depth is missing). This is how a VLM-grounded
# # # # #     goal pixel becomes the belief seed -- language -> where the goal is, in metres."""
# # # # #     intr = intrinsics_from_hfov(height, width, hfov_deg)
# # # # #     iu = int(np.clip(u, 0, width - 1))
# # # # #     iv = int(np.clip(v, 0, height - 1))
# # # # #     d = float(depth_img[iv, iu]) if depth_img is not None else 0.0
# # # # #     rng = d if (np.isfinite(d) and d > 0.1) else float(fallback_range)
# # # # #     right = (float(u) - intr["cx"]) * rng / max(intr["fx"], 1e-6)
# # # # #     return np.asarray([rng, -right], dtype=np.float32)  # [forward, left]


# # # # # def bbox_to_body(bbox_xyxy, depth_img, height, width, hfov_deg, fallback_range):
# # # # #     """Body-frame point [forward, left] from a VLM bbox: bearing from the bbox's center column,
# # # # #     range from the MEDIAN depth over the bbox's interior. More robust than pixel_to_body's single
# # # # #     center pixel, which can land on a depth discontinuity (a rock's silhouette edge, or a gap
# # # # #     between the rock and the background) and seed a badly wrong range that then dead-reckons,
# # # # #     uncorrected, for the rest of the episode."""
# # # # #     intr = intrinsics_from_hfov(height, width, hfov_deg)
# # # # #     x1, y1, x2, y2 = bbox_xyxy
# # # # #     iu1, iu2 = int(np.clip(min(x1, x2), 0, width - 1)), int(
# # # # #         np.clip(max(x1, x2), 0, width - 1)
# # # # #     )
# # # # #     iv1, iv2 = int(np.clip(min(y1, y2), 0, height - 1)), int(
# # # # #         np.clip(max(y1, y2), 0, height - 1)
# # # # #     )
# # # # #     patch = np.asarray(depth_img)[iv1 : iv2 + 1, iu1 : iu2 + 1]
# # # # #     valid = patch[np.isfinite(patch) & (patch > 0.1)]
# # # # #     rng = float(np.median(valid)) if valid.size > 0 else float(fallback_range)
# # # # #     u = 0.5 * (x1 + x2)
# # # # #     right = (u - intr["cx"]) * rng / max(intr["fx"], 1e-6)
# # # # #     return np.asarray([rng, -right], dtype=np.float32)  # [forward, left]


# # # # # def bbox_center_depth(bbox_xyxy, depth_img) -> Optional[float]:
# # # # #     """Depth-sensor reading at a detected object's bbox CENTER pixel -- the straight-line-forward
# # # # #     distance from the agent's camera to that object's surface in the frame the bbox came from.
# # # # #     Unlike bbox_to_body's median-over-interior (built for a robust control seed), this is the raw
# # # # #     single-pixel reading callers want for a literal "how far is this object" log entry.
# # # # #     """
# # # # #     depth = np.asarray(depth_img)
# # # # #     height, width = depth.shape[:2]
# # # # #     x1, y1, x2, y2 = bbox_xyxy
# # # # #     u = int(np.clip(round(0.5 * (float(x1) + float(x2))), 0, width - 1))
# # # # #     v = int(np.clip(round(0.5 * (float(y1) + float(y2))), 0, height - 1))
# # # # #     d = float(depth[v, u])
# # # # #     return d if (np.isfinite(d) and d > 0.0) else None


# # # # # class VlmSelectionPixelGoal:
# # # # #     """Adapts a one-shot VLM object selection (resolve_vlm_selection, run once on an
# # # # #     already-captured+annotated frame) to the .ground(rgb, instruction) grounder interface used by
# # # # #     --grounder stub/qwen, so --goal-from-vlm seeds the belief via the same image-pixel path --
# # # # #     never a world coordinate. The bbox center is stored as a FRACTION of the frame it was resolved
# # # # #     on so it reprojects correctly onto the live rollout frame, whose resolution can differ from
# # # # #     the capture resolution."""

# # # # #     # A one-shot capture-time selection, not a live re-detector: the belief should be seeded from
# # # # #     # it ONCE (dead-reckoned by odometry after), never re-queried on a cadence like --grounder
# # # # #     # stub/qwen (see the main loop's grounder-call gate).
# # # # #     one_shot = True

# # # # #     def __init__(self, bbox_xyxy, capture_hw):
# # # # #         cap_h, cap_w = capture_hw
# # # # #         x1, y1, x2, y2 = bbox_xyxy
# # # # #         self._u_frac = 0.5 * (x1 + x2) / cap_w
# # # # #         self._v_frac = 0.5 * (y1 + y2) / cap_h
# # # # #         self._x1_frac, self._y1_frac = x1 / cap_w, y1 / cap_h
# # # # #         self._x2_frac, self._y2_frac = x2 / cap_w, y2 / cap_h

# # # # #     def ground(self, rgb, instruction):
# # # # #         h, w = rgb.shape[0], rgb.shape[1]
# # # # #         bbox = (
# # # # #             self._x1_frac * w,
# # # # #             self._y1_frac * h,
# # # # #             self._x2_frac * w,
# # # # #             self._y2_frac * h,
# # # # #         )
# # # # #         return SimpleNamespace(
# # # # #             u=self._u_frac * w, v=self._v_frac * h, in_view=True, bbox=bbox
# # # # #         )


# # # # # def goal_pixel_ratio(goal_mask: np.ndarray) -> Dict[str, float]:
# # # # #     """Fraction of the current frame occupied by goal-object pixels vs. the rest of the image;
# # # # #     a general per-step metric fed into the rollout/episode logs."""
# # # # #     total_px = int(goal_mask.shape[0] * goal_mask.shape[1])
# # # # #     goal_px = int(np.count_nonzero(goal_mask))
# # # # #     rest_px = total_px - goal_px
# # # # #     return {
# # # # #         "goal_px": goal_px,
# # # # #         "rest_px": rest_px,
# # # # #         "frame_fraction": goal_px / total_px if total_px > 0 else 0.0,
# # # # #         "goal_to_rest_ratio": goal_px / rest_px if rest_px > 0 else float("inf"),
# # # # #     }


# # # # # def propagate_body_point(bg, action, dt, odom_noise=0.0, rng=None):
# # # # #     """Move a body-frame point [forward, left] under the robot's own SE(2) motion (dead-reckoning):
# # # # #     translate back by v*dt and rotate by -yaw*dt -- the same propagation the cone uses. Optional
# # # # #     Gaussian odom noise makes the belief DRIFT, so it must be corrected by sightings to stay good.
# # # # #     """
# # # # #     v_fwd, v_lat, yaw_rate = float(action[0]), float(action[1]), float(action[2])
# # # # #     if odom_noise > 0.0 and rng is not None:
# # # # #         v_fwd += float(rng.normal(0.0, odom_noise))
# # # # #         yaw_rate += float(rng.normal(0.0, odom_noise))
# # # # #     th = -yaw_rate * dt
# # # # #     c, s = math.cos(th), math.sin(th)
# # # # #     qx = float(bg[0]) - v_fwd * dt
# # # # #     qy = float(bg[1]) - v_lat * dt
# # # # #     return np.asarray([c * qx - s * qy, s * qx + c * qy], dtype=np.float32)


# # # # # def paint_obstacle_map_point(
# # # # #     obstacle_map: np.ndarray,
# # # # #     builder,
# # # # #     point_forward_left: Optional[np.ndarray],
# # # # #     radius_cells: int,
# # # # # ) -> np.ndarray:
# # # # #     out = np.asarray(obstacle_map, dtype=np.float32).copy()
# # # # #     if point_forward_left is None:
# # # # #         return out
# # # # #     p = np.asarray(point_forward_left, dtype=np.float32).reshape(-1)
# # # # #     if p.size < 2 or not np.isfinite(p[:2]).all():
# # # # #         return out
# # # # #     rows, cols = builder.world_to_grid(
# # # # #         np.asarray([p[0]], dtype=np.float32), np.asarray([p[1]], dtype=np.float32)
# # # # #     )
# # # # #     r = int(rows[0])
# # # # #     c = int(cols[0])
# # # # #     rad = max(int(radius_cells), 0)
# # # # #     h, w = out.shape
# # # # #     for rr in range(max(0, r - rad), min(h, r + rad + 1)):
# # # # #         for cc in range(max(0, c - rad), min(w, c + rad + 1)):
# # # # #             if (rr - r) ** 2 + (cc - c) ** 2 <= rad**2:
# # # # #                 out[rr, cc] = 1.0
# # # # #     return out


# # # # # def depth_obstacle_mask(
# # # # #     depth: np.ndarray, threshold: float, min_y_frac: float
# # # # # ) -> np.ndarray:
# # # # #     arr = np.asarray(depth, dtype=np.float32)
# # # # #     h, _ = arr.shape
# # # # #     yy = np.arange(h)[:, None]
# # # # #     mask = (
# # # # #         np.isfinite(arr)
# # # # #         & (arr > 0.0)
# # # # #         & (arr < float(threshold))
# # # # #         & (yy >= h * float(min_y_frac))
# # # # #     )
# # # # #     return mask.astype(np.uint8)


# # # # # def overlay_frame(
# # # # #     rgb: np.ndarray, goal_mask: np.ndarray, obstacle_mask: np.ndarray, text: str
# # # # # ) -> Image.Image:
# # # # #     img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
# # # # #     overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
# # # # #     pix = np.asarray(overlay).copy()
# # # # #     gm = np.asarray(goal_mask) > 0
# # # # #     om = np.asarray(obstacle_mask) > 0
# # # # #     pix[gm] = [0, 255, 0, 120]
# # # # #     pix[om] = [255, 0, 0, 100]
# # # # #     overlay = Image.fromarray(pix, mode="RGBA")
# # # # #     img = Image.alpha_composite(img.convert("RGBA"), overlay)
# # # # #     draw = ImageDraw.Draw(img)
# # # # #     draw.rectangle([0, 0, img.width, 46], fill=(0, 0, 0, 170))
# # # # #     draw.text((8, 6), text, fill=(255, 255, 255, 255))
# # # # #     return img.convert("RGB")


# # # # # def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
# # # # #     try:
# # # # #         import imageio.v2 as imageio
# # # # #     except Exception as exc:
# # # # #         print(f"[WARN] imageio unavailable; skipping video: {exc}", flush=True)
# # # # #         return
# # # # #     if not frames:
# # # # #         return
# # # # #     path.parent.mkdir(parents=True, exist_ok=True)
# # # # #     imageio.mimsave(path, [np.asarray(f) for f in frames], fps=float(fps))


# # # # # _LOG_SENTINEL = object()


# # # # # def _log_json_default(obj):
# # # # #     """Tolerate numpy scalars/arrays showing up in logged payloads -- the rollout loop this feeds
# # # # #     is numpy-heavy, and a stray np.float32 buried in a nested dict/list should not crash logging
# # # # #     mid-episode."""
# # # # #     item = getattr(obj, "item", None)
# # # # #     if callable(item) and hasattr(obj, "shape") and obj.shape == ():
# # # # #         return item()
# # # # #     tolist = getattr(obj, "tolist", None)
# # # # #     if callable(tolist):
# # # # #         return tolist()
# # # # #     raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# # # # # def _log_now_iso() -> str:
# # # # #     return datetime.now(timezone.utc).isoformat()


# # # # # def _log_slugify(value) -> str:
# # # # #     s = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
# # # # #     return s or "na"


# # # # # def make_run_id(config: Dict, timestamp=None) -> str:
# # # # #     """`<timestamp>_<goal_mode>-goal_<steering_mode>_obs<count>_seed<seed>`, so the ablation
# # # # #     condition is identifiable from the folder name alone."""
# # # # #     ts = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
# # # # #     goal_mode = _log_slugify(config.get("goal_mode", "na"))
# # # # #     steering_mode = _log_slugify(config.get("steering_mode", "na"))
# # # # #     obstacle_count = _log_slugify(config.get("obstacle_count", "na"))
# # # # #     seed = _log_slugify(config.get("obstacle_seed", "na"))
# # # # #     return f"{ts}_{goal_mode}-goal_{steering_mode}_obs{obstacle_count}_seed{seed}"


# # # # # class EpisodeLogger:
# # # # #     """Structured, per-episode JSON logging for ablation rollouts. One instance per episode; it
# # # # #     knows nothing about ablation semantics (goal mode, steering mode, CBF, ...) -- it just
# # # # #     persists whatever `config` dict it's constructed with, plus whatever the rollout loop reports
# # # # #     through `log_frame` / `log_qwen_query` / `log_cbf_event`.

# # # # #     Writes are non-blocking: JSON Lines files (frames/qwen_queries/cbf_events) are appended to by
# # # # #     a single background thread pulling off a queue, so `log_*` calls from the physics/render loop
# # # # #     only do cheap bookkeeping (dict construction + a queue.put) and never touch disk themselves.
# # # # #     `config.json`, `obstacles.json`, and `summary.json` are small, one-shot, pretty-printed writes
# # # # #     done directly since they don't recur every step."""

# # # # #     _JSONL_FILES = ("frames", "qwen_queries", "cbf_events")

# # # # #     def __init__(
# # # # #         self, run_id, config, log_root="logs", save_frames=False, flush_interval_s=1.0
# # # # #     ):
# # # # #         self.run_id = run_id
# # # # #         self.run_dir = Path(log_root) / run_id
# # # # #         self.run_dir.mkdir(parents=True, exist_ok=True)

# # # # #         self._save_frames = save_frames
# # # # #         self.frames_dir = self.run_dir / "frames"
# # # # #         if save_frames:
# # # # #             self.frames_dir.mkdir(parents=True, exist_ok=True)

# # # # #         self._start_monotonic = time.monotonic()
# # # # #         self._config = dict(config)
# # # # #         self._config.setdefault("run_id", run_id)
# # # # #         self._config.setdefault("timestamp_start", _log_now_iso())
# # # # #         self._write_json_now("config.json", self._config)

# # # # #         # Stats accumulated synchronously on the caller's thread (cheap, no I/O) so finalize() can
# # # # #         # summarize without waiting on the background writer.
# # # # #         self._total_steps = 0
# # # # #         self._last_distance_to_goal = None
# # # # #         self._closest_approach = {}
# # # # #         self._counts = {"cbf_events": 0, "qwen_queries": 0, "goal_proximity_events": 0}

# # # # #         self._files = {
# # # # #             name: open(self.run_dir / f"{name}.jsonl", "a", encoding="utf-8")
# # # # #             for name in self._JSONL_FILES
# # # # #         }
# # # # #         self._queue = queue.Queue()
# # # # #         self._flush_interval_s = flush_interval_s
# # # # #         self._closed = False
# # # # #         self._worker = threading.Thread(
# # # # #             target=self._drain_loop, name=f"episode-logger-{run_id}", daemon=True
# # # # #         )
# # # # #         self._worker.start()

# # # # #     def _write_json_now(self, filename, obj) -> None:
# # # # #         (self.run_dir / filename).write_text(
# # # # #             json.dumps(obj, indent=2, default=_log_json_default)
# # # # #         )

# # # # #     def _elapsed(self) -> float:
# # # # #         return time.monotonic() - self._start_monotonic

# # # # #     def _enqueue(self, kind, payload) -> None:
# # # # #         self._queue.put((kind, payload))

# # # # #     def _drain_loop(self) -> None:
# # # # #         last_flush = time.monotonic()
# # # # #         while True:
# # # # #             try:
# # # # #                 item = self._queue.get(timeout=self._flush_interval_s)
# # # # #             except queue.Empty:
# # # # #                 item = None

# # # # #             if item is _LOG_SENTINEL:
# # # # #                 self._flush_all()
# # # # #                 break
# # # # #             if item is not None:
# # # # #                 self._write_item(item)

# # # # #             now = time.monotonic()
# # # # #             if now - last_flush >= self._flush_interval_s:
# # # # #                 self._flush_all()
# # # # #                 last_flush = now

# # # # #         for fh in self._files.values():
# # # # #             fh.close()

# # # # #     def _write_item(self, item) -> None:
# # # # #         kind, payload = item
# # # # #         if kind == "image":
# # # # #             step, image = payload
# # # # #             try:
# # # # #                 import imageio.v3 as iio

# # # # #                 iio.imwrite(self.frames_dir / f"frame_{step:06d}.png", image)
# # # # #             except Exception as e:  # pragma: no cover - best-effort side channel
# # # # #                 print(f"EpisodeLogger: failed to save frame {step}: {e}")
# # # # #             return
# # # # #         self._files[kind].write(
# # # # #             json.dumps(payload, separators=(",", ":"), default=_log_json_default) + "\n"
# # # # #         )

# # # # #     def _flush_all(self) -> None:
# # # # #         for fh in self._files.values():
# # # # #             fh.flush()

# # # # #     def write_obstacles(self, obstacle_list, goal_id=None, seed=None) -> None:
# # # # #         payload = {
# # # # #             "seed": seed if seed is not None else self._config.get("obstacle_seed"),
# # # # #             "obstacles": list(obstacle_list),
# # # # #             "goal_id": goal_id,
# # # # #         }
# # # # #         self._write_json_now("obstacles.json", payload)

# # # # #     def write_object_depths(self, object_depths) -> None:
# # # # #         """Per-episode: the first-frame depth-sensor reading at each detected object's bbox
# # # # #         center, from the rover's own start pose -- how far each VLM-flagged object (goal +
# # # # #         obstacles) actually is from the agent's camera, independent of the world-seed
# # # # #         back-projection stored in obstacles.json / the VLM mission metadata."""
# # # # #         self._write_json_now("object_depths.json", {"objects": list(object_depths)})

# # # # #     def log_frame(
# # # # #         self,
# # # # #         step,
# # # # #         position,
# # # # #         orientation,
# # # # #         action,
# # # # #         distances_to_obstacles=None,
# # # # #         cbf_active=False,
# # # # #         goal_belief=None,
# # # # #         distance_to_goal=None,
# # # # #         yaw_to_goal=None,
# # # # #     ) -> None:
# # # # #         distances_to_obstacles = dict(distances_to_obstacles or {})
# # # # #         if distance_to_goal is None and goal_belief is not None:
# # # # #             distance_to_goal = goal_belief.get("range")
# # # # #         if yaw_to_goal is None and goal_belief is not None:
# # # # #             yaw_to_goal = goal_belief.get("bearing")

# # # # #         entry = {
# # # # #             "step": step,
# # # # #             "t": self._elapsed(),
# # # # #             "position": list(position),
# # # # #             "orientation": list(orientation),
# # # # #             "action": action,
# # # # #             "distance_to_goal": distance_to_goal,
# # # # #             "yaw_to_goal": yaw_to_goal,
# # # # #             "distances_to_obstacles": distances_to_obstacles,
# # # # #             "cbf_active": bool(cbf_active),
# # # # #             "goal_belief": goal_belief,
# # # # #         }
# # # # #         self._enqueue("frames", entry)

# # # # #         self._total_steps = max(self._total_steps, step + 1)
# # # # #         if distance_to_goal is not None:
# # # # #             self._last_distance_to_goal = distance_to_goal
# # # # #         for obs_id, dist in distances_to_obstacles.items():
# # # # #             prev = self._closest_approach.get(obs_id)
# # # # #             if prev is None or dist < prev:
# # # # #                 self._closest_approach[obs_id] = dist

# # # # #     def log_rendered_frame(self, step, image) -> None:
# # # # #         """Save a rendered RGB snapshot to frames/, if the logger was constructed with
# # # # #         save_frames=True; no-op otherwise."""
# # # # #         if not self._save_frames:
# # # # #             return
# # # # #         self._enqueue("image", (step, image))

# # # # #     def log_qwen_query(
# # # # #         self, step, query_type, trigger, input_data, output_data, latency_ms
# # # # #     ) -> None:
# # # # #         entry = {
# # # # #             "step": step,
# # # # #             "t": self._elapsed(),
# # # # #             "query_type": query_type,
# # # # #             "trigger": trigger,
# # # # #             "input": input_data,
# # # # #             "output": output_data,
# # # # #             "latency_ms": latency_ms,
# # # # #         }
# # # # #         self._enqueue("qwen_queries", entry)
# # # # #         self._counts["qwen_queries"] += 1
# # # # #         if trigger == "goal_proximity":
# # # # #             self._counts["goal_proximity_events"] += 1

# # # # #     def log_cbf_event(
# # # # #         self, step, obstacle_id, distance, nominal_action, overridden_action, mode
# # # # #     ) -> None:
# # # # #         entry = {
# # # # #             "step": step,
# # # # #             "t": self._elapsed(),
# # # # #             "obstacle_id": obstacle_id,
# # # # #             "distance": distance,
# # # # #             "nominal_action": nominal_action,
# # # # #             "overridden_action": overridden_action,
# # # # #             "mode": mode,
# # # # #         }
# # # # #         self._enqueue("cbf_events", entry)
# # # # #         self._counts["cbf_events"] += 1
# # # # #         prev = self._closest_approach.get(obstacle_id)
# # # # #         if prev is None or distance < prev:
# # # # #             self._closest_approach[obstacle_id] = distance

# # # # #     def finalize(self, summary=None) -> Dict:
# # # # #         """Write summary.json (caller-provided fields win over computed ones) and shut down the
# # # # #         background writer. Safe to call at most once."""
# # # # #         computed = {
# # # # #             "run_id": self.run_id,
# # # # #             "timestamp_end": _log_now_iso(),
# # # # #             "total_steps": self._total_steps,
# # # # #             "final_distance_to_goal": self._last_distance_to_goal,
# # # # #             "closest_approach_per_obstacle": dict(self._closest_approach),
# # # # #             "num_cbf_interventions": self._counts["cbf_events"],
# # # # #             "num_qwen_queries": self._counts["qwen_queries"],
# # # # #             "num_goal_proximity_events": self._counts["goal_proximity_events"],
# # # # #         }
# # # # #         computed.update(summary or {})
# # # # #         self._write_json_now("summary.json", computed)
# # # # #         self.close()
# # # # #         return computed

# # # # #     def close(self) -> None:
# # # # #         if self._closed:
# # # # #             return
# # # # #         self._closed = True
# # # # #         self._queue.put(_LOG_SENTINEL)
# # # # #         self._worker.join(timeout=10.0)

# # # # #     def __enter__(self) -> "EpisodeLogger":
# # # # #         return self

# # # # #     def __exit__(self, exc_type, exc, tb) -> bool:
# # # # #         if not self._closed:
# # # # #             self.finalize(
# # # # #                 {
# # # # #                     "success": False,
# # # # #                     "termination_reason": "exception" if exc_type else "unfinalized",
# # # # #                 }
# # # # #             )
# # # # #         return False


# # # # # def _parse_obstacle_count(value: str):
# # # # #     """argparse type for --obstacles: 'single' (1), 'all' (every non-goal detection,
# # # # #     unlimited), or a positive integer count."""
# # # # #     if value in ("single", "all"):
# # # # #         return value
# # # # #     try:
# # # # #         n = int(value)
# # # # #     except ValueError:
# # # # #         raise argparse.ArgumentTypeError(
# # # # #             f"--obstacles must be 'single', 'all', or a positive integer (got {value!r})"
# # # # #         )
# # # # #     if n < 1:
# # # # #         raise argparse.ArgumentTypeError("--obstacles integer value must be >= 1")
# # # # #     return n


# # # # # def resolve_obstacle_limit(obstacles) -> Optional[int]:
# # # # #     """Map the --obstacles flag's parsed value to the obstacle_limit resolve_vlm_selection
# # # # #     expects: None = every VLM-flagged obstacle except the goal ('all'), else a max count.
# # # # #     """
# # # # #     if obstacles == "all":
# # # # #         return None
# # # # #     if obstacles == "single":
# # # # #         return 1
# # # # #     return int(obstacles)


# # # # # def main() -> None:
# # # # #     ap = argparse.ArgumentParser(
# # # # #         description="Run a trained NavDP/S2DiT policy inside the Mars HabitatSim terrain."
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--navdp-root",
# # # # #         default=None,
# # # # #         help="Path to the navdp_sam repo containing model_s2_dit.py",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--ckpt", required=True, help="Path to trained NavDP/S2DiT checkpoint"
# # # # #     )
# # # # #     ap.add_argument("--scene", default=str(DEFAULT_SCENE))
# # # # #     ap.add_argument(
# # # # #         "--out",
# # # # #         default=f"rollouts/navdp_rollout{datetime.now().strftime('%Y%m%d_%H%M%S')}",
# # # # #         help="Output dir for rollout frame",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--log-root",
# # # # #         default="logs",
# # # # #         help="Root dir for structured per-episode ablation logs (config/frames/obstacles/"
# # # # #         "qwen_queries/cbf_events/summary); a timestamped run_id subdir is created "
# # # # #         "under it every rollout, separate from --out.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--log-save-frames",
# # # # #         action="store_true",
# # # # #         help="Also dump each step's raw RGB frame under the log run dir's frames/ folder.",
# # # # #     )
# # # # #     ap.add_argument("--device", default="cuda")
# # # # #     ap.add_argument("--weights", choices=["model", "ema"], default="model")
# # # # #     ap.add_argument(
# # # # #         "--cbf-cone-project",
# # # # #         action=argparse.BooleanOptionalAction,
# # # # #         default=True,
# # # # #         help="cone mode: apply project_chunk_cone's soft gradient correction to the "
# # # # #         "sampled chunk before execution. --no-cbf-cone-project disables ONLY this step -- "
# # # # #         "obstacle detection, orbit, and hard-gate all stay exactly as configured -- so this "
# # # # #         "isolates what the cone projection itself adds on top of orbit/hard-gate.",
# # # # #     )
# # # # #     # Compatibility knobs matching scripts/rollout_habitat_policy.py.
# # # # #     ap.add_argument(
# # # # #         "--scene-mode",
# # # # #         default="mars",
# # # # #         help="Accepted for NavDP command compatibility; ignored by Mars adapter.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--obstacle-pool",
# # # # #         default="none",
# # # # #         help="Accepted for NavDP command compatibility; ignored unless ghost/depth obstacles are provided.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--categories",
# # # # #         nargs="*",
# # # # #         default=["chair"],
# # # # #         help="Accepted for command compatibility; the Mars target is set by --goal-x/--goal-z.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--episodes-per-category",
# # # # #         type=int,
# # # # #         default=1,
# # # # #         help="Accepted for command compatibility; Mars adapter runs one rollout.",
# # # # #     )
# # # # #     ap.add_argument("--sample-steps", type=int, default=20)
# # # # #     ap.add_argument("--image-size", type=int, default=None)
# # # # #     ap.add_argument("--height", type=int, default=720)
# # # # #     ap.add_argument("--width", type=int, default=720)
# # # # #     ap.add_argument("--hfov-deg", type=float, default=90.0)
# # # # #     ap.add_argument("--hz", type=float, default=10.0)
# # # # #     ap.add_argument("--max-steps", type=int, default=300)
# # # # #     ap.add_argument("--stop-dist", type=float, default=1.0)
# # # # #     ap.add_argument(
# # # # #         "--vla-dump",
# # # # #         type=str,
# # # # #         default="",
# # # # #         help="If set, dump paired left/right counterfactual VLA training samples "
# # # # #         "(neutral-goal obs + orbit-generated left & right chunks) to this dir at blocked steps.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--vla-dump-every",
# # # # #         type=int,
# # # # #         default=3,
# # # # #         help="Dump one sample per N blocked steps.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--vla-horizon",
# # # # #         type=int,
# # # # #         default=8,
# # # # #         help="Orbit target chunk length (match the policy horizon).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--command",
# # # # #         type=str,
# # # # #         default="",
# # # # #         help="Real-time language command: 'pass left' / 'pass right' / 'stop' / '' (default). "
# # # # #         "Overrides the geometric side choice while an obstacle blocks the path.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--command-file",
# # # # #         type=str,
# # # # #         default="",
# # # # #         help="Path polled every tick for the current command (a human or a VLM writes to it). "
# # # # #         "Overrides --command. This is the LIVE inference interface.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--vla-adapter",
# # # # #         type=str,
# # # # #         default="",
# # # # #         help="Path to a trained vla_adapter.pt. REGIME B: the language-conditioned POLICY "
# # # # #         "produces the maneuver (orbit override + soft cone projection off; hard gate keeps it "
# # # # #         "safe). Without it, the command drives the orbit controller (Regime A).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--vla-alpha-scale",
# # # # #         type=float,
# # # # #         default=1.25,
# # # # #         help="Scale the adapter's language token at inference (ablation showed ~1.25 gives the "
# # # # #         "cleanest instruction-following).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--belief-goal",
# # # # #         action="store_true",
# # # # #         help="Track the goal via BELIEF: seed a body-frame estimate from the goal ONCE, then "
# # # # #         "propagate it by odometry and draw the ghost from the estimate. The known goal touches "
# # # # #         "the system only at t=0 (and, if enabled, to correct on sight) -- no per-frame geometry.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--belief-odom-noise",
# # # # #         type=float,
# # # # #         default=0.0,
# # # # #         help="Gaussian odom noise per step for the belief propagation. 0 = perfect dead-reckoning "
# # # # #         "(numerically equals geometry); >0 makes the belief drift (its value shows under sightings).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--belief-update-on-sight",
# # # # #         action=argparse.BooleanOptionalAction,
# # # # #         default=True,
# # # # #         help="Re-seed the belief from the goal whenever the goal is actually in view (corrects "
# # # # #         "drift). --no-belief-update-on-sight = pure dead-reckoning from the initial mask only.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--goal-bearing-deg",
# # # # #         type=float,
# # # # #         default=None,
# # # # #         help="IMAGE goal (no world xyz): seed the belief from a bearing (+ = right of forward) and "
# # # # #         "--goal-range in the FIRST view, then dead-reckon by odometry. --goal-x/z become only a "
# # # # #         "reference for the success metric, never used by control.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--goal-range",
# # # # #         type=float,
# # # # #         default=8.0,
# # # # #         help="Initial/fallback range (m) for image-grounded goals.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--instruction",
# # # # #         type=str,
# # # # #         default="",
# # # # #         help="Language instruction for VLM goal grounding (with --grounder).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--grounder",
# # # # #         choices=["none", "stub", "qwen"],
# # # # #         default="none",
# # # # #         help="GROUNDED GOAL: point the goal from RGB+instruction. stub=fixed pixel (test the wiring), "
# # # # #         "qwen=Qwen2.5-VL zero-shot. Implies --belief-goal; the pixel seeds the belief, odometry tracks it.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--grounder-every",
# # # # #         type=int,
# # # # #         default=15,
# # # # #         help="Re-ground every N steps (odometry tracks between).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--grounder-uv",
# # # # #         type=str,
# # # # #         default="0.5,0.5",
# # # # #         help="stub grounder pixel as fraction 'fx,fy'.",
# # # # #     )
# # # # #     ap.add_argument("--grounder-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
# # # # #     ap.add_argument("--start-x", type=float, default=0.0)
# # # # #     ap.add_argument("--start-z", type=float, default=8.0)
# # # # #     ap.add_argument("--start-yaw-deg", type=float, default=0.0)
# # # # #     ap.add_argument(
# # # # #         "--goal-x",
# # # # #         type=float,
# # # # #         default=None,
# # # # #         help="World goal X (required unless --goal-mesh-uv/--goal-from-vlm).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--goal-z",
# # # # #         type=float,
# # # # #         default=None,
# # # # #         help="World goal Z (required unless --goal-mesh-uv/--goal-from-vlm).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--goal-from-vlm",
# # # # #         action="store_true",
# # # # #         help="BELIEF-tracked goal (mode b): resolve a VLM object selection and seed the "
# # # # #         "belief via the same grounder pixel path as --grounder stub/qwen (implies "
# # # # #         "--belief-goal). DEFAULT: the frame is captured LIVE from this rollout's own "
# # # # #         "start pose and auto-annotated with SAM -- no pre-existing files needed. Pass "
# # # # #         "--manual-annotate to instead use a pre-existing, already-annotated frame from "
# # # # #         "vlm_nav_interactive's capture session. The resolved world position is kept "
# # # # #         "only as a logging/success-metric reference, like --goal-bearing-deg -- never "
# # # # #         "fed to control.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--manual-annotate",
# # # # #         action="store_true",
# # # # #         help="With --goal-from-vlm: use a pre-existing, manually labelme-annotated frame "
# # # # #         "(vlm_nav_interactive's OUT_DIR/ANNOTATIONS_DIR, keyed by --vlm-frame-idx) "
# # # # #         "instead of the default live-capture+SAM path. Warns if --start-x/z/yaw-deg "
# # # # #         "differ from the pose that frame was captured at, since the annotated bbox is "
# # # # #         "a fixed pixel fraction only valid for that pose.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--vlm-frame-idx",
# # # # #         type=int,
# # # # #         default=0,
# # # # #         help="Frame index for --goal-from-vlm. With --manual-annotate, selects the "
# # # # #         "pre-captured/annotated frame to load; by default (SAM live-capture), it "
# # # # #         "just names the live-captured frame's output/annotation files.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--obstacles",
# # # # #         type=_parse_obstacle_count,
# # # # #         default="all",
# # # # #         help="With --goal-from-vlm: how many of the VLM's flagged obstacles to register. "
# # # # #         "'single' = just the first one, 'all' = every detected object except the goal "
# # # # #         "(default), or an integer N = up to N of them. Obstacle selection runs "
# # # # #         "separately from (and after) goal selection: candidates are drawn from every "
# # # # #         "detection other than the chosen goal object, which can never itself be "
# # # # #         "selected as an obstacle.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--goal-y",
# # # # #         type=float,
# # # # #         default=None,
# # # # #         help="World Y of goal marker; default terrain height + goal-height",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--goal-height",
# # # # #         type=float,
# # # # #         default=1.2,
# # # # #         help="Goal marker height above terrain when --goal-y is omitted",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--goal-terrain-radius",
# # # # #         type=float,
# # # # #         default=0.8,
# # # # #         help="Raise ghost goal from local max terrain height in this radius",
# # # # #     )
# # # # #     ap.add_argument("--goal-radius", type=int, default=18)
# # # # #     ap.add_argument("--no-clamp-goal-to-edge", action="store_true")
# # # # #     ap.add_argument(
# # # # #         "--goal-mesh-uv",
# # # # #         type=str,
# # # # #         default=None,
# # # # #         help="RENDERED-MASK goal: place a semantic mesh at this first-frame pixel fraction "
# # # # #         "'fu,fv'; each step the mask is rendered and the belief is built from it "
# # # # #         "(the policy's goal channel IS the mask). Enables mesh mode; no --goal-x needed.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--obstacle-mesh-uv",
# # # # #         type=str,
# # # # #         default=None,
# # # # #         help="Optional: place a raised obstacle mesh at this pixel fraction; auto-enables "
# # # # #         "--obstacle-mode depth so the cone avoids it.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--mesh-half-px",
# # # # #         type=int,
# # # # #         default=26,
# # # # #         help="half-size (px) of the pixel window per patch mesh",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--mesh-obstacle-lift",
# # # # #         type=float,
# # # # #         default=0.5,
# # # # #         help="raise the obstacle mesh so depth sees it",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--belief-adapter",
# # # # #         type=str,
# # # # #         default=None,
# # # # #         help="trained belief-return adapter (belief_adapter.pt). When the goal is OFF-SCREEN "
# # # # #         "the belief token drives the POLICY back to it -- replaces the P-controller.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--lang-turn-hyst",
# # # # #         type=float,
# # # # #         default=0.6,
# # # # #         help="extra distance beyond cbf-d-safe+cbf-deadzone before the near-obstacle "
# # # # #         "maneuver gate releases (hysteresis; stops it flicking on/off at the boundary).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--belief-reacquire-px",
# # # # #         type=int,
# # # # #         default=None,
# # # # #         help="goal-pixel count above which the belief-return gate releases (hysteresis); "
# # # # #         "default 3x --lost-goal-min-px.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--dwa",
# # # # #         action="store_true",
# # # # #         help="ABLATION BASELINE: classic Dynamic Window Approach REPLACES the diffusion "
# # # # #         "policy's action + the collision cone entirely (a from-scratch reactive "
# # # # #         "planner with no learned prior). Disables cone/orbit/hard-gate/ghost-assist.",
# # # # #     )
# # # # #     ap.add_argument("--dwa-v-samples", type=int, default=9)
# # # # #     ap.add_argument("--dwa-w-samples", type=int, default=15)
# # # # #     ap.add_argument(
# # # # #         "--dwa-predict-time",
# # # # #         type=float,
# # # # #         default=1.5,
# # # # #         help="forward-simulation horizon (s)",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--dwa-max-accel",
# # # # #         type=float,
# # # # #         default=1.0,
# # # # #         help="max forward acceleration (m/s^2)",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--dwa-max-yaw-accel",
# # # # #         type=float,
# # # # #         default=3.0,
# # # # #         help="max yaw acceleration (rad/s^2)",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--dwa-obstacle-radius",
# # # # #         type=float,
# # # # #         default=0.9,
# # # # #         help="hard-reject candidates closer than this (m)",
# # # # #     )
# # # # #     ap.add_argument("--dwa-heading-weight", type=float, default=1.0)
# # # # #     ap.add_argument("--dwa-clearance-weight", type=float, default=2.0)
# # # # #     ap.add_argument("--dwa-velocity-weight", type=float, default=0.5)
# # # # #     ap.add_argument(
# # # # #         "--terrain-height-mode",
# # # # #         choices=["auto", "heightmap", "obj", "flat"],
# # # # #         default="auto",
# # # # #     )
# # # # #     ap.add_argument("--heightmap", default=None)
# # # # #     ap.add_argument("--terrain-obj", default=str(DEFAULT_OBJ))
# # # # #     ap.add_argument("--flat-y", type=float, default=0.0)
# # # # #     ap.add_argument("--clearance", type=float, default=1.4)
# # # # #     ap.add_argument(
# # # # #         "--pose-terrain-radius",
# # # # #         type=float,
# # # # #         default=0.8,
# # # # #         help="Use local max terrain height around rover footprint before adding clearance",
# # # # #     )
# # # # #     ap.add_argument("--size-x", type=float, default=SIZE_X)
# # # # #     ap.add_argument("--size-z", type=float, default=SIZE_Z)
# # # # #     ap.add_argument("--size-y", type=float, default=SIZE_Y)
# # # # #     ap.add_argument("--flip-heightmap-x", action="store_true")
# # # # #     ap.add_argument(
# # # # #         "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
# # # # #     )
# # # # #     ap.add_argument("--swap-heightmap-xz", action="store_true")
# # # # #     ap.add_argument(
# # # # #         "--scene-height-flip-x", action=argparse.BooleanOptionalAction, default=False
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--scene-height-flip-z",
# # # # #         action=argparse.BooleanOptionalAction,
# # # # #         default=True,
# # # # #         help="Mirror Habitat scene Z before terrain-height lookup; matches the Mars GLB export",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--scene-height-swap-xz", action=argparse.BooleanOptionalAction, default=False
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--habitat-proprio-mode", choices=["pose7", "planar3", "zero"], default=None
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--habitat-action-mode",
# # # # #         choices=["action3d", "action2d", "waypoint"],
# # # # #         default=None,
# # # # #     )
# # # # #     ap.add_argument("--habitat-yaw-axis", choices=["x", "y", "z"], default=None)
# # # # #     ap.add_argument(
# # # # #         "--habitat-use-obstacle-channel",
# # # # #         action=argparse.BooleanOptionalAction,
# # # # #         default=None,
# # # # #     )
# # # # #     ap.add_argument("--obstacle-mode", choices=["none", "depth"], default="none")
# # # # #     ap.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
# # # # #     ap.add_argument("--obstacle-min-y-frac", type=float, default=0.45)
# # # # #     ap.add_argument(
# # # # #         "--ghost-obstacle-x",
# # # # #         type=float,
# # # # #         default=None,
# # # # #         help="Optional world X for a synthetic/ghost obstacle mask.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--ghost-obstacle-z",
# # # # #         type=float,
# # # # #         default=None,
# # # # #         help="Optional world Z for a synthetic/ghost obstacle mask.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--ghost-obstacle-y",
# # # # #         type=float,
# # # # #         default=None,
# # # # #         help="World Y of ghost obstacle marker; default terrain height + ghost-obstacle-height.",
# # # # #     )
# # # # #     ap.add_argument("--ghost-obstacle-height", type=float, default=0.45)
# # # # #     ap.add_argument(
# # # # #         "--ghost-obstacle-radius",
# # # # #         type=int,
# # # # #         default=24,
# # # # #         help="Pixel radius for the synthetic obstacle mask.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--ghost-obstacle-map-radius",
# # # # #         type=int,
# # # # #         default=4,
# # # # #         help="Radius in 96x96 obstacle-map cells for the ghost obstacle.",
# # # # #     )
# # # # #     ap.add_argument("--no-clamp-obstacle-to-edge", action="store_true")
# # # # #     ap.add_argument(
# # # # #         "--zero-lateral", action=argparse.BooleanOptionalAction, default=True
# # # # #     )
# # # # #     ap.add_argument("--max-forward-speed", type=float, default=1.0)
# # # # #     ap.add_argument("--max-lateral-speed", type=float, default=1.0)
# # # # #     ap.add_argument("--max-yaw-rate", type=float, default=1.0)
# # # # #     ap.add_argument(
# # # # #         "--action-smoothing", choices=["ensemble", "ema", "none"], default="none"
# # # # #     )
# # # # #     ap.add_argument("--ensemble-decay", type=float, default=0.5)
# # # # #     ap.add_argument("--ema-alpha", type=float, default=0.6)
# # # # #     ap.add_argument("--cbf", action="store_true")
# # # # #     ap.add_argument("--cbf-mode", choices=["project", "cone"], default="cone")
# # # # #     ap.add_argument("--cbf-d-safe", type=float, default=0.75)
# # # # #     ap.add_argument("--cbf-gamma", type=float, default=0.3)
# # # # #     ap.add_argument("--cbf-deadzone", type=float, default=0.6)
# # # # #     ap.add_argument("--cbf-proj-iters", type=int, default=15)
# # # # #     ap.add_argument("--cbf-proj-lr", type=float, default=0.08)
# # # # #     ap.add_argument("--cbf-cone-margin", type=float, default=0.05)
# # # # #     ap.add_argument("--cbf-trust", type=float, default=0.3)
# # # # #     ap.add_argument("--cbf-smooth", type=float, default=0.0)
# # # # #     ap.add_argument("--cbf-keep-speed", type=float, default=1.0)
# # # # #     ap.add_argument(
# # # # #         "--cbf-metric", choices=["euclidean", "mahalanobis"], default="euclidean"
# # # # #     )
# # # # #     ap.add_argument("--cbf-cov-base", type=float, default=1.0)
# # # # #     ap.add_argument("--cbf-cov-growth", type=float, default=0.6)
# # # # #     ap.add_argument(
# # # # #         "--cbf-cov-mode", choices=["grow", "flat", "shrink"], default="shrink"
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--cbf-radius-mode", choices=["fixed", "perceived"], default="fixed"
# # # # #     )
# # # # #     ap.add_argument("--robot-radius", type=float, default=0.25)
# # # # #     ap.add_argument("--safety-margin", type=float, default=0.15)
# # # # #     ap.add_argument("--ghost-obstacle-world-radius", type=float, default=0.25)
# # # # #     # --- ported safety layer (per-tick hard gate + escape yaw + committed side) ---
# # # # #     ap.add_argument(
# # # # #         "--cbf-hard-gate",
# # # # #         action=argparse.BooleanOptionalAction,
# # # # #         default=True,
# # # # #         help="cone mode: re-check the FINAL executed action every tick against the obstacle "
# # # # #         "and brake forward if it would breach. The soft chunk projection alone is diluted by "
# # # # #         "the smoother / skipped between replans -> not safe without this.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--cbf-escape-yaw",
# # # # #         type=float,
# # # # #         default=0.6,
# # # # #         help="cone mode: ENABLE tangent-point pursuit around the obstacle (any value >0 turns it "
# # # # #         "on; 0=off, fall back to the plain distance brake). The turn rate itself is computed by "
# # # # #         "the pursuit law and capped at --max-yaw-rate, so the magnitude here is just the switch.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--cbf-pursuit-kp",
# # # # #         type=float,
# # # # #         default=1.8,
# # # # #         help="cone mode: proportional gain from tangent heading error to yaw-rate for the smooth "
# # # # #         "pursuit. Higher = turns onto the tangent sooner (crisper); too high can overshoot.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--cbf-orbit-kr",
# # # # #         type=float,
# # # # #         default=0.8,
# # # # #         help="cone mode: radial pull-back gain (rad/m) onto the d_safe circle. The orbit law is "
# # # # #         "tangential heading + this*(dist - d_safe); it settles ON the circle instead of the "
# # # # #         "asin-tangent's bounce. 0 = pure tangential (twitchy when hugging tight).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--cbf-orbit-hyst",
# # # # #         type=float,
# # # # #         default=0.4,
# # # # #         help="cone mode: extra clearance (m) required to LEAVE the orbit once committed. Hysteresis "
# # # # #         "on the orbit<->goal switch so it cannot rapid-toggle at the boundary (chatter).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--cbf-goaround-forward",
# # # # #         type=float,
# # # # #         default=0.5,
# # # # #         help="cone mode: constant cruise speed (m/s) while skirting the obstacle with tangent "
# # # # #         "pursuit. Keep <= max-yaw-rate * d_safe so the circle stays trackable (1.0*1.2=1.2 here).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--cbf-commit-side",
# # # # #         action=argparse.BooleanOptionalAction,
# # # # #         default=True,
# # # # #         help="cone mode: hold the go-around side while the obstacle stays in view instead of "
# # # # #         "recomputing sign(p_lat) every replan (which dithers -> yaw stutter).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--lost-goal-ghost",
# # # # #         action="store_true",
# # # # #         help="Steer toward the known ghost goal (proportional heading assist) when it drifts to/past the frame edge, where the mask-conditioned policy only steers weakly.",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--lost-goal-min-px",
# # # # #         type=int,
# # # # #         default=10,
# # # # #         help="Goal-mask pixels below this count means the goal is behind us (mask empty) -> pivot recovery.",
# # # # #     )
# # # # #     ap.add_argument("--lost-goal-turn-kp", type=float, default=1.4)
# # # # #     ap.add_argument(
# # # # #         "--lost-goal-forward",
# # # # #         type=float,
# # # # #         default=0.0,
# # # # #         help="Forward speed floor during recovery. When the goal is merely off to the side we keep the policy's forward and only override yaw; this floor applies when the goal is fully behind (pivot).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--lost-goal-bearing-deg",
# # # # #         type=float,
# # # # #         default=30.0,
# # # # #         help="Engage the proportional heading assist once |goal bearing| exceeds this angle. The ghost is clamped to the frame edge beyond ~hfov/2 (=45deg at hfov 90), where the policy's yaw response saturates weakly; a value below hfov/2 kicks the strong turn in just before the edge. 0 disables the angle trigger (mask-empty only).",
# # # # #     )
# # # # #     ap.add_argument(
# # # # #         "--replan-every",
# # # # #         type=int,
# # # # #         default=1,
# # # # #         help="Sample a fresh diffusion chunk every N control ticks.",
# # # # #     )
# # # # #     ap.add_argument("--save-every", type=int, default=1)
# # # # #     ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
# # # # #     args = ap.parse_args()

# # # # #     navdp_root = resolve_navdp_root(args.navdp_root)
# # # # #     add_navdp_to_path(navdp_root)

# # # # #     from navdp.data.habitat_route_dataset import (
# # # # #         _empty_belief_tensor,
# # # # #         _proprio_from_pose,
# # # # #     )
# # # # #     from navdp.extensions import (
# # # # #         DepthObstacleMap,
# # # # #         horizon_growth_covariance,
# # # # #         nearest_obstacle_point,
# # # # #         nearest_obstacle_state,
# # # # #         project_chunk_cone,
# # # # #         project_forward_velocity_cbf,
# # # # #     )
# # # # #     from rollout_habitat_policy import (
# # # # #         ActionSmoother,
# # # # #         action_to_control,
# # # # #         frame_to_spatial,
# # # # #         load_model,
# # # # #         resolve_modes,
# # # # #         resolve_obstacle_channel,
# # # # #     )

# # # # #     if args.goal_from_vlm:
# # # # #         from vlm_nav_interactive import (
# # # # #             OUT_DIR as VLM_OUT_DIR,
# # # # #             ANNOTATIONS_DIR as VLM_ANNOTATIONS_DIR,
# # # # #             RGBD_RESOLUTION as VLM_CAPTURE_HW,
# # # # #             START_X as VLM_START_X,
# # # # #             START_Z as VLM_START_Z,
# # # # #             START_YAW_DEG as VLM_START_YAW_DEG,
# # # # #             draw_annotation_overlay,
# # # # #             load_depth_frame as vlm_load_depth_frame,
# # # # #             resolve_vlm_selection,
# # # # #             save_mission_metadata,
# # # # #             save_pose as vlm_save_pose,
# # # # #         )

# # # # #     out_dir = Path(args.out).expanduser().resolve()
# # # # #     out_dir.mkdir(parents=True, exist_ok=True)
# # # # #     frame_dir = out_dir / "frames"
# # # # #     frame_dir.mkdir(parents=True, exist_ok=True)

# # # # #     raw_terrain = TerrainHeight(
# # # # #         mode=args.terrain_height_mode,
# # # # #         heightmap=(
# # # # #             Path(args.heightmap).expanduser().resolve() if args.heightmap else None
# # # # #         ),
# # # # #         obj=Path(args.terrain_obj).expanduser().resolve() if args.terrain_obj else None,
# # # # #         flat_y=args.flat_y,
# # # # #         size_x=args.size_x,
# # # # #         size_z=args.size_z,
# # # # #         size_y=args.size_y,
# # # # #         flip_x=args.flip_heightmap_x,
# # # # #         flip_z=args.flip_heightmap_z,
# # # # #         swap_xz=args.swap_heightmap_xz,
# # # # #     )
# # # # #     terrain = SceneMappedTerrain(
# # # # #         raw_terrain,
# # # # #         flip_x=bool(args.scene_height_flip_x),
# # # # #         flip_z=bool(args.scene_height_flip_z),
# # # # #         swap_xz=bool(args.scene_height_swap_xz),
# # # # #     )

# # # # #     device = args.device
# # # # #     model, train_args = load_model(
# # # # #         Path(args.ckpt).expanduser().resolve(), device, args.weights
# # # # #     )
# # # # #     modes = resolve_modes(args, train_args)
# # # # #     if modes["action_mode"] == "waypoint":
# # # # #         raise ValueError(
# # # # #             "Mars rollout executes velocity actions; use action3d or action2d checkpoint/mode."
# # # # #         )
# # # # #     use_obstacle_channel = resolve_obstacle_channel(args, train_args)
# # # # #     image_size = int(args.image_size or train_args.get("image_size", 224))
# # # # #     intr = intrinsics_from_hfov(args.height, args.width, args.hfov_deg)
# # # # #     obstacle_builder = DepthObstacleMap(camera_intrinsics=intr)
# # # # #     smoother = ActionSmoother(
# # # # #         args.action_smoothing, args.ensemble_decay, args.ema_alpha
# # # # #     )

# # # # #     # REGIME B: load the trained language adapter + text encoder (frozen). The command's text
# # # # #     # token is appended to the policy cond set so the POLICY produces the maneuver.
# # # # #     vla_adapter = None
# # # # #     vla_text_enc = None
# # # # #     vla_tok_cache = {}
# # # # #     belief_adapter = None  # trained belief-return adapter (--belief-adapter); replaces the P-controller
# # # # #     if args.vla_adapter:
# # # # #         from train_vla_adapter import VLAAdapter
# # # # #         from sentence_transformers import SentenceTransformer

# # # # #         _vck = torch.load(args.vla_adapter, map_location=device)
# # # # #         vla_text_enc = SentenceTransformer(_vck["text_encoder"], device=device)
# # # # #         vla_adapter = VLAAdapter(
# # # # #             _vck["text_dim"], _vck["dim"], num_tokens=_vck.get("num_tokens", 1)
# # # # #         ).to(device)
# # # # #         vla_adapter.load_state_dict(_vck["adapter"])
# # # # #         vla_adapter.eval()
# # # # #         print(
# # # # #             f"[VLA] Regime B: policy driven by language adapter {args.vla_adapter} "
# # # # #             f"(alpha_scale={args.vla_alpha_scale}, tokens={vla_adapter.num_tokens})",
# # # # #             flush=True,
# # # # #         )

# # # # #     if args.belief_adapter:
# # # # #         from train_vla_adapter import VLAAdapter

# # # # #         _bck = torch.load(args.belief_adapter, map_location=device)
# # # # #         belief_adapter = VLAAdapter(
# # # # #             _bck["belief_feat_dim"], _bck["dim"], num_tokens=_bck.get("num_tokens", 4)
# # # # #         ).to(device)
# # # # #         belief_adapter.load_state_dict(_bck["adapter"])
# # # # #         belief_adapter.eval()
# # # # #         print(
# # # # #             f"[BELIEF] learned return: the belief token drives the policy when the goal is off-screen "
# # # # #             f"(replaces the P-controller); tokens={belief_adapter.num_tokens}",
# # # # #             flush=True,
# # # # #         )

# # # # #     if args.dwa:
# # # # #         print(
# # # # #             "[DWA] ABLATION BASELINE: classic Dynamic Window Approach replaces the diffusion "
# # # # #             "policy's action + the collision cone entirely (cone/orbit/hard-gate/ghost-assist disabled)",
# # # # #             flush=True,
# # # # #         )

# # # # #     # GROUNDED GOAL: a VLM (or stub) points the goal from RGB + instruction; the pixel seeds the
# # # # #     # belief and odometry tracks it -> language decides WHERE, geometry never sees the world goal.
# # # # #     grounder = None
# # # # #     if args.grounder == "stub":
# # # # #         from navdp.extensions import StubPixelGoal

# # # # #         _uv = tuple(float(x) for x in args.grounder_uv.split(","))
# # # # #         grounder = StubPixelGoal(uv=_uv, as_fraction=True)
# # # # #     elif args.grounder == "qwen":
# # # # #         from navdp.extensions import QwenVLPixelGoal

# # # # #         grounder = QwenVLPixelGoal(model_id=args.grounder_model, device=device)
# # # # #     if grounder is not None:
# # # # #         args.belief_goal = True
# # # # #         print(
# # # # #             f"[VLA] grounded goal: {args.grounder} every {args.grounder_every} steps, "
# # # # #             f"instruction={args.instruction!r}",
# # # # #             flush=True,
# # # # #         )

# # # # #     mesh_goal_mode = bool(args.goal_mesh_uv)
# # # # #     if mesh_goal_mode:
# # # # #         args.belief_goal = (
# # # # #             True  # reuse belief propagation + ghost recovery, but seed from the mask
# # # # #         )
# # # # #         # NOTE: obstacle comes from the rendered obstacle-MESH mask (below), NOT depth thresholding,
# # # # #         # which would flag the whole near ground as an obstacle. Leave --obstacle-mode as-is.
# # # # #         print(
# # # # #             f"[MASK] rendered-mask goal at pixel {args.goal_mesh_uv}"
# # # # #             + (
# # # # #                 f" + obstacle mesh at {args.obstacle_mesh_uv}"
# # # # #                 if args.obstacle_mesh_uv
# # # # #                 else ""
# # # # #             ),
# # # # #             flush=True,
# # # # #         )

# # # # #     sim = make_sim(
# # # # #         Path(args.scene),
# # # # #         args.height,
# # # # #         args.width,
# # # # #         args.hfov_deg,
# # # # #         with_semantic=(mesh_goal_mode or args.goal_from_vlm),
# # # # #     )
# # # # #     agent = sim.initialize_agent(0)

# # # # #     x = float(args.start_x)
# # # # #     z = float(args.start_z)
# # # # #     yaw = math.radians(float(args.start_yaw_deg))
# # # # #     dt = 1.0 / float(args.hz)

# # # # #     cbf_obstacle_id = (
# # # # #         "obstacle"  # identifies the single obstacle the live CBF math tracks, for
# # # # #     )
# # # # #     # cbf_events.jsonl -- refined below once VLM/ghost obstacle resolve
# # # # #     vlm_goal_mesh = None
# # # # #     vlm_obstacle_meshes = []
# # # # #     vlm_mesh_tracking = (
# # # # #         False  # True once the VLM's resolved goal mesh is registered with the
# # # # #     )
# # # # #     # semantic sensor below, so the main loop tracks it from the live
# # # # #     # per-frame mask instead of one-shot pixel + odometry dead-reckoning.
# # # # #     if args.goal_from_vlm:
# # # # #         # SEMANTIC-MASK-tracked goal: resolve the VLM's object selection ONCE here, register its
# # # # #         # already-saved mesh (selected_bbox_to_object_mesh's on-disk .obj) with the semantic sensor,
# # # # #         # and re-render that mesh's mask every step (see MESH_GOAL_ID handling in the main loop) --
# # # # #         # mirrors --goal-mesh-uv's rendered-mask tracking instead of dead-reckoning by odometry alone.
# # # # #         frame_idx = args.vlm_frame_idx
# # # # #         rgb_path = f"{VLM_OUT_DIR}/rgb_{frame_idx:04d}.png"
# # # # #         overlay_path = f"{VLM_OUT_DIR}/rgb_{frame_idx:04d}_at.png"
# # # # #         annotation_path = f"{VLM_ANNOTATIONS_DIR}/rgb_{frame_idx:04d}.json"

# # # # #         if not args.manual_annotate:
# # # # #             # DEFAULT: capture the actual live first frame (RGB + depth + pose) at THIS rollout's
# # # # #             # own start pose -- run after sim/agent exist so it's a real render, not an assumption
# # # # #             # that vlm_nav_interactive.py already produced these files on disk. SAM then annotates
# # # # #             # it in place of a human labelme session; resolve_vlm_selection below is unchanged and
# # # # #             # can't tell the difference between the two annotation sources.
# # # # #             y0 = terrain.local_height_max(
# # # # #                 x, z, float(args.pose_terrain_radius)
# # # # #             ) + float(args.clearance)
# # # # #             set_agent_pose(agent, x, y0, z, yaw)
# # # # #             obs0 = sim.get_sensor_observations()
# # # # #             rgb0, depth0 = rgb_depth(obs0)
# # # # #             os.makedirs(VLM_OUT_DIR, exist_ok=True)
# # # # #             Image.fromarray(rgb0).save(rgb_path)
# # # # #             np.save(
# # # # #                 f"{VLM_OUT_DIR}/depth_{frame_idx:04d}.npy", depth0.astype(np.float32)
# # # # #             )
# # # # #             depth_vis = (np.clip(depth0, 0.0, 10.0) / 10.0 * 255.0).astype(np.uint8)
# # # # #             Image.fromarray(depth_vis).save(f"{VLM_OUT_DIR}/depth_{frame_idx:04d}.png")
# # # # #             vlm_save_pose(frame_idx, x, y0, z, yaw)

# # # # #             from sam_annotation_adapter import sam_frame_to_annotation

# # # # #             annotation_path, sam_valid, sam_status = sam_frame_to_annotation(
# # # # #                 rgb_path, annotation_path
# # # # #             )
# # # # #             if not sam_valid:
# # # # #                 raise SystemExit(
# # # # #                     f"--goal-from-vlm: SAM annotation invalid: {sam_status}"
# # # # #                 )
# # # # #             print(
# # # # #                 f"[SAM] live-captured frame {frame_idx} at this rollout's start pose "
# # # # #                 f"({x:.2f},{z:.2f},{math.degrees(yaw):.1f}deg) -> {annotation_path}",
# # # # #                 flush=True,
# # # # #             )

# # # # #         # ANNOTATED FRAME FOR REFERENCE: draw the (SAM- or labelme-) annotation's labeled boxes
# # # # #         # onto the raw frame and save it to overlay_path.
# # # # #         draw_annotation_overlay(rgb_path, annotation_path, overlay_path)

# # # # #         obstacle_limit = resolve_obstacle_limit(args.obstacles)
# # # # #         vlm_success, vlm_result, vlm_status = resolve_vlm_selection(
# # # # #             rgb_path,
# # # # #             overlay_path,
# # # # #             annotation_path,
# # # # #             frame_idx,
# # # # #             obstacle_limit=obstacle_limit,
# # # # #         )
# # # # #         if not vlm_success:
# # # # #             raise SystemExit(f"--goal-from-vlm: VLM selection failed: {vlm_status}")
# # # # #         vlm_response, vlm_goal_mesh, vlm_obstacle_meshes = vlm_result
# # # # #         register_semantic_mesh(sim, vlm_goal_mesh["mesh_path"], MESH_GOAL_ID)
# # # # #         vlm_mesh_tracking = True
# # # # #         # Kept only as an inert fallback (unused once vlm_mesh_tracking routes through the
# # # # #         # MESH_GOAL_ID branch below) in case the goal mesh ever fails to register.
# # # # #         grounder = VlmSelectionPixelGoal(vlm_goal_mesh["bbox"], VLM_CAPTURE_HW)
# # # # #         args.belief_goal = True
# # # # #         print(
# # # # #             f"[VLM] goal '{vlm_goal_mesh['label']}' mesh={vlm_goal_mesh['mesh_path']} registered as "
# # # # #             f"MESH_GOAL_ID; belief re-derived from the live rendered mask every step (no dead-reckoning "
# # # # #             f"while in view)",
# # # # #             flush=True,
# # # # #         )

# # # # #         # OBSTACLE: the VLM's prompt asks for exactly one goal and marks every OTHER detected
# # # # #         # object as an obstacle; --obstacles (single/all/N) then caps how many of THOSE
# # # # #         # goal-excluded candidates resolve_vlm_selection actually resolved (see
# # # # #         # resolve_obstacle_limit / resolve_mission_meshes' obstacle_limit) -- register all of
# # # # #         # whatever came back under MESH_OBST_ID so the rendered obstacle mask the policy/CBF/DWA
# # # # #         # see is the union of every registered object's footprint. ALSO wire the first one's
# # # # #         # resolved world seed into --ghost-obstacle-x/y/z so the CBF/orbit avoidance's cone-mode
# # # # #         # math (which prefers a single stable world point over the mask to avoid abeam-pass
# # # # #         # flicker) still has one, unless the caller passed an explicit ghost obstacle of their
# # # # #         # own; the mask-based CBF/DWA paths don't depend on this and see every registered mesh.
# # # # #         print(
# # # # #             f"[VLM] --obstacles={args.obstacles}: {len(vlm_obstacle_meshes)} obstacle(s) resolved "
# # # # #             f"(goal excluded)",
# # # # #             flush=True,
# # # # #         )
# # # # #         if vlm_obstacle_meshes:
# # # # #             cbf_obstacle_id = "vlm_obstacle"
# # # # #             for vlm_obstacle_mesh in vlm_obstacle_meshes:
# # # # #                 register_semantic_mesh(
# # # # #                     sim, vlm_obstacle_mesh["mesh_path"], MESH_OBST_ID
# # # # #                 )
# # # # #                 print(
# # # # #                     f"[VLM] obstacle '{vlm_obstacle_mesh['label']}' bbox={vlm_obstacle_mesh['bbox']} "
# # # # #                     f"mesh={vlm_obstacle_mesh['mesh_path']} registered as MESH_OBST_ID",
# # # # #                     flush=True,
# # # # #                 )
# # # # #             if args.ghost_obstacle_x is None and args.ghost_obstacle_z is None:
# # # # #                 obs_vx, obs_vy, obs_vz = vlm_obstacle_meshes[0]["seed_world"]
# # # # #                 args.ghost_obstacle_x = obs_vx
# # # # #                 args.ghost_obstacle_z = obs_vz
# # # # #                 # y is left to the existing terrain-height + --ghost-obstacle-height computation
# # # # #                 # below (unless the caller passed --ghost-obstacle-y explicitly) -- seed_world's y
# # # # #                 # sits at the rock's own surface, not the elevated marker height the rest of this
# # # # #                 # script expects for a ghost obstacle.
# # # # #                 print(
# # # # #                     f"[VLM] ghost world seeded from first obstacle=({obs_vx:.2f},{obs_vy:.2f},"
# # # # #                     f"{obs_vz:.2f}); cone-mode math tracks this point while the mask-based CBF/DWA "
# # # # #                     f"see all {len(vlm_obstacle_meshes)} registered obstacle meshes",
# # # # #                     flush=True,
# # # # #                 )
# # # # #         else:
# # # # #             print(
# # # # #                 "[VLM] no obstacle resolved from the VLM's selection; proceeding without one",
# # # # #                 flush=True,
# # # # #             )

# # # # #         # PER-OBJECT DEPTH: the raw depth-sensor reading at each detected object's bbox center in
# # # # #         # the first frame, from the rover's own start pose -- a literal "how far is this from the
# # # # #         # agent" figure. Distinct from seed_world (a median over the bbox interior used to seed a
# # # # #         # robust world position); this is the single center pixel, matching what a "distance to
# # # # #         # this object" readout would show a human looking at that frame.
# # # # #         vlm_first_frame_depth = vlm_load_depth_frame(frame_idx)
# # # # #         vlm_object_depths = [
# # # # #             {
# # # # #                 "role": vlm_goal_mesh["role"],
# # # # #                 "label": vlm_goal_mesh["label"],
# # # # #                 "bbox": vlm_goal_mesh["bbox"],
# # # # #                 "depth_m": bbox_center_depth(
# # # # #                     vlm_goal_mesh["bbox"], vlm_first_frame_depth
# # # # #                 ),
# # # # #             }
# # # # #         ]
# # # # #         for vlm_obstacle_mesh in vlm_obstacle_meshes:
# # # # #             vlm_object_depths.append(
# # # # #                 {
# # # # #                     "role": vlm_obstacle_mesh["role"],
# # # # #                     "label": vlm_obstacle_mesh["label"],
# # # # #                     "bbox": vlm_obstacle_mesh["bbox"],
# # # # #                     "depth_m": bbox_center_depth(
# # # # #                         vlm_obstacle_mesh["bbox"], vlm_first_frame_depth
# # # # #                     ),
# # # # #                 }
# # # # #             )
# # # # #         print(
# # # # #             f"[VLM] first-frame object depths (bbox-center, agent POV): "
# # # # #             + ", ".join(f"{o['label']}={o['depth_m']}" for o in vlm_object_depths),
# # # # #             flush=True,
# # # # #         )

# # # # #         if args.manual_annotate:
# # # # #             # The bbox is a FIXED pixel fraction from an OFFLINE labelme session (not a live
# # # # #             # re-detection), so it's only valid if the rollout starts from ~the pose that frame
# # # # #             # was captured at. Doesn't apply to the default SAM path above: that frame IS this
# # # # #             # rollout's own start pose, by construction, so it can't be stale.
# # # # #             if (
# # # # #                 abs(args.start_x - VLM_START_X) > 0.5
# # # # #                 or abs(args.start_z - VLM_START_Z) > 0.5
# # # # #                 or abs(args.start_yaw_deg - VLM_START_YAW_DEG) > 5.0
# # # # #             ):
# # # # #                 print(
# # # # #                     f"[WARN] --start-x/z/yaw-deg ({args.start_x},{args.start_z},{args.start_yaw_deg}) "
# # # # #                     f"differ from the capture pose ({VLM_START_X},{VLM_START_Z},{VLM_START_YAW_DEG}); "
# # # # #                     "the seeded pixel may not land on the object in the first live frame.",
# # # # #                     flush=True,
# # # # #                 )

# # # # #     if args.goal_from_vlm:
# # # # #         # World position of the VLM selection, kept ONLY as the success-metric/logging reference
# # # # #         # (mirrors --goal-bearing-deg) -- control is driven by the pixel-seeded belief wired
# # # # #         # above, this world point is never read by the control path.
# # # # #         goal_vx, goal_vy, goal_vz = vlm_goal_mesh["seed_world"]
# # # # #         print(
# # # # #             f"[VLM] goal '{vlm_goal_mesh['label']}' reference world=({goal_vx:.2f},{goal_vy:.2f},{goal_vz:.2f})",
# # # # #             flush=True,
# # # # #         )
# # # # #         goal_y = args.goal_y
# # # # #         if goal_y is None:
# # # # #             goal_y = terrain.local_height_max(
# # # # #                 goal_vx, goal_vz, float(args.goal_terrain_radius)
# # # # #             ) + float(args.goal_height)
# # # # #         goal = np.asarray([goal_vx, goal_y, goal_vz], dtype=np.float32)

# # # # #         # MISSION RECORD: first frame (rgb_path), its SAM-annotated overlay (overlay_path), the
# # # # #         # annotation JSON (annotation_path), and the raw VLM prompt/response are already on disk
# # # # #         # under VLM_OUT_DIR/VLM_ANNOTATIONS_DIR; this adds one consolidated JSON tying the VLM's
# # # # #         # parsed goal+obstacle choice to their resolved world positions/meshes.
# # # # #         save_mission_metadata(
# # # # #             frame_idx,
# # # # #             vlm_response,
# # # # #             vlm_goal_mesh,
# # # # #             vlm_obstacle_meshes,
# # # # #             goal_target_world=goal,
# # # # #         )
# # # # #     elif args.goal_x is None or args.goal_z is None:
# # # # #         if not mesh_goal_mode:
# # # # #             raise SystemExit(
# # # # #                 "Pass --goal-x and --goal-z, --goal-from-vlm, or use --goal-mesh-uv for a rendered-mask goal."
# # # # #             )
# # # # #         goal = np.zeros(
# # # # #             3, dtype=np.float32
# # # # #         )  # placeholder; set from the mesh centroid at step 0
# # # # #     else:
# # # # #         goal_y = args.goal_y
# # # # #         if goal_y is None:
# # # # #             goal_y = terrain.local_height_max(
# # # # #                 float(args.goal_x), float(args.goal_z), float(args.goal_terrain_radius)
# # # # #             ) + float(args.goal_height)
# # # # #         goal = np.asarray(
# # # # #             [float(args.goal_x), float(goal_y), float(args.goal_z)], dtype=np.float32
# # # # #         )

# # # # #     ghost_obstacle = None
# # # # #     if (args.ghost_obstacle_x is None) != (args.ghost_obstacle_z is None):
# # # # #         raise ValueError(
# # # # #             "pass both --ghost-obstacle-x and --ghost-obstacle-z, or neither"
# # # # #         )
# # # # #     if args.ghost_obstacle_x is not None and args.ghost_obstacle_z is not None:
# # # # #         obstacle_y = args.ghost_obstacle_y
# # # # #         if obstacle_y is None:
# # # # #             obstacle_y = terrain.local_height_max(
# # # # #                 float(args.ghost_obstacle_x),
# # # # #                 float(args.ghost_obstacle_z),
# # # # #                 float(args.pose_terrain_radius),
# # # # #             ) + float(args.ghost_obstacle_height)
# # # # #         ghost_obstacle = np.asarray(
# # # # #             [
# # # # #                 float(args.ghost_obstacle_x),
# # # # #                 float(obstacle_y),
# # # # #                 float(args.ghost_obstacle_z),
# # # # #             ],
# # # # #             dtype=np.float32,
# # # # #         )
# # # # #         if cbf_obstacle_id == "obstacle":
# # # # #             cbf_obstacle_id = "ghost"

# # # # #     # Structured per-episode ablation logs (separate from the --out npz/manifest dump above):
# # # # #     # config.json / obstacles.json up front, frames.jsonl / qwen_queries.jsonl / cbf_events.jsonl
# # # # #     # appended to every step, summary.json once at the end.
# # # # #     goal_mode = (
# # # # #         "vlm" if args.goal_from_vlm else ("uv" if args.goal_mesh_uv else "coord")
# # # # #     )
# # # # #     log_config = {
# # # # #         "scene_glb": str(Path(args.scene).expanduser().resolve()),
# # # # #         "goal_mode": goal_mode,
# # # # #         "goal_coord": (
# # # # #             [float(args.goal_x), float(args.goal_z)]
# # # # #             if (args.goal_x is not None and args.goal_z is not None)
# # # # #             else None
# # # # #         ),
# # # # #         "steering_mode": "none",
# # # # #         "cbf_enabled": bool(args.cbf),
# # # # #         "obstacle_count": 1 if ghost_obstacle is not None else 0,
# # # # #         "obstacle_seed": None,
# # # # #         "obstacle_distance_threshold_X": float(args.cbf_d_safe),
# # # # #         "goal_distance_threshold_Y": float(args.stop_dist),
# # # # #         "max_steps": int(args.max_steps),
# # # # #         "agent_height_offset": float(args.clearance),
# # # # #     }
# # # # #     run_id = make_run_id(log_config)
# # # # #     episode_logger = EpisodeLogger(
# # # # #         run_id, log_config, log_root=args.log_root, save_frames=args.log_save_frames
# # # # #     )
# # # # #     if ghost_obstacle is not None:
# # # # #         episode_logger.write_obstacles(
# # # # #             [
# # # # #                 {
# # # # #                     "id": "ghost",
# # # # #                     "position": [
# # # # #                         float(ghost_obstacle[0]),
# # # # #                         float(ghost_obstacle[1]),
# # # # #                         float(ghost_obstacle[2]),
# # # # #                     ],
# # # # #                     "orientation": [0.0, 0.0, 0.0, 1.0],
# # # # #                     "radius": float(args.ghost_obstacle_world_radius),
# # # # #                     "is_goal": False,
# # # # #                 }
# # # # #             ],
# # # # #             goal_id=("vlm_goal" if args.goal_from_vlm else None),
# # # # #         )
# # # # #     else:
# # # # #         episode_logger.write_obstacles(
# # # # #             [], goal_id=("vlm_goal" if args.goal_from_vlm else None)
# # # # #         )
# # # # #     if args.goal_from_vlm:
# # # # #         episode_logger.write_object_depths(vlm_object_depths)
# # # # #     termination_reason = "timeout"

# # # # #     rows = {
# # # # #         k: []
# # # # #         for k in [
# # # # #             "rgb",
# # # # #             "depth",
# # # # #             "goal_mask",
# # # # #             "obstacle_mask",
# # # # #             "seg_masks",
# # # # #             "pose",
# # # # #             "proprio",
# # # # #             "action_3d",
# # # # #             "pred_chunk",
# # # # #             "goal_visible_pixels",
# # # # #             "goal_u",
# # # # #             "goal_v",
# # # # #             "goal_distance",
# # # # #             "obstacle_visible_pixels",
# # # # #             "obstacle_u",
# # # # #             "obstacle_v",
# # # # #             "obstacle_distance",
# # # # #             "belief_fwd",
# # # # #             "belief_left",  # body-frame belief_g each tick (nan if not tracking) -> lets
# # # # #             "goal_frame_fraction",  # goal_px / total_px this tick, from goal_pixel_ratio()
# # # # #             "cone_correction_step0",
# # # # #             "cone_correction_last",
# # # # #             "hard_gate_tick",  # Euclidean vs
# # # # #         ]
# # # # #     }  # Mahalanobis mechanistic ablation (see project_chunk_cone)
# # # # #     video_frames = []
# # # # #     prev_obstacle_point = None
# # # # #     last_pred_chunk = None
# # # # #     chunk_len = 0
# # # # #     replan_every = max(int(args.replan_every), 1)
# # # # #     mesh_tracking_mode = (
# # # # #         mesh_goal_mode or vlm_mesh_tracking
# # # # #     )  # goal (+ obstacle, if resolved) tracked
# # # # #     # from the live rendered semantic mask each step, not dead-reckoned
# # # # #     cbf_active = 0
# # # # #     cone_side_latch = (
# # # # #         None  # committed cone-projection side while the obstacle is in view
# # # # #     )
# # # # #     around_side = None  # committed tangent-pursuit side (+1 = pass on obstacle's left)
# # # # #     hard_gate_fired = 0
# # # # #     escape_active = 0
# # # # #     vla_count = 0  # counter for --vla-dump paired-sample writing
# # # # #     belief_g = (
# # # # #         None  # body-frame [forward, left] belief estimate of the goal (--belief-goal)
# # # # #     )
# # # # #     belief_rng = np.random.default_rng(0)
# # # # #     near_obstacle_state = (
# # # # #         False  # hysteresis-latched maneuver gate (avoids flicker at the boundary)
# # # # #     )
# # # # #     belief_state = False  # hysteresis-latched belief-return gate
# # # # #     dwa_prev_v, dwa_prev_w = (
# # # # #         0.0,
# # # # #         0.0,
# # # # #     )  # DWA's own dynamic-window state (--dwa ablation baseline)
# # # # #     cone_correction_step0 = float(
# # # # #         "nan"
# # # # #     )  # ||corrected - raw|| on the EXECUTED step (mechanistic
# # # # #     cone_correction_last = float(
# # # # #         "nan"
# # # # #     )  # ablation metric: Euclidean vs Mahalanobis correction spread)

# # # # #     print("Mars NavDP rollout", flush=True)
# # # # #     print(f"  navdp_root : {navdp_root}", flush=True)
# # # # #     print(f"  scene      : {Path(args.scene).expanduser().resolve()}", flush=True)
# # # # #     print(f"  ckpt       : {Path(args.ckpt).expanduser().resolve()}", flush=True)
# # # # #     print(
# # # # #         f"  terrain    : {terrain.mode} scene_flip_x={args.scene_height_flip_x} "
# # # # #         f"scene_flip_z={args.scene_height_flip_z} scene_swap_xz={args.scene_height_swap_xz}",
# # # # #         flush=True,
# # # # #     )
# # # # #     print(f"  goal       : x={goal[0]:.2f} y={goal[1]:.2f} z={goal[2]:.2f}", flush=True)
# # # # #     if (
# # # # #         args.scene_mode != "mars"
# # # # #         or args.obstacle_pool != "none"
# # # # #         or args.episodes_per_category != 1
# # # # #     ):
# # # # #         print(
# # # # #             "  compat    : scene/category generator flags were accepted but Mars runs one explicit scene",
# # # # #             flush=True,
# # # # #         )
# # # # #     if args.lost_goal_ghost:
# # # # #         print(
# # # # #             f"  ghost     : lost-goal recovery enabled min_px={args.lost_goal_min_px} "
# # # # #             f"turn_kp={args.lost_goal_turn_kp:g} forward={args.lost_goal_forward:g}",
# # # # #             flush=True,
# # # # #         )
# # # # #     if ghost_obstacle is not None:
# # # # #         print(
# # # # #             f"  obstacle   : ghost x={ghost_obstacle[0]:.2f} "
# # # # #             f"y={ghost_obstacle[1]:.2f} z={ghost_obstacle[2]:.2f}",
# # # # #             flush=True,
# # # # #         )
# # # # #     print(
# # # # #         f"  modes      : action={modes['action_mode']} proprio={modes['proprio_mode']} obstacle_channel={use_obstacle_channel}",
# # # # #         flush=True,
# # # # #     )

# # # # #     try:
# # # # #         for step in range(int(args.max_steps)):
# # # # #             # LIVE language command (real-time inference). Read once per tick so it can drive the
# # # # #             # sample below. With --vla-adapter the intent becomes the policy's text token (Regime
# # # # #             # B: the POLICY executes the maneuver); without it, intent drives the orbit (Regime A).
# # # # #             cmd_txt = args.command
# # # # #             if args.command_file:
# # # # #                 try:
# # # # #                     cmd_txt = (
# # # # #                         Path(args.command_file).read_text(encoding="utf-8").strip()
# # # # #                         or args.command
# # # # #                     )
# # # # #                 except Exception:
# # # # #                     pass
# # # # #             intent = command_intent(cmd_txt)
# # # # #             force_side = (
# # # # #                 1.0 if intent == "left" else (-1.0 if intent == "right" else None)
# # # # #             )
# # # # #             vla_token = None  # set below, once the obstacle distance is known
# # # # #             stop_cmd = False  # set below (gated on obstacle proximity)

# # # # #             y = terrain.local_height_max(x, z, float(args.pose_terrain_radius)) + float(
# # # # #                 args.clearance
# # # # #             )
# # # # #             position = np.asarray([x, y, z], dtype=np.float32)
# # # # #             set_agent_pose(agent, x, y, z, yaw)
# # # # #             obs = sim.get_sensor_observations()
# # # # #             rgb, depth = rgb_depth(obs)
# # # # #             if mesh_tracking_mode:
# # # # #                 # RENDERED-MASK goal: a semantic mesh (placed at t=0 from a pixel, or registered
# # # # #                 # up-front from the VLM's resolved selection) is rendered each step; the belief is
# # # # #                 # RE-DERIVED from that live mask every step it's visible, so tracking follows the
# # # # #                 # mask's ground truth instead of dead-reckoning by odometry alone (which drifts) --
# # # # #                 # dead-reckoning only bridges the gap while the mask briefly drops out of view.
# # # # #                 if mesh_goal_mode and step == 0:
# # # # #                     _gw, _ow = place_mesh_goal_obstacle(
# # # # #                         sim, depth, position, yaw, intr, args, out_dir
# # # # #                     )
# # # # #                     if _gw is not None:
# # # # #                         goal[:] = np.asarray(_gw, dtype=np.float32)
# # # # #                     obs = (
# # # # #                         sim.get_sensor_observations()
# # # # #                     )  # re-render now that the meshes exist
# # # # #                     rgb, depth = rgb_depth(obs)
# # # # #                 _sem = semantic_from_obs(obs)
# # # # #                 _gm = np.where(_sem == MESH_GOAL_ID, 255, 0).astype(np.uint8)
# # # # #                 if int(_gm.sum()) >= int(args.lost_goal_min_px):
# # # # #                     belief_g = mask_to_body(
# # # # #                         _gm,
# # # # #                         depth,
# # # # #                         rgb.shape[0],
# # # # #                         rgb.shape[1],
# # # # #                         args.hfov_deg,
# # # # #                         float(args.goal_range),
# # # # #                     )
# # # # #                     _ys, _xs = np.where(_gm > 0)
# # # # #                     goal_mask = _gm
# # # # #                     goal_info = {
# # # # #                         "u": float(_xs.mean()),
# # # # #                         "v": float(_ys.mean()),
# # # # #                         "distance": (
# # # # #                             float(np.hypot(belief_g[0], belief_g[1]))
# # # # #                             if belief_g is not None
# # # # #                             else float("nan")
# # # # #                         ),
# # # # #                         "visible": 1.0,
# # # # #                     }
# # # # #                 elif (
# # # # #                     belief_g is not None and belief_adapter is None
# # # # #                 ):  # ghost recovery = the P-controller path
# # # # #                     goal_mask, goal_info = project_body_point_mask(
# # # # #                         belief_g,
# # # # #                         rgb.shape[0],
# # # # #                         rgb.shape[1],
# # # # #                         args.hfov_deg,
# # # # #                         args.goal_radius,
# # # # #                         clamp_to_edge=not args.no_clamp_goal_to_edge,
# # # # #                     )
# # # # #                 else:
# # # # #                     goal_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
# # # # #                     goal_info = {
# # # # #                         "u": -1.0,
# # # # #                         "v": -1.0,
# # # # #                         "distance": float("nan"),
# # # # #                         "visible": 0.0,
# # # # #                     }
# # # # #             elif args.belief_goal:
# # # # #                 # BELIEF-tracked goal: the ghost comes from a body-frame estimate propagated by
# # # # #                 # odometry. It is seeded either from an IMAGE bearing+range (no world xyz) or from
# # # # #                 # the world goal at t=0; with a world goal it can also correct on sight.
# # # # #                 if grounder is not None and (
# # # # #                     belief_g is None
# # # # #                     or (
# # # # #                         not getattr(grounder, "one_shot", False)
# # # # #                         and step % max(1, int(args.grounder_every)) == 0
# # # # #                     )
# # # # #                 ):
# # # # #                     # LANGUAGE grounds the goal: RGB + instruction -> pixel -> body point (belief)
# # # # #                     pg = grounder.ground(rgb, args.instruction)
# # # # #                     if pg.in_view:
# # # # #                         bbox = getattr(pg, "bbox", None)
# # # # #                         if bbox is not None:
# # # # #                             # Robust: median depth over the whole VLM bbox, not one pixel that can
# # # # #                             # land on a depth discontinuity (see bbox_to_body's docstring).
# # # # #                             belief_g = bbox_to_body(
# # # # #                                 bbox,
# # # # #                                 depth,
# # # # #                                 rgb.shape[0],
# # # # #                                 rgb.shape[1],
# # # # #                                 args.hfov_deg,
# # # # #                                 args.goal_range,
# # # # #                             )
# # # # #                         else:
# # # # #                             belief_g = pixel_to_body(
# # # # #                                 pg.u,
# # # # #                                 pg.v,
# # # # #                                 depth,
# # # # #                                 rgb.shape[0],
# # # # #                                 rgb.shape[1],
# # # # #                                 args.hfov_deg,
# # # # #                                 args.goal_range,
# # # # #                             )
# # # # #                 elif belief_g is None:
# # # # #                     if args.goal_bearing_deg is not None:
# # # # #                         _b = math.radians(
# # # # #                             float(args.goal_bearing_deg)
# # # # #                         )  # + = right of forward
# # # # #                         belief_g = np.asarray(
# # # # #                             [
# # # # #                                 float(args.goal_range) * math.cos(_b),
# # # # #                                 -float(args.goal_range) * math.sin(_b),
# # # # #                             ],
# # # # #                             dtype=np.float32,
# # # # #                         )
# # # # #                     else:
# # # # #                         _gr, _gu, _gf = camera_coords(goal, position, yaw)
# # # # #                         belief_g = np.asarray([_gf, -_gr], dtype=np.float32)
# # # # #                 elif (
# # # # #                     grounder is None
# # # # #                     and args.goal_bearing_deg is None
# # # # #                     and args.belief_update_on_sight
# # # # #                 ):
# # # # #                     _gr, _gu, _gf = camera_coords(goal, position, yaw)
# # # # #                     if _gf > 0.05:
# # # # #                         belief_g = np.asarray(
# # # # #                             [_gf, -_gr], dtype=np.float32
# # # # #                         )  # correct drift on sight
# # # # #                 goal_mask, goal_info = project_body_point_mask(
# # # # #                     belief_g,
# # # # #                     rgb.shape[0],
# # # # #                     rgb.shape[1],
# # # # #                     args.hfov_deg,
# # # # #                     args.goal_radius,
# # # # #                     clamp_to_edge=not args.no_clamp_goal_to_edge,
# # # # #                 )
# # # # #             else:
# # # # #                 goal_mask, goal_info = project_goal_mask(
# # # # #                     goal=goal,
# # # # #                     position=position,
# # # # #                     yaw=yaw,
# # # # #                     height=rgb.shape[0],
# # # # #                     width=rgb.shape[1],
# # # # #                     hfov_deg=args.hfov_deg,
# # # # #                     radius=args.goal_radius,
# # # # #                     clamp_to_edge=not args.no_clamp_goal_to_edge,
# # # # #                 )
# # # # #             if args.obstacle_mode == "depth":
# # # # #                 obstacle_mask = depth_obstacle_mask(
# # # # #                     depth, args.obstacle_depth_threshold, args.obstacle_min_y_frac
# # # # #                 )
# # # # #             else:
# # # # #                 obstacle_mask = np.zeros_like(goal_mask, dtype=np.uint8)

# # # # #             ghost_obstacle_mask = np.zeros_like(goal_mask, dtype=np.uint8)
# # # # #             obstacle_info = {
# # # # #                 "u": -1.0,
# # # # #                 "v": -1.0,
# # # # #                 "range": float("nan"),
# # # # #                 "visible": 0.0,
# # # # #             }
# # # # #             ghost_obstacle_point = None
# # # # #             if ghost_obstacle is not None:
# # # # #                 ghost_obstacle_mask, obstacle_info = project_goal_mask(
# # # # #                     goal=ghost_obstacle,
# # # # #                     position=position,
# # # # #                     yaw=yaw,
# # # # #                     height=rgb.shape[0],
# # # # #                     width=rgb.shape[1],
# # # # #                     hfov_deg=args.hfov_deg,
# # # # #                     radius=args.ghost_obstacle_radius,
# # # # #                     clamp_to_edge=not args.no_clamp_obstacle_to_edge,
# # # # #                 )
# # # # #                 ghost_obstacle_point = obstacle_point_from_world(
# # # # #                     ghost_obstacle, position, yaw
# # # # #                 )
# # # # #                 obstacle_mask = np.maximum(obstacle_mask, ghost_obstacle_mask).astype(
# # # # #                     np.uint8
# # # # #                 )

# # # # #             if mesh_tracking_mode:
# # # # #                 # obstacle = ONLY the rendered obstacle-mesh pixels (semantic id), never the ground.
# # # # #                 # All-zero if no obstacle mesh was registered (no --obstacle-mesh-uv / no VLM
# # # # #                 # obstacle resolved), same as before -- this only ever ADDS ground-truth precision.
# # # # #                 obstacle_mask = np.where(_sem == MESH_OBST_ID, 255, 0).astype(np.uint8)

# # # # #             goal_ratio = goal_pixel_ratio(goal_mask)

# # # # #             spatial = frame_to_spatial(
# # # # #                 depth,
# # # # #                 goal_mask,
# # # # #                 image_size,
# # # # #                 obstacle_mask,
# # # # #                 include_obstacle_channel=use_obstacle_channel,
# # # # #             ).to(device)
# # # # #             obstacle_map = (
# # # # #                 obstacle_builder.build(depth)
# # # # #                 if args.obstacle_mode == "depth"
# # # # #                 else np.zeros((96, 96), dtype=np.float32)
# # # # #             )
# # # # #             obstacle_map = paint_obstacle_map_point(
# # # # #                 obstacle_map,
# # # # #                 obstacle_builder,
# # # # #                 ghost_obstacle_point,
# # # # #                 args.ghost_obstacle_map_radius,
# # # # #             )
# # # # #             obstacle_t = torch.from_numpy(obstacle_map[None]).float().to(device)

# # # # #             qx, qy, qz, qw = yaw_quat_xyzw(yaw)
# # # # #             pose = np.asarray([x, y, z, qx, qy, qz, qw], dtype=np.float32)
# # # # #             proprio = _proprio_from_pose(
# # # # #                 pose,
# # # # #                 modes["proprio_mode"],
# # # # #                 planar_axes=(0, 2),
# # # # #                 yaw_axis=modes["yaw_axis"],
# # # # #             )
# # # # #             proprio_t = torch.from_numpy(proprio[None]).float().to(device)
# # # # #             belief_t = torch.from_numpy(_empty_belief_tensor()[None]).float().to(device)
# # # # #             route_index = torch.zeros(1, dtype=torch.long, device=device)
# # # # #             active_goal_index = torch.zeros(1, dtype=torch.long, device=device)

# # # # #             obstacle_point = None
# # # # #             obstacle_radius_perceived = (
# # # # #                 None  # mask-derived obstacle radius (mesh obstacles only;
# # # # #             )
# # # # #             # ghost obstacles use --ghost-obstacle-world-radius instead)
# # # # #             if (args.cbf or args.dwa) and int(obstacle_mask.sum()) > 0:
# # # # #                 obstacle_point = ghost_obstacle_point
# # # # #                 if obstacle_point is None:
# # # # #                     # nearest_obstacle_state gives BOTH the nearest point AND a robust (90th-
# # # # #                     # percentile, clamped) radius from the mask's own unprojected spatial extent
# # # # #                     # -- a real "obstacle size" measurement (pixel extent -> depth -> metres), not
# # # # #                     # a hand-picked constant. Falls back to nearest_obstacle_point's plain nearest-
# # # # #                     # point-only behavior if the mask/depth don't support a radius estimate.
# # # # #                     _obs_state = nearest_obstacle_state(obstacle_mask, depth, intr)
# # # # #                     if _obs_state is not None:
# # # # #                         obstacle_point = _obs_state["p0"]
# # # # #                         obstacle_radius_perceived = _obs_state["radius"]
# # # # #                     else:
# # # # #                         obstacle_point = nearest_obstacle_point(
# # # # #                             obstacle_mask, depth, intr
# # # # #                         )
# # # # #             if obstacle_point is not None and ghost_obstacle is None:
# # # # #                 # obstacle_info["range"] is only ever set by the ghost-obstacle world-coordinate
# # # # #                 # path (project_goal_mask below); mesh-obstacle mode never touches it, so
# # # # #                 # min_obstacle_dist stayed NaN in every logged run. Fill it from the perceived
# # # # #                 # obstacle_point directly so the safety-margin ablation metric is actually recorded.
# # # # #                 obstacle_info["range"] = float(
# # # # #                     np.hypot(obstacle_point[0], obstacle_point[1])
# # # # #                 )
# # # # #                 obstacle_info["visible"] = 1.0
# # # # #             if obstacle_point is None:
# # # # #                 cone_side_latch = (
# # # # #                     None  # obstacle gone -> release the committed cone-proj side
# # # # #                 )
# # # # #                 # (around_side is released below when the obstacle stops blocking the goal ray)

# # # # #             # Proximity gate: a maneuver command applies WHEN YOU REACH the obstacle. Drive
# # # # #             # straight toward the goal until the obstacle is close (within d_safe+deadzone AND
# # # # #             # ahead), then turn (left/right) or stop; after passing, continue to the goal. Without
# # # # #             # this a persistent command would turn/stop the whole way.
# # # # #             # Hysteresis: ENTER near-obstacle at d_safe+deadzone, only EXIT past an extra margin, so
# # # # #             # small distance jitter right at the boundary can't flip the gate back and forth every
# # # # #             # tick (that flicker -- not the action smoother -- was the source of the jerkiness).
# # # # #             near_obstacle_enter = (
# # # # #                 obstacle_point is not None
# # # # #                 and float(obstacle_point[0]) > 0.0
# # # # #                 and float(np.hypot(obstacle_point[0], obstacle_point[1]))
# # # # #                 < args.cbf_d_safe + args.cbf_deadzone
# # # # #             )
# # # # #             near_obstacle_exit = (
# # # # #                 obstacle_point is None
# # # # #                 or float(obstacle_point[0]) <= 0.0
# # # # #                 or float(np.hypot(obstacle_point[0], obstacle_point[1]))
# # # # #                 > args.cbf_d_safe + args.cbf_deadzone + float(args.lang_turn_hyst)
# # # # #             )
# # # # #             if near_obstacle_enter:
# # # # #                 near_obstacle_state = True
# # # # #             elif near_obstacle_exit:
# # # # #                 near_obstacle_state = False
# # # # #             near_obstacle = near_obstacle_state
# # # # #             # "stop" is a language TRIGGER, but the HALT itself is proximity-gated: engage once close
# # # # #             # to the obstacle OR the goal (whichever comes first), not the instant "stop" is said.
# # # # #             # Without the goal term, "stop" never fired on a goal-only run (no obstacle to be near).
# # # # #             goal_dist_now = float(
# # # # #                 np.linalg.norm(goal[[0, 2]] - np.asarray([x, z], dtype=np.float32))
# # # # #             )
# # # # #             near_goal = goal_dist_now <= float(args.stop_dist) + float(
# # # # #                 args.cbf_deadzone
# # # # #             )
# # # # #             stop_cmd = (intent == "stop") and (near_obstacle or near_goal)
# # # # #             if vla_adapter is not None:
# # # # #                 # Hard, full-strength switch (NOT a blend): interpolating between two different
# # # # #                 # instruction tokens landed off the trained manifold -- the adapter only ever
# # # # #                 # produces alpha*adapter(ONE instruction), never a weighted sum of two -- and since
# # # # #                 # diffusion sampling is nonlinear in its conditioning, every replan along a blend
# # # # #                 # sampled an uncorrelated, effectively random chunk (far worse than one clean cut).
# # # # #                 # The hysteresis above still does its job: it debounces near_obstacle so this switch
# # # # #                 # doesn't fire repeatedly from boundary jitter.
# # # # #                 if intent in ("left", "right") and near_obstacle:
# # # # #                     _phrase = cmd_txt
# # # # #                 elif intent in ("left", "right", "straight", "stop"):
# # # # #                     _phrase = "navigate to the goal"
# # # # #                 else:
# # # # #                     _phrase = None
# # # # #                 if _phrase is not None:
# # # # #                     if _phrase not in vla_tok_cache:
# # # # #                         with torch.no_grad():
# # # # #                             _e = (
# # # # #                                 torch.from_numpy(
# # # # #                                     vla_text_enc.encode(
# # # # #                                         [_phrase], normalize_embeddings=True
# # # # #                                     )
# # # # #                                 )
# # # # #                                 .float()
# # # # #                                 .to(device)
# # # # #                             )
# # # # #                             vla_tok_cache[_phrase] = float(
# # # # #                                 args.vla_alpha_scale
# # # # #                             ) * vla_adapter(_e)
# # # # #                     vla_token = vla_tok_cache[_phrase]

# # # # #             # BELIEF-RETURN: when the goal is OFF-SCREEN (mask gone) inject the belief token so the
# # # # #             # POLICY turns back toward it -- the learned return that replaces the P-controller.
# # # # #             # Hysteresis only (enter on empty mask, exit only once well re-acquired); full-strength,
# # # # #             # same reasoning as above -- no magnitude ramp, so every replan sees a token identical
# # # # #             # to the one the adapter was trained/ablated on, not an in-between value.
# # # # #             goal_px = int((goal_mask > 0).sum())
# # # # #             exit_px = (
# # # # #                 int(args.belief_reacquire_px)
# # # # #                 if args.belief_reacquire_px is not None
# # # # #                 else 3 * int(args.lost_goal_min_px)
# # # # #             )
# # # # #             if belief_adapter is not None and belief_g is not None:
# # # # #                 if goal_px < int(args.lost_goal_min_px):
# # # # #                     belief_state = True
# # # # #                 elif goal_px > exit_px:
# # # # #                     belief_state = False
# # # # #             else:
# # # # #                 belief_state = False
# # # # #             belief_token = None
# # # # #             if belief_adapter is not None and belief_g is not None and belief_state:
# # # # #                 with torch.no_grad():
# # # # #                     belief_token = belief_adapter(
# # # # #                         torch.from_numpy(belief_feat(belief_g)[None]).float().to(device)
# # # # #                     )
# # # # #             _toks = [t for t in (belief_token, vla_token) if t is not None]
# # # # #             extra_cond = torch.cat(_toks, dim=1) if _toks else None

# # # # #             do_replan = (step % replan_every == 0) or (last_pred_chunk is None)
# # # # #             if do_replan:
# # # # #                 pred = model.sample(
# # # # #                     spatial,
# # # # #                     proprio_t,
# # # # #                     steps=int(args.sample_steps),
# # # # #                     belief_tensor=belief_t,
# # # # #                     obstacle_map=obstacle_t,
# # # # #                     route_index=route_index,
# # # # #                     active_goal_index=active_goal_index,
# # # # #                     extra_cond_tokens=extra_cond,  # belief token (off-screen) and/or language token
# # # # #                 )

# # # # #                 if (
# # # # #                     args.cbf
# # # # #                     and args.cbf_mode == "cone"
# # # # #                     and obstacle_point is not None
# # # # #                     and not args.vla_adapter
# # # # #                     and args.cbf_cone_project
# # # # #                 ):
# # # # #                     cbf_active += 1
# # # # #                     v_o = np.zeros(2, dtype=np.float32)
# # # # #                     if args.zero_lateral and pred.shape[-1] >= 3:
# # # # #                         pred = pred.clone()
# # # # #                         pred[..., 1] = 0.0
# # # # #                     p_lat = float(obstacle_point[1])
# # # # #                     side = -1.0 if p_lat > 0.0 else 1.0
# # # # #                     if args.cbf_commit_side:
# # # # #                         if cone_side_latch is None:
# # # # #                             cone_side_latch = side
# # # # #                         side = cone_side_latch
# # # # #                     cone_sigma = None
# # # # #                     if args.cbf_metric == "mahalanobis":
# # # # #                         cone_sigma = horizon_growth_covariance(
# # # # #                             pred.shape[1],
# # # # #                             pred.shape[2],
# # # # #                             base=args.cbf_cov_base,
# # # # #                             growth=args.cbf_cov_growth,
# # # # #                             mode=args.cbf_cov_mode,
# # # # #                             device=pred.device,
# # # # #                             dtype=pred.dtype,
# # # # #                         )
# # # # #                     if (
# # # # #                         args.cbf_radius_mode == "perceived"
# # # # #                         and ghost_obstacle is not None
# # # # #                     ):
# # # # #                         r_used = (
# # # # #                             args.ghost_obstacle_world_radius
# # # # #                             + args.robot_radius
# # # # #                             + args.safety_margin
# # # # #                         )
# # # # #                     elif (
# # # # #                         args.cbf_radius_mode == "perceived"
# # # # #                         and obstacle_radius_perceived is not None
# # # # #                     ):
# # # # #                         # mesh/mask obstacle: robot radius + the mask-derived obstacle radius (pixel
# # # # #                         # extent -> depth -> metres, via nearest_obstacle_state) + margin -- an
# # # # #                         # ACTUAL collision cone radius, not a hand-set constant.
# # # # #                         r_used = (
# # # # #                             obstacle_radius_perceived
# # # # #                             + args.robot_radius
# # # # #                             + args.safety_margin
# # # # #                         )
# # # # #                     else:
# # # # #                         r_used = args.cbf_d_safe
# # # # #                     # Capture the PRE-cone chunk (after zero_lateral, so both raw/corrected share
# # # # #                     # that masking) to measure how much the cone projection itself moves the
# # # # #                     # EXECUTED step vs a later one -- the direct test of "does Euclidean spread the
# # # # #                     # correction uniformly while Mahalanobis concentrates it on the near step".
# # # # #                     pre_cone_step0 = pred[0, 0, :].detach().cpu().numpy().copy()
# # # # #                     pre_cone_last = pred[0, -1, :].detach().cpu().numpy().copy()
# # # # #                     pred = project_chunk_cone(
# # # # #                         pred,
# # # # #                         obstacle_point,
# # # # #                         v_o,
# # # # #                         r=r_used,
# # # # #                         dt=dt,
# # # # #                         vel_scale=1.0,
# # # # #                         iters=args.cbf_proj_iters,
# # # # #                         lr=args.cbf_proj_lr,
# # # # #                         trust=args.cbf_trust,
# # # # #                         margin=args.cbf_cone_margin,
# # # # #                         smooth_weight=args.cbf_smooth,
# # # # #                         keep_speed=args.cbf_keep_speed,
# # # # #                         sigma=cone_sigma,
# # # # #                         deadzone_range=r_used + args.cbf_deadzone,
# # # # #                         side=side,
# # # # #                     )
# # # # #                     post_cone_step0 = pred[0, 0, :].detach().cpu().numpy()
# # # # #                     post_cone_last = pred[0, -1, :].detach().cpu().numpy()
# # # # #                     cone_correction_step0 = float(
# # # # #                         np.linalg.norm(post_cone_step0 - pre_cone_step0)
# # # # #                     )
# # # # #                     cone_correction_last = float(
# # # # #                         np.linalg.norm(post_cone_last - pre_cone_last)
# # # # #                     )
# # # # #                     prev_obstacle_point = obstacle_point

# # # # #                 pred_chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
# # # # #                 chunk_ctrl = np.stack(
# # # # #                     [
# # # # #                         action_to_control(
# # # # #                             a,
# # # # #                             action_mode=modes["action_mode"],
# # # # #                             max_forward_speed=args.max_forward_speed,
# # # # #                             max_lateral_speed=args.max_lateral_speed,
# # # # #                             max_yaw_rate=args.max_yaw_rate,
# # # # #                         )
# # # # #                         for a in pred_chunk
# # # # #                     ]
# # # # #                 ).astype(np.float32)
# # # # #                 smoother.add(step, chunk_ctrl)
# # # # #                 last_pred_chunk = pred_chunk
# # # # #                 chunk_len = int(pred_chunk.shape[0])
# # # # #                 if step == 0 and replan_every > chunk_len:
# # # # #                     print(
# # # # #                         f"[WARN] --replan-every {replan_every} > chunk length {chunk_len}; "
# # # # #                         "actions will repeat after the buffer runs dry.",
# # # # #                         flush=True,
# # # # #                     )
# # # # #             else:
# # # # #                 pred_chunk = last_pred_chunk
# # # # #             _ = prev_obstacle_point

# # # # #             action_3d = smoother.get(step)
# # # # #             goal_lost = int(goal_mask.sum()) < int(args.lost_goal_min_px)

# # # # #             # Perceived safety radius (obstacle + rover + margin) or a fixed d_safe.
# # # # #             if args.cbf_radius_mode == "perceived" and ghost_obstacle is not None:
# # # # #                 r_gate = (
# # # # #                     args.ghost_obstacle_world_radius
# # # # #                     + args.robot_radius
# # # # #                     + args.safety_margin
# # # # #                 )
# # # # #             elif (
# # # # #                 args.cbf_radius_mode == "perceived"
# # # # #                 and obstacle_radius_perceived is not None
# # # # #             ):
# # # # #                 r_gate = (
# # # # #                     obstacle_radius_perceived + args.robot_radius + args.safety_margin
# # # # #                 )
# # # # #             else:
# # # # #                 r_gate = args.cbf_d_safe

# # # # #             # Physical collision radius (obstacle + rover + margin). The orbit hugs the larger
# # # # #             # r_gate circle; this smaller radius is only the hard-breach backstop. Always prefers
# # # # #             # a REAL measured radius (ghost world-radius, or the mask-derived one) over the
# # # # #             # --ghost-obstacle-world-radius fallback, regardless of --cbf-radius-mode -- this is
# # # # #             # the physical safety backstop, so it shouldn't silently use an unrelated constant
# # # # #             # whenever a real obstacle-size measurement is available.
# # # # #             if ghost_obstacle is not None:
# # # # #                 r_cone = (
# # # # #                     args.ghost_obstacle_world_radius
# # # # #                     + args.robot_radius
# # # # #                     + args.safety_margin
# # # # #                 )
# # # # #             elif obstacle_radius_perceived is not None:
# # # # #                 r_cone = (
# # # # #                     obstacle_radius_perceived + args.robot_radius + args.safety_margin
# # # # #                 )
# # # # #             else:
# # # # #                 r_cone = (
# # # # #                     args.ghost_obstacle_world_radius
# # # # #                     + args.robot_radius
# # # # #                     + args.safety_margin
# # # # #                 )

# # # # #             # Control obstacle point [forward, left]. For a GHOST obstacle use the RAW geometry
# # # # #             # (no forward>0.05 cutoff) so distance/bearing never flicker as it passes abeam --
# # # # #             # that flicker toggles the avoid state and chatters the commands. Perceived
# # # # #             # obstacles keep the mask-based obstacle_point.
# # # # #             ctrl_op = None
# # # # #             if (args.cbf and args.cbf_mode == "cone") or args.dwa:
# # # # #                 if ghost_obstacle is not None:
# # # # #                     _r, _u, _f = camera_coords(ghost_obstacle, position, yaw)
# # # # #                     ctrl_op = np.asarray([_f, -_r], dtype=np.float32)  # [forward, left]
# # # # #                 elif obstacle_point is not None:
# # # # #                     ctrl_op = np.asarray(obstacle_point, dtype=np.float32)

# # # # #             # Within the cone+deadzone shell, and is the obstacle actually BETWEEN us and the
# # # # #             # goal? Hysteresis on the perpendicular clearance (wider to leave than to enter) so
# # # # #             # the orbit<->goal decision cannot rapid-toggle at the boundary.
# # # # #             avoiding = (
# # # # #                 ctrl_op is not None
# # # # #                 and float(np.hypot(ctrl_op[0], ctrl_op[1])) < r_gate + args.cbf_deadzone
# # # # #             )
# # # # #             blocked = False
# # # # #             if ctrl_op is not None:
# # # # #                 ox, oy = float(ctrl_op[0]), float(ctrl_op[1])
# # # # #                 L = math.hypot(ox, oy)
# # # # #                 phi = math.atan2(oy, ox)
# # # # #                 beta = (
# # # # #                     math.atan2(float(belief_g[1]), float(belief_g[0]))
# # # # #                     if (args.belief_goal and belief_g is not None)
# # # # #                     else planar_goal_bearing(position, yaw, goal)
# # # # #                 )
# # # # #                 proj = ox * math.cos(beta) + oy * math.sin(beta)
# # # # #                 perp = math.sqrt(max(L * L - proj * proj, 0.0))
# # # # #                 thresh = r_gate + (
# # # # #                     args.cbf_orbit_hyst if around_side is not None else 0.0
# # # # #                 )
# # # # #                 blocked = avoiding and (proj > 0.0) and (perp < thresh)

# # # # #             # (language command already read at the top of the loop -> intent / force_side /
# # # # #             # stop_cmd / vla_token)

# # # # #             # Ghost heading assist. The goal ghost is a binary mask; once the goal drifts
# # # # #             # past ~hfov/2 it clamps to the SAME border pixel no matter how far off-axis it
# # # # #             # is, so the mask-conditioned policy only steers weakly and can't tell "just
# # # # #             # off-screen" from "behind me". Override the yaw with a proportional turn toward
# # # # #             # the true goal bearing whenever the goal is off-centre OR fully behind:
# # # # #             #   - off to the side (still ahead): KEEP the policy's forward and only steer
# # # # #             #     hard, so we arc toward the goal without stopping (no stop-and-pivot judder);
# # # # #             #   - fully behind (mask empty): pivot with the forward floor.
# # # # #             # Suppressed only while the rock BLOCKS the goal ray (the orbit owns steering then),
# # # # #             # so the goal pull can't drag us back through it; a nearby-but-clear rock still lets
# # # # #             # the assist run.
# # # # #             if (
# # # # #                 args.lost_goal_ghost
# # # # #                 and not blocked
# # # # #                 and belief_adapter is None
# # # # #                 and not args.dwa
# # # # #             ):  # belief adapter / DWA replace this P-controller
# # # # #                 bearing = (
# # # # #                     math.atan2(float(belief_g[1]), float(belief_g[0]))
# # # # #                     if (args.belief_goal and belief_g is not None)
# # # # #                     else planar_goal_bearing(position, yaw, goal)
# # # # #                 )
# # # # #                 goal_behind = goal_lost
# # # # #                 goal_offcentre = args.lost_goal_bearing_deg > 0.0 and abs(
# # # # #                     bearing
# # # # #                 ) > math.radians(float(args.lost_goal_bearing_deg))
# # # # #                 if goal_behind or goal_offcentre:
# # # # #                     yaw_cmd = float(
# # # # #                         np.clip(
# # # # #                             float(args.lost_goal_turn_kp) * bearing,
# # # # #                             -float(args.max_yaw_rate),
# # # # #                             float(args.max_yaw_rate),
# # # # #                         )
# # # # #                     )
# # # # #                     fwd = (
# # # # #                         float(args.lost_goal_forward)
# # # # #                         if goal_behind
# # # # #                         else float(action_3d[0])
# # # # #                     )
# # # # #                     fwd = max(fwd, float(args.lost_goal_forward))
# # # # #                     action_3d = np.asarray([fwd, 0.0, yaw_cmd], dtype=np.float32)
# # # # #             if args.zero_lateral and action_3d.shape[0] >= 2:
# # # # #                 action_3d = action_3d.copy()
# # # # #                 action_3d[1] = 0.0
# # # # #             if (
# # # # #                 args.cbf
# # # # #                 and args.cbf_mode == "project"
# # # # #                 and obstacle_point is not None
# # # # #                 and not args.dwa
# # # # #             ):
# # # # #                 action_3d, _ = project_forward_velocity_cbf(
# # # # #                     action_3d,
# # # # #                     obstacle_point,
# # # # #                     np.zeros(2, dtype=np.float32),
# # # # #                     d_safe=args.cbf_d_safe,
# # # # #                     gamma=args.cbf_gamma,
# # # # #                     deadzone=args.cbf_deadzone,
# # # # #                     trust=args.cbf_trust,
# # # # #                 )

# # # # #             # Smooth ORBIT controller around the obstacle's safety circle. While the rock blocks
# # # # #             # the goal ray, steer along the tangent (phi + side*90deg) plus a linear radial
# # # # #             # pull-back toward the r_gate circle: it settles ON the circle and traces a smooth
# # # # #             # line-arc-line detour at constant cruise -> no stop/rotate/go judder, and no
# # # # #             # asin-tangent bounce when hugging tight. Body frame is [forward, left]; +heading
# # # # #             # turns left. Released back to goal-seeking once the rock clears the ray (+hyst).
# # # # #             if (
# # # # #                 blocked
# # # # #                 and args.cbf_escape_yaw > 0.0
# # # # #                 and not args.vla_adapter
# # # # #                 and not args.dwa
# # # # #             ):
# # # # #                 # Commit which way around: the tangent heading closest to the goal bearing
# # # # #                 # (least detour, natural return). Latched until the rock stops blocking.
# # # # #                 if force_side is not None:
# # # # #                     around_side = (
# # # # #                         force_side  # language command overrides the geometric side
# # # # #                     )
# # # # #                 elif around_side is None:
# # # # #                     a = math.asin(min(1.0, r_gate / max(L, 1e-6)))
# # # # #                     dl = abs(wrap_angle(phi + a - beta))
# # # # #                     dr = abs(wrap_angle(phi - a - beta))
# # # # #                     around_side = 1.0 if dl <= dr else -1.0
# # # # #                 corr = max(-1.2, min(1.2, float(args.cbf_orbit_kr) * (L - r_gate)))
# # # # #                 psi = wrap_angle(phi + around_side * (0.5 * math.pi - corr))
# # # # #                 yaw_cmd = float(
# # # # #                     np.clip(
# # # # #                         float(args.cbf_pursuit_kp) * psi,
# # # # #                         -float(args.max_yaw_rate),
# # # # #                         float(args.max_yaw_rate),
# # # # #                     )
# # # # #                 )
# # # # #                 nominal_action_orbit = [float(v) for v in action_3d]
# # # # #                 action_3d = np.asarray(
# # # # #                     [float(args.cbf_goaround_forward), 0.0, yaw_cmd], dtype=np.float32
# # # # #                 )
# # # # #                 escape_active += 1
# # # # #                 episode_logger.log_cbf_event(
# # # # #                     step=step,
# # # # #                     obstacle_id=cbf_obstacle_id,
# # # # #                     distance=float(L),
# # # # #                     nominal_action=nominal_action_orbit,
# # # # #                     overridden_action=[float(v) for v in action_3d],
# # # # #                     mode="orbit",
# # # # #                 )
# # # # #             elif around_side is not None:
# # # # #                 around_side = (
# # # # #                     None  # rock no longer blocks the goal ray -> release the side
# # # # #                 )

# # # # #             # HARD per-tick safety backstop for cone mode. Tangent pursuit steers along the
# # # # #             # r_gate circle so it never approaches closer than r_gate (> the collision radius
# # # # #             # r_cone); the distance brake's approach-rate term would otherwise fight that by
# # # # #             # braking the cruise during the turn-in. So WHILE pursuit is active and we are
# # # # #             # still outside the collision radius, trust the steering and stay smooth; only if
# # # # #             # we somehow penetrate r_cone (a genuine breach) do we fall back to the brake.
# # # # #             # When pursuit is off (escape-yaw 0), the plain distance brake applies as before.
# # # # #             hard_gate_fired_tick = (
# # # # #                 False  # per-tick flag: did the cheap backup brake have to rescue
# # # # #             )
# # # # #             if (
# # # # #                 args.cbf
# # # # #                 and args.cbf_mode == "cone"
# # # # #                 and args.cbf_hard_gate
# # # # #                 and ctrl_op is not None
# # # # #                 and not args.dwa
# # # # #             ):
# # # # #                 p_fwd, p_lat = float(ctrl_op[0]), float(ctrl_op[1])
# # # # #                 # Release on LATERAL clearance, not distance: driving straight forward MISSES the
# # # # #                 # obstacle once its lateral offset exceeds the collision radius (i.e. we have
# # # # #                 # turned enough). Releasing on distance never holds once the policy drives up
# # # # #                 # close, so it brakes forever and pins the rover in front of the rock. Committing
# # # # #                 # the pass the moment the heading clears it lets the (turning) policy drive around.
# # # # #                 cone_clears = (p_fwd <= 0.0) or (abs(p_lat) >= r_cone)
# # # # #                 if cone_clears and float(action_3d[0]) > 0.0:
# # # # #                     pass  # turned enough that forward motion misses the obstacle -> let it drive
# # # # #                 else:
# # # # #                     nominal_action_gate = [float(v) for v in action_3d]
# # # # #                     action_3d, _gated = project_forward_velocity_cbf(
# # # # #                         action_3d,
# # # # #                         ctrl_op,
# # # # #                         np.zeros(2, dtype=np.float32),
# # # # #                         d_safe=r_gate,
# # # # #                         gamma=args.cbf_gamma,
# # # # #                         deadzone=args.cbf_deadzone,
# # # # #                         trust=None,
# # # # #                     )
# # # # #                     if _gated:
# # # # #                         hard_gate_fired += 1
# # # # #                         hard_gate_fired_tick = True
# # # # #                         episode_logger.log_cbf_event(
# # # # #                             step=step,
# # # # #                             obstacle_id=cbf_obstacle_id,
# # # # #                             distance=float(math.hypot(p_fwd, p_lat)),
# # # # #                             nominal_action=nominal_action_gate,
# # # # #                             overridden_action=[float(v) for v in action_3d],
# # # # #                             mode="brake",
# # # # #                         )

# # # # #             # ABLATION BASELINE (--dwa): classic Dynamic Window Approach REPLACES the diffusion
# # # # #             # policy's action + collision cone entirely -- a from-scratch reactive planner with no
# # # # #             # learned prior, for the "collision cone vs DWA" comparison. Overrides whatever the
# # # # #             # (now cone/orbit/ghost-disabled) blocks above produced.
# # # # #             if args.dwa:
# # # # #                 dwa_goal_bearing = (
# # # # #                     math.atan2(float(belief_g[1]), float(belief_g[0]))
# # # # #                     if (args.belief_goal and belief_g is not None)
# # # # #                     else planar_goal_bearing(position, yaw, goal)
# # # # #                 )
# # # # #                 dwa_v, dwa_w = dwa_action(
# # # # #                     dwa_goal_bearing, ctrl_op, dwa_prev_v, dwa_prev_w, dt, args
# # # # #                 )
# # # # #                 action_3d = np.asarray([dwa_v, 0.0, dwa_w], dtype=np.float32)
# # # # #                 dwa_prev_v, dwa_prev_w = dwa_v, dwa_w

# # # # #             # VLA counterfactual data: at blocked steps, save the NEUTRAL-goal observation the
# # # # #             # policy sees, plus ALL FOUR instruction targets on that SAME observation:
# # # # #             #   left / right = orbit around each side (homotopy classes),
# # # # #             #   stop         = decelerate-to-stop before the obstacle,
# # # # #             #   straight     = navigate to the goal (default / prior-preservation).
# # # # #             # Four targets, one observation -> the ONLY thing that can explain the difference is
# # # # #             # the instruction, so the language adapter is forced to use the text.
# # # # #             if args.vla_dump and blocked and ghost_obstacle is not None:
# # # # #                 if vla_count % max(1, int(args.vla_dump_every)) == 0:
# # # # #                     dump_dir = Path(args.vla_dump)
# # # # #                     dump_dir.mkdir(parents=True, exist_ok=True)
# # # # #                     Hc = int(args.vla_horizon)
# # # # #                     kr, cr, kp, mw = (
# # # # #                         args.cbf_orbit_kr,
# # # # #                         args.cbf_goaround_forward,
# # # # #                         args.cbf_pursuit_kp,
# # # # #                         args.max_yaw_rate,
# # # # #                     )
# # # # #                     ck_left = orbit_chunk(
# # # # #                         position,
# # # # #                         yaw,
# # # # #                         ghost_obstacle,
# # # # #                         1.0,
# # # # #                         Hc,
# # # # #                         dt,
# # # # #                         r_gate,
# # # # #                         kr,
# # # # #                         cr,
# # # # #                         kp,
# # # # #                         mw,
# # # # #                     )
# # # # #                     ck_right = orbit_chunk(
# # # # #                         position,
# # # # #                         yaw,
# # # # #                         ghost_obstacle,
# # # # #                         -1.0,
# # # # #                         Hc,
# # # # #                         dt,
# # # # #                         r_gate,
# # # # #                         kr,
# # # # #                         cr,
# # # # #                         kp,
# # # # #                         mw,
# # # # #                     )
# # # # #                     ck_stop = brake_chunk(cr, Hc)
# # # # #                     ck_straight = goal_chunk(
# # # # #                         position, yaw, goal, Hc, dt, cr, args.lost_goal_turn_kp, mw
# # # # #                     )
# # # # #                     ck_back = back_chunk(cr, Hc)
# # # # #                     np.savez_compressed(
# # # # #                         str(dump_dir / f"{Path(args.out).name}_{vla_count:06d}.npz"),
# # # # #                         spatial=spatial.detach().cpu().numpy()[0].astype(np.float32),
# # # # #                         proprio=proprio.astype(np.float32),
# # # # #                         obstacle_map=obstacle_map.astype(np.float32),
# # # # #                         chunk_left=ck_left.astype(np.float32),
# # # # #                         chunk_right=ck_right.astype(np.float32),
# # # # #                         chunk_stop=ck_stop.astype(np.float32),
# # # # #                         chunk_straight=ck_straight.astype(np.float32),
# # # # #                         chunk_back=ck_back.astype(np.float32),
# # # # #                         classes=np.array("left,right,stop,straight,back"),
# # # # #                     )
# # # # #                 vla_count += 1

# # # # #             if stop_cmd:
# # # # #                 action_3d = np.zeros(
# # # # #                     3, dtype=np.float32
# # # # #                 )  # real-time STOP command halts the rover

# # # # #             next_position, next_yaw = integrate_mars(position, yaw, action_3d, dt)
# # # # #             x = float(
# # # # #                 np.clip(
# # # # #                     next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
# # # # #                 )
# # # # #             )
# # # # #             z = float(
# # # # #                 np.clip(
# # # # #                     next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
# # # # #                 )
# # # # #             )
# # # # #             yaw = wrap_angle(next_yaw)

# # # # #             # Log belief_g BEFORE propagation, so it's relative to the SAME (pre-move) pose already
# # # # #             # saved in rows["pose"] -- logging it after propagate_body_point (post-move pose) but
# # # # #             # pairing it with the pre-move saved pose applies the WRONG yaw to reconstruct the world
# # # # #             # estimate: a one-tick rotation mismatch, which at real belief range (~10m) becomes a
# # # # #             # few-metre positional error -- flat across the episode (same mismatch every tick), not
# # # # #             # growing drift. That's what produced a suspiciously constant ~3.6m "error" before.
# # # # #             if belief_g is not None:
# # # # #                 bf, bl = float(belief_g[0]), float(belief_g[1])
# # # # #             else:
# # # # #                 bf, bl = float("nan"), float("nan")

# # # # #             # Propagate the goal belief by the executed motion (dead-reckoning; drifts if noisy).
# # # # #             if args.belief_goal and belief_g is not None:
# # # # #                 belief_g = propagate_body_point(
# # # # #                     belief_g, action_3d, dt, args.belief_odom_noise, belief_rng
# # # # #                 )

# # # # #             goal_dist = float(
# # # # #                 np.linalg.norm(goal[[0, 2]] - np.asarray([x, z], dtype=np.float32))
# # # # #             )
# # # # #             seg = np.zeros_like(goal_mask, dtype=np.uint8)
# # # # #             seg[goal_mask > 0] = 1
# # # # #             seg[obstacle_mask > 0] = 2

# # # # #             rows["rgb"].append(rgb)
# # # # #             rows["depth"].append(depth)
# # # # #             rows["goal_mask"].append(goal_mask.astype(np.uint8))
# # # # #             rows["obstacle_mask"].append(obstacle_mask.astype(np.uint8))
# # # # #             rows["seg_masks"].append(seg.astype(np.uint8))
# # # # #             rows["pose"].append(pose)
# # # # #             rows["proprio"].append(proprio.astype(np.float32))
# # # # #             rows["action_3d"].append(action_3d.astype(np.float32))
# # # # #             rows["pred_chunk"].append(pred_chunk.astype(np.float32))
# # # # #             rows["goal_visible_pixels"].append(int(goal_mask.sum()))
# # # # #             rows["goal_u"].append(float(goal_info["u"]))
# # # # #             rows["goal_v"].append(float(goal_info["v"]))
# # # # #             rows["goal_distance"].append(goal_dist)
# # # # #             rows["obstacle_visible_pixels"].append(int(obstacle_mask.sum()))
# # # # #             rows["obstacle_u"].append(float(obstacle_info["u"]))
# # # # #             rows["obstacle_v"].append(float(obstacle_info["v"]))
# # # # #             rows["obstacle_distance"].append(float(obstacle_info["range"]))
# # # # #             rows["belief_fwd"].append(bf)
# # # # #             rows["belief_left"].append(bl)
# # # # #             rows["goal_frame_fraction"].append(goal_ratio["frame_fraction"])
# # # # #             rows["cone_correction_step0"].append(cone_correction_step0)
# # # # #             rows["cone_correction_last"].append(cone_correction_last)
# # # # #             rows["hard_gate_tick"].append(bool(hard_gate_fired_tick))

# # # # #             distances_to_obstacles = (
# # # # #                 {cbf_obstacle_id: float(obstacle_info["range"])}
# # # # #                 if ghost_obstacle is not None
# # # # #                 else {}
# # # # #             )
# # # # #             episode_logger.log_frame(
# # # # #                 step=step,
# # # # #                 position=[float(pose[0]), float(pose[1]), float(pose[2])],
# # # # #                 orientation=[
# # # # #                     float(pose[3]),
# # # # #                     float(pose[4]),
# # # # #                     float(pose[5]),
# # # # #                     float(pose[6]),
# # # # #                 ],
# # # # #                 action={
# # # # #                     "v_fwd": float(action_3d[0]),
# # # # #                     "v_lat": float(action_3d[1]),
# # # # #                     "yaw_rate": float(action_3d[2]),
# # # # #                 },
# # # # #                 distances_to_obstacles=distances_to_obstacles,
# # # # #                 cbf_active=bool(avoiding),
# # # # #                 goal_belief={
# # # # #                     "range": goal_dist,
# # # # #                     "bearing": math.degrees(planar_goal_bearing(position, yaw, goal)),
# # # # #                     "source": (
# # # # #                         "observed"
# # # # #                         if int(goal_mask.sum()) >= int(args.lost_goal_min_px)
# # # # #                         else "dead_reckoned"
# # # # #                     ),
# # # # #                 },
# # # # #             )
# # # # #             episode_logger.log_rendered_frame(step, rgb)

# # # # #             if step % max(int(args.save_every), 1) == 0:
# # # # #                 lost_txt = (
# # # # #                     " LOST" if int(goal_mask.sum()) < int(args.lost_goal_min_px) else ""
# # # # #                 )
# # # # #                 text = f"t={step} dist={goal_dist:.2f} obs={int(obstacle_mask.sum())} v={action_3d[0]:.2f} yaw={math.degrees(yaw):.1f}{lost_txt}"
# # # # #                 frame = overlay_frame(rgb, goal_mask, obstacle_mask, text)
# # # # #                 frame.save(frame_dir / f"frame_{step:04d}.png")
# # # # #                 video_frames.append(frame)
# # # # #                 # binary mask: goal=white, obstacle=red, background=black
# # # # #                 mimg = np.zeros(
# # # # #                     (goal_mask.shape[0], goal_mask.shape[1], 3), dtype=np.uint8
# # # # #                 )
# # # # #                 mimg[goal_mask > 0] = (255, 255, 255)
# # # # #                 mimg[obstacle_mask > 0] = (255, 0, 0)
# # # # #                 Image.fromarray(mimg).save(frame_dir / f"mask_{step:04d}.png")

# # # # #             if step % 10 == 0:
# # # # #                 print(
# # # # #                     f"step {step:04d} | dist={goal_dist:.2f} | goal_px={int(goal_mask.sum())} "
# # # # #                     f"| obs_px={int(obstacle_mask.sum())} "
# # # # #                     f"| action=[{action_3d[0]:.2f},{action_3d[1]:.2f},{action_3d[2]:.2f}]",
# # # # #                     flush=True,
# # # # #                 )
# # # # #             # Stop on the TRUE world distance, not the belief estimate: belief_g can collapse much
# # # # #             # faster than the rover actually moves (its centroid drifts onto near ground as the mask
# # # # #             # grows), which stopped the run ~4-5m short while belief read <1.2m. goal_dist is real
# # # # #             # (computed from the actual world goal position each tick) -- always use it to arrive.
# # # # #             belief_dist = (
# # # # #                 float(np.hypot(belief_g[0], belief_g[1]))
# # # # #                 if belief_g is not None
# # # # #                 else float("nan")
# # # # #             )
# # # # #             if goal_dist <= float(args.stop_dist):
# # # # #                 print(
# # # # #                     f"Reached goal at step {step} dist={goal_dist:.2f}m (belief={belief_dist:.2f})",
# # # # #                     flush=True,
# # # # #                 )
# # # # #                 termination_reason = "goal_reached"
# # # # #                 break
# # # # #             if (
# # # # #                 stop_cmd
# # # # #             ):  # language "stop" already halted (action zeroed above) -- end the rollout,
# # # # #                 print(
# # # # #                     f"Stopped by language command at step {step} dist={goal_dist:.2f}m",
# # # # #                     flush=True,
# # # # #                 )
# # # # #                 termination_reason = "stop_command"
# # # # #                 break
# # # # #     except BaseException:
# # # # #         # Make sure the structured logs are flushed and closed even if the rollout crashes
# # # # #         # mid-episode -- the happy-path finalize() call below is skipped once this re-raises.
# # # # #         episode_logger.finalize({"success": False, "termination_reason": "exception"})
# # # # #         raise
# # # # #     finally:
# # # # #         sim.close()

# # # # #     print(
# # # # #         f"[CBF diag] cbf_active={cbf_active} hard_gate_fired={hard_gate_fired} "
# # # # #         f"escape_active={escape_active}",
# # # # #         flush=True,
# # # # #     )
# # # # #     success = bool(
# # # # #         rows["goal_distance"] and rows["goal_distance"][-1] <= float(args.stop_dist)
# # # # #     )
# # # # #     episode_logger.finalize(
# # # # #         {"success": success, "termination_reason": termination_reason}
# # # # #     )
# # # # #     npz_path = out_dir / "rollout.npz"
# # # # #     np.savez_compressed(
# # # # #         npz_path,
# # # # #         rgb=np.stack(rows["rgb"]).astype(np.uint8),
# # # # #         depth=np.stack(rows["depth"]).astype(np.float32),
# # # # #         goal_mask=np.stack(rows["goal_mask"]).astype(np.uint8),
# # # # #         obstacle_mask=np.stack(rows["obstacle_mask"]).astype(np.uint8),
# # # # #         seg_masks=np.stack(rows["seg_masks"]).astype(np.uint8),
# # # # #         pose=np.stack(rows["pose"]).astype(np.float32),
# # # # #         proprio=np.stack(rows["proprio"]).astype(np.float32),
# # # # #         action_3d=np.stack(rows["action_3d"]).astype(np.float32),
# # # # #         pred_chunk=np.stack(rows["pred_chunk"]).astype(np.float32),
# # # # #         goal_visible_pixels=np.asarray(rows["goal_visible_pixels"], dtype=np.int32),
# # # # #         goal_u=np.asarray(rows["goal_u"], dtype=np.float32),
# # # # #         goal_v=np.asarray(rows["goal_v"], dtype=np.float32),
# # # # #         goal_distance=np.asarray(rows["goal_distance"], dtype=np.float32),
# # # # #         obstacle_visible_pixels=np.asarray(
# # # # #             rows["obstacle_visible_pixels"], dtype=np.int32
# # # # #         ),
# # # # #         obstacle_u=np.asarray(rows["obstacle_u"], dtype=np.float32),
# # # # #         obstacle_v=np.asarray(rows["obstacle_v"], dtype=np.float32),
# # # # #         obstacle_distance=np.asarray(rows["obstacle_distance"], dtype=np.float32),
# # # # #         belief_fwd=np.asarray(rows["belief_fwd"], dtype=np.float32),
# # # # #         belief_left=np.asarray(rows["belief_left"], dtype=np.float32),
# # # # #         goal_frame_fraction=np.asarray(rows["goal_frame_fraction"], dtype=np.float32),
# # # # #         cone_correction_step0=np.asarray(
# # # # #             rows["cone_correction_step0"], dtype=np.float32
# # # # #         ),
# # # # #         cone_correction_last=np.asarray(rows["cone_correction_last"], dtype=np.float32),
# # # # #         hard_gate_tick=np.asarray(rows["hard_gate_tick"], dtype=bool),
# # # # #         goal_position=goal.astype(np.float32),
# # # # #         obstacle_position=(
# # # # #             ghost_obstacle.astype(np.float32)
# # # # #             if ghost_obstacle is not None
# # # # #             else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
# # # # #         ),
# # # # #         success=np.asarray(success, dtype=bool),
# # # # #         hz=np.asarray(float(args.hz), dtype=np.float32),
# # # # #     )
# # # # #     manifest = {
# # # # #         "success": success,
# # # # #         "frames": len(rows["rgb"]),
# # # # #         "final_distance": (
# # # # #             float(rows["goal_distance"][-1]) if rows["goal_distance"] else None
# # # # #         ),
# # # # #         "goal_position": goal.tolist(),
# # # # #         "ghost_obstacle_position": (
# # # # #             ghost_obstacle.tolist() if ghost_obstacle is not None else None
# # # # #         ),
# # # # #         "ckpt": str(Path(args.ckpt).expanduser().resolve()),
# # # # #         "scene": str(Path(args.scene).expanduser().resolve()),
# # # # #         "terrain_mode": terrain.mode,
# # # # #         "scene_height_flip_x": bool(args.scene_height_flip_x),
# # # # #         "scene_height_flip_z": bool(args.scene_height_flip_z),
# # # # #         "scene_height_swap_xz": bool(args.scene_height_swap_xz),
# # # # #         "clearance": float(args.clearance),
# # # # #         "pose_terrain_radius": float(args.pose_terrain_radius),
# # # # #         "goal_height": float(args.goal_height),
# # # # #         "goal_terrain_radius": float(args.goal_terrain_radius),
# # # # #         "replan_every": replan_every,
# # # # #         "cbf_active": cbf_active,
# # # # #         "hard_gate_fired": hard_gate_fired,
# # # # #         "escape_active": escape_active,
# # # # #         "cbf_metric": args.cbf_metric,
# # # # #         "cbf_cov_mode": args.cbf_cov_mode,
# # # # #         "cbf_radius_mode": args.cbf_radius_mode,
# # # # #         "npz": str(npz_path),
# # # # #     }
# # # # #     with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
# # # # #         json.dump(manifest, f, indent=2)
# # # # #     if args.save_video:
# # # # #         save_video(
# # # # #             video_frames,
# # # # #             out_dir / "rollout.mp4",
# # # # #             fps=max(float(args.hz) / max(int(args.save_every), 1), 1.0),
# # # # #         )
# # # # #     print(f"Saved rollout: {npz_path}", flush=True)
# # # # #     print(f"Output dir   : {out_dir}", flush=True)


# # # # # def dwa_action(goal_bearing_body, obstacle_point, v_prev, w_prev, dt, args):
# # # # #     """Classic Dynamic Window Approach -- a REACTIVE, from-scratch local planner with NO learned
# # # # #     policy involved, used as an ablation baseline against the collision cone.

# # # # #     Sample every (v, w) reachable from (v_prev, w_prev) under the acceleration limits (the
# # # # #     "dynamic window"); forward-simulate each for --dwa-predict-time seconds assuming the obstacle
# # # # #     stays fixed in the current body frame (standard DWA simplification); reject any candidate that
# # # # #     would come within --dwa-obstacle-radius of it; score the survivors by heading-to-goal +
# # # # #     clearance + forward speed; return the single best (v, w). This REPLACES the diffusion policy's
# # # # #     action entirely for that tick -- DWA is a full navigation decision, not a filter on top of one.
# # # # #     """
# # # # #     max_v, max_w = float(args.max_forward_speed), float(args.max_yaw_rate)
# # # # #     dv, dw = float(args.dwa_max_accel) * dt, float(args.dwa_max_yaw_accel) * dt
# # # # #     v_lo, v_hi = max(0.0, v_prev - dv), min(max_v, v_prev + dv)
# # # # #     w_lo, w_hi = max(-max_w, w_prev - dw), min(max_w, w_prev + dw)
# # # # #     steps = max(int(float(args.dwa_predict_time) / dt), 1)
# # # # #     ox, oy = (
# # # # #         (float(obstacle_point[0]), float(obstacle_point[1]))
# # # # #         if obstacle_point is not None
# # # # #         else (1e9, 1e9)
# # # # #     )

# # # # #     best_score, best = -1e18, (0.0, 0.0)
# # # # #     for v in np.linspace(v_lo, v_hi, max(int(args.dwa_v_samples), 1)):
# # # # #         for w in np.linspace(w_lo, w_hi, max(int(args.dwa_w_samples), 1)):
# # # # #             px = py = pth = 0.0
# # # # #             min_clear = float("inf")
# # # # #             collided = False
# # # # #             for _ in range(steps):
# # # # #                 pth += float(w) * dt
# # # # #                 px += float(v) * math.cos(pth) * dt
# # # # #                 py += float(v) * math.sin(pth) * dt
# # # # #                 d = math.hypot(ox - px, oy - py)
# # # # #                 min_clear = min(min_clear, d)
# # # # #                 if d < float(args.dwa_obstacle_radius):
# # # # #                     collided = True
# # # # #                     break
# # # # #             if collided:
# # # # #                 continue  # infeasible candidate -> hard-rejected (classic DWA, not a soft cost)
# # # # #             heading_score = math.cos(
# # # # #                 goal_bearing_body - pth
# # # # #             )  # 1.0 = ends pointed at the goal
# # # # #             clearance_score = min(
# # # # #                 min_clear, 3.0
# # # # #             )  # saturate so far obstacles don't dominate
# # # # #             velocity_score = float(v) / max(max_v, 1e-6)
# # # # #             score = (
# # # # #                 float(args.dwa_heading_weight) * heading_score
# # # # #                 + float(args.dwa_clearance_weight) * clearance_score
# # # # #                 + float(args.dwa_velocity_weight) * velocity_score
# # # # #             )
# # # # #             if score > best_score:
# # # # #                 best_score, best = score, (float(v), float(w))
# # # # #     if best_score <= -1e17:
# # # # #         return (
# # # # #             0.0,
# # # # #             0.0,
# # # # #         )  # every sampled candidate collides -> full stop (a known DWA failure mode:
# # # # #     return best  # it has no escape when the whole dynamic window is blocked)


# # # # # def planar_goal_bearing(position: np.ndarray, yaw: float, goal: np.ndarray) -> float:
# # # # #     dx = float(goal[0] - position[0])
# # # # #     dz = float(goal[2] - position[2])
# # # # #     desired_yaw = math.atan2(-dx, -dz)
# # # # #     return wrap_angle(desired_yaw - float(yaw))


# # # # # def integrate_mars(
# # # # #     position: np.ndarray, yaw: float, action_3d: np.ndarray, dt: float
# # # # # ) -> Tuple[np.ndarray, float]:
# # # # #     v_fwd, v_lat, yaw_rate = [
# # # # #         float(x) for x in np.asarray(action_3d, dtype=np.float32).reshape(-1)[:3]
# # # # #     ]
# # # # #     fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
# # # # #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# # # # #     out = np.asarray(position, dtype=np.float32).copy()
# # # # #     out[0] += (fwd_x * v_fwd + left_x * v_lat) * float(dt)
# # # # #     out[2] += (fwd_z * v_fwd + left_z * v_lat) * float(dt)
# # # # #     return out, float(yaw + yaw_rate * float(dt))


# # # # # def wrap_angle(angle: float) -> float:
# # # # #     return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# # # # # def orbit_chunk(
# # # # #     position, yaw, ghost_obstacle, side, H, dt, r_gate, kr, cruise, pursuit_kp, max_yaw
# # # # # ):
# # # # #     """Roll the ORBIT controller forced to `side` for H steps -> an [H, 3] action chunk
# # # # #     ([v_fwd, v_lat, yaw], already in the policy's action units) that goes around the obstacle
# # # # #     on that side. side=+1 passes on the obstacle's left, -1 on its right. This is the
# # # # #     counterfactual training TARGET for the language adapter: same observation, the two chunks
# # # # #     differ only by the requested homotopy class -> the text must carry the difference.
# # # # #     """
# # # # #     p = np.asarray(position, np.float32).copy()
# # # # #     y = float(yaw)
# # # # #     out = []
# # # # #     for _ in range(int(H)):
# # # # #         right, _u, fwd = camera_coords(ghost_obstacle, p, y)
# # # # #         ox, oy = float(fwd), float(-right)  # [forward, left]
# # # # #         L = math.hypot(ox, oy)
# # # # #         phi = math.atan2(oy, ox)
# # # # #         corr = max(-1.2, min(1.2, float(kr) * (L - float(r_gate))))
# # # # #         psi = wrap_angle(phi + float(side) * (0.5 * math.pi - corr))
# # # # #         yaw_cmd = float(
# # # # #             np.clip(float(pursuit_kp) * psi, -float(max_yaw), float(max_yaw))
# # # # #         )
# # # # #         a = np.asarray([float(cruise), 0.0, yaw_cmd], np.float32)
# # # # #         out.append(a)
# # # # #         p, y = integrate_mars(p, y, a, dt)
# # # # #     return np.stack(out, 0)


# # # # # def command_intent(text):
# # # # #     """Map a real-time language command to an intent: 'left' / 'right' / 'stop' / '' (default).
# # # # #     Keyword now; swap for the embedding grounder or a VLM call. Same interface either way.
# # # # #     """
# # # # #     t = (text or "").strip().lower()
# # # # #     if not t:
# # # # #         return ""
# # # # #     if any(k in t for k in ("stop", "halt", "brake", "wait", "hold")):
# # # # #         return "stop"
# # # # #     left, right = "left" in t, "right" in t
# # # # #     if left and not right:
# # # # #         return "left"
# # # # #     if right and not left:
# # # # #         return "right"
# # # # #     return ""  # navigate normally / unrecognised -> default geometric behaviour


# # # # # def brake_chunk(v0, H):
# # # # #     """Decelerate-to-stop chunk: forward ramps v0 -> 0, no turn. Target for 'stop before the
# # # # #     obstacle' -- a physically feasible braking horizon, distinct from all the moving chunks.
# # # # #     """
# # # # #     out = []
# # # # #     for k in range(int(H)):
# # # # #         v = float(v0) * max(0.0, 1.0 - k / max(int(H) - 1, 1))
# # # # #         out.append(np.asarray([v, 0.0, 0.0], np.float32))
# # # # #     return np.stack(out, 0)


# # # # # def back_chunk(v_back, H):
# # # # #     """Straight-reverse chunk: constant NEGATIVE forward velocity, no turn -- retreat away from
# # # # #     the obstacle in a straight line. Target for the 'back'/'reverse' instruction. Geometry-
# # # # #     independent (same for every observation), unlike orbit_chunk/goal_chunk."""
# # # # #     v = -abs(float(v_back))
# # # # #     return np.stack([np.asarray([v, 0.0, 0.0], np.float32) for _ in range(int(H))], 0)


# # # # # def goal_chunk(position, yaw, goal, H, dt, cruise, kp, max_yaw):
# # # # #     """Pursuit-to-goal chunk: steer toward the goal bearing at cruise, ignoring the obstacle.
# # # # #     Target for 'navigate normally / steer to the goal mask' -- the policy's default goal-seeking
# # # # #     behaviour (also the prior-preservation class: the adapter should ~reproduce the default).
# # # # #     """
# # # # #     p = np.asarray(position, np.float32).copy()
# # # # #     y = float(yaw)
# # # # #     out = []
# # # # #     for _ in range(int(H)):
# # # # #         bearing = planar_goal_bearing(p, y, goal)
# # # # #         yaw_cmd = float(np.clip(float(kp) * bearing, -float(max_yaw), float(max_yaw)))
# # # # #         a = np.asarray([float(cruise), 0.0, yaw_cmd], np.float32)
# # # # #         out.append(a)
# # # # #         p, y = integrate_mars(p, y, a, dt)
# # # # #     return np.stack(out, 0)


# # # # # if __name__ == "__main__":
# # # # #     main()

# # # # from __future__ import annotations

# # # # import argparse
# # # # import io
# # # # import json
# # # # import math
# # # # import os
# # # # import socket
# # # # import subprocess
# # # # import sys
# # # # import time
# # # # from dataclasses import dataclass
# # # # from pathlib import Path
# # # # from typing import Any, Optional, Sequence

# # # # import habitat_sim
# # # # import numpy as np
# # # # import quaternion
# # # # import requests
# # # # from habitat_sim.agent import AgentConfiguration
# # # # from PIL import Image, ImageDraw


# # # # HERE = Path(__file__).resolve().parent
# # # # SIZE_X = 50.0
# # # # SIZE_Z = 50.0
# # # # SIZE_Y = 4.820803273566
# # # # MESH_GOAL_ID = 10000
# # # # MESH_OBSTACLE_ID = 2


# # # # @dataclass(frozen=True)
# # # # class NavDPS2DiffOutput:
# # # #     trajectory: np.ndarray
# # # #     all_trajectories: np.ndarray
# # # #     all_values: np.ndarray
# # # #     selected_index: int
# # # #     fallback_stop: bool
# # # #     escape_turn: bool
# # # #     valid_obstacle_points: int
# # # #     selected_circulation_sign: float
# # # #     candidate_circulation_signs: np.ndarray
# # # #     selected_barrier_energy: float
# # # #     selected_circulation_energy: float
# # # #     minimum_clearance: np.ndarray
# # # #     selected_minimum_clearance: float
# # # #     mean_guidance_noise_correction: float
# # # #     final_guidance_noise_correction: float
# # # #     maximum_guidance_noise_correction: float
# # # #     mean_final_effective_sample_size: float


# # # # class NavDPS2DiffClient:
# # # #     def __init__(self, server_url: str, timeout: float = 180.0):
# # # #         self.server_url = server_url.rstrip("/")
# # # #         self.timeout = float(timeout)

# # # #     def reset(
# # # #         self,
# # # #         intrinsic: np.ndarray,
# # # #         *,
# # # #         stop_threshold: float = -3.0,
# # # #         batch_size: int = 1,
# # # #     ) -> str:
# # # #         intrinsic = np.asarray(intrinsic, dtype=np.float32)
# # # #         if intrinsic.shape != (3, 3):
# # # #             raise ValueError(f"intrinsic must have shape [3,3], got {intrinsic.shape}")
# # # #         response = requests.post(
# # # #             f"{self.server_url}/navigator_reset",
# # # #             json={
# # # #                 "intrinsic": intrinsic.tolist(),
# # # #                 "stop_threshold": float(stop_threshold),
# # # #                 "batch_size": int(batch_size),
# # # #             },
# # # #             timeout=self.timeout,
# # # #         )
# # # #         self._raise_for_error(response)
# # # #         return str(response.json().get("algo", ""))

# # # #     def plan(
# # # #         self,
# # # #         *,
# # # #         goal_xy: np.ndarray,
# # # #         rgb: np.ndarray,
# # # #         depth: np.ndarray,
# # # #         obstacle_pixels: np.ndarray,
# # # #         goal_mode: str = "point",
# # # #         forced_circulation_sign: float = 0.0,
# # # #     ) -> NavDPS2DiffOutput:
# # # #         goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
# # # #         if goal_xy.shape != (2,):
# # # #             raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")
# # # #         if goal_mode not in {"point", "pixel"}:
# # # #             raise ValueError("goal_mode must be point or pixel")
# # # #         forced_circulation_sign = float(forced_circulation_sign)
# # # #         if forced_circulation_sign not in {-1.0, 0.0, 1.0}:
# # # #             raise ValueError("forced_circulation_sign must be -1, 0, or +1")

# # # #         rgb = np.asarray(rgb, dtype=np.uint8)
# # # #         if rgb.ndim != 3 or rgb.shape[-1] < 3:
# # # #             raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
# # # #         rgb = rgb[..., :3]

# # # #         depth = np.asarray(depth, dtype=np.float32)
# # # #         if depth.ndim == 3 and depth.shape[-1] == 1:
# # # #             depth = depth[..., 0]
# # # #         if depth.shape != rgb.shape[:2]:
# # # #             raise ValueError(f"depth/rgb shape mismatch: {depth.shape} vs {rgb.shape[:2]}")

# # # #         if goal_mode == "pixel":
# # # #             if not np.all(np.isfinite(goal_xy)) or not np.allclose(
# # # #                 goal_xy, np.round(goal_xy)
# # # #             ):
# # # #                 raise ValueError("PixelGoal must be integer [u,v]")
# # # #             goal_xy = np.round(goal_xy).astype(np.int64)
# # # #             if not (0 <= goal_xy[0] < rgb.shape[1] and 0 <= goal_xy[1] < rgb.shape[0]):
# # # #                 raise ValueError("PixelGoal lies outside the RGB image")

# # # #         pixels = np.asarray(obstacle_pixels)
# # # #         if pixels.size == 0:
# # # #             pixels = np.zeros((0, 2), dtype=np.int32)
# # # #         else:
# # # #             pixels = pixels.reshape(-1, 2)
# # # #             if not np.all(np.isfinite(pixels)):
# # # #                 raise ValueError("obstacle pixels must be finite")
# # # #             if not np.allclose(pixels, np.round(pixels)):
# # # #                 raise ValueError("obstacle pixels must be integer [u,v] coordinates")
# # # #             pixels = np.round(pixels).astype(np.int32)

# # # #         rgb_bytes = io.BytesIO()
# # # #         Image.fromarray(rgb, mode="RGB").save(rgb_bytes, format="JPEG", quality=95)
# # # #         depth_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
# # # #         depth_bytes = io.BytesIO()
# # # #         Image.fromarray(depth_u16).save(depth_bytes, format="PNG")

# # # #         endpoint = "pixelgoal_step" if goal_mode == "pixel" else "pointgoal_step"
# # # #         response = requests.post(
# # # #             f"{self.server_url}/{endpoint}",
# # # #             files={
# # # #                 "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
# # # #                 "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
# # # #             },
# # # #             data={
# # # #                 "goal_data": json.dumps(
# # # #                     {
# # # #                         "goal_x": [float(goal_xy[0])],
# # # #                         "goal_y": [float(goal_xy[1])],
# # # #                         "obstacle_pixels": [pixels.tolist()],
# # # #                         "forced_circulation_signs": [forced_circulation_sign],
# # # #                     }
# # # #                 )
# # # #             },
# # # #             timeout=self.timeout,
# # # #         )
# # # #         self._raise_for_error(response)
# # # #         payload = response.json()
# # # #         diagnostics = payload["s2diff"]
# # # #         trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
# # # #         all_trajectories = np.asarray(payload["all_trajectory"], dtype=np.float32)
# # # #         all_values = np.asarray(payload["all_values"], dtype=np.float32)

# # # #         return NavDPS2DiffOutput(
# # # #             trajectory=trajectory[0],
# # # #             all_trajectories=all_trajectories[0],
# # # #             all_values=all_values[0],
# # # #             selected_index=int(diagnostics["selected_index"][0]),
# # # #             fallback_stop=bool(diagnostics["fallback_stop"][0]),
# # # #             escape_turn=bool(diagnostics["escape_turn"][0]),
# # # #             valid_obstacle_points=int(diagnostics["valid_obstacle_points"][0]),
# # # #             selected_circulation_sign=float(
# # # #                 diagnostics["selected_circulation_sign"][0]
# # # #             ),
# # # #             candidate_circulation_signs=np.asarray(
# # # #                 diagnostics["candidate_circulation_signs"][0], dtype=np.float32
# # # #             ),
# # # #             selected_barrier_energy=float(diagnostics["selected_barrier_energy"][0]),
# # # #             selected_circulation_energy=float(
# # # #                 diagnostics["selected_circulation_energy"][0]
# # # #             ),
# # # #             minimum_clearance=np.asarray(
# # # #                 diagnostics["minimum_clearance"][0], dtype=np.float32
# # # #             ),
# # # #             selected_minimum_clearance=float(
# # # #                 diagnostics["selected_minimum_clearance"][0]
# # # #             ),
# # # #             mean_guidance_noise_correction=float(
# # # #                 diagnostics["mean_guidance_noise_correction"][0]
# # # #             ),
# # # #             final_guidance_noise_correction=float(
# # # #                 diagnostics["final_guidance_noise_correction"][0]
# # # #             ),
# # # #             maximum_guidance_noise_correction=float(
# # # #                 diagnostics["maximum_guidance_noise_correction"][0]
# # # #             ),
# # # #             mean_final_effective_sample_size=float(
# # # #                 diagnostics.get("mean_final_effective_sample_size", [0.0])[0]
# # # #             ),
# # # #         )

# # # #     @staticmethod
# # # #     def _raise_for_error(response: requests.Response) -> None:
# # # #         try:
# # # #             payload = response.json()
# # # #         except ValueError:
# # # #             payload = None
# # # #         if isinstance(payload, dict) and "error" in payload:
# # # #             raise RuntimeError(str(payload["error"]))
# # # #         response.raise_for_status()


# # # # def port_is_open(host: str, port: int) -> bool:
# # # #     try:
# # # #         with socket.create_connection((host, port), timeout=1.0):
# # # #             return True
# # # #     except OSError:
# # # #         return False


# # # # def wait_for_server(
# # # #     process: subprocess.Popen[Any], host: str, port: int, timeout: float
# # # # ) -> None:
# # # #     deadline = time.time() + float(timeout)
# # # #     while time.time() < deadline:
# # # #         if process.poll() is not None:
# # # #             raise RuntimeError(
# # # #                 f"NavDP/S2Diff server exited with code {process.returncode}"
# # # #             )
# # # #         if port_is_open(host, port):
# # # #             return
# # # #         time.sleep(1.0)
# # # #     raise TimeoutError(f"NavDP server did not open port {port} within {timeout}s")


# # # # def stop_server(process: Optional[subprocess.Popen[Any]]) -> None:
# # # #     if process is None or process.poll() is not None:
# # # #         return
# # # #     process.terminate()
# # # #     try:
# # # #         process.wait(timeout=10.0)
# # # #     except subprocess.TimeoutExpired:
# # # #         process.kill()
# # # #         process.wait()


# # # # def start_server(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
# # # #     if not args.start_server:
# # # #         return None
# # # #     if port_is_open(args.server_host, args.server_port):
# # # #         raise RuntimeError(
# # # #             f"port {args.server_port} is already in use; use --no-start-server "
# # # #             "to connect to an existing guided server"
# # # #         )

# # # #     navdp_root = Path(args.navdp_root).expanduser().resolve()
# # # #     checkpoint = Path(args.navdp_checkpoint).expanduser().resolve()
# # # #     server_dir = navdp_root / "baselines" / "navdp"
# # # #     server_file = server_dir / "navdp_s2diff_server.py"
# # # #     if not server_file.is_file():
# # # #         raise FileNotFoundError(f"guided server not found: {server_file}")
# # # #     if not checkpoint.is_file():
# # # #         raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint}")

# # # #     command = [
# # # #         str(args.navdp_python),
# # # #         str(server_file),
# # # #         "--checkpoint",
# # # #         str(checkpoint),
# # # #         "--device",
# # # #         str(args.navdp_device),
# # # #         "--planner-mode",
# # # #         str(args.planner_mode),
# # # #         "--seed",
# # # #         str(args.seed),
# # # #         "--port",
# # # #         str(args.server_port),
# # # #         "--candidates",
# # # #         str(args.candidates),
# # # #         "--particles",
# # # #         str(args.particles),
# # # #         "--particle-std",
# # # #         str(args.particle_std),
# # # #         "--gradient-steps",
# # # #         str(args.gradient_steps),
# # # #         "--gradient-step-size",
# # # #         str(args.gradient_step_size),
# # # #         "--guidance-strength",
# # # #         str(args.guidance_strength),
# # # #         "--temperature",
# # # #         str(args.temperature),
# # # #         "--safe-distance",
# # # #         str(args.safe_distance),
# # # #         "--hard-collision-distance",
# # # #         str(args.hard_collision_distance),
# # # #         "--robot-radius",
# # # #         str(args.robot_radius),
# # # #         "--safety-weight",
# # # #         str(args.safety_weight),
# # # #         "--barrier-weight",
# # # #         str(args.barrier_weight),
# # # #         "--barrier-rate",
# # # #         str(args.barrier_rate),
# # # #         "--circulation-weight",
# # # #         str(args.circulation_weight),
# # # #         "--circulation-activation-distance",
# # # #         str(args.circulation_activation_distance),
# # # #         "--circulation-activation-sharpness",
# # # #         str(args.circulation_activation_sharpness),
# # # #         "--minimum-circulation-progress",
# # # #         str(args.minimum_circulation_progress),
# # # #         "--blocking-alignment-threshold",
# # # #         str(args.blocking_alignment_threshold),
# # # #         "--circulation-switch-weight",
# # # #         str(args.circulation_switch_weight),
# # # #         "--escape-lateral-target",
# # # #         str(args.escape_lateral_target),
# # # #         "--minimum-obstacle-depth",
# # # #         str(args.minimum_obstacle_depth),
# # # #         "--maximum-obstacle-depth",
# # # #         str(args.maximum_obstacle_depth),
# # # #         "--maximum-obstacle-pixels",
# # # #         str(args.maximum_obstacle_pixels),
# # # #     ]
# # # #     particle_flags = {
# # # #         "particle-anchor": args.particle_anchor,
# # # #         "particle-energy-reweighting": args.particle_energy_reweighting,
# # # #         "particle-collision-mask": args.particle_collision_mask,
# # # #         "particle-noise-schedule": args.particle_noise_schedule,
# # # #         "progressive-guidance": args.progressive_guidance,
# # # #     }
# # # #     for name, enabled in particle_flags.items():
# # # #         command.append(f"--{name}" if enabled else f"--no-{name}")
# # # #     command.append("--remove-critic" if args.remove_critic else "--no-remove-critic")
# # # #     print("[server]", " ".join(command), flush=True)
# # # #     process = subprocess.Popen(command, cwd=str(server_dir))
# # # #     wait_for_server(process, args.server_host, args.server_port, args.server_timeout)
# # # #     return process


# # # # def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
# # # #     height, width = grid.shape
# # # #     x0 = int(np.floor(px))
# # # #     y0 = int(np.floor(py))
# # # #     x1 = min(x0 + 1, width - 1)
# # # #     y1 = min(y0 + 1, height - 1)
# # # #     tx = px - x0
# # # #     ty = py - y0
# # # #     top = float(grid[y0, x0]) * (1.0 - tx) + float(grid[y0, x1]) * tx
# # # #     bottom = float(grid[y1, x0]) * (1.0 - tx) + float(grid[y1, x1]) * tx
# # # #     return top * (1.0 - ty) + bottom * ty


# # # # class TerrainHeight:
# # # #     def __init__(
# # # #         self,
# # # #         *,
# # # #         mode: str,
# # # #         heightmap: Optional[Path],
# # # #         obj: Optional[Path],
# # # #         flat_y: float,
# # # #         size_x: float,
# # # #         size_z: float,
# # # #         size_y: float,
# # # #         flip_x: bool,
# # # #         flip_z: bool,
# # # #         swap_xz: bool,
# # # #     ):
# # # #         if mode == "auto":
# # # #             mode = "heightmap" if heightmap and heightmap.exists() else (
# # # #                 "obj" if obj and obj.exists() else "flat"
# # # #             )
# # # #         self.mode = mode
# # # #         self.flat_y = float(flat_y)
# # # #         self.size_x = float(size_x)
# # # #         self.size_z = float(size_z)
# # # #         self.size_y = float(size_y)
# # # #         self.flip_x = bool(flip_x)
# # # #         self.flip_z = bool(flip_z)
# # # #         self.swap_xz = bool(swap_xz)
# # # #         self.height: Optional[np.ndarray] = None
# # # #         self.obj_xs: Optional[np.ndarray] = None
# # # #         self.obj_zs: Optional[np.ndarray] = None
# # # #         self.obj_h: Optional[np.ndarray] = None

# # # #         if mode == "heightmap":
# # # #             if heightmap is None or not heightmap.exists():
# # # #                 raise FileNotFoundError(f"heightmap not found: {heightmap}")
# # # #             array = np.asarray(Image.open(heightmap))
# # # #             if array.ndim == 3:
# # # #                 array = array[..., 0]
# # # #             array = array.astype(np.float32)
# # # #             array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
# # # #             self.height = array * self.size_y - float(np.mean(array * self.size_y))
# # # #         elif mode == "obj":
# # # #             if obj is None or not obj.exists():
# # # #                 raise FileNotFoundError(f"terrain OBJ not found: {obj}")
# # # #             vertices = []
# # # #             with obj.open("r", encoding="utf-8", errors="ignore") as file:
# # # #                 for line in file:
# # # #                     if line.startswith("v "):
# # # #                         parts = line.split()
# # # #                         if len(parts) >= 4:
# # # #                             vertices.append(tuple(float(value) for value in parts[1:4]))
# # # #             if not vertices:
# # # #                 raise RuntimeError(f"no vertices found in {obj}")
# # # #             array = np.asarray(vertices, dtype=np.float32)
# # # #             xs = np.unique(array[:, 0])
# # # #             zs = np.unique(array[:, 1])
# # # #             grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
# # # #             x_index = {float(value): index for index, value in enumerate(xs.tolist())}
# # # #             z_index = {float(value): index for index, value in enumerate(zs.tolist())}
# # # #             for x, z, height in array:
# # # #                 grid[z_index[float(z)], x_index[float(x)]] = height
# # # #             self.obj_xs = xs
# # # #             self.obj_zs = zs
# # # #             self.obj_h = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
# # # #         elif mode != "flat":
# # # #             raise ValueError(f"unknown terrain mode: {mode}")

# # # #     def _map(self, x: float, z: float) -> tuple[float, float]:
# # # #         if self.swap_xz:
# # # #             x, z = z, x
# # # #         u = (x + self.size_x / 2.0) / self.size_x
# # # #         v = (z + self.size_z / 2.0) / self.size_z
# # # #         if self.flip_x:
# # # #             u = 1.0 - u
# # # #         if self.flip_z:
# # # #             v = 1.0 - v
# # # #         return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

# # # #     def __call__(self, x: float, z: float) -> float:
# # # #         if self.mode == "flat":
# # # #             return self.flat_y
# # # #         if self.mode == "heightmap":
# # # #             assert self.height is not None
# # # #             u, v = self._map(x, z)
# # # #             return bilinear_grid(
# # # #                 self.height, u * (self.height.shape[1] - 1), v * (self.height.shape[0] - 1)
# # # #             )
# # # #         assert self.obj_xs is not None and self.obj_zs is not None and self.obj_h is not None
# # # #         xx = float(np.clip(x, self.obj_xs[0], self.obj_xs[-1]))
# # # #         zz = float(np.clip(z, self.obj_zs[0], self.obj_zs[-1]))
# # # #         column = int(np.clip(np.searchsorted(self.obj_xs, xx) - 1, 0, len(self.obj_xs) - 2))
# # # #         row = int(np.clip(np.searchsorted(self.obj_zs, zz) - 1, 0, len(self.obj_zs) - 2))
# # # #         x0, x1 = float(self.obj_xs[column]), float(self.obj_xs[column + 1])
# # # #         z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
# # # #         tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
# # # #         tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
# # # #         top = float(self.obj_h[row, column]) * (1.0 - tx) + float(self.obj_h[row, column + 1]) * tx
# # # #         bottom = float(self.obj_h[row + 1, column]) * (1.0 - tx) + float(self.obj_h[row + 1, column + 1]) * tx
# # # #         return top * (1.0 - tz) + bottom * tz

# # # #     def local_height_max(self, x: float, z: float, radius: float, samples: int = 5) -> float:
# # # #         if radius <= 1e-6:
# # # #             return float(self(x, z))
# # # #         values = [
# # # #             float(self(x + dx, z + dz))
# # # #             for dx in np.linspace(-radius, radius, samples)
# # # #             for dz in np.linspace(-radius, radius, samples)
# # # #             if dx * dx + dz * dz <= radius * radius + 1e-8
# # # #         ]
# # # #         return max(values) if values else float(self(x, z))


# # # # def make_sensor(
# # # #     uuid: str, sensor_type: Any, height: int, width: int, hfov_deg: float
# # # # ) -> habitat_sim.CameraSensorSpec:
# # # #     specification = habitat_sim.CameraSensorSpec()
# # # #     specification.uuid = uuid
# # # #     specification.sensor_type = sensor_type
# # # #     specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
# # # #     specification.resolution = [int(height), int(width)]
# # # #     specification.position = [0.0, 0.0, 0.0]
# # # #     specification.hfov = float(hfov_deg)
# # # #     return specification


# # # # def make_simulator(
# # # #     scene: Path,
# # # #     height: int,
# # # #     width: int,
# # # #     hfov_deg: float,
# # # #     *,
# # # #     with_semantic: bool,
# # # # ):
# # # #     simulator_configuration = habitat_sim.SimulatorConfiguration()
# # # #     simulator_configuration.scene_id = str(scene.expanduser().resolve())
# # # #     simulator_configuration.enable_physics = False
# # # #     sensors = [
# # # #         make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
# # # #         make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
# # # #     ]
# # # #     if with_semantic:
# # # #         sensors.append(
# # # #             make_sensor(
# # # #                 "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
# # # #             )
# # # #         )
# # # #     agent_configuration = AgentConfiguration()
# # # #     agent_configuration.sensor_specifications = sensors
# # # #     return habitat_sim.Simulator(
# # # #         habitat_sim.Configuration(simulator_configuration, [agent_configuration])
# # # #     )

# # # # def set_agent_pose(agent: Any, position: np.ndarray, yaw: float) -> None:
# # # #     state = agent.get_state()
# # # #     state.position = np.asarray(position, dtype=np.float32)
# # # #     state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
# # # #     agent.set_state(state)


# # # # def rgb_depth(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
# # # #     rgb = np.asarray(observation["rgb"])
# # # #     if rgb.ndim == 3 and rgb.shape[-1] == 4:
# # # #         rgb = rgb[..., :3]
# # # #     depth = np.asarray(observation["depth"], dtype=np.float32)
# # # #     if depth.ndim == 3:
# # # #         depth = depth[..., 0]
# # # #     return rgb.astype(np.uint8), depth.astype(np.float32)


# # # # def semantic_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
# # # #     semantic = np.asarray(observation["semantic"])
# # # #     if semantic.ndim == 3:
# # # #         semantic = semantic[..., 0]
# # # #     return semantic.astype(np.int32)


# # # # def pixel_to_world(
# # # #     u: float,
# # # #     v: float,
# # # #     depth: float,
# # # #     position: np.ndarray,
# # # #     yaw: float,
# # # #     intrinsic: np.ndarray,
# # # # ) -> np.ndarray:
# # # #     right = (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
# # # #     up = -(v - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
# # # #     forward_vector = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)])
# # # #     right_vector = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)])
# # # #     return (
# # # #         np.asarray(position, dtype=np.float64)
# # # #         + depth * forward_vector
# # # #         + right * right_vector
# # # #         + up * np.asarray([0.0, 1.0, 0.0])
# # # #     )


# # # # def depth_patch_mesh(
# # # #     u_center: float,
# # # #     v_center: float,
# # # #     half_size: int,
# # # #     stride: int,
# # # #     depth: np.ndarray,
# # # #     position: np.ndarray,
# # # #     yaw: float,
# # # #     intrinsic: np.ndarray,
# # # #     *,
# # # #     lift: float,
# # # #     maximum_depth_jump: float = 0.4,
# # # # ) -> tuple[np.ndarray, np.ndarray]:
# # # #     height, width = depth.shape
# # # #     columns = list(
# # # #         range(
# # # #             max(0, int(u_center - half_size)),
# # # #             min(width, int(u_center + half_size) + 1),
# # # #             max(int(stride), 1),
# # # #         )
# # # #     )
# # # #     rows = list(
# # # #         range(
# # # #             max(0, int(v_center - half_size)),
# # # #             min(height, int(v_center + half_size) + 1),
# # # #             max(int(stride), 1),
# # # #         )
# # # #     )
# # # #     indices = -np.ones((len(rows), len(columns)), dtype=np.int64)
# # # #     depths = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
# # # #     vertices: list[tuple[float, float, float]] = []
# # # #     for row_index, v in enumerate(rows):
# # # #         for column_index, u in enumerate(columns):
# # # #             metric_depth = float(depth[v, u])
# # # #             if not np.isfinite(metric_depth) or metric_depth <= 0.1:
# # # #                 continue
# # # #             indices[row_index, column_index] = len(vertices)
# # # #             depths[row_index, column_index] = metric_depth
# # # #             point = pixel_to_world(
# # # #                 u, v, metric_depth, position, yaw, intrinsic
# # # #             ) + float(lift) * np.asarray([0.0, 1.0, 0.0])
# # # #             vertices.append(tuple(float(value) for value in point))

# # # #     faces: list[tuple[int, int, int]] = []
# # # #     for row_index in range(len(rows) - 1):
# # # #         for column_index in range(len(columns) - 1):
# # # #             a = int(indices[row_index, column_index])
# # # #             b = int(indices[row_index, column_index + 1])
# # # #             c = int(indices[row_index + 1, column_index])
# # # #             d = int(indices[row_index + 1, column_index + 1])
# # # #             if min(a, b, c, d) < 0:
# # # #                 continue
# # # #             cell_depths = (
# # # #                 depths[row_index, column_index],
# # # #                 depths[row_index, column_index + 1],
# # # #                 depths[row_index + 1, column_index],
# # # #                 depths[row_index + 1, column_index + 1],
# # # #             )
# # # #             if max(cell_depths) - min(cell_depths) > maximum_depth_jump:
# # # #                 continue
# # # #             faces.append((a, c, d))
# # # #             faces.append((a, d, b))
# # # #     return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# # # # def save_obj(
# # # #     path: Path,
# # # #     vertices: np.ndarray,
# # # #     faces: np.ndarray,
# # # #     *,
# # # #     diffuse_rgb: Optional[tuple[float, float, float]] = None,
# # # # ) -> None:
# # # #     material_name = None
# # # #     if diffuse_rgb is not None:
# # # #         red, green, blue = (float(value) for value in diffuse_rgb)
# # # #         if not all(0.0 <= value <= 1.0 for value in (red, green, blue)):
# # # #             raise ValueError("OBJ diffuse material values must be in [0, 1]")
# # # #         material_name = "mesh_material"
# # # #         material_path = path.with_suffix(".mtl")
# # # #         with material_path.open("w", encoding="utf-8") as material:
# # # #             material.write(f"newmtl {material_name}\n")
# # # #             material.write(f"Ka {0.25 * red:.4f} {0.25 * green:.4f} {0.25 * blue:.4f}\n")
# # # #             material.write(f"Kd {red:.4f} {green:.4f} {blue:.4f}\n")
# # # #             material.write("Ks 0.1000 0.1000 0.1000\n")
# # # #             material.write("Ns 24.0000\n")
# # # #             material.write("d 1.0000\n")
# # # #             material.write("illum 2\n")
# # # #     with path.open("w", encoding="utf-8") as file:
# # # #         if material_name is not None:
# # # #             file.write(f"mtllib {path.with_suffix('.mtl').name}\n")
# # # #             file.write(f"usemtl {material_name}\n")
# # # #         for x, y, z in vertices:
# # # #             file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
# # # #         for a, b, c in faces:
# # # #             file.write(f"f {a + 1} {b + 1} {c + 1}\n")


# # # # def register_semantic_mesh(
# # # #     simulator: Any, mesh_path: Path, semantic_id: int
# # # # ) -> Any:
# # # #     template_manager = simulator.get_object_template_manager()
# # # #     object_manager = simulator.get_rigid_object_manager()
# # # #     template = template_manager.create_new_template(str(mesh_path))
# # # #     template.render_asset_handle = str(mesh_path)
# # # #     template.collision_asset_handle = str(mesh_path)
# # # #     template.is_collidable = False
# # # #     template_id = template_manager.register_template(
# # # #         template, f"s2diff_obstacle_{semantic_id}_{os.path.basename(mesh_path)}"
# # # #     )
# # # #     object_handle = template_manager.get_template_handle_by_id(template_id)
# # # #     obstacle = object_manager.add_object_by_template_handle(object_handle)
# # # #     obstacle.motion_type = habitat_sim.physics.MotionType.KINEMATIC
# # # #     obstacle.collidable = False
# # # #     obstacle.semantic_id = int(semantic_id)
# # # #     return obstacle


# # # # def parse_world_xz(specification: str) -> tuple[float, float]:
# # # #     values = [float(value) for value in str(specification).split(",")]
# # # #     if len(values) != 2 or not np.isfinite(values).all():
# # # #         raise ValueError(
# # # #             f"world mesh position must be finite X,Z, got {specification!r}"
# # # #         )
# # # #     return values[0], values[1]


# # # # def world_box_mesh(
# # # #     center_x: float,
# # # #     base_y: float,
# # # #     center_z: float,
# # # #     half_extent: float,
# # # #     height: float,
# # # # ) -> tuple[np.ndarray, np.ndarray]:
# # # #     """Create a closed axis-aligned box whose vertices are in world coordinates."""

# # # #     if half_extent <= 0.0 or height <= 0.0:
# # # #         raise ValueError("box half extent and height must be positive")
# # # #     x0, x1 = center_x - half_extent, center_x + half_extent
# # # #     z0, z1 = center_z - half_extent, center_z + half_extent
# # # #     y0, y1 = base_y, base_y + height
# # # #     vertices = np.asarray(
# # # #         [
# # # #             [x0, y0, z0],
# # # #             [x1, y0, z0],
# # # #             [x1, y0, z1],
# # # #             [x0, y0, z1],
# # # #             [x0, y1, z0],
# # # #             [x1, y1, z0],
# # # #             [x1, y1, z1],
# # # #             [x0, y1, z1],
# # # #         ],
# # # #         dtype=np.float64,
# # # #     )
# # # #     faces = np.asarray(
# # # #         [
# # # #             [0, 2, 1], [0, 3, 2],
# # # #             [4, 5, 6], [4, 6, 7],
# # # #             [0, 1, 5], [0, 5, 4],
# # # #             [1, 2, 6], [1, 6, 5],
# # # #             [2, 3, 7], [2, 7, 6],
# # # #             [3, 0, 4], [3, 4, 7],
# # # #         ],
# # # #         dtype=np.int64,
# # # #     )
# # # #     return vertices, faces


# # # # def place_world_obstacle_meshes(
# # # #     simulator: Any,
# # # #     terrain: Any,
# # # #     xz_specifications: Sequence[str],
# # # #     output_directory: Path,
# # # #     *,
# # # #     half_extent: float,
# # # #     height: float,
# # # # ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
# # # #     """Place static obstacle boxes at exact world X,Z coordinates."""

# # # #     mesh_directory = output_directory / "meshes"
# # # #     mesh_directory.mkdir(parents=True, exist_ok=True)
# # # #     objects: list[Any] = []
# # # #     centroids: list[np.ndarray] = []
# # # #     geometries: list[np.ndarray] = []
# # # #     for index, specification in enumerate(xz_specifications):
# # # #         center_x, center_z = parse_world_xz(specification)
# # # #         base_y = terrain.local_height_max(center_x, center_z, half_extent)
# # # #         vertices, faces = world_box_mesh(
# # # #             center_x, base_y, center_z, half_extent, height
# # # #         )
# # # #         mesh_path = mesh_directory / f"world_obstacle_{index}.obj"
# # # #         save_obj(
# # # #             mesh_path, vertices, faces, diffuse_rgb=(0.78, 0.16, 0.06)
# # # #         )
# # # #         semantic_id = MESH_OBSTACLE_ID + index
# # # #         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
# # # #         centroid = vertices.mean(axis=0).astype(np.float32)
# # # #         centroids.append(centroid)
# # # #         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
# # # #         print(
# # # #             f"[world-mesh] obstacle={index} semantic_id={semantic_id} "
# # # #             f"center_xz={[center_x, center_z]} half_extent={half_extent:.3f} "
# # # #             f"height={height:.3f}",
# # # #             flush=True,
# # # #         )
# # # #     return objects, centroids, geometries


# # # # def place_world_goal_mesh(
# # # #     simulator: Any,
# # # #     terrain: Any,
# # # #     goal_x: float,
# # # #     goal_z: float,
# # # #     output_directory: Path,
# # # #     *,
# # # #     half_extent: float,
# # # #     height: float,
# # # # ) -> Any:
# # # #     """Place a visible, non-obstacle semantic goal marker at the exact goal."""

# # # #     base_y = terrain.local_height_max(goal_x, goal_z, half_extent)
# # # #     vertices, faces = world_box_mesh(
# # # #         goal_x, base_y, goal_z, half_extent, height
# # # #     )
# # # #     mesh_directory = output_directory / "meshes"
# # # #     mesh_directory.mkdir(parents=True, exist_ok=True)
# # # #     mesh_path = mesh_directory / "goal_marker.obj"
# # # #     save_obj(
# # # #         mesh_path, vertices, faces, diffuse_rgb=(0.08, 0.85, 0.18)
# # # #     )
# # # #     goal_object = register_semantic_mesh(simulator, mesh_path, MESH_GOAL_ID)
# # # #     print(
# # # #         f"[world-mesh] goal semantic_id={MESH_GOAL_ID} "
# # # #         f"center_xz={[goal_x, goal_z]}",
# # # #         flush=True,
# # # #     )
# # # #     return goal_object


# # # # def parse_uv_fraction(specification: str, width: int, height: int) -> tuple[float, float]:
# # # #     u_fraction, v_fraction = (
# # # #         float(value) for value in str(specification).split(",")
# # # #     )
# # # #     if not (0.0 <= u_fraction <= 1.0 and 0.0 <= v_fraction <= 1.0):
# # # #         raise ValueError(
# # # #             f"mesh pixel fraction must be in [0,1], got {specification!r}"
# # # #         )
# # # #     return u_fraction * width, v_fraction * height


# # # # def place_obstacle_meshes(
# # # #     simulator: Any,
# # # #     depth: np.ndarray,
# # # #     position: np.ndarray,
# # # #     yaw: float,
# # # #     intrinsic: np.ndarray,
# # # #     uv_specifications: Sequence[str],
# # # #     output_directory: Path,
# # # #     *,
# # # #     mesh_half_pixels: int,
# # # #     mesh_lift: float,
# # # # ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
# # # #     mesh_directory = output_directory / "meshes"
# # # #     mesh_directory.mkdir(parents=True, exist_ok=True)
# # # #     height, width = depth.shape
# # # #     objects: list[Any] = []
# # # #     centroids: list[np.ndarray] = []
# # # #     geometries: list[np.ndarray] = []
# # # #     for index, specification in enumerate(uv_specifications):
# # # #         u, v = parse_uv_fraction(specification, width, height)
# # # #         vertices, faces = depth_patch_mesh(
# # # #             u,
# # # #             v,
# # # #             mesh_half_pixels,
# # # #             2,
# # # #             depth,
# # # #             position,
# # # #             yaw,
# # # #             intrinsic,
# # # #             lift=mesh_lift,
# # # #         )
# # # #         if len(vertices) == 0 or len(faces) == 0:
# # # #             raise RuntimeError(
# # # #                 f"obstacle mesh {index} at {specification!r} has no valid depth surface"
# # # #             )
# # # #         mesh_path = mesh_directory / f"obstacle_{index}.obj"
# # # #         save_obj(mesh_path, vertices, faces)
# # # #         semantic_id = MESH_OBSTACLE_ID + index
# # # #         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
# # # #         centroid = vertices.mean(axis=0).astype(np.float32)
# # # #         centroids.append(centroid)
# # # #         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
# # # #         print(
# # # #             f"[mesh] obstacle={index} semantic_id={semantic_id} "
# # # #             f"pixels={specification} vertices={len(vertices)} "
# # # #             f"world={centroid.tolist()}",
# # # #             flush=True,
# # # #         )
# # # #     return objects, centroids, geometries


# # # # def planar_mesh_clearance(
# # # #     point_xz: np.ndarray,
# # # #     geometries: Sequence[np.ndarray],
# # # # ) -> float:
# # # #     """Minimum 2-D distance from a robot center to projected mesh triangles."""
# # # #     point = np.asarray(point_xz, dtype=np.float64)
# # # #     best = float("inf")
# # # #     for triangles in geometries:
# # # #         triangles = np.asarray(triangles, dtype=np.float64)
# # # #         if triangles.size == 0:
# # # #             continue
# # # #         a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
# # # #         v0, v1, v2 = b - a, c - a, point[None, :] - a
# # # #         denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
# # # #         valid = np.abs(denominator) > 1.0e-12
# # # #         safe_denominator = np.where(valid, denominator, 1.0)
# # # #         u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_denominator
# # # #         v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_denominator
# # # #         if np.any(valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)):
# # # #             return 0.0

# # # #         starts = np.concatenate((a, b, c), axis=0)
# # # #         ends = np.concatenate((b, c, a), axis=0)
# # # #         segments = ends - starts
# # # #         squared_lengths = np.einsum("ij,ij->i", segments, segments)
# # # #         numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
# # # #         fractions = np.divide(
# # # #             numerators,
# # # #             squared_lengths,
# # # #             out=np.zeros_like(numerators),
# # # #             where=squared_lengths > 1.0e-12,
# # # #         )
# # # #         fractions = np.clip(fractions, 0.0, 1.0)
# # # #         closest = starts + fractions[:, None] * segments
# # # #         best = min(best, float(np.linalg.norm(point[None, :] - closest, axis=1).min()))
# # # #     return best


# # # # def parse_xz_velocity(specification: str) -> np.ndarray:
# # # #     values = [float(value) for value in str(specification).split(",")]
# # # #     if len(values) != 2 or not np.all(np.isfinite(values)):
# # # #         raise ValueError("obstacle velocity must be finite vx,vz")
# # # #     return np.asarray(values, dtype=np.float64)


# # # # def expand_obstacle_velocities(
# # # #     specifications: Sequence[str], obstacle_count: int
# # # # ) -> np.ndarray:
# # # #     if obstacle_count == 0:
# # # #         return np.zeros((0, 2), dtype=np.float64)
# # # #     if not specifications:
# # # #         return np.zeros((obstacle_count, 2), dtype=np.float64)
# # # #     velocities = np.stack([parse_xz_velocity(item) for item in specifications])
# # # #     if len(velocities) == 1 and obstacle_count > 1:
# # # #         velocities = np.repeat(velocities, obstacle_count, axis=0)
# # # #     if len(velocities) != obstacle_count:
# # # #         raise ValueError(
# # # #             "provide one obstacle velocity to broadcast or one velocity per mesh"
# # # #         )
# # # #     return velocities


# # # # def translated_mesh_geometry(
# # # #     base_geometries: Sequence[np.ndarray],
# # # #     base_centroids: Sequence[np.ndarray],
# # # #     velocities_xz: np.ndarray,
# # # #     elapsed_seconds: float,
# # # # ) -> tuple[list[np.ndarray], list[np.ndarray]]:
# # # #     geometries: list[np.ndarray] = []
# # # #     centroids: list[np.ndarray] = []
# # # #     for geometry, centroid, velocity in zip(
# # # #         base_geometries, base_centroids, velocities_xz
# # # #     ):
# # # #         offset_xz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
# # # #         geometries.append(np.asarray(geometry) + offset_xz[None, None, :])
# # # #         offset_xyz = np.asarray([offset_xz[0], 0.0, offset_xz[1]])
# # # #         centroids.append(np.asarray(centroid, dtype=np.float64) + offset_xyz)
# # # #     return geometries, centroids


# # # # def move_mesh_objects(
# # # #     objects: Sequence[Any], velocities_xz: np.ndarray, elapsed_seconds: float
# # # # ) -> None:
# # # #     for obstacle, velocity in zip(objects, velocities_xz):
# # # #         dx, dz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
# # # #         vector_type = type(obstacle.translation)
# # # #         obstacle.translation = vector_type(float(dx), 0.0, float(dz))


# # # # def camera_coordinates(
# # # #     point: np.ndarray, position: np.ndarray, yaw: float
# # # # ) -> tuple[float, float, float]:
# # # #     delta = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
# # # #     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
# # # #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# # # #     forward = forward_x * float(delta[0]) + forward_z * float(delta[2])
# # # #     left = left_x * float(delta[0]) + left_z * float(delta[2])
# # # #     return -left, float(delta[1]), forward


# # # # def camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
# # # #     hfov = math.radians(float(hfov_deg))
# # # #     focal = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
# # # #     return np.asarray(
# # # #         [
# # # #             [focal, 0.0, (width - 1) * 0.5],
# # # #             [0.0, focal, (height - 1) * 0.5],
# # # #             [0.0, 0.0, 1.0],
# # # #         ],
# # # #         dtype=np.float32,
# # # #     )


# # # # def world_goal_to_pixel(
# # # #     point: np.ndarray,
# # # #     position: np.ndarray,
# # # #     yaw: float,
# # # #     intrinsic: np.ndarray,
# # # #     height: int,
# # # #     width: int,
# # # # ) -> np.ndarray:
# # # #     """Project a world goal to a valid PixelGoal, clamping off-screen bearings."""

# # # #     right, up, forward = camera_coordinates(point, position, yaw)
# # # #     fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
# # # #     cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
# # # #     margin = 11
# # # #     bearing = math.atan2(right, forward)
# # # #     maximum_bearing = math.atan2(max(cx - margin, 1.0), fx)
# # # #     bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
# # # #     u = cx + fx * math.tan(bearing)
# # # #     v = cy - fy * up / forward if forward > 0.05 else 0.62 * height
# # # #     return np.asarray(
# # # #         [
# # # #             int(np.clip(round(u), margin, width - margin - 1)),
# # # #             int(np.clip(round(v), margin, height - margin - 1)),
# # # #         ],
# # # #         dtype=np.int32,
# # # #     )


# # # # def circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
# # # #     yy, xx = np.ogrid[:height, :width]
# # # #     return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


# # # # def project_world_mask(
# # # #     point: np.ndarray,
# # # #     position: np.ndarray,
# # # #     yaw: float,
# # # #     intrinsic: np.ndarray,
# # # #     height: int,
# # # #     width: int,
# # # #     radius: int,
# # # # ) -> tuple[np.ndarray, float]:
# # # #     right, up, forward = camera_coordinates(point, position, yaw)
# # # #     if forward <= 0.05:
# # # #         return np.zeros((height, width), dtype=np.uint8), forward
# # # #     u = float(intrinsic[0, 2] + intrinsic[0, 0] * right / forward)
# # # #     v = float(intrinsic[1, 2] - intrinsic[1, 1] * up / forward)
# # # #     if not (radius <= u < width - radius and radius <= v < height - radius):
# # # #         return np.zeros((height, width), dtype=np.uint8), forward
# # # #     return circle_mask(height, width, u, v, radius), forward


# # # # def depth_obstacle_mask(
# # # #     depth: np.ndarray, threshold: float, minimum_y_fraction: float
# # # # ) -> np.ndarray:
# # # #     mask = np.isfinite(depth) & (depth > 0.05) & (depth < float(threshold))
# # # #     mask[: int(depth.shape[0] * minimum_y_fraction)] = False
# # # #     return mask.astype(np.uint8)


# # # # def pixels_from_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
# # # #     v, u = np.nonzero(np.asarray(mask) > 0)
# # # #     if u.size == 0:
# # # #         return np.zeros((0, 2), dtype=np.int32)
# # # #     pixels = np.stack((u, v), axis=-1).astype(np.int32)
# # # #     if maximum > 0 and len(pixels) > maximum:
# # # #         indices = np.linspace(0, len(pixels) - 1, maximum).astype(np.int64)
# # # #         pixels = pixels[indices]
# # # #     return pixels


# # # # def waypoint_action(
# # # #     trajectory: np.ndarray,
# # # #     *,
# # # #     lookahead_index: int,
# # # #     maximum_forward_speed: float,
# # # #     maximum_yaw_rate: float,
# # # #     yaw_gain: float,
# # # # ) -> np.ndarray:
# # # #     trajectory = np.asarray(trajectory, dtype=np.float32)
# # # #     if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
# # # #         return np.zeros(3, dtype=np.float32)
# # # #     if np.max(np.linalg.norm(trajectory[:, :2], axis=-1)) < 1e-5:
# # # #         return np.zeros(3, dtype=np.float32)
# # # #     index = int(np.clip(lookahead_index, 0, trajectory.shape[0] - 1))
# # # #     forward, left = float(trajectory[index, 0]), float(trajectory[index, 1])
# # # #     bearing = math.atan2(left, max(forward, 1e-4))
# # # #     velocity = maximum_forward_speed * max(0.0, math.cos(bearing))
# # # #     yaw_rate = float(np.clip(yaw_gain * bearing, -maximum_yaw_rate, maximum_yaw_rate))
# # # #     return np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)


# # # # def integrate_mars(
# # # #     position: np.ndarray, yaw: float, action: np.ndarray, dt: float
# # # # ) -> tuple[np.ndarray, float]:
# # # #     forward_velocity, lateral_velocity, yaw_rate = [float(value) for value in action]
# # # #     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
# # # #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# # # #     output = np.asarray(position, dtype=np.float32).copy()
# # # #     output[0] += (forward_x * forward_velocity + left_x * lateral_velocity) * dt
# # # #     output[2] += (forward_z * forward_velocity + left_z * lateral_velocity) * dt
# # # #     return output, yaw + yaw_rate * dt


# # # # def wrap_angle(angle: float) -> float:
# # # #     return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# # # # def overlay_frame(
# # # #     rgb: np.ndarray,
# # # #     goal_mask: np.ndarray,
# # # #     obstacle_mask: np.ndarray,
# # # #     text: str,
# # # #     *,
# # # #     show_masks: bool,
# # # #     detection_box: Optional[np.ndarray] = None,
# # # #     detection_label: Optional[str] = None,
# # # # ) -> Image.Image:
# # # #     output = np.asarray(rgb, dtype=np.uint8).copy()
# # # #     if show_masks:
# # # #         output[goal_mask > 0] = (
# # # #             0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
# # # #         ).astype(np.uint8)
# # # #         output[obstacle_mask > 0] = (
# # # #             0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
# # # #         ).astype(np.uint8)
# # # #     image = Image.fromarray(output)
# # # #     draw = ImageDraw.Draw(image)
# # # #     if detection_box is not None:
# # # #         x1, y1, x2, y2 = [float(value) for value in detection_box]
# # # #         draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
# # # #         if detection_label:
# # # #             draw.text((x1 + 2, max(y1 - 14, 2)), detection_label, fill=(255, 255, 0))
# # # #     draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
# # # #     draw.text((10, 9), text, fill=(255, 255, 255))
# # # #     return image


# # # # def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
# # # #     import imageio.v2 as imageio

# # # #     with imageio.get_writer(path, fps=float(fps)) as writer:
# # # #         for frame in frames:
# # # #             writer.append_data(np.asarray(frame.convert("RGB")))


# # # # def parser() -> argparse.ArgumentParser:
# # # #     argument_parser = argparse.ArgumentParser(
# # # #         description="One-file released NavDP + in-denoising S2Diff Mars rollout"
# # # #     )
# # # #     argument_parser.add_argument("--navdp-root", required=True)
# # # #     argument_parser.add_argument("--navdp-checkpoint", required=True)
# # # #     argument_parser.add_argument("--navdp-python", default=sys.executable)
# # # #     argument_parser.add_argument("--navdp-device", default="cuda:0")
# # # #     argument_parser.add_argument(
# # # #         "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--goal-mode", choices=["point", "pixel"], default="point"
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct"
# # # #     )
# # # #     argument_parser.add_argument("--qwen-device", default="auto")
# # # #     argument_parser.add_argument(
# # # #         "--qwen-homotopy", action=argparse.BooleanOptionalAction, default=False,
# # # #         help=(
# # # #             "When a metric obstacle becomes relevant, Qwen chooses the single "
# # # #             "LEFT/RIGHT circulation sign used by every trajectory candidate."
# # # #         ),
# # # #     )
# # # #     argument_parser.add_argument("--homotopy-minimum-obstacle-pixels", type=int, default=30)
# # # #     argument_parser.add_argument("--homotopy-release-clear-frames", type=int, default=8)
# # # #     argument_parser.add_argument(
# # # #         "--homotopy-consistency-repeats", type=int, default=5,
# # # #         help="Repeat Qwen on the identical obstacle frame and use majority vote.",
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--remove-critic", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument("--seed", type=int, default=7)
# # # #     argument_parser.add_argument(
# # # #         "--start-server", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument("--server-host", default="127.0.0.1")
# # # #     argument_parser.add_argument("--server-port", type=int, default=8888)
# # # #     argument_parser.add_argument("--server-timeout", type=float, default=180.0)
# # # #     argument_parser.add_argument("--candidates", type=int, default=16)
# # # #     argument_parser.add_argument("--particles", type=int, default=8)
# # # #     argument_parser.add_argument("--particle-std", type=float, default=0.22)
# # # #     argument_parser.add_argument("--gradient-steps", type=int, default=3)
# # # #     argument_parser.add_argument("--gradient-step-size", type=float, default=0.04)
# # # #     argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
# # # #     argument_parser.add_argument("--temperature", type=float, default=0.35)
# # # #     argument_parser.add_argument(
# # # #         "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--particle-energy-reweighting",
# # # #         action=argparse.BooleanOptionalAction,
# # # #         default=True,
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument("--safe-distance", type=float, default=0.42)
# # # #     argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
# # # #     argument_parser.add_argument("--safety-weight", type=float, default=35.0)
# # # #     argument_parser.add_argument("--barrier-weight", type=float, default=25.0)
# # # #     argument_parser.add_argument("--barrier-rate", type=float, default=0.15)
# # # #     argument_parser.add_argument("--circulation-weight", type=float, default=18.0)
# # # #     argument_parser.add_argument(
# # # #         "--circulation-activation-distance", type=float, default=1.50
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--circulation-activation-sharpness", type=float, default=0.20
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--minimum-circulation-progress", type=float, default=0.025
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--blocking-alignment-threshold", type=float, default=0.25
# # # #     )
# # # #     argument_parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
# # # #     argument_parser.add_argument("--escape-lateral-target", type=float, default=0.35)
# # # #     argument_parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
# # # #     argument_parser.add_argument("--maximum-obstacle-depth", type=float, default=5.0)
# # # #     argument_parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)

# # # #     argument_parser.add_argument("--scene", required=True)
# # # #     argument_parser.add_argument("--terrain-obj", default=None)
# # # #     argument_parser.add_argument("--heightmap", default=None)
# # # #     argument_parser.add_argument(
# # # #         "--terrain-height-mode",
# # # #         choices=["auto", "heightmap", "obj", "flat"],
# # # #         default="auto",
# # # #     )
# # # #     argument_parser.add_argument("--flat-y", type=float, default=0.0)
# # # #     argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
# # # #     argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
# # # #     argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
# # # #     argument_parser.add_argument("--flip-heightmap-x", action="store_true")
# # # #     argument_parser.add_argument(
# # # #         "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument("--swap-heightmap-xz", action="store_true")
# # # #     argument_parser.add_argument("--clearance", type=float, default=1.4)
# # # #     argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)
# # # #     argument_parser.add_argument(
# # # #         "--robot-radius",
# # # #         type=float,
# # # #         default=0.24,
# # # #         help="Planar rover footprint radius used by both guidance and evaluation.",
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--evaluation-layout",
# # # #         default="default",
# # # #         help="Stable layout identifier stored in the rollout archive.",
# # # #     )

# # # #     argument_parser.add_argument("--height", type=int, default=720)
# # # #     argument_parser.add_argument("--width", type=int, default=720)
# # # #     argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
# # # #     argument_parser.add_argument("--hz", type=float, default=10.0)
# # # #     argument_parser.add_argument("--max-steps", type=int, default=300)
# # # #     argument_parser.add_argument("--stop-distance", type=float, default=1.0)
# # # #     argument_parser.add_argument("--start-x", type=float, default=0.0)
# # # #     argument_parser.add_argument("--start-z", type=float, default=8.0)
# # # #     argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
# # # #     argument_parser.add_argument("--goal-x", type=float, default=None)
# # # #     argument_parser.add_argument("--goal-z", type=float, default=None)
# # # #     argument_parser.add_argument("--goal-y", type=float, default=None)
# # # #     argument_parser.add_argument("--goal-height", type=float, default=1.2)
# # # #     argument_parser.add_argument("--goal-radius", type=int, default=18)
# # # #     argument_parser.add_argument(
# # # #         "--goal-mesh", action=argparse.BooleanOptionalAction, default=False
# # # #     )
# # # #     argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
# # # #     argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)

# # # #     argument_parser.add_argument(
# # # #         "--obstacle-mode", choices=["none", "depth", "mesh", "ghost"], default="none"
# # # #     )
# # # #     argument_parser.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
# # # #     argument_parser.add_argument("--obstacle-min-y-fraction", type=float, default=0.45)
# # # #     argument_parser.add_argument("--ghost-obstacle-x", type=float, default=None)
# # # #     argument_parser.add_argument("--ghost-obstacle-z", type=float, default=None)
# # # #     argument_parser.add_argument("--ghost-obstacle-y", type=float, default=None)
# # # #     argument_parser.add_argument("--ghost-obstacle-height", type=float, default=0.45)
# # # #     argument_parser.add_argument("--ghost-obstacle-radius", type=int, default=24)
# # # #     argument_parser.add_argument(
# # # #         "--obstacle-mesh-uv",
# # # #         nargs="+",
# # # #         default=[],
# # # #         help=(
# # # #             "Actual rendered obstacle mesh locations as image fractions u,v. "
# # # #             "Example: --obstacle-mesh-uv 0.50,0.72 0.30,0.68"
# # # #         ),
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--obstacle-world-xz",
# # # #         nargs="*",
# # # #         default=[],
# # # #         metavar="X,Z",
# # # #         help=(
# # # #             "Static rendered obstacle-box centers in world X,Z coordinates. "
# # # #             "Example: --obstacle-world-xz 0,0. Do not combine with "
# # # #             "--obstacle-mesh-uv."
# # # #         ),
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--obstacle-world-xz-item",
# # # #         action="append",
# # # #         default=[],
# # # #         metavar="X,Z",
# # # #         help=(
# # # #             "Repeatable form that safely accepts negative coordinates, e.g. "
# # # #             "--obstacle-world-xz-item=-3,0."
# # # #         ),
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--world-obstacle-half-extent", type=float, default=0.75
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--world-obstacle-height", type=float, default=1.40
# # # #     )
# # # #     argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
# # # #     argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)
# # # #     argument_parser.add_argument(
# # # #         "--obstacle-velocity-xz",
# # # #         nargs="*",
# # # #         default=[],
# # # #         metavar="VX,VZ",
# # # #         help=(
# # # #             "World-frame mesh velocities in m/s. Supply one value to broadcast "
# # # #             "or one value per obstacle. Example: --obstacle-velocity-xz 0.30,0.0"
# # # #         ),
# # # #     )

# # # #     argument_parser.add_argument("--lookahead-index", type=int, default=4)
# # # #     argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
# # # #     argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
# # # #     argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
# # # #     argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
# # # #     argument_parser.add_argument("--save-every", type=int, default=1)
# # # #     argument_parser.add_argument(
# # # #         "--save-frames", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--save-video", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--archive-observations",
# # # #         action=argparse.BooleanOptionalAction,
# # # #         default=True,
# # # #         help="Store RGB/depth/masks in rollout.npz; disable for large evaluations.",
# # # #     )
# # # #     argument_parser.add_argument(
# # # #         "--overlay-masks", action=argparse.BooleanOptionalAction, default=True
# # # #     )
# # # #     return argument_parser


# # # # def main() -> None:
# # # #     args = parser().parse_args()
# # # #     if args.obstacle_world_xz_item:
# # # #         args.obstacle_world_xz.extend(args.obstacle_world_xz_item)
# # # #     np.random.seed(args.seed)
# # # #     if args.goal_x is None or args.goal_z is None:
# # # #         raise ValueError("fixed PointGoal requires --goal-x and --goal-z")
# # # #     if args.qwen_homotopy and args.goal_mode != "point":
# # # #         raise ValueError("Qwen homotopy selection currently requires --goal-mode point")
# # # #     if args.robot_radius < 0.0:
# # # #         raise ValueError("robot-radius must be non-negative")
# # # #     if args.obstacle_velocity_xz and args.obstacle_mode != "mesh":
# # # #         raise ValueError("moving obstacle velocities require --obstacle-mode mesh")
# # # #     if args.obstacle_mode == "ghost" and (
# # # #         args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
# # # #     ):
# # # #         raise ValueError("ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z")
# # # #     if args.obstacle_mesh_uv and args.obstacle_world_xz:
# # # #         raise ValueError(
# # # #             "choose either --obstacle-mesh-uv or --obstacle-world-xz, not both"
# # # #         )
# # # #     if args.obstacle_mode == "mesh" and not (
# # # #         args.obstacle_mesh_uv or args.obstacle_world_xz
# # # #     ):
# # # #         raise ValueError(
# # # #             "mesh mode requires --obstacle-world-xz X,Z [X,Z ...] or "
# # # #             "--obstacle-mesh-uv u,v [u,v ...]"
# # # #         )
# # # #     if args.world_obstacle_half_extent <= 0.0 or args.world_obstacle_height <= 0.0:
# # # #         raise ValueError("world obstacle dimensions must be positive")
# # # #     if args.goal_mesh_half_extent <= 0.0 or args.goal_mesh_height <= 0.0:
# # # #         raise ValueError("goal mesh dimensions must be positive")


# # # #     homotopy_selector = None
# # # #     if args.qwen_homotopy:
# # # #         if args.planner_mode == "pure-navdp":
# # # #             raise ValueError("Qwen homotopy conditioning requires s2diff or gradient mode")
# # # #         from qwen_navdp_homotopy import VisualQwenHomotopySelector

# # # #         homotopy_selector = VisualQwenHomotopySelector(
# # # #             model_id=args.qwen_model_id,
# # # #             device=args.qwen_device,
# # # #             minimum_obstacle_pixels=args.homotopy_minimum_obstacle_pixels,
# # # #             release_clear_frames=args.homotopy_release_clear_frames,
# # # #             consistency_repeats=args.homotopy_consistency_repeats,
# # # #         )

# # # #     server_process: Optional[subprocess.Popen[Any]] = None
# # # #     simulator = None
# # # #     try:
# # # #         server_process = start_server(args)
# # # #         server_url = f"http://{args.server_host}:{args.server_port}"
# # # #         client = NavDPS2DiffClient(server_url)
# # # #         algorithm = client.reset(
# # # #             camera_intrinsic(args.height, args.width, args.hfov_deg),
# # # #             batch_size=1,
# # # #             stop_threshold=-3.0,
# # # #         )
# # # #         supported_algorithms = {
# # # #             "navdp-s2diff-pixels",
# # # #             "navdp-hlc-s2diff",
# # # #             "navdp-hlc-s2diff-no-critic",
# # # #             "navdp-hlc-gradient",
# # # #             "navdp-hlc-gradient-no-critic",
# # # #             "navdp-pure-critic",
# # # #         }
# # # #         if algorithm not in supported_algorithms:
# # # #             raise RuntimeError(f"unexpected planner response: {algorithm!r}")

# # # #         terrain = TerrainHeight(
# # # #             mode=args.terrain_height_mode,
# # # #             heightmap=Path(args.heightmap).expanduser().resolve() if args.heightmap else None,
# # # #             obj=Path(args.terrain_obj).expanduser().resolve() if args.terrain_obj else None,
# # # #             flat_y=args.flat_y,
# # # #             size_x=args.size_x,
# # # #             size_z=args.size_z,
# # # #             size_y=args.size_y,
# # # #             flip_x=args.flip_heightmap_x,
# # # #             flip_z=args.flip_heightmap_z,
# # # #             swap_xz=args.swap_heightmap_xz,
# # # #         )
# # # #         output_directory = Path(args.output).expanduser().resolve()
# # # #         frame_directory = output_directory / "frames"
# # # #         frame_directory.mkdir(parents=True, exist_ok=True)

# # # #         simulator = make_simulator(
# # # #             Path(args.scene),
# # # #             args.height,
# # # #             args.width,
# # # #             args.hfov_deg,
# # # #             with_semantic=args.obstacle_mode == "mesh" or args.goal_mesh,
# # # #         )
# # # #         agent = simulator.initialize_agent(0)
# # # #         intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
# # # #         x, z = float(args.start_x), float(args.start_z)
# # # #         yaw = math.radians(float(args.start_yaw_deg))
# # # #         dt = 1.0 / float(args.hz)

# # # #         goal_y = args.goal_y
# # # #         if goal_y is None:
# # # #             goal_y = terrain.local_height_max(args.goal_x, args.goal_z, 0.8) + args.goal_height
# # # #         goal = np.asarray([args.goal_x, goal_y, args.goal_z], dtype=np.float32)
# # # #         start_position_xz = np.asarray([x, z], dtype=np.float64)
# # # #         initial_goal_distance = float(
# # # #             np.linalg.norm(goal[[0, 2]].astype(np.float64) - start_position_xz)
# # # #         )
# # # #         goal_mesh_object = None
# # # #         if args.goal_mesh:
# # # #             goal_mesh_object = place_world_goal_mesh(
# # # #                 simulator,
# # # #                 terrain,
# # # #                 args.goal_x,
# # # #                 args.goal_z,
# # # #                 output_directory,
# # # #                 half_extent=args.goal_mesh_half_extent,
# # # #                 height=args.goal_mesh_height,
# # # #             )

# # # #         ghost = None
# # # #         if args.obstacle_mode == "ghost":
# # # #             ghost_y = args.ghost_obstacle_y
# # # #             if ghost_y is None:
# # # #                 ghost_y = terrain.local_height_max(
# # # #                     args.ghost_obstacle_x, args.ghost_obstacle_z, args.pose_terrain_radius
# # # #                 ) + args.ghost_obstacle_height
# # # #             ghost = np.asarray(
# # # #                 [args.ghost_obstacle_x, ghost_y, args.ghost_obstacle_z], dtype=np.float32
# # # #             )

# # # #         mesh_objects: list[Any] = []
# # # #         mesh_centroids: list[np.ndarray] = []
# # # #         mesh_current_centroids: list[np.ndarray] = []
# # # #         mesh_base_geometries: list[np.ndarray] = []
# # # #         mesh_geometries: list[np.ndarray] = []
# # # #         mesh_velocities = np.zeros((0, 2), dtype=np.float64)
# # # #         mesh_placed = False
# # # #         if args.obstacle_mode == "mesh" and args.obstacle_world_xz:
# # # #             mesh_objects, mesh_centroids, mesh_base_geometries = (
# # # #                 place_world_obstacle_meshes(
# # # #                     simulator,
# # # #                     terrain,
# # # #                     args.obstacle_world_xz,
# # # #                     output_directory,
# # # #                     half_extent=args.world_obstacle_half_extent,
# # # #                     height=args.world_obstacle_height,
# # # #                 )
# # # #             )
# # # #             mesh_velocities = expand_obstacle_velocities(
# # # #                 args.obstacle_velocity_xz, len(mesh_objects)
# # # #             )
# # # #             mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# # # #                 mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
# # # #             )
# # # #             mesh_placed = True

# # # #         row_keys = [
# # # #                 "pose",
# # # #                 "action_3d",
# # # #                 "point_goal",
# # # #                 "selected_trajectory",
# # # #                 "all_trajectories",
# # # #                 "all_values",
# # # #                 "selected_index",
# # # #                 "fallback_stop",
# # # #                 "escape_turn",
# # # #                 "valid_obstacle_points",
# # # #                 "selected_circulation_sign",
# # # #                 "candidate_circulation_signs",
# # # #                 "selected_barrier_energy",
# # # #                 "selected_circulation_energy",
# # # #                 "planning_time_seconds",
# # # #                 "selected_minimum_clearance",
# # # #                 "mean_guidance_noise_correction",
# # # #                 "final_guidance_noise_correction",
# # # #                 "maximum_guidance_noise_correction",
# # # #                 "mean_final_effective_sample_size",
# # # #                 "goal_distance",
# # # #                 "executed_center_clearance",
# # # #                 "executed_surface_clearance",
# # # #                 "geometric_collision",
# # # #                 "obstacle_positions_world",

# # # #                 "qwen_homotopy_sign",
# # # #                 "qwen_homotopy_side",
# # # #                 "qwen_homotopy_confidence",
# # # #                 "qwen_homotopy_queried",
# # # #         ]
# # # #         if args.archive_observations:
# # # #             row_keys.extend(("rgb", "depth", "goal_mask", "obstacle_mask"))
# # # #         rows: dict[str, list[Any]] = {key: [] for key in row_keys}
# # # #         video_frames: list[Image.Image] = []
# # # #         success = False
# # # #         homotopy_events: list[dict[str, Any]] = []

# # # #         for step in range(int(args.max_steps)):
# # # #             y = terrain.local_height_max(x, z, args.pose_terrain_radius) + args.clearance
# # # #             position = np.asarray([x, y, z], dtype=np.float32)
# # # #             set_agent_pose(agent, position, yaw)
# # # #             if mesh_placed:
# # # #                 elapsed_seconds = step * dt
# # # #                 move_mesh_objects(mesh_objects, mesh_velocities, elapsed_seconds)
# # # #                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# # # #                     mesh_base_geometries,
# # # #                     mesh_centroids,
# # # #                     mesh_velocities,
# # # #                     elapsed_seconds,
# # # #                 )
# # # #             observation = simulator.get_sensor_observations()
# # # #             rgb, depth = rgb_depth(observation)

# # # #             if args.obstacle_mode == "mesh" and not mesh_placed:
# # # #                 mesh_objects, mesh_centroids, mesh_base_geometries = place_obstacle_meshes(
# # # #                     simulator,
# # # #                     depth,
# # # #                     position,
# # # #                     yaw,
# # # #                     intrinsic,
# # # #                     args.obstacle_mesh_uv,
# # # #                     output_directory,
# # # #                     mesh_half_pixels=args.mesh_half_pixels,
# # # #                     mesh_lift=args.mesh_obstacle_lift,
# # # #                 )
# # # #                 mesh_velocities = expand_obstacle_velocities(
# # # #                     args.obstacle_velocity_xz, len(mesh_objects)
# # # #                 )
# # # #                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# # # #                     mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
# # # #                 )
# # # #                 mesh_placed = True
# # # #                 observation = simulator.get_sensor_observations()
# # # #                 rgb, depth = rgb_depth(observation)

# # # #             semantic = (
# # # #                 semantic_from_observation(observation)
# # # #                 if args.obstacle_mode == "mesh" or args.goal_mesh
# # # #                 else None
# # # #             )
# # # #             goal_right, _goal_up, goal_forward = camera_coordinates(
# # # #                 goal, position, yaw
# # # #             )
# # # #             point_goal = np.asarray(
# # # #                 [max(goal_forward, 0.0), -goal_right], dtype=np.float32
# # # #             )
# # # #             if args.goal_mesh:
# # # #                 assert semantic is not None
# # # #                 goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
# # # #                 # The obstacle may visually occlude the real goal mesh in the
# # # #                 # intentionally collinear test. Keep its projected fixed-goal
# # # #                 # overlay visible for Qwen without changing the numeric PointGoal.
# # # #                 if not np.any(goal_mask):
# # # #                     goal_mask, _ = project_world_mask(
# # # #                         goal,
# # # #                         position,
# # # #                         yaw,
# # # #                         intrinsic,
# # # #                         args.height,
# # # #                         args.width,
# # # #                         args.goal_radius,
# # # #                     )
# # # #             else:
# # # #                 goal_mask, _ = project_world_mask(
# # # #                     goal,
# # # #                     position,
# # # #                     yaw,
# # # #                     intrinsic,
# # # #                     args.height,
# # # #                     args.width,
# # # #                     args.goal_radius,
# # # #                 )
# # # #             planner_goal = point_goal
# # # #             if args.goal_mode == "pixel":
# # # #                 planner_goal = world_goal_to_pixel(
# # # #                     goal, position, yaw, intrinsic, args.height, args.width
# # # #                 )
# # # #                 goal_mask = circle_mask(
# # # #                     args.height,
# # # #                     args.width,
# # # #                     planner_goal[0],
# # # #                     planner_goal[1],
# # # #                     args.goal_radius,
# # # #                 )
# # # #             guidance_depth = depth.copy()
# # # #             if args.obstacle_mode == "depth":
# # # #                 obstacle_mask = depth_obstacle_mask(
# # # #                     depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
# # # #                 )
# # # #             elif args.obstacle_mode == "mesh":
# # # #                 assert semantic is not None
# # # #                 semantic_ids = list(
# # # #                     range(
# # # #                         MESH_OBSTACLE_ID,
# # # #                         MESH_OBSTACLE_ID + len(mesh_objects),
# # # #                     )
# # # #                 )
# # # #                 obstacle_mask = np.isin(semantic, semantic_ids).astype(np.uint8)
# # # #                 # The depth image was re-rendered after mesh placement, so
# # # #                 # guidance_depth already contains the real obstacle depth.
# # # #             elif args.obstacle_mode == "ghost":
# # # #                 assert ghost is not None
# # # #                 obstacle_mask, obstacle_forward = project_world_mask(
# # # #                     ghost,
# # # #                     position,
# # # #                     yaw,
# # # #                     intrinsic,
# # # #                     args.height,
# # # #                     args.width,
# # # #                     args.ghost_obstacle_radius,
# # # #                 )
# # # #                 if obstacle_forward > 0.05:
# # # #                     guidance_depth[obstacle_mask > 0] = obstacle_forward
# # # #             else:
# # # #                 obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)

# # # #             # Replace this mask-to-pixels line with your own detector's [u,v]
# # # #             # array if obstacle pixels already come directly from your system.
# # # #             obstacle_pixels = pixels_from_mask(
# # # #                 obstacle_mask, args.maximum_obstacle_pixels
# # # #             )
# # # #             homotopy_decision = None
# # # #             forced_circulation_sign = 0.0
# # # #             if homotopy_selector is not None:
# # # #                 homotopy_obstacle_mask = (
# # # #                     (obstacle_mask > 0)
# # # #                     & np.isfinite(guidance_depth)
# # # #                     & (guidance_depth >= args.minimum_obstacle_depth)
# # # #                     & (guidance_depth <= args.maximum_obstacle_depth)
# # # #                 ).astype(np.uint8)
# # # #                 qwen_overlay = overlay_frame(
# # # #                     rgb,
# # # #                     goal_mask,
# # # #                     homotopy_obstacle_mask,
# # # #                     "Qwen homotopy: choose LEFT or RIGHT",
# # # #                     show_masks=True,
# # # #                 )
# # # #                 homotopy_decision = homotopy_selector.step(
# # # #                     np.asarray(qwen_overlay.convert("RGB")), homotopy_obstacle_mask
# # # #                 )
# # # #                 forced_circulation_sign = homotopy_decision.circulation_sign
# # # #                 if homotopy_decision.queried_qwen:
# # # #                     event = {
# # # #                         "step": step,
# # # #                         "side": homotopy_decision.side,
# # # #                         "circulation_sign": forced_circulation_sign,
# # # #                         "confidence": homotopy_decision.confidence,
# # # #                         "repeat_sides": list(homotopy_decision.repeated_sides),
# # # #                         "repeat_confidences": list(
# # # #                             homotopy_decision.repeated_confidences
# # # #                         ),
# # # #                         "consistency_rate": homotopy_decision.consistency_rate,
# # # #                         "used_fallback": homotopy_decision.used_fallback,
# # # #                         "raw_response": homotopy_decision.raw_response,
# # # #                     }
# # # #                     homotopy_events.append(event)
# # # #                     query_directory = output_directory / "qwen_homotopy_queries"
# # # #                     query_directory.mkdir(parents=True, exist_ok=True)
# # # #                     qwen_overlay.save(query_directory / f"query_step_{step:04d}.png")
# # # #                     print(
# # # #                         f"[qwen-homotopy] side={homotopy_decision.side} "
# # # #                         f"sign={forced_circulation_sign:+.0f} "
# # # #                         f"confidence={homotopy_decision.confidence:.2f} "
# # # #                         f"consistency={homotopy_decision.consistency_rate:.2%} "
# # # #                         f"repeats={list(homotopy_decision.repeated_sides)} "
# # # #                         f"fallback={homotopy_decision.used_fallback}",
# # # #                         flush=True,
# # # #                     )
# # # #             planning_start = time.perf_counter()
# # # #             result = client.plan(
# # # #                 goal_xy=planner_goal,
# # # #                 rgb=rgb,
# # # #                 depth=guidance_depth,
# # # #                 obstacle_pixels=obstacle_pixels,
# # # #                 goal_mode=args.goal_mode,
# # # #                 forced_circulation_sign=forced_circulation_sign,
# # # #             )
# # # #             planning_time = time.perf_counter() - planning_start
# # # #             action = (
# # # #                 np.zeros(3, dtype=np.float32)
# # # #                 if result.fallback_stop
# # # #                 else waypoint_action(
# # # #                     result.trajectory,
# # # #                     lookahead_index=args.lookahead_index,
# # # #                     maximum_forward_speed=args.maximum_forward_speed,
# # # #                     maximum_yaw_rate=args.maximum_yaw_rate,
# # # #                     yaw_gain=args.yaw_gain,
# # # #                 )
# # # #             )

# # # #             next_position, next_yaw = integrate_mars(position, yaw, action, dt)
# # # #             x = float(np.clip(next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5))
# # # #             z = float(np.clip(next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5))
# # # #             yaw = wrap_angle(next_yaw)
# # # #             goal_distance = float(np.linalg.norm(goal[[0, 2]] - np.asarray([x, z])))
# # # #             center_clearance = planar_mesh_clearance(
# # # #                 np.asarray([x, z], dtype=np.float64), mesh_geometries
# # # #             )
# # # #             if np.isfinite(center_clearance):
# # # #                 surface_clearance = max(center_clearance - float(args.robot_radius), 0.0)
# # # #                 geometric_collision = center_clearance <= float(args.robot_radius)
# # # #             else:
# # # #                 surface_clearance = float("nan")
# # # #                 geometric_collision = False
# # # #             rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
# # # #             pose = np.asarray(
# # # #                 [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32
# # # #             )

# # # #             if args.archive_observations:
# # # #                 rows["rgb"].append(rgb)
# # # #                 rows["depth"].append(depth)
# # # #                 rows["goal_mask"].append(goal_mask)
# # # #                 rows["obstacle_mask"].append(obstacle_mask)
# # # #             rows["pose"].append(pose)
# # # #             rows["action_3d"].append(action)
# # # #             rows["point_goal"].append(planner_goal)
# # # #             rows["selected_trajectory"].append(result.trajectory)
# # # #             rows["all_trajectories"].append(result.all_trajectories)
# # # #             rows["all_values"].append(result.all_values)
# # # #             rows["selected_index"].append(result.selected_index)
# # # #             rows["fallback_stop"].append(result.fallback_stop)
# # # #             rows["escape_turn"].append(result.escape_turn)
# # # #             rows["valid_obstacle_points"].append(result.valid_obstacle_points)
# # # #             rows["selected_circulation_sign"].append(result.selected_circulation_sign)
# # # #             rows["candidate_circulation_signs"].append(
# # # #                 result.candidate_circulation_signs
# # # #             )
# # # #             rows["selected_barrier_energy"].append(result.selected_barrier_energy)
# # # #             rows["selected_circulation_energy"].append(
# # # #                 result.selected_circulation_energy
# # # #             )
# # # #             rows["planning_time_seconds"].append(planning_time)
# # # #             rows["selected_minimum_clearance"].append(result.selected_minimum_clearance)
# # # #             rows["mean_guidance_noise_correction"].append(
# # # #                 result.mean_guidance_noise_correction
# # # #             )
# # # #             rows["final_guidance_noise_correction"].append(
# # # #                 result.final_guidance_noise_correction
# # # #             )
# # # #             rows["maximum_guidance_noise_correction"].append(
# # # #                 result.maximum_guidance_noise_correction
# # # #             )
# # # #             rows["mean_final_effective_sample_size"].append(
# # # #                 result.mean_final_effective_sample_size
# # # #             )
# # # #             rows["goal_distance"].append(goal_distance)
# # # #             rows["executed_center_clearance"].append(center_clearance)
# # # #             rows["executed_surface_clearance"].append(surface_clearance)
# # # #             rows["geometric_collision"].append(geometric_collision)
# # # #             rows["obstacle_positions_world"].append(
# # # #                 np.stack(mesh_current_centroids)
# # # #                 if mesh_current_centroids
# # # #                 else np.zeros((0, 3), dtype=np.float64)
# # # #             )

# # # #             rows["qwen_homotopy_sign"].append(forced_circulation_sign)
# # # #             rows["qwen_homotopy_side"].append(
# # # #                 homotopy_decision.side if homotopy_decision is not None else "AUTO"
# # # #             )
# # # #             rows["qwen_homotopy_confidence"].append(
# # # #                 homotopy_decision.confidence if homotopy_decision is not None else 0.0
# # # #             )
# # # #             rows["qwen_homotopy_queried"].append(
# # # #                 homotopy_decision.queried_qwen if homotopy_decision is not None else False
# # # #             )

# # # #             if args.save_frames and step % max(int(args.save_every), 1) == 0:

# # # #                 side_label = (
# # # #                     homotopy_decision.side
# # # #                     if homotopy_decision is not None
# # # #                     else "AUTO"
# # # #                 )
# # # #                 label = (
# # # #                     f"t={step} goal={goal_distance:.2f}m qwen_side={side_label} pixels={len(obstacle_pixels)} "
# # # #                     f"pred={result.selected_minimum_clearance:.2f}m "
# # # #                     f"actual={surface_clearance:.2f}m "
# # # #                     f"mode={result.selected_circulation_sign:+.0f} "
# # # #                     f"escape={int(result.escape_turn)} "
# # # #                     f"guide_rms={result.mean_guidance_noise_correction:.4f} "
# # # #                     f"v={action[0]:.2f} w={action[2]:.2f}"
# # # #                 )
# # # #                 frame = overlay_frame(
# # # #                     rgb,
# # # #                     goal_mask,
# # # #                     obstacle_mask,
# # # #                     label,
# # # #                     show_masks=args.overlay_masks,
# # # #                 )
# # # #                 frame.save(frame_directory / f"frame_{step:04d}.png")
# # # #                 video_frames.append(frame)

# # # #             print(
# # # #                 f"step={step:04d} goal={goal_distance:.2f}m "
# # # #                 f"qwen_side={homotopy_decision.side if homotopy_decision else 'AUTO'} "
# # # #                 f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
# # # #                 f"selected={result.selected_index} fallback={result.fallback_stop} "
# # # #                 f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
# # # #                 f"pred_clear={result.selected_minimum_clearance:.3f}m "
# # # #                 f"actual_clear={surface_clearance:.3f}m "
# # # #                 f"collision={geometric_collision} "
# # # #                 f"barrier={result.selected_barrier_energy:.5f} "
# # # #                 f"circ={result.selected_circulation_energy:.5f} "
# # # #                 f"latency={planning_time * 1000.0:.1f}ms "
# # # #                 f"guide_rms={result.mean_guidance_noise_correction:.6f} "
# # # #                 f"ess={result.mean_final_effective_sample_size:.2f} "
# # # #                 f"action={action.tolist()}",
# # # #                 flush=True,
# # # #             )
# # # #             if goal_distance <= args.stop_distance:
# # # #                 success = True
# # # #                 break

# # # #         if not rows["goal_distance"]:
# # # #             raise RuntimeError("rollout produced no steps")
# # # #         rollout_path = output_directory / "rollout.npz"
# # # #         np.savez_compressed(
# # # #             rollout_path,
# # # #             **{
# # # #                 key: np.stack(values)
# # # #                 if isinstance(values[0], np.ndarray)
# # # #                 else np.asarray(values)
# # # #                 for key, values in rows.items()
# # # #             },
# # # #             goal_position=goal,
# # # #             obstacle_position=(
# # # #                 mesh_centroids[0]
# # # #                 if mesh_centroids
# # # #                 else (
# # # #                     ghost
# # # #                     if ghost is not None
# # # #                     else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
# # # #                 )
# # # #             ),
# # # #             obstacle_positions=(
# # # #                 np.stack(mesh_centroids)
# # # #                 if mesh_centroids
# # # #                 else np.zeros((0, 3), dtype=np.float32)
# # # #             ),
# # # #             obstacle_velocity_xz=mesh_velocities,
# # # #             success=np.asarray(success),
# # # #             hz=np.asarray(args.hz, dtype=np.float32),
# # # #             start_position_xz=start_position_xz,
# # # #             initial_goal_distance=np.asarray(initial_goal_distance, dtype=np.float64),
# # # #             stop_distance=np.asarray(args.stop_distance, dtype=np.float64),
# # # #             robot_radius=np.asarray(args.robot_radius, dtype=np.float64),
# # # #             evaluation_layout=np.asarray(args.evaluation_layout),
# # # #             seed=np.asarray(args.seed, dtype=np.int64),
# # # #             goal_mode=np.asarray(args.goal_mode),
# # # #         )
# # # #         with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
# # # #             json.dump(
# # # #                 {
# # # #                     "success": success,
# # # #                     "steps": len(rows["goal_distance"]),
# # # #                     "archived_observations": args.archive_observations,
# # # #                     "final_goal_distance": float(rows["goal_distance"][-1]),
# # # #                     "planner": "released_navdp_s2diff_pixels",
# # # #                     "controller": "direct_waypoint_no_optimizer",
# # # #                     "qwen_role": "obstacle_homotopy_only",
# # # #                     "qwen_creates_goal_or_action": False,
# # # #                     "qwen_homotopy": args.qwen_homotopy,
# # # #                     "qwen_homotopy_events": homotopy_events,
# # # #                     "qwen_homotopy_forces_all_candidates": args.qwen_homotopy,
# # # #                     "homotopy_sign_convention": {"LEFT": -1.0, "RIGHT": 1.0},
# # # #                     "homotopy_minimum_obstacle_pixels": args.homotopy_minimum_obstacle_pixels,
# # # #                     "homotopy_release_clear_frames": args.homotopy_release_clear_frames,
# # # #                     "homotopy_consistency_repeats": args.homotopy_consistency_repeats,
# # # #                     "uses_velocity_chunk": False,
# # # #                     "obstacle_mode": args.obstacle_mode,
# # # #                     "obstacle_world_xz": args.obstacle_world_xz,
# # # #                     "goal_mesh": args.goal_mesh,
# # # #                     "particle_anchor": args.particle_anchor,
# # # #                     "particle_energy_reweighting": args.particle_energy_reweighting,
# # # #                     "particle_collision_mask": args.particle_collision_mask,
# # # #                     "goal_mode": args.goal_mode,
# # # #                     "particle_noise_schedule": args.particle_noise_schedule,
# # # #                     "progressive_guidance": args.progressive_guidance,
# # # #                     "mesh_obstacle_count": len(mesh_centroids),
# # # #                     "moving_obstacles": bool(np.any(np.abs(mesh_velocities) > 0.0)),
# # # #                     "obstacle_velocity_xz": mesh_velocities.tolist(),
# # # #                     "evaluation_layout": args.evaluation_layout,
# # # #                     "seed": args.seed,
# # # #                     "robot_radius": args.robot_radius,
# # # #                     "minimum_executed_surface_clearance": (
# # # #                         float(np.nanmin(rows["executed_surface_clearance"]))
# # # #                         if np.any(np.isfinite(rows["executed_surface_clearance"]))
# # # #                         else None
# # # #                     ),
# # # #                     "geometric_collision": bool(
# # # #                         np.any(rows["geometric_collision"])
# # # #                     ),
# # # #                     "rollout": str(rollout_path),
# # # #                 },
# # # #                 file,
# # # #                 indent=2,
# # # #             )
# # # #         if args.save_video and video_frames:
# # # #             save_video(
# # # #                 video_frames,
# # # #                 output_directory / "rollout.mp4",
# # # #                 fps=max(args.hz / max(args.save_every, 1), 1.0),
# # # #             )
# # # #         print(f"Saved rollout: {rollout_path}", flush=True)
# # # #         print(f"Success: {success}", flush=True)
# # # #     finally:
# # # #         if simulator is not None:
# # # #             simulator.close()
# # # #         stop_server(server_process)


# # # # if __name__ == "__main__":
# # # #     main()

# # # from __future__ import annotations

# # # import argparse
# # # import io
# # # import json
# # # import math
# # # import os
# # # import socket
# # # import subprocess
# # # import sys
# # # import time
# # # from dataclasses import dataclass
# # # from pathlib import Path
# # # from typing import Any, Optional, Sequence

# # # import habitat_sim
# # # import numpy as np
# # # import quaternion
# # # import requests
# # # from habitat_sim.agent import AgentConfiguration
# # # from PIL import Image, ImageDraw, ImageFilter

# # # from belief_heading_recovery import belief_heading_recovery_action
# # # from belief_pixel_goal import GaussianGoalBelief

# # # HERE = Path(__file__).resolve().parent
# # # SIZE_X = 50.0
# # # SIZE_Z = 50.0
# # # SIZE_Y = 4.820803273566
# # # MESH_GOAL_ID = 10000
# # # MESH_OBSTACLE_ID = 2


# # # @dataclass(frozen=True)
# # # class NavDPS2DiffOutput:
# # #     trajectory: np.ndarray
# # #     all_trajectories: np.ndarray
# # #     all_values: np.ndarray
# # #     selected_index: int
# # #     fallback_stop: bool
# # #     escape_turn: bool
# # #     valid_obstacle_points: int
# # #     selected_circulation_sign: float
# # #     candidate_circulation_signs: np.ndarray
# # #     selected_barrier_energy: float
# # #     selected_circulation_energy: float
# # #     minimum_clearance: np.ndarray
# # #     selected_minimum_clearance: float
# # #     mean_guidance_noise_correction: float
# # #     final_guidance_noise_correction: float
# # #     maximum_guidance_noise_correction: float
# # #     mean_final_effective_sample_size: float


# # # class NavDPS2DiffClient:
# # #     def __init__(self, server_url: str, timeout: float = 180.0):
# # #         self.server_url = server_url.rstrip("/")
# # #         self.timeout = float(timeout)

# # #     def reset(
# # #         self,
# # #         intrinsic: np.ndarray,
# # #         *,
# # #         stop_threshold: float = -3.0,
# # #         batch_size: int = 1,
# # #     ) -> str:
# # #         intrinsic = np.asarray(intrinsic, dtype=np.float32)
# # #         if intrinsic.shape != (3, 3):
# # #             raise ValueError(f"intrinsic must have shape [3,3], got {intrinsic.shape}")
# # #         response = requests.post(
# # #             f"{self.server_url}/navigator_reset",
# # #             json={
# # #                 "intrinsic": intrinsic.tolist(),
# # #                 "stop_threshold": float(stop_threshold),
# # #                 "batch_size": int(batch_size),
# # #             },
# # #             timeout=self.timeout,
# # #         )
# # #         self._raise_for_error(response)
# # #         return str(response.json().get("algo", ""))

# # #     def plan(
# # #         self,
# # #         *,
# # #         goal_xy: np.ndarray,
# # #         rgb: np.ndarray,
# # #         depth: np.ndarray,
# # #         obstacle_pixels: np.ndarray,
# # #         goal_mode: str = "point",
# # #         forced_circulation_sign: float = 0.0,
# # #     ) -> NavDPS2DiffOutput:
# # #         goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
# # #         if goal_xy.shape != (2,):
# # #             raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")
# # #         if goal_mode not in {"point", "pixel"}:
# # #             raise ValueError("goal_mode must be point or pixel")
# # #         forced_circulation_sign = float(forced_circulation_sign)
# # #         if forced_circulation_sign not in {-1.0, 0.0, 1.0}:
# # #             raise ValueError("forced_circulation_sign must be -1, 0, or +1")

# # #         rgb = np.asarray(rgb, dtype=np.uint8)
# # #         if rgb.ndim != 3 or rgb.shape[-1] < 3:
# # #             raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
# # #         rgb = rgb[..., :3]

# # #         depth = np.asarray(depth, dtype=np.float32)
# # #         if depth.ndim == 3 and depth.shape[-1] == 1:
# # #             depth = depth[..., 0]
# # #         if depth.shape != rgb.shape[:2]:
# # #             raise ValueError(
# # #                 f"depth/rgb shape mismatch: {depth.shape} vs {rgb.shape[:2]}"
# # #             )

# # #         if goal_mode == "pixel":
# # #             if not np.all(np.isfinite(goal_xy)) or not np.allclose(
# # #                 goal_xy, np.round(goal_xy)
# # #             ):
# # #                 raise ValueError("PixelGoal must be integer [u,v]")
# # #             goal_xy = np.round(goal_xy).astype(np.int64)
# # #             if not (0 <= goal_xy[0] < rgb.shape[1] and 0 <= goal_xy[1] < rgb.shape[0]):
# # #                 raise ValueError("PixelGoal lies outside the RGB image")

# # #         pixels = np.asarray(obstacle_pixels)
# # #         if pixels.size == 0:
# # #             pixels = np.zeros((0, 2), dtype=np.int32)
# # #         else:
# # #             pixels = pixels.reshape(-1, 2)
# # #             if not np.all(np.isfinite(pixels)):
# # #                 raise ValueError("obstacle pixels must be finite")
# # #             if not np.allclose(pixels, np.round(pixels)):
# # #                 raise ValueError("obstacle pixels must be integer [u,v] coordinates")
# # #             pixels = np.round(pixels).astype(np.int32)

# # #         rgb_bytes = io.BytesIO()
# # #         Image.fromarray(rgb, mode="RGB").save(rgb_bytes, format="JPEG", quality=95)
# # #         depth_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
# # #         depth_bytes = io.BytesIO()
# # #         Image.fromarray(depth_u16).save(depth_bytes, format="PNG")

# # #         endpoint = "pixelgoal_step" if goal_mode == "pixel" else "pointgoal_step"
# # #         response = requests.post(
# # #             f"{self.server_url}/{endpoint}",
# # #             files={
# # #                 "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
# # #                 "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
# # #             },
# # #             data={
# # #                 "goal_data": json.dumps(
# # #                     {
# # #                         "goal_x": [float(goal_xy[0])],
# # #                         "goal_y": [float(goal_xy[1])],
# # #                         "obstacle_pixels": [pixels.tolist()],
# # #                         "forced_circulation_signs": [forced_circulation_sign],
# # #                     }
# # #                 )
# # #             },
# # #             timeout=self.timeout,
# # #         )
# # #         self._raise_for_error(response)
# # #         payload = response.json()
# # #         diagnostics = payload["s2diff"]
# # #         trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
# # #         all_trajectories = np.asarray(payload["all_trajectory"], dtype=np.float32)
# # #         all_values = np.asarray(payload["all_values"], dtype=np.float32)

# # #         return NavDPS2DiffOutput(
# # #             trajectory=trajectory[0],
# # #             all_trajectories=all_trajectories[0],
# # #             all_values=all_values[0],
# # #             selected_index=int(diagnostics["selected_index"][0]),
# # #             fallback_stop=bool(diagnostics["fallback_stop"][0]),
# # #             escape_turn=bool(diagnostics["escape_turn"][0]),
# # #             valid_obstacle_points=int(diagnostics["valid_obstacle_points"][0]),
# # #             selected_circulation_sign=float(
# # #                 diagnostics["selected_circulation_sign"][0]
# # #             ),
# # #             candidate_circulation_signs=np.asarray(
# # #                 diagnostics["candidate_circulation_signs"][0], dtype=np.float32
# # #             ),
# # #             selected_barrier_energy=float(diagnostics["selected_barrier_energy"][0]),
# # #             selected_circulation_energy=float(
# # #                 diagnostics["selected_circulation_energy"][0]
# # #             ),
# # #             minimum_clearance=np.asarray(
# # #                 diagnostics["minimum_clearance"][0], dtype=np.float32
# # #             ),
# # #             selected_minimum_clearance=float(
# # #                 diagnostics["selected_minimum_clearance"][0]
# # #             ),
# # #             mean_guidance_noise_correction=float(
# # #                 diagnostics["mean_guidance_noise_correction"][0]
# # #             ),
# # #             final_guidance_noise_correction=float(
# # #                 diagnostics["final_guidance_noise_correction"][0]
# # #             ),
# # #             maximum_guidance_noise_correction=float(
# # #                 diagnostics["maximum_guidance_noise_correction"][0]
# # #             ),
# # #             mean_final_effective_sample_size=float(
# # #                 diagnostics.get("mean_final_effective_sample_size", [0.0])[0]
# # #             ),
# # #         )

# # #     @staticmethod
# # #     def _raise_for_error(response: requests.Response) -> None:
# # #         try:
# # #             payload = response.json()
# # #         except ValueError:
# # #             payload = None
# # #         if isinstance(payload, dict) and "error" in payload:
# # #             raise RuntimeError(str(payload["error"]))
# # #         response.raise_for_status()


# # # @dataclass(frozen=True)
# # # class QwenHomotopyDecision:
# # #     side: str
# # #     circulation_sign: float
# # #     confidence: float
# # #     obstacle_relevant: bool
# # #     queried_qwen: bool
# # #     raw_response: Optional[str]
# # #     repeated_sides: tuple[str, ...]
# # #     repeated_confidences: tuple[float, ...]
# # #     consistency_rate: float
# # #     used_fallback: bool


# # # @dataclass(frozen=True)
# # # class QwenCommandDecision:
# # #     command: str
# # #     confidence: float
# # #     raw_response: str


# # # @dataclass(frozen=True)
# # # class QwenMissionPlanDecision:
# # #     plan: tuple[str, ...]
# # #     confidence: float
# # #     raw_response: str


# # # class QwenHomotopyClient:
# # #     """HTTP client for the isolated visual-Qwen process."""

# # #     def __init__(self, server_url: str, timeout: float = 300.0) -> None:
# # #         self.server_url = server_url.rstrip("/")
# # #         self.timeout = float(timeout)

# # #     def reset(self) -> None:
# # #         response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
# # #         self._raise_for_error(response)

# # #     def step(
# # #         self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
# # #     ) -> QwenHomotopyDecision:
# # #         image_bytes = io.BytesIO()
# # #         Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8)).save(
# # #             image_bytes, format="PNG"
# # #         )
# # #         mask_bytes = io.BytesIO()
# # #         Image.fromarray((np.asarray(obstacle_mask) > 0).astype(np.uint8) * 255).save(
# # #             mask_bytes, format="PNG"
# # #         )
# # #         response = requests.post(
# # #             f"{self.server_url}/select",
# # #             files={
# # #                 "image": ("overlay.png", image_bytes.getvalue(), "image/png"),
# # #                 "obstacle_mask": ("mask.png", mask_bytes.getvalue(), "image/png"),
# # #             },
# # #             timeout=self.timeout,
# # #         )
# # #         self._raise_for_error(response)
# # #         payload = response.json()
# # #         return QwenHomotopyDecision(
# # #             side=str(payload["side"]),
# # #             circulation_sign=float(payload["circulation_sign"]),
# # #             confidence=float(payload["confidence"]),
# # #             obstacle_relevant=bool(payload["obstacle_relevant"]),
# # #             queried_qwen=bool(payload["queried_qwen"]),
# # #             raw_response=payload.get("raw_response"),
# # #             repeated_sides=tuple(payload.get("repeated_sides", [])),
# # #             repeated_confidences=tuple(
# # #                 float(value) for value in payload.get("repeated_confidences", [])
# # #             ),
# # #             consistency_rate=float(payload.get("consistency_rate", 1.0)),
# # #             used_fallback=bool(payload.get("used_fallback", False)),
# # #         )

# # #     def classify_command(
# # #         self, image_rgb: np.ndarray, user_command: str
# # #     ) -> QwenCommandDecision:
# # #         image_bytes = io.BytesIO()
# # #         Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
# # #             image_bytes, format="PNG"
# # #         )
# # #         response = requests.post(
# # #             f"{self.server_url}/command",
# # #             files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
# # #             data={"command": str(user_command)},
# # #             timeout=self.timeout,
# # #         )
# # #         self._raise_for_error(response)
# # #         payload = response.json()
# # #         return QwenCommandDecision(
# # #             command=str(payload["command"]).upper(),
# # #             confidence=float(payload["confidence"]),
# # #             raw_response=str(payload["raw_response"]),
# # #         )

# # #     def classify_mission(
# # #         self, image_rgb: np.ndarray, user_command: str
# # #     ) -> QwenMissionPlanDecision:
# # #         image_bytes = io.BytesIO()
# # #         Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
# # #             image_bytes, format="PNG"
# # #         )
# # #         response = requests.post(
# # #             f"{self.server_url}/mission",
# # #             files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
# # #             data={"command": str(user_command)},
# # #             timeout=self.timeout,
# # #         )
# # #         self._raise_for_error(response)
# # #         payload = response.json()
# # #         return QwenMissionPlanDecision(
# # #             plan=tuple(str(item).upper() for item in payload["plan"]),
# # #             confidence=float(payload["confidence"]),
# # #             raw_response=str(payload["raw_response"]),
# # #         )

# # #     @staticmethod
# # #     def _raise_for_error(response: requests.Response) -> None:
# # #         try:
# # #             payload = response.json()
# # #         except ValueError:
# # #             payload = None
# # #         if isinstance(payload, dict) and "error" in payload:
# # #             raise RuntimeError(str(payload["error"]))
# # #         response.raise_for_status()


# # # def port_is_open(host: str, port: int) -> bool:
# # #     try:
# # #         with socket.create_connection((host, port), timeout=1.0):
# # #             return True
# # #     except OSError:
# # #         return False


# # # def wait_for_server(
# # #     process: subprocess.Popen[Any], host: str, port: int, timeout: float
# # # ) -> None:
# # #     deadline = time.time() + float(timeout)
# # #     while time.time() < deadline:
# # #         if process.poll() is not None:
# # #             raise RuntimeError(
# # #                 f"NavDP/S2Diff server exited with code {process.returncode}"
# # #             )
# # #         if port_is_open(host, port):
# # #             return
# # #         time.sleep(1.0)
# # #     raise TimeoutError(f"NavDP server did not open port {port} within {timeout}s")


# # # def stop_server(process: Optional[subprocess.Popen[Any]]) -> None:
# # #     if process is None or process.poll() is not None:
# # #         return
# # #     process.terminate()
# # #     try:
# # #         process.wait(timeout=10.0)
# # #     except subprocess.TimeoutExpired:
# # #         process.kill()
# # #         process.wait()


# # # def start_qwen_homotopy_server(
# # #     args: argparse.Namespace,
# # # ) -> Optional[subprocess.Popen[Any]]:
# # #     if not args.qwen_homotopy or not args.start_qwen_homotopy_server:
# # #         return None
# # #     if port_is_open(args.qwen_homotopy_host, args.qwen_homotopy_port):
# # #         raise RuntimeError(
# # #             f"Qwen homotopy port {args.qwen_homotopy_port} is already in use; "
# # #             "pass --no-start-qwen-homotopy-server to use an existing service"
# # #         )
# # #     server_file = HERE / "qwen_homotopy_server.py"
# # #     if not server_file.is_file():
# # #         raise FileNotFoundError(f"Qwen homotopy server not found: {server_file}")
# # #     command = [
# # #         str(args.qwen_homotopy_python),
# # #         str(server_file),
# # #         "--host",
# # #         str(args.qwen_homotopy_host),
# # #         "--port",
# # #         str(args.qwen_homotopy_port),
# # #         "--model-id",
# # #         str(args.qwen_model_id),
# # #         "--device",
# # #         str(args.qwen_device),
# # #         "--minimum-obstacle-pixels",
# # #         str(args.homotopy_minimum_obstacle_pixels),
# # #         "--release-clear-frames",
# # #         str(args.homotopy_release_clear_frames),
# # #         "--consistency-repeats",
# # #         str(args.homotopy_consistency_repeats),
# # #     ]
# # #     print("[qwen-server]", " ".join(command), flush=True)
# # #     process = subprocess.Popen(command, cwd=str(HERE))
# # #     wait_for_server(
# # #         process,
# # #         args.qwen_homotopy_host,
# # #         args.qwen_homotopy_port,
# # #         args.qwen_homotopy_timeout,
# # #     )
# # #     return process


# # # def start_server(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
# # #     if not args.start_server:
# # #         return None
# # #     if port_is_open(args.server_host, args.server_port):
# # #         raise RuntimeError(
# # #             f"port {args.server_port} is already in use; use --no-start-server "
# # #             "to connect to an existing guided server"
# # #         )

# # #     navdp_root = Path(args.navdp_root).expanduser().resolve()
# # #     checkpoint = Path(args.navdp_checkpoint).expanduser().resolve()
# # #     server_dir = navdp_root / "baselines" / "navdp"
# # #     server_file = server_dir / "navdp_s2diff_server.py"
# # #     if not server_file.is_file():
# # #         raise FileNotFoundError(f"guided server not found: {server_file}")
# # #     if not checkpoint.is_file():
# # #         raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint}")

# # #     command = [
# # #         str(args.navdp_python),
# # #         str(server_file),
# # #         "--checkpoint",
# # #         str(checkpoint),
# # #         "--device",
# # #         str(args.navdp_device),
# # #         "--planner-mode",
# # #         str(args.planner_mode),
# # #         "--seed",
# # #         str(args.seed),
# # #         "--port",
# # #         str(args.server_port),
# # #         "--candidates",
# # #         str(args.candidates),
# # #         "--particles",
# # #         str(args.particles),
# # #         "--particle-std",
# # #         str(args.particle_std),
# # #         "--gradient-steps",
# # #         str(args.gradient_steps),
# # #         "--gradient-step-size",
# # #         str(args.gradient_step_size),
# # #         "--guidance-strength",
# # #         str(args.guidance_strength),
# # #         "--temperature",
# # #         str(args.temperature),
# # #         "--safe-distance",
# # #         str(args.safe_distance),
# # #         "--hard-collision-distance",
# # #         str(args.hard_collision_distance),
# # #         "--robot-radius",
# # #         str(args.robot_radius),
# # #         "--safety-weight",
# # #         str(args.safety_weight),
# # #         "--barrier-weight",
# # #         str(args.barrier_weight),
# # #         "--barrier-rate",
# # #         str(args.barrier_rate),
# # #         "--circulation-weight",
# # #         str(args.circulation_weight),
# # #         "--circulation-activation-distance",
# # #         str(args.circulation_activation_distance),
# # #         "--circulation-activation-sharpness",
# # #         str(args.circulation_activation_sharpness),
# # #         "--minimum-circulation-progress",
# # #         str(args.minimum_circulation_progress),
# # #         "--blocking-alignment-threshold",
# # #         str(args.blocking_alignment_threshold),
# # #         "--circulation-switch-weight",
# # #         str(args.circulation_switch_weight),
# # #         "--escape-lateral-target",
# # #         str(args.escape_lateral_target),
# # #         "--minimum-obstacle-depth",
# # #         str(args.minimum_obstacle_depth),
# # #         "--maximum-obstacle-depth",
# # #         str(args.maximum_obstacle_depth),
# # #         "--maximum-obstacle-pixels",
# # #         str(args.maximum_obstacle_pixels),
# # #     ]
# # #     particle_flags = {
# # #         "particle-anchor": args.particle_anchor,
# # #         "particle-energy-reweighting": args.particle_energy_reweighting,
# # #         "particle-collision-mask": args.particle_collision_mask,
# # #         "particle-noise-schedule": args.particle_noise_schedule,
# # #         "progressive-guidance": args.progressive_guidance,
# # #     }
# # #     for name, enabled in particle_flags.items():
# # #         command.append(f"--{name}" if enabled else f"--no-{name}")
# # #     command.append("--remove-critic" if args.remove_critic else "--no-remove-critic")
# # #     print("[server]", " ".join(command), flush=True)
# # #     process = subprocess.Popen(command, cwd=str(server_dir))
# # #     wait_for_server(process, args.server_host, args.server_port, args.server_timeout)
# # #     return process


# # # def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
# # #     height, width = grid.shape
# # #     x0 = int(np.floor(px))
# # #     y0 = int(np.floor(py))
# # #     x1 = min(x0 + 1, width - 1)
# # #     y1 = min(y0 + 1, height - 1)
# # #     tx = px - x0
# # #     ty = py - y0
# # #     top = float(grid[y0, x0]) * (1.0 - tx) + float(grid[y0, x1]) * tx
# # #     bottom = float(grid[y1, x0]) * (1.0 - tx) + float(grid[y1, x1]) * tx
# # #     return top * (1.0 - ty) + bottom * ty


# # # class TerrainHeight:
# # #     def __init__(
# # #         self,
# # #         *,
# # #         mode: str,
# # #         heightmap: Optional[Path],
# # #         obj: Optional[Path],
# # #         flat_y: float,
# # #         size_x: float,
# # #         size_z: float,
# # #         size_y: float,
# # #         flip_x: bool,
# # #         flip_z: bool,
# # #         swap_xz: bool,
# # #     ):
# # #         if mode == "auto":
# # #             mode = (
# # #                 "heightmap"
# # #                 if heightmap and heightmap.exists()
# # #                 else ("obj" if obj and obj.exists() else "flat")
# # #             )
# # #         self.mode = mode
# # #         self.flat_y = float(flat_y)
# # #         self.size_x = float(size_x)
# # #         self.size_z = float(size_z)
# # #         self.size_y = float(size_y)
# # #         self.flip_x = bool(flip_x)
# # #         self.flip_z = bool(flip_z)
# # #         self.swap_xz = bool(swap_xz)
# # #         self.height: Optional[np.ndarray] = None
# # #         self.obj_xs: Optional[np.ndarray] = None
# # #         self.obj_zs: Optional[np.ndarray] = None
# # #         self.obj_h: Optional[np.ndarray] = None

# # #         if mode == "heightmap":
# # #             if heightmap is None or not heightmap.exists():
# # #                 raise FileNotFoundError(f"heightmap not found: {heightmap}")
# # #             array = np.asarray(Image.open(heightmap))
# # #             if array.ndim == 3:
# # #                 array = array[..., 0]
# # #             array = array.astype(np.float32)
# # #             array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
# # #             self.height = array * self.size_y - float(np.mean(array * self.size_y))
# # #         elif mode == "obj":
# # #             if obj is None or not obj.exists():
# # #                 raise FileNotFoundError(f"terrain OBJ not found: {obj}")
# # #             vertices = []
# # #             with obj.open("r", encoding="utf-8", errors="ignore") as file:
# # #                 for line in file:
# # #                     if line.startswith("v "):
# # #                         parts = line.split()
# # #                         if len(parts) >= 4:
# # #                             vertices.append(tuple(float(value) for value in parts[1:4]))
# # #             if not vertices:
# # #                 raise RuntimeError(f"no vertices found in {obj}")
# # #             array = np.asarray(vertices, dtype=np.float32)
# # #             xs = np.unique(array[:, 0])
# # #             zs = np.unique(array[:, 1])
# # #             grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
# # #             x_index = {float(value): index for index, value in enumerate(xs.tolist())}
# # #             z_index = {float(value): index for index, value in enumerate(zs.tolist())}
# # #             for x, z, height in array:
# # #                 grid[z_index[float(z)], x_index[float(x)]] = height
# # #             self.obj_xs = xs
# # #             self.obj_zs = zs
# # #             self.obj_h = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
# # #         elif mode != "flat":
# # #             raise ValueError(f"unknown terrain mode: {mode}")

# # #     def _map(self, x: float, z: float) -> tuple[float, float]:
# # #         if self.swap_xz:
# # #             x, z = z, x
# # #         u = (x + self.size_x / 2.0) / self.size_x
# # #         v = (z + self.size_z / 2.0) / self.size_z
# # #         if self.flip_x:
# # #             u = 1.0 - u
# # #         if self.flip_z:
# # #             v = 1.0 - v
# # #         return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

# # #     def __call__(self, x: float, z: float) -> float:
# # #         if self.mode == "flat":
# # #             return self.flat_y
# # #         if self.mode == "heightmap":
# # #             assert self.height is not None
# # #             u, v = self._map(x, z)
# # #             return bilinear_grid(
# # #                 self.height,
# # #                 u * (self.height.shape[1] - 1),
# # #                 v * (self.height.shape[0] - 1),
# # #             )
# # #         assert (
# # #             self.obj_xs is not None
# # #             and self.obj_zs is not None
# # #             and self.obj_h is not None
# # #         )
# # #         xx = float(np.clip(x, self.obj_xs[0], self.obj_xs[-1]))
# # #         zz = float(np.clip(z, self.obj_zs[0], self.obj_zs[-1]))
# # #         column = int(
# # #             np.clip(np.searchsorted(self.obj_xs, xx) - 1, 0, len(self.obj_xs) - 2)
# # #         )
# # #         row = int(
# # #             np.clip(np.searchsorted(self.obj_zs, zz) - 1, 0, len(self.obj_zs) - 2)
# # #         )
# # #         x0, x1 = float(self.obj_xs[column]), float(self.obj_xs[column + 1])
# # #         z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
# # #         tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
# # #         tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
# # #         top = (
# # #             float(self.obj_h[row, column]) * (1.0 - tx)
# # #             + float(self.obj_h[row, column + 1]) * tx
# # #         )
# # #         bottom = (
# # #             float(self.obj_h[row + 1, column]) * (1.0 - tx)
# # #             + float(self.obj_h[row + 1, column + 1]) * tx
# # #         )
# # #         return top * (1.0 - tz) + bottom * tz

# # #     def local_height_max(
# # #         self, x: float, z: float, radius: float, samples: int = 5
# # #     ) -> float:
# # #         if radius <= 1e-6:
# # #             return float(self(x, z))
# # #         values = [
# # #             float(self(x + dx, z + dz))
# # #             for dx in np.linspace(-radius, radius, samples)
# # #             for dz in np.linspace(-radius, radius, samples)
# # #             if dx * dx + dz * dz <= radius * radius + 1e-8
# # #         ]
# # #         return max(values) if values else float(self(x, z))


# # # def make_sensor(
# # #     uuid: str, sensor_type: Any, height: int, width: int, hfov_deg: float
# # # ) -> habitat_sim.CameraSensorSpec:
# # #     specification = habitat_sim.CameraSensorSpec()
# # #     specification.uuid = uuid
# # #     specification.sensor_type = sensor_type
# # #     specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
# # #     specification.resolution = [int(height), int(width)]
# # #     specification.position = [0.0, 0.0, 0.0]
# # #     specification.hfov = float(hfov_deg)
# # #     return specification


# # # def make_simulator(
# # #     scene: Path,
# # #     height: int,
# # #     width: int,
# # #     hfov_deg: float,
# # #     *,
# # #     with_semantic: bool,
# # # ):
# # #     simulator_configuration = habitat_sim.SimulatorConfiguration()
# # #     simulator_configuration.scene_id = str(scene.expanduser().resolve())
# # #     simulator_configuration.enable_physics = False
# # #     sensors = [
# # #         make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
# # #         make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
# # #     ]
# # #     if with_semantic:
# # #         sensors.append(
# # #             make_sensor(
# # #                 "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
# # #             )
# # #         )
# # #     agent_configuration = AgentConfiguration()
# # #     agent_configuration.sensor_specifications = sensors
# # #     return habitat_sim.Simulator(
# # #         habitat_sim.Configuration(simulator_configuration, [agent_configuration])
# # #     )


# # # def set_agent_pose(agent: Any, position: np.ndarray, yaw: float) -> None:
# # #     state = agent.get_state()
# # #     state.position = np.asarray(position, dtype=np.float32)
# # #     state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
# # #     agent.set_state(state)


# # # def rgb_depth(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
# # #     rgb = np.asarray(observation["rgb"])
# # #     if rgb.ndim == 3 and rgb.shape[-1] == 4:
# # #         rgb = rgb[..., :3]
# # #     depth = np.asarray(observation["depth"], dtype=np.float32)
# # #     if depth.ndim == 3:
# # #         depth = depth[..., 0]
# # #     return rgb.astype(np.uint8), depth.astype(np.float32)


# # # def semantic_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
# # #     semantic = np.asarray(observation["semantic"])
# # #     if semantic.ndim == 3:
# # #         semantic = semantic[..., 0]
# # #     return semantic.astype(np.int32)


# # # def pixel_to_world(
# # #     u: float,
# # #     v: float,
# # #     depth: float,
# # #     position: np.ndarray,
# # #     yaw: float,
# # #     intrinsic: np.ndarray,
# # # ) -> np.ndarray:
# # #     right = (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
# # #     up = -(v - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
# # #     forward_vector = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)])
# # #     right_vector = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)])
# # #     return (
# # #         np.asarray(position, dtype=np.float64)
# # #         + depth * forward_vector
# # #         + right * right_vector
# # #         + up * np.asarray([0.0, 1.0, 0.0])
# # #     )


# # # def depth_patch_mesh(
# # #     u_center: float,
# # #     v_center: float,
# # #     half_size: int,
# # #     stride: int,
# # #     depth: np.ndarray,
# # #     position: np.ndarray,
# # #     yaw: float,
# # #     intrinsic: np.ndarray,
# # #     *,
# # #     lift: float,
# # #     maximum_depth_jump: float = 0.4,
# # # ) -> tuple[np.ndarray, np.ndarray]:
# # #     height, width = depth.shape
# # #     columns = list(
# # #         range(
# # #             max(0, int(u_center - half_size)),
# # #             min(width, int(u_center + half_size) + 1),
# # #             max(int(stride), 1),
# # #         )
# # #     )
# # #     rows = list(
# # #         range(
# # #             max(0, int(v_center - half_size)),
# # #             min(height, int(v_center + half_size) + 1),
# # #             max(int(stride), 1),
# # #         )
# # #     )
# # #     indices = -np.ones((len(rows), len(columns)), dtype=np.int64)
# # #     depths = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
# # #     vertices: list[tuple[float, float, float]] = []
# # #     for row_index, v in enumerate(rows):
# # #         for column_index, u in enumerate(columns):
# # #             metric_depth = float(depth[v, u])
# # #             if not np.isfinite(metric_depth) or metric_depth <= 0.1:
# # #                 continue
# # #             indices[row_index, column_index] = len(vertices)
# # #             depths[row_index, column_index] = metric_depth
# # #             point = pixel_to_world(
# # #                 u, v, metric_depth, position, yaw, intrinsic
# # #             ) + float(lift) * np.asarray([0.0, 1.0, 0.0])
# # #             vertices.append(tuple(float(value) for value in point))

# # #     faces: list[tuple[int, int, int]] = []
# # #     for row_index in range(len(rows) - 1):
# # #         for column_index in range(len(columns) - 1):
# # #             a = int(indices[row_index, column_index])
# # #             b = int(indices[row_index, column_index + 1])
# # #             c = int(indices[row_index + 1, column_index])
# # #             d = int(indices[row_index + 1, column_index + 1])
# # #             if min(a, b, c, d) < 0:
# # #                 continue
# # #             cell_depths = (
# # #                 depths[row_index, column_index],
# # #                 depths[row_index, column_index + 1],
# # #                 depths[row_index + 1, column_index],
# # #                 depths[row_index + 1, column_index + 1],
# # #             )
# # #             if max(cell_depths) - min(cell_depths) > maximum_depth_jump:
# # #                 continue
# # #             faces.append((a, c, d))
# # #             faces.append((a, d, b))
# # #     return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# # # def save_obj(
# # #     path: Path,
# # #     vertices: np.ndarray,
# # #     faces: np.ndarray,
# # #     *,
# # #     diffuse_rgb: Optional[tuple[float, float, float]] = None,
# # # ) -> None:
# # #     material_name = None
# # #     if diffuse_rgb is not None:
# # #         red, green, blue = (float(value) for value in diffuse_rgb)
# # #         if not all(0.0 <= value <= 1.0 for value in (red, green, blue)):
# # #             raise ValueError("OBJ diffuse material values must be in [0, 1]")
# # #         material_name = "mesh_material"
# # #         material_path = path.with_suffix(".mtl")
# # #         with material_path.open("w", encoding="utf-8") as material:
# # #             material.write(f"newmtl {material_name}\n")
# # #             material.write(
# # #                 f"Ka {0.25 * red:.4f} {0.25 * green:.4f} {0.25 * blue:.4f}\n"
# # #             )
# # #             material.write(f"Kd {red:.4f} {green:.4f} {blue:.4f}\n")
# # #             material.write("Ks 0.1000 0.1000 0.1000\n")
# # #             material.write("Ns 24.0000\n")
# # #             material.write("d 1.0000\n")
# # #             material.write("illum 2\n")
# # #     with path.open("w", encoding="utf-8") as file:
# # #         if material_name is not None:
# # #             file.write(f"mtllib {path.with_suffix('.mtl').name}\n")
# # #             file.write(f"usemtl {material_name}\n")
# # #         for x, y, z in vertices:
# # #             file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
# # #         for a, b, c in faces:
# # #             file.write(f"f {a + 1} {b + 1} {c + 1}\n")


# # # def register_semantic_mesh(simulator: Any, mesh_path: Path, semantic_id: int) -> Any:
# # #     template_manager = simulator.get_object_template_manager()
# # #     object_manager = simulator.get_rigid_object_manager()
# # #     template = template_manager.create_new_template(str(mesh_path))
# # #     template.render_asset_handle = str(mesh_path)
# # #     template.collision_asset_handle = str(mesh_path)
# # #     template.is_collidable = False
# # #     template_id = template_manager.register_template(
# # #         template, f"s2diff_obstacle_{semantic_id}_{os.path.basename(mesh_path)}"
# # #     )
# # #     object_handle = template_manager.get_template_handle_by_id(template_id)
# # #     obstacle = object_manager.add_object_by_template_handle(object_handle)
# # #     obstacle.motion_type = habitat_sim.physics.MotionType.KINEMATIC
# # #     obstacle.collidable = False
# # #     obstacle.semantic_id = int(semantic_id)
# # #     return obstacle


# # # def parse_world_xz(specification: str) -> tuple[float, float]:
# # #     values = [float(value) for value in str(specification).split(",")]
# # #     if len(values) != 2 or not np.isfinite(values).all():
# # #         raise ValueError(
# # #             f"world mesh position must be finite X,Z, got {specification!r}"
# # #         )
# # #     return values[0], values[1]


# # # def world_box_mesh(
# # #     center_x: float,
# # #     base_y: float,
# # #     center_z: float,
# # #     half_extent: float,
# # #     height: float,
# # # ) -> tuple[np.ndarray, np.ndarray]:
# # #     """Create a closed axis-aligned box whose vertices are in world coordinates."""

# # #     if half_extent <= 0.0 or height <= 0.0:
# # #         raise ValueError("box half extent and height must be positive")
# # #     x0, x1 = center_x - half_extent, center_x + half_extent
# # #     z0, z1 = center_z - half_extent, center_z + half_extent
# # #     y0, y1 = base_y, base_y + height
# # #     vertices = np.asarray(
# # #         [
# # #             [x0, y0, z0],
# # #             [x1, y0, z0],
# # #             [x1, y0, z1],
# # #             [x0, y0, z1],
# # #             [x0, y1, z0],
# # #             [x1, y1, z0],
# # #             [x1, y1, z1],
# # #             [x0, y1, z1],
# # #         ],
# # #         dtype=np.float64,
# # #     )
# # #     faces = np.asarray(
# # #         [
# # #             [0, 2, 1],
# # #             [0, 3, 2],
# # #             [4, 5, 6],
# # #             [4, 6, 7],
# # #             [0, 1, 5],
# # #             [0, 5, 4],
# # #             [1, 2, 6],
# # #             [1, 6, 5],
# # #             [2, 3, 7],
# # #             [2, 7, 6],
# # #             [3, 0, 4],
# # #             [3, 4, 7],
# # #         ],
# # #         dtype=np.int64,
# # #     )
# # #     return vertices, faces


# # # def place_world_obstacle_meshes(
# # #     simulator: Any,
# # #     terrain: Any,
# # #     xz_specifications: Sequence[str],
# # #     output_directory: Path,
# # #     *,
# # #     half_extent: float,
# # #     height: float,
# # # ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
# # #     """Place static obstacle boxes at exact world X,Z coordinates."""

# # #     mesh_directory = output_directory / "meshes"
# # #     mesh_directory.mkdir(parents=True, exist_ok=True)
# # #     objects: list[Any] = []
# # #     centroids: list[np.ndarray] = []
# # #     geometries: list[np.ndarray] = []
# # #     for index, specification in enumerate(xz_specifications):
# # #         center_x, center_z = parse_world_xz(specification)
# # #         base_y = terrain.local_height_max(center_x, center_z, half_extent)
# # #         vertices, faces = world_box_mesh(
# # #             center_x, base_y, center_z, half_extent, height
# # #         )
# # #         mesh_path = mesh_directory / f"world_obstacle_{index}.obj"
# # #         save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.78, 0.16, 0.06))
# # #         semantic_id = MESH_OBSTACLE_ID + index
# # #         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
# # #         centroid = vertices.mean(axis=0).astype(np.float32)
# # #         centroids.append(centroid)
# # #         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
# # #         print(
# # #             f"[world-mesh] obstacle={index} semantic_id={semantic_id} "
# # #             f"center_xz={[center_x, center_z]} half_extent={half_extent:.3f} "
# # #             f"height={height:.3f}",
# # #             flush=True,
# # #         )
# # #     return objects, centroids, geometries


# # # def place_world_goal_mesh(
# # #     simulator: Any,
# # #     terrain: Any,
# # #     goal_x: float,
# # #     goal_z: float,
# # #     output_directory: Path,
# # #     *,
# # #     half_extent: float,
# # #     height: float,
# # # ) -> Any:
# # #     """Place a visible, non-obstacle semantic goal marker at the exact goal."""

# # #     base_y = terrain.local_height_max(goal_x, goal_z, half_extent)
# # #     vertices, faces = world_box_mesh(goal_x, base_y, goal_z, half_extent, height)
# # #     mesh_directory = output_directory / "meshes"
# # #     mesh_directory.mkdir(parents=True, exist_ok=True)
# # #     mesh_path = mesh_directory / "goal_marker.obj"
# # #     save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.08, 0.85, 0.18))
# # #     goal_object = register_semantic_mesh(simulator, mesh_path, MESH_GOAL_ID)
# # #     print(
# # #         f"[world-mesh] goal semantic_id={MESH_GOAL_ID} "
# # #         f"center_xz={[goal_x, goal_z]}",
# # #         flush=True,
# # #     )
# # #     return goal_object


# # # def parse_uv_fraction(
# # #     specification: str, width: int, height: int
# # # ) -> tuple[float, float]:
# # #     u_fraction, v_fraction = (float(value) for value in str(specification).split(","))
# # #     if not (0.0 <= u_fraction <= 1.0 and 0.0 <= v_fraction <= 1.0):
# # #         raise ValueError(f"mesh pixel fraction must be in [0,1], got {specification!r}")
# # #     return u_fraction * width, v_fraction * height


# # # def place_obstacle_meshes(
# # #     simulator: Any,
# # #     depth: np.ndarray,
# # #     position: np.ndarray,
# # #     yaw: float,
# # #     intrinsic: np.ndarray,
# # #     uv_specifications: Sequence[str],
# # #     output_directory: Path,
# # #     *,
# # #     mesh_half_pixels: int,
# # #     mesh_lift: float,
# # # ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
# # #     mesh_directory = output_directory / "meshes"
# # #     mesh_directory.mkdir(parents=True, exist_ok=True)
# # #     height, width = depth.shape
# # #     objects: list[Any] = []
# # #     centroids: list[np.ndarray] = []
# # #     geometries: list[np.ndarray] = []
# # #     for index, specification in enumerate(uv_specifications):
# # #         u, v = parse_uv_fraction(specification, width, height)
# # #         vertices, faces = depth_patch_mesh(
# # #             u,
# # #             v,
# # #             mesh_half_pixels,
# # #             2,
# # #             depth,
# # #             position,
# # #             yaw,
# # #             intrinsic,
# # #             lift=mesh_lift,
# # #         )
# # #         if len(vertices) == 0 or len(faces) == 0:
# # #             raise RuntimeError(
# # #                 f"obstacle mesh {index} at {specification!r} has no valid depth surface"
# # #             )
# # #         mesh_path = mesh_directory / f"obstacle_{index}.obj"
# # #         save_obj(mesh_path, vertices, faces)
# # #         semantic_id = MESH_OBSTACLE_ID + index
# # #         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
# # #         centroid = vertices.mean(axis=0).astype(np.float32)
# # #         centroids.append(centroid)
# # #         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
# # #         print(
# # #             f"[mesh] obstacle={index} semantic_id={semantic_id} "
# # #             f"pixels={specification} vertices={len(vertices)} "
# # #             f"world={centroid.tolist()}",
# # #             flush=True,
# # #         )
# # #     return objects, centroids, geometries


# # # def planar_mesh_clearance(
# # #     point_xz: np.ndarray,
# # #     geometries: Sequence[np.ndarray],
# # # ) -> float:
# # #     """Minimum 2-D distance from a robot center to projected mesh triangles."""
# # #     point = np.asarray(point_xz, dtype=np.float64)
# # #     best = float("inf")
# # #     for triangles in geometries:
# # #         triangles = np.asarray(triangles, dtype=np.float64)
# # #         if triangles.size == 0:
# # #             continue
# # #         a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
# # #         v0, v1, v2 = b - a, c - a, point[None, :] - a
# # #         denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
# # #         valid = np.abs(denominator) > 1.0e-12
# # #         safe_denominator = np.where(valid, denominator, 1.0)
# # #         u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_denominator
# # #         v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_denominator
# # #         if np.any(valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)):
# # #             return 0.0

# # #         starts = np.concatenate((a, b, c), axis=0)
# # #         ends = np.concatenate((b, c, a), axis=0)
# # #         segments = ends - starts
# # #         squared_lengths = np.einsum("ij,ij->i", segments, segments)
# # #         numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
# # #         fractions = np.divide(
# # #             numerators,
# # #             squared_lengths,
# # #             out=np.zeros_like(numerators),
# # #             where=squared_lengths > 1.0e-12,
# # #         )
# # #         fractions = np.clip(fractions, 0.0, 1.0)
# # #         closest = starts + fractions[:, None] * segments
# # #         best = min(best, float(np.linalg.norm(point[None, :] - closest, axis=1).min()))
# # #     return best


# # # def parse_xz_velocity(specification: str) -> np.ndarray:
# # #     values = [float(value) for value in str(specification).split(",")]
# # #     if len(values) != 2 or not np.all(np.isfinite(values)):
# # #         raise ValueError("obstacle velocity must be finite vx,vz")
# # #     return np.asarray(values, dtype=np.float64)


# # # def expand_obstacle_velocities(
# # #     specifications: Sequence[str], obstacle_count: int
# # # ) -> np.ndarray:
# # #     if obstacle_count == 0:
# # #         return np.zeros((0, 2), dtype=np.float64)
# # #     if not specifications:
# # #         return np.zeros((obstacle_count, 2), dtype=np.float64)
# # #     velocities = np.stack([parse_xz_velocity(item) for item in specifications])
# # #     if len(velocities) == 1 and obstacle_count > 1:
# # #         velocities = np.repeat(velocities, obstacle_count, axis=0)
# # #     if len(velocities) != obstacle_count:
# # #         raise ValueError(
# # #             "provide one obstacle velocity to broadcast or one velocity per mesh"
# # #         )
# # #     return velocities


# # # def translated_mesh_geometry(
# # #     base_geometries: Sequence[np.ndarray],
# # #     base_centroids: Sequence[np.ndarray],
# # #     velocities_xz: np.ndarray,
# # #     elapsed_seconds: float,
# # # ) -> tuple[list[np.ndarray], list[np.ndarray]]:
# # #     geometries: list[np.ndarray] = []
# # #     centroids: list[np.ndarray] = []
# # #     for geometry, centroid, velocity in zip(
# # #         base_geometries, base_centroids, velocities_xz
# # #     ):
# # #         offset_xz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
# # #         geometries.append(np.asarray(geometry) + offset_xz[None, None, :])
# # #         offset_xyz = np.asarray([offset_xz[0], 0.0, offset_xz[1]])
# # #         centroids.append(np.asarray(centroid, dtype=np.float64) + offset_xyz)
# # #     return geometries, centroids


# # # def move_mesh_objects(
# # #     objects: Sequence[Any], velocities_xz: np.ndarray, elapsed_seconds: float
# # # ) -> None:
# # #     for obstacle, velocity in zip(objects, velocities_xz):
# # #         dx, dz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
# # #         vector_type = type(obstacle.translation)
# # #         obstacle.translation = vector_type(float(dx), 0.0, float(dz))


# # # def camera_coordinates(
# # #     point: np.ndarray, position: np.ndarray, yaw: float
# # # ) -> tuple[float, float, float]:
# # #     delta = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
# # #     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
# # #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# # #     forward = forward_x * float(delta[0]) + forward_z * float(delta[2])
# # #     left = left_x * float(delta[0]) + left_z * float(delta[2])
# # #     return -left, float(delta[1]), forward


# # # def camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
# # #     hfov = math.radians(float(hfov_deg))
# # #     focal = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
# # #     return np.asarray(
# # #         [
# # #             [focal, 0.0, (width - 1) * 0.5],
# # #             [0.0, focal, (height - 1) * 0.5],
# # #             [0.0, 0.0, 1.0],
# # #         ],
# # #         dtype=np.float32,
# # #     )


# # # def world_goal_to_pixel(
# # #     point: np.ndarray,
# # #     position: np.ndarray,
# # #     yaw: float,
# # #     intrinsic: np.ndarray,
# # #     height: int,
# # #     width: int,
# # # ) -> np.ndarray:
# # #     """Project a world goal to a valid PixelGoal, clamping off-screen bearings."""

# # #     right, up, forward = camera_coordinates(point, position, yaw)
# # #     fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
# # #     cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
# # #     margin = 11
# # #     bearing = math.atan2(right, forward)
# # #     maximum_bearing = math.atan2(max(cx - margin, 1.0), fx)
# # #     bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
# # #     u = cx + fx * math.tan(bearing)
# # #     v = cy - fy * up / forward if forward > 0.05 else 0.62 * height
# # #     return np.asarray(
# # #         [
# # #             int(np.clip(round(u), margin, width - margin - 1)),
# # #             int(np.clip(round(v), margin, height - margin - 1)),
# # #         ],
# # #         dtype=np.int32,
# # #     )


# # # def circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
# # #     yy, xx = np.ogrid[:height, :width]
# # #     return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


# # # def project_world_mask(
# # #     point: np.ndarray,
# # #     position: np.ndarray,
# # #     yaw: float,
# # #     intrinsic: np.ndarray,
# # #     height: int,
# # #     width: int,
# # #     radius: int,
# # # ) -> tuple[np.ndarray, float]:
# # #     right, up, forward = camera_coordinates(point, position, yaw)
# # #     if forward <= 0.05:
# # #         return np.zeros((height, width), dtype=np.uint8), forward
# # #     u = float(intrinsic[0, 2] + intrinsic[0, 0] * right / forward)
# # #     v = float(intrinsic[1, 2] - intrinsic[1, 1] * up / forward)
# # #     if not (radius <= u < width - radius and radius <= v < height - radius):
# # #         return np.zeros((height, width), dtype=np.uint8), forward
# # #     return circle_mask(height, width, u, v, radius), forward


# # # def depth_obstacle_mask(
# # #     depth: np.ndarray, threshold: float, minimum_y_fraction: float
# # # ) -> np.ndarray:
# # #     mask = np.isfinite(depth) & (depth > 0.05) & (depth < float(threshold))
# # #     mask[: int(depth.shape[0] * minimum_y_fraction)] = False
# # #     return mask.astype(np.uint8)


# # # def pixels_from_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
# # #     v, u = np.nonzero(np.asarray(mask) > 0)
# # #     if u.size == 0:
# # #         return np.zeros((0, 2), dtype=np.int32)
# # #     pixels = np.stack((u, v), axis=-1).astype(np.int32)
# # #     if maximum > 0 and len(pixels) > maximum:
# # #         indices = np.linspace(0, len(pixels) - 1, maximum).astype(np.int64)
# # #         pixels = pixels[indices]
# # #     return pixels


# # # def waypoint_action(
# # #     trajectory: np.ndarray,
# # #     *,
# # #     lookahead_index: int,
# # #     maximum_forward_speed: float,
# # #     maximum_yaw_rate: float,
# # #     yaw_gain: float,
# # # ) -> np.ndarray:
# # #     trajectory = np.asarray(trajectory, dtype=np.float32)
# # #     if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
# # #         return np.zeros(3, dtype=np.float32)
# # #     if np.max(np.linalg.norm(trajectory[:, :2], axis=-1)) < 1e-5:
# # #         return np.zeros(3, dtype=np.float32)
# # #     index = int(np.clip(lookahead_index, 0, trajectory.shape[0] - 1))
# # #     forward, left = float(trajectory[index, 0]), float(trajectory[index, 1])
# # #     bearing = math.atan2(left, max(forward, 1e-4))
# # #     velocity = maximum_forward_speed * max(0.0, math.cos(bearing))
# # #     yaw_rate = float(np.clip(yaw_gain * bearing, -maximum_yaw_rate, maximum_yaw_rate))
# # #     return np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)


# # # def integrate_mars(
# # #     position: np.ndarray, yaw: float, action: np.ndarray, dt: float
# # # ) -> tuple[np.ndarray, float]:
# # #     forward_velocity, lateral_velocity, yaw_rate = [float(value) for value in action]
# # #     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
# # #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# # #     output = np.asarray(position, dtype=np.float32).copy()
# # #     output[0] += (forward_x * forward_velocity + left_x * lateral_velocity) * dt
# # #     output[2] += (forward_z * forward_velocity + left_z * lateral_velocity) * dt
# # #     return output, yaw + yaw_rate * dt


# # # def wrap_angle(angle: float) -> float:
# # #     return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# # # def overlay_frame(
# # #     rgb: np.ndarray,
# # #     goal_mask: np.ndarray,
# # #     obstacle_mask: np.ndarray,
# # #     text: str,
# # #     *,
# # #     show_masks: bool,
# # #     detection_box: Optional[np.ndarray] = None,
# # #     detection_label: Optional[str] = None,
# # # ) -> Image.Image:
# # #     output = np.asarray(rgb, dtype=np.uint8).copy()
# # #     if show_masks:
# # #         output[goal_mask > 0] = (
# # #             0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
# # #         ).astype(np.uint8)
# # #         output[obstacle_mask > 0] = (
# # #             0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
# # #         ).astype(np.uint8)
# # #     image = Image.fromarray(output)
# # #     draw = ImageDraw.Draw(image)
# # #     if detection_box is not None:
# # #         x1, y1, x2, y2 = [float(value) for value in detection_box]
# # #         draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
# # #         if detection_label:
# # #             draw.text((x1 + 2, max(y1 - 14, 2)), detection_label, fill=(255, 255, 0))
# # #     draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
# # #     draw.text((10, 9), text, fill=(255, 255, 255))
# # #     return image


# # # def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
# # #     """Dilate a binary image mask without adding a SciPy dependency."""

# # #     binary = (np.asarray(mask) > 0).astype(np.uint8)
# # #     if radius <= 0 or not np.any(binary):
# # #         return binary
# # #     kernel_size = 2 * int(radius) + 1
# # #     return (
# # #         np.asarray(
# # #             Image.fromarray(binary * 255).filter(ImageFilter.MaxFilter(kernel_size))
# # #         )
# # #         > 0
# # #     ).astype(np.uint8)


# # # def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
# # #     import imageio.v2 as imageio

# # #     with imageio.get_writer(path, fps=float(fps)) as writer:
# # #         for frame in frames:
# # #             writer.append_data(np.asarray(frame.convert("RGB")))


# # # def parser() -> argparse.ArgumentParser:
# # #     argument_parser = argparse.ArgumentParser(
# # #         description="One-file released NavDP + in-denoising S2Diff Mars rollout"
# # #     )
# # #     argument_parser.add_argument("--navdp-root", required=True)
# # #     argument_parser.add_argument("--navdp-checkpoint", required=True)
# # #     argument_parser.add_argument("--navdp-python", default=sys.executable)
# # #     argument_parser.add_argument("--navdp-device", default="cuda:0")
# # #     argument_parser.add_argument(
# # #         "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
# # #     )
# # #     argument_parser.add_argument(
# # #         "--goal-mode", choices=["point", "pixel"], default="point"
# # #     )
# # #     argument_parser.add_argument(
# # #         "--belief-pixel-goal",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=False,
# # #         help=(
# # #             "Use the live semantic goal mask to correct a body-frame Gaussian "
# # #             "belief and its projected mean as NavDP's PixelGoal while occluded."
# # #         ),
# # #     )
# # #     argument_parser.add_argument("--belief-minimum-goal-pixels", type=int, default=10)
# # #     argument_parser.add_argument("--belief-measurement-std", type=float, default=0.05)
# # #     argument_parser.add_argument(
# # #         "--belief-translation-process-std", type=float, default=0.03
# # #     )
# # #     argument_parser.add_argument(
# # #         "--belief-yaw-process-std-deg", type=float, default=1.0
# # #     )
# # #     argument_parser.add_argument(
# # #         "--belief-bootstrap-world-goal",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=False,
# # #         help=(
# # #             "Simulation-only bootstrap when the goal is initially invisible. "
# # #             "Disable for a strict detector-only evaluation."
# # #         ),
# # #     )
# # #     argument_parser.add_argument("--belief-bootstrap-std", type=float, default=0.50)
# # #     argument_parser.add_argument("--belief-ghost-base-radius", type=int, default=10)
# # #     argument_parser.add_argument(
# # #         "--belief-ghost-covariance-scale", type=float, default=2.0
# # #     )
# # #     argument_parser.add_argument("--belief-ghost-maximum-radius", type=int, default=80)
# # #     argument_parser.add_argument(
# # #         "--belief-heading-recovery",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=True,
# # #     )
# # #     argument_parser.add_argument(
# # #         "--belief-recovery-bearing-deg", type=float, default=35.0
# # #     )
# # #     argument_parser.add_argument("--belief-recovery-yaw-gain", type=float, default=1.5)
# # #     argument_parser.add_argument(
# # #         "--belief-recovery-maximum-yaw-rate", type=float, default=0.70
# # #     )
# # #     argument_parser.add_argument(
# # #         "--belief-recovery-maximum-forward-speed", type=float, default=0.12
# # #     )
# # #     argument_parser.add_argument(
# # #         "--interactive-return-home",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=False,
# # #         help=(
# # #             "At the outward goal, ask for a command, let Qwen classify RETURN "
# # #             "or STOP, and use a separately propagated spawn/home PixelGoal belief."
# # #         ),
# # #     )
# # #     argument_parser.add_argument(
# # #         "--return-command",
# # #         default=None,
# # #         help="Optional non-interactive command text, for example 'come back'.",
# # #     )
# # #     argument_parser.add_argument(
# # #         "--qwen-freeform-mission",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=False,
# # #         help=(
# # #             "Ask once at startup for a free-form instruction. Qwen emits either "
# # #             "GO_TO_GOAL or GO_TO_GOAL followed by RETURN_HOME."
# # #         ),
# # #     )
# # #     argument_parser.add_argument(
# # #         "--mission-command",
# # #         default=None,
# # #         help=(
# # #             "Optional non-interactive free-form mission, for example "
# # #             "'visit the target and report back'."
# # #         ),
# # #     )
# # #     argument_parser.add_argument(
# # #         "--return-goal-obstacle-activation-distance", type=float, default=1.35
# # #     )
# # #     argument_parser.add_argument(
# # #         "--return-goal-obstacle-dilation-pixels", type=int, default=30
# # #     )
# # #     argument_parser.add_argument(
# # #         "--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct"
# # #     )
# # #     argument_parser.add_argument("--qwen-device", default="auto")
# # #     argument_parser.add_argument("--qwen-homotopy-python", default=sys.executable)
# # #     argument_parser.add_argument("--qwen-homotopy-host", default="127.0.0.1")
# # #     argument_parser.add_argument("--qwen-homotopy-port", type=int, default=8890)
# # #     argument_parser.add_argument("--qwen-homotopy-timeout", type=float, default=600.0)
# # #     argument_parser.add_argument(
# # #         "--start-qwen-homotopy-server",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=True,
# # #     )
# # #     argument_parser.add_argument(
# # #         "--qwen-homotopy",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=False,
# # #         help=(
# # #             "When a metric obstacle becomes relevant, Qwen chooses the single "
# # #             "LEFT/RIGHT circulation sign used by every trajectory candidate."
# # #         ),
# # #     )
# # #     argument_parser.add_argument(
# # #         "--homotopy-minimum-obstacle-pixels", type=int, default=30
# # #     )
# # #     argument_parser.add_argument("--homotopy-release-clear-frames", type=int, default=8)
# # #     argument_parser.add_argument(
# # #         "--homotopy-consistency-repeats",
# # #         type=int,
# # #         default=5,
# # #         help="Repeat Qwen on the identical obstacle frame and use majority vote.",
# # #     )
# # #     argument_parser.add_argument(
# # #         "--remove-critic", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument("--seed", type=int, default=7)
# # #     argument_parser.add_argument(
# # #         "--start-server", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument("--server-host", default="127.0.0.1")
# # #     argument_parser.add_argument("--server-port", type=int, default=8888)
# # #     argument_parser.add_argument("--server-timeout", type=float, default=180.0)
# # #     argument_parser.add_argument("--candidates", type=int, default=16)
# # #     argument_parser.add_argument("--particles", type=int, default=8)
# # #     argument_parser.add_argument("--particle-std", type=float, default=0.22)
# # #     argument_parser.add_argument("--gradient-steps", type=int, default=3)
# # #     argument_parser.add_argument("--gradient-step-size", type=float, default=0.04)
# # #     argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
# # #     argument_parser.add_argument("--temperature", type=float, default=0.35)
# # #     argument_parser.add_argument(
# # #         "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument(
# # #         "--particle-energy-reweighting",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=True,
# # #     )
# # #     argument_parser.add_argument(
# # #         "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument(
# # #         "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument(
# # #         "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument("--safe-distance", type=float, default=0.42)
# # #     argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
# # #     argument_parser.add_argument("--safety-weight", type=float, default=35.0)
# # #     argument_parser.add_argument("--barrier-weight", type=float, default=25.0)
# # #     argument_parser.add_argument("--barrier-rate", type=float, default=0.15)
# # #     argument_parser.add_argument("--circulation-weight", type=float, default=18.0)
# # #     argument_parser.add_argument(
# # #         "--circulation-activation-distance", type=float, default=1.50
# # #     )
# # #     argument_parser.add_argument(
# # #         "--circulation-activation-sharpness", type=float, default=0.20
# # #     )
# # #     argument_parser.add_argument(
# # #         "--minimum-circulation-progress", type=float, default=0.025
# # #     )
# # #     argument_parser.add_argument(
# # #         "--blocking-alignment-threshold", type=float, default=0.25
# # #     )
# # #     argument_parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
# # #     argument_parser.add_argument("--escape-lateral-target", type=float, default=0.35)
# # #     argument_parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
# # #     argument_parser.add_argument("--maximum-obstacle-depth", type=float, default=5.0)
# # #     argument_parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)

# # #     argument_parser.add_argument("--scene", required=True)
# # #     argument_parser.add_argument("--terrain-obj", default=None)
# # #     argument_parser.add_argument("--heightmap", default=None)
# # #     argument_parser.add_argument(
# # #         "--terrain-height-mode",
# # #         choices=["auto", "heightmap", "obj", "flat"],
# # #         default="auto",
# # #     )
# # #     argument_parser.add_argument("--flat-y", type=float, default=0.0)
# # #     argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
# # #     argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
# # #     argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
# # #     argument_parser.add_argument("--flip-heightmap-x", action="store_true")
# # #     argument_parser.add_argument(
# # #         "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument("--swap-heightmap-xz", action="store_true")
# # #     argument_parser.add_argument("--clearance", type=float, default=1.4)
# # #     argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)
# # #     argument_parser.add_argument(
# # #         "--robot-radius",
# # #         type=float,
# # #         default=0.24,
# # #         help="Planar rover footprint radius used by both guidance and evaluation.",
# # #     )
# # #     argument_parser.add_argument(
# # #         "--evaluation-layout",
# # #         default="default",
# # #         help="Stable layout identifier stored in the rollout archive.",
# # #     )

# # #     argument_parser.add_argument("--height", type=int, default=720)
# # #     argument_parser.add_argument("--width", type=int, default=720)
# # #     argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
# # #     argument_parser.add_argument("--hz", type=float, default=10.0)
# # #     argument_parser.add_argument("--max-steps", type=int, default=300)
# # #     argument_parser.add_argument("--stop-distance", type=float, default=1.0)
# # #     argument_parser.add_argument("--start-x", type=float, default=0.0)
# # #     argument_parser.add_argument("--start-z", type=float, default=8.0)
# # #     argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
# # #     argument_parser.add_argument("--goal-x", type=float, default=None)
# # #     argument_parser.add_argument("--goal-z", type=float, default=None)
# # #     argument_parser.add_argument("--goal-y", type=float, default=None)
# # #     argument_parser.add_argument("--goal-height", type=float, default=1.2)
# # #     argument_parser.add_argument("--goal-radius", type=int, default=18)
# # #     argument_parser.add_argument(
# # #         "--goal-mesh", action=argparse.BooleanOptionalAction, default=False
# # #     )
# # #     argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
# # #     argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)

# # #     argument_parser.add_argument(
# # #         "--obstacle-mode", choices=["none", "depth", "mesh", "ghost"], default="none"
# # #     )
# # #     argument_parser.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
# # #     argument_parser.add_argument("--obstacle-min-y-fraction", type=float, default=0.45)
# # #     argument_parser.add_argument("--ghost-obstacle-x", type=float, default=None)
# # #     argument_parser.add_argument("--ghost-obstacle-z", type=float, default=None)
# # #     argument_parser.add_argument("--ghost-obstacle-y", type=float, default=None)
# # #     argument_parser.add_argument("--ghost-obstacle-height", type=float, default=0.45)
# # #     argument_parser.add_argument("--ghost-obstacle-radius", type=int, default=24)
# # #     argument_parser.add_argument(
# # #         "--obstacle-mesh-uv",
# # #         nargs="+",
# # #         default=[],
# # #         help=(
# # #             "Actual rendered obstacle mesh locations as image fractions u,v. "
# # #             "Example: --obstacle-mesh-uv 0.50,0.72 0.30,0.68"
# # #         ),
# # #     )
# # #     argument_parser.add_argument(
# # #         "--obstacle-world-xz",
# # #         nargs="*",
# # #         default=[],
# # #         metavar="X,Z",
# # #         help=(
# # #             "Static rendered obstacle-box centers in world X,Z coordinates. "
# # #             "Example: --obstacle-world-xz 0,0. Do not combine with "
# # #             "--obstacle-mesh-uv."
# # #         ),
# # #     )
# # #     argument_parser.add_argument(
# # #         "--obstacle-world-xz-item",
# # #         action="append",
# # #         default=[],
# # #         metavar="X,Z",
# # #         help=(
# # #             "Repeatable form that safely accepts negative coordinates, e.g. "
# # #             "--obstacle-world-xz-item=-3,0."
# # #         ),
# # #     )
# # #     argument_parser.add_argument(
# # #         "--world-obstacle-half-extent", type=float, default=0.75
# # #     )
# # #     argument_parser.add_argument("--world-obstacle-height", type=float, default=1.40)
# # #     argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
# # #     argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)
# # #     argument_parser.add_argument(
# # #         "--obstacle-velocity-xz",
# # #         nargs="*",
# # #         default=[],
# # #         metavar="VX,VZ",
# # #         help=(
# # #             "World-frame mesh velocities in m/s. Supply one value to broadcast "
# # #             "or one value per obstacle. Example: --obstacle-velocity-xz 0.30,0.0"
# # #         ),
# # #     )

# # #     argument_parser.add_argument("--lookahead-index", type=int, default=4)
# # #     argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
# # #     argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
# # #     argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
# # #     argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
# # #     argument_parser.add_argument("--save-every", type=int, default=1)
# # #     argument_parser.add_argument(
# # #         "--save-frames", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument(
# # #         "--save-video", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     argument_parser.add_argument(
# # #         "--archive-observations",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=True,
# # #         help="Store RGB/depth/masks in rollout.npz; disable for large evaluations.",
# # #     )
# # #     argument_parser.add_argument(
# # #         "--overlay-masks", action=argparse.BooleanOptionalAction, default=True
# # #     )
# # #     return argument_parser


# # # def main() -> None:
# # #     args = parser().parse_args()
# # #     if args.obstacle_world_xz_item:
# # #         args.obstacle_world_xz.extend(args.obstacle_world_xz_item)
# # #     np.random.seed(args.seed)
# # #     if args.goal_x is None or args.goal_z is None:
# # #         raise ValueError("fixed PointGoal requires --goal-x and --goal-z")
# # #     if args.belief_pixel_goal and args.goal_mode != "pixel":
# # #         raise ValueError("--belief-pixel-goal requires --goal-mode pixel")
# # #     if args.belief_pixel_goal and not args.goal_mesh:
# # #         raise ValueError(
# # #             "simulation belief tracking requires --goal-mesh so a live semantic "
# # #             "goal observation exists"
# # #         )
# # #     if args.belief_minimum_goal_pixels < 1:
# # #         raise ValueError("belief-minimum-goal-pixels must be positive")
# # #     return_home_enabled = args.interactive_return_home or args.qwen_freeform_mission
# # #     if return_home_enabled and not args.belief_pixel_goal:
# # #         raise ValueError("return-home modes require --belief-pixel-goal")
# # #     if return_home_enabled and not args.qwen_homotopy:
# # #         raise ValueError("return-home modes require --qwen-homotopy")
# # #     if args.mission_command is not None and not args.qwen_freeform_mission:
# # #         raise ValueError("--mission-command requires --qwen-freeform-mission")
# # #     if args.return_goal_obstacle_activation_distance <= 0.0:
# # #         raise ValueError("return goal obstacle activation distance must be positive")
# # #     if args.return_goal_obstacle_dilation_pixels < 0:
# # #         raise ValueError("return goal obstacle dilation pixels must be non-negative")
# # #     if (
# # #         min(
# # #             args.belief_measurement_std,
# # #             args.belief_translation_process_std,
# # #             args.belief_yaw_process_std_deg,
# # #             args.belief_bootstrap_std,
# # #         )
# # #         < 0.0
# # #     ):
# # #         raise ValueError("belief uncertainty parameters must be non-negative")
# # #     if args.robot_radius < 0.0:
# # #         raise ValueError("robot-radius must be non-negative")
# # #     if args.obstacle_velocity_xz and args.obstacle_mode != "mesh":
# # #         raise ValueError("moving obstacle velocities require --obstacle-mode mesh")
# # #     if args.obstacle_mode == "ghost" and (
# # #         args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
# # #     ):
# # #         raise ValueError(
# # #             "ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z"
# # #         )
# # #     if args.obstacle_mesh_uv and args.obstacle_world_xz:
# # #         raise ValueError(
# # #             "choose either --obstacle-mesh-uv or --obstacle-world-xz, not both"
# # #         )
# # #     if args.obstacle_mode == "mesh" and not (
# # #         args.obstacle_mesh_uv or args.obstacle_world_xz
# # #     ):
# # #         raise ValueError(
# # #             "mesh mode requires --obstacle-world-xz X,Z [X,Z ...] or "
# # #             "--obstacle-mesh-uv u,v [u,v ...]"
# # #         )
# # #     if args.world_obstacle_half_extent <= 0.0 or args.world_obstacle_height <= 0.0:
# # #         raise ValueError("world obstacle dimensions must be positive")
# # #     if args.goal_mesh_half_extent <= 0.0 or args.goal_mesh_height <= 0.0:
# # #         raise ValueError("goal mesh dimensions must be positive")

# # #     if args.qwen_homotopy and args.planner_mode == "pure-navdp":
# # #         raise ValueError("Qwen homotopy conditioning requires s2diff or gradient mode")

# # #     qwen_process: Optional[subprocess.Popen[Any]] = None
# # #     server_process: Optional[subprocess.Popen[Any]] = None
# # #     simulator = None
# # #     try:
# # #         qwen_process = start_qwen_homotopy_server(args)
# # #         homotopy_selector = None
# # #         if args.qwen_homotopy:
# # #             homotopy_selector = QwenHomotopyClient(
# # #                 f"http://{args.qwen_homotopy_host}:{args.qwen_homotopy_port}",
# # #                 timeout=args.qwen_homotopy_timeout,
# # #             )
# # #             homotopy_selector.reset()
# # #         server_process = start_server(args)
# # #         server_url = f"http://{args.server_host}:{args.server_port}"
# # #         client = NavDPS2DiffClient(server_url)
# # #         algorithm = client.reset(
# # #             camera_intrinsic(args.height, args.width, args.hfov_deg),
# # #             batch_size=1,
# # #             stop_threshold=-3.0,
# # #         )
# # #         supported_algorithms = {
# # #             "navdp-s2diff-pixels",
# # #             "navdp-hlc-s2diff",
# # #             "navdp-hlc-s2diff-no-critic",
# # #             "navdp-hlc-gradient",
# # #             "navdp-hlc-gradient-no-critic",
# # #             "navdp-pure-critic",
# # #         }
# # #         if algorithm not in supported_algorithms:
# # #             raise RuntimeError(f"unexpected planner response: {algorithm!r}")

# # #         terrain = TerrainHeight(
# # #             mode=args.terrain_height_mode,
# # #             heightmap=(
# # #                 Path(args.heightmap).expanduser().resolve() if args.heightmap else None
# # #             ),
# # #             obj=(
# # #                 Path(args.terrain_obj).expanduser().resolve()
# # #                 if args.terrain_obj
# # #                 else None
# # #             ),
# # #             flat_y=args.flat_y,
# # #             size_x=args.size_x,
# # #             size_z=args.size_z,
# # #             size_y=args.size_y,
# # #             flip_x=args.flip_heightmap_x,
# # #             flip_z=args.flip_heightmap_z,
# # #             swap_xz=args.swap_heightmap_xz,
# # #         )
# # #         output_directory = Path(args.output).expanduser().resolve()
# # #         frame_directory = output_directory / "frames"
# # #         frame_directory.mkdir(parents=True, exist_ok=True)

# # #         simulator = make_simulator(
# # #             Path(args.scene),
# # #             args.height,
# # #             args.width,
# # #             args.hfov_deg,
# # #             with_semantic=args.obstacle_mode == "mesh" or args.goal_mesh,
# # #         )
# # #         agent = simulator.initialize_agent(0)
# # #         intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
# # #         goal_belief = (
# # #             GaussianGoalBelief(
# # #                 intrinsic,
# # #                 (args.height, args.width),
# # #                 minimum_visible_pixels=args.belief_minimum_goal_pixels,
# # #                 measurement_std=args.belief_measurement_std,
# # #                 translation_process_std=args.belief_translation_process_std,
# # #                 yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
# # #             )
# # #             if args.belief_pixel_goal
# # #             else None
# # #         )
# # #         previous_executed_action = np.zeros(3, dtype=np.float32)
# # #         home_belief = (
# # #             GaussianGoalBelief(
# # #                 intrinsic,
# # #                 (args.height, args.width),
# # #                 minimum_visible_pixels=args.belief_minimum_goal_pixels,
# # #                 measurement_std=args.belief_measurement_std,
# # #                 translation_process_std=args.belief_translation_process_std,
# # #                 yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
# # #             )
# # #             if return_home_enabled
# # #             else None
# # #         )
# # #         if home_belief is not None:
# # #             # Home starts at the rover origin. Every executed action propagates
# # #             # this stationary world location into the current body frame.
# # #             home_belief.initialize(
# # #                 np.zeros(2, dtype=np.float32),
# # #                 args.belief_measurement_std,
# # #                 visible=False,
# # #             )
# # #         mission_phase = "OUTBOUND"
# # #         return_goal_obstacle_active = False
# # #         return_command_event: Optional[dict[str, Any]] = None
# # #         mission_plan_event: Optional[dict[str, Any]] = None
# # #         mission_plan_pending = bool(args.qwen_freeform_mission)
# # #         automatic_return_requested = False
# # #         roundtrip_completed = False
# # #         x, z = float(args.start_x), float(args.start_z)
# # #         yaw = math.radians(float(args.start_yaw_deg))
# # #         dt = 1.0 / float(args.hz)

# # #         goal_y = args.goal_y
# # #         if goal_y is None:
# # #             goal_y = (
# # #                 terrain.local_height_max(args.goal_x, args.goal_z, 0.8)
# # #                 + args.goal_height
# # #             )
# # #         goal = np.asarray([args.goal_x, goal_y, args.goal_z], dtype=np.float32)
# # #         start_position_xz = np.asarray([x, z], dtype=np.float64)
# # #         initial_goal_distance = float(
# # #             np.linalg.norm(goal[[0, 2]].astype(np.float64) - start_position_xz)
# # #         )
# # #         goal_mesh_object = None
# # #         if args.goal_mesh:
# # #             goal_mesh_object = place_world_goal_mesh(
# # #                 simulator,
# # #                 terrain,
# # #                 args.goal_x,
# # #                 args.goal_z,
# # #                 output_directory,
# # #                 half_extent=args.goal_mesh_half_extent,
# # #                 height=args.goal_mesh_height,
# # #             )

# # #         ghost = None
# # #         if args.obstacle_mode == "ghost":
# # #             ghost_y = args.ghost_obstacle_y
# # #             if ghost_y is None:
# # #                 ghost_y = (
# # #                     terrain.local_height_max(
# # #                         args.ghost_obstacle_x,
# # #                         args.ghost_obstacle_z,
# # #                         args.pose_terrain_radius,
# # #                     )
# # #                     + args.ghost_obstacle_height
# # #                 )
# # #             ghost = np.asarray(
# # #                 [args.ghost_obstacle_x, ghost_y, args.ghost_obstacle_z],
# # #                 dtype=np.float32,
# # #             )

# # #         mesh_objects: list[Any] = []
# # #         mesh_centroids: list[np.ndarray] = []
# # #         mesh_current_centroids: list[np.ndarray] = []
# # #         mesh_base_geometries: list[np.ndarray] = []
# # #         mesh_geometries: list[np.ndarray] = []
# # #         mesh_velocities = np.zeros((0, 2), dtype=np.float64)
# # #         mesh_placed = False
# # #         if args.obstacle_mode == "mesh" and args.obstacle_world_xz:
# # #             mesh_objects, mesh_centroids, mesh_base_geometries = (
# # #                 place_world_obstacle_meshes(
# # #                     simulator,
# # #                     terrain,
# # #                     args.obstacle_world_xz,
# # #                     output_directory,
# # #                     half_extent=args.world_obstacle_half_extent,
# # #                     height=args.world_obstacle_height,
# # #                 )
# # #             )
# # #             mesh_velocities = expand_obstacle_velocities(
# # #                 args.obstacle_velocity_xz, len(mesh_objects)
# # #             )
# # #             mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# # #                 mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
# # #             )
# # #             mesh_placed = True

# # #         row_keys = [
# # #             "pose",
# # #             "action_3d",
# # #             "mission_phase",
# # #             "return_goal_obstacle_active",
# # #             "point_goal",
# # #             "belief_goal_mu",
# # #             "belief_goal_covariance",
# # #             "belief_goal_pixel",
# # #             "belief_goal_visible",
# # #             "belief_goal_source",
# # #             "belief_goal_time_since_seen",
# # #             "belief_goal_bearing_rad",
# # #             "belief_goal_pixel_sigma",
# # #             "belief_heading_recovery_active",
# # #             "selected_trajectory",
# # #             "all_trajectories",
# # #             "all_values",
# # #             "selected_index",
# # #             "fallback_stop",
# # #             "escape_turn",
# # #             "valid_obstacle_points",
# # #             "selected_circulation_sign",
# # #             "candidate_circulation_signs",
# # #             "selected_barrier_energy",
# # #             "selected_circulation_energy",
# # #             "planning_time_seconds",
# # #             "selected_minimum_clearance",
# # #             "mean_guidance_noise_correction",
# # #             "final_guidance_noise_correction",
# # #             "maximum_guidance_noise_correction",
# # #             "mean_final_effective_sample_size",
# # #             "goal_distance",
# # #             "executed_center_clearance",
# # #             "executed_surface_clearance",
# # #             "geometric_collision",
# # #             "obstacle_positions_world",
# # #             "qwen_homotopy_sign",
# # #             "qwen_homotopy_side",
# # #             "qwen_homotopy_confidence",
# # #             "qwen_homotopy_queried",
# # #         ]
# # #         if args.archive_observations:
# # #             row_keys.extend(
# # #                 (
# # #                     "rgb",
# # #                     "depth",
# # #                     "goal_mask",
# # #                     "live_goal_mask",
# # #                     "ghost_goal_mask",
# # #                     "obstacle_mask",
# # #                 )
# # #             )
# # #         rows: dict[str, list[Any]] = {key: [] for key in row_keys}
# # #         video_frames: list[Image.Image] = []
# # #         success = False
# # #         homotopy_events: list[dict[str, Any]] = []

# # #         for step in range(int(args.max_steps)):
# # #             y = (
# # #                 terrain.local_height_max(x, z, args.pose_terrain_radius)
# # #                 + args.clearance
# # #             )
# # #             position = np.asarray([x, y, z], dtype=np.float32)
# # #             if goal_belief is not None and step > 0:
# # #                 goal_belief.predict(previous_executed_action, dt)
# # #             if home_belief is not None and step > 0:
# # #                 home_belief.predict(previous_executed_action, dt)
# # #             set_agent_pose(agent, position, yaw)
# # #             if mesh_placed:
# # #                 elapsed_seconds = step * dt
# # #                 move_mesh_objects(mesh_objects, mesh_velocities, elapsed_seconds)
# # #                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# # #                     mesh_base_geometries,
# # #                     mesh_centroids,
# # #                     mesh_velocities,
# # #                     elapsed_seconds,
# # #                 )
# # #             observation = simulator.get_sensor_observations()
# # #             rgb, depth = rgb_depth(observation)

# # #             if args.obstacle_mode == "mesh" and not mesh_placed:
# # #                 mesh_objects, mesh_centroids, mesh_base_geometries = (
# # #                     place_obstacle_meshes(
# # #                         simulator,
# # #                         depth,
# # #                         position,
# # #                         yaw,
# # #                         intrinsic,
# # #                         args.obstacle_mesh_uv,
# # #                         output_directory,
# # #                         mesh_half_pixels=args.mesh_half_pixels,
# # #                         mesh_lift=args.mesh_obstacle_lift,
# # #                     )
# # #                 )
# # #                 mesh_velocities = expand_obstacle_velocities(
# # #                     args.obstacle_velocity_xz, len(mesh_objects)
# # #                 )
# # #                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# # #                     mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
# # #                 )
# # #                 mesh_placed = True
# # #                 observation = simulator.get_sensor_observations()
# # #                 rgb, depth = rgb_depth(observation)

# # #             if mission_plan_pending:
# # #                 mission_command = args.mission_command
# # #                 if mission_command is None:
# # #                     print(
# # #                         "\nWhat should the rover do? You may use a vague command, "
# # #                         "for example: 'visit the goal and come back'.",
# # #                         flush=True,
# # #                     )
# # #                     try:
# # #                         mission_command = input("> ").strip()
# # #                     except EOFError as error:
# # #                         raise RuntimeError(
# # #                             "no startup command was available; set "
# # #                             "--mission-command for a non-interactive run"
# # #                         ) from error
# # #                 assert homotopy_selector is not None
# # #                 mission_decision = homotopy_selector.classify_mission(
# # #                     rgb, mission_command
# # #                 )
# # #                 automatic_return_requested = mission_decision.plan == (
# # #                     "GO_TO_GOAL",
# # #                     "RETURN_HOME",
# # #                 )
# # #                 mission_plan_event = {
# # #                     "step": step,
# # #                     "user_command": mission_command,
# # #                     "plan": list(mission_decision.plan),
# # #                     "confidence": mission_decision.confidence,
# # #                     "raw_response": mission_decision.raw_response,
# # #                 }
# # #                 Image.fromarray(rgb).save(output_directory / "qwen_mission_frame.png")
# # #                 print(
# # #                     f"[qwen-mission] text={mission_command!r} "
# # #                     f"plan={list(mission_decision.plan)} "
# # #                     f"confidence={mission_decision.confidence:.2f}",
# # #                     flush=True,
# # #                 )
# # #                 mission_plan_pending = False

# # #             semantic = (
# # #                 semantic_from_observation(observation)
# # #                 if args.obstacle_mode == "mesh" or args.goal_mesh
# # #                 else None
# # #             )
# # #             goal_right, _goal_up, goal_forward = camera_coordinates(goal, position, yaw)
# # #             point_goal = np.asarray(
# # #                 [max(goal_forward, 0.0), -goal_right], dtype=np.float32
# # #             )
# # #             live_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
# # #             ghost_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
# # #             belief_goal_visible = False
# # #             belief_goal_source = "DISABLED"
# # #             belief_goal_mu = np.full(2, np.nan, dtype=np.float32)
# # #             belief_goal_covariance = np.full((2, 2), np.nan, dtype=np.float32)
# # #             belief_goal_pixel = np.full(2, -1, dtype=np.int32)
# # #             belief_goal_time_since_seen = float("nan")
# # #             belief_goal_bearing = float("nan")
# # #             belief_goal_pixel_sigma = float("nan")

# # #             if goal_belief is not None:
# # #                 assert semantic is not None
# # #                 live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
# # #                 bootstrapped = False
# # #                 if mission_phase == "OUTBOUND":
# # #                     active_goal_belief = goal_belief
# # #                     belief_goal_visible = goal_belief.observe(live_goal_mask, depth)
# # #                     if not goal_belief.initialized:
# # #                         if not args.belief_bootstrap_world_goal:
# # #                             raise RuntimeError(
# # #                                 "goal belief is uninitialized because the live goal "
# # #                                 "mask has not been observed; start with the goal visible "
# # #                                 "or pass --belief-bootstrap-world-goal for simulation"
# # #                             )
# # #                         goal_belief.initialize(
# # #                             np.asarray([goal_forward, -goal_right], dtype=np.float32),
# # #                             args.belief_bootstrap_std,
# # #                         )
# # #                         bootstrapped = True
# # #                 else:
# # #                     assert home_belief is not None and home_belief.initialized
# # #                     active_goal_belief = home_belief
# # #                     belief_goal_visible = False

# # #                 belief_projection = active_goal_belief.project(
# # #                     base_radius=args.belief_ghost_base_radius,
# # #                     covariance_scale=args.belief_ghost_covariance_scale,
# # #                     maximum_radius=args.belief_ghost_maximum_radius,
# # #                 )
# # #                 planner_goal = belief_projection.pixel_uv
# # #                 ghost_goal_mask = belief_projection.mask
# # #                 if mission_phase == "OUTBOUND" and belief_goal_visible:
# # #                     goal_mask = live_goal_mask
# # #                     belief_goal_source = "LIVE"
# # #                 else:
# # #                     goal_mask = ghost_goal_mask
# # #                     belief_goal_source = (
# # #                         "HOME_BELIEF"
# # #                         if mission_phase == "RETURN_HOME"
# # #                         else ("WORLD_BOOTSTRAP" if bootstrapped else "GHOST")
# # #                     )
# # #                 assert (
# # #                     active_goal_belief.mu is not None
# # #                     and active_goal_belief.Sigma is not None
# # #                 )
# # #                 belief_goal_mu = active_goal_belief.mu.copy()
# # #                 belief_goal_covariance = active_goal_belief.Sigma.copy()
# # #                 belief_goal_pixel = belief_projection.pixel_uv.copy()
# # #                 belief_goal_time_since_seen = active_goal_belief.time_since_seen
# # #                 belief_goal_bearing = belief_projection.bearing_rad
# # #                 belief_goal_pixel_sigma = belief_projection.pixel_sigma
# # #             else:
# # #                 if args.goal_mesh:
# # #                     assert semantic is not None
# # #                     live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
# # #                     goal_mask = live_goal_mask
# # #                     if not np.any(goal_mask):
# # #                         goal_mask, _ = project_world_mask(
# # #                             goal,
# # #                             position,
# # #                             yaw,
# # #                             intrinsic,
# # #                             args.height,
# # #                             args.width,
# # #                             args.goal_radius,
# # #                         )
# # #                 else:
# # #                     goal_mask, _ = project_world_mask(
# # #                         goal,
# # #                         position,
# # #                         yaw,
# # #                         intrinsic,
# # #                         args.height,
# # #                         args.width,
# # #                         args.goal_radius,
# # #                     )
# # #                 planner_goal = point_goal
# # #             if args.goal_mode == "pixel" and goal_belief is None:
# # #                 planner_goal = world_goal_to_pixel(
# # #                     goal, position, yaw, intrinsic, args.height, args.width
# # #                 )
# # #                 goal_mask = circle_mask(
# # #                     args.height,
# # #                     args.width,
# # #                     planner_goal[0],
# # #                     planner_goal[1],
# # #                     args.goal_radius,
# # #                 )
# # #             guidance_depth = depth.copy()
# # #             if args.obstacle_mode == "depth":
# # #                 obstacle_mask = depth_obstacle_mask(
# # #                     depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
# # #                 )
# # #             elif args.obstacle_mode == "mesh":
# # #                 assert semantic is not None
# # #                 semantic_ids = list(
# # #                     range(
# # #                         MESH_OBSTACLE_ID,
# # #                         MESH_OBSTACLE_ID + len(mesh_objects),
# # #                     )
# # #                 )
# # #                 obstacle_mask = np.isin(semantic, semantic_ids).astype(np.uint8)
# # #                 # The depth image was re-rendered after mesh placement, so
# # #                 # guidance_depth already contains the real obstacle depth.
# # #             elif args.obstacle_mode == "ghost":
# # #                 assert ghost is not None
# # #                 obstacle_mask, obstacle_forward = project_world_mask(
# # #                     ghost,
# # #                     position,
# # #                     yaw,
# # #                     intrinsic,
# # #                     args.height,
# # #                     args.width,
# # #                     args.ghost_obstacle_radius,
# # #                 )
# # #                 if obstacle_forward > 0.05:
# # #                     guidance_depth[obstacle_mask > 0] = obstacle_forward
# # #             else:
# # #                 obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)

# # #             # Replace this mask-to-pixels line with your own detector's [u,v]
# # #             if mission_phase == "RETURN_HOME":
# # #                 distance_from_reached_goal = float(
# # #                     np.linalg.norm(goal[[0, 2]] - position[[0, 2]])
# # #                 )
# # #                 if (
# # #                     not return_goal_obstacle_active
# # #                     and distance_from_reached_goal
# # #                     >= args.return_goal_obstacle_activation_distance
# # #                 ):
# # #                     return_goal_obstacle_active = True
# # #                     print(
# # #                         "[roundtrip] reached-goal keep-out is now active",
# # #                         flush=True,
# # #                     )
# # #                 if return_goal_obstacle_active and np.any(live_goal_mask):
# # #                     reached_goal_keepout = dilate_binary_mask(
# # #                         live_goal_mask,
# # #                         args.return_goal_obstacle_dilation_pixels,
# # #                     )
# # #                     obstacle_mask = (
# # #                         (obstacle_mask > 0) | (reached_goal_keepout > 0)
# # #                     ).astype(np.uint8)
# # #                     target_depths = depth[
# # #                         (live_goal_mask > 0)
# # #                         & np.isfinite(depth)
# # #                         & (depth > args.minimum_obstacle_depth)
# # #                     ]
# # #                     if target_depths.size:
# # #                         guidance_depth[reached_goal_keepout > 0] = float(
# # #                             np.median(target_depths)
# # #                         )

# # #             # array if obstacle pixels already come directly from your system.
# # #             obstacle_pixels = pixels_from_mask(
# # #                 obstacle_mask, args.maximum_obstacle_pixels
# # #             )
# # #             homotopy_decision = None
# # #             forced_circulation_sign = 0.0
# # #             obstacle_relevant_for_homotopy = False
# # #             if homotopy_selector is not None:
# # #                 homotopy_obstacle_mask = (
# # #                     (obstacle_mask > 0)
# # #                     & np.isfinite(guidance_depth)
# # #                     & (guidance_depth >= args.minimum_obstacle_depth)
# # #                     & (guidance_depth <= args.maximum_obstacle_depth)
# # #                 ).astype(np.uint8)
# # #                 qwen_overlay = overlay_frame(
# # #                     rgb,
# # #                     goal_mask,
# # #                     homotopy_obstacle_mask,
# # #                     "Qwen homotopy: choose LEFT or RIGHT",
# # #                     show_masks=True,
# # #                 )
# # #                 homotopy_decision = homotopy_selector.step(
# # #                     np.asarray(qwen_overlay.convert("RGB")), homotopy_obstacle_mask
# # #                 )
# # #                 obstacle_relevant_for_homotopy = homotopy_decision.obstacle_relevant
# # #                 forced_circulation_sign = homotopy_decision.circulation_sign
# # #                 if homotopy_decision.queried_qwen:
# # #                     event = {
# # #                         "step": step,
# # #                         "side": homotopy_decision.side,
# # #                         "circulation_sign": forced_circulation_sign,
# # #                         "confidence": homotopy_decision.confidence,
# # #                         "repeat_sides": list(homotopy_decision.repeated_sides),
# # #                         "repeat_confidences": list(
# # #                             homotopy_decision.repeated_confidences
# # #                         ),
# # #                         "consistency_rate": homotopy_decision.consistency_rate,
# # #                         "used_fallback": homotopy_decision.used_fallback,
# # #                         "raw_response": homotopy_decision.raw_response,
# # #                     }
# # #                     homotopy_events.append(event)
# # #                     query_directory = output_directory / "qwen_homotopy_queries"
# # #                     query_directory.mkdir(parents=True, exist_ok=True)
# # #                     qwen_overlay.save(query_directory / f"query_step_{step:04d}.png")
# # #                     print(
# # #                         f"[qwen-homotopy] side={homotopy_decision.side} "
# # #                         f"sign={forced_circulation_sign:+.0f} "
# # #                         f"confidence={homotopy_decision.confidence:.2f} "
# # #                         f"consistency={homotopy_decision.consistency_rate:.2%} "
# # #                         f"repeats={list(homotopy_decision.repeated_sides)} "
# # #                         f"fallback={homotopy_decision.used_fallback}",
# # #                         flush=True,
# # #                     )
# # #             planning_start = time.perf_counter()
# # #             result = client.plan(
# # #                 goal_xy=planner_goal,
# # #                 rgb=rgb,
# # #                 depth=guidance_depth,
# # #                 obstacle_pixels=obstacle_pixels,
# # #                 goal_mode=args.goal_mode,
# # #                 forced_circulation_sign=forced_circulation_sign,
# # #             )
# # #             planning_time = time.perf_counter() - planning_start
# # #             action = (
# # #                 np.zeros(3, dtype=np.float32)
# # #                 if result.fallback_stop
# # #                 else waypoint_action(
# # #                     result.trajectory,
# # #                     lookahead_index=args.lookahead_index,
# # #                     maximum_forward_speed=args.maximum_forward_speed,
# # #                     maximum_yaw_rate=args.maximum_yaw_rate,
# # #                     yaw_gain=args.yaw_gain,
# # #                 )
# # #             )
# # #             action, belief_recovery_active = belief_heading_recovery_action(
# # #                 action,
# # #                 belief_bearing=belief_goal_bearing,
# # #                 obstacle_relevant=obstacle_relevant_for_homotopy,
# # #                 enabled=args.belief_pixel_goal and args.belief_heading_recovery,
# # #                 activation_bearing=math.radians(args.belief_recovery_bearing_deg),
# # #                 yaw_gain=args.belief_recovery_yaw_gain,
# # #                 maximum_yaw_rate=args.belief_recovery_maximum_yaw_rate,
# # #                 maximum_forward_speed=args.belief_recovery_maximum_forward_speed,
# # #             )

# # #             next_position, next_yaw = integrate_mars(position, yaw, action, dt)
# # #             previous_executed_action = action.copy()
# # #             x = float(
# # #                 np.clip(
# # #                     next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
# # #                 )
# # #             )
# # #             z = float(
# # #                 np.clip(
# # #                     next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
# # #                 )
# # #             )
# # #             rows["mission_phase"].append(mission_phase)
# # #             rows["return_goal_obstacle_active"].append(return_goal_obstacle_active)

# # #             yaw = wrap_angle(next_yaw)
# # #             outbound_goal_distance = float(
# # #                 np.linalg.norm(goal[[0, 2]] - np.asarray([x, z]))
# # #             )
# # #             home_distance = float(
# # #                 np.linalg.norm(start_position_xz - np.asarray([x, z]))
# # #             )
# # #             goal_distance = (
# # #                 home_distance
# # #                 if mission_phase == "RETURN_HOME"
# # #                 else outbound_goal_distance
# # #             )
# # #             center_clearance = planar_mesh_clearance(
# # #                 np.asarray([x, z], dtype=np.float64), mesh_geometries
# # #             )
# # #             if np.isfinite(center_clearance):
# # #                 surface_clearance = max(
# # #                     center_clearance - float(args.robot_radius), 0.0
# # #                 )
# # #                 geometric_collision = center_clearance <= float(args.robot_radius)
# # #             else:
# # #                 surface_clearance = float("nan")
# # #                 geometric_collision = False
# # #             rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
# # #             pose = np.asarray(
# # #                 [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w],
# # #                 dtype=np.float32,
# # #             )

# # #             if args.archive_observations:
# # #                 rows["rgb"].append(rgb)
# # #                 rows["depth"].append(depth)
# # #                 rows["goal_mask"].append(goal_mask)
# # #                 rows["live_goal_mask"].append(live_goal_mask)
# # #                 rows["ghost_goal_mask"].append(ghost_goal_mask)
# # #                 rows["obstacle_mask"].append(obstacle_mask)
# # #             rows["pose"].append(pose)
# # #             rows["action_3d"].append(action)
# # #             rows["point_goal"].append(planner_goal)
# # #             rows["belief_goal_mu"].append(belief_goal_mu)
# # #             rows["belief_goal_covariance"].append(belief_goal_covariance)
# # #             rows["belief_goal_pixel"].append(belief_goal_pixel)
# # #             rows["belief_goal_visible"].append(belief_goal_visible)
# # #             rows["belief_goal_source"].append(belief_goal_source)
# # #             rows["belief_goal_time_since_seen"].append(belief_goal_time_since_seen)
# # #             rows["belief_goal_bearing_rad"].append(belief_goal_bearing)
# # #             rows["belief_goal_pixel_sigma"].append(belief_goal_pixel_sigma)
# # #             rows["belief_heading_recovery_active"].append(belief_recovery_active)
# # #             rows["selected_trajectory"].append(result.trajectory)
# # #             rows["all_trajectories"].append(result.all_trajectories)
# # #             rows["all_values"].append(result.all_values)
# # #             rows["selected_index"].append(result.selected_index)
# # #             rows["fallback_stop"].append(result.fallback_stop)
# # #             rows["escape_turn"].append(result.escape_turn)
# # #             rows["valid_obstacle_points"].append(result.valid_obstacle_points)
# # #             rows["selected_circulation_sign"].append(result.selected_circulation_sign)
# # #             rows["candidate_circulation_signs"].append(
# # #                 result.candidate_circulation_signs
# # #             )
# # #             rows["selected_barrier_energy"].append(result.selected_barrier_energy)
# # #             rows["selected_circulation_energy"].append(
# # #                 result.selected_circulation_energy
# # #             )
# # #             rows["planning_time_seconds"].append(planning_time)
# # #             rows["selected_minimum_clearance"].append(result.selected_minimum_clearance)
# # #             rows["mean_guidance_noise_correction"].append(
# # #                 result.mean_guidance_noise_correction
# # #             )
# # #             rows["final_guidance_noise_correction"].append(
# # #                 result.final_guidance_noise_correction
# # #             )
# # #             rows["maximum_guidance_noise_correction"].append(
# # #                 result.maximum_guidance_noise_correction
# # #             )
# # #             rows["mean_final_effective_sample_size"].append(
# # #                 result.mean_final_effective_sample_size
# # #             )
# # #             rows["goal_distance"].append(goal_distance)
# # #             rows["executed_center_clearance"].append(center_clearance)
# # #             rows["executed_surface_clearance"].append(surface_clearance)
# # #             rows["geometric_collision"].append(geometric_collision)
# # #             rows["obstacle_positions_world"].append(
# # #                 np.stack(mesh_current_centroids)
# # #                 if mesh_current_centroids
# # #                 else np.zeros((0, 3), dtype=np.float64)
# # #             )

# # #             rows["qwen_homotopy_sign"].append(forced_circulation_sign)
# # #             rows["qwen_homotopy_side"].append(
# # #                 homotopy_decision.side if homotopy_decision is not None else "AUTO"
# # #             )
# # #             rows["qwen_homotopy_confidence"].append(
# # #                 homotopy_decision.confidence if homotopy_decision is not None else 0.0
# # #             )
# # #             rows["qwen_homotopy_queried"].append(
# # #                 homotopy_decision.queried_qwen
# # #                 if homotopy_decision is not None
# # #                 else False
# # #             )

# # #             if args.save_frames and step % max(int(args.save_every), 1) == 0:

# # #                 side_label = (
# # #                     homotopy_decision.side if homotopy_decision is not None else "AUTO"
# # #                 )
# # #                 label = (
# # #                     f"t={step} phase={mission_phase} goal={goal_distance:.2f}m "
# # #                     f"qwen_side={side_label} pixels={len(obstacle_pixels)} "
# # #                     f"goal_src={belief_goal_source} "
# # #                     f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
# # #                     f"recover={int(belief_recovery_active)} "
# # #                     f"pred={result.selected_minimum_clearance:.2f}m "
# # #                     f"actual={surface_clearance:.2f}m "
# # #                     f"mode={result.selected_circulation_sign:+.0f} "
# # #                     f"escape={int(result.escape_turn)} "
# # #                     f"guide_rms={result.mean_guidance_noise_correction:.4f} "
# # #                     f"v={action[0]:.2f} w={action[2]:.2f}"
# # #                 )
# # #                 frame = overlay_frame(
# # #                     rgb,
# # #                     goal_mask,
# # #                     obstacle_mask,
# # #                     label,
# # #                     show_masks=args.overlay_masks,
# # #                 )
# # #                 frame.save(frame_directory / f"frame_{step:04d}.png")
# # #                 video_frames.append(frame)

# # #             print(
# # #                 f"step={step:04d} phase={mission_phase} goal={goal_distance:.2f}m "
# # #                 f"qwen_side={homotopy_decision.side if homotopy_decision else 'AUTO'} "
# # #                 f"goal_src={belief_goal_source} "
# # #                 f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
# # #                 f"recover={int(belief_recovery_active)} "
# # #                 f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
# # #                 f"selected={result.selected_index} fallback={result.fallback_stop} "
# # #                 f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
# # #                 f"pred_clear={result.selected_minimum_clearance:.3f}m "
# # #                 f"actual_clear={surface_clearance:.3f}m "
# # #                 f"collision={geometric_collision} "
# # #                 f"barrier={result.selected_barrier_energy:.5f} "
# # #                 f"circ={result.selected_circulation_energy:.5f} "
# # #                 f"latency={planning_time * 1000.0:.1f}ms "
# # #                 f"guide_rms={result.mean_guidance_noise_correction:.6f} "
# # #                 f"ess={result.mean_final_effective_sample_size:.2f} "
# # #                 f"action={action.tolist()}",
# # #                 flush=True,
# # #             )
# # #             if goal_distance <= args.stop_distance:
# # #                 if args.qwen_freeform_mission and mission_phase == "OUTBOUND":
# # #                     if automatic_return_requested:
# # #                         print(
# # #                             "[mission] outward goal reached; advancing automatically "
# # #                             "to RETURN_HOME",
# # #                             flush=True,
# # #                         )
# # #                         mission_phase = "RETURN_HOME"
# # #                         assert homotopy_selector is not None
# # #                         homotopy_selector.reset()
# # #                         continue
# # #                     success = True
# # #                     print(
# # #                         "[mission] outward goal reached; GO_TO_GOAL plan complete",
# # #                         flush=True,
# # #                     )
# # #                     break

# # #                 if args.interactive_return_home and mission_phase == "OUTBOUND":
# # #                     user_command = args.return_command
# # #                     if user_command is None:
# # #                         print(
# # #                             "\nOutward goal reached. What should the rover do? "
# # #                             "(for example: come back / stop)",
# # #                             flush=True,
# # #                         )
# # #                         try:
# # #                             user_command = input("> ").strip()
# # #                         except EOFError as error:
# # #                             raise RuntimeError(
# # #                                 "no interactive command was available; set "
# # #                                 "--return-command 'come back' for a non-interactive run"
# # #                             ) from error
# # #                     assert homotopy_selector is not None
# # #                     command_overlay = overlay_frame(
# # #                         rgb,
# # #                         goal_mask,
# # #                         obstacle_mask,
# # #                         "Qwen command: RETURN or STOP",
# # #                         show_masks=True,
# # #                     )
# # #                     command_decision = homotopy_selector.classify_command(
# # #                         np.asarray(command_overlay.convert("RGB")),
# # #                         user_command,
# # #                     )
# # #                     return_command_event = {
# # #                         "step": step,
# # #                         "user_command": user_command,
# # #                         "command": command_decision.command,
# # #                         "confidence": command_decision.confidence,
# # #                         "raw_response": command_decision.raw_response,
# # #                     }
# # #                     command_overlay.save(output_directory / "qwen_return_command.png")
# # #                     print(
# # #                         f"[qwen-command] text={user_command!r} "
# # #                         f"decision={command_decision.command} "
# # #                         f"confidence={command_decision.confidence:.2f}",
# # #                         flush=True,
# # #                     )
# # #                     if command_decision.command == "RETURN":
# # #                         mission_phase = "RETURN_HOME"
# # #                         homotopy_selector.reset()
# # #                         continue
# # #                     success = True
# # #                     break
# # #                 success = True
# # #                 if mission_phase == "RETURN_HOME":
# # #                     roundtrip_completed = True
# # #                 break

# # #         if not rows["goal_distance"]:
# # #             raise RuntimeError("rollout produced no steps")
# # #         rollout_path = output_directory / "rollout.npz"
# # #         np.savez_compressed(
# # #             rollout_path,
# # #             **{
# # #                 key: (
# # #                     np.stack(values)
# # #                     if isinstance(values[0], np.ndarray)
# # #                     else np.asarray(values)
# # #                 )
# # #                 for key, values in rows.items()
# # #             },
# # #             goal_position=goal,
# # #             obstacle_position=(
# # #                 mesh_centroids[0]
# # #                 if mesh_centroids
# # #                 else (
# # #                     ghost
# # #                     if ghost is not None
# # #                     else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
# # #                 )
# # #             ),
# # #             obstacle_positions=(
# # #                 np.stack(mesh_centroids)
# # #                 if mesh_centroids
# # #                 else np.zeros((0, 3), dtype=np.float32)
# # #             ),
# # #             obstacle_velocity_xz=mesh_velocities,
# # #             success=np.asarray(success),
# # #             hz=np.asarray(args.hz, dtype=np.float32),
# # #             start_position_xz=start_position_xz,
# # #             initial_goal_distance=np.asarray(initial_goal_distance, dtype=np.float64),
# # #             stop_distance=np.asarray(args.stop_distance, dtype=np.float64),
# # #             robot_radius=np.asarray(args.robot_radius, dtype=np.float64),
# # #             evaluation_layout=np.asarray(args.evaluation_layout),
# # #             seed=np.asarray(args.seed, dtype=np.int64),
# # #             goal_mode=np.asarray(args.goal_mode),
# # #             belief_pixel_goal=np.asarray(args.belief_pixel_goal),
# # #             interactive_return_home=np.asarray(args.interactive_return_home),
# # #             qwen_freeform_mission=np.asarray(args.qwen_freeform_mission),
# # #             automatic_return_requested=np.asarray(automatic_return_requested),
# # #             roundtrip_completed=np.asarray(roundtrip_completed),
# # #             final_mission_phase=np.asarray(mission_phase),
# # #         )
# # #         with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
# # #             json.dump(
# # #                 {
# # #                     "success": success,
# # #                     "steps": len(rows["goal_distance"]),
# # #                     "archived_observations": args.archive_observations,
# # #                     "final_goal_distance": float(rows["goal_distance"][-1]),
# # #                     "planner": "released_navdp_s2diff_pixels",
# # #                     "controller": "direct_waypoint_no_optimizer",
# # #                     "qwen_role": "homotopy_return_command_and_mission_plan",
# # #                     "qwen_process_isolated_from_habitat": True,
# # #                     "qwen_creates_goal_or_action": False,
# # #                     "qwen_homotopy": args.qwen_homotopy,
# # #                     "qwen_homotopy_events": homotopy_events,
# # #                     "qwen_homotopy_forces_all_candidates": args.qwen_homotopy,
# # #                     "homotopy_sign_convention": {"LEFT": -1.0, "RIGHT": 1.0},
# # #                     "homotopy_minimum_obstacle_pixels": args.homotopy_minimum_obstacle_pixels,
# # #                     "homotopy_release_clear_frames": args.homotopy_release_clear_frames,
# # #                     "homotopy_consistency_repeats": args.homotopy_consistency_repeats,
# # #                     "uses_velocity_chunk": False,
# # #                     "obstacle_mode": args.obstacle_mode,
# # #                     "obstacle_world_xz": args.obstacle_world_xz,
# # #                     "goal_mesh": args.goal_mesh,
# # #                     "particle_anchor": args.particle_anchor,
# # #                     "particle_energy_reweighting": args.particle_energy_reweighting,
# # #                     "particle_collision_mask": args.particle_collision_mask,
# # #                     "goal_mode": args.goal_mode,
# # #                     "interactive_return_home": args.interactive_return_home,
# # #                     "qwen_freeform_mission": args.qwen_freeform_mission,
# # #                     "mission_plan_event": mission_plan_event,
# # #                     "automatic_return_requested": automatic_return_requested,
# # #                     "phase_completion_source": "metric_distance_state_machine",
# # #                     "roundtrip_completed": roundtrip_completed,
# # #                     "final_mission_phase": mission_phase,
# # #                     "return_command_event": return_command_event,
# # #                     "home_belief_source": "spawn_origin_plus_executed_odometry",
# # #                     "reached_goal_becomes_obstacle_on_return": True,
# # #                     "return_goal_obstacle_activation_distance": (
# # #                         args.return_goal_obstacle_activation_distance
# # #                     ),
# # #                     "return_goal_obstacle_dilation_pixels": (
# # #                         args.return_goal_obstacle_dilation_pixels
# # #                     ),
# # #                     "belief_pixel_goal": args.belief_pixel_goal,
# # #                     "belief_source": "semantic_goal_mask_plus_odometry",
# # #                     "belief_bootstrap_world_goal": args.belief_bootstrap_world_goal,
# # #                     "belief_measurement_std": args.belief_measurement_std,
# # #                     "belief_translation_process_std": args.belief_translation_process_std,
# # #                     "belief_yaw_process_std_deg": args.belief_yaw_process_std_deg,
# # #                     "belief_covariance_controls_navdp_mask_size": False,
# # #                     "belief_heading_recovery": args.belief_heading_recovery,
# # #                     "belief_recovery_obstacle_gated": True,
# # #                     "belief_recovery_bearing_deg": args.belief_recovery_bearing_deg,
# # #                     "belief_recovery_maximum_yaw_rate": args.belief_recovery_maximum_yaw_rate,
# # #                     "belief_recovery_maximum_forward_speed": args.belief_recovery_maximum_forward_speed,
# # #                     "particle_noise_schedule": args.particle_noise_schedule,
# # #                     "progressive_guidance": args.progressive_guidance,
# # #                     "mesh_obstacle_count": len(mesh_centroids),
# # #                     "moving_obstacles": bool(np.any(np.abs(mesh_velocities) > 0.0)),
# # #                     "obstacle_velocity_xz": mesh_velocities.tolist(),
# # #                     "evaluation_layout": args.evaluation_layout,
# # #                     "seed": args.seed,
# # #                     "robot_radius": args.robot_radius,
# # #                     "minimum_executed_surface_clearance": (
# # #                         float(np.nanmin(rows["executed_surface_clearance"]))
# # #                         if np.any(np.isfinite(rows["executed_surface_clearance"]))
# # #                         else None
# # #                     ),
# # #                     "geometric_collision": bool(np.any(rows["geometric_collision"])),
# # #                     "rollout": str(rollout_path),
# # #                 },
# # #                 file,
# # #                 indent=2,
# # #             )
# # #         if args.save_video and video_frames:
# # #             save_video(
# # #                 video_frames,
# # #                 output_directory / "rollout.mp4",
# # #                 fps=max(args.hz / max(args.save_every, 1), 1.0),
# # #             )
# # #         print(f"Saved rollout: {rollout_path}", flush=True)
# # #         print(f"Success: {success}", flush=True)
# # #     finally:
# # #         if simulator is not None:
# # #             simulator.close()
# # #         stop_server(server_process)
# # #         stop_server(qwen_process)


# # # if __name__ == "__main__":
# # #     main()
# # from __future__ import annotations

# # import argparse
# # import io
# # import json
# # import math
# # import os
# # import socket
# # import subprocess
# # import sys
# # import time
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Any, Optional, Sequence

# # import habitat_sim
# # import numpy as np
# # import quaternion
# # import requests
# # from habitat_sim.agent import AgentConfiguration
# # from PIL import Image, ImageDraw, ImageFilter

# # from belief_heading_recovery import belief_heading_recovery_action
# # from belief_pixel_goal import GaussianGoalBelief

# # HERE = Path(__file__).resolve().parent
# # SIZE_X = 50.0
# # SIZE_Z = 50.0
# # SIZE_Y = 4.820803273566
# # MESH_GOAL_ID = 10000
# # MESH_OBSTACLE_ID = 2


# # @dataclass(frozen=True)
# # class NavDPS2DiffOutput:
# #     trajectory: np.ndarray
# #     all_trajectories: np.ndarray
# #     all_values: np.ndarray
# #     selected_index: int
# #     fallback_stop: bool
# #     escape_turn: bool
# #     valid_obstacle_points: int
# #     selected_circulation_sign: float
# #     candidate_circulation_signs: np.ndarray
# #     selected_barrier_energy: float
# #     selected_circulation_energy: float
# #     minimum_clearance: np.ndarray
# #     selected_minimum_clearance: float
# #     mean_guidance_noise_correction: float
# #     final_guidance_noise_correction: float
# #     maximum_guidance_noise_correction: float
# #     mean_final_effective_sample_size: float


# # class NavDPS2DiffClient:
# #     def __init__(self, server_url: str, timeout: float = 180.0):
# #         self.server_url = server_url.rstrip("/")
# #         self.timeout = float(timeout)

# #     def reset(
# #         self,
# #         intrinsic: np.ndarray,
# #         *,
# #         stop_threshold: float = -3.0,
# #         batch_size: int = 1,
# #     ) -> str:
# #         intrinsic = np.asarray(intrinsic, dtype=np.float32)
# #         if intrinsic.shape != (3, 3):
# #             raise ValueError(f"intrinsic must have shape [3,3], got {intrinsic.shape}")
# #         response = requests.post(
# #             f"{self.server_url}/navigator_reset",
# #             json={
# #                 "intrinsic": intrinsic.tolist(),
# #                 "stop_threshold": float(stop_threshold),
# #                 "batch_size": int(batch_size),
# #             },
# #             timeout=self.timeout,
# #         )
# #         self._raise_for_error(response)
# #         return str(response.json().get("algo", ""))

# #     def plan(
# #         self,
# #         *,
# #         goal_xy: np.ndarray,
# #         rgb: np.ndarray,
# #         depth: np.ndarray,
# #         obstacle_pixels: np.ndarray,
# #         goal_mode: str = "point",
# #         forced_circulation_sign: float = 0.0,
# #     ) -> NavDPS2DiffOutput:
# #         goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
# #         if goal_xy.shape != (2,):
# #             raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")
# #         if goal_mode not in {"point", "pixel"}:
# #             raise ValueError("goal_mode must be point or pixel")
# #         forced_circulation_sign = float(forced_circulation_sign)
# #         if forced_circulation_sign not in {-1.0, 0.0, 1.0}:
# #             raise ValueError("forced_circulation_sign must be -1, 0, or +1")

# #         rgb = np.asarray(rgb, dtype=np.uint8)
# #         if rgb.ndim != 3 or rgb.shape[-1] < 3:
# #             raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
# #         rgb = rgb[..., :3]

# #         depth = np.asarray(depth, dtype=np.float32)
# #         if depth.ndim == 3 and depth.shape[-1] == 1:
# #             depth = depth[..., 0]
# #         if depth.shape != rgb.shape[:2]:
# #             raise ValueError(
# #                 f"depth/rgb shape mismatch: {depth.shape} vs {rgb.shape[:2]}"
# #             )

# #         if goal_mode == "pixel":
# #             if not np.all(np.isfinite(goal_xy)) or not np.allclose(
# #                 goal_xy, np.round(goal_xy)
# #             ):
# #                 raise ValueError("PixelGoal must be integer [u,v]")
# #             goal_xy = np.round(goal_xy).astype(np.int64)
# #             if not (0 <= goal_xy[0] < rgb.shape[1] and 0 <= goal_xy[1] < rgb.shape[0]):
# #                 raise ValueError("PixelGoal lies outside the RGB image")

# #         pixels = np.asarray(obstacle_pixels)
# #         if pixels.size == 0:
# #             pixels = np.zeros((0, 2), dtype=np.int32)
# #         else:
# #             pixels = pixels.reshape(-1, 2)
# #             if not np.all(np.isfinite(pixels)):
# #                 raise ValueError("obstacle pixels must be finite")
# #             if not np.allclose(pixels, np.round(pixels)):
# #                 raise ValueError("obstacle pixels must be integer [u,v] coordinates")
# #             pixels = np.round(pixels).astype(np.int32)

# #         rgb_bytes = io.BytesIO()
# #         Image.fromarray(rgb, mode="RGB").save(rgb_bytes, format="JPEG", quality=95)
# #         depth_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
# #         depth_bytes = io.BytesIO()
# #         Image.fromarray(depth_u16).save(depth_bytes, format="PNG")

# #         endpoint = "pixelgoal_step" if goal_mode == "pixel" else "pointgoal_step"
# #         response = requests.post(
# #             f"{self.server_url}/{endpoint}",
# #             files={
# #                 "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
# #                 "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
# #             },
# #             data={
# #                 "goal_data": json.dumps(
# #                     {
# #                         "goal_x": [float(goal_xy[0])],
# #                         "goal_y": [float(goal_xy[1])],
# #                         "obstacle_pixels": [pixels.tolist()],
# #                         "forced_circulation_signs": [forced_circulation_sign],
# #                     }
# #                 )
# #             },
# #             timeout=self.timeout,
# #         )
# #         self._raise_for_error(response)
# #         payload = response.json()
# #         diagnostics = payload["s2diff"]
# #         trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
# #         all_trajectories = np.asarray(payload["all_trajectory"], dtype=np.float32)
# #         all_values = np.asarray(payload["all_values"], dtype=np.float32)

# #         return NavDPS2DiffOutput(
# #             trajectory=trajectory[0],
# #             all_trajectories=all_trajectories[0],
# #             all_values=all_values[0],
# #             selected_index=int(diagnostics["selected_index"][0]),
# #             fallback_stop=bool(diagnostics["fallback_stop"][0]),
# #             escape_turn=bool(diagnostics["escape_turn"][0]),
# #             valid_obstacle_points=int(diagnostics["valid_obstacle_points"][0]),
# #             selected_circulation_sign=float(
# #                 diagnostics["selected_circulation_sign"][0]
# #             ),
# #             candidate_circulation_signs=np.asarray(
# #                 diagnostics["candidate_circulation_signs"][0], dtype=np.float32
# #             ),
# #             selected_barrier_energy=float(diagnostics["selected_barrier_energy"][0]),
# #             selected_circulation_energy=float(
# #                 diagnostics["selected_circulation_energy"][0]
# #             ),
# #             minimum_clearance=np.asarray(
# #                 diagnostics["minimum_clearance"][0], dtype=np.float32
# #             ),
# #             selected_minimum_clearance=float(
# #                 diagnostics["selected_minimum_clearance"][0]
# #             ),
# #             mean_guidance_noise_correction=float(
# #                 diagnostics["mean_guidance_noise_correction"][0]
# #             ),
# #             final_guidance_noise_correction=float(
# #                 diagnostics["final_guidance_noise_correction"][0]
# #             ),
# #             maximum_guidance_noise_correction=float(
# #                 diagnostics["maximum_guidance_noise_correction"][0]
# #             ),
# #             mean_final_effective_sample_size=float(
# #                 diagnostics.get("mean_final_effective_sample_size", [0.0])[0]
# #             ),
# #         )

# #     @staticmethod
# #     def _raise_for_error(response: requests.Response) -> None:
# #         try:
# #             payload = response.json()
# #         except ValueError:
# #             payload = None
# #         if isinstance(payload, dict) and "error" in payload:
# #             raise RuntimeError(str(payload["error"]))
# #         response.raise_for_status()


# # @dataclass(frozen=True)
# # class QwenHomotopyDecision:
# #     side: str
# #     circulation_sign: float
# #     confidence: float
# #     obstacle_relevant: bool
# #     queried_qwen: bool
# #     raw_response: Optional[str]
# #     repeated_sides: tuple[str, ...]
# #     repeated_confidences: tuple[float, ...]
# #     consistency_rate: float
# #     used_fallback: bool


# # @dataclass(frozen=True)
# # class QwenCommandDecision:
# #     command: str
# #     confidence: float
# #     raw_response: str


# # @dataclass(frozen=True)
# # class QwenMissionPlanDecision:
# #     plan: tuple[str, ...]
# #     confidence: float
# #     raw_response: str


# # class QwenHomotopyClient:
# #     """HTTP client for the isolated visual-Qwen process."""

# #     def __init__(self, server_url: str, timeout: float = 300.0) -> None:
# #         self.server_url = server_url.rstrip("/")
# #         self.timeout = float(timeout)

# #     def reset(self) -> None:
# #         response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
# #         self._raise_for_error(response)

# #     def step(
# #         self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
# #     ) -> QwenHomotopyDecision:
# #         image_bytes = io.BytesIO()
# #         Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8)).save(
# #             image_bytes, format="PNG"
# #         )
# #         mask_bytes = io.BytesIO()
# #         Image.fromarray((np.asarray(obstacle_mask) > 0).astype(np.uint8) * 255).save(
# #             mask_bytes, format="PNG"
# #         )
# #         response = requests.post(
# #             f"{self.server_url}/select",
# #             files={
# #                 "image": ("overlay.png", image_bytes.getvalue(), "image/png"),
# #                 "obstacle_mask": ("mask.png", mask_bytes.getvalue(), "image/png"),
# #             },
# #             timeout=self.timeout,
# #         )
# #         self._raise_for_error(response)
# #         payload = response.json()
# #         return QwenHomotopyDecision(
# #             side=str(payload["side"]),
# #             circulation_sign=float(payload["circulation_sign"]),
# #             confidence=float(payload["confidence"]),
# #             obstacle_relevant=bool(payload["obstacle_relevant"]),
# #             queried_qwen=bool(payload["queried_qwen"]),
# #             raw_response=payload.get("raw_response"),
# #             repeated_sides=tuple(payload.get("repeated_sides", [])),
# #             repeated_confidences=tuple(
# #                 float(value) for value in payload.get("repeated_confidences", [])
# #             ),
# #             consistency_rate=float(payload.get("consistency_rate", 1.0)),
# #             used_fallback=bool(payload.get("used_fallback", False)),
# #         )

# #     def classify_command(
# #         self, image_rgb: np.ndarray, user_command: str
# #     ) -> QwenCommandDecision:
# #         image_bytes = io.BytesIO()
# #         Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
# #             image_bytes, format="PNG"
# #         )
# #         response = requests.post(
# #             f"{self.server_url}/command",
# #             files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
# #             data={"command": str(user_command)},
# #             timeout=self.timeout,
# #         )
# #         self._raise_for_error(response)
# #         payload = response.json()
# #         return QwenCommandDecision(
# #             command=str(payload["command"]).upper(),
# #             confidence=float(payload["confidence"]),
# #             raw_response=str(payload["raw_response"]),
# #         )

# #     def classify_mission(
# #         self, image_rgb: np.ndarray, user_command: str
# #     ) -> QwenMissionPlanDecision:
# #         image_bytes = io.BytesIO()
# #         Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
# #             image_bytes, format="PNG"
# #         )
# #         response = requests.post(
# #             f"{self.server_url}/mission",
# #             files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
# #             data={"command": str(user_command)},
# #             timeout=self.timeout,
# #         )
# #         self._raise_for_error(response)
# #         payload = response.json()
# #         return QwenMissionPlanDecision(
# #             plan=tuple(str(item).upper() for item in payload["plan"]),
# #             confidence=float(payload["confidence"]),
# #             raw_response=str(payload["raw_response"]),
# #         )

# #     @staticmethod
# #     def _raise_for_error(response: requests.Response) -> None:
# #         try:
# #             payload = response.json()
# #         except ValueError:
# #             payload = None
# #         if isinstance(payload, dict) and "error" in payload:
# #             raise RuntimeError(str(payload["error"]))
# #         response.raise_for_status()


# # def port_is_open(host: str, port: int) -> bool:
# #     try:
# #         with socket.create_connection((host, port), timeout=1.0):
# #             return True
# #     except OSError:
# #         return False


# # def wait_for_server(
# #     process: subprocess.Popen[Any], host: str, port: int, timeout: float
# # ) -> None:
# #     deadline = time.time() + float(timeout)
# #     while time.time() < deadline:
# #         if process.poll() is not None:
# #             raise RuntimeError(
# #                 f"NavDP/S2Diff server exited with code {process.returncode}"
# #             )
# #         if port_is_open(host, port):
# #             return
# #         time.sleep(1.0)
# #     raise TimeoutError(f"NavDP server did not open port {port} within {timeout}s")


# # def stop_server(process: Optional[subprocess.Popen[Any]]) -> None:
# #     if process is None or process.poll() is not None:
# #         return
# #     process.terminate()
# #     try:
# #         process.wait(timeout=10.0)
# #     except subprocess.TimeoutExpired:
# #         process.kill()
# #         process.wait()


# # def start_qwen_homotopy_server(
# #     args: argparse.Namespace,
# # ) -> Optional[subprocess.Popen[Any]]:
# #     if not args.qwen_homotopy or not args.start_qwen_homotopy_server:
# #         return None
# #     if port_is_open(args.qwen_homotopy_host, args.qwen_homotopy_port):
# #         raise RuntimeError(
# #             f"Qwen homotopy port {args.qwen_homotopy_port} is already in use; "
# #             "pass --no-start-qwen-homotopy-server to use an existing service"
# #         )
# #     server_file = HERE / "qwen_homotopy_server.py"
# #     if not server_file.is_file():
# #         raise FileNotFoundError(f"Qwen homotopy server not found: {server_file}")
# #     command = [
# #         str(args.qwen_homotopy_python),
# #         str(server_file),
# #         "--host",
# #         str(args.qwen_homotopy_host),
# #         "--port",
# #         str(args.qwen_homotopy_port),
# #         "--model-id",
# #         str(args.qwen_model_id),
# #         "--device",
# #         str(args.qwen_device),
# #         "--minimum-obstacle-pixels",
# #         str(args.homotopy_minimum_obstacle_pixels),
# #         "--release-clear-frames",
# #         str(args.homotopy_release_clear_frames),
# #         "--consistency-repeats",
# #         str(args.homotopy_consistency_repeats),
# #     ]
# #     print("[qwen-server]", " ".join(command), flush=True)
# #     process = subprocess.Popen(command, cwd=str(HERE))
# #     wait_for_server(
# #         process,
# #         args.qwen_homotopy_host,
# #         args.qwen_homotopy_port,
# #         args.qwen_homotopy_timeout,
# #     )
# #     return process


# # def start_server(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
# #     if not args.start_server:
# #         return None
# #     if port_is_open(args.server_host, args.server_port):
# #         raise RuntimeError(
# #             f"port {args.server_port} is already in use; use --no-start-server "
# #             "to connect to an existing guided server"
# #         )

# #     navdp_root = Path(args.navdp_root).expanduser().resolve()
# #     checkpoint = Path(args.navdp_checkpoint).expanduser().resolve()
# #     server_dir = navdp_root / "baselines" / "navdp"
# #     server_file = server_dir / "navdp_s2diff_server.py"
# #     if not server_file.is_file():
# #         raise FileNotFoundError(f"guided server not found: {server_file}")
# #     if not checkpoint.is_file():
# #         raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint}")

# #     command = [
# #         str(args.navdp_python),
# #         str(server_file),
# #         "--checkpoint",
# #         str(checkpoint),
# #         "--device",
# #         str(args.navdp_device),
# #         "--planner-mode",
# #         str(args.planner_mode),
# #         "--seed",
# #         str(args.seed),
# #         "--port",
# #         str(args.server_port),
# #         "--candidates",
# #         str(args.candidates),
# #         "--particles",
# #         str(args.particles),
# #         "--particle-std",
# #         str(args.particle_std),
# #         "--gradient-steps",
# #         str(args.gradient_steps),
# #         "--gradient-step-size",
# #         str(args.gradient_step_size),
# #         "--guidance-strength",
# #         str(args.guidance_strength),
# #         "--temperature",
# #         str(args.temperature),
# #         "--safe-distance",
# #         str(args.safe_distance),
# #         "--hard-collision-distance",
# #         str(args.hard_collision_distance),
# #         "--robot-radius",
# #         str(args.robot_radius),
# #         "--safety-weight",
# #         str(args.safety_weight),
# #         "--barrier-weight",
# #         str(args.barrier_weight),
# #         "--barrier-rate",
# #         str(args.barrier_rate),
# #         "--circulation-weight",
# #         str(args.circulation_weight),
# #         "--circulation-activation-distance",
# #         str(args.circulation_activation_distance),
# #         "--circulation-activation-sharpness",
# #         str(args.circulation_activation_sharpness),
# #         "--minimum-circulation-progress",
# #         str(args.minimum_circulation_progress),
# #         "--blocking-alignment-threshold",
# #         str(args.blocking_alignment_threshold),
# #         "--circulation-switch-weight",
# #         str(args.circulation_switch_weight),
# #         "--escape-lateral-target",
# #         str(args.escape_lateral_target),
# #         "--minimum-obstacle-depth",
# #         str(args.minimum_obstacle_depth),
# #         "--maximum-obstacle-depth",
# #         str(args.maximum_obstacle_depth),
# #         "--maximum-obstacle-pixels",
# #         str(args.maximum_obstacle_pixels),
# #     ]
# #     particle_flags = {
# #         "particle-anchor": args.particle_anchor,
# #         "particle-energy-reweighting": args.particle_energy_reweighting,
# #         "particle-collision-mask": args.particle_collision_mask,
# #         "particle-noise-schedule": args.particle_noise_schedule,
# #         "progressive-guidance": args.progressive_guidance,
# #     }
# #     for name, enabled in particle_flags.items():
# #         command.append(f"--{name}" if enabled else f"--no-{name}")
# #     command.append("--remove-critic" if args.remove_critic else "--no-remove-critic")
# #     print("[server]", " ".join(command), flush=True)
# #     process = subprocess.Popen(command, cwd=str(server_dir))
# #     wait_for_server(process, args.server_host, args.server_port, args.server_timeout)
# #     return process


# # def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
# #     height, width = grid.shape
# #     x0 = int(np.floor(px))
# #     y0 = int(np.floor(py))
# #     x1 = min(x0 + 1, width - 1)
# #     y1 = min(y0 + 1, height - 1)
# #     tx = px - x0
# #     ty = py - y0
# #     top = float(grid[y0, x0]) * (1.0 - tx) + float(grid[y0, x1]) * tx
# #     bottom = float(grid[y1, x0]) * (1.0 - tx) + float(grid[y1, x1]) * tx
# #     return top * (1.0 - ty) + bottom * ty


# # class TerrainHeight:
# #     def __init__(
# #         self,
# #         *,
# #         mode: str,
# #         heightmap: Optional[Path],
# #         obj: Optional[Path],
# #         flat_y: float,
# #         size_x: float,
# #         size_z: float,
# #         size_y: float,
# #         flip_x: bool,
# #         flip_z: bool,
# #         swap_xz: bool,
# #     ):
# #         if mode == "auto":
# #             mode = (
# #                 "heightmap"
# #                 if heightmap and heightmap.exists()
# #                 else ("obj" if obj and obj.exists() else "flat")
# #             )
# #         self.mode = mode
# #         self.flat_y = float(flat_y)
# #         self.size_x = float(size_x)
# #         self.size_z = float(size_z)
# #         self.size_y = float(size_y)
# #         self.flip_x = bool(flip_x)
# #         self.flip_z = bool(flip_z)
# #         self.swap_xz = bool(swap_xz)
# #         self.height: Optional[np.ndarray] = None
# #         self.obj_xs: Optional[np.ndarray] = None
# #         self.obj_zs: Optional[np.ndarray] = None
# #         self.obj_h: Optional[np.ndarray] = None

# #         if mode == "heightmap":
# #             if heightmap is None or not heightmap.exists():
# #                 raise FileNotFoundError(f"heightmap not found: {heightmap}")
# #             array = np.asarray(Image.open(heightmap))
# #             if array.ndim == 3:
# #                 array = array[..., 0]
# #             array = array.astype(np.float32)
# #             array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
# #             self.height = array * self.size_y - float(np.mean(array * self.size_y))
# #         elif mode == "obj":
# #             if obj is None or not obj.exists():
# #                 raise FileNotFoundError(f"terrain OBJ not found: {obj}")
# #             vertices = []
# #             with obj.open("r", encoding="utf-8", errors="ignore") as file:
# #                 for line in file:
# #                     if line.startswith("v "):
# #                         parts = line.split()
# #                         if len(parts) >= 4:
# #                             vertices.append(tuple(float(value) for value in parts[1:4]))
# #             if not vertices:
# #                 raise RuntimeError(f"no vertices found in {obj}")
# #             array = np.asarray(vertices, dtype=np.float32)
# #             xs = np.unique(array[:, 0])
# #             zs = np.unique(array[:, 1])
# #             grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
# #             x_index = {float(value): index for index, value in enumerate(xs.tolist())}
# #             z_index = {float(value): index for index, value in enumerate(zs.tolist())}
# #             for x, z, height in array:
# #                 grid[z_index[float(z)], x_index[float(x)]] = height
# #             self.obj_xs = xs
# #             self.obj_zs = zs
# #             self.obj_h = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
# #         elif mode != "flat":
# #             raise ValueError(f"unknown terrain mode: {mode}")

# #     def _map(self, x: float, z: float) -> tuple[float, float]:
# #         if self.swap_xz:
# #             x, z = z, x
# #         u = (x + self.size_x / 2.0) / self.size_x
# #         v = (z + self.size_z / 2.0) / self.size_z
# #         if self.flip_x:
# #             u = 1.0 - u
# #         if self.flip_z:
# #             v = 1.0 - v
# #         return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

# #     def __call__(self, x: float, z: float) -> float:
# #         if self.mode == "flat":
# #             return self.flat_y
# #         if self.mode == "heightmap":
# #             assert self.height is not None
# #             u, v = self._map(x, z)
# #             return bilinear_grid(
# #                 self.height,
# #                 u * (self.height.shape[1] - 1),
# #                 v * (self.height.shape[0] - 1),
# #             )
# #         assert (
# #             self.obj_xs is not None
# #             and self.obj_zs is not None
# #             and self.obj_h is not None
# #         )
# #         xx = float(np.clip(x, self.obj_xs[0], self.obj_xs[-1]))
# #         zz = float(np.clip(z, self.obj_zs[0], self.obj_zs[-1]))
# #         column = int(
# #             np.clip(np.searchsorted(self.obj_xs, xx) - 1, 0, len(self.obj_xs) - 2)
# #         )
# #         row = int(
# #             np.clip(np.searchsorted(self.obj_zs, zz) - 1, 0, len(self.obj_zs) - 2)
# #         )
# #         x0, x1 = float(self.obj_xs[column]), float(self.obj_xs[column + 1])
# #         z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
# #         tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
# #         tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
# #         top = (
# #             float(self.obj_h[row, column]) * (1.0 - tx)
# #             + float(self.obj_h[row, column + 1]) * tx
# #         )
# #         bottom = (
# #             float(self.obj_h[row + 1, column]) * (1.0 - tx)
# #             + float(self.obj_h[row + 1, column + 1]) * tx
# #         )
# #         return top * (1.0 - tz) + bottom * tz

# #     def local_height_max(
# #         self, x: float, z: float, radius: float, samples: int = 5
# #     ) -> float:
# #         if radius <= 1e-6:
# #             return float(self(x, z))
# #         values = [
# #             float(self(x + dx, z + dz))
# #             for dx in np.linspace(-radius, radius, samples)
# #             for dz in np.linspace(-radius, radius, samples)
# #             if dx * dx + dz * dz <= radius * radius + 1e-8
# #         ]
# #         return max(values) if values else float(self(x, z))


# # def make_sensor(
# #     uuid: str, sensor_type: Any, height: int, width: int, hfov_deg: float
# # ) -> habitat_sim.CameraSensorSpec:
# #     specification = habitat_sim.CameraSensorSpec()
# #     specification.uuid = uuid
# #     specification.sensor_type = sensor_type
# #     specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
# #     specification.resolution = [int(height), int(width)]
# #     specification.position = [0.0, 0.0, 0.0]
# #     specification.hfov = float(hfov_deg)
# #     return specification


# # def make_simulator(
# #     scene: Path,
# #     height: int,
# #     width: int,
# #     hfov_deg: float,
# #     *,
# #     with_semantic: bool,
# # ):
# #     simulator_configuration = habitat_sim.SimulatorConfiguration()
# #     simulator_configuration.scene_id = str(scene.expanduser().resolve())
# #     simulator_configuration.enable_physics = False
# #     sensors = [
# #         make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
# #         make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
# #     ]
# #     if with_semantic:
# #         sensors.append(
# #             make_sensor(
# #                 "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
# #             )
# #         )
# #     agent_configuration = AgentConfiguration()
# #     agent_configuration.sensor_specifications = sensors
# #     return habitat_sim.Simulator(
# #         habitat_sim.Configuration(simulator_configuration, [agent_configuration])
# #     )


# # def set_agent_pose(agent: Any, position: np.ndarray, yaw: float) -> None:
# #     state = agent.get_state()
# #     state.position = np.asarray(position, dtype=np.float32)
# #     state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
# #     agent.set_state(state)


# # def rgb_depth(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
# #     rgb = np.asarray(observation["rgb"])
# #     if rgb.ndim == 3 and rgb.shape[-1] == 4:
# #         rgb = rgb[..., :3]
# #     depth = np.asarray(observation["depth"], dtype=np.float32)
# #     if depth.ndim == 3:
# #         depth = depth[..., 0]
# #     return rgb.astype(np.uint8), depth.astype(np.float32)


# # def semantic_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
# #     semantic = np.asarray(observation["semantic"])
# #     if semantic.ndim == 3:
# #         semantic = semantic[..., 0]
# #     return semantic.astype(np.int32)


# # def pixel_to_world(
# #     u: float,
# #     v: float,
# #     depth: float,
# #     position: np.ndarray,
# #     yaw: float,
# #     intrinsic: np.ndarray,
# # ) -> np.ndarray:
# #     right = (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
# #     up = -(v - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
# #     forward_vector = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)])
# #     right_vector = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)])
# #     return (
# #         np.asarray(position, dtype=np.float64)
# #         + depth * forward_vector
# #         + right * right_vector
# #         + up * np.asarray([0.0, 1.0, 0.0])
# #     )


# # def depth_patch_mesh(
# #     u_center: float,
# #     v_center: float,
# #     half_size: int,
# #     stride: int,
# #     depth: np.ndarray,
# #     position: np.ndarray,
# #     yaw: float,
# #     intrinsic: np.ndarray,
# #     *,
# #     lift: float,
# #     maximum_depth_jump: float = 0.4,
# # ) -> tuple[np.ndarray, np.ndarray]:
# #     height, width = depth.shape
# #     columns = list(
# #         range(
# #             max(0, int(u_center - half_size)),
# #             min(width, int(u_center + half_size) + 1),
# #             max(int(stride), 1),
# #         )
# #     )
# #     rows = list(
# #         range(
# #             max(0, int(v_center - half_size)),
# #             min(height, int(v_center + half_size) + 1),
# #             max(int(stride), 1),
# #         )
# #     )
# #     indices = -np.ones((len(rows), len(columns)), dtype=np.int64)
# #     depths = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
# #     vertices: list[tuple[float, float, float]] = []
# #     for row_index, v in enumerate(rows):
# #         for column_index, u in enumerate(columns):
# #             metric_depth = float(depth[v, u])
# #             if not np.isfinite(metric_depth) or metric_depth <= 0.1:
# #                 continue
# #             indices[row_index, column_index] = len(vertices)
# #             depths[row_index, column_index] = metric_depth
# #             point = pixel_to_world(
# #                 u, v, metric_depth, position, yaw, intrinsic
# #             ) + float(lift) * np.asarray([0.0, 1.0, 0.0])
# #             vertices.append(tuple(float(value) for value in point))

# #     faces: list[tuple[int, int, int]] = []
# #     for row_index in range(len(rows) - 1):
# #         for column_index in range(len(columns) - 1):
# #             a = int(indices[row_index, column_index])
# #             b = int(indices[row_index, column_index + 1])
# #             c = int(indices[row_index + 1, column_index])
# #             d = int(indices[row_index + 1, column_index + 1])
# #             if min(a, b, c, d) < 0:
# #                 continue
# #             cell_depths = (
# #                 depths[row_index, column_index],
# #                 depths[row_index, column_index + 1],
# #                 depths[row_index + 1, column_index],
# #                 depths[row_index + 1, column_index + 1],
# #             )
# #             if max(cell_depths) - min(cell_depths) > maximum_depth_jump:
# #                 continue
# #             faces.append((a, c, d))
# #             faces.append((a, d, b))
# #     return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# # def save_obj(
# #     path: Path,
# #     vertices: np.ndarray,
# #     faces: np.ndarray,
# #     *,
# #     diffuse_rgb: Optional[tuple[float, float, float]] = None,
# # ) -> None:
# #     material_name = None
# #     if diffuse_rgb is not None:
# #         red, green, blue = (float(value) for value in diffuse_rgb)
# #         if not all(0.0 <= value <= 1.0 for value in (red, green, blue)):
# #             raise ValueError("OBJ diffuse material values must be in [0, 1]")
# #         material_name = "mesh_material"
# #         material_path = path.with_suffix(".mtl")
# #         with material_path.open("w", encoding="utf-8") as material:
# #             material.write(f"newmtl {material_name}\n")
# #             material.write(
# #                 f"Ka {0.25 * red:.4f} {0.25 * green:.4f} {0.25 * blue:.4f}\n"
# #             )
# #             material.write(f"Kd {red:.4f} {green:.4f} {blue:.4f}\n")
# #             material.write("Ks 0.1000 0.1000 0.1000\n")
# #             material.write("Ns 24.0000\n")
# #             material.write("d 1.0000\n")
# #             material.write("illum 2\n")
# #     with path.open("w", encoding="utf-8") as file:
# #         if material_name is not None:
# #             file.write(f"mtllib {path.with_suffix('.mtl').name}\n")
# #             file.write(f"usemtl {material_name}\n")
# #         for x, y, z in vertices:
# #             file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
# #         for a, b, c in faces:
# #             file.write(f"f {a + 1} {b + 1} {c + 1}\n")


# # def register_semantic_mesh(simulator: Any, mesh_path: Path, semantic_id: int) -> Any:
# #     template_manager = simulator.get_object_template_manager()
# #     object_manager = simulator.get_rigid_object_manager()
# #     template = template_manager.create_new_template(str(mesh_path))
# #     template.render_asset_handle = str(mesh_path)
# #     template.collision_asset_handle = str(mesh_path)
# #     template.is_collidable = False
# #     template_id = template_manager.register_template(
# #         template, f"s2diff_obstacle_{semantic_id}_{os.path.basename(mesh_path)}"
# #     )
# #     object_handle = template_manager.get_template_handle_by_id(template_id)
# #     obstacle = object_manager.add_object_by_template_handle(object_handle)
# #     obstacle.motion_type = habitat_sim.physics.MotionType.KINEMATIC
# #     obstacle.collidable = False
# #     obstacle.semantic_id = int(semantic_id)
# #     return obstacle


# # def parse_world_xz(specification: str) -> tuple[float, float]:
# #     values = [float(value) for value in str(specification).split(",")]
# #     if len(values) != 2 or not np.isfinite(values).all():
# #         raise ValueError(
# #             f"world mesh position must be finite X,Z, got {specification!r}"
# #         )
# #     return values[0], values[1]


# # def world_box_mesh(
# #     center_x: float,
# #     base_y: float,
# #     center_z: float,
# #     half_extent: float,
# #     height: float,
# # ) -> tuple[np.ndarray, np.ndarray]:
# #     """Create a closed axis-aligned box whose vertices are in world coordinates."""

# #     if half_extent <= 0.0 or height <= 0.0:
# #         raise ValueError("box half extent and height must be positive")
# #     x0, x1 = center_x - half_extent, center_x + half_extent
# #     z0, z1 = center_z - half_extent, center_z + half_extent
# #     y0, y1 = base_y, base_y + height
# #     vertices = np.asarray(
# #         [
# #             [x0, y0, z0],
# #             [x1, y0, z0],
# #             [x1, y0, z1],
# #             [x0, y0, z1],
# #             [x0, y1, z0],
# #             [x1, y1, z0],
# #             [x1, y1, z1],
# #             [x0, y1, z1],
# #         ],
# #         dtype=np.float64,
# #     )
# #     faces = np.asarray(
# #         [
# #             [0, 2, 1],
# #             [0, 3, 2],
# #             [4, 5, 6],
# #             [4, 6, 7],
# #             [0, 1, 5],
# #             [0, 5, 4],
# #             [1, 2, 6],
# #             [1, 6, 5],
# #             [2, 3, 7],
# #             [2, 7, 6],
# #             [3, 0, 4],
# #             [3, 4, 7],
# #         ],
# #         dtype=np.int64,
# #     )
# #     return vertices, faces


# # def place_world_obstacle_meshes(
# #     simulator: Any,
# #     terrain: Any,
# #     xz_specifications: Sequence[str],
# #     output_directory: Path,
# #     *,
# #     half_extent: float,
# #     height: float,
# # ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
# #     """Place static obstacle boxes at exact world X,Z coordinates."""

# #     mesh_directory = output_directory / "meshes"
# #     mesh_directory.mkdir(parents=True, exist_ok=True)
# #     objects: list[Any] = []
# #     centroids: list[np.ndarray] = []
# #     geometries: list[np.ndarray] = []
# #     for index, specification in enumerate(xz_specifications):
# #         center_x, center_z = parse_world_xz(specification)
# #         base_y = terrain.local_height_max(center_x, center_z, half_extent)
# #         vertices, faces = world_box_mesh(
# #             center_x, base_y, center_z, half_extent, height
# #         )
# #         mesh_path = mesh_directory / f"world_obstacle_{index}.obj"
# #         save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.78, 0.16, 0.06))
# #         semantic_id = MESH_OBSTACLE_ID + index
# #         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
# #         centroid = vertices.mean(axis=0).astype(np.float32)
# #         centroids.append(centroid)
# #         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
# #         print(
# #             f"[world-mesh] obstacle={index} semantic_id={semantic_id} "
# #             f"center_xz={[center_x, center_z]} half_extent={half_extent:.3f} "
# #             f"height={height:.3f}",
# #             flush=True,
# #         )
# #     return objects, centroids, geometries


# # def place_world_goal_mesh(
# #     simulator: Any,
# #     terrain: Any,
# #     goal_x: float,
# #     goal_z: float,
# #     output_directory: Path,
# #     *,
# #     half_extent: float,
# #     height: float,
# # ) -> Any:
# #     """Place a visible, non-obstacle semantic goal marker at the exact goal."""

# #     base_y = terrain.local_height_max(goal_x, goal_z, half_extent)
# #     vertices, faces = world_box_mesh(goal_x, base_y, goal_z, half_extent, height)
# #     mesh_directory = output_directory / "meshes"
# #     mesh_directory.mkdir(parents=True, exist_ok=True)
# #     mesh_path = mesh_directory / "goal_marker.obj"
# #     save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.08, 0.85, 0.18))
# #     goal_object = register_semantic_mesh(simulator, mesh_path, MESH_GOAL_ID)
# #     print(
# #         f"[world-mesh] goal semantic_id={MESH_GOAL_ID} "
# #         f"center_xz={[goal_x, goal_z]}",
# #         flush=True,
# #     )
# #     return goal_object


# # def parse_uv_fraction(
# #     specification: str, width: int, height: int
# # ) -> tuple[float, float]:
# #     u_fraction, v_fraction = (float(value) for value in str(specification).split(","))
# #     if not (0.0 <= u_fraction <= 1.0 and 0.0 <= v_fraction <= 1.0):
# #         raise ValueError(f"mesh pixel fraction must be in [0,1], got {specification!r}")
# #     return u_fraction * width, v_fraction * height


# # def place_obstacle_meshes(
# #     simulator: Any,
# #     depth: np.ndarray,
# #     position: np.ndarray,
# #     yaw: float,
# #     intrinsic: np.ndarray,
# #     uv_specifications: Sequence[str],
# #     output_directory: Path,
# #     *,
# #     mesh_half_pixels: int,
# #     mesh_lift: float,
# # ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
# #     mesh_directory = output_directory / "meshes"
# #     mesh_directory.mkdir(parents=True, exist_ok=True)
# #     height, width = depth.shape
# #     objects: list[Any] = []
# #     centroids: list[np.ndarray] = []
# #     geometries: list[np.ndarray] = []
# #     for index, specification in enumerate(uv_specifications):
# #         u, v = parse_uv_fraction(specification, width, height)
# #         vertices, faces = depth_patch_mesh(
# #             u,
# #             v,
# #             mesh_half_pixels,
# #             2,
# #             depth,
# #             position,
# #             yaw,
# #             intrinsic,
# #             lift=mesh_lift,
# #         )
# #         if len(vertices) == 0 or len(faces) == 0:
# #             raise RuntimeError(
# #                 f"obstacle mesh {index} at {specification!r} has no valid depth surface"
# #             )
# #         mesh_path = mesh_directory / f"obstacle_{index}.obj"
# #         save_obj(mesh_path, vertices, faces)
# #         semantic_id = MESH_OBSTACLE_ID + index
# #         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
# #         centroid = vertices.mean(axis=0).astype(np.float32)
# #         centroids.append(centroid)
# #         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
# #         print(
# #             f"[mesh] obstacle={index} semantic_id={semantic_id} "
# #             f"pixels={specification} vertices={len(vertices)} "
# #             f"world={centroid.tolist()}",
# #             flush=True,
# #         )
# #     return objects, centroids, geometries


# # def planar_mesh_clearance(
# #     point_xz: np.ndarray,
# #     geometries: Sequence[np.ndarray],
# # ) -> float:
# #     """Minimum 2-D distance from a robot center to projected mesh triangles."""
# #     point = np.asarray(point_xz, dtype=np.float64)
# #     best = float("inf")
# #     for triangles in geometries:
# #         triangles = np.asarray(triangles, dtype=np.float64)
# #         if triangles.size == 0:
# #             continue
# #         a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
# #         v0, v1, v2 = b - a, c - a, point[None, :] - a
# #         denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
# #         valid = np.abs(denominator) > 1.0e-12
# #         safe_denominator = np.where(valid, denominator, 1.0)
# #         u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_denominator
# #         v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_denominator
# #         if np.any(valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)):
# #             return 0.0

# #         starts = np.concatenate((a, b, c), axis=0)
# #         ends = np.concatenate((b, c, a), axis=0)
# #         segments = ends - starts
# #         squared_lengths = np.einsum("ij,ij->i", segments, segments)
# #         numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
# #         fractions = np.divide(
# #             numerators,
# #             squared_lengths,
# #             out=np.zeros_like(numerators),
# #             where=squared_lengths > 1.0e-12,
# #         )
# #         fractions = np.clip(fractions, 0.0, 1.0)
# #         closest = starts + fractions[:, None] * segments
# #         best = min(best, float(np.linalg.norm(point[None, :] - closest, axis=1).min()))
# #     return best


# # def parse_xz_velocity(specification: str) -> np.ndarray:
# #     values = [float(value) for value in str(specification).split(",")]
# #     if len(values) != 2 or not np.all(np.isfinite(values)):
# #         raise ValueError("obstacle velocity must be finite vx,vz")
# #     return np.asarray(values, dtype=np.float64)


# # def expand_obstacle_velocities(
# #     specifications: Sequence[str], obstacle_count: int
# # ) -> np.ndarray:
# #     if obstacle_count == 0:
# #         return np.zeros((0, 2), dtype=np.float64)
# #     if not specifications:
# #         return np.zeros((obstacle_count, 2), dtype=np.float64)
# #     velocities = np.stack([parse_xz_velocity(item) for item in specifications])
# #     if len(velocities) == 1 and obstacle_count > 1:
# #         velocities = np.repeat(velocities, obstacle_count, axis=0)
# #     if len(velocities) != obstacle_count:
# #         raise ValueError(
# #             "provide one obstacle velocity to broadcast or one velocity per mesh"
# #         )
# #     return velocities


# # def translated_mesh_geometry(
# #     base_geometries: Sequence[np.ndarray],
# #     base_centroids: Sequence[np.ndarray],
# #     velocities_xz: np.ndarray,
# #     elapsed_seconds: float,
# # ) -> tuple[list[np.ndarray], list[np.ndarray]]:
# #     geometries: list[np.ndarray] = []
# #     centroids: list[np.ndarray] = []
# #     for geometry, centroid, velocity in zip(
# #         base_geometries, base_centroids, velocities_xz
# #     ):
# #         offset_xz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
# #         geometries.append(np.asarray(geometry) + offset_xz[None, None, :])
# #         offset_xyz = np.asarray([offset_xz[0], 0.0, offset_xz[1]])
# #         centroids.append(np.asarray(centroid, dtype=np.float64) + offset_xyz)
# #     return geometries, centroids


# # def move_mesh_objects(
# #     objects: Sequence[Any], velocities_xz: np.ndarray, elapsed_seconds: float
# # ) -> None:
# #     for obstacle, velocity in zip(objects, velocities_xz):
# #         dx, dz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
# #         vector_type = type(obstacle.translation)
# #         obstacle.translation = vector_type(float(dx), 0.0, float(dz))


# # def camera_coordinates(
# #     point: np.ndarray, position: np.ndarray, yaw: float
# # ) -> tuple[float, float, float]:
# #     delta = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
# #     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
# #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# #     forward = forward_x * float(delta[0]) + forward_z * float(delta[2])
# #     left = left_x * float(delta[0]) + left_z * float(delta[2])
# #     return -left, float(delta[1]), forward


# # def camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
# #     hfov = math.radians(float(hfov_deg))
# #     focal = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
# #     return np.asarray(
# #         [
# #             [focal, 0.0, (width - 1) * 0.5],
# #             [0.0, focal, (height - 1) * 0.5],
# #             [0.0, 0.0, 1.0],
# #         ],
# #         dtype=np.float32,
# #     )


# # def world_goal_to_pixel(
# #     point: np.ndarray,
# #     position: np.ndarray,
# #     yaw: float,
# #     intrinsic: np.ndarray,
# #     height: int,
# #     width: int,
# # ) -> np.ndarray:
# #     """Project a world goal to a valid PixelGoal, clamping off-screen bearings."""

# #     right, up, forward = camera_coordinates(point, position, yaw)
# #     fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
# #     cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
# #     margin = 11
# #     bearing = math.atan2(right, forward)
# #     maximum_bearing = math.atan2(max(cx - margin, 1.0), fx)
# #     bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
# #     u = cx + fx * math.tan(bearing)
# #     v = cy - fy * up / forward if forward > 0.05 else 0.62 * height
# #     return np.asarray(
# #         [
# #             int(np.clip(round(u), margin, width - margin - 1)),
# #             int(np.clip(round(v), margin, height - margin - 1)),
# #         ],
# #         dtype=np.int32,
# #     )


# # def circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
# #     yy, xx = np.ogrid[:height, :width]
# #     return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


# # def project_world_mask(
# #     point: np.ndarray,
# #     position: np.ndarray,
# #     yaw: float,
# #     intrinsic: np.ndarray,
# #     height: int,
# #     width: int,
# #     radius: int,
# # ) -> tuple[np.ndarray, float]:
# #     right, up, forward = camera_coordinates(point, position, yaw)
# #     if forward <= 0.05:
# #         return np.zeros((height, width), dtype=np.uint8), forward
# #     u = float(intrinsic[0, 2] + intrinsic[0, 0] * right / forward)
# #     v = float(intrinsic[1, 2] - intrinsic[1, 1] * up / forward)
# #     if not (radius <= u < width - radius and radius <= v < height - radius):
# #         return np.zeros((height, width), dtype=np.uint8), forward
# #     return circle_mask(height, width, u, v, radius), forward


# # def depth_obstacle_mask(
# #     depth: np.ndarray, threshold: float, minimum_y_fraction: float
# # ) -> np.ndarray:
# #     mask = np.isfinite(depth) & (depth > 0.05) & (depth < float(threshold))
# #     mask[: int(depth.shape[0] * minimum_y_fraction)] = False
# #     return mask.astype(np.uint8)


# # def pixels_from_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
# #     v, u = np.nonzero(np.asarray(mask) > 0)
# #     if u.size == 0:
# #         return np.zeros((0, 2), dtype=np.int32)
# #     pixels = np.stack((u, v), axis=-1).astype(np.int32)
# #     if maximum > 0 and len(pixels) > maximum:
# #         indices = np.linspace(0, len(pixels) - 1, maximum).astype(np.int64)
# #         pixels = pixels[indices]
# #     return pixels


# # def waypoint_action(
# #     trajectory: np.ndarray,
# #     *,
# #     lookahead_index: int,
# #     maximum_forward_speed: float,
# #     maximum_yaw_rate: float,
# #     yaw_gain: float,
# # ) -> np.ndarray:
# #     trajectory = np.asarray(trajectory, dtype=np.float32)
# #     if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
# #         return np.zeros(3, dtype=np.float32)
# #     if np.max(np.linalg.norm(trajectory[:, :2], axis=-1)) < 1e-5:
# #         return np.zeros(3, dtype=np.float32)
# #     index = int(np.clip(lookahead_index, 0, trajectory.shape[0] - 1))
# #     forward, left = float(trajectory[index, 0]), float(trajectory[index, 1])
# #     bearing = math.atan2(left, max(forward, 1e-4))
# #     velocity = maximum_forward_speed * max(0.0, math.cos(bearing))
# #     yaw_rate = float(np.clip(yaw_gain * bearing, -maximum_yaw_rate, maximum_yaw_rate))
# #     return np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)


# # def integrate_mars(
# #     position: np.ndarray, yaw: float, action: np.ndarray, dt: float
# # ) -> tuple[np.ndarray, float]:
# #     forward_velocity, lateral_velocity, yaw_rate = [float(value) for value in action]
# #     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
# #     left_x, left_z = -math.cos(yaw), math.sin(yaw)
# #     output = np.asarray(position, dtype=np.float32).copy()
# #     output[0] += (forward_x * forward_velocity + left_x * lateral_velocity) * dt
# #     output[2] += (forward_z * forward_velocity + left_z * lateral_velocity) * dt
# #     return output, yaw + yaw_rate * dt


# # def wrap_angle(angle: float) -> float:
# #     return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# # def overlay_frame(
# #     rgb: np.ndarray,
# #     goal_mask: np.ndarray,
# #     obstacle_mask: np.ndarray,
# #     text: str,
# #     *,
# #     show_masks: bool,
# #     detection_box: Optional[np.ndarray] = None,
# #     detection_label: Optional[str] = None,
# # ) -> Image.Image:
# #     output = np.asarray(rgb, dtype=np.uint8).copy()
# #     if show_masks:
# #         output[goal_mask > 0] = (
# #             0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
# #         ).astype(np.uint8)
# #         output[obstacle_mask > 0] = (
# #             0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
# #         ).astype(np.uint8)
# #     image = Image.fromarray(output)
# #     draw = ImageDraw.Draw(image)
# #     if detection_box is not None:
# #         x1, y1, x2, y2 = [float(value) for value in detection_box]
# #         draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
# #         if detection_label:
# #             draw.text((x1 + 2, max(y1 - 14, 2)), detection_label, fill=(255, 255, 0))
# #     draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
# #     draw.text((10, 9), text, fill=(255, 255, 255))
# #     return image


# # def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
# #     """Dilate a binary image mask without adding a SciPy dependency."""

# #     binary = (np.asarray(mask) > 0).astype(np.uint8)
# #     if radius <= 0 or not np.any(binary):
# #         return binary
# #     kernel_size = 2 * int(radius) + 1
# #     return (
# #         np.asarray(
# #             Image.fromarray(binary * 255).filter(ImageFilter.MaxFilter(kernel_size))
# #         )
# #         > 0
# #     ).astype(np.uint8)


# # def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
# #     import imageio.v2 as imageio

# #     with imageio.get_writer(path, fps=float(fps)) as writer:
# #         for frame in frames:
# #             writer.append_data(np.asarray(frame.convert("RGB")))


# # def parser() -> argparse.ArgumentParser:
# #     argument_parser = argparse.ArgumentParser(
# #         description="One-file released NavDP + in-denoising S2Diff Mars rollout"
# #     )
# #     argument_parser.add_argument("--navdp-root", required=True)
# #     argument_parser.add_argument("--navdp-checkpoint", required=True)
# #     argument_parser.add_argument("--navdp-python", default=sys.executable)
# #     argument_parser.add_argument("--navdp-device", default="cuda:0")
# #     argument_parser.add_argument(
# #         "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
# #     )
# #     argument_parser.add_argument(
# #         "--goal-mode", choices=["point", "pixel"], default="point"
# #     )
# #     argument_parser.add_argument(
# #         "--belief-pixel-goal",
# #         action=argparse.BooleanOptionalAction,
# #         default=False,
# #         help=(
# #             "Use the live semantic goal mask to correct a body-frame Gaussian "
# #             "belief and its projected mean as NavDP's PixelGoal while occluded."
# #         ),
# #     )
# #     argument_parser.add_argument("--belief-minimum-goal-pixels", type=int, default=10)
# #     argument_parser.add_argument("--belief-measurement-std", type=float, default=0.05)
# #     argument_parser.add_argument(
# #         "--belief-translation-process-std", type=float, default=0.03
# #     )
# #     argument_parser.add_argument(
# #         "--belief-yaw-process-std-deg", type=float, default=1.0
# #     )
# #     argument_parser.add_argument(
# #         "--belief-bootstrap-world-goal",
# #         action=argparse.BooleanOptionalAction,
# #         default=False,
# #         help=(
# #             "Simulation-only bootstrap when the goal is initially invisible. "
# #             "Disable for a strict detector-only evaluation."
# #         ),
# #     )
# #     argument_parser.add_argument("--belief-bootstrap-std", type=float, default=0.50)
# #     argument_parser.add_argument("--belief-ghost-base-radius", type=int, default=10)
# #     argument_parser.add_argument(
# #         "--belief-ghost-covariance-scale", type=float, default=2.0
# #     )
# #     argument_parser.add_argument("--belief-ghost-maximum-radius", type=int, default=80)
# #     argument_parser.add_argument(
# #         "--belief-heading-recovery",
# #         action=argparse.BooleanOptionalAction,
# #         default=True,
# #     )
# #     argument_parser.add_argument(
# #         "--belief-recovery-bearing-deg", type=float, default=35.0
# #     )
# #     argument_parser.add_argument("--belief-recovery-yaw-gain", type=float, default=1.5)
# #     argument_parser.add_argument(
# #         "--belief-recovery-maximum-yaw-rate", type=float, default=0.70
# #     )
# #     argument_parser.add_argument(
# #         "--belief-recovery-maximum-forward-speed", type=float, default=0.12
# #     )
# #     argument_parser.add_argument(
# #         "--interactive-return-home",
# #         action=argparse.BooleanOptionalAction,
# #         default=False,
# #         help=(
# #             "At the outward goal, ask for a command, let Qwen classify RETURN "
# #             "or STOP, and use a separately propagated spawn/home PixelGoal belief."
# #         ),
# #     )
# #     argument_parser.add_argument(
# #         "--return-command",
# #         default=None,
# #         help="Optional non-interactive command text, for example 'come back'.",
# #     )
# #     argument_parser.add_argument(
# #         "--qwen-freeform-mission",
# #         action=argparse.BooleanOptionalAction,
# #         default=False,
# #         help=(
# #             "Ask once at startup for a free-form instruction. Qwen emits either "
# #             "GO_TO_GOAL or GO_TO_GOAL followed by RETURN_HOME."
# #         ),
# #     )
# #     argument_parser.add_argument(
# #         "--mission-command",
# #         default=None,
# #         help=(
# #             "Optional non-interactive free-form mission, for example "
# #             "'visit the target and report back'."
# #         ),
# #     )
# #     argument_parser.add_argument(
# #         "--return-goal-obstacle-activation-distance", type=float, default=1.35
# #     )
# #     argument_parser.add_argument(
# #         "--return-goal-obstacle-dilation-pixels", type=int, default=30
# #     )
# #     argument_parser.add_argument(
# #         "--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct"
# #     )
# #     argument_parser.add_argument("--qwen-device", default="auto")
# #     argument_parser.add_argument("--qwen-homotopy-python", default=sys.executable)
# #     argument_parser.add_argument("--qwen-homotopy-host", default="127.0.0.1")
# #     argument_parser.add_argument("--qwen-homotopy-port", type=int, default=8890)
# #     argument_parser.add_argument("--qwen-homotopy-timeout", type=float, default=600.0)
# #     argument_parser.add_argument(
# #         "--start-qwen-homotopy-server",
# #         action=argparse.BooleanOptionalAction,
# #         default=True,
# #     )
# #     argument_parser.add_argument(
# #         "--qwen-homotopy",
# #         action=argparse.BooleanOptionalAction,
# #         default=False,
# #         help=(
# #             "When a metric obstacle becomes relevant, Qwen chooses the single "
# #             "LEFT/RIGHT circulation sign used by every trajectory candidate."
# #         ),
# #     )
# #     argument_parser.add_argument(
# #         "--homotopy-minimum-obstacle-pixels", type=int, default=30
# #     )
# #     argument_parser.add_argument("--homotopy-release-clear-frames", type=int, default=8)
# #     argument_parser.add_argument(
# #         "--homotopy-consistency-repeats",
# #         type=int,
# #         default=5,
# #         help="Repeat Qwen on the identical obstacle frame and use majority vote.",
# #     )
# #     argument_parser.add_argument(
# #         "--remove-critic", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument("--seed", type=int, default=7)
# #     argument_parser.add_argument(
# #         "--start-server", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument("--server-host", default="127.0.0.1")
# #     argument_parser.add_argument("--server-port", type=int, default=8888)
# #     argument_parser.add_argument("--server-timeout", type=float, default=180.0)
# #     argument_parser.add_argument("--candidates", type=int, default=16)
# #     argument_parser.add_argument("--particles", type=int, default=8)
# #     argument_parser.add_argument("--particle-std", type=float, default=0.22)
# #     argument_parser.add_argument("--gradient-steps", type=int, default=3)
# #     argument_parser.add_argument("--gradient-step-size", type=float, default=0.04)
# #     argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
# #     argument_parser.add_argument("--temperature", type=float, default=0.35)
# #     argument_parser.add_argument(
# #         "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument(
# #         "--particle-energy-reweighting",
# #         action=argparse.BooleanOptionalAction,
# #         default=True,
# #     )
# #     argument_parser.add_argument(
# #         "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument(
# #         "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument(
# #         "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument("--safe-distance", type=float, default=0.42)
# #     argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
# #     argument_parser.add_argument("--safety-weight", type=float, default=35.0)
# #     argument_parser.add_argument("--barrier-weight", type=float, default=25.0)
# #     argument_parser.add_argument("--barrier-rate", type=float, default=0.15)
# #     argument_parser.add_argument("--circulation-weight", type=float, default=18.0)
# #     argument_parser.add_argument(
# #         "--circulation-activation-distance", type=float, default=1.50
# #     )
# #     argument_parser.add_argument(
# #         "--circulation-activation-sharpness", type=float, default=0.20
# #     )
# #     argument_parser.add_argument(
# #         "--minimum-circulation-progress", type=float, default=0.025
# #     )
# #     argument_parser.add_argument(
# #         "--blocking-alignment-threshold", type=float, default=0.25
# #     )
# #     argument_parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
# #     argument_parser.add_argument("--escape-lateral-target", type=float, default=0.35)
# #     argument_parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
# #     argument_parser.add_argument("--maximum-obstacle-depth", type=float, default=5.0)
# #     argument_parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)

# #     argument_parser.add_argument("--scene", required=True)
# #     argument_parser.add_argument("--terrain-obj", default=None)
# #     argument_parser.add_argument("--heightmap", default=None)
# #     argument_parser.add_argument(
# #         "--terrain-height-mode",
# #         choices=["auto", "heightmap", "obj", "flat"],
# #         default="auto",
# #     )
# #     argument_parser.add_argument("--flat-y", type=float, default=0.0)
# #     argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
# #     argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
# #     argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
# #     argument_parser.add_argument("--flip-heightmap-x", action="store_true")
# #     argument_parser.add_argument(
# #         "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument("--swap-heightmap-xz", action="store_true")
# #     argument_parser.add_argument("--clearance", type=float, default=1.4)
# #     argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)
# #     argument_parser.add_argument(
# #         "--robot-radius",
# #         type=float,
# #         default=0.24,
# #         help="Planar rover footprint radius used by both guidance and evaluation.",
# #     )
# #     argument_parser.add_argument(
# #         "--evaluation-layout",
# #         default="default",
# #         help="Stable layout identifier stored in the rollout archive.",
# #     )

# #     argument_parser.add_argument("--height", type=int, default=720)
# #     argument_parser.add_argument("--width", type=int, default=720)
# #     argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
# #     argument_parser.add_argument("--hz", type=float, default=10.0)
# #     argument_parser.add_argument("--max-steps", type=int, default=300)
# #     argument_parser.add_argument("--stop-distance", type=float, default=1.0)
# #     argument_parser.add_argument("--start-x", type=float, default=0.0)
# #     argument_parser.add_argument("--start-z", type=float, default=8.0)
# #     argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
# #     argument_parser.add_argument("--goal-x", type=float, default=None)
# #     argument_parser.add_argument("--goal-z", type=float, default=None)
# #     argument_parser.add_argument("--goal-y", type=float, default=None)
# #     argument_parser.add_argument("--goal-height", type=float, default=1.2)
# #     argument_parser.add_argument("--goal-radius", type=int, default=18)
# #     argument_parser.add_argument(
# #         "--goal-mesh", action=argparse.BooleanOptionalAction, default=False
# #     )
# #     argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
# #     argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)

# #     argument_parser.add_argument(
# #         "--obstacle-mode", choices=["none", "depth", "mesh", "ghost"], default="none"
# #     )
# #     argument_parser.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
# #     argument_parser.add_argument("--obstacle-min-y-fraction", type=float, default=0.45)
# #     argument_parser.add_argument("--ghost-obstacle-x", type=float, default=None)
# #     argument_parser.add_argument("--ghost-obstacle-z", type=float, default=None)
# #     argument_parser.add_argument("--ghost-obstacle-y", type=float, default=None)
# #     argument_parser.add_argument("--ghost-obstacle-height", type=float, default=0.45)
# #     argument_parser.add_argument("--ghost-obstacle-radius", type=int, default=24)
# #     argument_parser.add_argument(
# #         "--obstacle-mesh-uv",
# #         nargs="+",
# #         default=[],
# #         help=(
# #             "Actual rendered obstacle mesh locations as image fractions u,v. "
# #             "Example: --obstacle-mesh-uv 0.50,0.72 0.30,0.68"
# #         ),
# #     )
# #     argument_parser.add_argument(
# #         "--obstacle-world-xz",
# #         nargs="*",
# #         default=[],
# #         metavar="X,Z",
# #         help=(
# #             "Static rendered obstacle-box centers in world X,Z coordinates. "
# #             "Example: --obstacle-world-xz 0,0. Do not combine with "
# #             "--obstacle-mesh-uv."
# #         ),
# #     )
# #     argument_parser.add_argument(
# #         "--obstacle-world-xz-item",
# #         action="append",
# #         default=[],
# #         metavar="X,Z",
# #         help=(
# #             "Repeatable form that safely accepts negative coordinates, e.g. "
# #             "--obstacle-world-xz-item=-3,0."
# #         ),
# #     )
# #     argument_parser.add_argument(
# #         "--world-obstacle-half-extent", type=float, default=0.75
# #     )
# #     argument_parser.add_argument("--world-obstacle-height", type=float, default=1.40)
# #     argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
# #     argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)
# #     argument_parser.add_argument(
# #         "--obstacle-velocity-xz",
# #         nargs="*",
# #         default=[],
# #         metavar="VX,VZ",
# #         help=(
# #             "World-frame mesh velocities in m/s. Supply one value to broadcast "
# #             "or one value per obstacle. Example: --obstacle-velocity-xz 0.30,0.0"
# #         ),
# #     )

# #     argument_parser.add_argument("--lookahead-index", type=int, default=4)
# #     argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
# #     argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
# #     argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
# #     argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
# #     argument_parser.add_argument("--save-every", type=int, default=1)
# #     argument_parser.add_argument(
# #         "--save-frames", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument(
# #         "--save-video", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     argument_parser.add_argument(
# #         "--archive-observations",
# #         action=argparse.BooleanOptionalAction,
# #         default=True,
# #         help="Store RGB/depth/masks in rollout.npz; disable for large evaluations.",
# #     )
# #     argument_parser.add_argument(
# #         "--overlay-masks", action=argparse.BooleanOptionalAction, default=True
# #     )
# #     return argument_parser


# # def main() -> None:
# #     args = parser().parse_args()
# #     if args.obstacle_world_xz_item:
# #         args.obstacle_world_xz.extend(args.obstacle_world_xz_item)
# #     np.random.seed(args.seed)
# #     if args.goal_x is None or args.goal_z is None:
# #         raise ValueError("fixed PointGoal requires --goal-x and --goal-z")
# #     if args.belief_pixel_goal and args.goal_mode != "pixel":
# #         raise ValueError("--belief-pixel-goal requires --goal-mode pixel")
# #     if args.belief_pixel_goal and not args.goal_mesh:
# #         raise ValueError(
# #             "simulation belief tracking requires --goal-mesh so a live semantic "
# #             "goal observation exists"
# #         )
# #     if args.belief_minimum_goal_pixels < 1:
# #         raise ValueError("belief-minimum-goal-pixels must be positive")
# #     return_home_enabled = args.interactive_return_home or args.qwen_freeform_mission
# #     if return_home_enabled and not args.belief_pixel_goal:
# #         raise ValueError("return-home modes require --belief-pixel-goal")
# #     if return_home_enabled and not args.qwen_homotopy:
# #         raise ValueError("return-home modes require --qwen-homotopy")
# #     if args.mission_command is not None and not args.qwen_freeform_mission:
# #         raise ValueError("--mission-command requires --qwen-freeform-mission")
# #     if args.return_goal_obstacle_activation_distance <= 0.0:
# #         raise ValueError("return goal obstacle activation distance must be positive")
# #     if args.return_goal_obstacle_dilation_pixels < 0:
# #         raise ValueError("return goal obstacle dilation pixels must be non-negative")
# #     if (
# #         min(
# #             args.belief_measurement_std,
# #             args.belief_translation_process_std,
# #             args.belief_yaw_process_std_deg,
# #             args.belief_bootstrap_std,
# #         )
# #         < 0.0
# #     ):
# #         raise ValueError("belief uncertainty parameters must be non-negative")
# #     if args.robot_radius < 0.0:
# #         raise ValueError("robot-radius must be non-negative")
# #     if args.obstacle_velocity_xz and args.obstacle_mode != "mesh":
# #         raise ValueError("moving obstacle velocities require --obstacle-mode mesh")
# #     if args.obstacle_mode == "ghost" and (
# #         args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
# #     ):
# #         raise ValueError(
# #             "ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z"
# #         )
# #     if args.obstacle_mesh_uv and args.obstacle_world_xz:
# #         raise ValueError(
# #             "choose either --obstacle-mesh-uv or --obstacle-world-xz, not both"
# #         )
# #     if args.obstacle_mode == "mesh" and not (
# #         args.obstacle_mesh_uv or args.obstacle_world_xz
# #     ):
# #         raise ValueError(
# #             "mesh mode requires --obstacle-world-xz X,Z [X,Z ...] or "
# #             "--obstacle-mesh-uv u,v [u,v ...]"
# #         )
# #     if args.world_obstacle_half_extent <= 0.0 or args.world_obstacle_height <= 0.0:
# #         raise ValueError("world obstacle dimensions must be positive")
# #     if args.goal_mesh_half_extent <= 0.0 or args.goal_mesh_height <= 0.0:
# #         raise ValueError("goal mesh dimensions must be positive")

# #     if args.qwen_homotopy and args.planner_mode == "pure-navdp":
# #         raise ValueError("Qwen homotopy conditioning requires s2diff or gradient mode")

# #     qwen_process: Optional[subprocess.Popen[Any]] = None
# #     server_process: Optional[subprocess.Popen[Any]] = None
# #     simulator = None
# #     try:
# #         qwen_process = start_qwen_homotopy_server(args)
# #         homotopy_selector = None
# #         if args.qwen_homotopy:
# #             homotopy_selector = QwenHomotopyClient(
# #                 f"http://{args.qwen_homotopy_host}:{args.qwen_homotopy_port}",
# #                 timeout=args.qwen_homotopy_timeout,
# #             )
# #             homotopy_selector.reset()
# #         server_process = start_server(args)
# #         server_url = f"http://{args.server_host}:{args.server_port}"
# #         client = NavDPS2DiffClient(server_url)
# #         algorithm = client.reset(
# #             camera_intrinsic(args.height, args.width, args.hfov_deg),
# #             batch_size=1,
# #             stop_threshold=-3.0,
# #         )
# #         supported_algorithms = {
# #             "navdp-s2diff-pixels",
# #             "navdp-hlc-s2diff",
# #             "navdp-hlc-s2diff-no-critic",
# #             "navdp-hlc-gradient",
# #             "navdp-hlc-gradient-no-critic",
# #             "navdp-pure-critic",
# #         }
# #         if algorithm not in supported_algorithms:
# #             raise RuntimeError(f"unexpected planner response: {algorithm!r}")

# #         terrain = TerrainHeight(
# #             mode=args.terrain_height_mode,
# #             heightmap=(
# #                 Path(args.heightmap).expanduser().resolve() if args.heightmap else None
# #             ),
# #             obj=(
# #                 Path(args.terrain_obj).expanduser().resolve()
# #                 if args.terrain_obj
# #                 else None
# #             ),
# #             flat_y=args.flat_y,
# #             size_x=args.size_x,
# #             size_z=args.size_z,
# #             size_y=args.size_y,
# #             flip_x=args.flip_heightmap_x,
# #             flip_z=args.flip_heightmap_z,
# #             swap_xz=args.swap_heightmap_xz,
# #         )
# #         output_directory = Path(args.output).expanduser().resolve()
# #         frame_directory = output_directory / "frames"
# #         frame_directory.mkdir(parents=True, exist_ok=True)

# #         simulator = make_simulator(
# #             Path(args.scene),
# #             args.height,
# #             args.width,
# #             args.hfov_deg,
# #             with_semantic=args.obstacle_mode == "mesh" or args.goal_mesh,
# #         )
# #         agent = simulator.initialize_agent(0)
# #         intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
# #         goal_belief = (
# #             GaussianGoalBelief(
# #                 intrinsic,
# #                 (args.height, args.width),
# #                 minimum_visible_pixels=args.belief_minimum_goal_pixels,
# #                 measurement_std=args.belief_measurement_std,
# #                 translation_process_std=args.belief_translation_process_std,
# #                 yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
# #             )
# #             if args.belief_pixel_goal
# #             else None
# #         )
# #         previous_executed_action = np.zeros(3, dtype=np.float32)
# #         home_belief = (
# #             GaussianGoalBelief(
# #                 intrinsic,
# #                 (args.height, args.width),
# #                 minimum_visible_pixels=args.belief_minimum_goal_pixels,
# #                 measurement_std=args.belief_measurement_std,
# #                 translation_process_std=args.belief_translation_process_std,
# #                 yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
# #             )
# #             if return_home_enabled
# #             else None
# #         )
# #         if home_belief is not None:
# #             # Home starts at the rover origin. Every executed action propagates
# #             # this stationary world location into the current body frame.
# #             home_belief.initialize(
# #                 np.zeros(2, dtype=np.float32),
# #                 args.belief_measurement_std,
# #                 visible=False,
# #             )
# #         mission_phase = "OUTBOUND"
# #         return_goal_obstacle_active = False
# #         return_command_event: Optional[dict[str, Any]] = None
# #         mission_plan_event: Optional[dict[str, Any]] = None
# #         mission_plan_pending = bool(args.qwen_freeform_mission)
# #         automatic_return_requested = False
# #         roundtrip_completed = False
# #         x, z = float(args.start_x), float(args.start_z)
# #         yaw = math.radians(float(args.start_yaw_deg))
# #         dt = 1.0 / float(args.hz)

# #         goal_y = args.goal_y
# #         if goal_y is None:
# #             goal_y = (
# #                 terrain.local_height_max(args.goal_x, args.goal_z, 0.8)
# #                 + args.goal_height
# #             )
# #         goal = np.asarray([args.goal_x, goal_y, args.goal_z], dtype=np.float32)
# #         start_position_xz = np.asarray([x, z], dtype=np.float64)
# #         initial_goal_distance = float(
# #             np.linalg.norm(goal[[0, 2]].astype(np.float64) - start_position_xz)
# #         )
# #         goal_mesh_object = None
# #         if args.goal_mesh:
# #             goal_mesh_object = place_world_goal_mesh(
# #                 simulator,
# #                 terrain,
# #                 args.goal_x,
# #                 args.goal_z,
# #                 output_directory,
# #                 half_extent=args.goal_mesh_half_extent,
# #                 height=args.goal_mesh_height,
# #             )

# #         ghost = None
# #         if args.obstacle_mode == "ghost":
# #             ghost_y = args.ghost_obstacle_y
# #             if ghost_y is None:
# #                 ghost_y = (
# #                     terrain.local_height_max(
# #                         args.ghost_obstacle_x,
# #                         args.ghost_obstacle_z,
# #                         args.pose_terrain_radius,
# #                     )
# #                     + args.ghost_obstacle_height
# #                 )
# #             ghost = np.asarray(
# #                 [args.ghost_obstacle_x, ghost_y, args.ghost_obstacle_z],
# #                 dtype=np.float32,
# #             )

# #         mesh_objects: list[Any] = []
# #         mesh_centroids: list[np.ndarray] = []
# #         mesh_current_centroids: list[np.ndarray] = []
# #         mesh_base_geometries: list[np.ndarray] = []
# #         mesh_geometries: list[np.ndarray] = []
# #         mesh_velocities = np.zeros((0, 2), dtype=np.float64)
# #         mesh_placed = False
# #         if args.obstacle_mode == "mesh" and args.obstacle_world_xz:
# #             mesh_objects, mesh_centroids, mesh_base_geometries = (
# #                 place_world_obstacle_meshes(
# #                     simulator,
# #                     terrain,
# #                     args.obstacle_world_xz,
# #                     output_directory,
# #                     half_extent=args.world_obstacle_half_extent,
# #                     height=args.world_obstacle_height,
# #                 )
# #             )
# #             mesh_velocities = expand_obstacle_velocities(
# #                 args.obstacle_velocity_xz, len(mesh_objects)
# #             )
# #             mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# #                 mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
# #             )
# #             mesh_placed = True

# #         row_keys = [
# #             "pose",
# #             "action_3d",
# #             "mission_phase",
# #             "return_goal_obstacle_active",
# #             "point_goal",
# #             "belief_goal_mu",
# #             "belief_goal_covariance",
# #             "belief_goal_pixel",
# #             "belief_goal_visible",
# #             "belief_goal_source",
# #             "belief_goal_time_since_seen",
# #             "belief_goal_bearing_rad",
# #             "belief_goal_pixel_sigma",
# #             "belief_heading_recovery_active",
# #             "selected_trajectory",
# #             "all_trajectories",
# #             "all_values",
# #             "selected_index",
# #             "fallback_stop",
# #             "escape_turn",
# #             "valid_obstacle_points",
# #             "selected_circulation_sign",
# #             "candidate_circulation_signs",
# #             "selected_barrier_energy",
# #             "selected_circulation_energy",
# #             "planning_time_seconds",
# #             "selected_minimum_clearance",
# #             "mean_guidance_noise_correction",
# #             "final_guidance_noise_correction",
# #             "maximum_guidance_noise_correction",
# #             "mean_final_effective_sample_size",
# #             "goal_distance",
# #             "executed_center_clearance",
# #             "executed_surface_clearance",
# #             "geometric_collision",
# #             "obstacle_positions_world",
# #             "qwen_homotopy_sign",
# #             "qwen_homotopy_side",
# #             "qwen_homotopy_confidence",
# #             "qwen_homotopy_queried",
# #         ]
# #         if args.archive_observations:
# #             row_keys.extend(
# #                 (
# #                     "rgb",
# #                     "depth",
# #                     "goal_mask",
# #                     "live_goal_mask",
# #                     "ghost_goal_mask",
# #                     "obstacle_mask",
# #                 )
# #             )
# #         rows: dict[str, list[Any]] = {key: [] for key in row_keys}
# #         video_frames: list[Image.Image] = []
# #         success = False
# #         homotopy_events: list[dict[str, Any]] = []

# #         for step in range(int(args.max_steps)):
# #             y = (
# #                 terrain.local_height_max(x, z, args.pose_terrain_radius)
# #                 + args.clearance
# #             )
# #             position = np.asarray([x, y, z], dtype=np.float32)
# #             if goal_belief is not None and step > 0:
# #                 goal_belief.predict(previous_executed_action, dt)
# #             if home_belief is not None and step > 0:
# #                 home_belief.predict(previous_executed_action, dt)
# #             set_agent_pose(agent, position, yaw)
# #             if mesh_placed:
# #                 elapsed_seconds = step * dt
# #                 move_mesh_objects(mesh_objects, mesh_velocities, elapsed_seconds)
# #                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# #                     mesh_base_geometries,
# #                     mesh_centroids,
# #                     mesh_velocities,
# #                     elapsed_seconds,
# #                 )
# #             observation = simulator.get_sensor_observations()
# #             rgb, depth = rgb_depth(observation)

# #             if args.obstacle_mode == "mesh" and not mesh_placed:
# #                 mesh_objects, mesh_centroids, mesh_base_geometries = (
# #                     place_obstacle_meshes(
# #                         simulator,
# #                         depth,
# #                         position,
# #                         yaw,
# #                         intrinsic,
# #                         args.obstacle_mesh_uv,
# #                         output_directory,
# #                         mesh_half_pixels=args.mesh_half_pixels,
# #                         mesh_lift=args.mesh_obstacle_lift,
# #                     )
# #                 )
# #                 mesh_velocities = expand_obstacle_velocities(
# #                     args.obstacle_velocity_xz, len(mesh_objects)
# #                 )
# #                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
# #                     mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
# #                 )
# #                 mesh_placed = True
# #                 observation = simulator.get_sensor_observations()
# #                 rgb, depth = rgb_depth(observation)

# #             if mission_plan_pending:
# #                 mission_command = args.mission_command
# #                 if mission_command is None:
# #                     print(
# #                         "\nWhat should the rover do? You may use a vague command, "
# #                         "for example: 'visit the goal and come back'.",
# #                         flush=True,
# #                     )
# #                     try:
# #                         mission_command = input("> ").strip()
# #                     except EOFError as error:
# #                         raise RuntimeError(
# #                             "no startup command was available; set "
# #                             "--mission-command for a non-interactive run"
# #                         ) from error
# #                 assert homotopy_selector is not None
# #                 mission_decision = homotopy_selector.classify_mission(
# #                     rgb, mission_command
# #                 )
# #                 automatic_return_requested = mission_decision.plan == (
# #                     "GO_TO_GOAL",
# #                     "RETURN_HOME",
# #                 )
# #                 mission_plan_event = {
# #                     "step": step,
# #                     "user_command": mission_command,
# #                     "plan": list(mission_decision.plan),
# #                     "confidence": mission_decision.confidence,
# #                     "raw_response": mission_decision.raw_response,
# #                 }
# #                 Image.fromarray(rgb).save(output_directory / "qwen_mission_frame.png")
# #                 print(
# #                     f"[qwen-mission] text={mission_command!r} "
# #                     f"plan={list(mission_decision.plan)} "
# #                     f"confidence={mission_decision.confidence:.2f}",
# #                     flush=True,
# #                 )
# #                 mission_plan_pending = False

# #             semantic = (
# #                 semantic_from_observation(observation)
# #                 if args.obstacle_mode == "mesh" or args.goal_mesh
# #                 else None
# #             )
# #             goal_right, _goal_up, goal_forward = camera_coordinates(goal, position, yaw)
# #             point_goal = np.asarray(
# #                 [max(goal_forward, 0.0), -goal_right], dtype=np.float32
# #             )
# #             live_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
# #             ghost_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
# #             belief_goal_visible = False
# #             belief_goal_source = "DISABLED"
# #             belief_goal_mu = np.full(2, np.nan, dtype=np.float32)
# #             belief_goal_covariance = np.full((2, 2), np.nan, dtype=np.float32)
# #             belief_goal_pixel = np.full(2, -1, dtype=np.int32)
# #             belief_goal_time_since_seen = float("nan")
# #             belief_goal_bearing = float("nan")
# #             belief_goal_pixel_sigma = float("nan")

# #             if goal_belief is not None:
# #                 assert semantic is not None
# #                 live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
# #                 bootstrapped = False
# #                 if mission_phase == "OUTBOUND":
# #                     active_goal_belief = goal_belief
# #                     belief_goal_visible = goal_belief.observe(live_goal_mask, depth)
# #                     if not goal_belief.initialized:
# #                         if not args.belief_bootstrap_world_goal:
# #                             raise RuntimeError(
# #                                 "goal belief is uninitialized because the live goal "
# #                                 "mask has not been observed; start with the goal visible "
# #                                 "or pass --belief-bootstrap-world-goal for simulation"
# #                             )
# #                         goal_belief.initialize(
# #                             np.asarray([goal_forward, -goal_right], dtype=np.float32),
# #                             args.belief_bootstrap_std,
# #                         )
# #                         bootstrapped = True
# #                 else:
# #                     assert home_belief is not None and home_belief.initialized
# #                     active_goal_belief = home_belief
# #                     belief_goal_visible = False

# #                 belief_projection = active_goal_belief.project(
# #                     base_radius=args.belief_ghost_base_radius,
# #                     covariance_scale=args.belief_ghost_covariance_scale,
# #                     maximum_radius=args.belief_ghost_maximum_radius,
# #                 )
# #                 planner_goal = belief_projection.pixel_uv
# #                 ghost_goal_mask = belief_projection.mask
# #                 if mission_phase == "OUTBOUND" and belief_goal_visible:
# #                     goal_mask = live_goal_mask
# #                     belief_goal_source = "LIVE"
# #                 else:
# #                     goal_mask = ghost_goal_mask
# #                     belief_goal_source = (
# #                         "HOME_BELIEF"
# #                         if mission_phase == "RETURN_HOME"
# #                         else ("WORLD_BOOTSTRAP" if bootstrapped else "GHOST")
# #                     )
# #                 assert (
# #                     active_goal_belief.mu is not None
# #                     and active_goal_belief.Sigma is not None
# #                 )
# #                 belief_goal_mu = active_goal_belief.mu.copy()
# #                 belief_goal_covariance = active_goal_belief.Sigma.copy()
# #                 belief_goal_pixel = belief_projection.pixel_uv.copy()
# #                 belief_goal_time_since_seen = active_goal_belief.time_since_seen
# #                 belief_goal_bearing = belief_projection.bearing_rad
# #                 belief_goal_pixel_sigma = belief_projection.pixel_sigma
# #             else:
# #                 if args.goal_mesh:
# #                     assert semantic is not None
# #                     live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
# #                     goal_mask = live_goal_mask
# #                     if not np.any(goal_mask):
# #                         goal_mask, _ = project_world_mask(
# #                             goal,
# #                             position,
# #                             yaw,
# #                             intrinsic,
# #                             args.height,
# #                             args.width,
# #                             args.goal_radius,
# #                         )
# #                 else:
# #                     goal_mask, _ = project_world_mask(
# #                         goal,
# #                         position,
# #                         yaw,
# #                         intrinsic,
# #                         args.height,
# #                         args.width,
# #                         args.goal_radius,
# #                     )
# #                 planner_goal = point_goal
# #             if args.goal_mode == "pixel" and goal_belief is None:
# #                 planner_goal = world_goal_to_pixel(
# #                     goal, position, yaw, intrinsic, args.height, args.width
# #                 )
# #                 goal_mask = circle_mask(
# #                     args.height,
# #                     args.width,
# #                     planner_goal[0],
# #                     planner_goal[1],
# #                     args.goal_radius,
# #                 )
# #             guidance_depth = depth.copy()
# #             if args.obstacle_mode == "depth":
# #                 obstacle_mask = depth_obstacle_mask(
# #                     depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
# #                 )
# #             elif args.obstacle_mode == "mesh":
# #                 assert semantic is not None
# #                 semantic_ids = list(
# #                     range(
# #                         MESH_OBSTACLE_ID,
# #                         MESH_OBSTACLE_ID + len(mesh_objects),
# #                     )
# #                 )
# #                 obstacle_mask = np.isin(semantic, semantic_ids).astype(np.uint8)
# #                 # The depth image was re-rendered after mesh placement, so
# #                 # guidance_depth already contains the real obstacle depth.
# #             elif args.obstacle_mode == "ghost":
# #                 assert ghost is not None
# #                 obstacle_mask, obstacle_forward = project_world_mask(
# #                     ghost,
# #                     position,
# #                     yaw,
# #                     intrinsic,
# #                     args.height,
# #                     args.width,
# #                     args.ghost_obstacle_radius,
# #                 )
# #                 if obstacle_forward > 0.05:
# #                     guidance_depth[obstacle_mask > 0] = obstacle_forward
# #             else:
# #                 obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)

# #             # Replace this mask-to-pixels line with your own detector's [u,v]
# #             if mission_phase == "RETURN_HOME":
# #                 distance_from_reached_goal = float(
# #                     np.linalg.norm(goal[[0, 2]] - position[[0, 2]])
# #                 )
# #                 if (
# #                     not return_goal_obstacle_active
# #                     and distance_from_reached_goal
# #                     >= args.return_goal_obstacle_activation_distance
# #                 ):
# #                     return_goal_obstacle_active = True
# #                     print(
# #                         "[roundtrip] reached-goal keep-out is now active",
# #                         flush=True,
# #                     )
# #                 if return_goal_obstacle_active and np.any(live_goal_mask):
# #                     reached_goal_keepout = dilate_binary_mask(
# #                         live_goal_mask,
# #                         args.return_goal_obstacle_dilation_pixels,
# #                     )
# #                     obstacle_mask = (
# #                         (obstacle_mask > 0) | (reached_goal_keepout > 0)
# #                     ).astype(np.uint8)
# #                     target_depths = depth[
# #                         (live_goal_mask > 0)
# #                         & np.isfinite(depth)
# #                         & (depth > args.minimum_obstacle_depth)
# #                     ]
# #                     if target_depths.size:
# #                         guidance_depth[reached_goal_keepout > 0] = float(
# #                             np.median(target_depths)
# #                         )

# #             # array if obstacle pixels already come directly from your system.
# #             obstacle_pixels = pixels_from_mask(
# #                 obstacle_mask, args.maximum_obstacle_pixels
# #             )
# #             homotopy_decision = None
# #             forced_circulation_sign = 0.0
# #             obstacle_relevant_for_homotopy = False
# #             if homotopy_selector is not None:
# #                 homotopy_obstacle_mask = (
# #                     (obstacle_mask > 0)
# #                     & np.isfinite(guidance_depth)
# #                     & (guidance_depth >= args.minimum_obstacle_depth)
# #                     & (guidance_depth <= args.maximum_obstacle_depth)
# #                 ).astype(np.uint8)
# #                 qwen_overlay = overlay_frame(
# #                     rgb,
# #                     goal_mask,
# #                     homotopy_obstacle_mask,
# #                     "Qwen homotopy: choose LEFT or RIGHT",
# #                     show_masks=True,
# #                 )
# #                 homotopy_decision = homotopy_selector.step(
# #                     np.asarray(qwen_overlay.convert("RGB")), homotopy_obstacle_mask
# #                 )
# #                 obstacle_relevant_for_homotopy = homotopy_decision.obstacle_relevant
# #                 forced_circulation_sign = homotopy_decision.circulation_sign
# #                 if homotopy_decision.queried_qwen:
# #                     event = {
# #                         "step": step,
# #                         "side": homotopy_decision.side,
# #                         "circulation_sign": forced_circulation_sign,
# #                         "confidence": homotopy_decision.confidence,
# #                         "repeat_sides": list(homotopy_decision.repeated_sides),
# #                         "repeat_confidences": list(
# #                             homotopy_decision.repeated_confidences
# #                         ),
# #                         "consistency_rate": homotopy_decision.consistency_rate,
# #                         "used_fallback": homotopy_decision.used_fallback,
# #                         "raw_response": homotopy_decision.raw_response,
# #                     }
# #                     homotopy_events.append(event)
# #                     query_directory = output_directory / "qwen_homotopy_queries"
# #                     query_directory.mkdir(parents=True, exist_ok=True)
# #                     qwen_overlay.save(query_directory / f"query_step_{step:04d}.png")
# #                     print(
# #                         f"[qwen-homotopy] side={homotopy_decision.side} "
# #                         f"sign={forced_circulation_sign:+.0f} "
# #                         f"confidence={homotopy_decision.confidence:.2f} "
# #                         f"consistency={homotopy_decision.consistency_rate:.2%} "
# #                         f"repeats={list(homotopy_decision.repeated_sides)} "
# #                         f"fallback={homotopy_decision.used_fallback}",
# #                         flush=True,
# #                     )
# #             planning_start = time.perf_counter()
# #             result = client.plan(
# #                 goal_xy=planner_goal,
# #                 rgb=rgb,
# #                 depth=guidance_depth,
# #                 obstacle_pixels=obstacle_pixels,
# #                 goal_mode=args.goal_mode,
# #                 forced_circulation_sign=forced_circulation_sign,
# #             )
# #             planning_time = time.perf_counter() - planning_start
# #             action = (
# #                 np.zeros(3, dtype=np.float32)
# #                 if result.fallback_stop
# #                 else waypoint_action(
# #                     result.trajectory,
# #                     lookahead_index=args.lookahead_index,
# #                     maximum_forward_speed=args.maximum_forward_speed,
# #                     maximum_yaw_rate=args.maximum_yaw_rate,
# #                     yaw_gain=args.yaw_gain,
# #                 )
# #             )
# #             action, belief_recovery_active = belief_heading_recovery_action(
# #                 action,
# #                 belief_bearing=belief_goal_bearing,
# #                 obstacle_relevant=obstacle_relevant_for_homotopy,
# #                 enabled=args.belief_pixel_goal and args.belief_heading_recovery,
# #                 activation_bearing=math.radians(args.belief_recovery_bearing_deg),
# #                 yaw_gain=args.belief_recovery_yaw_gain,
# #                 maximum_yaw_rate=args.belief_recovery_maximum_yaw_rate,
# #                 maximum_forward_speed=args.belief_recovery_maximum_forward_speed,
# #             )

# #             next_position, next_yaw = integrate_mars(position, yaw, action, dt)
# #             previous_executed_action = action.copy()
# #             x = float(
# #                 np.clip(
# #                     next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
# #                 )
# #             )
# #             z = float(
# #                 np.clip(
# #                     next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
# #                 )
# #             )
# #             rows["mission_phase"].append(mission_phase)
# #             rows["return_goal_obstacle_active"].append(return_goal_obstacle_active)

# #             yaw = wrap_angle(next_yaw)
# #             outbound_goal_distance = float(
# #                 np.linalg.norm(goal[[0, 2]] - np.asarray([x, z]))
# #             )
# #             home_distance = float(
# #                 np.linalg.norm(start_position_xz - np.asarray([x, z]))
# #             )
# #             goal_distance = (
# #                 home_distance
# #                 if mission_phase == "RETURN_HOME"
# #                 else outbound_goal_distance
# #             )
# #             center_clearance = planar_mesh_clearance(
# #                 np.asarray([x, z], dtype=np.float64), mesh_geometries
# #             )
# #             if np.isfinite(center_clearance):
# #                 surface_clearance = max(
# #                     center_clearance - float(args.robot_radius), 0.0
# #                 )
# #                 geometric_collision = center_clearance <= float(args.robot_radius)
# #             else:
# #                 surface_clearance = float("nan")
# #                 geometric_collision = False
# #             rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
# #             pose = np.asarray(
# #                 [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w],
# #                 dtype=np.float32,
# #             )

# #             if args.archive_observations:
# #                 rows["rgb"].append(rgb)
# #                 rows["depth"].append(depth)
# #                 rows["goal_mask"].append(goal_mask)
# #                 rows["live_goal_mask"].append(live_goal_mask)
# #                 rows["ghost_goal_mask"].append(ghost_goal_mask)
# #                 rows["obstacle_mask"].append(obstacle_mask)
# #             rows["pose"].append(pose)
# #             rows["action_3d"].append(action)
# #             rows["point_goal"].append(planner_goal)
# #             rows["belief_goal_mu"].append(belief_goal_mu)
# #             rows["belief_goal_covariance"].append(belief_goal_covariance)
# #             rows["belief_goal_pixel"].append(belief_goal_pixel)
# #             rows["belief_goal_visible"].append(belief_goal_visible)
# #             rows["belief_goal_source"].append(belief_goal_source)
# #             rows["belief_goal_time_since_seen"].append(belief_goal_time_since_seen)
# #             rows["belief_goal_bearing_rad"].append(belief_goal_bearing)
# #             rows["belief_goal_pixel_sigma"].append(belief_goal_pixel_sigma)
# #             rows["belief_heading_recovery_active"].append(belief_recovery_active)
# #             rows["selected_trajectory"].append(result.trajectory)
# #             rows["all_trajectories"].append(result.all_trajectories)
# #             rows["all_values"].append(result.all_values)
# #             rows["selected_index"].append(result.selected_index)
# #             rows["fallback_stop"].append(result.fallback_stop)
# #             rows["escape_turn"].append(result.escape_turn)
# #             rows["valid_obstacle_points"].append(result.valid_obstacle_points)
# #             rows["selected_circulation_sign"].append(result.selected_circulation_sign)
# #             rows["candidate_circulation_signs"].append(
# #                 result.candidate_circulation_signs
# #             )
# #             rows["selected_barrier_energy"].append(result.selected_barrier_energy)
# #             rows["selected_circulation_energy"].append(
# #                 result.selected_circulation_energy
# #             )
# #             rows["planning_time_seconds"].append(planning_time)
# #             rows["selected_minimum_clearance"].append(result.selected_minimum_clearance)
# #             rows["mean_guidance_noise_correction"].append(
# #                 result.mean_guidance_noise_correction
# #             )
# #             rows["final_guidance_noise_correction"].append(
# #                 result.final_guidance_noise_correction
# #             )
# #             rows["maximum_guidance_noise_correction"].append(
# #                 result.maximum_guidance_noise_correction
# #             )
# #             rows["mean_final_effective_sample_size"].append(
# #                 result.mean_final_effective_sample_size
# #             )
# #             rows["goal_distance"].append(goal_distance)
# #             rows["executed_center_clearance"].append(center_clearance)
# #             rows["executed_surface_clearance"].append(surface_clearance)
# #             rows["geometric_collision"].append(geometric_collision)
# #             rows["obstacle_positions_world"].append(
# #                 np.stack(mesh_current_centroids)
# #                 if mesh_current_centroids
# #                 else np.zeros((0, 3), dtype=np.float64)
# #             )

# #             rows["qwen_homotopy_sign"].append(forced_circulation_sign)
# #             rows["qwen_homotopy_side"].append(
# #                 homotopy_decision.side if homotopy_decision is not None else "AUTO"
# #             )
# #             rows["qwen_homotopy_confidence"].append(
# #                 homotopy_decision.confidence if homotopy_decision is not None else 0.0
# #             )
# #             rows["qwen_homotopy_queried"].append(
# #                 homotopy_decision.queried_qwen
# #                 if homotopy_decision is not None
# #                 else False
# #             )

# #             if args.save_frames and step % max(int(args.save_every), 1) == 0:

# #                 side_label = (
# #                     homotopy_decision.side if homotopy_decision is not None else "AUTO"
# #                 )
# #                 label = (
# #                     f"t={step} phase={mission_phase} goal={goal_distance:.2f}m "
# #                     f"qwen_side={side_label} pixels={len(obstacle_pixels)} "
# #                     f"goal_src={belief_goal_source} "
# #                     f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
# #                     f"recover={int(belief_recovery_active)} "
# #                     f"pred={result.selected_minimum_clearance:.2f}m "
# #                     f"actual={surface_clearance:.2f}m "
# #                     f"mode={result.selected_circulation_sign:+.0f} "
# #                     f"escape={int(result.escape_turn)} "
# #                     f"guide_rms={result.mean_guidance_noise_correction:.4f} "
# #                     f"v={action[0]:.2f} w={action[2]:.2f}"
# #                 )
# #                 frame = overlay_frame(
# #                     rgb,
# #                     goal_mask,
# #                     obstacle_mask,
# #                     label,
# #                     show_masks=args.overlay_masks,
# #                 )
# #                 frame.save(frame_directory / f"frame_{step:04d}.png")
# #                 video_frames.append(frame)

# #             print(
# #                 f"step={step:04d} phase={mission_phase} goal={goal_distance:.2f}m "
# #                 f"qwen_side={homotopy_decision.side if homotopy_decision else 'AUTO'} "
# #                 f"goal_src={belief_goal_source} "
# #                 f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
# #                 f"recover={int(belief_recovery_active)} "
# #                 f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
# #                 f"selected={result.selected_index} fallback={result.fallback_stop} "
# #                 f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
# #                 f"pred_clear={result.selected_minimum_clearance:.3f}m "
# #                 f"actual_clear={surface_clearance:.3f}m "
# #                 f"collision={geometric_collision} "
# #                 f"barrier={result.selected_barrier_energy:.5f} "
# #                 f"circ={result.selected_circulation_energy:.5f} "
# #                 f"latency={planning_time * 1000.0:.1f}ms "
# #                 f"guide_rms={result.mean_guidance_noise_correction:.6f} "
# #                 f"ess={result.mean_final_effective_sample_size:.2f} "
# #                 f"action={action.tolist()}",
# #                 flush=True,
# #             )
# #             if goal_distance <= args.stop_distance:
# #                 if args.qwen_freeform_mission and mission_phase == "OUTBOUND":
# #                     if automatic_return_requested:
# #                         print(
# #                             "[mission] outward goal reached; advancing automatically "
# #                             "to RETURN_HOME",
# #                             flush=True,
# #                         )
# #                         mission_phase = "RETURN_HOME"
# #                         assert homotopy_selector is not None
# #                         homotopy_selector.reset()
# #                         continue
# #                     success = True
# #                     print(
# #                         "[mission] outward goal reached; GO_TO_GOAL plan complete",
# #                         flush=True,
# #                     )
# #                     break

# #                 if args.interactive_return_home and mission_phase == "OUTBOUND":
# #                     user_command = args.return_command
# #                     if user_command is None:
# #                         print(
# #                             "\nOutward goal reached. What should the rover do? "
# #                             "(for example: come back / stop)",
# #                             flush=True,
# #                         )
# #                         try:
# #                             user_command = input("> ").strip()
# #                         except EOFError as error:
# #                             raise RuntimeError(
# #                                 "no interactive command was available; set "
# #                                 "--return-command 'come back' for a non-interactive run"
# #                             ) from error
# #                     assert homotopy_selector is not None
# #                     command_overlay = overlay_frame(
# #                         rgb,
# #                         goal_mask,
# #                         obstacle_mask,
# #                         "Qwen command: RETURN or STOP",
# #                         show_masks=True,
# #                     )
# #                     command_decision = homotopy_selector.classify_command(
# #                         np.asarray(command_overlay.convert("RGB")),
# #                         user_command,
# #                     )
# #                     return_command_event = {
# #                         "step": step,
# #                         "user_command": user_command,
# #                         "command": command_decision.command,
# #                         "confidence": command_decision.confidence,
# #                         "raw_response": command_decision.raw_response,
# #                     }
# #                     command_overlay.save(output_directory / "qwen_return_command.png")
# #                     print(
# #                         f"[qwen-command] text={user_command!r} "
# #                         f"decision={command_decision.command} "
# #                         f"confidence={command_decision.confidence:.2f}",
# #                         flush=True,
# #                     )
# #                     if command_decision.command == "RETURN":
# #                         mission_phase = "RETURN_HOME"
# #                         homotopy_selector.reset()
# #                         continue
# #                     success = True
# #                     break
# #                 success = True
# #                 if mission_phase == "RETURN_HOME":
# #                     roundtrip_completed = True
# #                 break

# #         if not rows["goal_distance"]:
# #             raise RuntimeError("rollout produced no steps")
# #         rollout_path = output_directory / "rollout.npz"
# #         np.savez_compressed(
# #             rollout_path,
# #             **{
# #                 key: (
# #                     np.stack(values)
# #                     if isinstance(values[0], np.ndarray)
# #                     else np.asarray(values)
# #                 )
# #                 for key, values in rows.items()
# #             },
# #             goal_position=goal,
# #             obstacle_position=(
# #                 mesh_centroids[0]
# #                 if mesh_centroids
# #                 else (
# #                     ghost
# #                     if ghost is not None
# #                     else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
# #                 )
# #             ),
# #             obstacle_positions=(
# #                 np.stack(mesh_centroids)
# #                 if mesh_centroids
# #                 else np.zeros((0, 3), dtype=np.float32)
# #             ),
# #             obstacle_velocity_xz=mesh_velocities,
# #             success=np.asarray(success),
# #             hz=np.asarray(args.hz, dtype=np.float32),
# #             start_position_xz=start_position_xz,
# #             initial_goal_distance=np.asarray(initial_goal_distance, dtype=np.float64),
# #             stop_distance=np.asarray(args.stop_distance, dtype=np.float64),
# #             robot_radius=np.asarray(args.robot_radius, dtype=np.float64),
# #             evaluation_layout=np.asarray(args.evaluation_layout),
# #             seed=np.asarray(args.seed, dtype=np.int64),
# #             goal_mode=np.asarray(args.goal_mode),
# #             belief_pixel_goal=np.asarray(args.belief_pixel_goal),
# #             interactive_return_home=np.asarray(args.interactive_return_home),
# #             qwen_freeform_mission=np.asarray(args.qwen_freeform_mission),
# #             automatic_return_requested=np.asarray(automatic_return_requested),
# #             roundtrip_completed=np.asarray(roundtrip_completed),
# #             final_mission_phase=np.asarray(mission_phase),
# #         )
# #         with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
# #             json.dump(
# #                 {
# #                     "success": success,
# #                     "steps": len(rows["goal_distance"]),
# #                     "archived_observations": args.archive_observations,
# #                     "final_goal_distance": float(rows["goal_distance"][-1]),
# #                     "planner": "released_navdp_s2diff_pixels",
# #                     "controller": "direct_waypoint_no_optimizer",
# #                     "qwen_role": "homotopy_return_command_and_mission_plan",
# #                     "qwen_process_isolated_from_habitat": True,
# #                     "qwen_creates_goal_or_action": False,
# #                     "qwen_homotopy": args.qwen_homotopy,
# #                     "qwen_homotopy_events": homotopy_events,
# #                     "qwen_homotopy_forces_all_candidates": args.qwen_homotopy,
# #                     "homotopy_sign_convention": {"LEFT": -1.0, "RIGHT": 1.0},
# #                     "homotopy_minimum_obstacle_pixels": args.homotopy_minimum_obstacle_pixels,
# #                     "homotopy_release_clear_frames": args.homotopy_release_clear_frames,
# #                     "homotopy_consistency_repeats": args.homotopy_consistency_repeats,
# #                     "uses_velocity_chunk": False,
# #                     "obstacle_mode": args.obstacle_mode,
# #                     "obstacle_world_xz": args.obstacle_world_xz,
# #                     "goal_mesh": args.goal_mesh,
# #                     "particle_anchor": args.particle_anchor,
# #                     "particle_energy_reweighting": args.particle_energy_reweighting,
# #                     "particle_collision_mask": args.particle_collision_mask,
# #                     "goal_mode": args.goal_mode,
# #                     "interactive_return_home": args.interactive_return_home,
# #                     "qwen_freeform_mission": args.qwen_freeform_mission,
# #                     "mission_plan_event": mission_plan_event,
# #                     "automatic_return_requested": automatic_return_requested,
# #                     "phase_completion_source": "metric_distance_state_machine",
# #                     "roundtrip_completed": roundtrip_completed,
# #                     "final_mission_phase": mission_phase,
# #                     "return_command_event": return_command_event,
# #                     "home_belief_source": "spawn_origin_plus_executed_odometry",
# #                     "reached_goal_becomes_obstacle_on_return": True,
# #                     "return_goal_obstacle_activation_distance": (
# #                         args.return_goal_obstacle_activation_distance
# #                     ),
# #                     "return_goal_obstacle_dilation_pixels": (
# #                         args.return_goal_obstacle_dilation_pixels
# #                     ),
# #                     "belief_pixel_goal": args.belief_pixel_goal,
# #                     "belief_source": "semantic_goal_mask_plus_odometry",
# #                     "belief_bootstrap_world_goal": args.belief_bootstrap_world_goal,
# #                     "belief_measurement_std": args.belief_measurement_std,
# #                     "belief_translation_process_std": args.belief_translation_process_std,
# #                     "belief_yaw_process_std_deg": args.belief_yaw_process_std_deg,
# #                     "belief_covariance_controls_navdp_mask_size": False,
# #                     "belief_heading_recovery": args.belief_heading_recovery,
# #                     "belief_recovery_obstacle_gated": True,
# #                     "belief_recovery_bearing_deg": args.belief_recovery_bearing_deg,
# #                     "belief_recovery_maximum_yaw_rate": args.belief_recovery_maximum_yaw_rate,
# #                     "belief_recovery_maximum_forward_speed": args.belief_recovery_maximum_forward_speed,
# #                     "particle_noise_schedule": args.particle_noise_schedule,
# #                     "progressive_guidance": args.progressive_guidance,
# #                     "mesh_obstacle_count": len(mesh_centroids),
# #                     "moving_obstacles": bool(np.any(np.abs(mesh_velocities) > 0.0)),
# #                     "obstacle_velocity_xz": mesh_velocities.tolist(),
# #                     "evaluation_layout": args.evaluation_layout,
# #                     "seed": args.seed,
# #                     "robot_radius": args.robot_radius,
# #                     "minimum_executed_surface_clearance": (
# #                         float(np.nanmin(rows["executed_surface_clearance"]))
# #                         if np.any(np.isfinite(rows["executed_surface_clearance"]))
# #                         else None
# #                     ),
# #                     "geometric_collision": bool(np.any(rows["geometric_collision"])),
# #                     "rollout": str(rollout_path),
# #                 },
# #                 file,
# #                 indent=2,
# #             )
# #         if args.save_video and video_frames:
# #             save_video(
# #                 video_frames,
# #                 output_directory / "rollout.mp4",
# #                 fps=max(args.hz / max(args.save_every, 1), 1.0),
# #             )
# #         print(f"Saved rollout: {rollout_path}", flush=True)
# #         print(f"Success: {success}", flush=True)
# #     finally:
# #         if simulator is not None:
# #             simulator.close()
# #         stop_server(server_process)
# #         stop_server(qwen_process)


# # if __name__ == "__main__":
# #     main()

# from __future__ import annotations

# import argparse
# import io
# import json
# import math
# import os
# import socket
# import subprocess
# import sys
# import time
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Optional, Sequence

# import habitat_sim
# import numpy as np
# import quaternion
# import requests
# from habitat_sim.agent import AgentConfiguration
# from PIL import Image, ImageDraw, ImageFilter

# from belief_heading_recovery import belief_heading_recovery_action
# from belief_pixel_goal import GaussianGoalBelief

# HERE = Path(__file__).resolve().parent
# SIZE_X = 50.0
# SIZE_Z = 50.0
# SIZE_Y = 4.820803273566
# MESH_GOAL_ID = 10000
# MESH_OBSTACLE_ID = 2


# @dataclass(frozen=True)
# class NavDPS2DiffOutput:
#     trajectory: np.ndarray
#     all_trajectories: np.ndarray
#     all_values: np.ndarray
#     selected_index: int
#     fallback_stop: bool
#     escape_turn: bool
#     valid_obstacle_points: int
#     selected_circulation_sign: float
#     candidate_circulation_signs: np.ndarray
#     selected_barrier_energy: float
#     selected_circulation_energy: float
#     minimum_clearance: np.ndarray
#     selected_minimum_clearance: float
#     mean_guidance_noise_correction: float
#     final_guidance_noise_correction: float
#     maximum_guidance_noise_correction: float
#     mean_final_effective_sample_size: float


# class NavDPS2DiffClient:
#     def __init__(self, server_url: str, timeout: float = 180.0):
#         self.server_url = server_url.rstrip("/")
#         self.timeout = float(timeout)

#     def reset(
#         self,
#         intrinsic: np.ndarray,
#         *,
#         stop_threshold: float = -3.0,
#         batch_size: int = 1,
#     ) -> str:
#         intrinsic = np.asarray(intrinsic, dtype=np.float32)
#         if intrinsic.shape != (3, 3):
#             raise ValueError(f"intrinsic must have shape [3,3], got {intrinsic.shape}")
#         response = requests.post(
#             f"{self.server_url}/navigator_reset",
#             json={
#                 "intrinsic": intrinsic.tolist(),
#                 "stop_threshold": float(stop_threshold),
#                 "batch_size": int(batch_size),
#             },
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         return str(response.json().get("algo", ""))

#     def plan(
#         self,
#         *,
#         goal_xy: np.ndarray,
#         rgb: np.ndarray,
#         depth: np.ndarray,
#         obstacle_pixels: np.ndarray,
#         goal_mode: str = "point",
#         forced_circulation_sign: float = 0.0,
#     ) -> NavDPS2DiffOutput:
#         goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
#         if goal_xy.shape != (2,):
#             raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")
#         if goal_mode not in {"point", "pixel"}:
#             raise ValueError("goal_mode must be point or pixel")
#         forced_circulation_sign = float(forced_circulation_sign)
#         if forced_circulation_sign not in {-1.0, 0.0, 1.0}:
#             raise ValueError("forced_circulation_sign must be -1, 0, or +1")

#         rgb = np.asarray(rgb, dtype=np.uint8)
#         if rgb.ndim != 3 or rgb.shape[-1] < 3:
#             raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
#         rgb = rgb[..., :3]

#         depth = np.asarray(depth, dtype=np.float32)
#         if depth.ndim == 3 and depth.shape[-1] == 1:
#             depth = depth[..., 0]
#         if depth.shape != rgb.shape[:2]:
#             raise ValueError(
#                 f"depth/rgb shape mismatch: {depth.shape} vs {rgb.shape[:2]}"
#             )

#         if goal_mode == "pixel":
#             if not np.all(np.isfinite(goal_xy)) or not np.allclose(
#                 goal_xy, np.round(goal_xy)
#             ):
#                 raise ValueError("PixelGoal must be integer [u,v]")
#             goal_xy = np.round(goal_xy).astype(np.int64)
#             if not (0 <= goal_xy[0] < rgb.shape[1] and 0 <= goal_xy[1] < rgb.shape[0]):
#                 raise ValueError("PixelGoal lies outside the RGB image")

#         pixels = np.asarray(obstacle_pixels)
#         if pixels.size == 0:
#             pixels = np.zeros((0, 2), dtype=np.int32)
#         else:
#             pixels = pixels.reshape(-1, 2)
#             if not np.all(np.isfinite(pixels)):
#                 raise ValueError("obstacle pixels must be finite")
#             if not np.allclose(pixels, np.round(pixels)):
#                 raise ValueError("obstacle pixels must be integer [u,v] coordinates")
#             pixels = np.round(pixels).astype(np.int32)

#         rgb_bytes = io.BytesIO()
#         Image.fromarray(rgb, mode="RGB").save(rgb_bytes, format="JPEG", quality=95)
#         depth_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
#         depth_bytes = io.BytesIO()
#         Image.fromarray(depth_u16).save(depth_bytes, format="PNG")

#         endpoint = "pixelgoal_step" if goal_mode == "pixel" else "pointgoal_step"
#         response = requests.post(
#             f"{self.server_url}/{endpoint}",
#             files={
#                 "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
#                 "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
#             },
#             data={
#                 "goal_data": json.dumps(
#                     {
#                         "goal_x": [float(goal_xy[0])],
#                         "goal_y": [float(goal_xy[1])],
#                         "obstacle_pixels": [pixels.tolist()],
#                         "forced_circulation_signs": [forced_circulation_sign],
#                     }
#                 )
#             },
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         payload = response.json()
#         diagnostics = payload["s2diff"]
#         trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
#         all_trajectories = np.asarray(payload["all_trajectory"], dtype=np.float32)
#         all_values = np.asarray(payload["all_values"], dtype=np.float32)

#         return NavDPS2DiffOutput(
#             trajectory=trajectory[0],
#             all_trajectories=all_trajectories[0],
#             all_values=all_values[0],
#             selected_index=int(diagnostics["selected_index"][0]),
#             fallback_stop=bool(diagnostics["fallback_stop"][0]),
#             escape_turn=bool(diagnostics["escape_turn"][0]),
#             valid_obstacle_points=int(diagnostics["valid_obstacle_points"][0]),
#             selected_circulation_sign=float(
#                 diagnostics["selected_circulation_sign"][0]
#             ),
#             candidate_circulation_signs=np.asarray(
#                 diagnostics["candidate_circulation_signs"][0], dtype=np.float32
#             ),
#             selected_barrier_energy=float(diagnostics["selected_barrier_energy"][0]),
#             selected_circulation_energy=float(
#                 diagnostics["selected_circulation_energy"][0]
#             ),
#             minimum_clearance=np.asarray(
#                 diagnostics["minimum_clearance"][0], dtype=np.float32
#             ),
#             selected_minimum_clearance=float(
#                 diagnostics["selected_minimum_clearance"][0]
#             ),
#             mean_guidance_noise_correction=float(
#                 diagnostics["mean_guidance_noise_correction"][0]
#             ),
#             final_guidance_noise_correction=float(
#                 diagnostics["final_guidance_noise_correction"][0]
#             ),
#             maximum_guidance_noise_correction=float(
#                 diagnostics["maximum_guidance_noise_correction"][0]
#             ),
#             mean_final_effective_sample_size=float(
#                 diagnostics.get("mean_final_effective_sample_size", [0.0])[0]
#             ),
#         )

#     @staticmethod
#     def _raise_for_error(response: requests.Response) -> None:
#         try:
#             payload = response.json()
#         except ValueError:
#             payload = None
#         if isinstance(payload, dict) and "error" in payload:
#             raise RuntimeError(str(payload["error"]))
#         response.raise_for_status()


# @dataclass(frozen=True)
# class QwenHomotopyDecision:
#     side: str
#     circulation_sign: float
#     confidence: float
#     obstacle_relevant: bool
#     queried_qwen: bool
#     raw_response: Optional[str]
#     repeated_sides: tuple[str, ...]
#     repeated_confidences: tuple[float, ...]
#     consistency_rate: float
#     used_fallback: bool


# @dataclass(frozen=True)
# class QwenCommandDecision:
#     command: str
#     confidence: float
#     raw_response: str


# @dataclass(frozen=True)
# class QwenMissionPlanDecision:
#     plan: tuple[str, ...]
#     confidence: float
#     raw_response: str


# class QwenHomotopyClient:
#     """HTTP client for the isolated visual-Qwen process."""

#     def __init__(self, server_url: str, timeout: float = 300.0) -> None:
#         self.server_url = server_url.rstrip("/")
#         self.timeout = float(timeout)

#     def reset(self) -> None:
#         response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
#         self._raise_for_error(response)

#     def step(
#         self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
#     ) -> QwenHomotopyDecision:
#         image_bytes = io.BytesIO()
#         Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8)).save(
#             image_bytes, format="PNG"
#         )
#         mask_bytes = io.BytesIO()
#         Image.fromarray((np.asarray(obstacle_mask) > 0).astype(np.uint8) * 255).save(
#             mask_bytes, format="PNG"
#         )
#         response = requests.post(
#             f"{self.server_url}/select",
#             files={
#                 "image": ("overlay.png", image_bytes.getvalue(), "image/png"),
#                 "obstacle_mask": ("mask.png", mask_bytes.getvalue(), "image/png"),
#             },
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         payload = response.json()
#         return QwenHomotopyDecision(
#             side=str(payload["side"]),
#             circulation_sign=float(payload["circulation_sign"]),
#             confidence=float(payload["confidence"]),
#             obstacle_relevant=bool(payload["obstacle_relevant"]),
#             queried_qwen=bool(payload["queried_qwen"]),
#             raw_response=payload.get("raw_response"),
#             repeated_sides=tuple(payload.get("repeated_sides", [])),
#             repeated_confidences=tuple(
#                 float(value) for value in payload.get("repeated_confidences", [])
#             ),
#             consistency_rate=float(payload.get("consistency_rate", 1.0)),
#             used_fallback=bool(payload.get("used_fallback", False)),
#         )

#     def classify_command(
#         self, image_rgb: np.ndarray, user_command: str
#     ) -> QwenCommandDecision:
#         image_bytes = io.BytesIO()
#         Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
#             image_bytes, format="PNG"
#         )
#         response = requests.post(
#             f"{self.server_url}/command",
#             files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
#             data={"command": str(user_command)},
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         payload = response.json()
#         return QwenCommandDecision(
#             command=str(payload["command"]).upper(),
#             confidence=float(payload["confidence"]),
#             raw_response=str(payload["raw_response"]),
#         )

#     def classify_mission(
#         self, image_rgb: np.ndarray, user_command: str
#     ) -> QwenMissionPlanDecision:
#         image_bytes = io.BytesIO()
#         Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
#             image_bytes, format="PNG"
#         )
#         response = requests.post(
#             f"{self.server_url}/mission",
#             files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
#             data={"command": str(user_command)},
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         payload = response.json()
#         return QwenMissionPlanDecision(
#             plan=tuple(str(item).upper() for item in payload["plan"]),
#             confidence=float(payload["confidence"]),
#             raw_response=str(payload["raw_response"]),
#         )

#     @staticmethod
#     def _raise_for_error(response: requests.Response) -> None:
#         try:
#             payload = response.json()
#         except ValueError:
#             payload = None
#         if isinstance(payload, dict) and "error" in payload:
#             raise RuntimeError(str(payload["error"]))
#         response.raise_for_status()


# def port_is_open(host: str, port: int) -> bool:
#     try:
#         with socket.create_connection((host, port), timeout=1.0):
#             return True
#     except OSError:
#         return False


# def wait_for_server(
#     process: subprocess.Popen[Any], host: str, port: int, timeout: float
# ) -> None:
#     deadline = time.time() + float(timeout)
#     while time.time() < deadline:
#         if process.poll() is not None:
#             raise RuntimeError(
#                 f"NavDP/S2Diff server exited with code {process.returncode}"
#             )
#         if port_is_open(host, port):
#             return
#         time.sleep(1.0)
#     raise TimeoutError(f"NavDP server did not open port {port} within {timeout}s")


# def stop_server(process: Optional[subprocess.Popen[Any]]) -> None:
#     if process is None or process.poll() is not None:
#         return
#     process.terminate()
#     try:
#         process.wait(timeout=10.0)
#     except subprocess.TimeoutExpired:
#         process.kill()
#         process.wait()


# def start_qwen_homotopy_server(
#     args: argparse.Namespace,
# ) -> Optional[subprocess.Popen[Any]]:
#     if not args.qwen_homotopy or not args.start_qwen_homotopy_server:
#         return None
#     if port_is_open(args.qwen_homotopy_host, args.qwen_homotopy_port):
#         raise RuntimeError(
#             f"Qwen homotopy port {args.qwen_homotopy_port} is already in use; "
#             "pass --no-start-qwen-homotopy-server to use an existing service"
#         )
#     server_file = HERE / "qwen_homotopy_server.py"
#     if not server_file.is_file():
#         raise FileNotFoundError(f"Qwen homotopy server not found: {server_file}")
#     command = [
#         str(args.qwen_homotopy_python),
#         str(server_file),
#         "--host",
#         str(args.qwen_homotopy_host),
#         "--port",
#         str(args.qwen_homotopy_port),
#         "--model-id",
#         str(args.qwen_model_id),
#         "--device",
#         str(args.qwen_device),
#         "--minimum-obstacle-pixels",
#         str(args.homotopy_minimum_obstacle_pixels),
#         "--release-clear-frames",
#         str(args.homotopy_release_clear_frames),
#         "--consistency-repeats",
#         str(args.homotopy_consistency_repeats),
#     ]
#     print("[qwen-server]", " ".join(command), flush=True)
#     process = subprocess.Popen(command, cwd=str(HERE))
#     wait_for_server(
#         process,
#         args.qwen_homotopy_host,
#         args.qwen_homotopy_port,
#         args.qwen_homotopy_timeout,
#     )
#     return process


# def start_server(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
#     if not args.start_server:
#         return None
#     if port_is_open(args.server_host, args.server_port):
#         raise RuntimeError(
#             f"port {args.server_port} is already in use; use --no-start-server "
#             "to connect to an existing guided server"
#         )

#     navdp_root = Path(args.navdp_root).expanduser().resolve()
#     checkpoint = Path(args.navdp_checkpoint).expanduser().resolve()
#     server_dir = navdp_root / "baselines" / "navdp"
#     server_file = server_dir / "navdp_s2diff_server.py"
#     if not server_file.is_file():
#         raise FileNotFoundError(f"guided server not found: {server_file}")
#     if not checkpoint.is_file():
#         raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint}")

#     command = [
#         str(args.navdp_python),
#         str(server_file),
#         "--checkpoint",
#         str(checkpoint),
#         "--device",
#         str(args.navdp_device),
#         "--planner-mode",
#         str(args.planner_mode),
#         "--seed",
#         str(args.seed),
#         "--port",
#         str(args.server_port),
#         "--candidates",
#         str(args.candidates),
#         "--particles",
#         str(args.particles),
#         "--particle-std",
#         str(args.particle_std),
#         "--gradient-steps",
#         str(args.gradient_steps),
#         "--gradient-step-size",
#         str(args.gradient_step_size),
#         "--guidance-strength",
#         str(args.guidance_strength),
#         "--temperature",
#         str(args.temperature),
#         "--safe-distance",
#         str(args.safe_distance),
#         "--hard-collision-distance",
#         str(args.hard_collision_distance),
#         "--robot-radius",
#         str(args.robot_radius),
#         "--safety-weight",
#         str(args.safety_weight),
#         "--barrier-weight",
#         str(args.barrier_weight),
#         "--barrier-rate",
#         str(args.barrier_rate),
#         "--circulation-weight",
#         str(args.circulation_weight),
#         "--circulation-activation-distance",
#         str(args.circulation_activation_distance),
#         "--circulation-activation-sharpness",
#         str(args.circulation_activation_sharpness),
#         "--minimum-circulation-progress",
#         str(args.minimum_circulation_progress),
#         "--blocking-alignment-threshold",
#         str(args.blocking_alignment_threshold),
#         "--circulation-switch-weight",
#         str(args.circulation_switch_weight),
#         "--escape-lateral-target",
#         str(args.escape_lateral_target),
#         "--minimum-obstacle-depth",
#         str(args.minimum_obstacle_depth),
#         "--maximum-obstacle-depth",
#         str(args.maximum_obstacle_depth),
#         "--maximum-obstacle-pixels",
#         str(args.maximum_obstacle_pixels),
#     ]
#     particle_flags = {
#         "particle-anchor": args.particle_anchor,
#         "particle-energy-reweighting": args.particle_energy_reweighting,
#         "particle-collision-mask": args.particle_collision_mask,
#         "particle-noise-schedule": args.particle_noise_schedule,
#         "progressive-guidance": args.progressive_guidance,
#     }
#     for name, enabled in particle_flags.items():
#         command.append(f"--{name}" if enabled else f"--no-{name}")
#     command.append("--remove-critic" if args.remove_critic else "--no-remove-critic")
#     print("[server]", " ".join(command), flush=True)
#     process = subprocess.Popen(command, cwd=str(server_dir))
#     wait_for_server(process, args.server_host, args.server_port, args.server_timeout)
#     return process


# def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
#     height, width = grid.shape
#     x0 = int(np.floor(px))
#     y0 = int(np.floor(py))
#     x1 = min(x0 + 1, width - 1)
#     y1 = min(y0 + 1, height - 1)
#     tx = px - x0
#     ty = py - y0
#     top = float(grid[y0, x0]) * (1.0 - tx) + float(grid[y0, x1]) * tx
#     bottom = float(grid[y1, x0]) * (1.0 - tx) + float(grid[y1, x1]) * tx
#     return top * (1.0 - ty) + bottom * ty


# class TerrainHeight:
#     def __init__(
#         self,
#         *,
#         mode: str,
#         heightmap: Optional[Path],
#         obj: Optional[Path],
#         flat_y: float,
#         size_x: float,
#         size_z: float,
#         size_y: float,
#         flip_x: bool,
#         flip_z: bool,
#         swap_xz: bool,
#     ):
#         if mode == "auto":
#             mode = (
#                 "heightmap"
#                 if heightmap and heightmap.exists()
#                 else ("obj" if obj and obj.exists() else "flat")
#             )
#         self.mode = mode
#         self.flat_y = float(flat_y)
#         self.size_x = float(size_x)
#         self.size_z = float(size_z)
#         self.size_y = float(size_y)
#         self.flip_x = bool(flip_x)
#         self.flip_z = bool(flip_z)
#         self.swap_xz = bool(swap_xz)
#         self.height: Optional[np.ndarray] = None
#         self.obj_xs: Optional[np.ndarray] = None
#         self.obj_zs: Optional[np.ndarray] = None
#         self.obj_h: Optional[np.ndarray] = None

#         if mode == "heightmap":
#             if heightmap is None or not heightmap.exists():
#                 raise FileNotFoundError(f"heightmap not found: {heightmap}")
#             array = np.asarray(Image.open(heightmap))
#             if array.ndim == 3:
#                 array = array[..., 0]
#             array = array.astype(np.float32)
#             array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
#             self.height = array * self.size_y - float(np.mean(array * self.size_y))
#         elif mode == "obj":
#             if obj is None or not obj.exists():
#                 raise FileNotFoundError(f"terrain OBJ not found: {obj}")
#             vertices = []
#             with obj.open("r", encoding="utf-8", errors="ignore") as file:
#                 for line in file:
#                     if line.startswith("v "):
#                         parts = line.split()
#                         if len(parts) >= 4:
#                             vertices.append(tuple(float(value) for value in parts[1:4]))
#             if not vertices:
#                 raise RuntimeError(f"no vertices found in {obj}")
#             array = np.asarray(vertices, dtype=np.float32)
#             xs = np.unique(array[:, 0])
#             zs = np.unique(array[:, 1])
#             grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
#             x_index = {float(value): index for index, value in enumerate(xs.tolist())}
#             z_index = {float(value): index for index, value in enumerate(zs.tolist())}
#             for x, z, height in array:
#                 grid[z_index[float(z)], x_index[float(x)]] = height
#             self.obj_xs = xs
#             self.obj_zs = zs
#             self.obj_h = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
#         elif mode != "flat":
#             raise ValueError(f"unknown terrain mode: {mode}")

#     def _map(self, x: float, z: float) -> tuple[float, float]:
#         if self.swap_xz:
#             x, z = z, x
#         u = (x + self.size_x / 2.0) / self.size_x
#         v = (z + self.size_z / 2.0) / self.size_z
#         if self.flip_x:
#             u = 1.0 - u
#         if self.flip_z:
#             v = 1.0 - v
#         return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

#     def __call__(self, x: float, z: float) -> float:
#         if self.mode == "flat":
#             return self.flat_y
#         if self.mode == "heightmap":
#             assert self.height is not None
#             u, v = self._map(x, z)
#             return bilinear_grid(
#                 self.height,
#                 u * (self.height.shape[1] - 1),
#                 v * (self.height.shape[0] - 1),
#             )
#         assert (
#             self.obj_xs is not None
#             and self.obj_zs is not None
#             and self.obj_h is not None
#         )
#         xx = float(np.clip(x, self.obj_xs[0], self.obj_xs[-1]))
#         zz = float(np.clip(z, self.obj_zs[0], self.obj_zs[-1]))
#         column = int(
#             np.clip(np.searchsorted(self.obj_xs, xx) - 1, 0, len(self.obj_xs) - 2)
#         )
#         row = int(
#             np.clip(np.searchsorted(self.obj_zs, zz) - 1, 0, len(self.obj_zs) - 2)
#         )
#         x0, x1 = float(self.obj_xs[column]), float(self.obj_xs[column + 1])
#         z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
#         tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
#         tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
#         top = (
#             float(self.obj_h[row, column]) * (1.0 - tx)
#             + float(self.obj_h[row, column + 1]) * tx
#         )
#         bottom = (
#             float(self.obj_h[row + 1, column]) * (1.0 - tx)
#             + float(self.obj_h[row + 1, column + 1]) * tx
#         )
#         return top * (1.0 - tz) + bottom * tz

#     def local_height_max(
#         self, x: float, z: float, radius: float, samples: int = 5
#     ) -> float:
#         if radius <= 1e-6:
#             return float(self(x, z))
#         values = [
#             float(self(x + dx, z + dz))
#             for dx in np.linspace(-radius, radius, samples)
#             for dz in np.linspace(-radius, radius, samples)
#             if dx * dx + dz * dz <= radius * radius + 1e-8
#         ]
#         return max(values) if values else float(self(x, z))


# def make_sensor(
#     uuid: str, sensor_type: Any, height: int, width: int, hfov_deg: float
# ) -> habitat_sim.CameraSensorSpec:
#     specification = habitat_sim.CameraSensorSpec()
#     specification.uuid = uuid
#     specification.sensor_type = sensor_type
#     specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     specification.resolution = [int(height), int(width)]
#     specification.position = [0.0, 0.0, 0.0]
#     specification.hfov = float(hfov_deg)
#     return specification


# def make_simulator(
#     scene: Path,
#     height: int,
#     width: int,
#     hfov_deg: float,
#     *,
#     with_semantic: bool,
# ):
#     simulator_configuration = habitat_sim.SimulatorConfiguration()
#     simulator_configuration.scene_id = str(scene.expanduser().resolve())
#     simulator_configuration.enable_physics = False
#     sensors = [
#         make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
#         make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
#     ]
#     if with_semantic:
#         sensors.append(
#             make_sensor(
#                 "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
#             )
#         )
#     agent_configuration = AgentConfiguration()
#     agent_configuration.sensor_specifications = sensors
#     return habitat_sim.Simulator(
#         habitat_sim.Configuration(simulator_configuration, [agent_configuration])
#     )


# def set_agent_pose(agent: Any, position: np.ndarray, yaw: float) -> None:
#     state = agent.get_state()
#     state.position = np.asarray(position, dtype=np.float32)
#     state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
#     agent.set_state(state)


# def rgb_depth(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
#     rgb = np.asarray(observation["rgb"])
#     if rgb.ndim == 3 and rgb.shape[-1] == 4:
#         rgb = rgb[..., :3]
#     depth = np.asarray(observation["depth"], dtype=np.float32)
#     if depth.ndim == 3:
#         depth = depth[..., 0]
#     return rgb.astype(np.uint8), depth.astype(np.float32)


# def semantic_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
#     semantic = np.asarray(observation["semantic"])
#     if semantic.ndim == 3:
#         semantic = semantic[..., 0]
#     return semantic.astype(np.int32)


# def pixel_to_world(
#     u: float,
#     v: float,
#     depth: float,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
# ) -> np.ndarray:
#     right = (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
#     up = -(v - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
#     forward_vector = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)])
#     right_vector = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)])
#     return (
#         np.asarray(position, dtype=np.float64)
#         + depth * forward_vector
#         + right * right_vector
#         + up * np.asarray([0.0, 1.0, 0.0])
#     )


# def depth_patch_mesh(
#     u_center: float,
#     v_center: float,
#     half_size: int,
#     stride: int,
#     depth: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     *,
#     lift: float,
#     maximum_depth_jump: float = 0.4,
# ) -> tuple[np.ndarray, np.ndarray]:
#     height, width = depth.shape
#     columns = list(
#         range(
#             max(0, int(u_center - half_size)),
#             min(width, int(u_center + half_size) + 1),
#             max(int(stride), 1),
#         )
#     )
#     rows = list(
#         range(
#             max(0, int(v_center - half_size)),
#             min(height, int(v_center + half_size) + 1),
#             max(int(stride), 1),
#         )
#     )
#     indices = -np.ones((len(rows), len(columns)), dtype=np.int64)
#     depths = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
#     vertices: list[tuple[float, float, float]] = []
#     for row_index, v in enumerate(rows):
#         for column_index, u in enumerate(columns):
#             metric_depth = float(depth[v, u])
#             if not np.isfinite(metric_depth) or metric_depth <= 0.1:
#                 continue
#             indices[row_index, column_index] = len(vertices)
#             depths[row_index, column_index] = metric_depth
#             point = pixel_to_world(
#                 u, v, metric_depth, position, yaw, intrinsic
#             ) + float(lift) * np.asarray([0.0, 1.0, 0.0])
#             vertices.append(tuple(float(value) for value in point))

#     faces: list[tuple[int, int, int]] = []
#     for row_index in range(len(rows) - 1):
#         for column_index in range(len(columns) - 1):
#             a = int(indices[row_index, column_index])
#             b = int(indices[row_index, column_index + 1])
#             c = int(indices[row_index + 1, column_index])
#             d = int(indices[row_index + 1, column_index + 1])
#             if min(a, b, c, d) < 0:
#                 continue
#             cell_depths = (
#                 depths[row_index, column_index],
#                 depths[row_index, column_index + 1],
#                 depths[row_index + 1, column_index],
#                 depths[row_index + 1, column_index + 1],
#             )
#             if max(cell_depths) - min(cell_depths) > maximum_depth_jump:
#                 continue
#             faces.append((a, c, d))
#             faces.append((a, d, b))
#     return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# def save_obj(
#     path: Path,
#     vertices: np.ndarray,
#     faces: np.ndarray,
#     *,
#     diffuse_rgb: Optional[tuple[float, float, float]] = None,
# ) -> None:
#     material_name = None
#     if diffuse_rgb is not None:
#         red, green, blue = (float(value) for value in diffuse_rgb)
#         if not all(0.0 <= value <= 1.0 for value in (red, green, blue)):
#             raise ValueError("OBJ diffuse material values must be in [0, 1]")
#         material_name = "mesh_material"
#         material_path = path.with_suffix(".mtl")
#         with material_path.open("w", encoding="utf-8") as material:
#             material.write(f"newmtl {material_name}\n")
#             material.write(
#                 f"Ka {0.25 * red:.4f} {0.25 * green:.4f} {0.25 * blue:.4f}\n"
#             )
#             material.write(f"Kd {red:.4f} {green:.4f} {blue:.4f}\n")
#             material.write("Ks 0.1000 0.1000 0.1000\n")
#             material.write("Ns 24.0000\n")
#             material.write("d 1.0000\n")
#             material.write("illum 2\n")
#     with path.open("w", encoding="utf-8") as file:
#         if material_name is not None:
#             file.write(f"mtllib {path.with_suffix('.mtl').name}\n")
#             file.write(f"usemtl {material_name}\n")
#         for x, y, z in vertices:
#             file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
#         for a, b, c in faces:
#             file.write(f"f {a + 1} {b + 1} {c + 1}\n")


# def register_semantic_mesh(simulator: Any, mesh_path: Path, semantic_id: int) -> Any:
#     template_manager = simulator.get_object_template_manager()
#     object_manager = simulator.get_rigid_object_manager()
#     template = template_manager.create_new_template(str(mesh_path))
#     template.render_asset_handle = str(mesh_path)
#     template.collision_asset_handle = str(mesh_path)
#     template.is_collidable = False
#     template_id = template_manager.register_template(
#         template, f"s2diff_obstacle_{semantic_id}_{os.path.basename(mesh_path)}"
#     )
#     object_handle = template_manager.get_template_handle_by_id(template_id)
#     obstacle = object_manager.add_object_by_template_handle(object_handle)
#     obstacle.motion_type = habitat_sim.physics.MotionType.KINEMATIC
#     obstacle.collidable = False
#     obstacle.semantic_id = int(semantic_id)
#     return obstacle


# def parse_world_xz(specification: str) -> tuple[float, float]:
#     values = [float(value) for value in str(specification).split(",")]
#     if len(values) != 2 or not np.isfinite(values).all():
#         raise ValueError(
#             f"world mesh position must be finite X,Z, got {specification!r}"
#         )
#     return values[0], values[1]


# def world_box_mesh(
#     center_x: float,
#     base_y: float,
#     center_z: float,
#     half_extent: float,
#     height: float,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Create a closed axis-aligned box whose vertices are in world coordinates."""

#     if half_extent <= 0.0 or height <= 0.0:
#         raise ValueError("box half extent and height must be positive")
#     x0, x1 = center_x - half_extent, center_x + half_extent
#     z0, z1 = center_z - half_extent, center_z + half_extent
#     y0, y1 = base_y, base_y + height
#     vertices = np.asarray(
#         [
#             [x0, y0, z0],
#             [x1, y0, z0],
#             [x1, y0, z1],
#             [x0, y0, z1],
#             [x0, y1, z0],
#             [x1, y1, z0],
#             [x1, y1, z1],
#             [x0, y1, z1],
#         ],
#         dtype=np.float64,
#     )
#     faces = np.asarray(
#         [
#             [0, 2, 1],
#             [0, 3, 2],
#             [4, 5, 6],
#             [4, 6, 7],
#             [0, 1, 5],
#             [0, 5, 4],
#             [1, 2, 6],
#             [1, 6, 5],
#             [2, 3, 7],
#             [2, 7, 6],
#             [3, 0, 4],
#             [3, 4, 7],
#         ],
#         dtype=np.int64,
#     )
#     return vertices, faces


# def place_world_obstacle_meshes(
#     simulator: Any,
#     terrain: Any,
#     xz_specifications: Sequence[str],
#     output_directory: Path,
#     *,
#     half_extent: float,
#     height: float,
# ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
#     """Place static obstacle boxes at exact world X,Z coordinates."""

#     mesh_directory = output_directory / "meshes"
#     mesh_directory.mkdir(parents=True, exist_ok=True)
#     objects: list[Any] = []
#     centroids: list[np.ndarray] = []
#     geometries: list[np.ndarray] = []
#     for index, specification in enumerate(xz_specifications):
#         center_x, center_z = parse_world_xz(specification)
#         base_y = terrain.local_height_max(center_x, center_z, half_extent)
#         vertices, faces = world_box_mesh(
#             center_x, base_y, center_z, half_extent, height
#         )
#         mesh_path = mesh_directory / f"world_obstacle_{index}.obj"
#         save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.78, 0.16, 0.06))
#         semantic_id = MESH_OBSTACLE_ID + index
#         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
#         centroid = vertices.mean(axis=0).astype(np.float32)
#         centroids.append(centroid)
#         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
#         print(
#             f"[world-mesh] obstacle={index} semantic_id={semantic_id} "
#             f"center_xz={[center_x, center_z]} half_extent={half_extent:.3f} "
#             f"height={height:.3f}",
#             flush=True,
#         )
#     return objects, centroids, geometries


# def place_world_goal_mesh(
#     simulator: Any,
#     terrain: Any,
#     goal_x: float,
#     goal_z: float,
#     output_directory: Path,
#     *,
#     half_extent: float,
#     height: float,
# ) -> Any:
#     """Place a visible, non-obstacle semantic goal marker at the exact goal."""

#     base_y = terrain.local_height_max(goal_x, goal_z, half_extent)
#     vertices, faces = world_box_mesh(goal_x, base_y, goal_z, half_extent, height)
#     mesh_directory = output_directory / "meshes"
#     mesh_directory.mkdir(parents=True, exist_ok=True)
#     mesh_path = mesh_directory / "goal_marker.obj"
#     save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.08, 0.85, 0.18))
#     goal_object = register_semantic_mesh(simulator, mesh_path, MESH_GOAL_ID)
#     print(
#         f"[world-mesh] goal semantic_id={MESH_GOAL_ID} "
#         f"center_xz={[goal_x, goal_z]}",
#         flush=True,
#     )
#     return goal_object


# def parse_uv_fraction(
#     specification: str, width: int, height: int
# ) -> tuple[float, float]:
#     u_fraction, v_fraction = (float(value) for value in str(specification).split(","))
#     if not (0.0 <= u_fraction <= 1.0 and 0.0 <= v_fraction <= 1.0):
#         raise ValueError(f"mesh pixel fraction must be in [0,1], got {specification!r}")
#     return u_fraction * width, v_fraction * height


# def place_obstacle_meshes(
#     simulator: Any,
#     depth: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     uv_specifications: Sequence[str],
#     output_directory: Path,
#     *,
#     mesh_half_pixels: int,
#     mesh_lift: float,
# ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
#     mesh_directory = output_directory / "meshes"
#     mesh_directory.mkdir(parents=True, exist_ok=True)
#     height, width = depth.shape
#     objects: list[Any] = []
#     centroids: list[np.ndarray] = []
#     geometries: list[np.ndarray] = []
#     for index, specification in enumerate(uv_specifications):
#         u, v = parse_uv_fraction(specification, width, height)
#         vertices, faces = depth_patch_mesh(
#             u,
#             v,
#             mesh_half_pixels,
#             2,
#             depth,
#             position,
#             yaw,
#             intrinsic,
#             lift=mesh_lift,
#         )
#         if len(vertices) == 0 or len(faces) == 0:
#             raise RuntimeError(
#                 f"obstacle mesh {index} at {specification!r} has no valid depth surface"
#             )
#         mesh_path = mesh_directory / f"obstacle_{index}.obj"
#         save_obj(mesh_path, vertices, faces)
#         semantic_id = MESH_OBSTACLE_ID + index
#         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
#         centroid = vertices.mean(axis=0).astype(np.float32)
#         centroids.append(centroid)
#         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
#         print(
#             f"[mesh] obstacle={index} semantic_id={semantic_id} "
#             f"pixels={specification} vertices={len(vertices)} "
#             f"world={centroid.tolist()}",
#             flush=True,
#         )
#     return objects, centroids, geometries


# def planar_mesh_clearance(
#     point_xz: np.ndarray,
#     geometries: Sequence[np.ndarray],
# ) -> float:
#     """Minimum 2-D distance from a robot center to projected mesh triangles."""
#     point = np.asarray(point_xz, dtype=np.float64)
#     best = float("inf")
#     for triangles in geometries:
#         triangles = np.asarray(triangles, dtype=np.float64)
#         if triangles.size == 0:
#             continue
#         a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
#         v0, v1, v2 = b - a, c - a, point[None, :] - a
#         denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
#         valid = np.abs(denominator) > 1.0e-12
#         safe_denominator = np.where(valid, denominator, 1.0)
#         u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_denominator
#         v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_denominator
#         if np.any(valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)):
#             return 0.0

#         starts = np.concatenate((a, b, c), axis=0)
#         ends = np.concatenate((b, c, a), axis=0)
#         segments = ends - starts
#         squared_lengths = np.einsum("ij,ij->i", segments, segments)
#         numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
#         fractions = np.divide(
#             numerators,
#             squared_lengths,
#             out=np.zeros_like(numerators),
#             where=squared_lengths > 1.0e-12,
#         )
#         fractions = np.clip(fractions, 0.0, 1.0)
#         closest = starts + fractions[:, None] * segments
#         best = min(best, float(np.linalg.norm(point[None, :] - closest, axis=1).min()))
#     return best


# def parse_xz_velocity(specification: str) -> np.ndarray:
#     values = [float(value) for value in str(specification).split(",")]
#     if len(values) != 2 or not np.all(np.isfinite(values)):
#         raise ValueError("obstacle velocity must be finite vx,vz")
#     return np.asarray(values, dtype=np.float64)


# def expand_obstacle_velocities(
#     specifications: Sequence[str], obstacle_count: int
# ) -> np.ndarray:
#     if obstacle_count == 0:
#         return np.zeros((0, 2), dtype=np.float64)
#     if not specifications:
#         return np.zeros((obstacle_count, 2), dtype=np.float64)
#     velocities = np.stack([parse_xz_velocity(item) for item in specifications])
#     if len(velocities) == 1 and obstacle_count > 1:
#         velocities = np.repeat(velocities, obstacle_count, axis=0)
#     if len(velocities) != obstacle_count:
#         raise ValueError(
#             "provide one obstacle velocity to broadcast or one velocity per mesh"
#         )
#     return velocities


# def translated_mesh_geometry(
#     base_geometries: Sequence[np.ndarray],
#     base_centroids: Sequence[np.ndarray],
#     velocities_xz: np.ndarray,
#     elapsed_seconds: float,
# ) -> tuple[list[np.ndarray], list[np.ndarray]]:
#     geometries: list[np.ndarray] = []
#     centroids: list[np.ndarray] = []
#     for geometry, centroid, velocity in zip(
#         base_geometries, base_centroids, velocities_xz
#     ):
#         offset_xz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
#         geometries.append(np.asarray(geometry) + offset_xz[None, None, :])
#         offset_xyz = np.asarray([offset_xz[0], 0.0, offset_xz[1]])
#         centroids.append(np.asarray(centroid, dtype=np.float64) + offset_xyz)
#     return geometries, centroids


# def move_mesh_objects(
#     objects: Sequence[Any], velocities_xz: np.ndarray, elapsed_seconds: float
# ) -> None:
#     for obstacle, velocity in zip(objects, velocities_xz):
#         dx, dz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
#         vector_type = type(obstacle.translation)
#         obstacle.translation = vector_type(float(dx), 0.0, float(dz))


# def camera_coordinates(
#     point: np.ndarray, position: np.ndarray, yaw: float
# ) -> tuple[float, float, float]:
#     delta = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
#     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
#     left_x, left_z = -math.cos(yaw), math.sin(yaw)
#     forward = forward_x * float(delta[0]) + forward_z * float(delta[2])
#     left = left_x * float(delta[0]) + left_z * float(delta[2])
#     return -left, float(delta[1]), forward


# def camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
#     hfov = math.radians(float(hfov_deg))
#     focal = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
#     return np.asarray(
#         [
#             [focal, 0.0, (width - 1) * 0.5],
#             [0.0, focal, (height - 1) * 0.5],
#             [0.0, 0.0, 1.0],
#         ],
#         dtype=np.float32,
#     )


# def world_goal_to_pixel(
#     point: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     height: int,
#     width: int,
# ) -> np.ndarray:
#     """Project a world goal to a valid PixelGoal, clamping off-screen bearings."""

#     right, up, forward = camera_coordinates(point, position, yaw)
#     fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
#     cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
#     margin = 11
#     bearing = math.atan2(right, forward)
#     maximum_bearing = math.atan2(max(cx - margin, 1.0), fx)
#     bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
#     u = cx + fx * math.tan(bearing)
#     v = cy - fy * up / forward if forward > 0.05 else 0.62 * height
#     return np.asarray(
#         [
#             int(np.clip(round(u), margin, width - margin - 1)),
#             int(np.clip(round(v), margin, height - margin - 1)),
#         ],
#         dtype=np.int32,
#     )


# def circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
#     yy, xx = np.ogrid[:height, :width]
#     return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


# def project_world_mask(
#     point: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     height: int,
#     width: int,
#     radius: int,
# ) -> tuple[np.ndarray, float]:
#     right, up, forward = camera_coordinates(point, position, yaw)
#     if forward <= 0.05:
#         return np.zeros((height, width), dtype=np.uint8), forward
#     u = float(intrinsic[0, 2] + intrinsic[0, 0] * right / forward)
#     v = float(intrinsic[1, 2] - intrinsic[1, 1] * up / forward)
#     if not (radius <= u < width - radius and radius <= v < height - radius):
#         return np.zeros((height, width), dtype=np.uint8), forward
#     return circle_mask(height, width, u, v, radius), forward


# def depth_obstacle_mask(
#     depth: np.ndarray, threshold: float, minimum_y_fraction: float
# ) -> np.ndarray:
#     mask = np.isfinite(depth) & (depth > 0.05) & (depth < float(threshold))
#     mask[: int(depth.shape[0] * minimum_y_fraction)] = False
#     return mask.astype(np.uint8)


# def pixels_from_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
#     v, u = np.nonzero(np.asarray(mask) > 0)
#     if u.size == 0:
#         return np.zeros((0, 2), dtype=np.int32)
#     pixels = np.stack((u, v), axis=-1).astype(np.int32)
#     if maximum > 0 and len(pixels) > maximum:
#         indices = np.linspace(0, len(pixels) - 1, maximum).astype(np.int64)
#         pixels = pixels[indices]
#     return pixels


# def waypoint_action(
#     trajectory: np.ndarray,
#     *,
#     lookahead_index: int,
#     maximum_forward_speed: float,
#     maximum_yaw_rate: float,
#     yaw_gain: float,
# ) -> np.ndarray:
#     trajectory = np.asarray(trajectory, dtype=np.float32)
#     if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
#         return np.zeros(3, dtype=np.float32)
#     if np.max(np.linalg.norm(trajectory[:, :2], axis=-1)) < 1e-5:
#         return np.zeros(3, dtype=np.float32)
#     index = int(np.clip(lookahead_index, 0, trajectory.shape[0] - 1))
#     forward, left = float(trajectory[index, 0]), float(trajectory[index, 1])
#     bearing = math.atan2(left, max(forward, 1e-4))
#     velocity = maximum_forward_speed * max(0.0, math.cos(bearing))
#     yaw_rate = float(np.clip(yaw_gain * bearing, -maximum_yaw_rate, maximum_yaw_rate))
#     return np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)


# def integrate_mars(
#     position: np.ndarray, yaw: float, action: np.ndarray, dt: float
# ) -> tuple[np.ndarray, float]:
#     forward_velocity, lateral_velocity, yaw_rate = [float(value) for value in action]
#     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
#     left_x, left_z = -math.cos(yaw), math.sin(yaw)
#     output = np.asarray(position, dtype=np.float32).copy()
#     output[0] += (forward_x * forward_velocity + left_x * lateral_velocity) * dt
#     output[2] += (forward_z * forward_velocity + left_z * lateral_velocity) * dt
#     return output, yaw + yaw_rate * dt


# def wrap_angle(angle: float) -> float:
#     return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# def overlay_frame(
#     rgb: np.ndarray,
#     goal_mask: np.ndarray,
#     obstacle_mask: np.ndarray,
#     text: str,
#     *,
#     show_masks: bool,
#     detection_box: Optional[np.ndarray] = None,
#     detection_label: Optional[str] = None,
# ) -> Image.Image:
#     output = np.asarray(rgb, dtype=np.uint8).copy()
#     if show_masks:
#         output[goal_mask > 0] = (
#             0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
#         ).astype(np.uint8)
#         output[obstacle_mask > 0] = (
#             0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
#         ).astype(np.uint8)
#     image = Image.fromarray(output)
#     draw = ImageDraw.Draw(image)
#     if detection_box is not None:
#         x1, y1, x2, y2 = [float(value) for value in detection_box]
#         draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
#         if detection_label:
#             draw.text((x1 + 2, max(y1 - 14, 2)), detection_label, fill=(255, 255, 0))
#     draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
#     draw.text((10, 9), text, fill=(255, 255, 255))
#     return image


# def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
#     """Dilate a binary image mask without adding a SciPy dependency."""

#     binary = (np.asarray(mask) > 0).astype(np.uint8)
#     if radius <= 0 or not np.any(binary):
#         return binary
#     kernel_size = 2 * int(radius) + 1
#     return (
#         np.asarray(
#             Image.fromarray(binary * 255).filter(ImageFilter.MaxFilter(kernel_size))
#         )
#         > 0
#     ).astype(np.uint8)


# def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
#     import imageio.v2 as imageio

#     with imageio.get_writer(path, fps=float(fps)) as writer:
#         for frame in frames:
#             writer.append_data(np.asarray(frame.convert("RGB")))


# def parser() -> argparse.ArgumentParser:
#     argument_parser = argparse.ArgumentParser(
#         description="One-file released NavDP + in-denoising S2Diff Mars rollout"
#     )
#     argument_parser.add_argument("--navdp-root", required=True)
#     argument_parser.add_argument("--navdp-checkpoint", required=True)
#     argument_parser.add_argument("--navdp-python", default=sys.executable)
#     argument_parser.add_argument("--navdp-device", default="cuda:0")
#     argument_parser.add_argument(
#         "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
#     )
#     argument_parser.add_argument(
#         "--goal-mode", choices=["point", "pixel"], default="point"
#     )
#     argument_parser.add_argument(
#         "--belief-pixel-goal",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help=(
#             "Use the live semantic goal mask to correct a body-frame Gaussian "
#             "belief and its projected mean as NavDP's PixelGoal while occluded."
#         ),
#     )
#     argument_parser.add_argument("--belief-minimum-goal-pixels", type=int, default=10)
#     argument_parser.add_argument("--belief-measurement-std", type=float, default=0.05)
#     argument_parser.add_argument(
#         "--belief-translation-process-std", type=float, default=0.03
#     )
#     argument_parser.add_argument(
#         "--belief-yaw-process-std-deg", type=float, default=1.0
#     )
#     argument_parser.add_argument(
#         "--belief-bootstrap-world-goal",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help=(
#             "Simulation-only bootstrap when the goal is initially invisible. "
#             "Disable for a strict detector-only evaluation."
#         ),
#     )
#     argument_parser.add_argument("--belief-bootstrap-std", type=float, default=0.50)
#     argument_parser.add_argument("--belief-ghost-base-radius", type=int, default=10)
#     argument_parser.add_argument(
#         "--belief-ghost-covariance-scale", type=float, default=2.0
#     )
#     argument_parser.add_argument("--belief-ghost-maximum-radius", type=int, default=80)
#     argument_parser.add_argument(
#         "--belief-heading-recovery",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#     )
#     argument_parser.add_argument(
#         "--belief-recovery-bearing-deg", type=float, default=35.0
#     )
#     argument_parser.add_argument("--belief-recovery-yaw-gain", type=float, default=1.5)
#     argument_parser.add_argument(
#         "--belief-recovery-maximum-yaw-rate", type=float, default=0.70
#     )
#     argument_parser.add_argument(
#         "--belief-recovery-maximum-forward-speed", type=float, default=0.12
#     )
#     argument_parser.add_argument(
#         "--interactive-return-home",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help=(
#             "At the outward goal, ask for a command, let Qwen classify RETURN "
#             "or STOP, and use a separately propagated spawn/home PixelGoal belief."
#         ),
#     )
#     argument_parser.add_argument(
#         "--return-command",
#         default=None,
#         help="Optional non-interactive command text, for example 'come back'.",
#     )
#     argument_parser.add_argument(
#         "--qwen-freeform-mission",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help=(
#             "Ask once at startup for a free-form instruction. Qwen emits either "
#             "GO_TO_GOAL or GO_TO_GOAL followed by RETURN_HOME."
#         ),
#     )
#     argument_parser.add_argument(
#         "--mission-command",
#         default=None,
#         help=(
#             "Optional non-interactive free-form mission, for example "
#             "'visit the target and report back'."
#         ),
#     )
#     argument_parser.add_argument(
#         "--return-goal-obstacle-activation-distance", type=float, default=1.35
#     )
#     argument_parser.add_argument(
#         "--return-goal-obstacle-dilation-pixels", type=int, default=30
#     )
#     argument_parser.add_argument(
#         "--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct"
#     )
#     argument_parser.add_argument("--qwen-device", default="auto")
#     argument_parser.add_argument("--qwen-homotopy-python", default=sys.executable)
#     argument_parser.add_argument("--qwen-homotopy-host", default="127.0.0.1")
#     argument_parser.add_argument("--qwen-homotopy-port", type=int, default=8890)
#     argument_parser.add_argument("--qwen-homotopy-timeout", type=float, default=600.0)
#     argument_parser.add_argument(
#         "--start-qwen-homotopy-server",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#     )
#     argument_parser.add_argument(
#         "--qwen-homotopy",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help=(
#             "When a metric obstacle becomes relevant, Qwen chooses the single "
#             "LEFT/RIGHT circulation sign used by every trajectory candidate."
#         ),
#     )
#     argument_parser.add_argument(
#         "--homotopy-minimum-obstacle-pixels", type=int, default=30
#     )
#     argument_parser.add_argument("--homotopy-release-clear-frames", type=int, default=8)
#     argument_parser.add_argument(
#         "--homotopy-consistency-repeats",
#         type=int,
#         default=5,
#         help="Repeat Qwen on the identical obstacle frame and use majority vote.",
#     )
#     argument_parser.add_argument(
#         "--remove-critic", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--seed", type=int, default=7)
#     argument_parser.add_argument(
#         "--start-server", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--server-host", default="127.0.0.1")
#     argument_parser.add_argument("--server-port", type=int, default=8888)
#     argument_parser.add_argument("--server-timeout", type=float, default=180.0)
#     argument_parser.add_argument("--candidates", type=int, default=16)
#     argument_parser.add_argument("--particles", type=int, default=8)
#     argument_parser.add_argument("--particle-std", type=float, default=0.22)
#     argument_parser.add_argument("--gradient-steps", type=int, default=3)
#     argument_parser.add_argument("--gradient-step-size", type=float, default=0.04)
#     argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
#     argument_parser.add_argument("--temperature", type=float, default=0.35)
#     argument_parser.add_argument(
#         "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--particle-energy-reweighting",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#     )
#     argument_parser.add_argument(
#         "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--safe-distance", type=float, default=0.42)
#     argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
#     argument_parser.add_argument("--safety-weight", type=float, default=35.0)
#     argument_parser.add_argument("--barrier-weight", type=float, default=25.0)
#     argument_parser.add_argument("--barrier-rate", type=float, default=0.15)
#     argument_parser.add_argument("--circulation-weight", type=float, default=18.0)
#     argument_parser.add_argument(
#         "--circulation-activation-distance", type=float, default=1.50
#     )
#     argument_parser.add_argument(
#         "--circulation-activation-sharpness", type=float, default=0.20
#     )
#     argument_parser.add_argument(
#         "--minimum-circulation-progress", type=float, default=0.025
#     )
#     argument_parser.add_argument(
#         "--blocking-alignment-threshold", type=float, default=0.25
#     )
#     argument_parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
#     argument_parser.add_argument("--escape-lateral-target", type=float, default=0.35)
#     argument_parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
#     argument_parser.add_argument("--maximum-obstacle-depth", type=float, default=5.0)
#     argument_parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)

#     argument_parser.add_argument("--scene", required=True)
#     argument_parser.add_argument("--terrain-obj", default=None)
#     argument_parser.add_argument("--heightmap", default=None)
#     argument_parser.add_argument(
#         "--terrain-height-mode",
#         choices=["auto", "heightmap", "obj", "flat"],
#         default="auto",
#     )
#     argument_parser.add_argument("--flat-y", type=float, default=0.0)
#     argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
#     argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
#     argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
#     argument_parser.add_argument("--flip-heightmap-x", action="store_true")
#     argument_parser.add_argument(
#         "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--swap-heightmap-xz", action="store_true")
#     argument_parser.add_argument("--clearance", type=float, default=1.4)
#     argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)
#     argument_parser.add_argument(
#         "--robot-radius",
#         type=float,
#         default=0.24,
#         help="Planar rover footprint radius used by both guidance and evaluation.",
#     )
#     argument_parser.add_argument(
#         "--evaluation-layout",
#         default="default",
#         help="Stable layout identifier stored in the rollout archive.",
#     )

#     argument_parser.add_argument("--height", type=int, default=720)
#     argument_parser.add_argument("--width", type=int, default=720)
#     argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
#     argument_parser.add_argument("--hz", type=float, default=10.0)
#     argument_parser.add_argument("--max-steps", type=int, default=300)
#     argument_parser.add_argument("--stop-distance", type=float, default=1.0)
#     argument_parser.add_argument("--start-x", type=float, default=0.0)
#     argument_parser.add_argument("--start-z", type=float, default=8.0)
#     argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
#     argument_parser.add_argument("--goal-x", type=float, default=None)
#     argument_parser.add_argument("--goal-z", type=float, default=None)
#     argument_parser.add_argument("--goal-y", type=float, default=None)
#     argument_parser.add_argument("--goal-height", type=float, default=1.2)
#     argument_parser.add_argument("--goal-radius", type=int, default=18)
#     argument_parser.add_argument(
#         "--goal-mesh", action=argparse.BooleanOptionalAction, default=False
#     )
#     argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
#     argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)

#     argument_parser.add_argument(
#         "--obstacle-mode", choices=["none", "depth", "mesh", "ghost"], default="none"
#     )
#     argument_parser.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
#     argument_parser.add_argument("--obstacle-min-y-fraction", type=float, default=0.45)
#     argument_parser.add_argument("--ghost-obstacle-x", type=float, default=None)
#     argument_parser.add_argument("--ghost-obstacle-z", type=float, default=None)
#     argument_parser.add_argument("--ghost-obstacle-y", type=float, default=None)
#     argument_parser.add_argument("--ghost-obstacle-height", type=float, default=0.45)
#     argument_parser.add_argument("--ghost-obstacle-radius", type=int, default=24)
#     argument_parser.add_argument(
#         "--obstacle-mesh-uv",
#         nargs="+",
#         default=[],
#         help=(
#             "Actual rendered obstacle mesh locations as image fractions u,v. "
#             "Example: --obstacle-mesh-uv 0.50,0.72 0.30,0.68"
#         ),
#     )
#     argument_parser.add_argument(
#         "--obstacle-world-xz",
#         nargs="*",
#         default=[],
#         metavar="X,Z",
#         help=(
#             "Static rendered obstacle-box centers in world X,Z coordinates. "
#             "Example: --obstacle-world-xz 0,0. Do not combine with "
#             "--obstacle-mesh-uv."
#         ),
#     )
#     argument_parser.add_argument(
#         "--obstacle-world-xz-item",
#         action="append",
#         default=[],
#         metavar="X,Z",
#         help=(
#             "Repeatable form that safely accepts negative coordinates, e.g. "
#             "--obstacle-world-xz-item=-3,0."
#         ),
#     )
#     argument_parser.add_argument(
#         "--world-obstacle-half-extent", type=float, default=0.75
#     )
#     argument_parser.add_argument("--world-obstacle-height", type=float, default=1.40)
#     argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
#     argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)
#     argument_parser.add_argument(
#         "--obstacle-velocity-xz",
#         nargs="*",
#         default=[],
#         metavar="VX,VZ",
#         help=(
#             "World-frame mesh velocities in m/s. Supply one value to broadcast "
#             "or one value per obstacle. Example: --obstacle-velocity-xz 0.30,0.0"
#         ),
#     )

#     argument_parser.add_argument("--lookahead-index", type=int, default=4)
#     argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
#     argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
#     argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
#     argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
#     argument_parser.add_argument("--save-every", type=int, default=1)
#     argument_parser.add_argument(
#         "--save-frames", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--save-video", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--archive-observations",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help="Store RGB/depth/masks in rollout.npz; disable for large evaluations.",
#     )
#     argument_parser.add_argument(
#         "--overlay-masks", action=argparse.BooleanOptionalAction, default=True
#     )
#     return argument_parser


# def main() -> None:
#     args = parser().parse_args()
#     if args.obstacle_world_xz_item:
#         args.obstacle_world_xz.extend(args.obstacle_world_xz_item)
#     np.random.seed(args.seed)
#     if args.goal_x is None or args.goal_z is None:
#         raise ValueError("fixed PointGoal requires --goal-x and --goal-z")
#     if args.belief_pixel_goal and args.goal_mode != "pixel":
#         raise ValueError("--belief-pixel-goal requires --goal-mode pixel")
#     if args.belief_pixel_goal and not args.goal_mesh:
#         raise ValueError(
#             "simulation belief tracking requires --goal-mesh so a live semantic "
#             "goal observation exists"
#         )
#     if args.belief_minimum_goal_pixels < 1:
#         raise ValueError("belief-minimum-goal-pixels must be positive")
#     return_home_enabled = args.interactive_return_home or args.qwen_freeform_mission
#     if return_home_enabled and not args.belief_pixel_goal:
#         raise ValueError("return-home modes require --belief-pixel-goal")
#     if return_home_enabled and not args.qwen_homotopy:
#         raise ValueError("return-home modes require --qwen-homotopy")
#     if args.mission_command is not None and not args.qwen_freeform_mission:
#         raise ValueError("--mission-command requires --qwen-freeform-mission")
#     if args.return_goal_obstacle_activation_distance <= 0.0:
#         raise ValueError("return goal obstacle activation distance must be positive")
#     if args.return_goal_obstacle_dilation_pixels < 0:
#         raise ValueError("return goal obstacle dilation pixels must be non-negative")
#     if (
#         min(
#             args.belief_measurement_std,
#             args.belief_translation_process_std,
#             args.belief_yaw_process_std_deg,
#             args.belief_bootstrap_std,
#         )
#         < 0.0
#     ):
#         raise ValueError("belief uncertainty parameters must be non-negative")
#     if args.robot_radius < 0.0:
#         raise ValueError("robot-radius must be non-negative")
#     if args.obstacle_velocity_xz and args.obstacle_mode != "mesh":
#         raise ValueError("moving obstacle velocities require --obstacle-mode mesh")
#     if args.obstacle_mode == "ghost" and (
#         args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
#     ):
#         raise ValueError(
#             "ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z"
#         )
#     if args.obstacle_mesh_uv and args.obstacle_world_xz:
#         raise ValueError(
#             "choose either --obstacle-mesh-uv or --obstacle-world-xz, not both"
#         )
#     if args.obstacle_mode == "mesh" and not (
#         args.obstacle_mesh_uv or args.obstacle_world_xz
#     ):
#         raise ValueError(
#             "mesh mode requires --obstacle-world-xz X,Z [X,Z ...] or "
#             "--obstacle-mesh-uv u,v [u,v ...]"
#         )
#     if args.world_obstacle_half_extent <= 0.0 or args.world_obstacle_height <= 0.0:
#         raise ValueError("world obstacle dimensions must be positive")
#     if args.goal_mesh_half_extent <= 0.0 or args.goal_mesh_height <= 0.0:
#         raise ValueError("goal mesh dimensions must be positive")

#     if args.qwen_homotopy and args.planner_mode == "pure-navdp":
#         raise ValueError("Qwen homotopy conditioning requires s2diff or gradient mode")

#     qwen_process: Optional[subprocess.Popen[Any]] = None
#     server_process: Optional[subprocess.Popen[Any]] = None
#     simulator = None
#     try:
#         qwen_process = start_qwen_homotopy_server(args)
#         homotopy_selector = None
#         if args.qwen_homotopy:
#             homotopy_selector = QwenHomotopyClient(
#                 f"http://{args.qwen_homotopy_host}:{args.qwen_homotopy_port}",
#                 timeout=args.qwen_homotopy_timeout,
#             )
#             homotopy_selector.reset()
#         server_process = start_server(args)
#         server_url = f"http://{args.server_host}:{args.server_port}"
#         client = NavDPS2DiffClient(server_url)
#         algorithm = client.reset(
#             camera_intrinsic(args.height, args.width, args.hfov_deg),
#             batch_size=1,
#             stop_threshold=-3.0,
#         )
#         supported_algorithms = {
#             "navdp-s2diff-pixels",
#             "navdp-hlc-s2diff",
#             "navdp-hlc-s2diff-no-critic",
#             "navdp-hlc-gradient",
#             "navdp-hlc-gradient-no-critic",
#             "navdp-pure-critic",
#         }
#         if algorithm not in supported_algorithms:
#             raise RuntimeError(f"unexpected planner response: {algorithm!r}")

#         terrain = TerrainHeight(
#             mode=args.terrain_height_mode,
#             heightmap=(
#                 Path(args.heightmap).expanduser().resolve() if args.heightmap else None
#             ),
#             obj=(
#                 Path(args.terrain_obj).expanduser().resolve()
#                 if args.terrain_obj
#                 else None
#             ),
#             flat_y=args.flat_y,
#             size_x=args.size_x,
#             size_z=args.size_z,
#             size_y=args.size_y,
#             flip_x=args.flip_heightmap_x,
#             flip_z=args.flip_heightmap_z,
#             swap_xz=args.swap_heightmap_xz,
#         )
#         output_directory = Path(args.output).expanduser().resolve()
#         frame_directory = output_directory / "frames"
#         frame_directory.mkdir(parents=True, exist_ok=True)

#         simulator = make_simulator(
#             Path(args.scene),
#             args.height,
#             args.width,
#             args.hfov_deg,
#             with_semantic=args.obstacle_mode == "mesh" or args.goal_mesh,
#         )
#         agent = simulator.initialize_agent(0)
#         intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
#         goal_belief = (
#             GaussianGoalBelief(
#                 intrinsic,
#                 (args.height, args.width),
#                 minimum_visible_pixels=args.belief_minimum_goal_pixels,
#                 measurement_std=args.belief_measurement_std,
#                 translation_process_std=args.belief_translation_process_std,
#                 yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
#             )
#             if args.belief_pixel_goal
#             else None
#         )
#         previous_executed_action = np.zeros(3, dtype=np.float32)
#         home_belief = (
#             GaussianGoalBelief(
#                 intrinsic,
#                 (args.height, args.width),
#                 minimum_visible_pixels=args.belief_minimum_goal_pixels,
#                 measurement_std=args.belief_measurement_std,
#                 translation_process_std=args.belief_translation_process_std,
#                 yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
#             )
#             if return_home_enabled
#             else None
#         )
#         if home_belief is not None:
#             # Home starts at the rover origin. Every executed action propagates
#             # this stationary world location into the current body frame.
#             home_belief.initialize(
#                 np.zeros(2, dtype=np.float32),
#                 args.belief_measurement_std,
#                 visible=False,
#             )
#         mission_phase = "OUTBOUND"
#         return_goal_obstacle_active = False
#         return_command_event: Optional[dict[str, Any]] = None
#         mission_plan_event: Optional[dict[str, Any]] = None
#         mission_plan_pending = bool(args.qwen_freeform_mission)
#         automatic_return_requested = False
#         roundtrip_completed = False
#         x, z = float(args.start_x), float(args.start_z)
#         yaw = math.radians(float(args.start_yaw_deg))
#         dt = 1.0 / float(args.hz)

#         goal_y = args.goal_y
#         if goal_y is None:
#             goal_y = (
#                 terrain.local_height_max(args.goal_x, args.goal_z, 0.8)
#                 + args.goal_height
#             )
#         goal = np.asarray([args.goal_x, goal_y, args.goal_z], dtype=np.float32)
#         start_position_xz = np.asarray([x, z], dtype=np.float64)
#         initial_goal_distance = float(
#             np.linalg.norm(goal[[0, 2]].astype(np.float64) - start_position_xz)
#         )
#         goal_mesh_object = None
#         if args.goal_mesh:
#             goal_mesh_object = place_world_goal_mesh(
#                 simulator,
#                 terrain,
#                 args.goal_x,
#                 args.goal_z,
#                 output_directory,
#                 half_extent=args.goal_mesh_half_extent,
#                 height=args.goal_mesh_height,
#             )

#         ghost = None
#         if args.obstacle_mode == "ghost":
#             ghost_y = args.ghost_obstacle_y
#             if ghost_y is None:
#                 ghost_y = (
#                     terrain.local_height_max(
#                         args.ghost_obstacle_x,
#                         args.ghost_obstacle_z,
#                         args.pose_terrain_radius,
#                     )
#                     + args.ghost_obstacle_height
#                 )
#             ghost = np.asarray(
#                 [args.ghost_obstacle_x, ghost_y, args.ghost_obstacle_z],
#                 dtype=np.float32,
#             )

#         mesh_objects: list[Any] = []
#         mesh_centroids: list[np.ndarray] = []
#         mesh_current_centroids: list[np.ndarray] = []
#         mesh_base_geometries: list[np.ndarray] = []
#         mesh_geometries: list[np.ndarray] = []
#         mesh_velocities = np.zeros((0, 2), dtype=np.float64)
#         mesh_placed = False
#         if args.obstacle_mode == "mesh" and args.obstacle_world_xz:
#             mesh_objects, mesh_centroids, mesh_base_geometries = (
#                 place_world_obstacle_meshes(
#                     simulator,
#                     terrain,
#                     args.obstacle_world_xz,
#                     output_directory,
#                     half_extent=args.world_obstacle_half_extent,
#                     height=args.world_obstacle_height,
#                 )
#             )
#             mesh_velocities = expand_obstacle_velocities(
#                 args.obstacle_velocity_xz, len(mesh_objects)
#             )
#             mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
#                 mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
#             )
#             mesh_placed = True

#         row_keys = [
#             "pose",
#             "action_3d",
#             "mission_phase",
#             "return_goal_obstacle_active",
#             "point_goal",
#             "belief_goal_mu",
#             "belief_goal_covariance",
#             "belief_goal_pixel",
#             "belief_goal_visible",
#             "belief_goal_source",
#             "belief_goal_time_since_seen",
#             "belief_goal_bearing_rad",
#             "belief_goal_pixel_sigma",
#             "belief_heading_recovery_active",
#             "selected_trajectory",
#             "all_trajectories",
#             "all_values",
#             "selected_index",
#             "fallback_stop",
#             "escape_turn",
#             "valid_obstacle_points",
#             "selected_circulation_sign",
#             "candidate_circulation_signs",
#             "selected_barrier_energy",
#             "selected_circulation_energy",
#             "planning_time_seconds",
#             "selected_minimum_clearance",
#             "mean_guidance_noise_correction",
#             "final_guidance_noise_correction",
#             "maximum_guidance_noise_correction",
#             "mean_final_effective_sample_size",
#             "goal_distance",
#             "executed_center_clearance",
#             "executed_surface_clearance",
#             "geometric_collision",
#             "obstacle_positions_world",
#             "qwen_homotopy_sign",
#             "qwen_homotopy_side",
#             "qwen_homotopy_confidence",
#             "qwen_homotopy_queried",
#         ]
#         if args.archive_observations:
#             row_keys.extend(
#                 (
#                     "rgb",
#                     "depth",
#                     "goal_mask",
#                     "live_goal_mask",
#                     "ghost_goal_mask",
#                     "obstacle_mask",
#                 )
#             )
#         rows: dict[str, list[Any]] = {key: [] for key in row_keys}
#         video_frames: list[Image.Image] = []
#         success = False
#         homotopy_events: list[dict[str, Any]] = []

#         for step in range(int(args.max_steps)):
#             y = (
#                 terrain.local_height_max(x, z, args.pose_terrain_radius)
#                 + args.clearance
#             )
#             position = np.asarray([x, y, z], dtype=np.float32)
#             if goal_belief is not None and step > 0:
#                 goal_belief.predict(previous_executed_action, dt)
#             if home_belief is not None and step > 0:
#                 home_belief.predict(previous_executed_action, dt)
#             set_agent_pose(agent, position, yaw)
#             if mesh_placed:
#                 elapsed_seconds = step * dt
#                 move_mesh_objects(mesh_objects, mesh_velocities, elapsed_seconds)
#                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
#                     mesh_base_geometries,
#                     mesh_centroids,
#                     mesh_velocities,
#                     elapsed_seconds,
#                 )
#             observation = simulator.get_sensor_observations()
#             rgb, depth = rgb_depth(observation)

#             if args.obstacle_mode == "mesh" and not mesh_placed:
#                 mesh_objects, mesh_centroids, mesh_base_geometries = (
#                     place_obstacle_meshes(
#                         simulator,
#                         depth,
#                         position,
#                         yaw,
#                         intrinsic,
#                         args.obstacle_mesh_uv,
#                         output_directory,
#                         mesh_half_pixels=args.mesh_half_pixels,
#                         mesh_lift=args.mesh_obstacle_lift,
#                     )
#                 )
#                 mesh_velocities = expand_obstacle_velocities(
#                     args.obstacle_velocity_xz, len(mesh_objects)
#                 )
#                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
#                     mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
#                 )
#                 mesh_placed = True
#                 observation = simulator.get_sensor_observations()
#                 rgb, depth = rgb_depth(observation)

#             if mission_plan_pending:
#                 mission_command = args.mission_command
#                 if mission_command is None:
#                     print(
#                         "\nWhat should the rover do? You may use a vague command, "
#                         "for example: 'visit the goal and come back'.",
#                         flush=True,
#                     )
#                     try:
#                         mission_command = input("> ").strip()
#                     except EOFError as error:
#                         raise RuntimeError(
#                             "no startup command was available; set "
#                             "--mission-command for a non-interactive run"
#                         ) from error
#                 assert homotopy_selector is not None
#                 mission_decision = homotopy_selector.classify_mission(
#                     rgb, mission_command
#                 )
#                 automatic_return_requested = mission_decision.plan == (
#                     "GO_TO_GOAL",
#                     "RETURN_HOME",
#                 )
#                 mission_plan_event = {
#                     "step": step,
#                     "user_command": mission_command,
#                     "plan": list(mission_decision.plan),
#                     "confidence": mission_decision.confidence,
#                     "raw_response": mission_decision.raw_response,
#                 }
#                 Image.fromarray(rgb).save(output_directory / "qwen_mission_frame.png")
#                 print(
#                     f"[qwen-mission] text={mission_command!r} "
#                     f"plan={list(mission_decision.plan)} "
#                     f"confidence={mission_decision.confidence:.2f}",
#                     flush=True,
#                 )
#                 mission_plan_pending = False

#             semantic = (
#                 semantic_from_observation(observation)
#                 if args.obstacle_mode == "mesh" or args.goal_mesh
#                 else None
#             )
#             goal_right, _goal_up, goal_forward = camera_coordinates(goal, position, yaw)
#             point_goal = np.asarray(
#                 [max(goal_forward, 0.0), -goal_right], dtype=np.float32
#             )
#             live_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
#             ghost_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
#             belief_goal_visible = False
#             belief_goal_source = "DISABLED"
#             belief_goal_mu = np.full(2, np.nan, dtype=np.float32)
#             belief_goal_covariance = np.full((2, 2), np.nan, dtype=np.float32)
#             belief_goal_pixel = np.full(2, -1, dtype=np.int32)
#             belief_goal_time_since_seen = float("nan")
#             belief_goal_bearing = float("nan")
#             belief_goal_pixel_sigma = float("nan")

#             if goal_belief is not None:
#                 assert semantic is not None
#                 live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
#                 bootstrapped = False
#                 if mission_phase == "OUTBOUND":
#                     active_goal_belief = goal_belief
#                     belief_goal_visible = goal_belief.observe(live_goal_mask, depth)
#                     if not goal_belief.initialized:
#                         if not args.belief_bootstrap_world_goal:
#                             raise RuntimeError(
#                                 "goal belief is uninitialized because the live goal "
#                                 "mask has not been observed; start with the goal visible "
#                                 "or pass --belief-bootstrap-world-goal for simulation"
#                             )
#                         goal_belief.initialize(
#                             np.asarray([goal_forward, -goal_right], dtype=np.float32),
#                             args.belief_bootstrap_std,
#                         )
#                         bootstrapped = True
#                 else:
#                     assert home_belief is not None and home_belief.initialized
#                     active_goal_belief = home_belief
#                     belief_goal_visible = False

#                 belief_projection = active_goal_belief.project(
#                     base_radius=args.belief_ghost_base_radius,
#                     covariance_scale=args.belief_ghost_covariance_scale,
#                     maximum_radius=args.belief_ghost_maximum_radius,
#                 )
#                 planner_goal = belief_projection.pixel_uv
#                 ghost_goal_mask = belief_projection.mask
#                 if mission_phase == "OUTBOUND" and belief_goal_visible:
#                     goal_mask = live_goal_mask
#                     belief_goal_source = "LIVE"
#                 else:
#                     goal_mask = ghost_goal_mask
#                     belief_goal_source = (
#                         "HOME_BELIEF"
#                         if mission_phase == "RETURN_HOME"
#                         else ("WORLD_BOOTSTRAP" if bootstrapped else "GHOST")
#                     )
#                 assert (
#                     active_goal_belief.mu is not None
#                     and active_goal_belief.Sigma is not None
#                 )
#                 belief_goal_mu = active_goal_belief.mu.copy()
#                 belief_goal_covariance = active_goal_belief.Sigma.copy()
#                 belief_goal_pixel = belief_projection.pixel_uv.copy()
#                 belief_goal_time_since_seen = active_goal_belief.time_since_seen
#                 belief_goal_bearing = belief_projection.bearing_rad
#                 belief_goal_pixel_sigma = belief_projection.pixel_sigma
#             else:
#                 if args.goal_mesh:
#                     assert semantic is not None
#                     live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
#                     goal_mask = live_goal_mask
#                     if not np.any(goal_mask):
#                         goal_mask, _ = project_world_mask(
#                             goal,
#                             position,
#                             yaw,
#                             intrinsic,
#                             args.height,
#                             args.width,
#                             args.goal_radius,
#                         )
#                 else:
#                     goal_mask, _ = project_world_mask(
#                         goal,
#                         position,
#                         yaw,
#                         intrinsic,
#                         args.height,
#                         args.width,
#                         args.goal_radius,
#                     )
#                 planner_goal = point_goal
#             if args.goal_mode == "pixel" and goal_belief is None:
#                 planner_goal = world_goal_to_pixel(
#                     goal, position, yaw, intrinsic, args.height, args.width
#                 )
#                 goal_mask = circle_mask(
#                     args.height,
#                     args.width,
#                     planner_goal[0],
#                     planner_goal[1],
#                     args.goal_radius,
#                 )
#             guidance_depth = depth.copy()
#             if args.obstacle_mode == "depth":
#                 obstacle_mask = depth_obstacle_mask(
#                     depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
#                 )
#             elif args.obstacle_mode == "mesh":
#                 assert semantic is not None
#                 semantic_ids = list(
#                     range(
#                         MESH_OBSTACLE_ID,
#                         MESH_OBSTACLE_ID + len(mesh_objects),
#                     )
#                 )
#                 obstacle_mask = np.isin(semantic, semantic_ids).astype(np.uint8)
#                 # The depth image was re-rendered after mesh placement, so
#                 # guidance_depth already contains the real obstacle depth.
#             elif args.obstacle_mode == "ghost":
#                 assert ghost is not None
#                 obstacle_mask, obstacle_forward = project_world_mask(
#                     ghost,
#                     position,
#                     yaw,
#                     intrinsic,
#                     args.height,
#                     args.width,
#                     args.ghost_obstacle_radius,
#                 )
#                 if obstacle_forward > 0.05:
#                     guidance_depth[obstacle_mask > 0] = obstacle_forward
#             else:
#                 obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)

#             # Replace this mask-to-pixels line with your own detector's [u,v]
#             if mission_phase == "RETURN_HOME":
#                 distance_from_reached_goal = float(
#                     np.linalg.norm(goal[[0, 2]] - position[[0, 2]])
#                 )
#                 if (
#                     not return_goal_obstacle_active
#                     and distance_from_reached_goal
#                     >= args.return_goal_obstacle_activation_distance
#                 ):
#                     return_goal_obstacle_active = True
#                     print(
#                         "[roundtrip] reached-goal keep-out is now active",
#                         flush=True,
#                     )
#                 if return_goal_obstacle_active and np.any(live_goal_mask):
#                     reached_goal_keepout = dilate_binary_mask(
#                         live_goal_mask,
#                         args.return_goal_obstacle_dilation_pixels,
#                     )
#                     obstacle_mask = (
#                         (obstacle_mask > 0) | (reached_goal_keepout > 0)
#                     ).astype(np.uint8)
#                     target_depths = depth[
#                         (live_goal_mask > 0)
#                         & np.isfinite(depth)
#                         & (depth > args.minimum_obstacle_depth)
#                     ]
#                     if target_depths.size:
#                         guidance_depth[reached_goal_keepout > 0] = float(
#                             np.median(target_depths)
#                         )

#             # array if obstacle pixels already come directly from your system.
#             obstacle_pixels = pixels_from_mask(
#                 obstacle_mask, args.maximum_obstacle_pixels
#             )
#             homotopy_decision = None
#             forced_circulation_sign = 0.0
#             obstacle_relevant_for_homotopy = False
#             if homotopy_selector is not None:
#                 homotopy_obstacle_mask = (
#                     (obstacle_mask > 0)
#                     & np.isfinite(guidance_depth)
#                     & (guidance_depth >= args.minimum_obstacle_depth)
#                     & (guidance_depth <= args.maximum_obstacle_depth)
#                 ).astype(np.uint8)
#                 qwen_overlay = overlay_frame(
#                     rgb,
#                     goal_mask,
#                     homotopy_obstacle_mask,
#                     "Qwen homotopy: choose LEFT or RIGHT",
#                     show_masks=True,
#                 )
#                 homotopy_decision = homotopy_selector.step(
#                     np.asarray(qwen_overlay.convert("RGB")), homotopy_obstacle_mask
#                 )
#                 obstacle_relevant_for_homotopy = homotopy_decision.obstacle_relevant
#                 forced_circulation_sign = homotopy_decision.circulation_sign
#                 if homotopy_decision.queried_qwen:
#                     event = {
#                         "step": step,
#                         "side": homotopy_decision.side,
#                         "circulation_sign": forced_circulation_sign,
#                         "confidence": homotopy_decision.confidence,
#                         "repeat_sides": list(homotopy_decision.repeated_sides),
#                         "repeat_confidences": list(
#                             homotopy_decision.repeated_confidences
#                         ),
#                         "consistency_rate": homotopy_decision.consistency_rate,
#                         "used_fallback": homotopy_decision.used_fallback,
#                         "raw_response": homotopy_decision.raw_response,
#                     }
#                     homotopy_events.append(event)
#                     query_directory = output_directory / "qwen_homotopy_queries"
#                     query_directory.mkdir(parents=True, exist_ok=True)
#                     qwen_overlay.save(query_directory / f"query_step_{step:04d}.png")
#                     print(
#                         f"[qwen-homotopy] side={homotopy_decision.side} "
#                         f"sign={forced_circulation_sign:+.0f} "
#                         f"confidence={homotopy_decision.confidence:.2f} "
#                         f"consistency={homotopy_decision.consistency_rate:.2%} "
#                         f"repeats={list(homotopy_decision.repeated_sides)} "
#                         f"fallback={homotopy_decision.used_fallback}",
#                         flush=True,
#                     )
#             planning_start = time.perf_counter()
#             result = client.plan(
#                 goal_xy=planner_goal,
#                 rgb=rgb,
#                 depth=guidance_depth,
#                 obstacle_pixels=obstacle_pixels,
#                 goal_mode=args.goal_mode,
#                 forced_circulation_sign=forced_circulation_sign,
#             )
#             planning_time = time.perf_counter() - planning_start
#             action = (
#                 np.zeros(3, dtype=np.float32)
#                 if result.fallback_stop
#                 else waypoint_action(
#                     result.trajectory,
#                     lookahead_index=args.lookahead_index,
#                     maximum_forward_speed=args.maximum_forward_speed,
#                     maximum_yaw_rate=args.maximum_yaw_rate,
#                     yaw_gain=args.yaw_gain,
#                 )
#             )
#             action, belief_recovery_active = belief_heading_recovery_action(
#                 action,
#                 belief_bearing=belief_goal_bearing,
#                 obstacle_relevant=obstacle_relevant_for_homotopy,
#                 enabled=args.belief_pixel_goal and args.belief_heading_recovery,
#                 activation_bearing=math.radians(args.belief_recovery_bearing_deg),
#                 yaw_gain=args.belief_recovery_yaw_gain,
#                 maximum_yaw_rate=args.belief_recovery_maximum_yaw_rate,
#                 maximum_forward_speed=args.belief_recovery_maximum_forward_speed,
#             )

#             next_position, next_yaw = integrate_mars(position, yaw, action, dt)
#             previous_executed_action = action.copy()
#             x = float(
#                 np.clip(
#                     next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
#                 )
#             )
#             z = float(
#                 np.clip(
#                     next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
#                 )
#             )
#             rows["mission_phase"].append(mission_phase)
#             rows["return_goal_obstacle_active"].append(return_goal_obstacle_active)

#             yaw = wrap_angle(next_yaw)
#             outbound_goal_distance = float(
#                 np.linalg.norm(goal[[0, 2]] - np.asarray([x, z]))
#             )
#             home_distance = float(
#                 np.linalg.norm(start_position_xz - np.asarray([x, z]))
#             )
#             goal_distance = (
#                 home_distance
#                 if mission_phase == "RETURN_HOME"
#                 else outbound_goal_distance
#             )
#             center_clearance = planar_mesh_clearance(
#                 np.asarray([x, z], dtype=np.float64), mesh_geometries
#             )
#             if np.isfinite(center_clearance):
#                 surface_clearance = max(
#                     center_clearance - float(args.robot_radius), 0.0
#                 )
#                 geometric_collision = center_clearance <= float(args.robot_radius)
#             else:
#                 surface_clearance = float("nan")
#                 geometric_collision = False
#             rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
#             pose = np.asarray(
#                 [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w],
#                 dtype=np.float32,
#             )

#             if args.archive_observations:
#                 rows["rgb"].append(rgb)
#                 rows["depth"].append(depth)
#                 rows["goal_mask"].append(goal_mask)
#                 rows["live_goal_mask"].append(live_goal_mask)
#                 rows["ghost_goal_mask"].append(ghost_goal_mask)
#                 rows["obstacle_mask"].append(obstacle_mask)
#             rows["pose"].append(pose)
#             rows["action_3d"].append(action)
#             rows["point_goal"].append(planner_goal)
#             rows["belief_goal_mu"].append(belief_goal_mu)
#             rows["belief_goal_covariance"].append(belief_goal_covariance)
#             rows["belief_goal_pixel"].append(belief_goal_pixel)
#             rows["belief_goal_visible"].append(belief_goal_visible)
#             rows["belief_goal_source"].append(belief_goal_source)
#             rows["belief_goal_time_since_seen"].append(belief_goal_time_since_seen)
#             rows["belief_goal_bearing_rad"].append(belief_goal_bearing)
#             rows["belief_goal_pixel_sigma"].append(belief_goal_pixel_sigma)
#             rows["belief_heading_recovery_active"].append(belief_recovery_active)
#             rows["selected_trajectory"].append(result.trajectory)
#             rows["all_trajectories"].append(result.all_trajectories)
#             rows["all_values"].append(result.all_values)
#             rows["selected_index"].append(result.selected_index)
#             rows["fallback_stop"].append(result.fallback_stop)
#             rows["escape_turn"].append(result.escape_turn)
#             rows["valid_obstacle_points"].append(result.valid_obstacle_points)
#             rows["selected_circulation_sign"].append(result.selected_circulation_sign)
#             rows["candidate_circulation_signs"].append(
#                 result.candidate_circulation_signs
#             )
#             rows["selected_barrier_energy"].append(result.selected_barrier_energy)
#             rows["selected_circulation_energy"].append(
#                 result.selected_circulation_energy
#             )
#             rows["planning_time_seconds"].append(planning_time)
#             rows["selected_minimum_clearance"].append(result.selected_minimum_clearance)
#             rows["mean_guidance_noise_correction"].append(
#                 result.mean_guidance_noise_correction
#             )
#             rows["final_guidance_noise_correction"].append(
#                 result.final_guidance_noise_correction
#             )
#             rows["maximum_guidance_noise_correction"].append(
#                 result.maximum_guidance_noise_correction
#             )
#             rows["mean_final_effective_sample_size"].append(
#                 result.mean_final_effective_sample_size
#             )
#             rows["goal_distance"].append(goal_distance)
#             rows["executed_center_clearance"].append(center_clearance)
#             rows["executed_surface_clearance"].append(surface_clearance)
#             rows["geometric_collision"].append(geometric_collision)
#             rows["obstacle_positions_world"].append(
#                 np.stack(mesh_current_centroids)
#                 if mesh_current_centroids
#                 else np.zeros((0, 3), dtype=np.float64)
#             )

#             rows["qwen_homotopy_sign"].append(forced_circulation_sign)
#             rows["qwen_homotopy_side"].append(
#                 homotopy_decision.side if homotopy_decision is not None else "AUTO"
#             )
#             rows["qwen_homotopy_confidence"].append(
#                 homotopy_decision.confidence if homotopy_decision is not None else 0.0
#             )
#             rows["qwen_homotopy_queried"].append(
#                 homotopy_decision.queried_qwen
#                 if homotopy_decision is not None
#                 else False
#             )

#             if args.save_frames and step % max(int(args.save_every), 1) == 0:

#                 side_label = (
#                     homotopy_decision.side if homotopy_decision is not None else "AUTO"
#                 )
#                 label = (
#                     f"t={step} phase={mission_phase} goal={goal_distance:.2f}m "
#                     f"qwen_side={side_label} pixels={len(obstacle_pixels)} "
#                     f"goal_src={belief_goal_source} "
#                     f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
#                     f"recover={int(belief_recovery_active)} "
#                     f"pred={result.selected_minimum_clearance:.2f}m "
#                     f"actual={surface_clearance:.2f}m "
#                     f"mode={result.selected_circulation_sign:+.0f} "
#                     f"escape={int(result.escape_turn)} "
#                     f"guide_rms={result.mean_guidance_noise_correction:.4f} "
#                     f"v={action[0]:.2f} w={action[2]:.2f}"
#                 )
#                 frame = overlay_frame(
#                     rgb,
#                     goal_mask,
#                     obstacle_mask,
#                     label,
#                     show_masks=args.overlay_masks,
#                 )
#                 frame.save(frame_directory / f"frame_{step:04d}.png")
#                 video_frames.append(frame)

#             print(
#                 f"step={step:04d} phase={mission_phase} goal={goal_distance:.2f}m "
#                 f"qwen_side={homotopy_decision.side if homotopy_decision else 'AUTO'} "
#                 f"goal_src={belief_goal_source} "
#                 f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
#                 f"recover={int(belief_recovery_active)} "
#                 f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
#                 f"selected={result.selected_index} fallback={result.fallback_stop} "
#                 f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
#                 f"pred_clear={result.selected_minimum_clearance:.3f}m "
#                 f"actual_clear={surface_clearance:.3f}m "
#                 f"collision={geometric_collision} "
#                 f"barrier={result.selected_barrier_energy:.5f} "
#                 f"circ={result.selected_circulation_energy:.5f} "
#                 f"latency={planning_time * 1000.0:.1f}ms "
#                 f"guide_rms={result.mean_guidance_noise_correction:.6f} "
#                 f"ess={result.mean_final_effective_sample_size:.2f} "
#                 f"action={action.tolist()}",
#                 flush=True,
#             )
#             if goal_distance <= args.stop_distance:
#                 if args.qwen_freeform_mission and mission_phase == "OUTBOUND":
#                     if automatic_return_requested:
#                         print(
#                             "[mission] outward goal reached; advancing automatically "
#                             "to RETURN_HOME",
#                             flush=True,
#                         )
#                         mission_phase = "RETURN_HOME"
#                         assert homotopy_selector is not None
#                         homotopy_selector.reset()
#                         continue
#                     success = True
#                     print(
#                         "[mission] outward goal reached; GO_TO_GOAL plan complete",
#                         flush=True,
#                     )
#                     break

#                 if args.interactive_return_home and mission_phase == "OUTBOUND":
#                     user_command = args.return_command
#                     if user_command is None:
#                         print(
#                             "\nOutward goal reached. What should the rover do? "
#                             "(for example: come back / stop)",
#                             flush=True,
#                         )
#                         try:
#                             user_command = input("> ").strip()
#                         except EOFError as error:
#                             raise RuntimeError(
#                                 "no interactive command was available; set "
#                                 "--return-command 'come back' for a non-interactive run"
#                             ) from error
#                     assert homotopy_selector is not None
#                     command_overlay = overlay_frame(
#                         rgb,
#                         goal_mask,
#                         obstacle_mask,
#                         "Qwen command: RETURN or STOP",
#                         show_masks=True,
#                     )
#                     command_decision = homotopy_selector.classify_command(
#                         np.asarray(command_overlay.convert("RGB")),
#                         user_command,
#                     )
#                     return_command_event = {
#                         "step": step,
#                         "user_command": user_command,
#                         "command": command_decision.command,
#                         "confidence": command_decision.confidence,
#                         "raw_response": command_decision.raw_response,
#                     }
#                     command_overlay.save(output_directory / "qwen_return_command.png")
#                     print(
#                         f"[qwen-command] text={user_command!r} "
#                         f"decision={command_decision.command} "
#                         f"confidence={command_decision.confidence:.2f}",
#                         flush=True,
#                     )
#                     if command_decision.command == "RETURN":
#                         mission_phase = "RETURN_HOME"
#                         homotopy_selector.reset()
#                         continue
#                     success = True
#                     break
#                 success = True
#                 if mission_phase == "RETURN_HOME":
#                     roundtrip_completed = True
#                 break

#         if not rows["goal_distance"]:
#             raise RuntimeError("rollout produced no steps")
#         rollout_path = output_directory / "rollout.npz"
#         np.savez_compressed(
#             rollout_path,
#             **{
#                 key: (
#                     np.stack(values)
#                     if isinstance(values[0], np.ndarray)
#                     else np.asarray(values)
#                 )
#                 for key, values in rows.items()
#             },
#             goal_position=goal,
#             obstacle_position=(
#                 mesh_centroids[0]
#                 if mesh_centroids
#                 else (
#                     ghost
#                     if ghost is not None
#                     else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
#                 )
#             ),
#             obstacle_positions=(
#                 np.stack(mesh_centroids)
#                 if mesh_centroids
#                 else np.zeros((0, 3), dtype=np.float32)
#             ),
#             obstacle_velocity_xz=mesh_velocities,
#             success=np.asarray(success),
#             hz=np.asarray(args.hz, dtype=np.float32),
#             start_position_xz=start_position_xz,
#             initial_goal_distance=np.asarray(initial_goal_distance, dtype=np.float64),
#             stop_distance=np.asarray(args.stop_distance, dtype=np.float64),
#             robot_radius=np.asarray(args.robot_radius, dtype=np.float64),
#             evaluation_layout=np.asarray(args.evaluation_layout),
#             seed=np.asarray(args.seed, dtype=np.int64),
#             planner_mode=np.asarray(args.planner_mode),
#             candidate_count=np.asarray(args.candidates, dtype=np.int64),
#             particles_per_candidate=np.asarray(args.particles, dtype=np.int64),
#             particle_std=np.asarray(args.particle_std, dtype=np.float64),
#             guidance_strength=np.asarray(args.guidance_strength, dtype=np.float64),
#             temperature=np.asarray(args.temperature, dtype=np.float64),
#             particle_anchor=np.asarray(args.particle_anchor),
#             particle_energy_reweighting=np.asarray(args.particle_energy_reweighting),
#             particle_collision_mask=np.asarray(args.particle_collision_mask),
#             particle_noise_schedule=np.asarray(args.particle_noise_schedule),
#             progressive_guidance=np.asarray(args.progressive_guidance),
#             goal_mode=np.asarray(args.goal_mode),
#             belief_pixel_goal=np.asarray(args.belief_pixel_goal),
#             interactive_return_home=np.asarray(args.interactive_return_home),
#             qwen_freeform_mission=np.asarray(args.qwen_freeform_mission),
#             automatic_return_requested=np.asarray(automatic_return_requested),
#             roundtrip_completed=np.asarray(roundtrip_completed),
#             final_mission_phase=np.asarray(mission_phase),
#         )
#         with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
#             json.dump(
#                 {
#                     "success": success,
#                     "steps": len(rows["goal_distance"]),
#                     "archived_observations": args.archive_observations,
#                     "final_goal_distance": float(rows["goal_distance"][-1]),
#                     "planner": "released_navdp_s2diff_pixels",
#                     "controller": "direct_waypoint_no_optimizer",
#                     "qwen_role": "homotopy_return_command_and_mission_plan",
#                     "qwen_process_isolated_from_habitat": True,
#                     "qwen_creates_goal_or_action": False,
#                     "qwen_homotopy": args.qwen_homotopy,
#                     "qwen_homotopy_events": homotopy_events,
#                     "qwen_homotopy_forces_all_candidates": args.qwen_homotopy,
#                     "homotopy_sign_convention": {"LEFT": -1.0, "RIGHT": 1.0},
#                     "homotopy_minimum_obstacle_pixels": args.homotopy_minimum_obstacle_pixels,
#                     "homotopy_release_clear_frames": args.homotopy_release_clear_frames,
#                     "homotopy_consistency_repeats": args.homotopy_consistency_repeats,
#                     "uses_velocity_chunk": False,
#                     "obstacle_mode": args.obstacle_mode,
#                     "obstacle_world_xz": args.obstacle_world_xz,
#                     "goal_mesh": args.goal_mesh,
#                     "candidate_count": args.candidates,
#                     "particles_per_candidate": args.particles,
#                     "particle_std": args.particle_std,
#                     "guidance_strength": args.guidance_strength,
#                     "temperature": args.temperature,
#                     "particle_anchor": args.particle_anchor,
#                     "particle_energy_reweighting": args.particle_energy_reweighting,
#                     "particle_collision_mask": args.particle_collision_mask,
#                     "goal_mode": args.goal_mode,
#                     "interactive_return_home": args.interactive_return_home,
#                     "qwen_freeform_mission": args.qwen_freeform_mission,
#                     "mission_plan_event": mission_plan_event,
#                     "automatic_return_requested": automatic_return_requested,
#                     "phase_completion_source": "metric_distance_state_machine",
#                     "roundtrip_completed": roundtrip_completed,
#                     "final_mission_phase": mission_phase,
#                     "return_command_event": return_command_event,
#                     "home_belief_source": "spawn_origin_plus_executed_odometry",
#                     "reached_goal_becomes_obstacle_on_return": True,
#                     "return_goal_obstacle_activation_distance": (
#                         args.return_goal_obstacle_activation_distance
#                     ),
#                     "return_goal_obstacle_dilation_pixels": (
#                         args.return_goal_obstacle_dilation_pixels
#                     ),
#                     "belief_pixel_goal": args.belief_pixel_goal,
#                     "belief_source": "semantic_goal_mask_plus_odometry",
#                     "belief_bootstrap_world_goal": args.belief_bootstrap_world_goal,
#                     "belief_measurement_std": args.belief_measurement_std,
#                     "belief_translation_process_std": args.belief_translation_process_std,
#                     "belief_yaw_process_std_deg": args.belief_yaw_process_std_deg,
#                     "belief_covariance_controls_navdp_mask_size": False,
#                     "belief_heading_recovery": args.belief_heading_recovery,
#                     "belief_recovery_obstacle_gated": True,
#                     "belief_recovery_bearing_deg": args.belief_recovery_bearing_deg,
#                     "belief_recovery_maximum_yaw_rate": args.belief_recovery_maximum_yaw_rate,
#                     "belief_recovery_maximum_forward_speed": args.belief_recovery_maximum_forward_speed,
#                     "particle_noise_schedule": args.particle_noise_schedule,
#                     "progressive_guidance": args.progressive_guidance,
#                     "mesh_obstacle_count": len(mesh_centroids),
#                     "moving_obstacles": bool(np.any(np.abs(mesh_velocities) > 0.0)),
#                     "obstacle_velocity_xz": mesh_velocities.tolist(),
#                     "evaluation_layout": args.evaluation_layout,
#                     "seed": args.seed,
#                     "robot_radius": args.robot_radius,
#                     "minimum_executed_surface_clearance": (
#                         float(np.nanmin(rows["executed_surface_clearance"]))
#                         if np.any(np.isfinite(rows["executed_surface_clearance"]))
#                         else None
#                     ),
#                     "geometric_collision": bool(np.any(rows["geometric_collision"])),
#                     "rollout": str(rollout_path),
#                 },
#                 file,
#                 indent=2,
#             )
#         if args.save_video and video_frames:
#             save_video(
#                 video_frames,
#                 output_directory / "rollout.mp4",
#                 fps=max(args.hz / max(args.save_every, 1), 1.0),
#             )
#         print(f"Saved rollout: {rollout_path}", flush=True)
#         print(f"Success: {success}", flush=True)
#     finally:
#         if simulator is not None:
#             simulator.close()
#         stop_server(server_process)
#         stop_server(qwen_process)


# if __name__ == "__main__":
#     main()
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
from PIL import Image, ImageDraw, ImageFilter

from belief_heading_recovery import belief_heading_recovery_action
from belief_pixel_goal import GaussianGoalBelief

HERE = Path(__file__).resolve().parent
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566
MESH_GOAL_ID = 10000
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
    candidate_circulation_signs: np.ndarray
    selected_barrier_energy: float
    selected_circulation_energy: float
    minimum_clearance: np.ndarray
    selected_minimum_clearance: float
    mean_guidance_noise_correction: float
    final_guidance_noise_correction: float
    maximum_guidance_noise_correction: float
    mean_final_effective_sample_size: float
    selected_lyapunov_energy: float


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
        goal_mode: str = "point",
        forced_circulation_sign: float = 0.0,
        metric_goal_xy: np.ndarray | None = None,
    ) -> NavDPS2DiffOutput:
        goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        if goal_xy.shape != (2,):
            raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")
        if goal_mode not in {"point", "pixel"}:
            raise ValueError("goal_mode must be point or pixel")
        forced_circulation_sign = float(forced_circulation_sign)
        if metric_goal_xy is not None:
            metric_goal_xy = np.asarray(metric_goal_xy, dtype=np.float32).reshape(-1)
            if metric_goal_xy.shape != (2,) or not np.all(np.isfinite(metric_goal_xy)):
                raise ValueError(
                    "metric_goal_xy must be a finite [forward,left] vector"
                )
        if forced_circulation_sign not in {-1.0, 0.0, 1.0}:
            raise ValueError("forced_circulation_sign must be -1, 0, or +1")

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

        if goal_mode == "pixel":
            if not np.all(np.isfinite(goal_xy)) or not np.allclose(
                goal_xy, np.round(goal_xy)
            ):
                raise ValueError("PixelGoal must be integer [u,v]")
            goal_xy = np.round(goal_xy).astype(np.int64)
            if not (0 <= goal_xy[0] < rgb.shape[1] and 0 <= goal_xy[1] < rgb.shape[0]):
                raise ValueError("PixelGoal lies outside the RGB image")

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

        endpoint = "pixelgoal_step" if goal_mode == "pixel" else "pointgoal_step"
        response = requests.post(
            f"{self.server_url}/{endpoint}",
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
                        "forced_circulation_signs": [forced_circulation_sign],
                        "metric_goal_xy": (
                            [metric_goal_xy.tolist()]
                            if metric_goal_xy is not None
                            else None
                        ),
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
            candidate_circulation_signs=np.asarray(
                diagnostics["candidate_circulation_signs"][0], dtype=np.float32
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
            mean_final_effective_sample_size=float(
                diagnostics.get("mean_final_effective_sample_size", [0.0])[0]
            ),
            selected_lyapunov_energy=float(
                diagnostics.get("selected_lyapunov_energy", [0.0])[0]
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


@dataclass(frozen=True)
class QwenHomotopyDecision:
    side: str
    circulation_sign: float
    confidence: float
    obstacle_relevant: bool
    queried_qwen: bool
    raw_response: Optional[str]
    repeated_sides: tuple[str, ...]
    repeated_confidences: tuple[float, ...]
    consistency_rate: float
    used_fallback: bool


@dataclass(frozen=True)
class QwenCommandDecision:
    command: str
    confidence: float
    raw_response: str


@dataclass(frozen=True)
class QwenMissionPlanDecision:
    plan: tuple[str, ...]
    confidence: float
    raw_response: str


class QwenHomotopyClient:
    """HTTP client for the isolated visual-Qwen process."""

    def __init__(self, server_url: str, timeout: float = 300.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)

    def reset(self) -> None:
        response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
        self._raise_for_error(response)

    def step(
        self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
    ) -> QwenHomotopyDecision:
        image_bytes = io.BytesIO()
        Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8)).save(
            image_bytes, format="PNG"
        )
        mask_bytes = io.BytesIO()
        Image.fromarray((np.asarray(obstacle_mask) > 0).astype(np.uint8) * 255).save(
            mask_bytes, format="PNG"
        )
        response = requests.post(
            f"{self.server_url}/select",
            files={
                "image": ("overlay.png", image_bytes.getvalue(), "image/png"),
                "obstacle_mask": ("mask.png", mask_bytes.getvalue(), "image/png"),
            },
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        return QwenHomotopyDecision(
            side=str(payload["side"]),
            circulation_sign=float(payload["circulation_sign"]),
            confidence=float(payload["confidence"]),
            obstacle_relevant=bool(payload["obstacle_relevant"]),
            queried_qwen=bool(payload["queried_qwen"]),
            raw_response=payload.get("raw_response"),
            repeated_sides=tuple(payload.get("repeated_sides", [])),
            repeated_confidences=tuple(
                float(value) for value in payload.get("repeated_confidences", [])
            ),
            consistency_rate=float(payload.get("consistency_rate", 1.0)),
            used_fallback=bool(payload.get("used_fallback", False)),
        )

    def classify_command(
        self, image_rgb: np.ndarray, user_command: str
    ) -> QwenCommandDecision:
        image_bytes = io.BytesIO()
        Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
            image_bytes, format="PNG"
        )
        response = requests.post(
            f"{self.server_url}/command",
            files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
            data={"command": str(user_command)},
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        return QwenCommandDecision(
            command=str(payload["command"]).upper(),
            confidence=float(payload["confidence"]),
            raw_response=str(payload["raw_response"]),
        )

    def classify_mission(
        self, image_rgb: np.ndarray, user_command: str
    ) -> QwenMissionPlanDecision:
        image_bytes = io.BytesIO()
        Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
            image_bytes, format="PNG"
        )
        response = requests.post(
            f"{self.server_url}/mission",
            files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
            data={"command": str(user_command)},
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        return QwenMissionPlanDecision(
            plan=tuple(str(item).upper() for item in payload["plan"]),
            confidence=float(payload["confidence"]),
            raw_response=str(payload["raw_response"]),
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


def start_qwen_homotopy_server(
    args: argparse.Namespace,
) -> Optional[subprocess.Popen[Any]]:
    if not args.qwen_homotopy or not args.start_qwen_homotopy_server:
        return None
    if port_is_open(args.qwen_homotopy_host, args.qwen_homotopy_port):
        raise RuntimeError(
            f"Qwen homotopy port {args.qwen_homotopy_port} is already in use; "
            "pass --no-start-qwen-homotopy-server to use an existing service"
        )
    server_file = HERE / "qwen_homotopy_server.py"
    if not server_file.is_file():
        raise FileNotFoundError(f"Qwen homotopy server not found: {server_file}")
    command = [
        str(args.qwen_homotopy_python),
        str(server_file),
        "--host",
        str(args.qwen_homotopy_host),
        "--port",
        str(args.qwen_homotopy_port),
        "--model-id",
        str(args.qwen_model_id),
        "--device",
        str(args.qwen_device),
        "--minimum-obstacle-pixels",
        str(args.homotopy_minimum_obstacle_pixels),
        "--release-clear-frames",
        str(args.homotopy_release_clear_frames),
        "--consistency-repeats",
        str(args.homotopy_consistency_repeats),
    ]
    print("[qwen-server]", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(HERE))
    wait_for_server(
        process,
        args.qwen_homotopy_host,
        args.qwen_homotopy_port,
        args.qwen_homotopy_timeout,
    )
    return process


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
        "--gradient-steps",
        str(args.gradient_steps),
        "--gradient-step-size",
        str(args.gradient_step_size),
        "--guidance-strength",
        str(args.guidance_strength),
        "--temperature",
        str(args.temperature),
        "--safe-distance",
        str(args.safe_distance),
        "--hard-collision-distance",
        str(args.hard_collision_distance),
        "--robot-radius",
        str(args.robot_radius),
        "--safety-weight",
        str(args.safety_weight),
        "--lyapunov-weight",
        str(args.lyapunov_weight),
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
    particle_flags = {
        "particle-anchor": args.particle_anchor,
        "particle-energy-reweighting": args.particle_energy_reweighting,
        "particle-collision-mask": args.particle_collision_mask,
        "particle-noise-schedule": args.particle_noise_schedule,
        "progressive-guidance": args.progressive_guidance,
    }
    for name, enabled in particle_flags.items():
        command.append(f"--{name}" if enabled else f"--no-{name}")
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


def save_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    diffuse_rgb: Optional[tuple[float, float, float]] = None,
) -> None:
    material_name = None
    if diffuse_rgb is not None:
        red, green, blue = (float(value) for value in diffuse_rgb)
        if not all(0.0 <= value <= 1.0 for value in (red, green, blue)):
            raise ValueError("OBJ diffuse material values must be in [0, 1]")
        material_name = "mesh_material"
        material_path = path.with_suffix(".mtl")
        with material_path.open("w", encoding="utf-8") as material:
            material.write(f"newmtl {material_name}\n")
            material.write(
                f"Ka {0.25 * red:.4f} {0.25 * green:.4f} {0.25 * blue:.4f}\n"
            )
            material.write(f"Kd {red:.4f} {green:.4f} {blue:.4f}\n")
            material.write("Ks 0.1000 0.1000 0.1000\n")
            material.write("Ns 24.0000\n")
            material.write("d 1.0000\n")
            material.write("illum 2\n")
    with path.open("w", encoding="utf-8") as file:
        if material_name is not None:
            file.write(f"mtllib {path.with_suffix('.mtl').name}\n")
            file.write(f"usemtl {material_name}\n")
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


def parse_world_xz(specification: str) -> tuple[float, float]:
    values = [float(value) for value in str(specification).split(",")]
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError(
            f"world mesh position must be finite X,Z, got {specification!r}"
        )
    return values[0], values[1]


def world_box_mesh(
    center_x: float,
    base_y: float,
    center_z: float,
    half_extent: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed axis-aligned box whose vertices are in world coordinates."""

    if half_extent <= 0.0 or height <= 0.0:
        raise ValueError("box half extent and height must be positive")
    x0, x1 = center_x - half_extent, center_x + half_extent
    z0, z1 = center_z - half_extent, center_z + half_extent
    y0, y1 = base_y, base_y + height
    vertices = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x0, y0, z1],
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def place_world_obstacle_meshes(
    simulator: Any,
    terrain: Any,
    xz_specifications: Sequence[str],
    output_directory: Path,
    *,
    half_extent: float,
    height: float,
) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
    """Place static obstacle boxes at exact world X,Z coordinates."""

    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    objects: list[Any] = []
    centroids: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    for index, specification in enumerate(xz_specifications):
        center_x, center_z = parse_world_xz(specification)
        base_y = terrain.local_height_max(center_x, center_z, half_extent)
        vertices, faces = world_box_mesh(
            center_x, base_y, center_z, half_extent, height
        )
        mesh_path = mesh_directory / f"world_obstacle_{index}.obj"
        save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.78, 0.16, 0.06))
        semantic_id = MESH_OBSTACLE_ID + index
        objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
        centroid = vertices.mean(axis=0).astype(np.float32)
        centroids.append(centroid)
        geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
        print(
            f"[world-mesh] obstacle={index} semantic_id={semantic_id} "
            f"center_xz={[center_x, center_z]} half_extent={half_extent:.3f} "
            f"height={height:.3f}",
            flush=True,
        )
    return objects, centroids, geometries


def place_world_goal_mesh(
    simulator: Any,
    terrain: Any,
    goal_x: float,
    goal_z: float,
    output_directory: Path,
    *,
    half_extent: float,
    height: float,
) -> Any:
    """Place a visible, non-obstacle semantic goal marker at the exact goal."""

    base_y = terrain.local_height_max(goal_x, goal_z, half_extent)
    vertices, faces = world_box_mesh(goal_x, base_y, goal_z, half_extent, height)
    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_directory / "goal_marker.obj"
    save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.08, 0.85, 0.18))
    goal_object = register_semantic_mesh(simulator, mesh_path, MESH_GOAL_ID)
    print(
        f"[world-mesh] goal semantic_id={MESH_GOAL_ID} "
        f"center_xz={[goal_x, goal_z]}",
        flush=True,
    )
    return goal_object


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
) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    height, width = depth.shape
    objects: list[Any] = []
    centroids: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
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
        geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
        print(
            f"[mesh] obstacle={index} semantic_id={semantic_id} "
            f"pixels={specification} vertices={len(vertices)} "
            f"world={centroid.tolist()}",
            flush=True,
        )
    return objects, centroids, geometries


def planar_mesh_clearance(
    point_xz: np.ndarray,
    geometries: Sequence[np.ndarray],
) -> float:
    """Minimum 2-D distance from a robot center to projected mesh triangles."""
    point = np.asarray(point_xz, dtype=np.float64)
    best = float("inf")
    for triangles in geometries:
        triangles = np.asarray(triangles, dtype=np.float64)
        if triangles.size == 0:
            continue
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        v0, v1, v2 = b - a, c - a, point[None, :] - a
        denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        valid = np.abs(denominator) > 1.0e-12
        safe_denominator = np.where(valid, denominator, 1.0)
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_denominator
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_denominator
        if np.any(valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)):
            return 0.0

        starts = np.concatenate((a, b, c), axis=0)
        ends = np.concatenate((b, c, a), axis=0)
        segments = ends - starts
        squared_lengths = np.einsum("ij,ij->i", segments, segments)
        numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
        fractions = np.divide(
            numerators,
            squared_lengths,
            out=np.zeros_like(numerators),
            where=squared_lengths > 1.0e-12,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = starts + fractions[:, None] * segments
        best = min(best, float(np.linalg.norm(point[None, :] - closest, axis=1).min()))
    return best


def parse_xz_velocity(specification: str) -> np.ndarray:
    values = [float(value) for value in str(specification).split(",")]
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise ValueError("obstacle velocity must be finite vx,vz")
    return np.asarray(values, dtype=np.float64)


def expand_obstacle_velocities(
    specifications: Sequence[str], obstacle_count: int
) -> np.ndarray:
    if obstacle_count == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if not specifications:
        return np.zeros((obstacle_count, 2), dtype=np.float64)
    velocities = np.stack([parse_xz_velocity(item) for item in specifications])
    if len(velocities) == 1 and obstacle_count > 1:
        velocities = np.repeat(velocities, obstacle_count, axis=0)
    if len(velocities) != obstacle_count:
        raise ValueError(
            "provide one obstacle velocity to broadcast or one velocity per mesh"
        )
    return velocities


def translated_mesh_geometry(
    base_geometries: Sequence[np.ndarray],
    base_centroids: Sequence[np.ndarray],
    velocities_xz: np.ndarray,
    elapsed_seconds: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    geometries: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    for geometry, centroid, velocity in zip(
        base_geometries, base_centroids, velocities_xz
    ):
        offset_xz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
        geometries.append(np.asarray(geometry) + offset_xz[None, None, :])
        offset_xyz = np.asarray([offset_xz[0], 0.0, offset_xz[1]])
        centroids.append(np.asarray(centroid, dtype=np.float64) + offset_xyz)
    return geometries, centroids


def move_mesh_objects(
    objects: Sequence[Any], velocities_xz: np.ndarray, elapsed_seconds: float
) -> None:
    for obstacle, velocity in zip(objects, velocities_xz):
        dx, dz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
        vector_type = type(obstacle.translation)
        obstacle.translation = vector_type(float(dx), 0.0, float(dz))


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


def world_goal_to_pixel(
    point: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Project a world goal to a valid PixelGoal, clamping off-screen bearings."""

    right, up, forward = camera_coordinates(point, position, yaw)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    margin = 11
    bearing = math.atan2(right, forward)
    maximum_bearing = math.atan2(max(cx - margin, 1.0), fx)
    bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
    u = cx + fx * math.tan(bearing)
    v = cy - fy * up / forward if forward > 0.05 else 0.62 * height
    return np.asarray(
        [
            int(np.clip(round(u), margin, width - margin - 1)),
            int(np.clip(round(v), margin, height - margin - 1)),
        ],
        dtype=np.int32,
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
    rgb: np.ndarray,
    goal_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    text: str,
    *,
    show_masks: bool,
    detection_box: Optional[np.ndarray] = None,
    detection_label: Optional[str] = None,
) -> Image.Image:
    output = np.asarray(rgb, dtype=np.uint8).copy()
    if show_masks:
        output[goal_mask > 0] = (
            0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
        ).astype(np.uint8)
        output[obstacle_mask > 0] = (
            0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
        ).astype(np.uint8)
    image = Image.fromarray(output)
    draw = ImageDraw.Draw(image)
    if detection_box is not None:
        x1, y1, x2, y2 = [float(value) for value in detection_box]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
        if detection_label:
            draw.text((x1 + 2, max(y1 - 14, 2)), detection_label, fill=(255, 255, 0))
    draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
    draw.text((10, 9), text, fill=(255, 255, 255))
    return image


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary image mask without adding a SciPy dependency."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if radius <= 0 or not np.any(binary):
        return binary
    kernel_size = 2 * int(radius) + 1
    return (
        np.asarray(
            Image.fromarray(binary * 255).filter(ImageFilter.MaxFilter(kernel_size))
        )
        > 0
    ).astype(np.uint8)


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
        "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
    )
    argument_parser.add_argument(
        "--goal-mode", choices=["point", "pixel"], default="point"
    )
    argument_parser.add_argument(
        "--belief-pixel-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the live semantic goal mask to correct a body-frame Gaussian "
            "belief and its projected mean as NavDP's PixelGoal while occluded."
        ),
    )
    argument_parser.add_argument("--belief-minimum-goal-pixels", type=int, default=10)
    argument_parser.add_argument("--belief-measurement-std", type=float, default=0.05)
    argument_parser.add_argument(
        "--belief-translation-process-std", type=float, default=0.03
    )
    argument_parser.add_argument(
        "--belief-yaw-process-std-deg", type=float, default=1.0
    )
    argument_parser.add_argument(
        "--belief-bootstrap-world-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Simulation-only bootstrap when the goal is initially invisible. "
            "Disable for a strict detector-only evaluation."
        ),
    )
    argument_parser.add_argument("--belief-bootstrap-std", type=float, default=0.50)
    argument_parser.add_argument("--belief-ghost-base-radius", type=int, default=10)
    argument_parser.add_argument(
        "--belief-ghost-covariance-scale", type=float, default=2.0
    )
    argument_parser.add_argument("--belief-ghost-maximum-radius", type=int, default=80)
    argument_parser.add_argument(
        "--belief-heading-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument(
        "--belief-recovery-bearing-deg", type=float, default=35.0
    )
    argument_parser.add_argument("--belief-recovery-yaw-gain", type=float, default=1.5)
    argument_parser.add_argument(
        "--belief-recovery-maximum-yaw-rate", type=float, default=0.70
    )
    argument_parser.add_argument(
        "--belief-recovery-maximum-forward-speed", type=float, default=0.12
    )
    argument_parser.add_argument(
        "--interactive-return-home",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "At the outward goal, ask for a command, let Qwen classify RETURN "
            "or STOP, and use a separately propagated spawn/home PixelGoal belief."
        ),
    )
    argument_parser.add_argument(
        "--return-command",
        default=None,
        help="Optional non-interactive command text, for example 'come back'.",
    )
    argument_parser.add_argument(
        "--qwen-freeform-mission",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ask once at startup for a free-form instruction. Qwen emits either "
            "GO_TO_GOAL or GO_TO_GOAL followed by RETURN_HOME."
        ),
    )
    argument_parser.add_argument(
        "--mission-command",
        default=None,
        help=(
            "Optional non-interactive free-form mission, for example "
            "'visit the target and report back'."
        ),
    )
    argument_parser.add_argument(
        "--return-goal-obstacle-activation-distance", type=float, default=1.35
    )
    argument_parser.add_argument(
        "--return-goal-obstacle-dilation-pixels", type=int, default=30
    )
    argument_parser.add_argument(
        "--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    argument_parser.add_argument("--qwen-device", default="auto")
    argument_parser.add_argument("--qwen-homotopy-python", default=sys.executable)
    argument_parser.add_argument("--qwen-homotopy-host", default="127.0.0.1")
    argument_parser.add_argument("--qwen-homotopy-port", type=int, default=8890)
    argument_parser.add_argument("--qwen-homotopy-timeout", type=float, default=600.0)
    argument_parser.add_argument(
        "--start-qwen-homotopy-server",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument(
        "--qwen-homotopy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When a metric obstacle becomes relevant, Qwen chooses the single "
            "LEFT/RIGHT circulation sign used by every trajectory candidate."
        ),
    )
    argument_parser.add_argument(
        "--homotopy-minimum-obstacle-pixels", type=int, default=30
    )
    argument_parser.add_argument("--homotopy-release-clear-frames", type=int, default=8)
    argument_parser.add_argument(
        "--homotopy-consistency-repeats",
        type=int,
        default=5,
        help="Repeat Qwen on the identical obstacle frame and use majority vote.",
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
    argument_parser.add_argument("--gradient-steps", type=int, default=3)
    argument_parser.add_argument("--gradient-step-size", type=float, default=0.04)
    argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
    argument_parser.add_argument("--temperature", type=float, default=0.35)
    argument_parser.add_argument(
        "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--particle-energy-reweighting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument(
        "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--safe-distance", type=float, default=0.42)
    argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
    argument_parser.add_argument("--safety-weight", type=float, default=35.0)
    argument_parser.add_argument("--lyapunov-weight", type=float, default=4.0)
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
    argument_parser.add_argument(
        "--robot-radius",
        type=float,
        default=0.24,
        help="Planar rover footprint radius used by both guidance and evaluation.",
    )
    argument_parser.add_argument(
        "--evaluation-layout",
        default="default",
        help="Stable layout identifier stored in the rollout archive.",
    )

    argument_parser.add_argument("--height", type=int, default=720)
    argument_parser.add_argument("--width", type=int, default=720)
    argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
    argument_parser.add_argument("--hz", type=float, default=10.0)
    argument_parser.add_argument("--max-steps", type=int, default=300)
    argument_parser.add_argument("--stop-distance", type=float, default=1.0)
    argument_parser.add_argument("--start-x", type=float, default=0.0)
    argument_parser.add_argument("--start-z", type=float, default=8.0)
    argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    argument_parser.add_argument("--goal-x", type=float, default=None)
    argument_parser.add_argument("--goal-z", type=float, default=None)
    argument_parser.add_argument("--goal-y", type=float, default=None)
    argument_parser.add_argument("--goal-height", type=float, default=1.2)
    argument_parser.add_argument("--goal-radius", type=int, default=18)
    argument_parser.add_argument(
        "--goal-mesh", action=argparse.BooleanOptionalAction, default=False
    )
    argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
    argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)

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
    argument_parser.add_argument(
        "--obstacle-world-xz",
        nargs="*",
        default=[],
        metavar="X,Z",
        help=(
            "Static rendered obstacle-box centers in world X,Z coordinates. "
            "Example: --obstacle-world-xz 0,0. Do not combine with "
            "--obstacle-mesh-uv."
        ),
    )
    argument_parser.add_argument(
        "--obstacle-world-xz-item",
        action="append",
        default=[],
        metavar="X,Z",
        help=(
            "Repeatable form that safely accepts negative coordinates, e.g. "
            "--obstacle-world-xz-item=-3,0."
        ),
    )
    argument_parser.add_argument(
        "--world-obstacle-half-extent", type=float, default=0.75
    )
    argument_parser.add_argument("--world-obstacle-height", type=float, default=1.40)
    argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
    argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)
    argument_parser.add_argument(
        "--obstacle-velocity-xz",
        nargs="*",
        default=[],
        metavar="VX,VZ",
        help=(
            "World-frame mesh velocities in m/s. Supply one value to broadcast "
            "or one value per obstacle. Example: --obstacle-velocity-xz 0.30,0.0"
        ),
    )

    argument_parser.add_argument("--lookahead-index", type=int, default=4)
    argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
    argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
    argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
    argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
    argument_parser.add_argument("--save-every", type=int, default=1)
    argument_parser.add_argument(
        "--save-frames", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--archive-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store RGB/depth/masks in rollout.npz; disable for large evaluations.",
    )
    argument_parser.add_argument(
        "--overlay-masks", action=argparse.BooleanOptionalAction, default=True
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if args.obstacle_world_xz_item:
        args.obstacle_world_xz.extend(args.obstacle_world_xz_item)
    np.random.seed(args.seed)
    if args.goal_x is None or args.goal_z is None:
        raise ValueError("fixed PointGoal requires --goal-x and --goal-z")
    if args.belief_pixel_goal and args.goal_mode != "pixel":
        raise ValueError("--belief-pixel-goal requires --goal-mode pixel")
    if args.belief_pixel_goal and not args.goal_mesh:
        raise ValueError(
            "simulation belief tracking requires --goal-mesh so a live semantic "
            "goal observation exists"
        )
    if args.belief_minimum_goal_pixels < 1:
        raise ValueError("belief-minimum-goal-pixels must be positive")
    return_home_enabled = args.interactive_return_home or args.qwen_freeform_mission
    if return_home_enabled and not args.belief_pixel_goal:
        raise ValueError("return-home modes require --belief-pixel-goal")
    if return_home_enabled and not args.qwen_homotopy:
        raise ValueError("return-home modes require --qwen-homotopy")
    if args.mission_command is not None and not args.qwen_freeform_mission:
        raise ValueError("--mission-command requires --qwen-freeform-mission")
    if args.return_goal_obstacle_activation_distance <= 0.0:
        raise ValueError("return goal obstacle activation distance must be positive")
    if args.return_goal_obstacle_dilation_pixels < 0:
        raise ValueError("return goal obstacle dilation pixels must be non-negative")
    if (
        min(
            args.belief_measurement_std,
            args.belief_translation_process_std,
            args.belief_yaw_process_std_deg,
            args.belief_bootstrap_std,
        )
        < 0.0
    ):
        raise ValueError("belief uncertainty parameters must be non-negative")
    if args.robot_radius < 0.0:
        raise ValueError("robot-radius must be non-negative")
    if args.obstacle_velocity_xz and args.obstacle_mode != "mesh":
        raise ValueError("moving obstacle velocities require --obstacle-mode mesh")
    if args.obstacle_mode == "ghost" and (
        args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
    ):
        raise ValueError(
            "ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z"
        )
    if args.obstacle_mesh_uv and args.obstacle_world_xz:
        raise ValueError(
            "choose either --obstacle-mesh-uv or --obstacle-world-xz, not both"
        )
    if args.obstacle_mode == "mesh" and not (
        args.obstacle_mesh_uv or args.obstacle_world_xz
    ):
        raise ValueError(
            "mesh mode requires --obstacle-world-xz X,Z [X,Z ...] or "
            "--obstacle-mesh-uv u,v [u,v ...]"
        )
    if args.world_obstacle_half_extent <= 0.0 or args.world_obstacle_height <= 0.0:
        raise ValueError("world obstacle dimensions must be positive")
    if args.goal_mesh_half_extent <= 0.0 or args.goal_mesh_height <= 0.0:
        raise ValueError("goal mesh dimensions must be positive")

    if args.qwen_homotopy and args.planner_mode == "pure-navdp":
        raise ValueError("Qwen homotopy conditioning requires s2diff or gradient mode")

    qwen_process: Optional[subprocess.Popen[Any]] = None
    server_process: Optional[subprocess.Popen[Any]] = None
    simulator = None
    try:
        qwen_process = start_qwen_homotopy_server(args)
        homotopy_selector = None
        if args.qwen_homotopy:
            homotopy_selector = QwenHomotopyClient(
                f"http://{args.qwen_homotopy_host}:{args.qwen_homotopy_port}",
                timeout=args.qwen_homotopy_timeout,
            )
            homotopy_selector.reset()
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
            "navdp-hlc-gradient",
            "navdp-hlc-gradient-no-critic",
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
            with_semantic=args.obstacle_mode == "mesh" or args.goal_mesh,
        )
        agent = simulator.initialize_agent(0)
        intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
        goal_belief = (
            GaussianGoalBelief(
                intrinsic,
                (args.height, args.width),
                minimum_visible_pixels=args.belief_minimum_goal_pixels,
                measurement_std=args.belief_measurement_std,
                translation_process_std=args.belief_translation_process_std,
                yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
            )
            if args.belief_pixel_goal
            else None
        )
        previous_executed_action = np.zeros(3, dtype=np.float32)
        home_belief = (
            GaussianGoalBelief(
                intrinsic,
                (args.height, args.width),
                minimum_visible_pixels=args.belief_minimum_goal_pixels,
                measurement_std=args.belief_measurement_std,
                translation_process_std=args.belief_translation_process_std,
                yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
            )
            if return_home_enabled
            else None
        )
        if home_belief is not None:
            # Home starts at the rover origin. Every executed action propagates
            # this stationary world location into the current body frame.
            home_belief.initialize(
                np.zeros(2, dtype=np.float32),
                args.belief_measurement_std,
                visible=False,
            )
        mission_phase = "OUTBOUND"
        return_goal_obstacle_active = False
        return_command_event: Optional[dict[str, Any]] = None
        mission_plan_event: Optional[dict[str, Any]] = None
        mission_plan_pending = bool(args.qwen_freeform_mission)
        automatic_return_requested = False
        roundtrip_completed = False
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
        start_position_xz = np.asarray([x, z], dtype=np.float64)
        initial_goal_distance = float(
            np.linalg.norm(goal[[0, 2]].astype(np.float64) - start_position_xz)
        )
        goal_mesh_object = None
        if args.goal_mesh:
            goal_mesh_object = place_world_goal_mesh(
                simulator,
                terrain,
                args.goal_x,
                args.goal_z,
                output_directory,
                half_extent=args.goal_mesh_half_extent,
                height=args.goal_mesh_height,
            )

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
        mesh_current_centroids: list[np.ndarray] = []
        mesh_base_geometries: list[np.ndarray] = []
        mesh_geometries: list[np.ndarray] = []
        mesh_velocities = np.zeros((0, 2), dtype=np.float64)
        mesh_placed = False
        if args.obstacle_mode == "mesh" and args.obstacle_world_xz:
            mesh_objects, mesh_centroids, mesh_base_geometries = (
                place_world_obstacle_meshes(
                    simulator,
                    terrain,
                    args.obstacle_world_xz,
                    output_directory,
                    half_extent=args.world_obstacle_half_extent,
                    height=args.world_obstacle_height,
                )
            )
            mesh_velocities = expand_obstacle_velocities(
                args.obstacle_velocity_xz, len(mesh_objects)
            )
            mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
                mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
            )
            mesh_placed = True

        row_keys = [
            "pose",
            "action_3d",
            "mission_phase",
            "return_goal_obstacle_active",
            "point_goal",
            "belief_goal_mu",
            "belief_goal_covariance",
            "belief_goal_pixel",
            "belief_goal_visible",
            "belief_goal_source",
            "belief_goal_time_since_seen",
            "belief_goal_bearing_rad",
            "belief_goal_pixel_sigma",
            "belief_heading_recovery_active",
            "selected_trajectory",
            "all_trajectories",
            "all_values",
            "selected_index",
            "fallback_stop",
            "escape_turn",
            "valid_obstacle_points",
            "selected_circulation_sign",
            "candidate_circulation_signs",
            "selected_barrier_energy",
            "selected_lyapunov_energy",
            "selected_circulation_energy",
            "planning_time_seconds",
            "selected_minimum_clearance",
            "mean_guidance_noise_correction",
            "final_guidance_noise_correction",
            "maximum_guidance_noise_correction",
            "mean_final_effective_sample_size",
            "goal_distance",
            "executed_center_clearance",
            "executed_surface_clearance",
            "geometric_collision",
            "obstacle_positions_world",
            "qwen_homotopy_sign",
            "qwen_homotopy_side",
            "qwen_homotopy_confidence",
            "qwen_homotopy_queried",
        ]
        if args.archive_observations:
            row_keys.extend(
                (
                    "rgb",
                    "depth",
                    "goal_mask",
                    "live_goal_mask",
                    "ghost_goal_mask",
                    "obstacle_mask",
                )
            )
        rows: dict[str, list[Any]] = {key: [] for key in row_keys}
        video_frames: list[Image.Image] = []
        success = False
        homotopy_events: list[dict[str, Any]] = []

        for step in range(int(args.max_steps)):
            y = (
                terrain.local_height_max(x, z, args.pose_terrain_radius)
                + args.clearance
            )
            position = np.asarray([x, y, z], dtype=np.float32)
            if goal_belief is not None and step > 0:
                goal_belief.predict(previous_executed_action, dt)
            if home_belief is not None and step > 0:
                home_belief.predict(previous_executed_action, dt)
            set_agent_pose(agent, position, yaw)
            if mesh_placed:
                elapsed_seconds = step * dt
                move_mesh_objects(mesh_objects, mesh_velocities, elapsed_seconds)
                mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
                    mesh_base_geometries,
                    mesh_centroids,
                    mesh_velocities,
                    elapsed_seconds,
                )
            observation = simulator.get_sensor_observations()
            rgb, depth = rgb_depth(observation)

            if args.obstacle_mode == "mesh" and not mesh_placed:
                mesh_objects, mesh_centroids, mesh_base_geometries = (
                    place_obstacle_meshes(
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
                )
                mesh_velocities = expand_obstacle_velocities(
                    args.obstacle_velocity_xz, len(mesh_objects)
                )
                mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
                    mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
                )
                mesh_placed = True
                observation = simulator.get_sensor_observations()
                rgb, depth = rgb_depth(observation)

            if mission_plan_pending:
                mission_command = args.mission_command
                if mission_command is None:
                    print(
                        "\nWhat should the rover do? You may use a vague command, "
                        "for example: 'visit the goal and come back'.",
                        flush=True,
                    )
                    try:
                        mission_command = input("> ").strip()
                    except EOFError as error:
                        raise RuntimeError(
                            "no startup command was available; set "
                            "--mission-command for a non-interactive run"
                        ) from error
                assert homotopy_selector is not None
                mission_decision = homotopy_selector.classify_mission(
                    rgb, mission_command
                )
                automatic_return_requested = mission_decision.plan == (
                    "GO_TO_GOAL",
                    "RETURN_HOME",
                )
                mission_plan_event = {
                    "step": step,
                    "user_command": mission_command,
                    "plan": list(mission_decision.plan),
                    "confidence": mission_decision.confidence,
                    "raw_response": mission_decision.raw_response,
                }
                Image.fromarray(rgb).save(output_directory / "qwen_mission_frame.png")
                print(
                    f"[qwen-mission] text={mission_command!r} "
                    f"plan={list(mission_decision.plan)} "
                    f"confidence={mission_decision.confidence:.2f}",
                    flush=True,
                )
                mission_plan_pending = False

            semantic = (
                semantic_from_observation(observation)
                if args.obstacle_mode == "mesh" or args.goal_mesh
                else None
            )
            goal_right, _goal_up, goal_forward = camera_coordinates(goal, position, yaw)
            point_goal = np.asarray(
                [max(goal_forward, 0.0), -goal_right], dtype=np.float32
            )
            live_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
            ghost_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
            belief_goal_visible = False
            belief_goal_source = "DISABLED"
            belief_goal_mu = np.full(2, np.nan, dtype=np.float32)
            belief_goal_covariance = np.full((2, 2), np.nan, dtype=np.float32)
            belief_goal_pixel = np.full(2, -1, dtype=np.int32)
            belief_goal_time_since_seen = float("nan")
            belief_goal_bearing = float("nan")
            belief_goal_pixel_sigma = float("nan")

            if goal_belief is not None:
                assert semantic is not None
                live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
                bootstrapped = False
                if mission_phase == "OUTBOUND":
                    active_goal_belief = goal_belief
                    belief_goal_visible = goal_belief.observe(live_goal_mask, depth)
                    if not goal_belief.initialized:
                        if not args.belief_bootstrap_world_goal:
                            raise RuntimeError(
                                "goal belief is uninitialized because the live goal "
                                "mask has not been observed; start with the goal visible "
                                "or pass --belief-bootstrap-world-goal for simulation"
                            )
                        goal_belief.initialize(
                            np.asarray([goal_forward, -goal_right], dtype=np.float32),
                            args.belief_bootstrap_std,
                        )
                        bootstrapped = True
                else:
                    assert home_belief is not None and home_belief.initialized
                    active_goal_belief = home_belief
                    belief_goal_visible = False

                belief_projection = active_goal_belief.project(
                    base_radius=args.belief_ghost_base_radius,
                    covariance_scale=args.belief_ghost_covariance_scale,
                    maximum_radius=args.belief_ghost_maximum_radius,
                )
                planner_goal = belief_projection.pixel_uv
                ghost_goal_mask = belief_projection.mask
                if mission_phase == "OUTBOUND" and belief_goal_visible:
                    goal_mask = live_goal_mask
                    belief_goal_source = "LIVE"
                else:
                    goal_mask = ghost_goal_mask
                    belief_goal_source = (
                        "HOME_BELIEF"
                        if mission_phase == "RETURN_HOME"
                        else ("WORLD_BOOTSTRAP" if bootstrapped else "GHOST")
                    )
                assert (
                    active_goal_belief.mu is not None
                    and active_goal_belief.Sigma is not None
                )
                belief_goal_mu = active_goal_belief.mu.copy()
                belief_goal_covariance = active_goal_belief.Sigma.copy()
                belief_goal_pixel = belief_projection.pixel_uv.copy()
                belief_goal_time_since_seen = active_goal_belief.time_since_seen
                belief_goal_bearing = belief_projection.bearing_rad
                belief_goal_pixel_sigma = belief_projection.pixel_sigma
            else:
                if args.goal_mesh:
                    assert semantic is not None
                    live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
                    goal_mask = live_goal_mask
                    if not np.any(goal_mask):
                        goal_mask, _ = project_world_mask(
                            goal,
                            position,
                            yaw,
                            intrinsic,
                            args.height,
                            args.width,
                            args.goal_radius,
                        )
                else:
                    goal_mask, _ = project_world_mask(
                        goal,
                        position,
                        yaw,
                        intrinsic,
                        args.height,
                        args.width,
                        args.goal_radius,
                    )
                planner_goal = point_goal
            if args.goal_mode == "pixel" and goal_belief is None:
                planner_goal = world_goal_to_pixel(
                    goal, position, yaw, intrinsic, args.height, args.width
                )
                goal_mask = circle_mask(
                    args.height,
                    args.width,
                    planner_goal[0],
                    planner_goal[1],
                    args.goal_radius,
                )
            guidance_depth = depth.copy()
            if args.obstacle_mode == "depth":
                obstacle_mask = depth_obstacle_mask(
                    depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
                )
            elif args.obstacle_mode == "mesh":
                assert semantic is not None
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
            if mission_phase == "RETURN_HOME":
                distance_from_reached_goal = float(
                    np.linalg.norm(goal[[0, 2]] - position[[0, 2]])
                )
                if (
                    not return_goal_obstacle_active
                    and distance_from_reached_goal
                    >= args.return_goal_obstacle_activation_distance
                ):
                    return_goal_obstacle_active = True
                    print(
                        "[roundtrip] reached-goal keep-out is now active",
                        flush=True,
                    )
                if return_goal_obstacle_active and np.any(live_goal_mask):
                    reached_goal_keepout = dilate_binary_mask(
                        live_goal_mask,
                        args.return_goal_obstacle_dilation_pixels,
                    )
                    obstacle_mask = (
                        (obstacle_mask > 0) | (reached_goal_keepout > 0)
                    ).astype(np.uint8)
                    target_depths = depth[
                        (live_goal_mask > 0)
                        & np.isfinite(depth)
                        & (depth > args.minimum_obstacle_depth)
                    ]
                    if target_depths.size:
                        guidance_depth[reached_goal_keepout > 0] = float(
                            np.median(target_depths)
                        )

            # array if obstacle pixels already come directly from your system.
            obstacle_pixels = pixels_from_mask(
                obstacle_mask, args.maximum_obstacle_pixels
            )
            homotopy_decision = None
            forced_circulation_sign = 0.0
            obstacle_relevant_for_homotopy = False
            if homotopy_selector is not None:
                homotopy_obstacle_mask = (
                    (obstacle_mask > 0)
                    & np.isfinite(guidance_depth)
                    & (guidance_depth >= args.minimum_obstacle_depth)
                    & (guidance_depth <= args.maximum_obstacle_depth)
                ).astype(np.uint8)
                qwen_overlay = overlay_frame(
                    rgb,
                    goal_mask,
                    homotopy_obstacle_mask,
                    "Qwen homotopy: choose LEFT or RIGHT",
                    show_masks=True,
                )
                homotopy_decision = homotopy_selector.step(
                    np.asarray(qwen_overlay.convert("RGB")), homotopy_obstacle_mask
                )
                obstacle_relevant_for_homotopy = homotopy_decision.obstacle_relevant
                forced_circulation_sign = homotopy_decision.circulation_sign
                if homotopy_decision.queried_qwen:
                    event = {
                        "step": step,
                        "side": homotopy_decision.side,
                        "circulation_sign": forced_circulation_sign,
                        "confidence": homotopy_decision.confidence,
                        "repeat_sides": list(homotopy_decision.repeated_sides),
                        "repeat_confidences": list(
                            homotopy_decision.repeated_confidences
                        ),
                        "consistency_rate": homotopy_decision.consistency_rate,
                        "used_fallback": homotopy_decision.used_fallback,
                        "raw_response": homotopy_decision.raw_response,
                    }
                    homotopy_events.append(event)
                    query_directory = output_directory / "qwen_homotopy_queries"
                    query_directory.mkdir(parents=True, exist_ok=True)
                    qwen_overlay.save(query_directory / f"query_step_{step:04d}.png")
                    print(
                        f"[qwen-homotopy] side={homotopy_decision.side} "
                        f"sign={forced_circulation_sign:+.0f} "
                        f"confidence={homotopy_decision.confidence:.2f} "
                        f"consistency={homotopy_decision.consistency_rate:.2%} "
                        f"repeats={list(homotopy_decision.repeated_sides)} "
                        f"fallback={homotopy_decision.used_fallback}",
                        flush=True,
                    )
            planning_start = time.perf_counter()
            result = client.plan(
                goal_xy=planner_goal,
                rgb=rgb,
                depth=guidance_depth,
                obstacle_pixels=obstacle_pixels,
                goal_mode=args.goal_mode,
                forced_circulation_sign=forced_circulation_sign,
                metric_goal_xy=(
                    belief_goal_mu
                    if args.goal_mode == "pixel" and np.all(np.isfinite(belief_goal_mu))
                    else None
                ),
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
            action, belief_recovery_active = belief_heading_recovery_action(
                action,
                belief_bearing=belief_goal_bearing,
                obstacle_relevant=obstacle_relevant_for_homotopy,
                enabled=args.belief_pixel_goal and args.belief_heading_recovery,
                activation_bearing=math.radians(args.belief_recovery_bearing_deg),
                yaw_gain=args.belief_recovery_yaw_gain,
                maximum_yaw_rate=args.belief_recovery_maximum_yaw_rate,
                maximum_forward_speed=args.belief_recovery_maximum_forward_speed,
            )

            next_position, next_yaw = integrate_mars(position, yaw, action, dt)
            previous_executed_action = action.copy()
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
            rows["mission_phase"].append(mission_phase)
            rows["return_goal_obstacle_active"].append(return_goal_obstacle_active)

            yaw = wrap_angle(next_yaw)
            outbound_goal_distance = float(
                np.linalg.norm(goal[[0, 2]] - np.asarray([x, z]))
            )
            home_distance = float(
                np.linalg.norm(start_position_xz - np.asarray([x, z]))
            )
            goal_distance = (
                home_distance
                if mission_phase == "RETURN_HOME"
                else outbound_goal_distance
            )
            center_clearance = planar_mesh_clearance(
                np.asarray([x, z], dtype=np.float64), mesh_geometries
            )
            if np.isfinite(center_clearance):
                surface_clearance = max(
                    center_clearance - float(args.robot_radius), 0.0
                )
                geometric_collision = center_clearance <= float(args.robot_radius)
            else:
                surface_clearance = float("nan")
                geometric_collision = False
            rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
            pose = np.asarray(
                [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w],
                dtype=np.float32,
            )

            if args.archive_observations:
                rows["rgb"].append(rgb)
                rows["depth"].append(depth)
                rows["goal_mask"].append(goal_mask)
                rows["live_goal_mask"].append(live_goal_mask)
                rows["ghost_goal_mask"].append(ghost_goal_mask)
                rows["obstacle_mask"].append(obstacle_mask)
            rows["pose"].append(pose)
            rows["action_3d"].append(action)
            rows["point_goal"].append(planner_goal)
            rows["belief_goal_mu"].append(belief_goal_mu)
            rows["belief_goal_covariance"].append(belief_goal_covariance)
            rows["belief_goal_pixel"].append(belief_goal_pixel)
            rows["belief_goal_visible"].append(belief_goal_visible)
            rows["belief_goal_source"].append(belief_goal_source)
            rows["belief_goal_time_since_seen"].append(belief_goal_time_since_seen)
            rows["belief_goal_bearing_rad"].append(belief_goal_bearing)
            rows["belief_goal_pixel_sigma"].append(belief_goal_pixel_sigma)
            rows["belief_heading_recovery_active"].append(belief_recovery_active)
            rows["selected_trajectory"].append(result.trajectory)
            rows["all_trajectories"].append(result.all_trajectories)
            rows["all_values"].append(result.all_values)
            rows["selected_index"].append(result.selected_index)
            rows["fallback_stop"].append(result.fallback_stop)
            rows["escape_turn"].append(result.escape_turn)
            rows["valid_obstacle_points"].append(result.valid_obstacle_points)
            rows["selected_circulation_sign"].append(result.selected_circulation_sign)
            rows["candidate_circulation_signs"].append(
                result.candidate_circulation_signs
            )
            rows["selected_barrier_energy"].append(result.selected_barrier_energy)
            rows["selected_lyapunov_energy"].append(result.selected_lyapunov_energy)
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
            rows["mean_final_effective_sample_size"].append(
                result.mean_final_effective_sample_size
            )
            rows["goal_distance"].append(goal_distance)
            rows["executed_center_clearance"].append(center_clearance)
            rows["executed_surface_clearance"].append(surface_clearance)
            rows["geometric_collision"].append(geometric_collision)
            rows["obstacle_positions_world"].append(
                np.stack(mesh_current_centroids)
                if mesh_current_centroids
                else np.zeros((0, 3), dtype=np.float64)
            )

            rows["qwen_homotopy_sign"].append(forced_circulation_sign)
            rows["qwen_homotopy_side"].append(
                homotopy_decision.side if homotopy_decision is not None else "AUTO"
            )
            rows["qwen_homotopy_confidence"].append(
                homotopy_decision.confidence if homotopy_decision is not None else 0.0
            )
            rows["qwen_homotopy_queried"].append(
                homotopy_decision.queried_qwen
                if homotopy_decision is not None
                else False
            )

            if args.save_frames and step % max(int(args.save_every), 1) == 0:

                side_label = (
                    homotopy_decision.side if homotopy_decision is not None else "AUTO"
                )
                label = (
                    f"t={step} phase={mission_phase} goal={goal_distance:.2f}m "
                    f"qwen_side={side_label} pixels={len(obstacle_pixels)} "
                    f"goal_src={belief_goal_source} "
                    f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
                    f"recover={int(belief_recovery_active)} "
                    f"pred={result.selected_minimum_clearance:.2f}m "
                    f"actual={surface_clearance:.2f}m "
                    f"mode={result.selected_circulation_sign:+.0f} "
                    f"escape={int(result.escape_turn)} "
                    f"guide_rms={result.mean_guidance_noise_correction:.4f} "
                    f"v={action[0]:.2f} w={action[2]:.2f}"
                )
                frame = overlay_frame(
                    rgb,
                    goal_mask,
                    obstacle_mask,
                    label,
                    show_masks=args.overlay_masks,
                )
                frame.save(frame_directory / f"frame_{step:04d}.png")
                video_frames.append(frame)

            print(
                f"step={step:04d} phase={mission_phase} goal={goal_distance:.2f}m "
                f"qwen_side={homotopy_decision.side if homotopy_decision else 'AUTO'} "
                f"goal_src={belief_goal_source} "
                f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
                f"recover={int(belief_recovery_active)} "
                f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
                f"selected={result.selected_index} fallback={result.fallback_stop} "
                f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
                f"pred_clear={result.selected_minimum_clearance:.3f}m "
                f"actual_clear={surface_clearance:.3f}m "
                f"collision={geometric_collision} "
                f"barrier={result.selected_barrier_energy:.5f} "
                f"circ={result.selected_circulation_energy:.5f} "
                f"latency={planning_time * 1000.0:.1f}ms "
                f"guide_rms={result.mean_guidance_noise_correction:.6f} "
                f"ess={result.mean_final_effective_sample_size:.2f} "
                f"action={action.tolist()}",
                flush=True,
            )
            if goal_distance <= args.stop_distance:
                if args.qwen_freeform_mission and mission_phase == "OUTBOUND":
                    if automatic_return_requested:
                        print(
                            "[mission] outward goal reached; advancing automatically "
                            "to RETURN_HOME",
                            flush=True,
                        )
                        mission_phase = "RETURN_HOME"
                        assert homotopy_selector is not None
                        homotopy_selector.reset()
                        continue
                    success = True
                    print(
                        "[mission] outward goal reached; GO_TO_GOAL plan complete",
                        flush=True,
                    )
                    break

                if args.interactive_return_home and mission_phase == "OUTBOUND":
                    user_command = args.return_command
                    if user_command is None:
                        print(
                            "\nOutward goal reached. What should the rover do? "
                            "(for example: come back / stop)",
                            flush=True,
                        )
                        try:
                            user_command = input("> ").strip()
                        except EOFError as error:
                            raise RuntimeError(
                                "no interactive command was available; set "
                                "--return-command 'come back' for a non-interactive run"
                            ) from error
                    assert homotopy_selector is not None
                    command_overlay = overlay_frame(
                        rgb,
                        goal_mask,
                        obstacle_mask,
                        "Qwen command: RETURN or STOP",
                        show_masks=True,
                    )
                    command_decision = homotopy_selector.classify_command(
                        np.asarray(command_overlay.convert("RGB")),
                        user_command,
                    )
                    return_command_event = {
                        "step": step,
                        "user_command": user_command,
                        "command": command_decision.command,
                        "confidence": command_decision.confidence,
                        "raw_response": command_decision.raw_response,
                    }
                    command_overlay.save(output_directory / "qwen_return_command.png")
                    print(
                        f"[qwen-command] text={user_command!r} "
                        f"decision={command_decision.command} "
                        f"confidence={command_decision.confidence:.2f}",
                        flush=True,
                    )
                    if command_decision.command == "RETURN":
                        mission_phase = "RETURN_HOME"
                        homotopy_selector.reset()
                        continue
                    success = True
                    break
                success = True
                if mission_phase == "RETURN_HOME":
                    roundtrip_completed = True
                break

        if not rows["goal_distance"]:
            raise RuntimeError("rollout produced no steps")
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
            obstacle_velocity_xz=mesh_velocities,
            success=np.asarray(success),
            hz=np.asarray(args.hz, dtype=np.float32),
            start_position_xz=start_position_xz,
            initial_goal_distance=np.asarray(initial_goal_distance, dtype=np.float64),
            stop_distance=np.asarray(args.stop_distance, dtype=np.float64),
            robot_radius=np.asarray(args.robot_radius, dtype=np.float64),
            evaluation_layout=np.asarray(args.evaluation_layout),
            seed=np.asarray(args.seed, dtype=np.int64),
            planner_mode=np.asarray(args.planner_mode),
            candidate_count=np.asarray(args.candidates, dtype=np.int64),
            particles_per_candidate=np.asarray(args.particles, dtype=np.int64),
            particle_std=np.asarray(args.particle_std, dtype=np.float64),
            lyapunov_weight=np.asarray(args.lyapunov_weight, dtype=np.float64),
            barrier_weight=np.asarray(args.barrier_weight, dtype=np.float64),
            guidance_strength=np.asarray(args.guidance_strength, dtype=np.float64),
            temperature=np.asarray(args.temperature, dtype=np.float64),
            particle_anchor=np.asarray(args.particle_anchor),
            particle_energy_reweighting=np.asarray(args.particle_energy_reweighting),
            particle_collision_mask=np.asarray(args.particle_collision_mask),
            particle_noise_schedule=np.asarray(args.particle_noise_schedule),
            progressive_guidance=np.asarray(args.progressive_guidance),
            goal_mode=np.asarray(args.goal_mode),
            belief_pixel_goal=np.asarray(args.belief_pixel_goal),
            interactive_return_home=np.asarray(args.interactive_return_home),
            qwen_freeform_mission=np.asarray(args.qwen_freeform_mission),
            automatic_return_requested=np.asarray(automatic_return_requested),
            roundtrip_completed=np.asarray(roundtrip_completed),
            final_mission_phase=np.asarray(mission_phase),
        )
        with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "success": success,
                    "steps": len(rows["goal_distance"]),
                    "archived_observations": args.archive_observations,
                    "final_goal_distance": float(rows["goal_distance"][-1]),
                    "planner": "released_navdp_s2diff_pixels",
                    "controller": "direct_waypoint_no_optimizer",
                    "qwen_role": "homotopy_return_command_and_mission_plan",
                    "qwen_process_isolated_from_habitat": True,
                    "qwen_creates_goal_or_action": False,
                    "qwen_homotopy": args.qwen_homotopy,
                    "qwen_homotopy_events": homotopy_events,
                    "qwen_homotopy_forces_all_candidates": args.qwen_homotopy,
                    "homotopy_sign_convention": {"LEFT": -1.0, "RIGHT": 1.0},
                    "homotopy_minimum_obstacle_pixels": args.homotopy_minimum_obstacle_pixels,
                    "homotopy_release_clear_frames": args.homotopy_release_clear_frames,
                    "homotopy_consistency_repeats": args.homotopy_consistency_repeats,
                    "uses_velocity_chunk": False,
                    "obstacle_mode": args.obstacle_mode,
                    "obstacle_world_xz": args.obstacle_world_xz,
                    "goal_mesh": args.goal_mesh,
                    "candidate_count": args.candidates,
                    "particles_per_candidate": args.particles,
                    "particle_std": args.particle_std,
                    "lyapunov_weight": args.lyapunov_weight,
                    "barrier_weight": args.barrier_weight,
                    "pixelgoal_metric_belief_guidance": bool(
                        args.goal_mode == "pixel" and args.belief_pixel_goal
                    ),
                    "guidance_strength": args.guidance_strength,
                    "temperature": args.temperature,
                    "particle_anchor": args.particle_anchor,
                    "particle_energy_reweighting": args.particle_energy_reweighting,
                    "particle_collision_mask": args.particle_collision_mask,
                    "goal_mode": args.goal_mode,
                    "interactive_return_home": args.interactive_return_home,
                    "qwen_freeform_mission": args.qwen_freeform_mission,
                    "mission_plan_event": mission_plan_event,
                    "automatic_return_requested": automatic_return_requested,
                    "phase_completion_source": "metric_distance_state_machine",
                    "roundtrip_completed": roundtrip_completed,
                    "final_mission_phase": mission_phase,
                    "return_command_event": return_command_event,
                    "home_belief_source": "spawn_origin_plus_executed_odometry",
                    "reached_goal_becomes_obstacle_on_return": True,
                    "return_goal_obstacle_activation_distance": (
                        args.return_goal_obstacle_activation_distance
                    ),
                    "return_goal_obstacle_dilation_pixels": (
                        args.return_goal_obstacle_dilation_pixels
                    ),
                    "belief_pixel_goal": args.belief_pixel_goal,
                    "belief_source": "semantic_goal_mask_plus_odometry",
                    "belief_bootstrap_world_goal": args.belief_bootstrap_world_goal,
                    "belief_measurement_std": args.belief_measurement_std,
                    "belief_translation_process_std": args.belief_translation_process_std,
                    "belief_yaw_process_std_deg": args.belief_yaw_process_std_deg,
                    "belief_covariance_controls_navdp_mask_size": False,
                    "belief_heading_recovery": args.belief_heading_recovery,
                    "belief_recovery_obstacle_gated": True,
                    "belief_recovery_bearing_deg": args.belief_recovery_bearing_deg,
                    "belief_recovery_maximum_yaw_rate": args.belief_recovery_maximum_yaw_rate,
                    "belief_recovery_maximum_forward_speed": args.belief_recovery_maximum_forward_speed,
                    "particle_noise_schedule": args.particle_noise_schedule,
                    "progressive_guidance": args.progressive_guidance,
                    "mesh_obstacle_count": len(mesh_centroids),
                    "moving_obstacles": bool(np.any(np.abs(mesh_velocities) > 0.0)),
                    "obstacle_velocity_xz": mesh_velocities.tolist(),
                    "evaluation_layout": args.evaluation_layout,
                    "seed": args.seed,
                    "robot_radius": args.robot_radius,
                    "minimum_executed_surface_clearance": (
                        float(np.nanmin(rows["executed_surface_clearance"]))
                        if np.any(np.isfinite(rows["executed_surface_clearance"]))
                        else None
                    ),
                    "geometric_collision": bool(np.any(rows["geometric_collision"])),
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
        stop_server(qwen_process)


if __name__ == "__main__":
    main()
