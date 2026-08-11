"""Ghost home-base marker: a single, fixed, non-collidable blue cuboid placed
a fixed distance directly behind the rover's spawn pose -- a visual anchor
for "home" (nav/rover_controller.py's go_home already drives back to the
spawn (x, z); this only adds something to see there), not a goal/obstacle
candidate (see HOME_BASE_SEMANTIC_ID in sam_vla.core.goal_geometry).

"Ghost" because, like sam_vla.env.flag_placement's markers, it's render-only
and non-collidable -- the rover can drive straight through it. Solid flat
blue (rather than a Mars-palette rock/terrain color) so it reads as
obviously artificial against the yard. Built the same way rock_generation
bakes rocks: a local mesh with the world position baked straight into its
vertices, written to a companion .mtl for color via sim_utils.save_obj, then
registered as a render-only semantic mesh -- no cached manifest, regenerated
fresh into a tempdir every env startup.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

from sam_vla.core.goal_geometry import HOME_BASE_SEMANTIC_ID
from sam_vla.env.sim_utils import register_semantic_mesh, save_obj

HOME_BASE_DISTANCE_M = 5.0
HOME_BASE_HALF_WIDTH_M = 1.0  # 2x2m footprint
HOME_BASE_HEIGHT_M = 1.5
HOME_BASE_COLOR = (0.1, 0.35, 0.95)  # flat blue, distinct from the ochre/red terrain
HOME_BASE_ALPHA = 0.6  # 40% transparent


def home_base_xz(
    start_x: float, start_z: float, start_yaw: float, distance: float = HOME_BASE_DISTANCE_M
) -> Tuple[float, float]:
    """World (x, z) `distance` meters directly behind the rover's spawn pose.

    Uses pose_integrator's heading convention (forward = (-sin(yaw),
    -cos(yaw)) in the x-z plane) -- behind is that vector's negation."""
    return (
        start_x + distance * math.sin(start_yaw),
        start_z + distance * math.cos(start_yaw),
    )


def _cuboid_mesh(
    center_x: float, base_y: float, center_z: float, half_width: float, height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box, footprint (2*half_width)^2 in x-z, resting on
    base_y and extending straight up `height` -- verts/faces wound so every
    face's normal points outward (matters for backface culling, see
    rock_generation._make_rock_mesh's winding)."""
    hx = hz = float(half_width)
    x0, y0, z0 = float(center_x), float(base_y), float(center_z)
    verts = np.asarray(
        [
            (x0 - hx, y0, z0 - hz),  # 0 bottom
            (x0 + hx, y0, z0 - hz),  # 1
            (x0 + hx, y0, z0 + hz),  # 2
            (x0 - hx, y0, z0 + hz),  # 3
            (x0 - hx, y0 + height, z0 - hz),  # 4 top
            (x0 + hx, y0 + height, z0 - hz),  # 5
            (x0 + hx, y0 + height, z0 + hz),  # 6
            (x0 - hx, y0 + height, z0 + hz),  # 7
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (3, 2, 6), (3, 6, 7),  # +Z front
            (1, 0, 4), (1, 4, 5),  # -Z back
            (2, 1, 5), (2, 5, 6),  # +X right
            (0, 3, 7), (0, 7, 4),  # -X left
            (4, 6, 5), (4, 7, 6),  # +Y top
            (0, 1, 2), (0, 2, 3),  # -Y bottom
        ],
        dtype=np.int64,
    )
    return verts, faces


def register_home_base(
    sim,
    terrain,
    start_x: float,
    start_z: float,
    start_yaw: float,
    distance: float = HOME_BASE_DISTANCE_M,
):
    """Bake and register the ghost home-base cuboid `distance` meters behind
    (start_x, start_z, start_yaw), sitting on the terrain height under it.
    Returns the registered rigid object."""
    x, z = home_base_xz(start_x, start_z, start_yaw, distance)
    y = terrain.local_height_max(x, z, HOME_BASE_HALF_WIDTH_M)
    verts, faces = _cuboid_mesh(x, y, z, HOME_BASE_HALF_WIDTH_M, HOME_BASE_HEIGHT_M)

    mesh_dir = Path(tempfile.mkdtemp(prefix="ghost_home_base_"))
    mesh_path = str(mesh_dir / "ghost_home_base.obj")
    save_obj(mesh_path, verts, faces, diffuse_rgb=HOME_BASE_COLOR, alpha=HOME_BASE_ALPHA)
    return register_semantic_mesh(sim, mesh_path, HOME_BASE_SEMANTIC_ID)
