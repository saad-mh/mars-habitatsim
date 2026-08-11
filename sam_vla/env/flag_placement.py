"""Randomized, seed-reproducible flag-marker placement using the .glb assets
under assets/flags/ -- decorative waypoint markers, never a goal/obstacle
candidate (see FLAG_SEMANTIC_ID in sam_vla.core.goal_geometry).

Unlike sam_vla.env.rock_generation, flag geometry is fixed (five pre-made
.glb files) -- only placement is randomized, so there's no per-instance mesh
to bake/cache. Positions are generated live from a seed every env startup
(same seed + same terrain + same config => same flags, every call), no
manifest file needed. Generation only needs a `Terrain` height sampler (see
sam_vla.env.terrain) -- no habitat_sim / GPU required. Registering the
flags into a live sim is a separate step (`register_flags`), called from
MarsHabitatEnv once a Simulator exists.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from sam_vla.core.goal_geometry import FLAG_SEMANTIC_ID
from sam_vla.env.terrain import SIZE_X, SIZE_Z, Terrain

ExcludeZone = Tuple[float, float, float]  # (x, z, radius)

FLAGS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "flags"
# Keys are the clean color names; values are the exact on-disk asset paths
# (note: yelow_flag.glb is genuinely misspelled on disk -- keep the filename
# as-is, only the dict key is corrected).
FLAG_COLORS = {
    "white": FLAGS_DIR / "white_flag.glb",
    "blue": FLAGS_DIR / "blue_flag.glb",
    "green": FLAGS_DIR / "green_flag.glb",
    "red": FLAGS_DIR / "red_flag.glb",
    "yellow": FLAGS_DIR / "yelow_flag.glb",
}

# Used for placement-only bookkeeping (min-spacing between flags, terrain
# sampling radius) -- flags are non-collidable, this is not a physics size.
FLAG_FOOTPRINT_RADIUS = 0.3


@dataclass
class FlagFieldConfig:
    seed: int = 0
    num_flags: int = 6
    min_spacing: float = 1.5
    boundary_margin: float = 2.0  # keep every flag this far from the scene bound
    exclude_zones: List[ExcludeZone] = field(
        default_factory=list
    )  # e.g. a keep-out around the rover's spawn point
    max_attempts_per_flag: int = 300


@dataclass
class FlagSpec:
    id: int
    x: float
    y: float
    z: float
    yaw: float
    color: str
    mesh_path: str


def _too_close(
    x: float, z: float, placed: Sequence[Tuple[float, float]], min_spacing: float
) -> bool:
    return any(math.hypot(x - px, z - pz) < min_spacing for px, pz in placed)


def _in_exclude_zone(x: float, z: float, exclude_zones: Sequence[ExcludeZone]) -> bool:
    return any(math.hypot(x - ex, z - ez) < er for ex, ez, er in exclude_zones)


def generate_flag_field(config: FlagFieldConfig, terrain: Terrain) -> List[FlagSpec]:
    """Rejection-sample `config.num_flags` non-overlapping flag positions within
    the scene bounds, each dropped onto the terrain height under it and
    assigned a random color from FLAG_COLORS. Deterministic given
    `config.seed` -- same config + same terrain => same flags, every call."""
    rng = random.Random(config.seed)
    half_x = SIZE_X / 2.0 - config.boundary_margin
    half_z = SIZE_Z / 2.0 - config.boundary_margin
    if half_x <= 0.0 or half_z <= 0.0:
        raise ValueError(
            f"boundary_margin ({config.boundary_margin}) leaves no room inside the "
            f"{SIZE_X}x{SIZE_Z}m scene"
        )

    colors = list(FLAG_COLORS.keys())
    placed: List[Tuple[float, float]] = []
    flags: List[FlagSpec] = []

    for i in range(config.num_flags):
        placement = None
        for _attempt in range(config.max_attempts_per_flag):
            x = rng.uniform(-half_x, half_x)
            z = rng.uniform(-half_z, half_z)
            if _in_exclude_zone(x, z, config.exclude_zones):
                continue
            if _too_close(x, z, placed, config.min_spacing):
                continue
            placement = (x, z)
            break

        if placement is None:
            print(
                f"[flags] WARN could not place flag {i} after {config.max_attempts_per_flag} attempts; skipping"
            )
            continue

        x, z = placement
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        y = terrain.local_height_max(x, z, FLAG_FOOTPRINT_RADIUS)
        color = rng.choice(colors)

        placed.append((x, z))
        flags.append(
            FlagSpec(
                id=i,
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                color=color,
                mesh_path=str(FLAG_COLORS[color]),
            )
        )

    return flags


def register_flags(
    sim, flags: Sequence[FlagSpec], semantic_id: int = FLAG_SEMANTIC_ID
) -> List:
    """Add every placed flag's .glb asset into the sim as a render-only,
    non-collidable object -- the flags counterpart of
    rock_generation.register_rocks, via sim_utils.register_glb_object since
    (unlike rocks' per-instance baked .obj meshes) flag geometry is shared
    across instances and needs its position/yaw set explicitly."""
    from sam_vla.env.sim_utils import register_glb_object

    return [
        register_glb_object(
            sim,
            flag.mesh_path,
            position=(flag.x, flag.y, flag.z),
            yaw=flag.yaw,
            semantic_id=semantic_id,
            template_name=f"flag_{flag.id}_{flag.color}",
        )
        for flag in flags
    ]
