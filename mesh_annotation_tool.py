"""
Interactive 3D mesh segmentation tool (next.md, Pipeline Step 1).

Displays the terrain (built from the heightmap via the same HeightmapGrid
used at runtime, so it matches the world frame the rover/rocks live in) and
any extra OBJ meshes (e.g. procedurally placed rocks) in one 3D scene.
Rotate/zoom to find an object, enable "Pick points", then click directly on
the rendered mesh surface -- each click is ray-cast against the real
triangle geometry to get an exact 3D point on the surface. Click a handful
of points around/over one object (e.g. one rock or one crater), hit
"Finish Object", and the convex hull of those points becomes that object's
mask boundary (per next.md: the hull *is* the mask, not a collision proxy;
overshoot on jagged rocks is accepted label noise). A dialog then asks for
the category + a name, a persistent mesh_id is assigned, and the hull is
written out as its own OBJ alongside a mesh_id -> category JSON registry.

Work is autosaved after every finished/deleted object (including any
in-progress, unfinished points) to <out-dir>/session.json, and is resumed
automatically the next time the tool is pointed at the same --out-dir.

Usage:
    python mesh_annotation_tool.py --out-dir annotations/mesh_segmentation
    python mesh_annotation_tool.py --out-dir annotations/mesh_segmentation \
        --obj-glob "rock_envs/run1/rocks/*.obj"

Run with the `habitat` conda env's python (has tkinter + matplotlib + scipy):
    /home/gpu/miniconda3/envs/habitat/bin/python mesh_annotation_tool.py ...
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sam_vla.env.terrain import HeightmapGrid  # noqa: E402

DEFAULT_HEIGHTMAP = HERE / "marsyard2022_terrain_hm.png"
DEFAULT_TEXTURE = HERE / "marsyard2022_terrain_texture.png"

CATEGORIES = ["small_rock", "big_rock", "bedrock", "hole_in_ground"]
CATEGORY_COLORS = {
    "small_rock": "#e0813e",
    "big_rock": "#b23a2f",
    "bedrock": "#8a8a70",
    "hole_in_ground": "#2f5c8a",
}
FALLBACK_COLORS = ["#7a4fa3", "#3f9e6e", "#c2a83e", "#4f7fa3", "#a34f7f"]

CLICK_MOVE_THRESHOLD_PX = 4
RAY_EPS = 1e-8


# --------------------------------------------------------------------------
# Mesh loading / building
# --------------------------------------------------------------------------

def color_for_category(category: str) -> str:
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    idx = sum(map(ord, category)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ loader: returns (vertices Nx3, triangles Mx3 index array).
    Faces with >3 vertices are fan-triangulated. v/vt/vn face refs are
    accepted; only the vertex index is used."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()[1:4]
                verts.append([float(p) for p in parts])
            elif line.startswith("f "):
                raw = line.split()[1:]
                idxs = [int(tok.split("/")[0]) for tok in raw]
                idxs = [i - 1 if i > 0 else len(verts) + i for i in idxs]
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


@dataclass
class SceneMesh:
    name: str
    vertices: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (M, 3) int index into vertices
    # Rendering-only extras (terrain uses plot_surface, obj meshes use Poly3DCollection)
    grid_xyz: Optional[tuple] = None  # (X, Z, Y) 2D arrays, terrain only
    facecolors: Optional[np.ndarray] = None  # terrain only


def _sample_grid_bilinear(arr: np.ndarray, xs: np.ndarray, zs: np.ndarray, size_x: float, size_z: float,
                           flip_x: bool, flip_z: bool, swap_xz: bool) -> np.ndarray:
    """Vectorized bilinear sample of a (h, w) or (h, w, C) source array at
    world (xs, zs) (2D grids of matching shape), using the exact same
    world->uv convention as HeightmapGrid._to_uv/bilinear_sample -- so
    display sampling stays consistent with the height lookup used at
    runtime, just fast enough to run at high grid_res instead of a scalar
    per-vertex Python loop."""
    xx, zz = (zs, xs) if swap_xz else (xs, zs)
    u = (xx + size_x / 2.0) / size_x
    v = (zz + size_z / 2.0) / size_z
    if flip_x:
        u = 1.0 - u
    if flip_z:
        v = 1.0 - v
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    h, w = arr.shape[0], arr.shape[1]
    px = u * (w - 1)
    py = v * (h - 1)
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    dx = px - x0
    dy = py - y0
    if arr.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    v00, v10 = arr[y0, x0], arr[y0, x1]
    v01, v11 = arr[y1, x0], arr[y1, x1]
    top = v00 * (1 - dx) + v10 * dx
    bot = v01 * (1 - dx) + v11 * dx
    return top * (1 - dy) + bot * dy


def build_terrain_mesh(
    heightmap_path: Path,
    texture_path: Optional[Path],
    size_x: float,
    size_z: float,
    size_y: float,
    flip_x: bool,
    flip_z: bool,
    swap_xz: bool,
    grid_res: int,
) -> SceneMesh:
    grid = HeightmapGrid(
        heightmap_path,
        size_x=size_x,
        size_z=size_z,
        size_y=size_y,
        flip_x=flip_x,
        flip_z=flip_z,
        swap_xz=swap_xz,
    )
    xs = np.linspace(-size_x / 2.0, size_x / 2.0, grid_res)
    zs = np.linspace(-size_z / 2.0, size_z / 2.0, grid_res)
    X, Z = np.meshgrid(xs, zs)
    Y = _sample_grid_bilinear(grid._height, X, Z, size_x, size_z, flip_x, flip_z, swap_xz)

    facecolors = None
    if texture_path is not None and texture_path.exists():
        from PIL import Image

        tex = np.asarray(Image.open(texture_path).convert("RGB")).astype(np.float32) / 255.0
        vertex_colors = _sample_grid_bilinear(tex, X, Z, size_x, size_z, flip_x, flip_z, swap_xz)
        # Per-quad facecolor = average of its 4 corners.
        fc_rgb = 0.25 * (
            vertex_colors[:-1, :-1] + vertex_colors[1:, :-1] + vertex_colors[:-1, 1:] + vertex_colors[1:, 1:]
        )

        # Lambertian shading from the actual 3D surface normal, so rock/crater
        # *shape* shows up as highlight/shadow -- otherwise (flat texture
        # color only) camouflage-colored rocks are nearly invisible against
        # matching-color sand, defeating the entire point of doing this
        # segmentation in 3D instead of on a flat photo.
        p00 = np.stack([X[:-1, :-1], Z[:-1, :-1], Y[:-1, :-1]], axis=-1)
        p10 = np.stack([X[:-1, 1:], Z[:-1, 1:], Y[:-1, 1:]], axis=-1)
        p01 = np.stack([X[1:, :-1], Z[1:, :-1], Y[1:, :-1]], axis=-1)
        p11 = np.stack([X[1:, 1:], Z[1:, 1:], Y[1:, 1:]], axis=-1)
        normal = np.cross(p11 - p00, p10 - p01)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12
        light_dir = np.array([0.35, 0.35, 0.87])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip(normal @ light_dir, 0.0, 1.0)
        ambient = 0.45
        brightness = ambient + (1.0 - ambient) * diffuse
        fc_rgb = np.clip(fc_rgb * brightness[..., None], 0.0, 1.0)

        facecolors = np.dstack([fc_rgb, np.ones(fc_rgb.shape[:2], dtype=np.float32)])

    # Flatten grid into a vertex array + triangle list for ray picking.
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    row, col = np.meshgrid(np.arange(grid_res - 1), np.arange(grid_res - 1), indexing="ij")
    i0 = (row * grid_res + col).ravel()
    i1 = (row * grid_res + col + 1).ravel()
    i2 = ((row + 1) * grid_res + col + 1).ravel()
    i3 = ((row + 1) * grid_res + col).ravel()
    triangles = np.concatenate(
        [np.stack([i0, i1, i2], axis=-1), np.stack([i0, i2, i3], axis=-1)], axis=0
    ).astype(np.int64)

    return SceneMesh(
        name="terrain",
        vertices=verts,
        triangles=triangles,
        grid_xyz=(X, Z, Y),
        facecolors=facecolors,
    )


# --------------------------------------------------------------------------
# Ray picking
# --------------------------------------------------------------------------

def ray_triangle_intersect_batch(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray, t_min: float = RAY_EPS):
    """Vectorized Moeller-Trumbore against all triangles. Returns the closest
    hit point (3,) or None. `tris` has shape (K, 3, 3) = (tri, vertex, xyz).
    `t_min` is the smallest acceptable distance-along-ray-from-`origin` --
    pass an offset (not just RAY_EPS) when `origin` has been recentered away
    from the camera's true near point (see `unproject_ray`)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(direction, e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(a) > RAY_EPS
    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    valid &= (u >= -1e-6) & (u <= 1.0 + 1e-6)
    q = np.cross(s, e1)
    dir_b = np.broadcast_to(direction, e2.shape)
    v = f * np.einsum("ij,ij->i", dir_b, q)
    valid &= (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
    t = f * np.einsum("ij,ij->i", e2, q)
    valid &= t > t_min
    if not np.any(valid):
        return None
    t_masked = np.where(valid, t, np.inf)
    idx = int(np.argmin(t_masked))
    if not np.isfinite(t_masked[idx]):
        return None
    return origin + t_masked[idx] * direction


def unproject_ray(ax, xdata: float, ydata: float):
    """Build a world-space ray through the given (already-projected) 2D
    axes coordinates of a click on a 3D axes. Uses two safely-nonzero NDC
    depths (matplotlib's inv_transform is singular at depth 0).

    Returns (origin, direction, t_min): the raw unprojected point can land
    arbitrarily far from the actual scene (hundreds of units away,
    depending on the current view/projection scaling) even though the ray
    direction itself is correct. Downstream ray-triangle intersection then
    computes `origin + t*direction` for a large `t`, subtracting two large
    near-equal floats and destroying precision. Recentering the origin onto
    the point of the same line nearest the axes' own data bounds keeps
    everything numerically close to the mesh without changing the line
    itself -- but that shifts what "in front of the camera" (t > 0) means
    relative to the new origin, so `t_min` (the shifted validity threshold,
    NOT a fixed epsilon) must be passed through to
    `ray_triangle_intersect_batch` instead of assuming t=0 is the boundary."""
    M = ax.get_proj()
    invM = np.linalg.inv(M)
    p1 = np.array(proj3d.inv_transform(xdata, ydata, 0.4, invM)).reshape(3)
    p2 = np.array(proj3d.inv_transform(xdata, ydata, 0.6, invM)).reshape(3)
    direction = p2 - p1
    direction /= np.linalg.norm(direction)
    center = np.array(
        [sum(ax.get_xlim3d()) / 2.0, sum(ax.get_ylim3d()) / 2.0, sum(ax.get_zlim3d()) / 2.0]
    )
    t0 = np.dot(center - p1, direction)
    origin = p1 + t0 * direction
    t_min = -t0 + RAY_EPS
    return origin, direction, t_min


def world_to_uv(x: float, z: float, size_x: float, size_z: float, flip_x: bool, flip_z: bool, swap_xz: bool):
    """Mirrors HeightmapGrid._to_uv exactly -- world (x, z) -> texture (u, v)
    in [0, 1], for placing markers on the 2D texture reference panel."""
    if swap_xz:
        x, z = z, x
    u = (x + size_x / 2.0) / size_x
    v = (z + size_z / 2.0) / size_z
    if flip_x:
        u = 1.0 - u
    if flip_z:
        v = 1.0 - v
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def uv_to_world(u: float, v: float, size_x: float, size_z: float, flip_x: bool, flip_z: bool, swap_xz: bool):
    """Inverse of world_to_uv -- a click on the 2D texture panel back to
    world (x, z) (height is then looked up separately via the terrain
    mesh's own grid, see bilinear_height_query)."""
    if flip_x:
        u = 1.0 - u
    if flip_z:
        v = 1.0 - v
    x = u * size_x - size_x / 2.0
    z = v * size_z - size_z / 2.0
    if swap_xz:
        x, z = z, x
    return float(x), float(z)


def bilinear_height_query(mesh: "SceneMesh", xq: np.ndarray, zq: np.ndarray) -> np.ndarray:
    """Vectorized bilinear height sample over a terrain SceneMesh's own
    (grid_res x grid_res) display grid -- deliberately the same resolution
    that's rendered/picked against, not the full-res heightmap, so a click
    always lands exactly on the surface the user is actually looking at."""
    X, Z, Y = mesh.grid_xyz
    xs, zs = X[0, :], Z[:, 0]
    nx, nz = len(xs), len(zs)
    fx = np.clip((xq - xs[0]) / (xs[-1] - xs[0]) * (nx - 1), 0, nx - 1 - 1e-6)
    fz = np.clip((zq - zs[0]) / (zs[-1] - zs[0]) * (nz - 1), 0, nz - 1 - 1e-6)
    x0 = np.floor(fx).astype(int)
    z0 = np.floor(fz).astype(int)
    x1, z1 = x0 + 1, z0 + 1
    tx, tz = fx - x0, fz - z0
    h00, h10 = Y[z0, x0], Y[z0, x1]
    h01, h11 = Y[z1, x0], Y[z1, x1]
    h0 = h00 * (1 - tx) + h10 * tx
    h1 = h01 * (1 - tx) + h11 * tx
    return h0 * (1 - tz) + h1 * tz


def raymarch_terrain(mesh: "SceneMesh", origin_render: np.ndarray, direction_render: np.ndarray, t_min: float, steps: int = 800):
    """Ray-march a (render-order x,z,y) camera ray against the terrain's own
    height grid and return (world_xyz_hit, t) or None.

    Möller-Trumbore against the heightfield's triangles is numerically
    unstable for near-horizontal viewing angles: the ray-to-triangle-plane
    angle (and hence the algorithm's denominator) shrinks with the ray's
    vertical component, not with how "edge-on" the click actually is, so
    ordinary oblique/low camera angles amplify float noise by orders of
    magnitude and silently return a wrong triangle (verified empirically --
    same failure with 0.0001 and 4.82 unit terrain height ranges alike).
    Marching the height function directly sidesteps that entirely, since it
    only ever solves for where ray-height crosses terrain-height along x,z,
    with no edge-cross-product ratio to blow up."""
    origin = origin_render[[0, 2, 1]]  # -> world (x, y, z)
    direction = direction_render[[0, 2, 1]]
    X, Z, _ = mesh.grid_xyz
    xs, zs = X[0, :], Z[:, 0]
    x_lo, x_hi = float(xs.min()), float(xs.max())
    z_lo, z_hi = float(zs.min()), float(zs.max())

    def axis_bounds(o_c, d_c, lo, hi):
        if abs(d_c) < 1e-12:
            return (-np.inf, np.inf) if lo <= o_c <= hi else (np.inf, -np.inf)
        t_a, t_b = (lo - o_c) / d_c, (hi - o_c) / d_c
        return (min(t_a, t_b), max(t_a, t_b))

    tx_lo, tx_hi = axis_bounds(origin[0], direction[0], x_lo, x_hi)
    tz_lo, tz_hi = axis_bounds(origin[2], direction[2], z_lo, z_hi)
    t_start = max(t_min, tx_lo, tz_lo)
    t_end = min(tx_hi, tz_hi)
    if not (np.isfinite(t_start) and np.isfinite(t_end)) or t_start >= t_end:
        return None

    ts = np.linspace(t_start, t_end, steps)
    xq = origin[0] + ts * direction[0]
    zq = origin[2] + ts * direction[2]
    yq = origin[1] + ts * direction[1]
    diff = yq - bilinear_height_query(mesh, xq, zq)
    crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(crossings) == 0:
        return None
    i = int(crossings[0])  # first (smallest-t) crossing == nearest the camera
    d_a, d_b = diff[i], diff[i + 1]
    frac = d_a / (d_a - d_b) if (d_a - d_b) != 0 else 0.5
    t_hit = ts[i] + frac * (ts[i + 1] - ts[i])
    hit = np.array([origin[0] + t_hit * direction[0], origin[1] + t_hit * direction[1], origin[2] + t_hit * direction[2]])
    return hit, t_hit


# --------------------------------------------------------------------------
# Session / registry persistence
# --------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip())
    return name or "object"


@dataclass
class AnnotatedObject:
    mesh_id: int
    name: str
    category: str
    points: list  # raw clicked points [[x,y,z], ...]
    hull_points: list  # points actually fed to ConvexHull (post coplanar-fix)
    hull_faces: list  # [[i,j,k], ...] indices into hull_points
    obj_path: str

    def to_json(self):
        return {
            "mesh_id": self.mesh_id,
            "name": self.name,
            "category": self.category,
            "points": self.points,
            "hull_points": self.hull_points,
            "hull_faces": self.hull_faces,
            "obj_path": self.obj_path,
        }

    @staticmethod
    def from_json(d):
        return AnnotatedObject(
            mesh_id=d["mesh_id"],
            name=d["name"],
            category=d["category"],
            points=d["points"],
            hull_points=d["hull_points"],
            hull_faces=d["hull_faces"],
            obj_path=d["obj_path"],
        )


class SessionStore:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.subobj_dir = out_dir / "subobjects"
        self.session_path = out_dir / "session.json"
        self.registry_path = out_dir / "mesh_id_map.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.subobj_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[list[AnnotatedObject], int, list]:
        if not self.session_path.exists():
            return [], 1000, []
        data = json.loads(self.session_path.read_text())
        objects = [AnnotatedObject.from_json(d) for d in data.get("objects", [])]
        next_id = data.get("next_mesh_id", 1000)
        pending_points = data.get("pending_points", [])
        return objects, next_id, pending_points

    def save(self, objects: list[AnnotatedObject], next_mesh_id: int, pending_points: list):
        self.session_path.write_text(
            json.dumps(
                {
                    "next_mesh_id": next_mesh_id,
                    "pending_points": pending_points,
                    "objects": [o.to_json() for o in objects],
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        self.registry_path.write_text(
            json.dumps(
                {
                    "mesh_id_map": {
                        str(o.mesh_id): {"category": o.category, "name": o.name} for o in objects
                    }
                },
                indent=2,
            )
        )

    def write_hull_obj(self, obj: AnnotatedObject):
        path = self.subobj_dir / Path(obj.obj_path).name
        with open(path, "w") as f:
            f.write(f"o mesh_{obj.mesh_id}_{obj.name}\n")
            for p in obj.hull_points:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for tri in obj.hull_faces:
                f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

    def delete_hull_obj(self, obj: AnnotatedObject):
        path = self.subobj_dir / Path(obj.obj_path).name
        path.unlink(missing_ok=True)


def compute_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull of `points`. Falls back to a thin extrusion along the
    point cloud's own normal (found via SVD, not assumed to be world-up) if
    the points are (near-)coplanar or collinear, since a valid 3D hull needs
    actual volume -- this is the common case since 3-4 marked points are
    often nearly flat."""
    try:
        hull = ConvexHull(points)
        return points, hull.simplices
    except QhullError:
        centered = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered)
        normal = vt[-1]  # least-variance direction
        eps = 0.01
        thick = np.vstack([points + eps * normal, points - eps * normal])
        hull = ConvexHull(thick)
        return thick, hull.simplices


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class MeshAnnotatorApp:
    def __init__(
        self,
        root: tk.Tk,
        meshes: list[SceneMesh],
        out_dir: Path,
        texture_path: Optional[Path] = None,
        size_x: float = 50.0,
        size_z: float = 50.0,
        flip_x: bool = False,
        flip_z: bool = True,
        swap_xz: bool = False,
    ):
        self.root = root
        self.meshes = meshes
        self.store = SessionStore(out_dir)
        self.uv_params = dict(size_x=size_x, size_z=size_z, flip_x=flip_x, flip_z=flip_z, swap_xz=swap_xz)

        self.objects, self.next_mesh_id, pending = self.store.load()
        self.current_points: list[list[float]] = [list(p) for p in pending]

        # Full-resolution texture reference panel: the 3D view colors each
        # mesh quad by *averaging* the texels inside it, so at any geometry
        # resolution that's still interactive to rotate, pebble-sized rocks
        # get averaged away into the surrounding sand color even though
        # they're clearly visible in the source photo. Clicking here marks
        # a point via the same uv_to_world + height-lookup path a 3D pick
        # would produce, feeding the same current_points list.
        self.tex_img = None
        if texture_path is not None and Path(texture_path).exists():
            from PIL import Image

            self.tex_img = Image.open(texture_path).convert("RGB")
        self.tex_center = None  # (px, py) in source-image pixels; set on first layout
        self.tex_zoom = None  # source pixels per canvas pixel
        self.tex_crop_box = None  # last-rendered (left, top, right, bottom) in source pixels
        self._tex_photo = None  # keep a reference so Tk doesn't GC the PhotoImage
        self._tex_pan_xy = None

        # Terrain (a heightfield) is picked via raymarch_terrain -- Moeller-
        # Trumbore against its near-horizontal, grid-thin triangles is
        # numerically unstable for ordinary oblique/low camera angles (see
        # raymarch_terrain's docstring). Generic OBJ meshes (rocks etc.)
        # don't have that systemic degeneracy, so they still use the
        # triangle soup below. `ax.get_proj()` reflects whatever axis order
        # things are plotted in -- terrain uses plot_surface(X, Z, Y) and
        # obj meshes are drawn via a (x, z, y) swap in _redraw() -- so
        # picking must ray-cast in that SAME render order (x, z, y), not the
        # (x, y, z) world order `vertices` is stored in (world order is kept
        # for hull/export). Hits are swapped back to world order in
        # `_pick_point`.
        self.terrain_meshes = [m for m in meshes if m.grid_xyz is not None]
        obj_meshes = [m for m in meshes if m.grid_xyz is None]
        tri_arrays = [m.vertices[m.triangles][:, :, [0, 2, 1]] for m in obj_meshes if len(m.triangles)]
        self.pick_tris = np.concatenate(tri_arrays, axis=0) if tri_arrays else np.zeros((0, 3, 3))

        self.pick_mode = tk.BooleanVar(value=False)
        self._press_xy = None

        self._build_ui()
        self._refresh_listbox()
        self._redraw()
        if self.objects or self.current_points:
            self._status(f"Resumed session: {len(self.objects)} object(s), "
                         f"{len(self.current_points)} pending point(s).")

    # -- UI construction ---------------------------------------------------

    def _build_ui(self):
        self.root.title("Mesh Segmentation Annotator")
        self.root.geometry("1650x900" if self.tex_img is not None else "1280x820")

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(paned)
        control_frame = ttk.Frame(paned, width=320)
        paned.add(canvas_frame, weight=3)
        if self.tex_img is not None:
            texture_frame = ttk.Frame(paned)
            paned.add(texture_frame, weight=3)
        paned.add(control_frame, weight=2)

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        if self.tex_img is not None:
            ttk.Label(texture_frame, text="Full-res texture (reference + click to mark)").pack(
                anchor="w", padx=4, pady=(4, 0)
            )
            self.tex_canvas = tk.Canvas(texture_frame, bg="#222", highlightthickness=0)
            self.tex_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            self.tex_canvas.bind("<Configure>", self._on_tex_configure)
            self.tex_canvas.bind("<Button-1>", self._on_tex_click)
            self.tex_canvas.bind("<ButtonPress-3>", self._on_tex_pan_press)
            self.tex_canvas.bind("<B3-Motion>", self._on_tex_pan_move)
            self.tex_canvas.bind("<MouseWheel>", self._on_tex_scroll)  # Windows/Mac
            self.tex_canvas.bind("<Button-4>", self._on_tex_scroll)  # Linux wheel up
            self.tex_canvas.bind("<Button-5>", self._on_tex_scroll)  # Linux wheel down

        # -- controls --
        ttk.Checkbutton(control_frame, text="Pick points (click mesh to add)", variable=self.pick_mode).pack(
            anchor="w", padx=8, pady=(10, 2)
        )
        ttk.Label(
            control_frame,
            text="3D: scroll = zoom, drag = orbit. Texture: scroll = zoom, right-drag = pan.",
            foreground="#666", wraplength=300,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        self.point_count_label = ttk.Label(control_frame, text="Current object: 0 points")
        self.point_count_label.pack(anchor="w", padx=8, pady=(0, 4))

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(btn_frame, text="Undo last point", command=self.undo_point).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Cancel current object", command=self.cancel_current).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Finish object -> hull + label", command=self.finish_object).pack(
            fill=tk.X, pady=(2, 10)
        )

        ttk.Separator(control_frame).pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(control_frame, text="Finished objects:").pack(anchor="w", padx=8)
        list_frame = ttk.Frame(control_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.listbox = tk.Listbox(list_frame)
        self.listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scrollbar = ttk.Scrollbar(list_frame, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        ttk.Button(control_frame, text="Delete selected object", command=self.delete_selected).pack(
            fill=tk.X, padx=8, pady=(2, 10)
        )

        ttk.Separator(control_frame).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(control_frame, text="Save session now", command=self._save).pack(fill=tk.X, padx=8, pady=2)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=300, foreground="#345").pack(
            anchor="w", padx=8, pady=(10, 8)
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _status(self, msg: str):
        self.status_var.set(msg)

    # -- rendering -----------------------------------------------------

    def _redraw(self):
        self._redraw_3d()
        self._redraw_texture_panel()

    def _redraw_3d(self):
        self.ax.cla()
        for mesh in self.meshes:
            if mesh.grid_xyz is not None:
                X, Z, Y = mesh.grid_xyz
                if mesh.facecolors is not None:
                    self.ax.plot_surface(
                        X, Z, Y, facecolors=mesh.facecolors, rstride=1, cstride=1, shade=False,
                        linewidth=0, antialiased=False,
                    )
                else:
                    self.ax.plot_surface(
                        X, Z, Y, cmap="terrain", rstride=1, cstride=1, linewidth=0, antialiased=False,
                    )
            else:
                verts = mesh.vertices[mesh.triangles]  # (K,3,3) in (x,y,z)
                verts_render = verts[:, :, [0, 2, 1]]  # plot as (x, z, y)
                poly = Poly3DCollection(verts_render, facecolor="#c98a55", edgecolor="none", alpha=0.9)
                self.ax.add_collection3d(poly)

        for obj in self.objects:
            self._draw_object(obj)

        if self.current_points:
            pts = np.array(self.current_points)
            self.ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], color="red", s=40, depthshade=False)
            if len(pts) > 1:
                loop = np.vstack([pts, pts[:1]])
                self.ax.plot(loop[:, 0], loop[:, 2], loop[:, 1], color="red", linewidth=1.5)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("z")
        self.ax.set_zlabel("y (height)")
        self.canvas.draw_idle()
        self.point_count_label.config(text=f"Current object: {len(self.current_points)} points")

    def _draw_object(self, obj: AnnotatedObject):
        pts = np.array(obj.hull_points)
        faces = np.array(obj.hull_faces)
        color = color_for_category(obj.category)
        tris = pts[faces][:, :, [0, 2, 1]]
        poly = Poly3DCollection(tris, facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.55)
        self.ax.add_collection3d(poly)

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for obj in self.objects:
            self.listbox.insert(tk.END, f"[{obj.mesh_id}] {obj.name}  ({obj.category})")

    # -- 2D texture reference panel --------------------------------------

    def _on_tex_configure(self, _event):
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        if self.tex_center is None:
            # First layout: fit the whole image to the canvas.
            tw, th = self.tex_img.size
            self.tex_center = [tw / 2.0, th / 2.0]
            self.tex_zoom = max(tw / cw, th / ch)
        self._redraw_texture_panel()

    def _tex_view_box(self):
        """Current (left, top, right, bottom) crop box in source pixels,
        clamped to stay inside the image."""
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        tw, th = self.tex_img.size
        half_w, half_h = cw * self.tex_zoom / 2.0, ch * self.tex_zoom / 2.0
        cx = min(max(self.tex_center[0], half_w), tw - half_w) if tw > 2 * half_w else tw / 2.0
        cy = min(max(self.tex_center[1], half_h), th - half_h) if th > 2 * half_h else th / 2.0
        self.tex_center = [cx, cy]
        left, top = cx - half_w, cy - half_h
        right, bottom = cx + half_w, cy + half_h
        return left, top, right, bottom

    def _redraw_texture_panel(self):
        if self.tex_img is None or self.tex_center is None:
            return
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        from PIL import ImageTk

        left, top, right, bottom = self._tex_view_box()
        crop = self.tex_img.crop((int(left), int(top), int(right), int(bottom)))
        crop = crop.resize((cw, ch))
        self.tex_crop_box = (left, top, right, bottom)
        self._tex_photo = ImageTk.PhotoImage(crop)
        self.tex_canvas.delete("all")
        self.tex_canvas.create_image(0, 0, anchor="nw", image=self._tex_photo)

        def to_canvas_xy(x, z):
            u, v = world_to_uv(x, z, **self.uv_params)
            tw, th = self.tex_img.size
            px, py = u * (tw - 1), v * (th - 1)
            return (px - left) / self.tex_zoom, (py - top) / self.tex_zoom

        for obj in self.objects:
            color = color_for_category(obj.category)
            for x, _y, z in obj.hull_points:
                cx, cy = to_canvas_xy(x, z)
                self.tex_canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=color, outline="")
        for x, _y, z in self.current_points:
            cx, cy = to_canvas_xy(x, z)
            self.tex_canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="red", outline="white")

    def _on_tex_scroll(self, event):
        factor = 0.9 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else 1.1
        tw, th = self.tex_img.size
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        min_zoom = 0.02
        max_zoom = max(tw / cw, th / ch)
        self.tex_zoom = float(np.clip(self.tex_zoom * factor, min_zoom, max_zoom))
        self._redraw_texture_panel()

    def _on_tex_pan_press(self, event):
        self._tex_pan_xy = (event.x, event.y)

    def _on_tex_pan_move(self, event):
        if self._tex_pan_xy is None:
            return
        dx = (event.x - self._tex_pan_xy[0]) * self.tex_zoom
        dy = (event.y - self._tex_pan_xy[1]) * self.tex_zoom
        self.tex_center[0] -= dx
        self.tex_center[1] -= dy
        self._tex_pan_xy = (event.x, event.y)
        self._redraw_texture_panel()

    def _on_tex_click(self, event):
        if not self.pick_mode.get() or self.tex_crop_box is None:
            return
        left, top, _right, _bottom = self.tex_crop_box
        tw, th = self.tex_img.size
        px = left + event.x * self.tex_zoom
        py = top + event.y * self.tex_zoom
        u, v = px / (tw - 1), py / (th - 1)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return
        x, z = uv_to_world(u, v, **self.uv_params)
        if not self.terrain_meshes:
            self._status("No terrain mesh loaded to sample height from.")
            return
        y = float(bilinear_height_query(self.terrain_meshes[0], np.array([x]), np.array([z]))[0])
        self.current_points.append([x, y, z])
        self._redraw()
        self._status(f"Added point at ({x:.2f}, {y:.2f}, {z:.2f}) from texture panel.")

    # -- picking ---------------------------------------------------------

    def _on_press(self, event):
        self._press_xy = (event.x, event.y)

    def _on_release(self, event):
        if not self.pick_mode.get() or event.inaxes != self.ax:
            return
        if self._press_xy is None:
            return
        dx = event.x - self._press_xy[0]
        dy = event.y - self._press_xy[1]
        self._press_xy = None
        if (dx * dx + dy * dy) ** 0.5 > CLICK_MOVE_THRESHOLD_PX:
            return  # was an orbit-drag, not a click
        if event.xdata is None or event.ydata is None:
            return
        self._pick_point(event.xdata, event.ydata)

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.9 if event.button == "up" else 1.1
        for get_lim, set_lim in (
            (self.ax.get_xlim3d, self.ax.set_xlim3d),
            (self.ax.get_ylim3d, self.ax.set_ylim3d),
            (self.ax.get_zlim3d, self.ax.set_zlim3d),
        ):
            lo, hi = get_lim()
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            set_lim(mid - half, mid + half)
        self.canvas.draw_idle()

    def _pick_point(self, xdata, ydata):
        if self.pick_tris.shape[0] == 0 and not self.terrain_meshes:
            self._status("No mesh geometry loaded to pick against.")
            return
        origin, direction, t_min = unproject_ray(self.ax, xdata, ydata)

        best_world = None
        best_t = np.inf
        for terrain in self.terrain_meshes:
            result = raymarch_terrain(terrain, origin, direction, t_min)
            if result is not None and result[1] < best_t:
                best_world, best_t = result

        if self.pick_tris.shape[0] > 0:
            hit = ray_triangle_intersect_batch(origin, direction, self.pick_tris, t_min=t_min)
            if hit is not None:
                t_hit = float(np.dot(hit - origin, direction))
                if t_hit < best_t:
                    best_world = np.array([hit[0], hit[2], hit[1]])  # render (x,z,y) -> world (x,y,z)
                    best_t = t_hit

        if best_world is None:
            self._status("No surface hit under click -- try again on the mesh.")
            return
        world = (float(best_world[0]), float(best_world[1]), float(best_world[2]))
        self.current_points.append([world[0], world[1], world[2]])
        self._redraw()
        self._status(f"Added point at ({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f}).")

    # -- object lifecycle --------------------------------------------------

    def undo_point(self):
        if self.current_points:
            self.current_points.pop()
            self._redraw()

    def cancel_current(self):
        if self.current_points and not messagebox.askyesno(
            "Cancel object", f"Discard {len(self.current_points)} marked point(s)?"
        ):
            return
        self.current_points = []
        self._redraw()

    def finish_object(self):
        if len(self.current_points) < 3:
            messagebox.showwarning("Not enough points", "Mark at least 3 points before finishing an object.")
            return

        points = np.array(self.current_points)
        try:
            hull_points, hull_faces = compute_hull(points)
        except QhullError as exc:
            messagebox.showerror("Hull failed", f"Could not compute a convex hull: {exc}")
            return

        dialog = LabelDialog(self.root, default_name=f"object_{self.next_mesh_id}")
        self.root.wait_window(dialog.top)
        if dialog.result is None:
            return  # cancelled, keep points as-is
        category, name = dialog.result
        name = sanitize_name(name)

        mesh_id = self.next_mesh_id
        self.next_mesh_id += 1
        obj_path = f"subobjects/mesh_{mesh_id}_{name}.obj"
        obj = AnnotatedObject(
            mesh_id=mesh_id,
            name=name,
            category=category,
            points=[list(p) for p in self.current_points],
            hull_points=[list(p) for p in hull_points],
            hull_faces=[list(map(int, f)) for f in hull_faces],
            obj_path=obj_path,
        )
        self.store.write_hull_obj(obj)
        self.objects.append(obj)
        self.current_points = []
        self._refresh_listbox()
        self._redraw()
        self._save()
        self._status(f"Saved mesh_id {mesh_id} '{name}' ({category}) -> {obj_path}")

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        obj = self.objects[sel[0]]
        if not messagebox.askyesno("Delete object", f"Delete '{obj.name}' (mesh_id {obj.mesh_id})?"):
            return
        self.store.delete_hull_obj(obj)
        del self.objects[sel[0]]
        self._refresh_listbox()
        self._redraw()
        self._save()
        self._status(f"Deleted mesh_id {obj.mesh_id}.")

    def _save(self):
        self.store.save(self.objects, self.next_mesh_id, self.current_points)

    def _on_close(self):
        self._save()
        self.root.destroy()


class LabelDialog:
    """Modal category + name prompt used when finishing an object."""

    def __init__(self, parent, default_name: str):
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title("Label object")
        self.top.transient(parent)
        self.top.grab_set()

        ttk.Label(self.top, text="Category:").grid(row=0, column=0, sticky="w", padx=8, pady=(10, 2))
        self.category_var = tk.StringVar(value=CATEGORIES[0])
        combo = ttk.Combobox(self.top, textvariable=self.category_var, values=CATEGORIES)
        combo.grid(row=1, column=0, sticky="ew", padx=8)

        ttk.Label(self.top, text="Name:").grid(row=2, column=0, sticky="w", padx=8, pady=(10, 2))
        self.name_var = tk.StringVar(value=default_name)
        name_entry = ttk.Entry(self.top, textvariable=self.name_var)
        name_entry.grid(row=3, column=0, sticky="ew", padx=8)
        name_entry.select_range(0, tk.END)

        btn_frame = ttk.Frame(self.top)
        btn_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=12)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side=tk.RIGHT, padx=4)

        self.top.columnconfigure(0, weight=1)
        combo.focus_set()
        self.top.bind("<Return>", lambda _e: self._ok())
        self.top.bind("<Escape>", lambda _e: self._cancel())

    def _ok(self):
        category = self.category_var.get().strip()
        name = self.name_var.get().strip()
        if not category or not name:
            messagebox.showwarning("Missing info", "Both category and name are required.", parent=self.top)
            return
        self.result = (category, name)
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heightmap", default=str(DEFAULT_HEIGHTMAP))
    ap.add_argument("--texture", default=str(DEFAULT_TEXTURE))
    ap.add_argument("--size-x", type=float, default=50.0)
    ap.add_argument("--size-z", type=float, default=50.0)
    ap.add_argument("--size-y", type=float, default=4.820803273566)
    ap.add_argument("--flip-x", action="store_true", default=False)
    ap.add_argument("--flip-z", action="store_true", default=True)
    ap.add_argument("--no-flip-z", dest="flip_z", action="store_false")
    ap.add_argument("--swap-xz", action="store_true", default=False)
    ap.add_argument("--grid-res", type=int, default=140, help="Terrain display/pick grid resolution per axis")
    ap.add_argument(
        "--obj-glob", action="append", default=[],
        help="Glob for extra OBJ meshes to load into the scene (e.g. rock_envs/run1/rocks/*.obj). Repeatable.",
    )
    ap.add_argument("--out-dir", default="annotations/mesh_segmentation")
    args = ap.parse_args()

    heightmap = Path(args.heightmap)
    if not heightmap.exists():
        raise FileNotFoundError(f"heightmap not found: {heightmap}")
    texture = Path(args.texture) if args.texture else None

    print(f"[1/2] building terrain mesh from {heightmap} (grid_res={args.grid_res}) ...")
    meshes = [
        build_terrain_mesh(
            heightmap,
            texture,
            size_x=args.size_x,
            size_z=args.size_z,
            size_y=args.size_y,
            flip_x=args.flip_x,
            flip_z=args.flip_z,
            swap_xz=args.swap_xz,
            grid_res=args.grid_res,
        )
    ]

    obj_paths = []
    for pattern in args.obj_glob:
        obj_paths.extend(sorted(glob.glob(pattern)))
    for p in obj_paths:
        p = Path(p)
        verts, tris = load_obj_mesh(p)
        if len(verts) == 0:
            continue
        meshes.append(SceneMesh(name=p.stem, vertices=verts, triangles=tris))
    if obj_paths:
        print(f"[2/2] loaded {len(obj_paths)} extra OBJ mesh(es).")

    root = tk.Tk()
    MeshAnnotatorApp(
        root, meshes, Path(args.out_dir),
        texture_path=texture,
        size_x=args.size_x, size_z=args.size_z,
        flip_x=args.flip_x, flip_z=args.flip_z, swap_xz=args.swap_xz,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
