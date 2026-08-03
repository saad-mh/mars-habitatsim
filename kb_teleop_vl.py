"""
Keyboard teleop for marsyard2022 with vl_direction wired in live: every
rendered frame is checked against a synthetic obstacle field (random
ground-plane points, each a 0.5m-radius "rock" with no real geometry --
there's no obstacle detector in this repo yet, so this stands in for one),
overlaid in red directly on the RGB panel, and handed to
vl_direction.directive_engine.query() so the console prints exactly what was
asked for: prompt mode, raw model output / parsed direction, and per-call
latency, once per rendered frame.

Mode selection: "cbf" (go-around bbox prompt) when the nearest obstacle's
edge is within CBF_DISTANCE_THRESHOLD_M and it actually projects into the
current view; "exploration" (four-way prior) otherwise, including the
fallback case where the nearest obstacle is behind the rover and has no
pixel bbox to hand to CBFContext.

Reuses kb_teleop.py's habitat_sim setup (make_sim, terrain height lookup,
boundary clamping) via import rather than duplicating it -- this script only
adds the obstacle field, projection/overlay, and the VL call.

Run inside the "habitat" conda env from the repo root (so vl_direction is
importable and InternVLServerManager's subprocess cwd is correct):
    conda activate habitat && python kb_teleop_vl.py [--vl-every-n-frames N]
InternVLServerManager spawns internvl_server.py in the "vl" conda env on
first use and loads InternVL3-8B (weights are already cached locally, so
this is mostly GPU-load time); the window opens once that health check
passes.

VL queries are dispatched on a background thread and gated to run every
--vl-every-n-frames frames or every VL_QUERY_EVERY_N_SECONDS seconds,
whichever comes first (see _should_query_vl). Movement/rendering never
blocks on the VL call -- the rover keeps stepping and the frame keeps
rendering on whatever the last-known VL directive was (or with none at all,
before the first result lands), the same way a real rover has to keep
running even when its perception stack hasn't produced an output yet.
"""

import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageDraw

import tkinter as tk

import quaternion

import kb_teleop as kb
from vl_direction.client import get_client
from vl_direction.directive_engine import query as vl_query
from vl_direction.internvl_server_manager import InternVLServerManager
from vl_direction.schemas import CBFContext, ExplorationContext

# --- synthetic obstacle field (stand-in for a real obstacle detector) ---
NUM_OBSTACLES = 6
OBSTACLE_RADIUS_M = 0.5
OBSTACLE_SEED = 7
OBSTACLE_SPAWN_HALF_EXTENT = 18.0  # sampled uniformly in [-extent, extent]^2, comfortably inside kb.BOUNDARY_LIMIT

# --- mode selection ---
CBF_DISTANCE_THRESHOLD_M = 3.5  # rover-to-nearest-obstacle-edge distance below which we ask for a go-around side
EXPLORATION_TASK_STR = "explore the terrain ahead"

# --- camera projection (must match kb_teleop.make_sensor's hfov, which
# habitat_sim defaults to 90deg -- unchanged here, see make_sensor) ---
CAMERA_HFOV_DEG = 90.0
_FRAME_H, _FRAME_W = kb.RGBD_RESOLUTION
_FOCAL_PX = (_FRAME_W / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))

# --- overlay ---
OVERLAY_ALPHA = 0.5
OVERLAY_MIN_PIXEL_RADIUS = 3
OVERLAY_MAX_PIXEL_RADIUS = 260

# --- backend ---
USE_MOCK_CLIENT = False  # flip to True to smoke-test this script with no GPU/model

EPISODE_ID = f"kb-teleop-vl-{int(time.time())}"

# --- VL query cadence: fires on whichever comes first (frame count is
# manually configurable at startup; the seconds trigger is a fixed backstop
# so a stalled frame counter still keeps directives fresh) ---
DEFAULT_VL_QUERY_EVERY_N_FRAMES = 3
VL_QUERY_EVERY_N_SECONDS = 3.0


def make_obstacles(rng, count, half_extent):
    xs = rng.uniform(-half_extent, half_extent, size=count)
    zs = rng.uniform(-half_extent, half_extent, size=count)
    return list(zip(xs.tolist(), zs.tolist()))


def project_point(px_world, py_world, pz_world, rover_x, rover_y, rover_z, yaw):
    """World point -> (pixel_x, pixel_y, depth), or None if behind the camera.
    Matches kb_teleop's movement convention: forward at yaw=0 is -Z, right is
    +X (habitat_sim's camera looks down -Z in its local frame, +Y up)."""
    dx = px_world - rover_x
    dy = py_world - rover_y
    dz = pz_world - rover_z

    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    right_x, right_z = math.cos(yaw), -math.sin(yaw)

    depth = dx * forward_x + dz * forward_z
    if depth <= 1e-3:
        return None

    local_x = dx * right_x + dz * right_z
    local_y = dy

    pixel_x = _FRAME_W / 2.0 + _FOCAL_PX * local_x / depth
    pixel_y = _FRAME_H / 2.0 - _FOCAL_PX * local_y / depth
    return pixel_x, pixel_y, depth


def overlay_obstacles(rgb, projected_circles):
    """Alpha-blends solid red circles onto rgb (uint8 HWC) at each
    (pixel_x, pixel_y, pixel_radius) in projected_circles."""
    if not projected_circles:
        return rgb.copy()

    annotated = rgb.astype(np.float32)
    yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for pixel_x, pixel_y, pixel_radius in projected_circles:
        mask |= (xx - pixel_x) ** 2 + (yy - pixel_y) ** 2 <= pixel_radius**2

    red = np.array([255.0, 0.0, 0.0])
    annotated[mask] = annotated[mask] * (1.0 - OVERLAY_ALPHA) + red * OVERLAY_ALPHA
    return annotated.astype(np.uint8)


class VLTeleopApp:
    def __init__(self, vl_query_every_n_frames=DEFAULT_VL_QUERY_EVERY_N_FRAMES):
        self.vl_query_every_n_frames = vl_query_every_n_frames
        self.vl_query_every_n_seconds = VL_QUERY_EVERY_N_SECONDS
        self.last_vl_frame_idx = None
        self.last_vl_time = None
        self.vl_query_in_flight = False
        self.vl_lock = threading.Lock()

        self.sim = kb.make_sim()
        self.agent = self.sim.initialize_agent(0)

        self.x = kb.START_X
        self.z = kb.START_Z
        self.yaw = np.deg2rad(kb.START_YAW_DEG)
        self.clearance = kb.INITIAL_CLEARANCE

        rng = np.random.default_rng(OBSTACLE_SEED)
        obstacle_xz = make_obstacles(rng, NUM_OBSTACLES, OBSTACLE_SPAWN_HALF_EXTENT)
        self.obstacles = [
            (ox, kb.terrain_height_at(ox, oz) + OBSTACLE_RADIUS_M, oz)
            for ox, oz in obstacle_xz
        ]

        print(
            f"[VLTeleopApp] {NUM_OBSTACLES} synthetic obstacles (r={OBSTACLE_RADIUS_M}m): "
            + ", ".join(f"({o[0]:.1f},{o[2]:.1f})" for o in self.obstacles)
        )

        self.server_manager = InternVLServerManager()
        if USE_MOCK_CLIENT:
            self.client = get_client("mock")
        else:
            print(
                "[VLTeleopApp] starting InternVL server (loads InternVL3-8B in the 'vl' conda env)"
            )
            self.server_manager.start()
            self.client = get_client()

        print(
            f"[VLTeleopApp] VL query cadence: every {self.vl_query_every_n_frames} frames "
            f"or every {self.vl_query_every_n_seconds:.1f}s, whichever comes first"
        )

        self.frame_idx = 0
        self.closed = False
        self.last_vl_line = ""

        self.root = tk.Tk()
        self.root.title("Kb Teleop + InternVL")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text="W/S move | A/D turn | Q/E height | X quit",
            font=("Arial", 12),
        )
        self.info_label.pack()

        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.set_agent_pose()
        self.render()

    def set_agent_pose(self):
        self.terrain_y = kb.terrain_height_at(self.x, self.z)
        self.y = self.terrain_y + self.clearance

        state = self.agent.get_state()
        state.position = np.array([self.x, self.y, self.z], dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector([0.0, self.yaw, 0.0])
        self.agent.set_state(state)

    def _project_obstacles(self):
        """Returns (circles_for_overlay, nearest_any, nearest_visible).
        edge_distance is ground-plane distance from the rover to the
        obstacle surface (negative if the rover is already inside it).
        nearest_any drives mode selection; nearest_visible (a subset that
        actually projects into frame) is what CBFContext gets a bbox from."""
        circles = []
        nearest_any = None
        nearest_visible = None

        for ox, oy, oz in self.obstacles:
            edge_distance = math.hypot(ox - self.x, oz - self.z) - OBSTACLE_RADIUS_M
            if nearest_any is None or edge_distance < nearest_any["edge_distance"]:
                nearest_any = {"edge_distance": edge_distance}

            projected = project_point(ox, oy, oz, self.x, self.y, self.z, self.yaw)
            if projected is None:
                continue
            pixel_x, pixel_y, depth = projected
            pixel_radius = float(
                np.clip(
                    _FOCAL_PX * OBSTACLE_RADIUS_M / depth,
                    OVERLAY_MIN_PIXEL_RADIUS,
                    OVERLAY_MAX_PIXEL_RADIUS,
                )
            )

            x1, y1 = pixel_x - pixel_radius, pixel_y - pixel_radius
            x2, y2 = pixel_x + pixel_radius, pixel_y + pixel_radius
            x1c, y1c = max(0.0, x1), max(0.0, y1)
            x2c, y2c = min(float(_FRAME_W), x2), min(float(_FRAME_H), y2)
            if x2c <= x1c or y2c <= y1c:
                continue  # projects fully outside the frame

            circles.append((pixel_x, pixel_y, pixel_radius))
            bbox = (int(x1c), int(y1c), int(math.ceil(x2c)), int(math.ceil(y2c)))
            if (
                nearest_visible is None
                or edge_distance < nearest_visible["edge_distance"]
            ):
                nearest_visible = {"edge_distance": edge_distance, "bbox": bbox}

        return circles, nearest_any, nearest_visible

    def _should_query_vl(self):
        """Gate on whichever cadence trigger fires first: every N frames
        (manually configured, default 3) or every VL_QUERY_EVERY_N_SECONDS
        wall-clock seconds since the last dispatched call. Always true on
        the first frame. Skips while a previous call is still in flight so
        slow inference can't pile up overlapping requests -- the render
        loop itself is never gated on this, only whether a *new* dispatch
        happens."""
        if self.vl_query_in_flight:
            return False
        if self.last_vl_frame_idx is None:
            return True
        if self.frame_idx - self.last_vl_frame_idx >= self.vl_query_every_n_frames:
            return True
        if time.time() - self.last_vl_time >= self.vl_query_every_n_seconds:
            return True
        return False

    def _dispatch_vl_query(self, annotated_rgb, nearest_any, nearest_visible):
        """Builds the VL request and hands the actual model call to a
        background thread so render()/movement never blocks on inference
        latency -- the sim keeps stepping on whatever self.last_vl_line
        already holds (possibly still "", if no result has landed yet)."""
        mode = (
            "cbf"
            if nearest_any is not None
            and nearest_any["edge_distance"] <= CBF_DISTANCE_THRESHOLD_M
            else "exploration"
        )
        fallback_note = ""

        if mode == "cbf" and nearest_visible is not None:
            context = CBFContext(
                bbox_xyxy=nearest_visible["bbox"], frame_wh=(_FRAME_W, _FRAME_H)
            )
        else:
            if mode == "cbf":
                fallback_note = (
                    " (nearest obstacle not in view, fell back to exploration)"
                )
            mode = "exploration"
            hint = (
                f"nearest obstacle is {nearest_any['edge_distance']:.1f}m away"
                if nearest_any is not None
                else None
            )
            context = ExplorationContext(task_str=EXPLORATION_TASK_STR, vague_hint=hint)

        frame_idx_snapshot = self.frame_idx
        self.last_vl_frame_idx = self.frame_idx
        self.last_vl_time = time.time()
        self.vl_query_in_flight = True

        thread = threading.Thread(
            target=self._vl_worker,
            args=(mode, annotated_rgb, context, fallback_note, frame_idx_snapshot),
            daemon=True,
        )
        thread.start()

    def _vl_worker(self, mode, annotated_rgb, context, fallback_note, frame_idx_snapshot):
        """Runs on a background thread -- the actual (slow) model call.
        Never touches Tkinter directly; schedules the UI update back onto
        the main loop via root.after so this stays thread-safe."""
        try:
            result = vl_query(
                mode, [annotated_rgb], context, EPISODE_ID, client=self.client
            )
        except Exception as e:
            line = f"[{frame_idx_snapshot:05d}] {mode} -> ERROR -> n/a -> {e}"
        else:
            direction_str = (
                result.direction.value if result.direction is not None else "NONE"
            )
            line = (
                f"[{frame_idx_snapshot:05d}] {mode} -> {direction_str} -> "
                f"{result.latency_ms:.1f}ms -> {result.raw_response!r}{fallback_note}"
            )

        print(line)

        with self.vl_lock:
            self.vl_query_in_flight = False

        if not self.closed:
            try:
                self.root.after(0, self._apply_vl_line, line)
            except Exception:
                pass  # window may have been torn down between the check and this call

    def _apply_vl_line(self, line):
        """Runs on the main/Tkinter thread via root.after. Only updates the
        cached line; does not force a redraw so a slow VL result never
        stalls movement -- it just appears baked into the next
        keypress-triggered render()."""
        self.last_vl_line = line

    def render(self):
        obs = self.sim.get_sensor_observations()
        self.latest_obs = obs

        rgb, _, depth_rgb = kb.rgb_depth_from_obs(obs)

        circles, nearest_any, nearest_visible = self._project_obstacles()
        annotated_rgb = overlay_obstacles(rgb, circles)

        if self._should_query_vl():
            self._dispatch_vl_query(annotated_rgb, nearest_any, nearest_visible)
        self.frame_idx += 1

        if kb.SHOW_DEPTH_BESIDE_RGB:
            img_arr = np.hstack([annotated_rgb, depth_rgb])
        else:
            img_arr = annotated_rgb

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        status = f"x={self.x:.2f} y={self.y:.2f} z={self.z:.2f} yaw={np.rad2deg(self.yaw):.1f} clearance={self.clearance:.2f}"

        draw.rectangle([0, 0, img.width, 55], fill=(0, 0, 0))
        draw.text((10, 8), status, fill=(255, 255, 255))
        draw.text((10, 30), self.last_vl_line, fill=(255, 255, 0))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def on_key(self, event):
        key = event.keysym.lower()

        old_x = self.x
        old_z = self.z

        if key == "x" or key == "escape":
            self.close()
            return
        elif key == "w":
            self.x += -np.sin(self.yaw) * kb.MOVE_STEP
            self.z += -np.cos(self.yaw) * kb.MOVE_STEP
        elif key == "s":
            self.x -= -np.sin(self.yaw) * kb.MOVE_STEP
            self.z -= -np.cos(self.yaw) * kb.MOVE_STEP
        elif key == "a":
            self.yaw += kb.TURN_STEP
        elif key == "d":
            self.yaw -= kb.TURN_STEP
        elif key == "q":
            self.clearance = max(kb.MIN_CLEARANCE, self.clearance - kb.CLEARANCE_STEP)
        elif key == "e":
            self.clearance = min(kb.MAX_CLEARANCE, self.clearance + kb.CLEARANCE_STEP)
        else:
            return

        self.x, self.z = kb.apply_boundary(self.x, self.z, old_x, old_z)

        self.set_agent_pose()
        self.render()

    def close(self):
        if self.closed:
            return
        self.closed = True

        try:
            self.sim.close()
        except Exception:
            pass
        try:
            self.server_manager.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

        print("Done.")

    def run(self):
        self.root.mainloop()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vl-every-n-frames",
        type=int,
        default=DEFAULT_VL_QUERY_EVERY_N_FRAMES,
        help=f"query VL every N frames (default {DEFAULT_VL_QUERY_EVERY_N_FRAMES}); "
        f"also fires every {VL_QUERY_EVERY_N_SECONDS:.1f}s regardless, whichever comes first",
    )
    args = parser.parse_args()
    if args.vl_every_n_frames < 1:
        parser.error("--vl-every-n-frames must be >= 1")
    return args


if __name__ == "__main__":
    args = _parse_args()
    app = VLTeleopApp(vl_query_every_n_frames=args.vl_every_n_frames)
    app.run()
