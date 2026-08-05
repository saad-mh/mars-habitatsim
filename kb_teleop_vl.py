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

Also drives vl_direction's third mode, "uncertainty", via UncertaintySession --
this script has no real belief tracker, so a synthetic covariance proxy drifts
up while no synthetic obstacle is close enough to act as a visual anchor and
resets when one is. Crossing the threshold halts keyboard movement and normal
cbf/exploration dispatch, asks Qwen for a sweep description, and waits for
a human-supplied heading (numpad keys 1/2/3/4/6/7/8/9, or R to retry) before
resuming -- press U at any time to force-trigger the halt for testing instead
of waiting on/tuning the organic drift.

Also drives vl_direction's fourth mode, "ghost_mask", whenever the exploration
mode fires: the belief (bearing/distance/uncertainty to the stand-in lost
target) is handed to the VLM as text alongside the obstacle-only frame, and
the VLM answers with JSON pixel coordinates (u, v, radius_px) for where the
translucent ghost circle should be drawn -- placement is the model's call,
not caller-side trigonometry (--no-ghost-mask-vlm reverts to the old
deterministic projection for comparison). Advisory-only, same as exploration/
cbf: the parsed result only ever changes what gets drawn/printed, never an
Action. Because this is a second, throttled VLM call, the on-screen circle
only updates on the existing VL query cadence and holds its last position
between calls (falls back to the deterministic projection whenever no VLM
placement has landed yet, or the model's JSON failed to parse).

Reuses kb_teleop.py's habitat_sim setup (make_sim, terrain height lookup,
boundary clamping) via import rather than duplicating it -- this script only
adds the obstacle field, projection/overlay, and the VL call.

Run inside the "habitat" conda env from the repo root (so vl_direction is
importable and QwenServerManager's subprocess cwd is correct):
    conda activate habitat && python kb_teleop_vl.py [--vl-every-n-frames N]
QwenServerManager spawns vl_direction/qwen_server.py in the "qwen_vlm" conda
env on first use and loads Qwen2.5-VL-3B-Instruct (weights are already
cached locally, so this is mostly GPU-load time); the window opens once that
health check passes.

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
from sam_vla.core.belief_tracking import uncertainty_growth_increment
from sam_vla.core.ghost_mask import (
    draw_ghost_mask,
    project_body_point_to_pixel,
    uncertainty_to_radius_px,
)
from vl_direction import config as vl_dir_config
from vl_direction.client import get_client
from vl_direction.directive_engine import query as vl_query
from vl_direction.qwen_server_manager import QwenServerManager
from vl_direction.schemas import CBFContext, ExplorationContext, GhostMaskContext
from vl_direction.uncertainty_session import UncertaintySession

NUM_OBSTACLES = 6
OBSTACLE_RADIUS_M = 0.5
OBSTACLE_SEED = 7
OBSTACLE_SPAWN_HALF_EXTENT = 18.0  # sampled uniformly in [-extent, extent]^2, comfortably inside kb.BOUNDARY_LIMIT

CBF_DISTANCE_THRESHOLD_M = 3.5
EXPLORATION_TASK_STR = "explore the terrain ahead"

# --- uncertainty halt: synthetic covariance proxy (this script has no real belief
# tracker) -- confidence drifts up while no synthetic obstacle is close enough to act
# as a visual anchor, resets when one is. U forces an immediate halt for testing. ---
UNCERTAINTY_GROUNDING_RANGE_M = CBF_DISTANCE_THRESHOLD_M
DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD = vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD
DEFAULT_UNCERTAINTY_GROWTH_PER_STEP = 0.1
DEFAULT_UNCERTAINTY_GROWTH_RATE = 0.0

# --- ghost mask: translucent green circle at the stand-in "lost goal" (nearest
# obstacle's body-frame point, reused here only because this script has no
# real goal-tracking subsystem) whose radius tracks self.uncertainty_covariance,
# same value that drives the uncertainty halt above -- see sam_vla/core/ghost_mask.py
# and next.md's Phase 1. Scale is picked so the ghost saturates to
# OVERLAY_MAX_PIXEL_RADIUS right as uncertainty reaches the halt threshold. ---
GHOST_ALPHA = 0.45
UNCERTAINTY_HEADING_KEYS = {
    "8": 0.0,
    "9": 45.0,
    "6": 90.0,
    "3": 135.0,
    "2": 180.0,
    "1": -135.0,
    "4": -90.0,
    "7": -45.0,
}

CAMERA_HFOV_DEG = 90.0
_FRAME_H, _FRAME_W = kb.RGBD_RESOLUTION
_FOCAL_PX = (_FRAME_W / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))

OVERLAY_ALPHA = 0.5
OVERLAY_MIN_PIXEL_RADIUS = 3
OVERLAY_MAX_PIXEL_RADIUS = 260

USE_MOCK_CLIENT = False

EPISODE_ID = f"kb-teleop-vl-{int(time.time())}"

# --- VL query cadence: fires on whichever comes first (frame count is
# manually configurable at startup; the seconds trigger is a fixed backstop
# so a stalled frame counter still keeps directives fresh) ---
DEFAULT_VL_QUERY_EVERY_N_FRAMES = 3
VL_QUERY_EVERY_N_SECONDS = 5.0


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


def body_frame_forward_left(px_world, pz_world, rover_x, rover_z, yaw):
    """World-plane point -> body-frame [forward, left], the same convention
    BeliefGoalTracker/ghost_mask use. Ground-plane-only counterpart of
    project_point's forward/right math above (no vertical/pixel component)."""
    dx = px_world - rover_x
    dz = pz_world - rover_z
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    right_x, right_z = math.cos(yaw), -math.sin(yaw)
    forward = dx * forward_x + dz * forward_z
    right = dx * right_x + dz * right_z
    return forward, -right


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
    def __init__(
        self,
        vl_query_every_n_frames=DEFAULT_VL_QUERY_EVERY_N_FRAMES,
        uncertainty_covariance_threshold=DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD,
        uncertainty_growth_per_step=DEFAULT_UNCERTAINTY_GROWTH_PER_STEP,
        uncertainty_growth_rate=DEFAULT_UNCERTAINTY_GROWTH_RATE,
        ghost_mask_use_vlm=True,
    ):
        self.ghost_mask_use_vlm = ghost_mask_use_vlm
        self.last_ghost_mask_payload = None
        self.last_ghost_mask_line = ""

        self.vl_query_every_n_frames = vl_query_every_n_frames
        self.vl_query_every_n_seconds = VL_QUERY_EVERY_N_SECONDS
        self.last_vl_frame_idx = None
        self.last_vl_time = None
        self.vl_query_in_flight = False
        self.vl_lock = threading.Lock()

        self.uncertainty_covariance_threshold = uncertainty_covariance_threshold
        self.uncertainty_growth_per_step = uncertainty_growth_per_step
        self.uncertainty_growth_rate = uncertainty_growth_rate
        self.uncertainty_covariance = 0.0
        self.uncertainty_time_since_seen = 0.0
        self.halted_for_uncertainty = False
        self.uncertainty_request_in_flight = False
        self.last_uncertainty_line = ""
        self.last_annotated_rgb = None
        # Ghost radius saturates to OVERLAY_MAX_PIXEL_RADIUS right as
        # uncertainty_covariance reaches the halt threshold.
        self.ghost_radius_scale = OVERLAY_MAX_PIXEL_RADIUS / max(
            uncertainty_covariance_threshold, 1e-6
        )

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

        self.server_manager = QwenServerManager()
        if USE_MOCK_CLIENT:
            self.client = get_client("mock")
        else:
            print("[VLTeleopApp] starting Qwen server")
            self.server_manager.start()
            self.client = get_client("qwen")

        print(
            f"[VLTeleopApp] VL query cadence: every {self.vl_query_every_n_frames} frames "
            f"or every {self.vl_query_every_n_seconds:.1f}s, whichever comes first"
        )

        self.uncertainty_session = UncertaintySession(
            episode_id=EPISODE_ID,
            covariance_threshold=self.uncertainty_covariance_threshold,
            covariance_value=self.uncertainty_covariance,
            client=self.client,
        )
        print(
            f"[VLTeleopApp] uncertainty halt: growth={self.uncertainty_growth_per_step}/step, "
            f"threshold={self.uncertainty_covariance_threshold} (press U to force-trigger)"
        )

        self.frame_idx = 0
        self.closed = False
        self.last_vl_line = ""

        self.root = tk.Tk()
        self.root.title("kb teleop")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text=(
                "W/S move | A/D turn | Q/E height | U force-uncertainty-halt | X quit  "
            ),
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
                forward, left = body_frame_forward_left(
                    ox, oz, self.x, self.z, self.yaw
                )
                nearest_any = {
                    "edge_distance": edge_distance,
                    "forward": forward,
                    "left": left,
                }

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

    def _dispatch_vl_query(self, annotated_rgb, ghost_frame, nearest_any, nearest_visible):
        """Builds the VL request and hands the actual model call to a
        background thread so render()/movement never blocks on inference
        latency -- the sim keeps stepping on whatever self.last_vl_line
        already holds (possibly still "", if no result has landed yet).
        ghost_frame is the obstacle-only frame (no ghost mask baked in yet)
        -- handed separately to the ghost_mask query so the model isn't
        shown a circle it's being asked to place."""
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
            args=(
                mode,
                annotated_rgb,
                ghost_frame,
                context,
                fallback_note,
                frame_idx_snapshot,
                nearest_any,
            ),
            daemon=True,
        )
        thread.start()

    def _vl_worker(
        self,
        mode,
        annotated_rgb,
        ghost_frame,
        context,
        fallback_note,
        frame_idx_snapshot,
        nearest_any,
    ):
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

        ghost_payload, ghost_line = self._maybe_query_ghost_mask(
            mode, ghost_frame, nearest_any, frame_idx_snapshot
        )
        if ghost_line:
            print(ghost_line)

        with self.vl_lock:
            self.vl_query_in_flight = False

        if not self.closed:
            try:
                self.root.after(0, self._apply_vl_line, line, ghost_payload, ghost_line)
            except Exception:
                pass  # window may have been torn down between the check and this call

    def _maybe_query_ghost_mask(self, mode, ghost_frame, nearest_any, frame_idx_snapshot):
        """Second VLM call, gated on --ghost-mask-vlm and only while
        exploring (a lost-goal cue has no meaning mid-CBF-avoidance -- see
        next.md's cbf/exploration binary). Belief is handed to the model as
        text (GhostMaskContext); the model answers with pixel placement
        instead of sam_vla.core.ghost_mask's trigonometry deciding it.
        Runs on the same background thread as the direction call above, so
        this only ever blocks the (already async) worker, never render()."""
        if not self.ghost_mask_use_vlm or mode != "exploration" or nearest_any is None:
            return None, None

        bearing_deg = math.degrees(
            math.atan2(nearest_any["left"], nearest_any["forward"])
        )
        distance_m = math.hypot(nearest_any["forward"], nearest_any["left"])
        ghost_context = GhostMaskContext(
            bearing_deg=bearing_deg,
            distance_m=distance_m,
            uncertainty=self.uncertainty_covariance,
            frame_wh=(_FRAME_W, _FRAME_H),
            min_radius_px=float(OVERLAY_MIN_PIXEL_RADIUS),
            max_radius_px=float(OVERLAY_MAX_PIXEL_RADIUS),
        )
        try:
            ghost_result = vl_query(
                "ghost_mask", [ghost_frame], ghost_context, EPISODE_ID, client=self.client
            )
        except Exception as e:
            return None, f"[{frame_idx_snapshot:05d}] ghost_mask -> ERROR -> {e}"

        if not ghost_result.parse_ok:
            return None, (
                f"[{frame_idx_snapshot:05d}] ghost_mask -> PARSE FAILED, "
                f"keeping last placement -> {ghost_result.raw_response!r}"
            )

        payload = ghost_result.ghost_mask_payload
        line = (
            f"[{frame_idx_snapshot:05d}] ghost_mask -> "
            f"u={payload.u:.0f} v={payload.v:.0f} r={payload.radius_px:.0f} -> "
            f"{ghost_result.latency_ms:.1f}ms -> {ghost_result.raw_response!r}"
        )
        return payload, line

    def _apply_vl_line(self, line, ghost_payload=None, ghost_line=None):
        """Runs on the main/Tkinter thread via root.after. Only updates the
        cached line(s); does not force a redraw so a slow VL result never
        stalls movement -- it just appears baked into the next
        keypress-triggered render(). A failed ghost-mask parse leaves
        self.last_ghost_mask_payload untouched (keeps the last good VLM
        placement rather than snapping back to the deterministic fallback)."""
        self.last_vl_line = line
        if ghost_payload is not None:
            self.last_ghost_mask_payload = ghost_payload
        if ghost_line is not None:
            self.last_ghost_mask_line = ghost_line

    def _update_uncertainty_covariance(self, nearest_any):
        """Synthetic proxy for a real covariance signal -- this script has no
        belief tracker to source one from. Drifts up while no obstacle is
        close enough to act as a visual anchor, resets when one is, using the
        same accelerating-drift formula as BeliefGoalTracker.propagate()
        (sam_vla.core.belief_tracking.uncertainty_growth_increment) so this
        demo and the real Phase 2 wiring exercise one formula, not two
        divergent ones. Frozen while already halted so it can't re-trigger
        mid-halt."""
        if self.halted_for_uncertainty:
            return
        grounded = (
            nearest_any is not None
            and nearest_any["edge_distance"] <= UNCERTAINTY_GROUNDING_RANGE_M
        )
        if grounded:
            self.uncertainty_covariance = 0.0
            self.uncertainty_time_since_seen = 0.0
        else:
            self.uncertainty_time_since_seen += 1.0  # one render() call per step
            self.uncertainty_covariance += uncertainty_growth_increment(
                self.uncertainty_growth_per_step,
                self.uncertainty_growth_rate,
                self.uncertainty_time_since_seen,
            )

    def _dispatch_uncertainty_request(self, retry):
        """Enters (or re-enters, on retry) the uncertainty halt: dispatches
        the sweep-description call to a background thread, same pattern as
        _dispatch_vl_query, so it doesn't block rendering/input handling."""
        self.halted_for_uncertainty = True
        self.uncertainty_request_in_flight = True
        frame = self.last_annotated_rgb
        target = (
            self.uncertainty_session.retry
            if retry
            else self.uncertainty_session.request_human_heading
        )

        thread = threading.Thread(
            target=self._uncertainty_worker, args=(target, frame), daemon=True
        )
        thread.start()

    def _uncertainty_worker(self, session_method, frame):
        """Runs on a background thread -- calls UncertaintySession's request
        or retry method (a real Qwen sweep-description call). Never
        touches Tkinter directly; schedules the UI update via root.after."""
        try:
            result = session_method(frame)
        except Exception as e:
            line = f"[uncertainty] ERROR -> {e}"
        else:
            payload = result.uncertainty_payload
            line = (
                f"[uncertainty] attempt={payload.attempt} -> {payload.status.value} -> "
                f"sweep: {result.raw_response!r} -- "
                f"press 1/2/3/4/6/7/8/9 for heading, R to retry"
            )

        print(line)

        with self.vl_lock:
            self.uncertainty_request_in_flight = False

        if not self.closed:
            try:
                self.root.after(0, self._apply_uncertainty_line, line)
            except Exception:
                pass  # window may have been torn down between the check and this call

    def _apply_uncertainty_line(self, line):
        """Runs on the main/Tkinter thread via root.after. Re-renders (unlike
        _apply_vl_line) so the halted-state prompt on screen picks up the
        freshly landed sweep description right away."""
        self.last_uncertainty_line = line
        self.render()

    def _handle_halted_key(self, key):
        """Only reachable while self.halted_for_uncertainty is True -- this
        is the actual movement halt, called from on_key instead of the usual
        WASD handling. Ignores everything but retry/heading-submit while a
        sweep-description call is in flight, to avoid double-dispatching."""
        if self.uncertainty_request_in_flight:
            return

        if key == "r":
            self._dispatch_uncertainty_request(retry=True)
            return

        angle = UNCERTAINTY_HEADING_KEYS.get(key)
        if angle is None:
            return

        result = self.uncertainty_session.submit_heading(angle_deg=angle)
        payload = result.uncertainty_payload
        line = (
            f"[uncertainty] submit_heading({angle:.0f}deg) -> {payload.status.value} -> "
            f"max_units={payload.max_units:.1f}"
        )
        print(line)
        self.last_uncertainty_line = line
        self.uncertainty_covariance = 0.0
        self.uncertainty_time_since_seen = 0.0
        self.halted_for_uncertainty = False
        self.render()

    def render(self):
        obs = self.sim.get_sensor_observations()
        self.latest_obs = obs

        rgb, _, depth_rgb = kb.rgb_depth_from_obs(obs)

        circles, nearest_any, nearest_visible = self._project_obstacles()
        annotated_rgb = overlay_obstacles(rgb, circles)

        # Ghost mask: translucent green circle at a stand-in "lost goal" point
        # (nearest obstacle's body-frame position -- this script has no real
        # goal to track), radius driven by self.uncertainty_covariance, the
        # same synthetic uncertainty signal that drives the halt above. Purely
        # visual/advisory: only ever touches this VLM/display copy of the
        # frame, never anything fed to a policy (there is none here).
        if nearest_any is not None:
            ghost_px = project_body_point_to_pixel(
                nearest_any["forward"],
                nearest_any["left"],
                CAMERA_HFOV_DEG,
                _FRAME_H,
                _FRAME_W,
            )
            if ghost_px is not None:
                ghost_u, ghost_v = ghost_px
                ghost_radius = uncertainty_to_radius_px(
                    self.uncertainty_covariance,
                    OVERLAY_MIN_PIXEL_RADIUS,
                    OVERLAY_MAX_PIXEL_RADIUS,
                    self.ghost_radius_scale,
                )
                annotated_rgb = draw_ghost_mask(
                    annotated_rgb, ghost_u, ghost_v, ghost_radius, alpha=GHOST_ALPHA
                )

        self.last_annotated_rgb = annotated_rgb

        self._update_uncertainty_covariance(nearest_any)
        if (
            not self.halted_for_uncertainty
            and not self.uncertainty_request_in_flight
            and self.uncertainty_covariance >= self.uncertainty_covariance_threshold
        ):
            self._dispatch_uncertainty_request(retry=False)

        if not self.halted_for_uncertainty and self._should_query_vl():
            self._dispatch_vl_query(annotated_rgb, nearest_any, nearest_visible)
        self.frame_idx += 1

        if kb.SHOW_DEPTH_BESIDE_RGB:
            img_arr = np.hstack([annotated_rgb, depth_rgb])
        else:
            img_arr = annotated_rgb

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        status = f"x={self.x:.2f} y={self.y:.2f} z={self.z:.2f} yaw={np.rad2deg(self.yaw):.1f} clearance={self.clearance:.2f} cov={self.uncertainty_covariance:.2f}"
        if self.halted_for_uncertainty:
            status += "  [HALTED: awaiting heading]"

        draw.rectangle([0, 0, img.width, 75], fill=(0, 0, 0))
        draw.text((10, 8), status, fill=(255, 255, 255))
        draw.text((10, 30), self.last_vl_line, fill=(255, 255, 0))
        draw.text((10, 52), self.last_uncertainty_line, fill=(0, 255, 255))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def on_key(self, event):
        key = event.keysym.lower()

        if key == "x" or key == "escape":
            self.close()
            return

        if self.halted_for_uncertainty:
            self._handle_halted_key(key)
            return

        if key == "u":
            self.uncertainty_covariance = self.uncertainty_covariance_threshold
            self.render()
            return

        old_x = self.x
        old_z = self.z

        if key == "w":
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
        "--qwery-n",
        type=int,
        default=DEFAULT_VL_QUERY_EVERY_N_FRAMES,
        help=f"query VL every N frames (default {DEFAULT_VL_QUERY_EVERY_N_FRAMES}); "
        f"also fires every {VL_QUERY_EVERY_N_SECONDS:.1f}s regardless, whichever comes first",
    )
    parser.add_argument(
        "--cov-thresh",
        type=float,
        default=DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD,
        help="synthetic covariance value at which the uncertainty halt fires "
        f"(default {DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--cov-growth",
        type=float,
        default=DEFAULT_UNCERTAINTY_GROWTH_PER_STEP,
        help="base per-frame covariance growth while no synthetic obstacle is nearby "
        f"(default {DEFAULT_UNCERTAINTY_GROWTH_PER_STEP}); press U in-app to "
        "force-trigger the halt instead of waiting on this",
    )
    parser.add_argument(
        "--cov-growth-rate",
        type=float,
        default=DEFAULT_UNCERTAINTY_GROWTH_RATE,
        help="accelerating-drift factor: growth speeds up the longer the "
        f"target has been ungrounded (default {DEFAULT_UNCERTAINTY_GROWTH_RATE}, "
        "i.e. flat per-frame growth unless overridden); also drives the "
        "ghost-mask radius alongside --cov-growth",
    )
    args = parser.parse_args()
    if args.qwery_n < 1:
        parser.error("--qwery-n must be >= 1")
    return args


if __name__ == "__main__":
    args = _parse_args()
    app = VLTeleopApp(
        vl_query_every_n_frames=args.qwery_n,
        uncertainty_covariance_threshold=args.cov_thresh,
        uncertainty_growth_per_step=args.cov_growth,
        uncertainty_growth_rate=args.cov_growth_rate,
    )
    app.run()
