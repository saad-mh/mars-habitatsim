"""Threaded, in-process controller wrapping MarsHabitatEnv + the real,
published NavDP model (sam_vla.policy.navdp_upstream_policy.NavdpUpstreamPolicy)
-- not this repo's own in-house S2DiT/VL3-DP model
(sam_vla.policy.navdp_policy.NavdpPolicy) -- into a single control loop a GUI
can drive interactively.

Runs the sim + policy + belief/CBF stepping on one dedicated background
thread (habitat-sim's render context is not meant to be touched from more
than one thread), exposing a small thread-safe command API
(set_manual/random_goal/go_home/request_resolve/request_reset/stop_driving)
plus snapshot() for a GUI to poll at its own refresh rate. No zenoh, no
second conda-env process for the sim itself: MarsHabitatEnv runs in-process
here (the `habitat` conda env already has everything this module imports --
see CLAUDE.md's env table), the real NavDP model and the Qwen VLM used for
one-shot goal resolution are the only subprocesses, spawned automatically by
NavdpUpstreamPolicy/QwenServerManager exactly as sam_vla.run_navdp_rollout
already does.

Driving modes:
  idle    -- zero action, no policy call.
  manual  -- direct (v_fwd, yaw_rate) from the caller, bypassing the policy
             and CBF entirely (a human takes full responsibility, same
             convention every manual-drive tool in this repo uses).
  point   -- drive to a world-frame (x, z) point (random-ahead / go-home /
             click-to-goal / any caller-supplied point) with no detector in
             the loop: goal_math.body_frame_goal recovers the ground-truth
             body-frame point every tick, fed straight into BeliefGoalTracker
             via observe_body_point (see its docstring -- built for exactly
             this, ground-truth callers with no mask/depth to source a
             sighting from). A click-to-goal point is anchored once by
             request_pixel_goal's depth backprojection (goal_geometry.
             bbox_to_world on a small patch around the clicked pixel), not
             re-derived from anything that could drift -- and kept visually
             marked in the live camera view every frame by reprojecting it
             back to pixel coords (goal_geometry.project_world_to_pixel)
             rather than registering a scene object, since dynamically
             registered objects render zero pixels on this machine (see
             CLAUDE.md's dynamic-object-render-bug note).
  resolve -- one-shot first_frame_resolver.resolve_verbose() on the current
             frame (SAM2 detections + Qwen VLM salience pick, no per-preset
             text targeting -- qwen_client.select_goal has no such
             parameter, see next.md), then track the resulting goal/obstacle
             masks every tick exactly as sam_vla.run_navdp_rollout's
             single-goal path does (BeliefGoalTracker.observe(mask, depth),
             holding position rather than driving on a fabricated default
             goal point if the mask has never been sighted -- next.md's
             Integration-project Phase 5 fix, reproduced here since this is
             an independent implementation of the same loop, not an import
             of it).

CBF cone-mode obstacle avoidance (sam_vla.safety.cbf_avoidance) wraps every
mode except manual, same as sam_vla.run_navdp_rollout.
"""

from __future__ import annotations

import copy
import math
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
    project_world_to_pixel,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, GoalSpec, Pose
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.terrain import SIZE_X, SIZE_Z
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.perception.semantic_overlay import (
    draw_point_marker,
    overlay_semantic_masks,
)
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy
from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance
from sam_vla.vlm.qwen_server_manager import QwenServerManager

from nav.goal_math import body_frame_goal, random_ahead_point

MODE_IDLE = "idle"
MODE_POINT = "point"
MODE_RESOLVE = "resolve"
MODE_MANUAL = "manual"
# Holding state between a completed SAM2/Qwen resolve and the user accepting
# it -- entered by _do_resolve instead of MODE_RESOLVE, left via
# request_confirm_segmentation (-> MODE_RESOLVE, starts driving),
# request_rerun_segmentation (stays here, re-invokes _do_resolve), or
# request_pick_manually (-> MODE_IDLE, falls through to click-to-goal).
MODE_REVIEW_SEGMENTATION = "review_segmentation"

# Ground-truth distance at which a point goal counts as reached
POINT_GOAL_REACHED_M = 0.7

# Default annotations dir for seg_overlay="mesh" -- the dataset
# sam_lora_runs/exp10 (the default seg_backend="lora" checkpoint) was
# trained against. Repo-relative so this holds regardless of caller cwd,
# same convention nav/gui.py's REPO_ROOT uses.
DEFAULT_ANNOTATIONS_DIR = str(
    Path(__file__).resolve().parent.parent / "annotations" / "mesh_tight_bound2"
)


@dataclass
class DisplayState:
    """Snapshot of controller state for the GUI thread to read. Never mutated
    in place by the GUI -- RoverController.snapshot() hands out a shallow
    copy each call."""

    vis_rgb: Optional[np.ndarray] = None
    pose: Optional[Pose] = None
    mode: str = MODE_IDLE
    status_text: str = "starting"
    action: Action = field(default_factory=lambda: Action(0.0, 0.0, 0.0))
    distance: Optional[float] = None
    goal_world: Optional[tuple] = None
    belief_g: Optional[tuple] = None  # (forward, left), body frame
    trajectory: Optional[np.ndarray] = None  # chosen NavDP waypoints, body frame
    obstacle_point: Optional[tuple] = None  # nearest CBF obstacle, body frame
    cbf_info: dict = field(default_factory=dict)
    step: int = 0
    frame_count: int = 0
    goal_reached: bool = False
    error_text: str = ""
    click_status: str = ""


class RoverController:
    def __init__(
        self,
        *,
        scene_path: str,
        heightmap_path: str,
        navdp_upstream_ckpt: str,
        navdp_upstream_root: Optional[str] = None,
        navdp_root: Optional[str] = None,
        rock_field_path: Optional[str] = None,
        start_x: float = 0.0,
        start_z: float = 8.0,
        start_yaw_deg: float = 0.0,
        dt: float = 0.1,
        hz: float = 10.0,
        cbf_enabled: bool = True,
        cbf_d_safe: float = 0.75,
        cbf_gamma: float = 0.3,
        cbf_deadzone: float = 0.6,
        cbf_orbit_kr: float = 0.8,
        cbf_orbit_hyst: float = 0.4,
        cbf_pursuit_kp: float = 1.8,
        cbf_goaround_forward: float = 0.5,
        cbf_hard_gate: bool = True,
        robot_radius: float = 0.25,
        safety_margin: float = 0.15,
        obstacle_radius: float = 0.25,
        obj_mask_radius: float = 0.5,
        belief_goal_range: float = 8.0,
        lost_goal_min_px: int = 10,
        max_forward_speed: float = 0.6,
        max_yaw_rate: float = 0.6,
        navdp_upstream_port: Optional[int] = None,
        navdp_upstream_lookahead: int = 3,
        navdp_upstream_replan_every: int = 1,
        world_margin: float = 2.0,
        random_goal_bearing_deg: float = 60.0,
        random_goal_dist_range: tuple = (4.0, 8.0),
        seg_backend: str = "lora",
        seg_checkpoint: Optional[str] = None,
        seg_overlay: str = "mesh",
        annotations_dir: Optional[str] = DEFAULT_ANNOTATIONS_DIR,
        annotation_categories: Optional[Sequence[str]] = None,
    ):
        self.scene_path = scene_path
        self.heightmap_path = heightmap_path
        self.navdp_upstream_ckpt = navdp_upstream_ckpt
        self.navdp_upstream_root = navdp_upstream_root
        self.navdp_root = navdp_root
        self.rock_field_path = rock_field_path
        self.start_x = float(start_x)
        self.start_z = float(start_z)
        self.start_yaw_deg = float(start_yaw_deg)
        self.dt = float(dt)
        self.hz = float(hz)
        self.cbf_enabled = bool(cbf_enabled)
        self._cbf_kwargs = dict(
            d_safe=cbf_d_safe,
            gamma=cbf_gamma,
            deadzone=cbf_deadzone,
            orbit_kr=cbf_orbit_kr,
            orbit_hyst=cbf_orbit_hyst,
            pursuit_kp=cbf_pursuit_kp,
            goaround_forward=cbf_goaround_forward,
            hard_gate=cbf_hard_gate,
            robot_radius=robot_radius,
            safety_margin=safety_margin,
            obstacle_radius=obstacle_radius,
            max_yaw_rate=max_yaw_rate,
        )
        self.obj_mask_radius = float(obj_mask_radius)
        self.belief_goal_range = float(belief_goal_range)
        self.lost_goal_min_px = int(lost_goal_min_px)
        self.max_forward_speed = float(max_forward_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.navdp_upstream_port = navdp_upstream_port
        self.navdp_upstream_lookahead = int(navdp_upstream_lookahead)
        self.navdp_upstream_replan_every = int(navdp_upstream_replan_every)
        self.random_goal_bearing_deg = float(random_goal_bearing_deg)
        self.random_goal_dist_range = tuple(random_goal_dist_range)
        self.world_limit = max(SIZE_X, SIZE_Z) / 2.0 - float(world_margin)

        # Segmentation backend used by the "resolve" mode's SAM2 detection
        # (see first_frame_resolver._detect): which checkpoint, and whether
        # to feed it a mesh-overlay frame (see _do_resolve /
        # MarsHabitatEnv.get_mesh_overlay_rgb) instead of the plain camera
        # frame -- the overlay is only ever handed to the segmentation
        # model, never shown in the GUI or sent to the goal-selection VLM.
        self.seg_backend = seg_backend
        self.seg_checkpoint = seg_checkpoint
        self.seg_overlay = seg_overlay
        self.annotations_dir = annotations_dir if self.seg_overlay == "mesh" else None
        self.annotation_categories = annotation_categories

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._mode = MODE_IDLE
        self._manual_action = Action(0.0, 0.0, 0.0)
        self._world_goal: Optional[tuple] = None
        self._pending_resolve = False
        self._pending_reset = False
        self._pending_pixel_click: Optional[tuple] = None
        self._pending_confirm_segmentation = False
        self._pending_rerun_segmentation = False
        self._pending_pick_manually = False

        self.display = DisplayState()
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # thread-safe command API -- called from the GUI thread
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rover-controller"
        )
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_manual(self, v_fwd: float, yaw_rate: float) -> None:
        with self._lock:
            self._mode = MODE_MANUAL
            self._manual_action = Action(
                v_fwd=float(v_fwd), v_lat=0.0, yaw_rate=float(yaw_rate)
            )

    def random_goal(self) -> None:
        with self._lock:
            pose = self.display.pose
        if pose is None:
            return
        goal = random_ahead_point(
            pose,
            self._rng,
            bearing_range_deg=self.random_goal_bearing_deg,
            dist_range=self.random_goal_dist_range,
            world_limit=self.world_limit,
        )
        with self._lock:
            self._world_goal = goal
            self._mode = MODE_POINT
            self.display.goal_reached = False

    def go_home(self) -> None:
        with self._lock:
            self._world_goal = (self.start_x, self.start_z)
            self._mode = MODE_POINT
            self.display.goal_reached = False

    def request_pixel_goal(self, x_norm: float, y_norm: float) -> None:
        """Queue a click-to-goal request: (x_norm, y_norm) are normalized
        [0, 1] image coords in the *displayed* camera frame (origin
        top-left). Resolved against the live depth frame on the controller
        thread next tick -- see _handle_pixel_click."""
        with self._lock:
            self._pending_pixel_click = (float(x_norm), float(y_norm))

    def request_resolve(self) -> None:
        with self._lock:
            self._pending_resolve = True

    def request_confirm_segmentation(self) -> None:
        """Accept the goal/obstacle masks from the most recent resolve/rerun
        -- MODE_REVIEW_SEGMENTATION -> MODE_RESOLVE, driving starts next
        tick. No-op if not currently in review (stale click after a reset)."""
        with self._lock:
            self._pending_confirm_segmentation = True

    def request_rerun_segmentation(self) -> None:
        """Reject the current segmentation and re-invoke the resolver.
        Stays in MODE_REVIEW_SEGMENTATION for another round of review."""
        with self._lock:
            self._pending_rerun_segmentation = True

    def request_pick_manually(self) -> None:
        """Reject the auto-resolved goal entirely and fall through to the
        existing click-to-goal flow: untags the resolved masks and returns
        to MODE_IDLE so a subsequent request_pixel_goal drives normally."""
        with self._lock:
            self._pending_pick_manually = True

    def request_reset(self) -> None:
        with self._lock:
            self._pending_reset = True

    def stop_driving(self) -> None:
        with self._lock:
            self._mode = MODE_IDLE
            self._world_goal = None
            self._manual_action = Action(0.0, 0.0, 0.0)

    def snapshot(self) -> DisplayState:
        with self._lock:
            return copy.copy(self.display)

    def _run(self) -> None:
        # CbfObstacleAvoidance needs `navdp.extensions` (this repo's own navdp/
        # package's generic CBF/obstacle math -- unrelated to which driving
        # policy is active) importable; this is the same sys.path setup
        # sam_vla.run_navdp_rollout does before constructing it, reused as
        # plain infrastructure, not as "the custom model" itself.
        from sam_vla.policy.navdp_policy import _add_navdp_to_path, _resolve_navdp_root

        _add_navdp_to_path(_resolve_navdp_root(self.navdp_root))

        qwen_manager = QwenServerManager()
        try:
            with MarsHabitatEnv(
                self.scene_path,
                self.heightmap_path,
                services=[qwen_manager],
                start_x=self.start_x,
                start_z=self.start_z,
                start_yaw=math.radians(self.start_yaw_deg),
                with_semantic=True,
                rock_field_path=self.rock_field_path,
                annotations_dir=self.annotations_dir,
                annotation_categories=self.annotation_categories,
            ) as env:
                self._env_loop(env)
        except Exception as exc:  # pragma: no cover - surfaced to the GUI, not raised
            traceback.print_exc()
            with self._lock:
                self.display.status_text = f"FATAL: {exc}"
                self.display.error_text = str(exc)

    def _env_loop(self, env: MarsHabitatEnv) -> None:
        mask_dir = tempfile.mkdtemp(prefix="mars_nav_masks_")
        obs0 = env.get_observation(frame_idx=0)

        policy = NavdpUpstreamPolicy(
            checkpoint_path=self.navdp_upstream_ckpt,
            navdp_upstream_root=self.navdp_upstream_root,
            port=self.navdp_upstream_port,
            image_hw=obs0.rgb.shape[:2],
            hfov_deg=HFOV_DEG,
            lookahead=self.navdp_upstream_lookahead,
            replan_every=self.navdp_upstream_replan_every,
            max_forward_speed=self.max_forward_speed,
            max_yaw_rate=self.max_yaw_rate,
        )
        with self._lock:
            self.display.status_text = "NavDP policy is being loaded"
        policy.start()

        belief_tracker = self._new_belief_tracker()
        avoidance = self._new_avoidance() if self.cbf_enabled else None
        # goal/obstacle mask objects registered by the last "resolve" --
        # actually removed (env.remove_object_mask) rather than merely
        # untagged whenever they stop being current: on the next
        # resolve/rerun, on reset, on abandoning an unconfirmed review (Pick
        # Manually or switching to another driving mode without Confirm), or
        # on reaching a confirmed resolve goal -- see _clear_masks. Without
        # this, every resolve within one run leaves its old mesh behind.
        goal_objects: list = []
        obstacle_objects: list = []
        goal_spec = GoalSpec(
            goal_bbox_norm=(0.0, 0.0, 1.0, 1.0),
            obstacle_bboxes_norm=[],
            instruction_text="(no goal resolved yet)",
        )

        with self._lock:
            self.display.pose = obs0.pose
            self.display.status_text = "idle -- pick a driving mode"

        step = 0
        period = 1.0 / self.hz
        was_reviewing = False
        try:
            while self._running:
                t0 = time.time()

                (
                    do_resolve,
                    do_reset,
                    pixel_click,
                    do_confirm_seg,
                    do_rerun_seg,
                    do_pick_manual,
                ) = self._consume_pending()

                if do_reset:
                    goal_objects, obstacle_objects = self._clear_masks(
                        env, goal_objects, obstacle_objects
                    )
                    belief_tracker = self._new_belief_tracker()
                    if avoidance is not None:
                        avoidance = self._new_avoidance()
                    y = env.get_height_at_xz(self.start_x, self.start_z)
                    env.step(
                        Pose(
                            x=self.start_x,
                            y=y,
                            z=self.start_z,
                            yaw=math.radians(self.start_yaw_deg),
                        )
                    )
                    with self._lock:
                        self._mode = MODE_IDLE
                        self._world_goal = None
                        self._manual_action = Action(0.0, 0.0, 0.0)
                        self.display.status_text = "reset to spawn"
                        self.display.goal_reached = False
                        self.display.click_status = ""

                if do_resolve or do_rerun_seg:
                    goal_objects, obstacle_objects, goal_spec = self._do_resolve(
                        env, step, mask_dir, goal_objects, obstacle_objects, goal_spec
                    )
                    belief_tracker.belief_g = None

                if do_confirm_seg:
                    with self._lock:
                        if self._mode == MODE_REVIEW_SEGMENTATION:
                            self._mode = MODE_RESOLVE
                            self.display.status_text = (
                                f"resolved: {goal_spec.instruction_text}"
                            )
                            self.display.goal_reached = False

                if do_pick_manual:
                    with self._lock:
                        pick_manual_was_reviewing = (
                            self._mode == MODE_REVIEW_SEGMENTATION
                        )
                        if pick_manual_was_reviewing:
                            self._mode = MODE_IDLE
                            self._world_goal = None
                            self.display.status_text = (
                                "pick a goal point manually -- click the camera view"
                            )
                            self.display.goal_reached = False
                    if pick_manual_was_reviewing:
                        goal_objects, obstacle_objects = self._clear_masks(
                            env, goal_objects, obstacle_objects
                        )

                obs = env.get_observation(frame_idx=step)
                semantic = env.get_semantic_frame()
                goal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
                obstacle_mask = (semantic == MESH_OBST_ID).astype("uint8") * 255

                if pixel_click is not None:
                    self._handle_pixel_click(obs, pixel_click)

                with self._lock:
                    mode = self._mode
                    manual_action = self._manual_action
                    world_goal = self._world_goal

                if was_reviewing and mode not in (
                    MODE_REVIEW_SEGMENTATION,
                    MODE_RESOLVE,
                ):
                    # Review was abandoned by switching to another driving
                    # mode directly (random goal / go home / manual drive /
                    # click-to-goal / stop) instead of going through Confirm,
                    # Rerun, or Pick Manually -- those already clear their own
                    # masks, so this only fires for the remaining paths.
                    goal_objects, obstacle_objects = self._clear_masks(
                        env, goal_objects, obstacle_objects
                    )
                was_reviewing = mode == MODE_REVIEW_SEGMENTATION

                action = Action(0.0, 0.0, 0.0)
                trajectory = None
                goal_bearing: Optional[float] = None
                unresolved = False
                dist_to_goal: Optional[float] = None

                if mode == MODE_MANUAL:
                    action = manual_action
                elif mode == MODE_POINT and world_goal is not None:
                    forward, left = body_frame_goal(obs.pose, world_goal)
                    belief_tracker.observe_body_point(forward, left)
                    goal_bearing = belief_tracker.bearing()
                    if math.hypot(forward, left) < POINT_GOAL_REACHED_M:
                        with self._lock:
                            self._mode = MODE_IDLE
                            self._world_goal = None
                            self.display.goal_reached = True
                        mode = MODE_IDLE
                    else:
                        policy.set_goal_body(forward, left)
                        action, _vla_result = policy.act_verbose(
                            obs, semantic, goal_spec, step
                        )
                        trajectory = getattr(policy, "_last_trajectory", None)
                elif mode == MODE_RESOLVE:
                    belief_tracker.observe(goal_mask, obs.depth)
                    if belief_tracker.belief_g is not None:
                        forward, left = (float(v) for v in belief_tracker.belief_g)
                        if math.hypot(forward, left) < POINT_GOAL_REACHED_M:
                            with self._lock:
                                self._mode = MODE_IDLE
                                self.display.goal_reached = True
                            mode = MODE_IDLE
                            goal_objects, obstacle_objects = self._clear_masks(
                                env, goal_objects, obstacle_objects
                            )
                        else:
                            policy.set_goal_body(forward, left)
                            action, _vla_result = policy.act_verbose(
                                obs, semantic, goal_spec, step
                            )
                            trajectory = getattr(policy, "_last_trajectory", None)
                            goal_bearing = belief_tracker.bearing()
                    else:
                        # Never sighted the goal mask yet this episode -- hold
                        # rather than drive on NavdpUpstreamPolicy's hidden
                        # constructor default (see this module's docstring /
                        # next.md's Integration-project Phase 5).
                        unresolved = True

                obstacle_point = None
                cbf_info: dict = {}
                if avoidance is not None and mode != MODE_MANUAL:
                    height, width = obs.depth.shape[:2]
                    intr = intrinsics_from_hfov(height, width, HFOV_DEG)
                    obstacle_point = avoidance.nearest_obstacle(
                        obstacle_mask, obs.depth, intr
                    )
                    action, cbf_info = avoidance.apply(
                        action, obstacle_point, goal_bearing
                    )

                new_pose = integrate_mars(obs.pose, action, self.dt)
                env.step(new_pose)

                if mode in (MODE_POINT, MODE_RESOLVE):
                    belief_tracker.propagate(action, self.dt)
                if mode == MODE_POINT and world_goal is not None:
                    dist_to_goal = math.hypot(
                        world_goal[0] - new_pose.x, world_goal[1] - new_pose.z
                    )
                elif mode == MODE_RESOLVE and belief_tracker.belief_g is not None:
                    dist_to_goal = belief_tracker.distance()

                goal_pixel = None
                if mode == MODE_POINT and world_goal is not None:
                    goal_y = env.get_height_at_xz(world_goal[0], world_goal[1])
                    goal_pixel = project_world_to_pixel(
                        obs.pose,
                        (world_goal[0], goal_y, world_goal[1]),
                        HFOV_DEG,
                        obs.rgb.shape[1],
                        obs.rgb.shape[0],
                    )

                status = self._status_text(mode, unresolved, goal_spec, dist_to_goal)
                vis_rgb = overlay_semantic_masks(obs.rgb, semantic, text=status)
                if goal_pixel is not None:
                    vis_rgb = draw_point_marker(vis_rgb, goal_pixel)

                with self._lock:
                    d = self.display
                    d.vis_rgb = vis_rgb
                    d.pose = new_pose
                    d.mode = mode
                    d.status_text = status
                    d.action = action
                    d.distance = dist_to_goal
                    d.goal_world = world_goal
                    d.belief_g = (
                        None
                        if belief_tracker.belief_g is None
                        else tuple(float(v) for v in belief_tracker.belief_g)
                    )
                    d.trajectory = trajectory
                    d.obstacle_point = (
                        None
                        if obstacle_point is None
                        else tuple(float(v) for v in obstacle_point)
                    )
                    d.cbf_info = cbf_info
                    d.step = step
                    d.frame_count += 1

                step += 1
                remaining = period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            policy.stop()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _consume_pending(
        self,
    ) -> tuple[bool, bool, Optional[tuple], bool, bool, bool]:
        with self._lock:
            do_resolve = self._pending_resolve
            do_reset = self._pending_reset
            pixel_click = self._pending_pixel_click
            do_confirm_seg = self._pending_confirm_segmentation
            do_rerun_seg = self._pending_rerun_segmentation
            do_pick_manual = self._pending_pick_manually
            self._pending_resolve = False
            self._pending_reset = False
            self._pending_pixel_click = None
            self._pending_confirm_segmentation = False
            self._pending_rerun_segmentation = False
            self._pending_pick_manually = False
        return (
            do_resolve,
            do_reset,
            pixel_click,
            do_confirm_seg,
            do_rerun_seg,
            do_pick_manual,
        )

    def _handle_pixel_click(self, obs, pixel_click: tuple) -> None:
        """Backproject a clicked (x_norm, y_norm) into a world-frame (x, z)
        point via the live depth frame and switch to MODE_POINT driving
        toward it -- same bbox_to_world median-over-patch machinery
        first_frame_resolver's goal already uses, applied to a small patch
        around the clicked pixel (not just the single pixel) so a click that
        lands exactly on a depth discontinuity (a rock's silhouette edge)
        doesn't seed a wildly wrong point. Unlike the belief-tracked RESOLVE
        goal, this point is never re-derived from odometry/dead-reckoning --
        it's a fixed world coordinate, recovered fresh from ground-truth pose
        every tick by body_frame_goal, so it can't drift."""
        x_norm, y_norm = pixel_click
        height, width = np.asarray(obs.depth).shape[:2]
        px, py = x_norm * width, y_norm * height
        margin = 4.0
        bbox_norm = (
            max(0.0, (px - margin) / width),
            max(0.0, (py - margin) / height),
            min(1.0, (px + margin) / width),
            min(1.0, (py + margin) / height),
        )
        goal_xyz = bbox_to_world(obs, bbox_norm, hfov_deg=HFOV_DEG)
        with self._lock:
            if goal_xyz is None:
                self.display.click_status = "click ignored: no valid depth there"
                return
            gx, _gy, gz = goal_xyz
            self._world_goal = (gx, gz)
            self._mode = MODE_POINT
            self.display.goal_reached = False
            self.display.click_status = f"point goal set at world ({gx:.1f}, {gz:.1f})"

    def _new_belief_tracker(self) -> BeliefGoalTracker:
        return BeliefGoalTracker(
            hfov_deg=HFOV_DEG,
            goal_range=self.belief_goal_range,
            min_px=self.lost_goal_min_px,
        )

    def _new_avoidance(self) -> CbfObstacleAvoidance:
        return CbfObstacleAvoidance(**self._cbf_kwargs)

    @staticmethod
    def _clear_masks(env, goal_objects: list, obstacle_objects: list) -> tuple:
        """Actually remove every registered goal/obstacle mask mesh from the
        scene (not just untag it) and return the emptied (goal, obstacle)
        lists -- see _env_loop's goal_objects/obstacle_objects docstring."""
        for obj in goal_objects + obstacle_objects:
            env.remove_object_mask(obj)
        return [], []

    def _do_resolve(
        self, env, step, mask_dir, goal_objects, obstacle_objects, current_goal_spec
    ):
        obs_r = env.get_observation(frame_idx=step)
        # detect_rgb (if any) is a separate frame with the mesh_tight_bound2
        # annotation hulls composited in, fed only to the segmentation
        # model -- obs_r.rgb (the plain camera frame) is what the VLM sees
        # and what the GUI/backprojection use, unmodified.
        detect_rgb = env.get_mesh_overlay_rgb() if self.seg_overlay == "mesh" else None
        try:
            goal_spec_r, _vlm_result, _dets = first_frame_resolver.resolve_verbose(
                obs_r.rgb,
                detect_rgb=detect_rgb,
                backend=self.seg_backend,
                checkpoint_path=self.seg_checkpoint,
            )
        except Exception as exc:
            with self._lock:
                self.display.status_text = f"resolve failed: {exc}"
            # Leave the mode/masks/goal_spec exactly as they were -- a failed
            # resolve shouldn't clobber whatever was previously resolved.
            return goal_objects, obstacle_objects, current_goal_spec

        goal_position = backproject_goal_position(obs_r, goal_spec_r, hfov_deg=HFOV_DEG)
        goal_objects, obstacle_objects = self._clear_masks(
            env, goal_objects, obstacle_objects
        )
        new_goal_objects: list = []
        new_obstacle_objects: list = []
        if goal_position is not None:
            new_goal_objects.append(
                env.register_object_mask(
                    goal_position, MESH_GOAL_ID, self.obj_mask_radius, mask_dir, "goal"
                )
            )
            status_msg = (
                f"reviewing resolved goal: '{goal_spec_r.instruction_text}' -- "
                "Confirm to drive, Rerun to retry, or Pick Manually"
            )
        else:
            status_msg = (
                "resolved goal has no valid depth -- skipping mask "
                "(Rerun or Pick Manually)"
            )
        for i, obstacle_bbox in enumerate(goal_spec_r.obstacle_bboxes_norm):
            obstacle_position = bbox_to_world(obs_r, obstacle_bbox, hfov_deg=HFOV_DEG)
            if obstacle_position is None:
                continue
            new_obstacle_objects.append(
                env.register_object_mask(
                    obstacle_position,
                    MESH_OBST_ID,
                    self.obj_mask_radius,
                    mask_dir,
                    f"obstacle_{i}",
                )
            )

        with self._lock:
            self._mode = MODE_REVIEW_SEGMENTATION
            self._world_goal = None
            self.display.status_text = status_msg
            self.display.goal_reached = False
        return new_goal_objects, new_obstacle_objects, goal_spec_r

    @staticmethod
    def _status_text(mode, unresolved, goal_spec, dist) -> str:
        dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
        if mode == MODE_IDLE:
            return "IDLE"
        if mode == MODE_MANUAL:
            return "MANUAL DRIVE"
        if mode == MODE_POINT:
            return f"POINT GOAL  dist={dist_txt}"
        if mode == MODE_REVIEW_SEGMENTATION:
            label = goal_spec.instruction_text if goal_spec is not None else "?"
            return f"REVIEW SEGMENTATION '{label}' -- Confirm / Rerun / Pick Manually"
        if mode == MODE_RESOLVE:
            if unresolved:
                return "RESOLVE  waiting for goal sighting..."
            label = goal_spec.instruction_text if goal_spec is not None else "?"
            return f"RESOLVE '{label}'  dist={dist_txt}"
        return mode
