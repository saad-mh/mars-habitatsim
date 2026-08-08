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
             any caller-supplied point) with no detector in the loop:
             goal_math.body_frame_goal recovers the ground-truth body-frame
             point every tick, fed straight into BeliefGoalTracker via
             observe_body_point (see its docstring -- built for exactly this,
             ground-truth callers with no mask/depth to source a sighting
             from).
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
from typing import Optional

import numpy as np

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, GoalSpec, Pose
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.terrain import SIZE_X, SIZE_Z
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.perception.semantic_overlay import overlay_semantic_masks
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy
from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance
from sam_vla.vlm.qwen_server_manager import QwenServerManager

from nav.goal_math import body_frame_goal, random_ahead_point

MODE_IDLE = "idle"
MODE_POINT = "point"
MODE_RESOLVE = "resolve"
MODE_MANUAL = "manual"

# Ground-truth distance at which a point goal counts as reached -- independently
# chosen (not imported), same rover-scale ballpark as launch_mars.sh's own GUI.
POINT_GOAL_REACHED_M = 0.7


@dataclass
class DisplayState:
    """Snapshot of controller state for the GUI thread to read. Never mutated
    in place by the GUI -- RoverController.snapshot() hands out a shallow
    copy each call."""

    vis_rgb: Optional[np.ndarray] = None
    pose: Optional[Pose] = None
    mode: str = MODE_IDLE
    status_text: str = "starting..."
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

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._mode = MODE_IDLE
        self._manual_action = Action(0.0, 0.0, 0.0)
        self._world_goal: Optional[tuple] = None
        self._pending_resolve = False
        self._pending_reset = False

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
            self._manual_action = Action(v_fwd=float(v_fwd), v_lat=0.0, yaw_rate=float(yaw_rate))

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

    def request_resolve(self) -> None:
        with self._lock:
            self._pending_resolve = True

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

    # ------------------------------------------------------------------ #
    # controller-thread body
    # ------------------------------------------------------------------ #
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
            self.display.status_text = "loading NavDP upstream server..."
        policy.start()

        belief_tracker = self._new_belief_tracker()
        avoidance = self._new_avoidance() if self.cbf_enabled else None
        # goal/obstacle mask objects registered by the last "resolve" -- kept
        # around (rather than deleted, no handle for that without reaching
        # into MarsHabitatEnv's private sim) and neutralised (semantic_id=0,
        # the same "untag, don't remove" convention sam_vla.run_navdp_rollout
        # already uses for its base-station marker) before the next batch.
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
        try:
            while self._running:
                t0 = time.time()

                do_resolve, do_reset = self._consume_pending()

                if do_reset:
                    for obj in goal_objects + obstacle_objects:
                        obj.semantic_id = 0
                    goal_objects, obstacle_objects = [], []
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

                if do_resolve:
                    goal_objects, obstacle_objects, goal_spec = self._do_resolve(
                        env, step, mask_dir, goal_objects, obstacle_objects, goal_spec
                    )
                    belief_tracker.belief_g = None

                with self._lock:
                    mode = self._mode
                    manual_action = self._manual_action
                    world_goal = self._world_goal

                obs = env.get_observation(frame_idx=step)
                semantic = env.get_semantic_frame()
                goal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
                obstacle_mask = (semantic == MESH_OBST_ID).astype("uint8") * 255

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
                    action, cbf_info = avoidance.apply(action, obstacle_point, goal_bearing)

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

                status = self._status_text(mode, unresolved, goal_spec, dist_to_goal)
                vis_rgb = overlay_semantic_masks(obs.rgb, semantic, text=status)

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
                        None if obstacle_point is None else tuple(float(v) for v in obstacle_point)
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
    def _consume_pending(self) -> tuple[bool, bool]:
        with self._lock:
            do_resolve = self._pending_resolve
            do_reset = self._pending_reset
            self._pending_resolve = False
            self._pending_reset = False
        return do_resolve, do_reset

    def _new_belief_tracker(self) -> BeliefGoalTracker:
        return BeliefGoalTracker(
            hfov_deg=HFOV_DEG,
            goal_range=self.belief_goal_range,
            min_px=self.lost_goal_min_px,
        )

    def _new_avoidance(self) -> CbfObstacleAvoidance:
        return CbfObstacleAvoidance(**self._cbf_kwargs)

    def _do_resolve(
        self, env, step, mask_dir, goal_objects, obstacle_objects, current_goal_spec
    ):
        obs_r = env.get_observation(frame_idx=step)
        try:
            goal_spec_r, _vlm_result, _dets = first_frame_resolver.resolve_verbose(
                obs_r.rgb
            )
        except Exception as exc:
            with self._lock:
                self.display.status_text = f"resolve failed: {exc}"
            # Leave the mode/masks/goal_spec exactly as they were -- a failed
            # resolve shouldn't clobber whatever was previously resolved.
            return goal_objects, obstacle_objects, current_goal_spec

        goal_position = backproject_goal_position(obs_r, goal_spec_r, hfov_deg=HFOV_DEG)
        for obj in goal_objects + obstacle_objects:
            obj.semantic_id = 0
        new_goal_objects: list = []
        new_obstacle_objects: list = []
        if goal_position is not None:
            new_goal_objects.append(
                env.register_object_mask(
                    goal_position, MESH_GOAL_ID, self.obj_mask_radius, mask_dir, "goal"
                )
            )
        else:
            with self._lock:
                self.display.status_text = "resolved goal has no valid depth -- skipping mask"
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
            self._mode = MODE_RESOLVE
            self._world_goal = None
            self.display.status_text = f"resolved: {goal_spec_r.instruction_text}"
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
        if mode == MODE_RESOLVE:
            if unresolved:
                return "RESOLVE  waiting for goal sighting..."
            label = goal_spec.instruction_text if goal_spec is not None else "?"
            return f"RESOLVE '{label}'  dist={dist_txt}"
        return mode
