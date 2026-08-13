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
salience-pick auto-resolve/nav-command-parsing/uncertainty-halt prompts are
the only subprocesses, spawned automatically by
NavdpUpstreamPolicy/QwenServerManager exactly as sam_vla.run_navdp_rollout
already does. Open-vocabulary target grounding (Ground Target / a mission's
GO_TO/FIND step) runs in-process instead, via
sam_vla.goal_resolution.dino_grounding_resolver (navdp.extensions.
GroundingDINODetector, lazily loaded onto self.dino_device on first use --
no subprocess/server of its own).

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
  resolve -- with no target_text: one-shot first_frame_resolver.resolve_verbose()
             on the current frame (SAM2 detections + Qwen VLM salience pick,
             no per-preset text targeting -- qwen_client.select_goal has no
             such parameter, see next.md). With a target_text (Ground
             Target / a mission GO_TO/FIND step): dino_grounding_resolver
             instead -- GroundingDINO detects the named open-vocabulary
             object directly (flags, the home-base cuboid; SAM2 still runs
             to seed obstacles). Either way, the resulting goal/obstacle
             masks are tracked every tick exactly as sam_vla.run_navdp_rollout's
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
from sam_vla.core.ghost_mask import draw_ghost_mask, project_body_point_to_pixel
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
    project_world_point_with_pixel_radius,
    project_world_to_pixel,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, GoalSpec, Pose
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.terrain import SIZE_X, SIZE_Z
from sam_vla.goal_resolution import dino_grounding_resolver, first_frame_resolver
from sam_vla.perception.semantic_overlay import (
    draw_point_marker,
    overlay_semantic_masks,
)
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy
from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance
from sam_vla.vlm import qwen_client
from sam_vla.vlm.qwen_config import QWEN_SERVER_HOST, QWEN_SERVER_PORT
from sam_vla.vlm.qwen_server_manager import QwenServerManager

from nav.event_log import EventLogger
from nav.goal_math import body_frame_goal, heading_ahead_point, random_ahead_point
from nav.mission import GoalKind, Mission

# Uncertainty-halt: reuses vl_direction's own "uncertainty" prompt/session
# machinery (its documented sole integration point, see vl_direction's module
# docstring) against the real BeliefGoalTracker.uncertainty_value() computed
# below -- not scripts/habitat_tests/kb_teleop_vl.py's synthetic covariance
# proxy, and not imported from that script, to keep nav/ decoupled from it.
# Talks to the same qwen_server this module already starts for goal
# resolution/VLA actions (sam_vla.vlm.qwen_server now answers vl_direction's
# "generate" wire mode too) instead of spawning a second qwen_vlm-env model
# copy on its own port -- see QwenSocketClient construction in _run() below.
from vl_direction import config as vl_dir_config
from vl_direction.client import QwenSocketClient as VlDirectionQwenClient
from vl_direction.uncertainty_session import UncertaintySession

MODE_IDLE = "idle"
MODE_POINT = "point"
MODE_RESOLVE = "resolve"
MODE_MANUAL = "manual"
# Drives NavdpUpstreamPolicy toward a ghost point placed along the bearing to
# self._mission_target_yaw (see TURN_GHOST_* below) rather than spinning in
# place -- only ever entered by a Mission's TURN sub-goal (see
# _start_mission_subgoal), not user-reachable directly.
MODE_TURN = "turn"
# Holding state between a completed SAM2/Qwen resolve and the user accepting
# it -- entered by _do_resolve instead of MODE_RESOLVE, left via
# request_confirm_segmentation (-> MODE_RESOLVE, starts driving),
# request_rerun_segmentation (stays here, re-invokes _do_resolve), or
# request_pick_manually (-> MODE_IDLE, falls through to click-to-goal).
MODE_REVIEW_SEGMENTATION = "review_segmentation"

# Ground-truth distance at which a point goal counts as reached
POINT_GOAL_REACHED_M = 1.5

# Ground-truth yaw error at which a MODE_TURN sub-goal counts as reached
TURN_GOAL_REACHED_RAD = math.radians(3.0)

# MODE_TURN no longer spins in place -- it drives NavdpUpstreamPolicy toward a
# body-frame "ghost point" set via policy.set_goal_body(forward, left), same
# point-goal mechanism MODE_POINT/MODE_RESOLVE already use. The point sits
# TURN_GHOST_LOOKAHEAD_M ahead along the remaining-yaw-error bearing, clamped
# to +/- half the camera's HFOV so forward stays positive (always drivable/
# projectable) -- full remaining turn pins it to the frame edge, and it
# slides to dead center as the turn completes. Deliberately not routed
# through any depth/3D backprojection: project_body_point_to_pixel always
# resolves to the single horizontal line at the frame's vertical center, so
# the on-screen position is a deterministic function of the clamped bearing
# alone.
TURN_GHOST_LOOKAHEAD_M = 3.0
TURN_GHOST_MAX_BEARING_RAD = math.radians(HFOV_DEG / 2.0)
TURN_GHOST_RADIUS_PX = 18.0

# World-space radius of the ghost-mask circle drawn over a DINO-grounded
# open-vocabulary target (see _do_resolve/self._ground_target_world) --
# reprojected to a depth-dependent pixel radius every frame, not a fixed
# on-screen size (see project_world_point_with_pixel_radius).
GHOST_TARGET_RADIUS_M = 0.5

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
    # Active Mission's status() line (nav/gui.py's Command panel), "" when no
    # mission is running -- see nav.mission.Mission.
    mission_status: str = ""
    # Uncertainty-halt (vl_direction's "uncertainty" prompt against the real
    # BeliefGoalTracker, see this module's imports): uncertainty_value grows
    # while a MODE_RESOLVE goal mask stays unseen; once it reaches
    # uncertainty_threshold, driving halts and a heading is requested.
    uncertainty_enabled: bool = False
    uncertainty_value: float = 0.0
    uncertainty_threshold: float = 0.0
    uncertainty_halted: bool = False
    uncertainty_request_in_flight: bool = False
    uncertainty_searching: bool = False
    uncertainty_line: str = ""


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
        flag_seed: Optional[int] = None,
        num_flags: int = 6,
        flag_min_spacing: float = 1.5,
        flag_boundary_margin: float = 2.0,
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
        navdp_upstream_server_variant: str = "navdp",
        navdp_upstream_planner_mode: str = "s2diff",
        world_margin: float = 2.0,
        random_goal_bearing_deg: float = 60.0,
        random_goal_dist_range: tuple = (4.0, 8.0),
        go_direction_distance_m: float = 4.0,
        seg_backend: str = "lora",
        seg_checkpoint: Optional[str] = None,
        seg_overlay: str = "mesh",
        dino_model_id: str = dino_grounding_resolver.DEFAULT_DINO_MODEL_ID,
        dino_device: str = dino_grounding_resolver.DEFAULT_DINO_DEVICE,
        dino_box_threshold: float = dino_grounding_resolver.DEFAULT_BOX_THRESHOLD,
        dino_text_threshold: float = dino_grounding_resolver.DEFAULT_TEXT_THRESHOLD,
        mission_sweep_yaws: int = 8,
        mission_belief_sweep_every: int = 15,
        mission_belief_cov_growth: float = 0.0002,
        mission_belief_cov_growth_rate: float = 0.0,
        annotations_dir: Optional[str] = DEFAULT_ANNOTATIONS_DIR,
        annotation_categories: Optional[Sequence[str]] = None,
        uncertainty_enabled: bool = True,
        uncertainty_cov_threshold: float = vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD,
        uncertainty_cov_growth: float = 0.01,
        uncertainty_cov_growth_rate: float = 0.0,
        uncertainty_search_dist: float = 4.0,
    ):
        self.scene_path = scene_path
        self.heightmap_path = heightmap_path
        self.navdp_upstream_ckpt = navdp_upstream_ckpt
        self.navdp_upstream_root = navdp_upstream_root
        self.navdp_root = navdp_root
        self.rock_field_path = rock_field_path
        self.flag_seed = flag_seed
        self.num_flags = int(num_flags)
        self.flag_min_spacing = float(flag_min_spacing)
        self.flag_boundary_margin = float(flag_boundary_margin)
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
        self.navdp_upstream_server_variant = navdp_upstream_server_variant
        self.navdp_upstream_planner_mode = navdp_upstream_planner_mode
        self.random_goal_bearing_deg = float(random_goal_bearing_deg)
        self.random_goal_dist_range = tuple(random_goal_dist_range)
        self.go_direction_distance_m = float(go_direction_distance_m)
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

        # GroundingDINO grounder config (see _do_resolve's target_text path
        # -> dino_grounding_resolver). Independent of seg_backend/seg_checkpoint
        # above, which stay SAM2/SAM2-LoRA -- DINO only replaces the Qwen
        # open-vocabulary grounder, not the SAM2 obstacle/salience path.
        self.dino_model_id = dino_model_id
        self.dino_device = dino_device
        self.dino_box_threshold = float(dino_box_threshold)
        self.dino_text_threshold = float(dino_text_threshold)

        # In-place 360deg scan a GO_TO/FIND mission sub-goal runs (see
        # _start_mission_subgoal) before the existing single-frame
        # _do_resolve -- the search/scan behaviour mission.py's GoalKind.FIND
        # docstring already promises but _start_mission_subgoal previously
        # never actually did (a target outside the current single frame
        # just failed to resolve and got skipped).
        self.mission_sweep_yaws = int(mission_sweep_yaws)

        # Periodic multi-goal DINO belief sweep while a Mission is driving (see
        # _sweep_goal_beliefs) -- checks the current frame against every remaining
        # GO_TO/FIND target text every this-many ticks, seeding/refreshing a
        # persistent per-goal-text BeliefGoalTracker (goal_beliefs in _env_loop) so
        # a goal sighted while driving toward an earlier one isn't forgotten by the
        # time the mission reaches it. <= 0 disables the periodic sweep entirely
        # (a mission step can still resolve via _do_resolve's own front-facing
        # check when it starts, same as before this feature existed).
        self.mission_belief_sweep_every = int(mission_belief_sweep_every)

        # Growth rate for beliefs seeded passively by _sweep_goal_beliefs (a goal
        # other than the one currently being driven to, possibly not seen again for
        # the rest of that leg -- tens of seconds to minutes). Deliberately much
        # slower than uncertainty_cov_growth below: that rate is tuned for the
        # uncertainty-halt's live-tracking use case (a goal briefly losing its mask
        # for a couple seconds), and reusing it here made any sweep-seeded belief
        # decay past uncertainty_cov_threshold within ~1s of the sighting -- almost
        # always before the mission actually reached that goal, defeating the sweep
        # entirely (see nav/rover_controller.py's _start_mission_subgoal "usable
        # prior belief" check). Trackers promoted to actively-driven (see
        # _activate_belief_tracker) switch back to the tight halt-tuned rate.
        self.mission_belief_cov_growth = float(mission_belief_cov_growth)
        self.mission_belief_cov_growth_rate = float(mission_belief_cov_growth_rate)

        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_cov_threshold = float(uncertainty_cov_threshold)
        self.uncertainty_cov_growth = float(uncertainty_cov_growth)
        self.uncertainty_cov_growth_rate = float(uncertainty_cov_growth_rate)
        self.uncertainty_search_dist = float(uncertainty_search_dist)

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._mode = MODE_IDLE
        self._manual_action = Action(0.0, 0.0, 0.0)
        self._world_goal: Optional[tuple] = None
        self._pending_resolve = False
        # None -> first_frame_resolver's SAM2+Qwen-salience path (rocks etc,
        # unchanged). Set -> dino_grounding_resolver's direct-DINO-bbox path
        # for open-vocabulary targets (flags, the home-base cuboid) SAM2
        # wasn't trained on. Persists across a Rerun of the same resolve
        # (see request_rerun_segmentation), cleared on Reset.
        self._resolve_target_text: Optional[str] = None
        # World position of the last DINO-grounded target (see _do_resolve),
        # for the ghost-mask overlay drawn every frame in _run. Only ever
        # touched from _run's own background thread (_do_resolve/_clear_masks
        # write it, _run's render section reads it, all on that one thread) --
        # no lock needed, unlike the self._lock-guarded fields above/below
        # that cross threads.
        self._ground_target_world: Optional[tuple] = None
        self._pending_reset = False
        self._pending_clear_all = False
        self._pending_pixel_click: Optional[tuple] = None
        self._pending_confirm_segmentation = False
        self._pending_rerun_segmentation = False
        self._pending_pick_manually = False
        self._pending_uncertainty_heading: Optional[float] = None
        self._pending_uncertainty_retry = False
        # Active/queued Mission (nav/gui.py's Command panel) -- only ever
        # touched under self._lock: submit_nav_command (GUI thread) queues,
        # _env_loop's mission-advance block and every manual-override command
        # method (GUI thread, cancelling a stale mission) both read/write
        # self._mission, so unlike self._ground_target_world this can't be
        # left controller-thread-only.
        # Raw text waiting for a frame so its (directions, goals) VLM split
        # can be dispatched (see submit_nav_command/_dispatch_nav_command) --
        # distinct from self._pending_mission below, which holds the
        # *already-parsed* Mission once that background call returns.
        self._pending_nav_command_text: Optional[str] = None
        self._pending_mission: Optional[Mission] = None
        self._mission: Optional[Mission] = None
        # Absolute target world yaw (radians, Pose.yaw's convention) for an
        # active MODE_TURN sub-goal -- controller-thread-only (set and read
        # inside _env_loop/_start_mission_subgoal alone), same pattern as
        # self._ground_target_world.
        self._mission_target_yaw: Optional[float] = None

        # Only ever touched inside self._lock (read/written from both the
        # controller thread and the uncertainty-request background thread).
        self._uncertainty_halted = False
        self._uncertainty_request_in_flight = False
        self._uncertainty_search_goal: Optional[tuple] = None

        self.display = DisplayState()
        self.display.uncertainty_enabled = self.uncertainty_enabled
        self.display.uncertainty_threshold = self.uncertainty_cov_threshold
        self._rng = np.random.default_rng()
        # Goal-detected/goal-reached event log (nav/logs/, gitignored) --
        # see EventLogger's docstring. Created once per RoverController (one
        # log file per GUI process lifetime), not per-episode/reset.
        self._event_log = EventLogger()

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
            self._mission = None

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
            self._mission = None

    def go_home(self) -> None:
        with self._lock:
            self._world_goal = (self.start_x, self.start_z)
            self._mode = MODE_POINT
            self.display.goal_reached = False
            self._mission = None

    def request_pixel_goal(self, x_norm: float, y_norm: float) -> None:
        """Queue a click-to-goal request: (x_norm, y_norm) are normalized
        [0, 1] image coords in the *displayed* camera frame (origin
        top-left). Resolved against the live depth frame on the controller
        thread next tick -- see _handle_pixel_click."""
        with self._lock:
            self._pending_pixel_click = (float(x_norm), float(y_norm))
            self._mission = None

    def request_resolve(self, target_text: Optional[str] = None) -> None:
        """target_text=None -> the original SAM2+Qwen-salience auto-resolve
        (unchanged). A non-empty target_text (e.g. "flag", "blue cuboid")
        switches to direct GroundingDINO grounding for that object instead --
        see dino_grounding_resolver, wired in via _do_resolve. A manual
        override, so it cancels any active Mission (see submit_nav_command)
        the same way every other manual command below does."""
        with self._lock:
            self._pending_resolve = True
            self._resolve_target_text = target_text or None
            self._mission = None

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
            self._mission = None

    def clear_all_goals(self) -> None:
        """Hard reset of every goal/mission/mask/uncertainty-halt state back
        to a blank idle slate -- everything request_reset does *except*
        teleporting the rover back to spawn (see _env_loop's
        `do_reset or do_clear_all` block, which shares the rest of that
        logic). Distinct from stop_driving(), which only clears the active
        world_goal/mission and leaves resolved masks and uncertainty state
        alone."""
        with self._lock:
            self._pending_clear_all = True
            self._mission = None

    def submit_uncertainty_heading(self, angle_deg: float) -> None:
        """Answer an active uncertainty halt with a rover-front-relative
        heading (degrees, same convention as heading_ahead_point/nav/gui.py's
        numpad panel). No-op (consumed and dropped) if not currently
        halted -- guards a stale click racing a halt that's already resolved
        by a fresh sighting."""
        with self._lock:
            self._pending_uncertainty_heading = float(angle_deg)

    def retry_uncertainty_request(self) -> None:
        """Re-request the VLM sweep description for an active uncertainty
        halt. No-op if not currently halted or a request is already in
        flight."""
        with self._lock:
            self._pending_uncertainty_retry = True

    def submit_nav_command(self, text: str) -> None:
        """Queue a free-text nav command (nav/gui.py's Command panel) for
        background splitting into (directions, goals) via the Qwen VLM
        (qwen_client.parse_nav_command, see
        qwen_prompts.build_parse_nav_command_prompt) -- consumed next tick by
        _env_loop, which dispatches the VLM call on a background thread
        (_dispatch_nav_command) using that tick's live frame. The result
        becomes a new Mission (nav.mission.Mission, nav.mission.parse_parts:
        every direction first as an in-place TURN, then every goal as
        GO_TO/RETURN), picked up by _env_loop's mission stepper
        (_start_mission_subgoal) once parsing completes: GO_TO/FIND ground
        the named target via dino_grounding_resolver and auto-confirm (no
        review pause -- a human isn't supervising each step of a multi-part
        instruction), RETURN drives to the spawn point, TURN rotates in
        place. Advances on each step's ground-truth goal_reached signal,
        same as a manually driven point/resolve goal."""
        with self._lock:
            self._pending_nav_command_text = text

    def stop_driving(self) -> None:
        with self._lock:
            self._mode = MODE_IDLE
            self._world_goal = None
            self._manual_action = Action(0.0, 0.0, 0.0)
            self._mission = None

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
        services = [qwen_manager]
        # Uncertainty-halt reuses this same qwen_server (see this module's
        # import comment) rather than spawning vl_direction's own -- no
        # second service to start here, just a client pointed at the
        # sam_vla.vlm qwen_config host/port.
        uncertainty_client = None
        if self.uncertainty_enabled:
            uncertainty_client = VlDirectionQwenClient(
                host=QWEN_SERVER_HOST, port=QWEN_SERVER_PORT
            )
        try:
            with MarsHabitatEnv(
                self.scene_path,
                self.heightmap_path,
                services=services,
                start_x=self.start_x,
                start_z=self.start_z,
                start_yaw=math.radians(self.start_yaw_deg),
                with_semantic=True,
                rock_field_path=self.rock_field_path,
                flag_seed=self.flag_seed,
                num_flags=self.num_flags,
                flag_min_spacing=self.flag_min_spacing,
                flag_boundary_margin=self.flag_boundary_margin,
                annotations_dir=self.annotations_dir,
                annotation_categories=self.annotation_categories,
            ) as env:
                self._env_loop(env, uncertainty_client)
        except Exception as exc:  # pragma: no cover - surfaced to the GUI, not raised
            traceback.print_exc()
            with self._lock:
                self.display.status_text = f"FATAL: {exc}"
                self.display.error_text = str(exc)

    def _env_loop(self, env: MarsHabitatEnv, uncertainty_client=None) -> None:
        mask_dir = tempfile.mkdtemp(prefix="mars_nav_masks_")
        obs0 = env.get_observation(frame_idx=0)

        uncertainty_session: Optional[UncertaintySession] = None
        if self.uncertainty_enabled:
            uncertainty_session = UncertaintySession(
                episode_id=f"nav-gui-{int(time.time())}",
                covariance_threshold=self.uncertainty_cov_threshold,
                covariance_value=0.0,
                client=uncertainty_client,
            )

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
            server_variant=self.navdp_upstream_server_variant,
            planner_mode=self.navdp_upstream_planner_mode,
        )
        with self._lock:
            self.display.status_text = "NavDP policy is being loaded"
        policy.start()

        belief_tracker = self._new_belief_tracker()
        # Persistent per-goal-text belief store for an active Mission (nav.mission.
        # Mission) -- distinct from belief_tracker above, which stays the ad-hoc
        # single-target tracker for Ground Target / plain Segment / click-to-goal
        # (self._mission is None in all of those). While a Mission is active,
        # belief_tracker is reassigned to alias goal_beliefs[current sub-goal's
        # target] (see the _start_mission_subgoal call sites below) so the rest of
        # this loop's MODE_RESOLVE handling needs no changes -- it just always
        # operates on "whichever tracker is active". See _sweep_goal_beliefs for
        # how entries other than the active one get seeded/refreshed.
        goal_beliefs: dict[str, BeliefGoalTracker] = {}
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
                    do_clear_all,
                    pixel_click,
                    do_confirm_seg,
                    do_rerun_seg,
                    do_pick_manual,
                    uncertainty_heading,
                    do_uncertainty_retry,
                    nav_command_text,
                    resolve_target_text,
                ) = self._consume_pending()

                if do_reset or do_clear_all:
                    # Shared by request_reset (do_reset) and clear_all_goals
                    # (do_clear_all, the Escape-key hard clear) -- everything
                    # below except the env.step teleport back to spawn, which
                    # only do_reset wants (clear_all_goals leaves the rover
                    # exactly where it is).
                    goal_objects, obstacle_objects = self._clear_masks(
                        env, goal_objects, obstacle_objects
                    )
                    with self._lock:
                        self._resolve_target_text = None
                        self._mission = None
                        self._pending_mission = None
                    self._mission_target_yaw = None
                    belief_tracker = self._new_belief_tracker()
                    goal_beliefs = {}
                    if avoidance is not None:
                        avoidance = self._new_avoidance()
                    if do_reset:
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
                        self._uncertainty_halted = False
                        self._uncertainty_search_goal = None
                        self.display.status_text = (
                            "reset to spawn" if do_reset else "cleared -- idle"
                        )
                        self.display.goal_reached = False
                        self.display.click_status = ""
                        self.display.uncertainty_halted = False
                        self.display.uncertainty_searching = False
                        self.display.uncertainty_line = ""

                if do_resolve or do_rerun_seg:
                    goal_objects, obstacle_objects, goal_spec = self._do_resolve(
                        env,
                        step,
                        mask_dir,
                        goal_objects,
                        obstacle_objects,
                        goal_spec,
                        target_text=resolve_target_text,
                    )
                    belief_tracker.belief_g = None
                    with self._lock:
                        self._uncertainty_halted = False
                        self._uncertainty_search_goal = None
                        self.display.uncertainty_halted = False
                        self.display.uncertainty_searching = False

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

                # A Mission finished parsing (background thread from a prior
                # tick's _dispatch_nav_command, see submit_nav_command) --
                # pick it up and start its first sub-goal.
                with self._lock:
                    pending_mission = self._pending_mission
                    self._pending_mission = None
                if pending_mission is not None:
                    with self._lock:
                        self._mission = pending_mission
                    # Record the rover's spawn point as a "home base" entry in
                    # the persistent belief bank (goal_beliefs) the moment a
                    # mission is dispatched, keyed the same way _sweep_goal_beliefs
                    # keys everything else -- so home is remembered right from
                    # the prompt, not only once a RETURN sub-goal is reached.
                    # This is bookkeeping only: RETURN's own drive (below) still
                    # goes straight to the ground-truth (self.start_x, self.start_z)
                    # every tick via MODE_POINT, which is exact and never needs
                    # correcting -- this entry just makes home visible in the
                    # same belief bank flags/other goals live in.
                    home_forward, home_left = body_frame_goal(
                        self.display.pose, (self.start_x, self.start_z)
                    )
                    goal_beliefs.setdefault(
                        "home base", self._new_belief_tracker()
                    ).observe_body_point(home_forward, home_left)
                    self._event_log.log(
                        "home_base_seeded",
                        world=(round(self.start_x, 2), round(self.start_z, 2)),
                    )
                    goal_objects, obstacle_objects, goal_spec, belief_tracker = (
                        self._start_mission_subgoal(
                            env,
                            step,
                            mask_dir,
                            goal_objects,
                            obstacle_objects,
                            goal_spec,
                            goal_beliefs,
                            belief_tracker,
                        )
                    )

                obs = env.get_observation(frame_idx=step)
                semantic = env.get_semantic_frame()
                goal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
                obstacle_mask = (semantic == MESH_OBST_ID).astype("uint8") * 255

                with self._lock:
                    mission_snapshot = self._mission
                if (
                    mission_snapshot is not None
                    and self.mission_belief_sweep_every > 0
                    and step % self.mission_belief_sweep_every == 0
                ):
                    self._sweep_goal_beliefs(
                        mission_snapshot, obs.rgb, obs.depth, goal_beliefs
                    )
                    with self._lock:
                        turning_now = self._mode == MODE_TURN
                    if turning_now and self._check_turn_interrupt(
                        mission_snapshot, goal_beliefs
                    ):
                        goal_objects, obstacle_objects, goal_spec, belief_tracker = (
                            self._start_mission_subgoal(
                                env,
                                step,
                                mask_dir,
                                goal_objects,
                                obstacle_objects,
                                goal_spec,
                                goal_beliefs,
                                belief_tracker,
                            )
                        )

                if pixel_click is not None:
                    self._handle_pixel_click(obs, pixel_click)

                if nav_command_text is not None:
                    self._dispatch_nav_command(nav_command_text, obs.rgb)

                if uncertainty_session is not None:
                    self._handle_uncertainty_commands(
                        uncertainty_session,
                        belief_tracker,
                        obs,
                        uncertainty_heading,
                        do_uncertainty_retry,
                    )

                with self._lock:
                    mode = self._mode
                    manual_action = self._manual_action
                    world_goal = self._world_goal

                if mode != MODE_RESOLVE:
                    with self._lock:
                        if (
                            self._uncertainty_halted
                            or self._uncertainty_search_goal is not None
                        ):
                            self._uncertainty_halted = False
                            self._uncertainty_search_goal = None
                            self.display.uncertainty_halted = False
                            self.display.uncertainty_searching = False

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
                turn_ghost_body: Optional[tuple[float, float]] = None

                if self._uncertainty_halted:
                    # Frozen awaiting a human heading -- same "ignore drive
                    # commands while halted" contract kb_teleop_vl.py's
                    # uncertainty halt uses, just enforced here instead of at
                    # the input layer (nav/gui.py never blocks its own D-pad).
                    goal_bearing = belief_tracker.bearing()
                elif mode == MODE_MANUAL:
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
                        self._event_log.log(
                            "goal_reached", kind="point", world=world_goal
                        )
                    else:
                        policy.set_goal_body(forward, left)
                        action, _vla_result = policy.act_verbose(
                            obs, semantic, goal_spec, step
                        )
                        trajectory = getattr(policy, "_last_trajectory", None)
                elif mode == MODE_RESOLVE:
                    seen = belief_tracker.observe(goal_mask, obs.depth)
                    if not seen and self._ground_target_world is not None:
                        # DINO-grounded targets never actually re-seed belief
                        # via observe() above: register_object_mask's MESH_GOAL_ID
                        # object renders 0px on this GPU/driver (see CLAUDE.md's
                        # dynamic-object-render-bug note), so goal_mask is always
                        # empty and uncertainty grows unbounded even while the
                        # ghost mask is visibly on-screen. Fall back to seeding
                        # belief straight from the ground-truth world position
                        # (same mechanism MODE_POINT's observe_body_point uses),
                        # gated on the exact same in-frame test project_world_to_pixel
                        # runs below to decide whether to draw the ghost mask at
                        # all -- so belief resets precisely when the ghost mask
                        # is showing, not on some independent visibility notion.
                        gx, gy, gz = self._ground_target_world
                        if (
                            project_world_to_pixel(
                                obs.pose,
                                (gx, gy, gz),
                                HFOV_DEG,
                                obs.rgb.shape[1],
                                obs.rgb.shape[0],
                            )
                            is not None
                        ):
                            forward, left = body_frame_goal(obs.pose, (gx, gz))
                            belief_tracker.observe_body_point(forward, left)
                            seen = True
                    if seen and self._uncertainty_search_goal is not None:
                        # Goal re-sighted mid-search -- observe() above
                        # already reset uncertainty_value() to the sighted
                        # floor, so abandon the human-chosen heading and
                        # resume tracking the (now freshly re-anchored)
                        # belief directly.
                        self._uncertainty_search_goal = None
                        with self._lock:
                            self.display.uncertainty_searching = False
                    if belief_tracker.belief_g is not None:
                        forward, left = (float(v) for v in belief_tracker.belief_g)
                        searching = self._uncertainty_search_goal is not None
                        if searching:
                            search_forward, search_left = body_frame_goal(
                                obs.pose, self._uncertainty_search_goal
                            )
                            if (
                                math.hypot(search_forward, search_left)
                                < POINT_GOAL_REACHED_M
                            ):
                                # Reached the searched heading without a
                                # re-sighting -- fall back to the (still
                                # uncertain) dead-reckoned belief; the halt
                                # trigger below will ask again since
                                # uncertainty_value() is still >= threshold.
                                self._uncertainty_search_goal = None
                                searching = False
                                with self._lock:
                                    self.display.uncertainty_searching = False
                            else:
                                forward, left = search_forward, search_left
                        # "Reached" detection (Qwen touch-bottom query) was
                        # removed; this is the ground-truth-adjacent
                        # replacement -- same distance threshold MODE_POINT
                        # uses (POINT_GOAL_REACHED_M), checked against
                        # belief_tracker.distance() (the belief-tracked
                        # position, since a resolved/grounded goal never has
                        # a ground-truth one) rather than the possibly
                        # search-heading-overridden local forward/left.
                        # Gated on not-searching: the search sub-flow's own
                        # heading-reached fallback above is a different
                        # mechanic -- it just resumes belief tracking, it
                        # doesn't complete the goal. Setting goal_reached
                        # here feeds the same self._mission advance-on-
                        # goal_reached edge MODE_POINT/MODE_TURN already
                        # drive, so a mission's GO_TO/FIND sub-goal (and the
                        # belief bank behind it) now advances too.
                        if (
                            not searching
                            and belief_tracker.distance() < POINT_GOAL_REACHED_M
                        ):
                            with self._lock:
                                self._mode = MODE_IDLE
                                self.display.goal_reached = True
                            mode = MODE_IDLE
                            self._event_log.log(
                                "goal_reached",
                                kind="resolve",
                                name=goal_spec.instruction_text,
                            )
                        else:
                            policy.set_goal_body(forward, left)
                            action, _vla_result = policy.act_verbose(
                                obs, semantic, goal_spec, step
                            )
                            trajectory = getattr(policy, "_last_trajectory", None)
                            # CBF's steering hint must track whichever target
                            # is actually being driven to -- the searched
                            # heading while searching, the dead-reckoned
                            # belief otherwise -- not always belief_g's own
                            # (possibly stale) bearing.
                            goal_bearing = math.atan2(left, forward)
                    else:
                        # Never sighted the goal mask yet this episode -- hold
                        # rather than drive on NavdpUpstreamPolicy's hidden
                        # constructor default (see this module's docstring /
                        # next.md's Integration-project Phase 5).
                        unresolved = True
                elif mode == MODE_TURN and self._mission_target_yaw is not None:
                    yaw_err = (self._mission_target_yaw - obs.pose.yaw + math.pi) % (
                        2.0 * math.pi
                    ) - math.pi
                    if abs(yaw_err) < TURN_GOAL_REACHED_RAD:
                        with self._lock:
                            self._mode = MODE_IDLE
                            self.display.goal_reached = True
                        mode = MODE_IDLE
                        self._mission_target_yaw = None
                        self._event_log.log("goal_reached", kind="turn")
                    else:
                        # Ghost point proportional to how much turn is left
                        # -- see TURN_GHOST_* docstring above. Clamping the
                        # bearing (not just the resulting pixel) keeps
                        # forward strictly positive, so this is always a
                        # normal in-front point goal for the policy, never
                        # a behind-the-camera one.
                        turn_bearing = max(
                            -TURN_GHOST_MAX_BEARING_RAD,
                            min(TURN_GHOST_MAX_BEARING_RAD, yaw_err),
                        )
                        forward = TURN_GHOST_LOOKAHEAD_M * math.cos(turn_bearing)
                        left = TURN_GHOST_LOOKAHEAD_M * math.sin(turn_bearing)
                        turn_ghost_body = (forward, left)
                        goal_bearing = turn_bearing
                        policy.set_goal_body(forward, left)
                        action, _vla_result = policy.act_verbose(
                            obs, semantic, goal_spec, step
                        )
                        trajectory = getattr(policy, "_last_trajectory", None)

                if self._mission is not None and self.display.goal_reached:
                    # Ground-truth "reached" edge for the mission's current
                    # sub-goal (same signal a manually driven point/resolve
                    # goal already produces) -- advance and kick off whatever
                    # driving mode the next sub-goal needs. _start_mission_subgoal
                    # clears goal_reached again (or clears self._mission
                    # entirely on GoalKind.DONE), so this only fires once per
                    # sub-goal completion, not every tick after.
                    with self._lock:
                        if self._mission is not None:
                            completed_step = self._mission.current
                            self._event_log.log(
                                "mission_subgoal_reached",
                                step=completed_step.raw,
                                kind=completed_step.kind.name,
                            )
                            self._mission.advance()
                    goal_objects, obstacle_objects, goal_spec, belief_tracker = (
                        self._start_mission_subgoal(
                            env,
                            step,
                            mask_dir,
                            goal_objects,
                            obstacle_objects,
                            goal_spec,
                            goal_beliefs,
                            belief_tracker,
                        )
                    )

                obstacle_point = None
                cbf_info: dict = {}
                if (
                    avoidance is not None
                    and mode != MODE_MANUAL
                    and not self._uncertainty_halted
                ):
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

                if mode in (MODE_POINT, MODE_RESOLVE) and self._mission is None:
                    belief_tracker.propagate(action, self.dt)

                if self._mission is not None:
                    # Every tracked goal belief (not just the active one) dead-
                    # reckons against the rover's actual executed motion this tick,
                    # regardless of which mode that motion happened under -- a
                    # MODE_TURN/MODE_MANUAL leg still moves the rover, and a
                    # previously-sighted goal's remembered bearing must still
                    # decay/rotate with it. belief_tracker (the active goal's
                    # tracker, aliased into goal_beliefs by _start_mission_subgoal)
                    # is one of these entries, so it's propagated here instead of
                    # the ad-hoc-only line above.
                    for bt in goal_beliefs.values():
                        bt.propagate(action, self.dt)

                if (
                    uncertainty_session is not None
                    and mode == MODE_RESOLVE
                    and not self._uncertainty_halted
                    and not self._uncertainty_request_in_flight
                    and self._uncertainty_search_goal is None
                    and belief_tracker.belief_g is not None
                    and belief_tracker.uncertainty_value()
                    >= self.uncertainty_cov_threshold
                ):
                    self._uncertainty_halted = True
                    with self._lock:
                        self.display.uncertainty_halted = True
                    self._dispatch_uncertainty_request(
                        uncertainty_session, belief_tracker, obs.rgb, retry=False
                    )

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
                if self._ground_target_world is not None:
                    # Reprojected every frame (not baked in once) so the
                    # ghost mask tracks correctly as the rover moves -- only
                    # drawn when the target's fixed world position actually
                    # reprojects into this frame (project_world_point_with_pixel_radius
                    # returns None otherwise), i.e. only ever shown over what
                    # can currently be seen, never extrapolated off-screen.
                    ground_proj = project_world_point_with_pixel_radius(
                        obs.pose,
                        self._ground_target_world,
                        GHOST_TARGET_RADIUS_M,
                        HFOV_DEG,
                        obs.rgb.shape[1],
                        obs.rgb.shape[0],
                    )
                    if ground_proj is not None:
                        gu, gv, gr = ground_proj
                        vis_rgb = draw_ghost_mask(vis_rgb, gu, gv, gr)
                if turn_ghost_body is not None:
                    # Same single-horizontal-line projection MODE_TURN's
                    # driving goal was built from -- forward > 0 is
                    # guaranteed by the bearing clamp above, so this always
                    # resolves to a real pixel, never None in practice.
                    turn_pixel = project_body_point_to_pixel(
                        turn_ghost_body[0],
                        turn_ghost_body[1],
                        HFOV_DEG,
                        obs.rgb.shape[0],
                        obs.rgb.shape[1],
                    )
                    if turn_pixel is not None:
                        tu, tv = turn_pixel
                        vis_rgb = draw_ghost_mask(vis_rgb, tu, tv, TURN_GHOST_RADIUS_PX)

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
                    d.uncertainty_value = belief_tracker.uncertainty_value()
                    d.mission_status = (
                        self._mission.status() if self._mission is not None else ""
                    )

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
    ) -> tuple[
        bool,
        bool,
        bool,
        Optional[tuple],
        bool,
        bool,
        bool,
        Optional[float],
        bool,
        Optional[str],
        Optional[str],
    ]:
        with self._lock:
            do_resolve = self._pending_resolve
            resolve_target_text = self._resolve_target_text
            do_reset = self._pending_reset
            do_clear_all = self._pending_clear_all
            pixel_click = self._pending_pixel_click
            do_confirm_seg = self._pending_confirm_segmentation
            do_rerun_seg = self._pending_rerun_segmentation
            do_pick_manual = self._pending_pick_manually
            uncertainty_heading = self._pending_uncertainty_heading
            do_uncertainty_retry = self._pending_uncertainty_retry
            nav_command_text = self._pending_nav_command_text
            self._pending_resolve = False
            self._pending_reset = False
            self._pending_clear_all = False
            self._pending_pixel_click = None
            self._pending_confirm_segmentation = False
            self._pending_rerun_segmentation = False
            self._pending_pick_manually = False
            self._pending_uncertainty_heading = None
            self._pending_uncertainty_retry = False
            self._pending_nav_command_text = None
        return (
            do_resolve,
            do_reset,
            do_clear_all,
            pixel_click,
            do_confirm_seg,
            do_rerun_seg,
            do_pick_manual,
            uncertainty_heading,
            do_uncertainty_retry,
            nav_command_text,
            resolve_target_text,
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

    def _handle_uncertainty_commands(
        self,
        session: UncertaintySession,
        belief_tracker: BeliefGoalTracker,
        obs,
        heading_deg: Optional[float],
        do_retry: bool,
    ) -> None:
        """Applies a queued retry/heading-submit command against an active
        uncertainty halt (see _env_loop's halt-trigger block). Both are
        no-ops if not currently halted, or a request is already in flight --
        guards a stale GUI click racing a halt that's since been cleared by
        a fresh sighting or a mode switch (see the `mode != MODE_RESOLVE`
        cleanup in _env_loop)."""
        if (
            do_retry
            and self._uncertainty_halted
            and not self._uncertainty_request_in_flight
        ):
            self._dispatch_uncertainty_request(
                session, belief_tracker, obs.rgb, retry=True
            )

        if (
            heading_deg is None
            or not self._uncertainty_halted
            or self._uncertainty_request_in_flight
        ):
            return

        # Pure local packaging, no VLM call (directive_engine._query_uncertainty's
        # "submit phase") -- safe to resolve synchronously on this thread.
        result = session.submit_heading(angle_deg=heading_deg)
        payload = result.uncertainty_payload
        self._uncertainty_search_goal = heading_ahead_point(
            obs.pose, heading_deg, self.uncertainty_search_dist, self.world_limit
        )
        self._uncertainty_halted = False
        with self._lock:
            self.display.uncertainty_halted = False
            self.display.uncertainty_searching = True
            self.display.uncertainty_line = (
                f"heading {heading_deg:+.0f}deg -> traverse up to "
                f"{payload.max_units:.1f} units, or until the goal is re-sighted"
            )

    def _dispatch_uncertainty_request(
        self,
        session: UncertaintySession,
        belief_tracker: BeliefGoalTracker,
        frame: np.ndarray,
        retry: bool,
    ) -> None:
        """Kicks off the (slow) VLM sweep-description call on a background
        thread -- the control loop above must keep ticking at self.hz
        regardless of inference latency, same reasoning as
        NavdpUpstreamPolicy's own async replanning. session.covariance_value
        is refreshed here (kb_teleop_vl.py's version leaves it fixed at
        construction) so the sweep-description prompt reflects the actual
        uncertainty that crossed the threshold, not a stale value."""
        session.covariance_value = belief_tracker.uncertainty_value()
        self._uncertainty_request_in_flight = True
        with self._lock:
            self.display.uncertainty_request_in_flight = True
            self.display.uncertainty_line = (
                "retrying VLM sweep description..."
                if retry
                else "requesting VLM sweep description..."
            )
        target = session.retry if retry else session.request_human_heading
        thread = threading.Thread(
            target=self._uncertainty_worker, args=(target, frame), daemon=True
        )
        thread.start()

    def _uncertainty_worker(self, session_method, frame: np.ndarray) -> None:
        """Runs on a background thread -- the actual sweep-description VLM
        call. Never touches habitat-sim; only writes self.display under
        self._lock, same cross-thread pattern the rest of this controller
        uses for its snapshot() contract."""
        try:
            result = session_method(frame)
            payload = result.uncertainty_payload
            line = (
                f"attempt {payload.attempt}: {result.raw_response!r} -- "
                "pick a heading, or Retry"
            )
        except Exception as exc:
            line = f"uncertainty request failed: {exc}"
        with self._lock:
            self._uncertainty_request_in_flight = False
            self.display.uncertainty_request_in_flight = False
            self.display.uncertainty_line = line

    def _dispatch_nav_command(self, text: str, frame: np.ndarray) -> None:
        """Kicks off the (slow) nav-command-parsing VLM call on a background
        thread -- same reasoning as _dispatch_uncertainty_request, the
        control loop must keep ticking at self.hz regardless of inference
        latency."""
        thread = threading.Thread(
            target=self._nav_command_worker, args=(text, frame), daemon=True
        )
        thread.start()

    def _nav_command_worker(self, text: str, frame: np.ndarray) -> None:
        """Runs on a background thread -- splits a free-text nav command
        into (directions, goals) via qwen_client.parse_nav_command (see
        qwen_prompts.build_parse_nav_command_prompt), then turns that split
        into a Mission (nav.mission.Mission/parse_parts: every direction
        first as an in-place TURN, then every goal as GO_TO/RETURN). Never
        touches habitat-sim; only writes self._pending_mission under
        self._lock, picked up next tick by _env_loop."""
        try:
            directions, goals = qwen_client.parse_nav_command(frame, text)
            print(f"[nav command] {text!r} -> directions={directions} goals={goals}")
            mission = Mission(text, directions=directions, goal_texts=goals)
        except Exception as exc:
            print(f"[nav command] {text!r} failed: {exc}")
            return
        self._event_log.log(
            "mission_started", text=text, directions=directions, goals=goals
        )
        with self._lock:
            self._pending_mission = mission

    def _start_mission_subgoal(
        self,
        env,
        step,
        mask_dir,
        goal_objects,
        obstacle_objects,
        goal_spec,
        goal_beliefs,
        belief_tracker,
    ):
        """Kicks off whichever driving mode the active Mission's current
        sub-goal needs (see nav.mission.Mission) -- called once when a new
        Mission is queued (submit_nav_command) and again every time the
        mission-advance edge in _env_loop fires. Only ever called from the
        controller thread. A GO_TO/FIND resolve that fails (DINO doesn't
        ground the target in the frame this step starts on) no longer means
        the step is abandoned outright -- if a previous sighting (this
        step's own earlier attempt, or _sweep_goal_beliefs picking it up
        while an earlier leg of the mission was driving) left a confident
        belief for this exact target text in `goal_beliefs`, resume driving
        toward that remembered position instead. Only truly skipped (as
        before this feature existed) when neither a fresh resolve nor a
        usable prior belief is available.

        `goal_beliefs` is the persistent dict[goal-text -> BeliefGoalTracker]
        for the whole Mission (see _env_loop) -- entries survive across
        sub-goals so a target sighted while driving toward an earlier one
        isn't forgotten by the time the mission reaches it. `belief_tracker`
        is the CURRENTLY active tracker (whichever one _env_loop's tick loop
        is calling propagate()/observe() on); RETURN/TURN/DONE have no goal
        text of their own and just pass it through unchanged. Both are
        returned so the caller can rebind its own locals, same pattern
        goal_spec already uses."""
        while True:
            with self._lock:
                mission = self._mission
                goal = mission.current if mission is not None else None
            if goal is None:
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind == GoalKind.DONE:
                with self._lock:
                    self._mode = MODE_IDLE
                    self._world_goal = None
                    self._mission = None
                self._mission_target_yaw = None
                self._event_log.log("mission_complete")
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind in (GoalKind.GO_TO, GoalKind.FIND):
                # Single front-facing DINO resolve on whatever's already in
                # the current frame -- no 360 sweep. sweep_and_seed_beliefs
                # (dino_grounding_resolver) rotated through 8 discrete
                # headings with no cross-frame consistency check on what it
                # found, which let a single high-scoring false positive
                # (e.g. a blue cuboid grounding "flag") pick the bearing the
                # rover then turned to face and re-resolved against. Disabled
                # for now -- see this session's flag-misdetection debugging.
                # (_sweep_goal_beliefs below is a different, lower-stakes use
                # of the same detector -- it only ever seeds a passive belief,
                # never itself re-orients the rover.)
                goal_objects, obstacle_objects, goal_spec = self._do_resolve(
                    env,
                    step,
                    mask_dir,
                    goal_objects,
                    obstacle_objects,
                    goal_spec,
                    target_text=goal.target,
                )
                with self._lock:
                    # _do_resolve always leaves a *successful* resolve in
                    # MODE_REVIEW_SEGMENTATION with an empty click_status
                    # (goal_mask_warning) -- see its docstring. A mission
                    # step auto-confirms straight through that review pause
                    # instead of waiting for a human Confirm click.
                    resolved = (
                        self._mode == MODE_REVIEW_SEGMENTATION
                        and not self.display.click_status
                    )
                    if resolved:
                        self._mode = MODE_RESOLVE
                        self.display.status_text = (
                            f"[mission] resolved: {goal_spec.instruction_text}"
                        )
                if resolved:
                    # Fresh sighting -- (re)anchor this goal's persistent
                    # belief and discard whatever it held before: belief_g is
                    # reset to None so MODE_RESOLVE's own per-tick fallback
                    # (self._ground_target_world -> body_frame_goal ->
                    # observe_body_point, see the MODE_RESOLVE branch below)
                    # re-seeds it from the same ground_target_world
                    # _do_resolve just set, as soon as the target is next
                    # in-frame -- same reset _do_resolve's ad-hoc caller
                    # already relies on for its own single shared tracker.
                    belief_tracker = goal_beliefs.setdefault(
                        goal.target, self._new_belief_tracker()
                    )
                    belief_tracker.belief_g = None
                    self._activate_belief_tracker(belief_tracker)
                    return goal_objects, obstacle_objects, goal_spec, belief_tracker

                prior = goal_beliefs.get(goal.target)
                if (
                    prior is not None
                    and prior.belief_g is not None
                    and prior.uncertainty_value() < self.uncertainty_cov_threshold
                ):
                    # Not in the frame this step started on, but a previous
                    # sighting (an earlier leg's periodic _sweep_goal_beliefs
                    # hit, most likely) left a still-confident belief for this
                    # exact target -- resume driving toward it rather than
                    # abandoning the step. No mask/obstacle registration here
                    # (goal_objects/obstacle_objects are already empty --
                    # _do_resolve's failure path clears them same as always),
                    # and goal_spec content doesn't matter to the driving
                    # policy (NavdpUpstreamPolicy.act_verbose only reads the
                    # body-frame point set via policy.set_goal_body, see this
                    # module's docstring) -- a placeholder is safe, same one
                    # _env_loop starts with before any resolve has happened.
                    with self._lock:
                        self._mode = MODE_RESOLVE
                        self.display.status_text = (
                            f"[mission] resuming toward previously-sighted "
                            f"'{goal.target}'"
                        )
                        self.display.goal_reached = False
                    placeholder_spec = GoalSpec(
                        goal_bbox_norm=(0.0, 0.0, 1.0, 1.0),
                        obstacle_bboxes_norm=[],
                        instruction_text=(
                            f"Navigate to the {goal.target} (from memory)."
                        ),
                    )
                    self._activate_belief_tracker(prior)
                    return goal_objects, obstacle_objects, placeholder_spec, prior

                print(
                    f"[mission] step {goal.raw!r} failed to resolve and no "
                    "usable prior belief -- skipping"
                )
                self._event_log.log(
                    "mission_subgoal_skipped", step=goal.raw, target=goal.target
                )
                with self._lock:
                    if self._mission is mission:
                        self._mission.advance()
                continue

            if goal.kind == GoalKind.RETURN:
                with self._lock:
                    self._world_goal = (self.start_x, self.start_z)
                    self._mode = MODE_POINT
                    self.display.goal_reached = False
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind == GoalKind.TURN:
                pose = self.display.pose
                heading_deg = {
                    "right": 90.0,
                    "left": -90.0,
                    "back": 180.0,
                    "around": 180.0,
                }.get(goal.target, 180.0)
                self._mission_target_yaw = pose.yaw - math.radians(heading_deg)
                with self._lock:
                    self._mode = MODE_TURN
                    self.display.goal_reached = False
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind == GoalKind.ADVANCE:
                # "go left"/"go right"/"go straight" -- unlike TURN (rotate
                # only), this actually drives: a world-frame point
                # go_direction_distance_m straight ahead along whatever
                # heading the rover is CURRENTLY facing (the preceding TURN
                # sub-goal, if any, has already finished by the time this
                # runs -- see nav.mission._direction_subgoals). Same
                # ground-truth MODE_POINT mechanism RETURN/random-goal
                # driving already use (CBF obstacle avoidance included), just
                # aimed at a point instead of the spawn.
                pose = self.display.pose
                dist = self.go_direction_distance_m
                tx = pose.x - dist * math.sin(pose.yaw)
                tz = pose.z - dist * math.cos(pose.yaw)
                with self._lock:
                    self._world_goal = (tx, tz)
                    self._mode = MODE_POINT
                    self.display.goal_reached = False
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

    def _sweep_goal_beliefs(self, mission, rgb, depth, goal_beliefs) -> None:
        """Opportunistic multi-goal check against the current frame while a
        Mission is driving normally (no rotation, unlike
        dino_grounding_resolver.sweep_and_seed_beliefs's 360deg scan) --
        called every self.mission_belief_sweep_every ticks from _env_loop.
        Checks every GO_TO/FIND target from the mission's CURRENT step
        onward (already-completed steps aren't worth the DINO call) against
        this one frame, and seeds/refreshes goal_beliefs for whichever ones
        hit -- including the currently active goal: its own live-mask/
        ground-truth-anchor path (self._ground_target_world) already
        re-derives it every tick when in view, so a sweep hit on it is either
        redundant (overwritten again next tick) or, when that anchor is
        itself out of view, the only correction available -- both cases want
        inclusion, not exclusion. This never changes self._mode/drives the
        rover by itself; it only ever seeds a passive belief later consumed
        (or not) by _start_mission_subgoal."""
        queries = {
            g.target
            for g in mission.goals[mission.idx :]
            if g.kind in (GoalKind.GO_TO, GoalKind.FIND)
        }
        if not queries:
            return
        hits = dino_grounding_resolver.detect_in_frame(
            rgb,
            depth,
            list(queries),
            HFOV_DEG,
            dino_model_id=self.dino_model_id,
            dino_device=self.dino_device,
            dino_box_threshold=self.dino_box_threshold,
            dino_text_threshold=self.dino_text_threshold,
        )
        for query, (forward, left, score) in hits.items():
            goal_beliefs.setdefault(
                query, self._new_belief_tracker(mission_sweep=True)
            ).observe_body_point(forward, left)
            # Silent before this: a sweep hit only ever showed up indirectly, as
            # whatever _start_mission_subgoal later did with the belief it seeded
            # -- logged explicitly so a periodic sweep is actually observable/
            # debuggable from the event log instead of inferred after the fact.
            # score is DINO's own box confidence for this query's best box --
            # logged so a confident cross-class mixup (e.g. "green flag"
            # grounding onto the white flag once the real one leaves frame)
            # is distinguishable from a low-confidence noise hit.
            self._event_log.log(
                "mission_sweep_hit",
                name=query,
                forward=round(float(forward), 2),
                left=round(float(left), 2),
                score=round(float(score), 3),
            )

    def _check_turn_interrupt(self, mission, goal_beliefs) -> bool:
        """While a mission TURN sub-goal is spinning, don't finish the spin
        (or any ADVANCE step after it) if the goal it was turning toward is
        already confidently sighted -- called right after _sweep_goal_beliefs
        so it's reacting to the same detection, not an extra DINO call of its
        own. Only ever considers the NEAREST upcoming GO_TO/FIND sub-goal
        (the one this turn exists to face), same as a person stopping mid-
        turn the moment they spot what they were looking for. Jumps the
        mission straight to that sub-goal (skipping any ADVANCE in between --
        no point blindly driving forward first when the target is already in
        view) and lets the caller kick off _start_mission_subgoal, whose own
        GO_TO/FIND handling picks the belief seeded here right back up (fresh
        resolve, or this prior belief as fallback -- see its docstring).
        Returns whether an interrupt happened."""
        upcoming = next(
            (
                g
                for g in mission.goals[mission.idx :]
                if g.kind in (GoalKind.GO_TO, GoalKind.FIND)
            ),
            None,
        )
        if upcoming is None:
            return False
        bt = goal_beliefs.get(upcoming.target)
        if (
            bt is None
            or bt.belief_g is None
            or bt.uncertainty_value() >= self.uncertainty_cov_threshold
        ):
            return False
        with self._lock:
            if self._mission is not mission:
                return False
            while self._mission.current is not upcoming:
                self._mission.advance()
            self._mission_target_yaw = None
        self._event_log.log("turn_interrupted_goal_sighted", target=upcoming.target)
        return True

    def _new_belief_tracker(self, *, mission_sweep: bool = False) -> BeliefGoalTracker:
        # odom_noise/odom_noise_growth_rate drive uncertainty_value()'s real
        # growth while the goal mask is unseen (sam_vla.core.belief_tracking's
        # accelerating-drift formula) -- what the uncertainty halt below
        # compares against uncertainty_cov_threshold. Zero unless the halt is
        # enabled, so a disabled run doesn't silently start dead-reckoning
        # noisier than before.
        #
        # mission_sweep=True is for trackers _sweep_goal_beliefs seeds passively
        # for a goal not currently being driven to -- see mission_belief_cov_growth
        # above for why these need a much slower rate than the actively-driven
        # tracker's halt-tuned one. Callers that promote such a tracker to actively
        # driven must switch it back via _activate_belief_tracker.
        if mission_sweep:
            odom_noise = (
                self.mission_belief_cov_growth if self.uncertainty_enabled else 0.0
            )
            odom_noise_growth_rate = (
                self.mission_belief_cov_growth_rate if self.uncertainty_enabled else 0.0
            )
        else:
            odom_noise = self.uncertainty_cov_growth if self.uncertainty_enabled else 0.0
            odom_noise_growth_rate = (
                self.uncertainty_cov_growth_rate if self.uncertainty_enabled else 0.0
            )
        return BeliefGoalTracker(
            hfov_deg=HFOV_DEG,
            goal_range=self.belief_goal_range,
            min_px=self.lost_goal_min_px,
            odom_noise=odom_noise,
            odom_noise_growth_rate=odom_noise_growth_rate,
        )

    def _activate_belief_tracker(self, bt: BeliefGoalTracker) -> None:
        """Switch a belief tracker to the tight, halt-tuned growth rate at the
        moment it becomes the actively-driven belief_tracker (see
        _start_mission_subgoal's two return points below) -- undoes the loose
        mission_sweep=True rate a passively-seeded tracker may have started
        with, so the uncertainty halt still fires on its normal schedule once
        the rover is actually depending on this belief to drive."""
        bt.odom_noise = self.uncertainty_cov_growth if self.uncertainty_enabled else 0.0
        bt.odom_noise_growth_rate = (
            self.uncertainty_cov_growth_rate if self.uncertainty_enabled else 0.0
        )

    def _new_avoidance(self) -> CbfObstacleAvoidance:
        return CbfObstacleAvoidance(**self._cbf_kwargs)

    def _clear_masks(self, env, goal_objects: list, obstacle_objects: list) -> tuple:
        """Actually remove every registered goal/obstacle mask mesh from the
        scene (not just untag it) and return the emptied (goal, obstacle)
        lists -- see _env_loop's goal_objects/obstacle_objects docstring.
        Also drops any ground-target ghost-mask overlay (self._ground_target_world,
        see _do_resolve) -- every call site here already means "this resolve's
        state is being invalidated", the same boundary the ghost overlay
        should disappear at."""
        self._ground_target_world = None
        for obj in goal_objects + obstacle_objects:
            env.remove_object_mask(obj)
        return [], []

    def _do_resolve(
        self,
        env,
        step,
        mask_dir,
        goal_objects,
        obstacle_objects,
        current_goal_spec,
        target_text: Optional[str] = None,
    ):
        # A new detection task (DINO-grounded or plain SAM2 "Segment")
        # invalidates whatever the previous one left registered -- drop it
        # up front, before running the new detector, rather than waiting for
        # a successful result to overwrite it. This also means a FAILED new
        # resolve (below) now leaves the rover with no masks at all instead
        # of the stale previous target, which used to survive a failed
        # resolve deliberately; that tradeoff was intentionally accepted so
        # "new task requested" always reads as "old detections gone", even
        # when the new one doesn't pan out.
        goal_objects, obstacle_objects = self._clear_masks(
            env, goal_objects, obstacle_objects
        )

        obs_r = env.get_observation(frame_idx=step)
        # detect_rgb (if any) is a separate frame with the mesh_tight_bound2
        # annotation hulls composited in, fed only to the segmentation
        # model -- obs_r.rgb (the plain camera frame) is what the VLM sees
        # and what the GUI/backprojection use, unmodified.
        detect_rgb = env.get_mesh_overlay_rgb() if self.seg_overlay == "mesh" else None
        try:
            if target_text:
                # Open-vocabulary target (flag, blue cuboid, ...):
                # GroundingDINO grounds it directly, SAM2 only seeds
                # obstacles. See dino_grounding_resolver's module docstring
                # for why this is a separate path from first_frame_resolver
                # rather than a branch inside it.
                goal_spec_r, vlm_result, dets = dino_grounding_resolver.resolve_verbose(
                    obs_r.rgb,
                    target_text,
                    detect_rgb=detect_rgb,
                    backend=self.seg_backend,
                    checkpoint_path=self.seg_checkpoint,
                    dino_model_id=self.dino_model_id,
                    dino_device=self.dino_device,
                    dino_box_threshold=self.dino_box_threshold,
                    dino_text_threshold=self.dino_text_threshold,
                )
            else:
                goal_spec_r, vlm_result, dets = first_frame_resolver.resolve_verbose(
                    obs_r.rgb,
                    detect_rgb=detect_rgb,
                    backend=self.seg_backend,
                    checkpoint_path=self.seg_checkpoint,
                )
        except Exception as exc:
            # Console print too, not just the status banner -- a
            # dino_grounding_resolver "not found" RuntimeError carries
            # GroundingDINO's own reasoning (see its message), which was
            # previously only visible by reading the (often truncated)
            # sidebar status text.
            print(f"[nav resolve] failed: {exc}")
            self._event_log.log(
                "goal_detect_failed", name=target_text, reason=str(exc)
            )
            with self._lock:
                self.display.status_text = f"resolve failed: {exc}"
            # goal_objects/obstacle_objects are already emptied (cleared
            # above, before this detector call) -- only goal_spec falls back
            # to whatever was previously resolved, since there's no new one
            # to replace it with.
            return goal_objects, obstacle_objects, current_goal_spec

        # resolve_verbose's raw VLM result/detections were previously discarded
        # here -- surfaced now so a resolve can actually be audited from the
        # terminal (which detection the VLM picked as goal_index, and whether
        # that specific bbox's depth backprojection is what silently dropped
        # the goal mask below, as opposed to the VLM never picking one).
        if target_text:
            # dino_grounding_resolver's result shape (found/u/v/score/reasoning),
            # not select_goal's (goal_index) -- dets here are SAM2's
            # obstacle-only detections (see dino_grounding_resolver), not
            # goal candidates, so labeled accordingly to avoid implying they
            # were candidates DINO picked among.
            print(
                f"[nav resolve] dino ground_object(target={target_text!r}): "
                f"found={vlm_result.get('found')}, "
                f"u={vlm_result.get('u')}, v={vlm_result.get('v')}, "
                f"score={vlm_result.get('score')}, "
                f"reasoning={vlm_result.get('reasoning', '')!r}, "
                f"{len(dets)} SAM2 obstacle detection(s)"
            )
        else:
            print(
                f"[nav resolve] {len(dets)} detection(s), "
                f"goal_index={vlm_result.get('goal_index')}, "
                f"reasoning={vlm_result.get('reasoning', '')!r}"
            )

        goal_position = backproject_goal_position(obs_r, goal_spec_r, hfov_deg=HFOV_DEG)
        goal_bbox_norm = goal_spec_r.goal_bbox_norm
        obstacle_bboxes_norm = list(goal_spec_r.obstacle_bboxes_norm)
        used_fallback = False
        if goal_position is None:
            # bbox_to_world only returns None when literally every pixel in
            # the *padded* interior patch is non-finite or <= 0 -- dump those
            # same patch stats so a "bad depth" print above is diagnosable
            # (void/sky hit vs. NaN vs. a degenerate 0-area bbox) instead of
            # a bare yes/no. pad_px must match bbox_to_world's own default.
            pad_px = 6
            depth = np.asarray(obs_r.depth)
            h, w = depth.shape[:2]
            x0, y0, x1, y1 = goal_bbox_norm
            ix0, ix1 = sorted(
                (min(max(int(x0 * w), 0), w - 1), min(max(int(x1 * w), 0), w - 1))
            )
            iy0, iy1 = sorted(
                (min(max(int(y0 * h), 0), h - 1), min(max(int(y1 * h), 0), h - 1))
            )
            ix0, ix1 = max(ix0 - pad_px, 0), min(ix1 + pad_px, w - 1)
            iy0, iy1 = max(iy0 - pad_px, 0), min(iy1 + pad_px, h - 1)
            patch = depth[iy0 : iy1 + 1, ix0 : ix1 + 1]
            print(
                f"[nav resolve] goal bbox_norm={goal_bbox_norm} -> "
                f"pixels x[{ix0}:{ix1}] y[{iy0}:{iy1}] of {w}x{h}, "
                f"depth patch min={np.nanmin(patch):.4f} max={np.nanmax(patch):.4f} "
                f"nonfinite={int((~np.isfinite(patch)).sum())}/{patch.size} "
                f"zero_or_neg={int((patch <= 0.0).sum())}/{patch.size}"
            )
            # A whole padded neighborhood reading uniformly invalid means
            # this specific detection isn't backed by real geometry at all
            # (see CLAUDE.md's "Annotation masks are thin silhouette
            # slivers" known issue -- these checkpoints predict artifact
            # detections, not just imprecise ones) -- no amount of local
            # padding fixes that. Rather than leaving the episode with
            # obstacles and no goal, fall through the VLM's other ranked
            # detections (in the order it returned them) and promote the
            # first one that actually backprojects. Only if none of them do
            # does this genuinely have no usable goal this resolve.
            for i, candidate_bbox in enumerate(obstacle_bboxes_norm):
                candidate_position = bbox_to_world(
                    obs_r, candidate_bbox, hfov_deg=HFOV_DEG
                )
                if candidate_position is not None:
                    print(
                        f"[nav resolve] falling back to candidate {i} "
                        "(top pick had no valid depth) as goal"
                    )
                    goal_position = candidate_position
                    goal_bbox_norm = candidate_bbox
                    del obstacle_bboxes_norm[i]
                    obstacle_bboxes_norm.append(goal_spec_r.goal_bbox_norm)
                    used_fallback = True
                    break

        # Ghost-mask overlay for a DINO-grounded target (register_object_mask
        # above is a live 3D mesh, but dynamically registered objects render
        # 0px on this GPU/driver -- see CLAUDE.md's known-issues memory --
        # so it alone gives no visual confirmation grounding worked. This is
        # read every frame in _run's render loop and reprojected via
        # project_world_point_with_pixel_radius + ghost_mask.draw_ghost_mask,
        # which (unlike register_object_mask) is a real 2D image overlay and
        # does render. The clear at the top of this function already reset
        # this to None; only set it back when this was actually a
        # DINO-grounded resolve with a valid position -- the classic SAM2
        # auto-resolve path (target_text=None) keeps relying on the
        # mesh-mask overlay as before. Excludes used_fallback: that promotes
        # an *obstacle* rock's position as the goal when the grounded
        # point's own depth was invalid, which would otherwise show the
        # ghost mask over a rock instead of the actual requested target.
        if target_text and goal_position is not None and not used_fallback:
            self._ground_target_world = goal_position
        new_goal_objects: list = []
        new_obstacle_objects: list = []
        goal_mask_warning = ""
        if goal_position is not None:
            new_goal_objects.append(
                env.register_object_mask(
                    goal_position, MESH_GOAL_ID, self.obj_mask_radius, mask_dir, "goal"
                )
            )
            # goal_spec_r.instruction_text still describes the VLM's original
            # top pick (its reasoning references that specific detection) --
            # note the substitution explicitly rather than let the sidebar
            # imply the reasoning still matches whichever bbox actually got
            # registered as the goal.
            instruction = goal_spec_r.instruction_text
            if used_fallback:
                instruction += (
                    " [top pick had no valid depth -- fell back to another "
                    "detection as the actual goal]"
                )
            status_msg = (
                f"reviewing resolved goal: '{instruction}' -- "
                "Confirm to drive, Rerun to retry, or Pick Manually"
            )
            gx, gy, gz = goal_position
            self._event_log.log(
                "goal_detected",
                name=target_text or goal_spec_r.instruction_text,
                method="dino" if target_text else "sam2+qwen",
                world=(round(gx, 2), round(gy, 2), round(gz, 2)),
                fallback=used_fallback,
                score=vlm_result.get("score"),
            )
        else:
            status_msg = (
                "resolved goal has no valid depth -- skipping mask "
                "(Rerun or Pick Manually)"
            )
            self._event_log.log(
                "goal_detect_failed",
                name=target_text or goal_spec_r.instruction_text,
                reason="no valid depth",
            )
            # This is the case the user needs to actually see: the VLM did
            # pick a goal (goal_spec_r.instruction_text above reflects it),
            # but neither its bbox nor any fallback candidate got a mask in
            # the scene, so the rover has only obstacles registered and
            # nothing to drive toward. The per-tick `d.status_text = status`
            # in _env_loop overwrites display.status_text with the generic
            # mode banner on the very next frame, which would otherwise
            # erase this warning before it's ever seen -- click_status
            # isn't touched every tick, so park it there instead.
            goal_mask_warning = (
                "resolve: no detection had valid depth, only obstacle "
                "mask(s) registered -- Rerun or Pick Manually"
            )
        for i, obstacle_bbox in enumerate(obstacle_bboxes_norm):
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
        print(
            f"[nav resolve] goal_mask={'registered' if new_goal_objects else 'MISSING (bad depth)'}, "
            f"obstacle_masks={len(new_obstacle_objects)}/{len(goal_spec_r.obstacle_bboxes_norm)} registered"
        )

        with self._lock:
            self._mode = MODE_REVIEW_SEGMENTATION
            self._world_goal = None
            self.display.status_text = status_msg
            self.display.click_status = goal_mask_warning
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
        if mode == MODE_TURN:
            return "TURN"
        return mode
