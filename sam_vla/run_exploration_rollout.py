"""
conda activate habitat
python -m sam_vla.run_exploration_rollout \
    --scene-path assets/marsyard2022.glb \
    --heightmap-path marsyard2022_terrain_hm_1025.tif \
    --command area --cbf --max-steps 300 \
    --out-dir output/explore_smoke_test

Drives sam_vla.policy.exploration_policy.ExplorationPolicy end to end: real
first-frame VLM goal/obstacle detection (first_frame_resolver, same as
run_navdp_rollout.py's default single-goal path) registers actual MESH_GOAL_ID /
MESH_OBST_ID meshes so CBF has real obstacles to see (unlike kb_teleop_vl.py's
synthetic, undetectable obstacle field), a BeliefGoalTracker confirms when the
goal has actually come into view (GOAL_CONFIRM_STEPS consecutive sightings) to
stop the episode, and per-step composition (policy.act_verbose -> safety_filter
-> CbfObstacleAvoidance.apply -> integrate_mars -> env.step) exactly mirrors
run_navdp_rollout.py:666-716 with NavdpPolicy swapped for ExplorationPolicy.

Two separate Qwen server managers are needed (different ports, see their
respective configs): sam_vla.vlm.qwen_server_manager for first_frame_resolver's
one-shot goal/obstacle detection, and vl_direction.qwen_server_manager for
ExplorationPolicy's per-leg direction queries. register_goal_obstacle_masks
below is a direct copy of run_navdp_rollout.py's helper of the same name rather
than an import from it, since that module imports NavdpPolicy (and therefore
torch) at module scope -- this script has no diffusion-policy dependency and
shouldn't need torch installed to run without --cbf.
"""

import argparse
import datetime
import math
import time
from pathlib import Path

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.sim_utils import distance_to_goal
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.logging.rollout_logger import RolloutLogger
from sam_vla.perception.semantic_overlay import overlay_semantic_masks
from sam_vla.policy.exploration_policy import ExplorationConfig, ExplorationPolicy
from sam_vla.safety.safety_filter import filter as safety_filter_fn
from sam_vla.vlm.qwen_server_manager import QwenServerManager as SamQwenServerManager
from vl_direction.client import get_client
from vl_direction.qwen_server_manager import QwenServerManager as VlQwenServerManager

GOAL_CONFIRM_STEPS = 5


def register_goal_obstacle_masks(
    env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir
):
    """Give the chosen goal object a goal-mask mesh and every other detected
    object an obstacle-mask mesh -- copy of run_navdp_rollout.py's helper of
    the same name, see module docstring for why this isn't an import."""
    if goal_position is not None:
        env.register_object_mask(
            goal_position, MESH_GOAL_ID, obj_mask_radius, out_dir, "goal"
        )
    else:
        print("[WARN] goal bbox had no valid depth; skipping goal mask", flush=True)

    for i, obstacle_bbox in enumerate(goal_spec.obstacle_bboxes_norm):
        obstacle_position = bbox_to_world(obs0, obstacle_bbox, hfov_deg=HFOV_DEG)
        if obstacle_position is None:
            print(
                f"[WARN] obstacle[{i}] bbox had no valid depth; skipping obstacle mask",
                flush=True,
            )
            continue
        env.register_object_mask(
            obstacle_position, MESH_OBST_ID, obj_mask_radius, out_dir, f"obstacle_{i}"
        )


def run(
    scene_path: str,
    heightmap_path: str,
    out_dir: str,
    task_str: str = "explore the terrain",
    command: str = "area",
    max_steps: int = 500,
    dt: float = 0.1,
    save_video: bool = False,
    save_frames: bool = False,
    video_fps: int = 10,
    start_x: float = 0.0,
    start_z: float = 8.0,
    start_yaw_deg: float = 0.0,
    randomise_spawn: bool = False,
    obj_mask_radius: float = 0.5,
    cbf: bool = False,
    navdp_root: str = None,
    cbf_d_safe: float = 0.75,
    cbf_gamma: float = 0.3,
    cbf_deadzone: float = 0.6,
    cbf_orbit_kr: float = 0.8,
    cbf_orbit_hyst: float = 0.4,
    cbf_pursuit_kp: float = 1.8,
    cbf_goaround_forward: float = 0.5,
    cbf_escape_yaw: bool = True,
    cbf_hard_gate: bool = True,
    robot_radius: float = 0.25,
    safety_margin: float = 0.15,
    obstacle_radius: float = 0.25,
    max_yaw_rate: float = 1.0,
    zero_lateral: bool = True,
    belief_goal_range: float = 8.0,
    lost_goal_min_px: int = 10,
    leg_length_m: float = 3.0,
    leg_length_jitter_m: float = 1.0,
    cruise_speed: float = 1.0,
    turn_deg: float = 60.0,
    turn_jitter_deg: float = 20.0,
    cell_size_m: float = 1.5,
    stall_window_steps: int = 15,
    stall_distance_m: float = 0.5,
    backtrack_distance_m: float = 2.0,
    seed: int = None,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sam_qwen_manager = SamQwenServerManager()
    vl_qwen_manager = VlQwenServerManager()
    logger = RolloutLogger()
    episode_id = f"explore-{int(time.time())}"

    with MarsHabitatEnv(
        scene_path,
        heightmap_path,
        services=[sam_qwen_manager, vl_qwen_manager],
        start_x=start_x,
        start_z=start_z,
        start_yaw=math.radians(start_yaw_deg),
        randomise_spawn=randomise_spawn,
        with_semantic=True,
    ) as env:
        obs0 = env.get_observation(frame_idx=0)

        goal_spec, goal_vlm_result, sam_detections = (
            first_frame_resolver.resolve_verbose(obs0.rgb)
        )
        goal_position = backproject_goal_position(obs0, goal_spec, hfov_deg=HFOV_DEG)
        logger.log_goal_resolution(goal_spec, goal_vlm_result, goal_position)
        logger.save_sam_first_frame(obs0.rgb, sam_detections, goal_spec, out_dir)
        print(
            f"resolved goal_spec: {goal_spec.instruction_text} | goal_position={goal_position}"
        )
        register_goal_obstacle_masks(
            env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir
        )

        policy = ExplorationPolicy(
            ExplorationConfig(
                task_str=task_str,
                leg_length_m=leg_length_m,
                leg_length_jitter_m=leg_length_jitter_m,
                cruise_speed=cruise_speed,
                turn_deg=turn_deg,
                turn_jitter_deg=turn_jitter_deg,
                cell_size_m=cell_size_m,
                stall_window_steps=stall_window_steps,
                stall_distance_m=stall_distance_m,
                backtrack_distance_m=backtrack_distance_m,
                seed=seed,
            ),
            episode_id=episode_id,
            client=get_client("qwen"),
        )
        policy.set_command(command)

        belief_tracker = BeliefGoalTracker(
            hfov_deg=HFOV_DEG, goal_range=belief_goal_range, min_px=lost_goal_min_px
        )

        avoidance = None
        if cbf:
            # CbfObstacleAvoidance.__init__ does `from navdp.extensions import ...`,
            # which needs navdp_root on sys.path first -- normally done by
            # NavdpPolicy's constructor (see its module docstring), but this script
            # has no NavdpPolicy, so do it directly with the same helpers
            # run_navdp_rollout.py uses.
            from sam_vla.policy.navdp_policy import _add_navdp_to_path, _resolve_navdp_root
            from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance

            _add_navdp_to_path(_resolve_navdp_root(navdp_root))

            avoidance = CbfObstacleAvoidance(
                d_safe=cbf_d_safe,
                gamma=cbf_gamma,
                deadzone=cbf_deadzone,
                orbit_kr=cbf_orbit_kr,
                orbit_hyst=cbf_orbit_hyst,
                pursuit_kp=cbf_pursuit_kp,
                goaround_forward=cbf_goaround_forward,
                escape_yaw=cbf_escape_yaw,
                hard_gate=cbf_hard_gate,
                robot_radius=robot_radius,
                safety_margin=safety_margin,
                obstacle_radius=obstacle_radius,
                max_yaw_rate=max_yaw_rate,
            )
        cbf_active_steps = 0
        hard_gate_fired_steps = 0
        goal_seen_steps = 0

        for step in range(max_steps):
            obs = env.get_observation(frame_idx=step)
            semantic = env.get_semantic_frame()

            raw_action, vla_result = policy.act_verbose(obs, semantic, goal_spec, step)
            action = safety_filter_fn(raw_action, obs)

            goal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
            obstacle_mask = (semantic == MESH_OBST_ID).astype("uint8") * 255
            goal_visible = belief_tracker.observe(goal_mask, obs.depth)
            goal_bearing = belief_tracker.bearing()

            obstacle_point = None
            if avoidance is not None:
                height, width = obs.depth.shape[:2]
                intr = intrinsics_from_hfov(height, width, HFOV_DEG)
                obstacle_point = avoidance.nearest_obstacle(
                    obstacle_mask, obs.depth, intr
                )

            if zero_lateral and avoidance is not None:
                action = Action(v_fwd=action.v_fwd, v_lat=0.0, yaw_rate=action.yaw_rate)

            cbf_info = {}
            if avoidance is not None:
                action, cbf_info = avoidance.apply(action, obstacle_point, goal_bearing)
                if cbf_info.get("blocked"):
                    cbf_active_steps += 1
                if cbf_info.get("hard_gate_fired"):
                    hard_gate_fired_steps += 1

            new_pose = integrate_mars(obs.pose, action, dt)
            env.step(new_pose)
            belief_tracker.propagate(action, dt)

            dist = (
                distance_to_goal(new_pose, goal_position)
                if goal_position is not None
                else None
            )
            dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
            overlay_text = (
                f"t={step} dist={dist_txt} v=[{action.v_fwd:.2f},{action.v_lat:.2f}] "
                f"yaw_rate={action.yaw_rate:.2f} {vla_result['command']}/{vla_result['leg_kind']}"
            )
            vis_rgb = overlay_semantic_masks(obs.rgb, semantic, text=overlay_text)

            vla_result = {**vla_result, "goal_visible": goal_visible, **cbf_info}
            logger.log_step(
                obs, action, new_pose, vla_result=vla_result, vis_rgb=vis_rgb
            )

            if step % 10 == 0:
                print(
                    f"[traj] step={step} | distance_to_goal={dist} | action={action} | "
                    f"{vla_result}"
                )

            if goal_visible:
                goal_seen_steps += 1
                if goal_seen_steps >= GOAL_CONFIRM_STEPS:
                    print(
                        f"[explore] goal confirmed visible at step={step}, stopping",
                        flush=True,
                    )
                    break
            else:
                goal_seen_steps = 0

        if avoidance is not None:
            print(
                f"[CBF diag] blocked_steps={cbf_active_steps} hard_gate_fired={hard_gate_fired_steps}",
                flush=True,
            )
        logger.flush(out_dir)
        if save_frames:
            logger.save_frames(out_dir)
        if save_video:
            logger.save_video(out_dir, fps=video_fps)

    print("[explore] qwen managers: stop confirmed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--heightmap-path", required=True)
    parser.add_argument(
        "--out-dir",
        default=f"explore_rollout{datetime.datetime.now().strftime('%d%m%y%H%M')}",
    )
    parser.add_argument(
        "--task-str",
        default="explore the terrain",
        help="high-level task string passed to vl_direction's exploration prompt",
    )
    parser.add_argument(
        "--command",
        choices=["area", "left", "right"],
        default="area",
        help="which of ExplorationPolicy's three commandable patterns to run",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--start-x", type=float, default=8.0)
    parser.add_argument("--start-z", type=float, default=10.0)
    parser.add_argument("--start-yaw", dest="start_yaw_deg", type=float, default=0.0)
    parser.add_argument("--randomise-spawn", action="store_true")
    parser.add_argument("--obj-mask-radius", type=float, default=0.5)

    parser.add_argument("--cbf", action="store_true")
    parser.add_argument(
        "--navdp-root",
        default=None,
        help="Path to the navdp repo, needed by --cbf (default: ./navdp or $NAVDP_ROOT)",
    )
    parser.add_argument("--cbf-d-safe", type=float, default=0.75)
    parser.add_argument("--cbf-gamma", type=float, default=0.3)
    parser.add_argument("--cbf-deadzone", type=float, default=0.6)
    parser.add_argument("--cbf-orbit-kr", type=float, default=0.8)
    parser.add_argument("--cbf-orbit-hyst", type=float, default=0.4)
    parser.add_argument("--cbf-pursuit-kp", type=float, default=1.8)
    parser.add_argument("--cbf-goaround-forward", type=float, default=0.5)
    parser.add_argument(
        "--cbf-escape-yaw", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cbf-hard-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.15)
    parser.add_argument("--obstacle-radius", type=float, default=0.25)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument(
        "--zero-lateral", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--belief-goal-range", type=float, default=8.0)
    parser.add_argument("--lost-goal-min-px", type=int, default=10)

    parser.add_argument("--leg-length-m", type=float, default=3.0)
    parser.add_argument("--leg-length-jitter-m", type=float, default=1.0)
    parser.add_argument("--cruise-speed", type=float, default=1.0)
    parser.add_argument("--turn-deg", type=float, default=60.0)
    parser.add_argument("--turn-jitter-deg", type=float, default=20.0)
    parser.add_argument("--cell-size-m", type=float, default=1.5)
    parser.add_argument("--stall-window-steps", type=int, default=15)
    parser.add_argument("--stall-distance-m", type=float, default=0.5)
    parser.add_argument("--backtrack-distance-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    run(**vars(args))
