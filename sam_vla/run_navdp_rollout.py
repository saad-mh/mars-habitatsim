import argparse
import math
import time, datetime
from pathlib import Path

from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.sim_utils import distance_to_goal
from sam_vla.vlm.qwen_server_manager import QwenServerManager
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.policy.navdp_policy import NavdpPolicy
from sam_vla.safety.safety_filter import filter as safety_filter_fn
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    goal_pixel_center,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.logging.rollout_logger import RolloutLogger
from sam_vla.perception.semantic_overlay import overlay_semantic_masks


def register_goal_obstacle_masks(env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir):
    """Give the chosen goal object a goal-mask mesh and every other detected
    object an obstacle-mask mesh, each a disc of `obj_mask_radius` around its
    bbox's backprojected world coords. The rest of the scene is untouched."""
    if goal_position is not None:
        env.register_object_mask(goal_position, MESH_GOAL_ID, obj_mask_radius, out_dir, "goal")
    else:
        print("[WARN] goal bbox had no valid depth; skipping goal mask", flush=True)

    for i, obstacle_bbox in enumerate(goal_spec.obstacle_bboxes_norm):
        obstacle_position = bbox_to_world(obs0, obstacle_bbox, hfov_deg=HFOV_DEG)
        if obstacle_position is None:
            print(f"[WARN] obstacle[{i}] bbox had no valid depth; skipping obstacle mask", flush=True)
            continue
        env.register_object_mask(obstacle_position, MESH_OBST_ID, obj_mask_radius, out_dir, f"obstacle_{i}")


def run(
    scene_path: str,
    heightmap_path: str,
    ckpt_path: str,
    out_dir: str,
    navdp_root: str = None,
    device: str = "cuda",
    sample_steps: int = 20,
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
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Still needed for the one-shot first-frame goal selection below
    # (first_frame_resolver -> qwen_client.select_goal_verbose); the driving
    # loop itself no longer calls the VLM per frame -- NavdpPolicy drives.
    qwen_manager = QwenServerManager()
    logger = RolloutLogger()

    with MarsHabitatEnv(
        scene_path,
        heightmap_path,
        services=[qwen_manager],
        start_x=start_x,
        start_z=start_z,
        start_yaw=math.radians(start_yaw_deg),
        randomise_spawn=randomise_spawn,
        with_semantic=True,
    ) as env:
        obs0 = env.get_observation(frame_idx=0)
        goal_spec, goal_vlm_result, sam_detections = first_frame_resolver.resolve_verbose(obs0.rgb)
        goal_position = backproject_goal_position(obs0, goal_spec, hfov_deg=HFOV_DEG)
        logger.log_goal_resolution(goal_spec, goal_vlm_result, goal_position)
        logger.save_sam_first_frame(obs0.rgb, sam_detections, goal_spec, out_dir)
        print(f"resolved goal_spec: {goal_spec.instruction_text} | goal_position={goal_position}")
        register_goal_obstacle_masks(env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir)

        # <-- policy plugged in here: NavdpPolicy replaces QwenDiscreteDirectionPolicy.
        # Same act_verbose(..., goal_spec, step) -> (Action, dict) shape as the VLA
        # policy it swaps out; see the loop below for the call site.
        policy = NavdpPolicy(
            ckpt_path=ckpt_path,
            navdp_root=navdp_root,
            device=device,
            sample_steps=sample_steps,
        )

        for step in range(max_steps):
            obs = env.get_observation(frame_idx=step)
            semantic = env.get_semantic_frame()
            raw_action, vla_result = policy.act_verbose(obs, semantic, goal_spec, step)
            action = safety_filter_fn(raw_action, obs)
            new_pose = integrate_mars(obs.pose, action, dt)
            env.step(new_pose)
            dist = (
                distance_to_goal(new_pose, goal_position)
                if goal_position is not None
                else None
            )
            dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
            overlay_text = (
                f"t={step} dist={dist_txt} "
                f"v=[{action.v_fwd:.2f},{action.v_lat:.2f}] yaw_rate={action.yaw_rate:.2f}"
            )
            vis_rgb = overlay_semantic_masks(obs.rgb, semantic, text=overlay_text)
            logger.log_step(obs, action, new_pose, vla_result=vla_result, vis_rgb=vis_rgb)

            if step % 10 == 0:
                height, width = obs.rgb.shape[:2]
                goal_pixel = goal_pixel_center(goal_spec.goal_bbox_norm, width, height)
                print(
                    f"[traj] step={step} | distance_to_goal={dist} | "
                    f"goal_pixel={goal_pixel} | action={action}"
                )

        logger.flush(out_dir)
        if save_frames:
            logger.save_frames(out_dir)
        if save_video:
            logger.save_video(out_dir, fps=video_fps)

    print("[inf] qwen_manager: stop confirmed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--heightmap-path", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to trained NavDP/S2DiT checkpoint")
    parser.add_argument("--navdp-root", default=None, help="Path to the navdp repo (default: ./navdp or $NAVDP_ROOT)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--out-dir", default=f"navdp_rollout{datetime.datetime.now().strftime('%d%m%y%H%M')}")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--save-video", action="store_true", help="Save rollout.mp4 from logged RGB frames")
    parser.add_argument("--save-frames", action="store_true", help="Save individual PNG frames under out_dir/frames/")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--start-x", type=float, default=0.0, help="Rover spawn x coordinate")
    parser.add_argument("--start-z", type=float, default=8.0, help="Rover spawn z coordinate")
    parser.add_argument("--start-yaw", type=float, default=0.0, help="Rover spawn yaw in degrees")
    parser.add_argument(
        "--randomise-spawn",
        action="store_true",
        help="Ignore --start-x/--start-z/--start-yaw and pick a random (x, z) spawn within the "
        "heightmap bounds, with height sampled from the heightmap",
    )
    parser.add_argument(
        "--obj-mask-radius",
        type=float,
        default=0.5,
        help="Radius (m) of the goal/obstacle mask mesh placed around each detected object's "
        "backprojected world coords",
    )
    args = parser.parse_args()

    run(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        navdp_root=args.navdp_root,
        device=args.device,
        sample_steps=args.sample_steps,
        max_steps=args.max_steps,
        dt=args.dt,
        save_video=args.save_video,
        save_frames=args.save_frames,
        video_fps=args.video_fps,
        start_x=args.start_x,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw,
        randomise_spawn=args.randomise_spawn,
        obj_mask_radius=args.obj_mask_radius,
    )
