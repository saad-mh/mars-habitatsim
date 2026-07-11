import argparse
from pathlib import Path

from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.vlm.qwen_server_manager import QwenServerManager
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.policy.qwen_vla_policy import QwenVlaPolicy
from sam_vla.safety.safety_filter import filter as safety_filter_fn
from sam_vla.core.goal_geometry import backproject_goal_position, distance_to_goal
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.logging.rollout_logger import RolloutLogger


def run(
    scene_path: str,
    heightmap_path: str,
    out_dir: str,
    max_steps: int = 500,
    dt: float = 0.1,
    save_video: bool = False,
    save_frames: bool = False,
    video_fps: int = 10,
) -> None:
    qwen_manager = QwenServerManager()
    logger = RolloutLogger()

    with MarsHabitatEnv(scene_path, heightmap_path, services=[qwen_manager]) as env:
        obs0 = env.get_observation(frame_idx=0)
        goal_spec, goal_vlm_result = first_frame_resolver.resolve_verbose(obs0.rgb)
        goal_position = backproject_goal_position(obs0, goal_spec, hfov_deg=HFOV_DEG)
        logger.log_goal_resolution(goal_spec, goal_vlm_result, goal_position)
        print(f"resolved goal_spec: {goal_spec.instruction_text} | goal_position={goal_position}")

        policy = QwenVlaPolicy()

        for step in range(max_steps):
            obs = env.get_observation(frame_idx=step)
            raw_action, vla_result = policy.act_verbose(obs, goal_spec)
            action = safety_filter_fn(raw_action, obs)
            new_pose = integrate_mars(obs.pose, action, dt)
            env.step(new_pose)
            logger.log_step(obs, action, new_pose, vla_result=vla_result)

            if step % 50 == 0:
                dist = (
                    distance_to_goal(new_pose, goal_position)
                    if goal_position is not None
                    else None
                )
                print(
                    f"[inf] step {step}: pose={new_pose} | action={action} | "
                    f"distance_to_goal={dist}"
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
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--save-video", action="store_true", help="Save rollout.mp4 from logged RGB frames")
    parser.add_argument("--save-frames", action="store_true", help="Save individual PNG frames under out_dir/frames/")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    run(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        dt=args.dt,
        save_video=args.save_video,
        save_frames=args.save_frames,
        video_fps=args.video_fps,
    )
