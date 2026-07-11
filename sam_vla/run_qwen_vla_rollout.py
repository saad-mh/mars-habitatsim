import argparse
from pathlib import Path

from sam_vla.env.habitat_env import MarsHabitatEnv
from sam_vla.vlm.qwen_server_manager import QwenServerManager
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.policy.qwen_vla_policy import QwenVlaPolicy
from sam_vla.safety.safety_filter import filter as safety_filter_fn
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.logging.rollout_logger import RolloutLogger


def run(
    scene_path: str,
    heightmap_path: str,
    out_dir: str,
    max_steps: int = 500,
    dt: float = 0.1,
) -> None:
    qwen_manager = QwenServerManager()
    logger = RolloutLogger()

    with MarsHabitatEnv(scene_path, heightmap_path, services=[qwen_manager]) as env:
        obs0 = env.get_observation(frame_idx=0)
        goal_spec = first_frame_resolver.resolve(obs0.rgb)
        print(f"resolved goal_spec: {goal_spec.instruction_text}")

        policy = QwenVlaPolicy()

        for step in range(max_steps):
            obs = env.get_observation(frame_idx=step)
            raw_action = policy.act(obs, goal_spec)
            action = safety_filter_fn(raw_action, obs)
            new_pose = integrate_mars(obs.pose, action, dt)
            env.step(new_pose)
            logger.log_step(obs, action, new_pose)

            if step % 50 == 0:
                print(f"[inf] step {step}: pose={new_pose} | action={action}")

        logger.flush(out_dir)

    print("[inf] qwen_manager: stop confirmed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--heightmap_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    run(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        dt=args.dt,
    )
