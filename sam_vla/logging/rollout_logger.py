import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from sam_vla.core.goal_geometry import GoalPosition, distance_to_goal
from sam_vla.core.types import Action, GoalSpec, Observation, Pose


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RolloutLogger:
    def __init__(self):
        self._rgb_frames = []
        self._poses = []
        self._actions = []
        self._frame_indices = []
        self._timestamps = []
        self._distances_to_goal = []
        self._manifest = {"start_time": _now_iso(), "steps": [], "goal_resolution": None}
        self._goal_position: Optional[GoalPosition] = None

    def log_goal_resolution(
        self,
        goal_spec: GoalSpec,
        vlm_result: dict,
        goal_position: Optional[GoalPosition] = None,
    ) -> None:
        """Record the one-shot first-frame VLM goal-selection call (raw result + resolved spec)."""
        self._goal_position = goal_position
        self._manifest["goal_resolution"] = {
            "vlm_result": vlm_result,
            "goal_bbox_norm": list(goal_spec.goal_bbox_norm),
            "instruction_text": goal_spec.instruction_text,
            "goal_position": list(goal_position) if goal_position is not None else None,
            "timestamp": _now_iso(),
        }

    def log_step(
        self,
        obs: Observation,
        action: Action,
        pose: Pose,
        vla_result: Optional[dict] = None,
    ) -> None:
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        pose_tuple = (pose.x, pose.y, pose.z, pose.yaw)
        action_tuple = (action.v_fwd, action.v_lat, action.yaw_rate)
        timestamp = _now_iso()

        dist = (
            distance_to_goal(pose, self._goal_position)
            if self._goal_position is not None
            else None
        )

        self._rgb_frames.append(rgb)
        self._poses.append(pose_tuple)
        self._actions.append(action_tuple)
        self._frame_indices.append(obs.frame_idx)
        self._timestamps.append(timestamp)
        self._distances_to_goal.append(dist if dist is not None else float("nan"))

        self._manifest["steps"].append(
            {
                "frame_idx": obs.frame_idx,
                "pose": list(pose_tuple),
                "action": list(action_tuple),
                "vla_result": vla_result,
                "goal_position": list(self._goal_position) if self._goal_position is not None else None,
                "distance_to_goal": dist,
                "timestamp": timestamp,
            }
        )

    def flush(self, out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        npz_path = out_path / "rollout.npz"
        manifest_path = out_path / "manifest.json"

        np.savez(
            npz_path,
            rgb_frames=np.stack(self._rgb_frames),
            poses=np.array(self._poses, dtype=np.float64),
            actions=np.array(self._actions, dtype=np.float64),
            frame_indices=np.array(self._frame_indices, dtype=np.int64),
            timestamps=np.array(self._timestamps, dtype=str),
            distances_to_goal=np.array(self._distances_to_goal, dtype=np.float64),
        )

        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

        print(
            f"RolloutLogger: flushed {len(self._frame_indices)} steps -> "
            f"{npz_path}, {manifest_path}"
        )

    def save_frames(self, out_dir: str) -> None:
        """Dump each logged RGB frame as a PNG (frame_<frame_idx>.png)."""
        import imageio.v3 as iio

        frames_dir = Path(out_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, rgb in zip(self._frame_indices, self._rgb_frames):
            iio.imwrite(frames_dir / f"frame_{frame_idx:06d}.png", rgb)

        print(f"RolloutLogger: saved {len(self._rgb_frames)} frames -> {frames_dir}")

    def save_video(self, out_dir: str, fps: int = 10) -> None:
        """Encode the logged RGB frames into rollout.mp4 (imageio + ffmpeg backend)."""
        import imageio.v3 as iio

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        video_path = out_path / "rollout.mp4"

        iio.imwrite(video_path, np.stack(self._rgb_frames), fps=fps, codec="libx264")

        print(f"RolloutLogger: saved video ({len(self._rgb_frames)} frames @ {fps}fps) -> {video_path}")


if __name__ == "__main__":
    import tempfile

    from sam_vla.core.types import GoalSpec

    logger = RolloutLogger()
    logger.log_goal_resolution(
        goal_spec=GoalSpec(
            goal_bbox_norm=(0.4, 0.4, 0.6, 0.6),
            obstacle_bboxes_norm=[],
            instruction_text="Navigate to the rock target.",
        ),
        vlm_result={"goal_index": 1, "reasoning": "closest large rock"},
        goal_position=(10.0, 0.0, 0.0),
    )

    for i in range(5):
        obs = Observation(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=None,
            pose=Pose(x=float(i), y=0.0, z=0.0, yaw=0.0),
            frame_idx=i,
        )
        action = Action(v_fwd=0.5, v_lat=0.0, yaw_rate=0.1 * i)
        pose = Pose(x=float(i), y=0.0, z=0.0, yaw=0.0)
        vla_result = {"v_fwd": 0.5, "v_lat": 0.0, "yaw_rate": 0.1 * i}
        logger.log_step(obs, action, pose, vla_result=vla_result)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.flush(tmp_dir)

        npz_data = np.load(Path(tmp_dir) / "rollout.npz")
        with open(Path(tmp_dir) / "manifest.json") as f:
            manifest = json.load(f)

        print("rgb_frames shape:", npz_data["rgb_frames"].shape)
        print("poses shape:", npz_data["poses"].shape)
        print("actions shape:", npz_data["actions"].shape)
        print("frame_indices shape:", npz_data["frame_indices"].shape)
        print("timestamps shape:", npz_data["timestamps"].shape)
        print("distances_to_goal:", npz_data["distances_to_goal"])
        print("manifest steps:", len(manifest["steps"]))
        print("manifest goal_resolution:", manifest["goal_resolution"])

        assert npz_data["rgb_frames"].shape[0] == 5
        assert npz_data["poses"].shape == (5, 4)
        assert npz_data["actions"].shape == (5, 3)
        assert npz_data["distances_to_goal"].shape == (5,)
        assert npz_data["distances_to_goal"][0] == 10.0
        assert manifest["goal_resolution"]["vlm_result"]["goal_index"] == 1
        assert manifest["steps"][2]["vla_result"]["yaw_rate"] == 0.2
        assert npz_data["frame_indices"].shape == (5,)
        assert npz_data["timestamps"].shape == (5,)
        assert len(manifest["steps"]) == 5
        print("Round-trip check passed: step counts match.")
