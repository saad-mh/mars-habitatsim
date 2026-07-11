import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sam_vla.core.types import Action, Observation, Pose


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RolloutLogger:
    def __init__(self):
        self._rgb_frames = []
        self._poses = []
        self._actions = []
        self._frame_indices = []
        self._timestamps = []
        self._manifest = {"start_time": _now_iso(), "steps": []}

    def log_step(self, obs: Observation, action: Action, pose: Pose) -> None:
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        pose_tuple = (pose.x, pose.y, pose.z, pose.yaw)
        action_tuple = (action.v_fwd, action.v_lat, action.yaw_rate)
        timestamp = _now_iso()

        self._rgb_frames.append(rgb)
        self._poses.append(pose_tuple)
        self._actions.append(action_tuple)
        self._frame_indices.append(obs.frame_idx)
        self._timestamps.append(timestamp)

        self._manifest["steps"].append(
            {
                "frame_idx": obs.frame_idx,
                "pose": list(pose_tuple),
                "action": list(action_tuple),
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
        )

        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

        print(
            f"RolloutLogger: flushed {len(self._frame_indices)} steps -> "
            f"{npz_path}, {manifest_path}"
        )


if __name__ == "__main__":
    import tempfile

    logger = RolloutLogger()

    for i in range(5):
        obs = Observation(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=None,
            pose=Pose(x=float(i), y=0.0, z=0.0, yaw=0.0),
            frame_idx=i,
        )
        action = Action(v_fwd=0.5, v_lat=0.0, yaw_rate=0.1 * i)
        pose = Pose(x=float(i), y=0.0, z=0.0, yaw=0.0)
        logger.log_step(obs, action, pose)

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
        print("manifest steps:", len(manifest["steps"]))

        assert npz_data["rgb_frames"].shape[0] == 5
        assert npz_data["poses"].shape == (5, 4)
        assert npz_data["actions"].shape == (5, 3)
        assert npz_data["frame_indices"].shape == (5,)
        assert npz_data["timestamps"].shape == (5,)
        assert len(manifest["steps"]) == 5
        print("Round-trip check passed: step counts match.")
