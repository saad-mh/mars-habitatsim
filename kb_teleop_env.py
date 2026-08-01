"""Same WASD/space/x teleop UI as kb_teleop.py, but driven through
sam_vla.env.habitat_env.MarsHabitatEnv instead of a standalone habitat_sim
setup -- the exact sensor specs (480x640, HFOV 90) and camera-height formula
(local terrain max + spawn clearance) that sam_vla/run_segmentation_sweep.py
uses to build training data, so recorded frames match that distribution
instead of kb_teleop.py's own lower/point-sampled camera height.
"""

import os
import time
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageDraw

import tkinter as tk

from sam_vla.core.types import Pose
from sam_vla.env.habitat_env import MarsHabitatEnv
from sam_vla.env.terrain import SIZE_X, SIZE_Z

HERE = Path(__file__).resolve().parent

SCENE = str(HERE / "assets" / "marsyard2022.glb")
HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm_1025.tif")

OUT_DIR = f"mars_teleop_out{int(time.time())}"

# Start pose
START_X = 0.0
START_Z = 8.0
START_YAW_DEG = 0.0

# Movement
MOVE_STEP = 0.35
TURN_STEP_DEG = 10.0

# Bounds -- matches run_segmentation_sweep.py's --boundary-margin default
BOUNDARY_MARGIN = 2.0
AUTOSTOP_AT_BOUNDARY = True

# Recording
START_RECORDING = False
SAVE_ON_RECORDING_MOVEMENT_ONLY = True
SAVE_FRAME_ON_RECORDING_START = True

# Display
SHOW_DEPTH_BESIDE_RGB = True
DEPTH_VIS_MAX_METERS = 10.0

# ============================================================

TURN_STEP = np.deg2rad(TURN_STEP_DEG)


def depth_vis_from(depth: np.ndarray) -> np.ndarray:
    depth_clip = np.clip(depth, 0.0, DEPTH_VIS_MAX_METERS)
    depth_vis = (depth_clip / DEPTH_VIS_MAX_METERS * 255.0).astype(np.uint8)
    return np.stack([depth_vis, depth_vis, depth_vis], axis=-1)


def save_obs(rgb, depth, idx, x, y, z, yaw, recording):
    os.makedirs(OUT_DIR, exist_ok=True)

    depth_vis = depth_vis_from(depth)[:, :, 0]
    Image.fromarray(rgb).save(f"{OUT_DIR}/rgb_{idx:04d}.png")
    Image.fromarray(depth_vis).save(f"{OUT_DIR}/depth_{idx:04d}.png")

    with open(f"{OUT_DIR}/poses.txt", "a") as f:
        f.write(
            f"{idx:04d} "
            f"x={x:.4f} y={y:.4f} z={z:.4f} "
            f"yaw_rad={yaw:.4f} yaw_deg={np.rad2deg(yaw):.2f} "
            f"recording={int(recording)}\n"
        )


def apply_boundary(x, z, old_x, old_z):
    half_x = SIZE_X / 2.0 - BOUNDARY_MARGIN
    half_z = SIZE_Z / 2.0 - BOUNDARY_MARGIN

    inside = -half_x <= x <= half_x and -half_z <= z <= half_z

    if inside:
        return x, z

    if AUTOSTOP_AT_BOUNDARY:
        return old_x, old_z

    return float(np.clip(x, -half_x, half_x)), float(np.clip(z, -half_z, half_z))


class MarsTeleopApp:
    def __init__(self):
        self.env = MarsHabitatEnv(
            SCENE,
            HEIGHTMAP,
            start_x=START_X,
            start_z=START_Z,
            start_yaw=np.deg2rad(START_YAW_DEG),
        )
        self.env.__enter__()

        self.x = START_X
        self.z = START_Z
        self.yaw = np.deg2rad(START_YAW_DEG)

        self.recording = START_RECORDING
        self.frame_idx = 0
        self.recorded = False
        self.closed = False

        os.makedirs(OUT_DIR, exist_ok=True)
        poses_path = f"{OUT_DIR}/poses.txt"
        if os.path.exists(poses_path):
            os.remove(poses_path)

        self.root = tk.Tk()
        self.root.title("Kb Teleop (sweep-matched env)")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text="W/S move | A/D turn | SPACE record | P save | X quit",
            font=("Arial", 12),
        )
        self.info_label.pack()

        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.env.step(Pose(x=self.x, y=0.0, z=self.z, yaw=self.yaw))
        self.render()

    def render(self):
        obs = self.env.get_observation(self.frame_idx)
        self.latest_obs = obs
        self.pose_y = obs.pose.y

        depth_rgb = depth_vis_from(obs.depth)

        if SHOW_DEPTH_BESIDE_RGB:
            img_arr = np.hstack([obs.rgb, depth_rgb])
        else:
            img_arr = obs.rgb

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        status = (
            f"x={self.x:.2f} y={self.pose_y:.2f} z={self.z:.2f} "
            f"yaw={np.rad2deg(self.yaw):.1f} "
            f"REC={'ON' if self.recording else 'OFF'}"
        )

        draw.rectangle([0, 0, img.width, 55], fill=(0, 0, 0))
        draw.text((10, 8), status, fill=(255, 255, 255))
        draw.text(
            (10, 30),
            "W/S move | A/D turn | SPACE record | P save | X quit",
            fill=(255, 255, 255),
        )

        if self.recording:
            draw.ellipse([10, 65, 30, 85], fill=(255, 0, 0))
            draw.text((38, 66), "RECORDING", fill=(255, 0, 0))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def save_current_frame(self):
        save_obs(
            self.latest_obs.rgb,
            self.latest_obs.depth,
            self.frame_idx,
            self.x,
            self.pose_y,
            self.z,
            self.yaw,
            self.recording,
        )

        self.recorded = True

        print(f"saved frame {self.frame_idx:04d}")
        self.frame_idx += 1

    def on_key(self, event):
        key = event.keysym.lower()

        old_x = self.x
        old_z = self.z
        moved = False

        if key == "x" or key == "escape":
            self.close()
            return

        elif key == "space":
            self.recording = not self.recording
            print(f"Recording {'ON' if self.recording else 'OFF'}")

            if self.recording and SAVE_FRAME_ON_RECORDING_START:
                self.save_current_frame()

        elif key == "w":
            self.x += -np.sin(self.yaw) * MOVE_STEP
            self.z += -np.cos(self.yaw) * MOVE_STEP
            moved = True

        elif key == "s":
            self.x -= -np.sin(self.yaw) * MOVE_STEP
            self.z -= -np.cos(self.yaw) * MOVE_STEP
            moved = True

        elif key == "a":
            self.yaw += TURN_STEP
            moved = True

        elif key == "d":
            self.yaw -= TURN_STEP
            moved = True

        elif key == "p":
            self.save_current_frame()

        self.x, self.z = apply_boundary(self.x, self.z, old_x, old_z)

        self.env.step(Pose(x=self.x, y=0.0, z=self.z, yaw=self.yaw))
        self.render()

        if self.recording:
            if SAVE_ON_RECORDING_MOVEMENT_ONLY:
                if moved:
                    self.save_current_frame()
            else:
                self.save_current_frame()

    def close(self):
        if self.closed:
            return

        self.closed = True

        try:
            self.env.__exit__(None, None, None)
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

        if self.recorded:
            print(f"Done. Output: {OUT_DIR}")
        else:
            print("Done. No frames recorded.")
            shutil.rmtree(OUT_DIR, ignore_errors=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MarsTeleopApp()
    app.run()
