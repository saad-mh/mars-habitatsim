"""
VLM-driven target navigation demo for Mars rover in Habitat-Sim.

Uses continuous velocity control via pose integration.
"""

import os
import time
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
import quaternion

import habitat_sim
from habitat_sim.agent import AgentConfiguration

HERE = Path(__file__).resolve().parent

SCENE = str(HERE / "marsyard2022_tri.glb")
HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm.png")

OUT_DIR = f"vlm_nav_out_{int(time.time())}"

# Terrain scale
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566

# Start pose
START_X = 12.2
START_Z = 10.0
START_YAW_DEG = 10.0

# Camera height above terrain
INITIAL_CLEARANCE = 0.9

# Display
RGBD_RESOLUTION = [480, 640]

# Heightmap correction
FLIP_HEIGHTMAP_X = False
FLIP_HEIGHTMAP_Z = True
SWAP_HEIGHTMAP_XZ = False


TARGET_X = 1.0
TARGET_Z = -3.0
TARGET_Y = None

# Controller parameters
HEADING_ERROR_THRESHOLD = np.deg2rad(15.0)  # rotate in place if error > 15°
MAX_LINEAR_VELOCITY = 0.8  # m/s
MAX_ANGULAR_VELOCITY = np.deg2rad(45.0)  # deg(rad)/s ?? how do i convey what unit this is in?
PROPORTIONAL_GAIN_HEADING = 1.0  # heading correction gain while driving
PROPORTIONAL_GAIN_ANGULAR = 2.0  # rotation-only gain when heading error large
STANDOFF_DISTANCE = 0.5  # Stop this distance from target

# Simulation
DT = 0.1  # time step (seconds)
MAX_STEPS = 1000  # Max simulation steps
SAVE_FRAMES = True  # Save RGB/depth frames to OUT_DIR

def load_heightmap(path):
    """Load and normalize heightmap from PNG."""
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)

    y = arr * SIZE_Y
    y = y - np.mean(y)

    return y


HEIGHT = load_heightmap(HEIGHTMAP)
HM_H, HM_W = HEIGHT.shape


def terrain_height_at(x, z):
    """Bilinear interpolation of terrain height at (x, z)."""
    if SWAP_HEIGHTMAP_XZ:
        x, z = z, x
    u = (x + SIZE_X / 2.0) / SIZE_X
    v = (z + SIZE_Z / 2.0) / SIZE_Z
    if FLIP_HEIGHTMAP_X:
        u = 1.0 - u
    if FLIP_HEIGHTMAP_Z:
        v = 1.0 - v
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    px = u * (HM_W - 1)
    py = v * (HM_H - 1)
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, HM_W - 1)
    y1 = min(y0 + 1, HM_H - 1)
    dx = px - x0
    dy = py - y0
    h00 = HEIGHT[y0, x0]
    h10 = HEIGHT[y0, x1]
    h01 = HEIGHT[y1, x0]
    h11 = HEIGHT[y1, x1]
    h0 = h00 * (1.0 - dx) + h10 * dx
    h1 = h01 * (1.0 - dx) + h11 * dx
    return float(h0 * (1.0 - dy) + h1 * dy)


def make_sensor(uuid, sensor_type):
    """Create a camera sensor spec (RGB or depth)."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = RGBD_RESOLUTION
    spec.position = [0.0, 0.0, 0.0]
    return spec


def make_sim():
    """Initialize Habitat simulator with scene and sensors."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
    sim_cfg.enable_physics = False

    rgb = make_sensor("rgb", habitat_sim.SensorType.COLOR)
    depth = make_sensor("depth", habitat_sim.SensorType.DEPTH)

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]

    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [agent_cfg])
    )


def rgb_depth_from_obs(obs):
    """Extract and process RGB and depth from sensor observation."""
    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    rgb = rgb.astype(np.uint8)

    depth = obs["depth"]
    return rgb, depth


def save_frame(obs, step_idx, x, y, z, yaw, linear_v, angular_v, distance, bearing_error):
    """Save RGB and depth frame along with metadata."""
    os.makedirs(OUT_DIR, exist_ok=True)

    rgb, depth = rgb_depth_from_obs(obs)

    Image.fromarray(rgb).save(f"{OUT_DIR}/rgb_{step_idx:04d}.png")

    # Normalize depth for visualization
    depth_clip = np.clip(depth, 0.0, 10.0)
    depth_vis = (depth_clip / 10.0 * 255.0).astype(np.uint8)
    Image.fromarray(depth_vis).save(f"{OUT_DIR}/depth_{step_idx:04d}.png")

    with open(f"{OUT_DIR}/nav_log.txt", "a") as f:
        f.write(
            f"step={step_idx:04d} "
            f"x={x:.3f} y={y:.3f} z={z:.3f} "
            f"yaw_deg={np.rad2deg(yaw):.1f} "
            f"linear_v={linear_v:.3f} angular_v_deg={np.rad2deg(angular_v):.1f} "
            f"distance={distance:.3f} bearing_error_deg={np.rad2deg(bearing_error):.1f}\n"
        )


class GoToGoalController:
    """Proportional go-to-goal controller for rover navigation."""

    def __init__(self, target_x, target_y, target_z):
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.at_target = False

    def update(self, rover_x, rover_y, rover_z, rover_yaw, debug=False):
        """
        Compute velocity command to reach target.

        Returns:
            (linear_x, angular_y): forward velocity (m/s) and rotation rate (rad/s)
            (distance, bearing_error): diagnostic info
        """
        # Compute target in rover's local frame
        dx = self.target_x - rover_x
        dz = self.target_z - rover_z
        distance = np.sqrt(dx**2 + dz**2)

        # Bearing to target in global frame
        # Note: negated to match velocity integration convention (positive yaw = left turn)
        target_bearing = -np.arctan2(dx, -dz)

        if debug:
            print(f"[DEBUG] rover: ({rover_x:.2f}, {rover_z:.2f}) yaw={np.rad2deg(rover_yaw):.1f}°")
            print(f"[DEBUG] target: ({self.target_x:.2f}, {self.target_z:.2f})")
            print(f"[DEBUG] delta: dx={dx:.2f}, dz={dz:.2f}")
            print(f"[DEBUG] bearing={np.rad2deg(target_bearing):.1f}°, distance={distance:.2f}m")

        # Heading error: wrap to [-pi, pi]
        heading_error = target_bearing - rover_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        linear_x = 0.0
        angular_y = 0.0

        # Stop if at target
        if distance <= STANDOFF_DISTANCE:
            self.at_target = True
            linear_x = 0.0
            angular_y = 0.0
        # Rotate in place if heading error is large
        elif abs(heading_error) > HEADING_ERROR_THRESHOLD:
            linear_x = 0.0
            angular_y = np.sign(heading_error) * MAX_ANGULAR_VELOCITY * PROPORTIONAL_GAIN_ANGULAR
        # Drive forward with heading correction
        else:
            linear_x = MAX_LINEAR_VELOCITY
            angular_y = heading_error * MAX_ANGULAR_VELOCITY * PROPORTIONAL_GAIN_HEADING

        return linear_x, angular_y, distance, heading_error

    def obstacle_avoidance_hook(self, linear_x, angular_y, depth_frame):
        """
        Placeholder for obstacle avoidance logic.

        This will be implemented in a follow-up task. For now, returns velocities unchanged.

        Args:
            linear_x, angular_y: commanded velocities from go-to-goal
            depth_frame: depth image from current frame

        Returns:
            (linear_x, angular_y): potentially modified velocities
        """
        # TODO: Implement obstacle avoidance here
        # - Compute obstacle map from depth frame (e.g., distance-to-obstacle in forward direction)
        # - If obstacle ahead, reduce linear_x or adjust angular_y to steer around
        # - Log obstacle detections for debugging
        return linear_x, angular_y


def create_video_from_frames(output_dir, frame_type):
    """
    Create an MP4 video from a sequence of PNG frames using ffmpeg.

    Args:
        output_dir: Directory containing the frame PNGs
        frame_type: "rgb" or "depth" to specify which frames to use

    Returns:
        Path to created video or None if ffmpeg failed
    """
    pattern = os.path.join(output_dir, f"{frame_type}_%04d.png")
    video_path = os.path.join(output_dir, f"{frame_type}_video.mp4")

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-framerate", "10",  # 10 fps for visualization
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        video_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"[inf] Created {frame_type} video: {video_path}")
            return video_path
        else:
            print(f"[err] ffmpeg error creating {frame_type} video:")
            print(result.stderr)
            return None
    except FileNotFoundError:
        print(f"[err] ffmpeg not found. Install with: sudo apt-get install ffmpeg")
        return None
    except subprocess.TimeoutExpired:
        print(f"[err] ffmpeg timeout creating {frame_type} video")
        return None
    except Exception as e:
        print(f"[err] Error creating {frame_type} video: {e}")
        return None


class RoverSim:
    """Main rover simulation and navigation loop."""

    def __init__(self):
        self.sim = make_sim()
        self.agent = self.sim.initialize_agent(0)

        self.x = START_X
        self.z = START_Z
        self.yaw = np.deg2rad(START_YAW_DEG)

        # Compute target Y from heightmap
        self.target_y = terrain_height_at(TARGET_X, TARGET_Z) + INITIAL_CLEARANCE
        self.terrain_y = terrain_height_at(self.x, self.z)
        self.y = self.terrain_y + INITIAL_CLEARANCE

        # Initialize controller
        self.controller = GoToGoalController(TARGET_X, self.target_y, TARGET_Z)

        # Clean up old output directory
        if os.path.exists(OUT_DIR):
            import shutil
            shutil.rmtree(OUT_DIR, ignore_errors=True)
        os.makedirs(OUT_DIR, exist_ok=True)

        # Write header to nav log
        with open(f"{OUT_DIR}/nav_log.txt", "w") as f:
            f.write(
                f"# Navigation log for go-to-goal\n"
                f"# Target: ({TARGET_X:.2f}, {self.target_y:.2f}, {TARGET_Z:.2f})\n"
                f"# Standoff distance: {STANDOFF_DISTANCE:.2f}m\n"
                f"# Start pose: x={START_X:.2f} z={START_Z:.2f} yaw_deg={START_YAW_DEG:.1f}\n\n"
            )

        self.set_agent_pose()
        self.step_idx = 0

    def set_agent_pose(self):
        """Update agent pose in simulator."""
        state = self.agent.get_state()
        state.position = np.array([self.x, self.y, self.z], dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector([0.0, self.yaw, 0.0])
        self.agent.set_state(state)

    def step(self, debug_first_n=3):
        """Execute one navigation step."""
        # Get sensor observations
        obs = self.sim.get_sensor_observations()
        rgb, depth = rgb_depth_from_obs(obs)

        # Run controller (debug first few steps)
        debug = self.step_idx < debug_first_n
        linear_x, angular_y, distance, bearing_error = self.controller.update(
            self.x, self.y, self.z, self.yaw, debug=debug
        )

        # Obstacle avoidance hook (stub for now)
        linear_x, angular_y = self.controller.obstacle_avoidance_hook(
            linear_x, angular_y, depth
        )

        # Integrate velocities into pose
        self.x += linear_x * (-np.sin(self.yaw)) * DT
        self.z += linear_x * (-np.cos(self.yaw)) * DT
        self.yaw += angular_y * DT

        # Update terrain height and agent Y
        self.terrain_y = terrain_height_at(self.x, self.z)
        self.y = self.terrain_y + INITIAL_CLEARANCE

        self.set_agent_pose()

        # Log and save
        if SAVE_FRAMES:
            save_frame(
                obs, self.step_idx, self.x, self.y, self.z, self.yaw,
                linear_x, angular_y, distance, bearing_error
            )

        # Print to console
        print(
            f"step {self.step_idx:4d} | "
            f"pos=({self.x:7.3f}, {self.z:7.3f}) | "
            f"yaw={np.rad2deg(self.yaw):7.1f}° | "
            f"distance={distance:6.3f}m | "
            f"heading_err={np.rad2deg(bearing_error):7.1f}° | "
            f"v_lin={linear_x:6.3f} v_ang={np.rad2deg(angular_y):7.1f}°/s"
        )

        self.step_idx += 1

        return self.controller.at_target

    def close(self):
        """Cleanup."""
        try:
            self.sim.close()
        except Exception:
            pass

    def run(self):
        """Main simulation loop."""
        print(f"\n\n [inf] VLM Navigation GoToGoal Controller \n\n")
        print(f"Target: ({TARGET_X:.2f}, {self.target_y:.2f}, {TARGET_Z:.2f})")
        print(f"Start:  ({START_X:.2f}, {self.y:.2f}, {START_Z:.2f})")
        print(f"Standoff distance: {STANDOFF_DISTANCE:.2f}m")
        print(f"Max steps: {MAX_STEPS}")
        print(f"Output directory: {OUT_DIR}\n")

        try:
            for step in range(MAX_STEPS):
                at_target = self.step()
                if at_target:
                    print(f"\n[inf] Target reached in {step} steps!")
                    break
        finally:
            self.close()
            print(f"\n[inf] Outputs saved to: {OUT_DIR}")

            # Create videos from frames
            if SAVE_FRAMES:
                print(f"\n[inf] Generating videos from frames")
                create_video_from_frames(OUT_DIR, "rgb")
                create_video_from_frames(OUT_DIR, "depth")
                print(f"[inf] Video generation complete!")


if __name__ == "__main__":
    rover = RoverSim()
    rover.run()
