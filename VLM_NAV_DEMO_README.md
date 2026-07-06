# VLM Navigation Demo for Mars Rover

## Overview

Standalone Habitat-Sim demo implementing a proportional go-to-goal controller for autonomous rover navigation. Reuses the same scene, sensor configuration, and rover setup from `kb_teleop.py`.

## What's Implemented

### Scene & Sensor Configuration
- **Scene**: `marsyard2022_tri.glb` (same as kb_teleop.py)
- **Heightmap**: Dynamic terrain height interpolation
- **Sensors**: RGB (480×640) + Depth (480×640), captured every frame
- **Physics**: Disabled (visual simulation only)
- **Rover**: Habitat's default agent (no URDF articulation)

### Go-to-Goal Controller
Proportional controller that:
1. **Computes bearing** to target using global XZ coordinates
2. **Calculates heading error** (wrap to [-π, π])
3. **Selects behavior**:
   - If |heading_error| > 15°: **Rotate in place** (proportional angular velocity)
   - Else: **Drive forward** with minor heading correction
   - When distance ≤ 0.5m standoff: **Stop and face target**

### Action Space
**Continuous velocity control via pose integration** — matches the `ht_vel_client.py` interface expectations:
- `linear_x` (m/s): forward velocity
- `angular_y` (rad/s): rotation rate around Y axis
- Velocities are integrated directly into rover pose each timestep

### Output & Logging
- **RGB frames**: `rgb_XXXX.png` — raw sensor observations
- **Depth frames**: `depth_XXXX.png` — normalized [0, 255] depth visualization
- **RGB video**: `rgb_video.mp4` — 10 fps H.264 video of full navigation sequence
- **Depth video**: `depth_video.mp4` — 10 fps H.264 video of depth observations
- **Navigation log** (`nav_log.txt`): Step-by-step state for analysis
  - Pose, velocity, distance-to-target, bearing error
  - Compact format optimized for post-run analysis

**Videos are automatically generated at run completion** using ffmpeg. Both are stored in the output folder alongside frame images and log file. If ffmpeg is not installed, use: `sudo apt-get install ffmpeg`

## Running the Demo

```bash
source activate habitat
python vlm_nav_demo.py
```

The rover will:
1. Start at (0, 8) facing forward (yaw=0°)
2. Navigate toward hardcoded target at (0, -5) with 0.5m standoff
3. Reach target and stop in ~251 steps (~25 seconds)
4. Save outputs to `vlm_nav_out_TIMESTAMP/`
5. **Automatically generate** `rgb_video.mp4` and `depth_video.mp4` from frame sequences

**Example output structure:**
```
vlm_nav_out_1783326084/
├── rgb_0000.png ... rgb_0251.png       (252 RGB frames)
├── depth_0000.png ... depth_0251.png   (252 depth frames)
├── rgb_video.mp4                        (auto-generated, ~25 sec @ 10 fps)
├── depth_video.mp4                      (auto-generated, ~25 sec @ 10 fps)
└── nav_log.txt                          (full step-by-step log)
```

## Customization

Edit these constants in `vlm_nav_demo.py`:

```python
# Target position (modify the XZ coordinates)
TARGET_X = 0.0
TARGET_Z = -5.0

# Controller tuning
HEADING_ERROR_THRESHOLD = np.deg2rad(15.0)  # When to rotate vs. drive
MAX_LINEAR_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = np.deg2rad(45.0)  # rad/s
STANDOFF_DISTANCE = 0.5  # Stop when this close
```

## Future Work: Obstacle Avoidance

A stub function `obstacle_avoidance_hook()` in the `GoToGoalController` class is ready for extension:

```python
def obstacle_avoidance_hook(self, linear_x, angular_y, depth_frame):
    # TODO: Implement obstacle avoidance here
    # - Compute obstacle distance from depth_frame
    # - Modify velocities if obstacles detected
    return linear_x, angular_y  # Currently passes through unchanged
```

This will be implemented in a follow-up task.

## Design Notes

**Why continuous velocity control?**
- Better fit for proportional bearing/distance feedback
- Matches `ht_vel_client.py` interface (linear_x, angular_y)
- Smoother control than discrete turn/move actions
- Easier to implement heading corrections

**Why direct pose integration (not Habitat actions)?**
- Matches `kb_teleop.py` approach (proven working)
- Cleaner for research/demo code with physics disabled
- Full control over dynamics without action-space constraints

**Coordinate system (inherited from Habitat):**
- +X: right
- +Y: up
- +Z: backward
- Forward movement at yaw=0°: moves in -Z direction
- Bearing calculation: `atan2(dx, -dz)` gives angle where 0° = forward
