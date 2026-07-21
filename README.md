# Mars Habitat Sim

This repository contains Python scripts for 3D Mars terrain conversion, automated and interactive Habitat-Sim setups, telemetry control via UDP, and post-processing tools to compile frame recordings into videos.

---

## Table of Contents

1. [Pipeline Overview](#-pipeline-overview)
2. [Setup & Requirements](#-setup--requirements)
3. [Script Directions & Usage](#-script-directions--usage)
   - [3D Model Processing](#1-3d-model-processing)
     - [hm2obj.py (Heightmap to OBJ)](#hm2objpy)
     - [obj2glb.py (OBJ to GLB)](#obj2glbpy)
   - [Interactive Simulators](#2-interactive-simulators)
     - [kb_teleop.py (Keyboard Teleoperation)](#kb_teleoppy)
     - [ht_vel_server.py (UDP Velocity Server)](#ht_vel_serverpy)
     - [ht_vel_client.py (UDP Velocity Client)](#ht_vel_clientpy)
   - [Automated Scripts & Tests](#3-automated-scripts--tests)
     - [rgbd_drive.py (Automated Heightmap Following)](#rgbd_drivepy)
     - [rgbd_test.py (Multi-pose Static Test)](#rgbd_testpy)
     - [round1_walk_rgbd.py (Habitat Action Verification)](#round1_walk_rgbdpy)
   - [Post-Processing](#4-post-processing)
     - [pix2vid.py (Frame Compiler)](#pix2vidpy)

---

## Pipeline Overview

```mermaid
graph TD
    HM -->|hm2obj.py| OBJ[OBJ Mesh]
    OBJ -->|obj2glb.py| GLB[GLB Scene Asset]
    GLB --> Sim[Habitat Simulators]
    Sim -->|Recordings| PNGs[RGB/Depth Frames]
    PNGs -->|pix2vid.py| MP4[MP4 Videos]
```

---

## Setup & Requirements

Ensure you have a conda/python environment with the required dependencies installed:

- **Habitat-Sim**: `conda install -c aihabitat -c conda-forge habitat-sim`
- **Blender (with python API)**: Required for running `obj2glb.py` background conversions.
- **Python Libraries**: `numpy`, `pillow`, `numpy-quaternion`, `tkinter` (for GUIs)
- **FFmpeg**: Required for compiling videos in `pix2vid.py`.

---

## Script Directions & Usage

### 1. 3D Model Processing

#### [hm2obj.py](file:///marsHabitat/hm2obj.py)

Converts a 2D grayscale heightmap and a texture map into a 3D terrain mesh in Wavefront OBJ format. It normalizes heights, centers the terrain around the origin, downsamples the resolution using a stride to optimize performance, and generates corresponding UV coordinates.

- **Usage:**
  ```bash
  python hm2obj.py \
    --heightmap <path_to_heightmap_png> \
    --texture <path_to_texture_png> \
    --size-x <width_meters> \
    --size-y <length_meters> \
    --size-z <height_meters> \
    [--out <output_obj_path>] \
    [--stride <int>]
  ```
- **Arguments:**
  - `--heightmap`: (Required) Path to the heightmap image.
  - `--texture`: (Required) Path to the texture image.
  - `--size-x` / `--size-y` / `--size-z`: (Required) Dimensions of the generated terrain in meters.
  - `--out`: Output OBJ file name (defaults to `marsyard2022.obj`).
  - `--stride`: Downsampling factor for terrain grid density (defaults to `4` for optimal performance).

---

#### [obj2glb.py](file:///marsHabitat/obj2glb.py)

A Blender-dependent script to convert the generated OBJ model into a GLB file compatible with `habitat_sim`. It clears the Blender scene, imports the OBJ mesh, applies object scaling, and exports the final GLB asset.

- **Usage:**
  Run using Blender in background mode:
  ```bash
  blender --background --python obj2glb.py -- <input_mesh.obj> <output_mesh.glb>
  ```

---

### 2. Interactive Simulators

#### [kb_teleop.py](file:///kb_teleop.py)

A Tkinter-based interactive GUI application that spawns the agent inside the Habitat environment. It reads the heightmap dynamically, aligning the agent's height (`Y`) with the terrain as it walks.

- **Usage:**
  ```bash
  python kb_teleop.py
  ```
- **Controls (Tkinter Window Focus):**
  - `W` / `S`: Move forward / backward.
  - `A` / `D`: Rotate left / right.
  - `Q` / `E`: Decrease / Increase clearance height above the terrain.
  - `Space`: Toggle camera frame recording. Saves matching RGB/depth frame sequences.
  - `P`: Capture and save a single frame.
  - `X` / `Escape`: Safely close the simulator.
- **Output:** Saves recorded runs to a newly created folder matching `mars_teleop_out<timestamp>/` containing RGB, depth, and a `poses.txt` telemetry log.

---

#### [ht_vel_server.py](file:///ht_vel_server.py)

A Tkinter GUI server that starts the Habitat simulation and opens a UDP socket (default `127.0.0.1:5055`) to listen for incoming movement and utility commands. Like `kb_teleop.py`, it supports terrain height alignment.

- **Usage:**
  ```bash
  python ht_vel_server.py
  ```
- **UDP Command Interface:**
  The server listens for strings in the following formats:
  - `ht_vel <linear_x> <angular_y>`: Command linear velocity (m/s) and turning rate (rad/s).
  - `ht_stop`: Stop all movement.
  - `ht_rec [on|off|toggle]`: Control frame recording.
  - `ht_save`: Save current frame.
  - `ht_clearance <meters>`: Dynamically adjust camera height.
  - `ht_quit`: Shutdown server.
- **Output:** Records frame sequence outputs in folders matching `mars_ht_vel_out<timestamp>/`.

---

#### [ht_vel_client.py](file:///ht_vel_client.py)

A command-line script to send UDP control signals to the running `ht_vel_server.py`.

- **Usage:**

  ```bash
  # Send velocity commands (if duration is specified, loops client command at a rate, then stops)
  python ht_vel_client.py vel <linear_x> <angular_y> [--rate <hz>] [--duration <seconds>]

  # Trigger other utility commands
  python ht_vel_client.py stop
  python ht_vel_client.py rec <on|off|toggle>
  python ht_vel_client.py save
  python ht_vel_client.py clearance <meters>
  python ht_vel_client.py quit
  ```

---

### 3. Automated Scripts & Tests

#### [rgbd_drive.py](file:///rgbd_drive.py)

Runs a script-controlled traverse across the Marsyard terrain. The agent moves along a straight line in the X-axis (`X` from `-20` to `20` meters at `Z = 8`), dynamically queries the heightmap coordinates, takes RGB-D observations, and outputs them sequentially.

- **Usage:**
  ```bash
  python rgbd_drive.py
  ```
- **Output:** Saves 120 steps to `mars_follow_terrain_out/`.

---

#### [rgbd_test.py](file:///rgbd_test.py)

A validation script to verify that the Habitat simulator is rendering correctly by capturing screenshots from 4 static coordinates in 3D space.

- **Usage:**
  ```bash
  python rgbd_test.py
  ```
- **Output:** Saves verification observations to `mars_rgbd_test_out/`.

---

#### [round1_walk_rgbd.py](file:///round1_walk_rgbd.py)

A test runner configured to use Habitat's default discrete action space (`move_forward`, `turn_left`, `turn_right`) inside a standard test scene (`skokloster-castle.glb`). It performs a short walk and saves observations.

- **Usage:**
  ```bash
  python round1_walk_rgbd.py
  ```
- **Output:** Saves output frames to `round1_outputs/`.

---

### 4. Post-Processing

#### [pix2vid.py](file:///pix2vid.py)

Compiles saved frame folders into high-quality MP4 videos. By default, it will look for the latest `mars_teleop_out*` directory, extract all RGB and Depth PNG images, and compile them into separate `.mp4` video files.

- **Usage:**
  ```bash
  python pix2vid.py [--input <folder>] [--fps <fps>] [--rgb-out <path>] [--depth-out <path>]
  ```
- **Arguments:**
  - `--input`: Recording directory. If omitted, uses the latest folder matching `mars_teleop_out*`.
  - `--fps`: Target frames per second (defaults to `15`).
  - `--rgb-out` / `--depth-out`: Path to write output MP4 files. Defaults to `<input_folder>/rgb_video.mp4` and `<input_folder>/depth_video.mp4`.
