# VLM Navigation - Interactive Frame Capture & Annotation

## Overview

`vlm_nav_interactive.py` is a standalone interactive script for capturing frames from the Mars rover navigation simulation and immediately annotating them with labelme for VLM-driven target selection.

## Features

- **Interactive windowed display** (tkinter-based, 480×640 RGB stream)
- **On-demand frame capture** via spacebar (saves RGB + Depth frames)
- **Automatic labelme launch** immediately after capture
- **Annotation validation** (structural JSON check against labelme schema)
- **Sequential frame numbering** (rgb_0000.png, rgb_0001.png, etc.)

## Setup

### 1. Verify environment

The script requires the `habitat` conda environment:

```bash
conda activate habitat
```

### 2. Verify labelme in annotate environment

```bash
/home/nahar/miniconda3/envs/annotate/bin/labelme --version
# Expected: Labelme 6.3.1+
```

### 3. Verify labels.txt exists

```bash
ls -la labels.txt
# Should exist with labels: sand, rock
```

## Usage

```bash
conda activate habitat
python vlm_nav_interactive.py
```

### Controls

| Key | Action |
|-----|--------|
| **SPACE** | Capture current frame (RGB + Depth) and launch labelme |
| **Q** | Quit |

### Workflow

1. Window opens showing live RGB stream from the rover's camera
2. Navigate using the simulation (currently fixed start pose: x=12.2, z=10.0, yaw=10°)
3. When you want to annotate: **press SPACE**
   - Saves `vlm_nav_out/rgb_0000.png` and `depth_0000.png`
   - Launches labelme on the RGB frame
   - Window blocks until you finish annotating and close labelme
   - Validates `annotations/rgb_0000.json` and prints status
4. Press **Q** to exit the interactive loop

### Output

- `vlm_nav_out/rgb_XXXX.png` — Captured RGB frames
- `vlm_nav_out/depth_XXXX.png` — Corresponding depth frames (8-bit visualization)
- `annotations/rgb_XXXX.json` — Labelme annotation JSON (one per captured frame)

### Example Annotation JSON

After labeling rocks and sand in labelme, the JSON will contain:

```json
{
  "version": "6.3.1",
  "flags": {},
  "shapes": [
    {
      "label": "rock",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "rectangle",
      ...
    },
    {
      "label": "sand",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "rectangle",
      ...
    }
  ],
  "imagePath": "../vlm_nav_out/rgb_0000.png",
  "imageHeight": 480,
  "imageWidth": 640
}
```

## Validation

The script validates each annotation JSON for:
- ✅ File exists
- ✅ Valid JSON structure
- ✅ Required keys: `version`, `flags`, `shapes`, `imagePath`, `imageHeight`, `imageWidth`
- ✅ Each shape has `label` (non-empty string) and `points` (list of [x, y] pairs)
- ✅ At least 2 points per shape

On success: prints `"accepted"`  
On failure: prints a specific error message (e.g., `"shape 0: 'label' must be a non-empty string"`)

## Scene Configuration

Reuses the same scene setup as `vlm_nav_demo.py`:

- **Scene**: `marsyard2022_tri.glb`
- **Heightmap**: `marsyard2022_terrain_hm.png` (terrain-aligned camera)
- **Resolution**: 480×640 (RGB & Depth)
- **Start pose**: x=12.2, z=10.0, yaw=10° (fixed, no teleoperation yet)
- **Physics**: disabled

## Troubleshooting

### Labelme doesn't launch

- Check labelme binary exists: `ls /home/nahar/miniconda3/envs/annotate/bin/labelme`
- Check `labels.txt` in current directory
- Ensure `annotations/` and `masks/` directories can be created (write permissions)

### Annotation not saved

- Verify you closed the labelme window (don't just minimize)
- Check `annotations/` directory for the JSON file
- Look at the console output for the specific error message

### Window doesn't appear or is frozen

- Ensure you have X11/display forwarding if using SSH
- Try setting `DISPLAY` if needed

## Next Steps (Future Integration)

This script produces annotated frames in a standardized format ready for:
1. VLM prompt construction (pass RGB + annotation labels to VLM for target selection)
2. Bbox-to-3D back-projection (convert image-space annotations to 3D world coordinates)
3. Integration with go-to-goal controller (`vlm_nav_demo.py`)

---

For questions or issues, refer to the main README.md and vlm_nav_demo.py for context on the go-to-goal controller.
