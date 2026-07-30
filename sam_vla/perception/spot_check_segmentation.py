"""Renders human-checkable overlay images from a segmentation-sweep run
(sam_vla.run_segmentation_sweep): alpha-blends the category mask over RGB
(one color per class, extending perception.semantic_overlay's blend approach
from 2 fixed ids to however many classes a run used) and draws a bounding
box + label (mesh_id, category) for every object in the frame's record.
Serves next.md Step 6's "manually spot-check ~20 frames: does the projected
mask align with the RGB rock silhouette" -- without opening RGB/mask pairs
by hand. Saved to disk under <run_dir>/spot_check/ so the check is a
persistent artifact of the run, not a throwaway plot.

Usage:
    python -m sam_vla.perception.spot_check_segmentation --run-dir <run_dir> [--out-dir <out_dir>] [--n <n>] [--seed <seed>]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

_PALETTE: List[Tuple[int, int, int]] = [
    (255, 196, 0),  # amber -- small_rock by convention (first non-background class)
    (220, 20, 60),  # crimson -- big_rock
    (30, 144, 255),  # dodger blue -- bedrock
    (218, 112, 214),  # orchid -- hole_in_ground
    (50, 205, 50),  # lime green -- overflow classes
    (255, 140, 0),  # dark orange
]


def build_palette(class_names: Sequence[str]) -> Dict[str, Tuple[int, int, int]]:
    """class_names[0] is always background (see segmentation_capture.build_category_lut)
    and gets no color -- background pixels are left untouched by the overlay."""
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(class_names[1:])}


def overlay_category_mask(
    rgb: np.ndarray,
    category_mask: np.ndarray,
    class_names: Sequence[str],
    palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    alpha: float = 0.45,
) -> np.ndarray:
    palette = palette or build_palette(class_names)
    overlaid = np.asarray(rgb, dtype=np.float32).copy()
    category_mask = np.asarray(category_mask)
    for class_idx, name in enumerate(class_names[1:], start=1):
        color = np.array(palette[name], dtype=np.float32)
        pixel_mask = category_mask == class_idx
        overlaid[pixel_mask] = (1.0 - alpha) * overlaid[pixel_mask] + alpha * color
    return np.clip(overlaid, 0, 255).astype(np.uint8)


def draw_object_labels(
    image: np.ndarray,
    objects: List[dict],
    palette: Dict[str, Tuple[int, int, int]],
    min_label_area: int = 150,
) -> np.ndarray:
    """Draws a bounding box + `"<mesh_id> <category>"` label for every object
    record (as stored in segmentation_frames.jsonl -- mesh_id, category, bbox),
    color-matched to the same per-class palette used for the mask blend.
    Objects smaller than `min_label_area` px get the outline only, no text --
    a scene can have 100+ small_rock hulls in one frame, and unconditional
    text labels there just overlap into unreadable clutter; the outline
    alone still shows where every detection landed."""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for obj in objects:
        x, y, w, h = obj["bbox"]
        color = palette.get(obj["category"], (255, 255, 255))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
        if w * h < min_label_area:
            continue
        label = f"{obj['mesh_id']} {obj['category']}"
        text_y = y - 10 if y - 10 >= 0 else y + h + 1
        draw.rectangle([x, text_y, x + 6 * len(label), text_y + 9], fill=(0, 0, 0))
        draw.text((x + 1, text_y), label, fill=color)
    return np.asarray(img, dtype=np.uint8)


def spot_check_run(
    run_dir: Path, out_dir: Optional[Path] = None, n: int = 20, seed: int = 0
) -> List[Path]:
    """Sample up to `n` frames from run_dir/segmentation_frames.jsonl, render
    mask-overlay + per-object labels for each, and write them to
    out_dir (default run_dir/spot_check/). Returns the written paths."""
    import imageio.v3 as iio

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "spot_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "summary.json").read_text())
    class_names = summary["class_names"]
    palette = build_palette(class_names)

    lines = (run_dir / "segmentation_frames.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    sample = records if len(records) <= n else random.Random(seed).sample(records, n)

    written: List[Path] = []
    for rec in sample:
        rgb = iio.imread(run_dir / rec["rgb_path"])
        category_mask = iio.imread(run_dir / rec["category_mask_path"])
        overlaid = overlay_category_mask(rgb, category_mask, class_names, palette)
        annotated = draw_object_labels(overlaid, rec["objects"], palette)

        out_path = out_dir / f"{rec['frame_id']}.png"
        iio.imwrite(out_path, annotated)
        written.append(out_path)

    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="a run directory produced by run_segmentation_sweep.py",
    )
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/spot_check/")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    written = spot_check_run(
        Path(args.run_dir),
        Path(args.out_dir) if args.out_dir else None,
        args.n,
        args.seed,
    )
    out_dir = (
        written[0].parent
        if written
        else (Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "spot_check")
    )
    print(f"wrote {len(written)} spot-check overlays -> {out_dir}")
