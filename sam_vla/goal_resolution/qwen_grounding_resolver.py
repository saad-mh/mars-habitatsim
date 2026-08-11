"""Open-vocabulary goal resolution via direct Qwen grounding, for object
classes SAM2/SAM2-LoRA wasn't trained on and that first_frame_resolver
deliberately never sees as goal/obstacle candidates (the placed flag
markers, the home-base ghost cuboid -- see FLAG_SEMANTIC_ID/
HOME_BASE_SEMANTIC_ID in sam_vla.core.goal_geometry). Mirrors
first_frame_resolver's resolve/resolve_verbose -> GoalSpec shape exactly so
nav/rover_controller.py's _do_resolve can swap between the two by whether a
target_text was given, but the goal bbox here comes straight from
qwen_client.ground_object_verbose instead of a SAM2 detection Qwen picks
among. SAM2 still runs in this module -- only to seed obstacle_bboxes_norm
(rocks etc.) via first_frame_resolver.resolve_obstacles -- per the reason
this module exists at all: keep SAM2 for rocks, use Qwen grounding only for
open-vocabulary targets.
"""

from typing import Optional

import numpy as np

from sam_vla.core.types import Detection, GoalSpec
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.vlm import qwen_client


def resolve_verbose(
    rgb: np.ndarray,
    target_text: str,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> tuple[GoalSpec, dict, list[Detection]]:
    """Grounds `target_text` directly via Qwen (no SAM2 candidate list
    involved in picking the goal). Raises RuntimeError if Qwen doesn't find
    the object in the current frame -- the same "no usable goal this
    resolve" contract first_frame_resolver.resolve_verbose has when SAM2
    finds zero detections."""
    goal_bbox_norm, vlm_result = qwen_client.ground_object_verbose(rgb, target_text)
    if goal_bbox_norm is None:
        raise RuntimeError(
            f'Qwen did not find "{target_text}" in the current frame '
            f'(reasoning: {vlm_result.get("reasoning", "")!r})'
        )

    try:
        obstacle_detections = first_frame_resolver.resolve_obstacles(
            rgb,
            detect_rgb=detect_rgb,
            backend=backend,
            checkpoint_path=checkpoint_path,
        )
    except RuntimeError:
        # Unlike first_frame_resolver's own path, the goal here doesn't
        # depend on SAM2 finding anything -- only obstacles do, so zero
        # detections is a legitimate (obstacle-free) outcome, not a failure.
        obstacle_detections = []

    goal_spec = GoalSpec(
        goal_bbox_norm=goal_bbox_norm,
        obstacle_bboxes_norm=[d.bbox_norm for d in obstacle_detections],
        instruction_text=f"Navigate to the {target_text}.",
    )
    return goal_spec, vlm_result, obstacle_detections


def resolve(
    rgb: np.ndarray,
    target_text: str,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> GoalSpec:
    goal_spec, _vlm_result, _dets = resolve_verbose(
        rgb,
        target_text,
        detect_rgb=detect_rgb,
        backend=backend,
        checkpoint_path=checkpoint_path,
    )
    return goal_spec
