"""Open-vocabulary goal resolution via direct GroundingDINO detection, for
object classes SAM2/SAM2-LoRA wasn't trained on and that first_frame_resolver
deliberately never sees as goal/obstacle candidates (the placed flag
markers, the home-base ghost cuboid -- see FLAG_SEMANTIC_ID/
HOME_BASE_SEMANTIC_ID in sam_vla.core.goal_geometry). Mirrors
qwen_grounding_resolver's resolve/resolve_verbose -> GoalSpec shape exactly
so nav/rover_controller.py's _do_resolve can call this instead without
changing anything downstream, but the grounding backend here is
navdp.extensions.GroundingDINODetector (zero-shot open-vocab detector) in
place of a Qwen VLM point-pick (qwen_client.ground_object_verbose) --
ported from scripts/vlm_nav_tests/qwen_search_dino.py's own DINO usage.
That script only uses DINO to CONFIRM a geometrically-known point (the
flag's true world position is already known there); here there is no known
position yet, so DINO's own detected box IS the goal region -- used
directly as goal_bbox_norm, which is actually a strictly better anchor for
bbox_to_world's depth-backprojection than qwen_grounding_resolver's
synthetic small patch around a single point. SAM2 still runs in this
module -- only to seed obstacle_bboxes_norm (rocks etc.) via
first_frame_resolver.resolve_obstacles -- per the reason
qwen_grounding_resolver existed at all: keep SAM2 for rocks, use the
open-vocabulary detector only for open-vocabulary targets.

GroundingDINODetector lives under navdp/navdp/extensions -- this repo's own
navdp/ package (see CLAUDE.md), not the external NavdpUpstreamPolicy
checkout -- so it needs navdp_root on sys.path first (see
sam_vla.policy.navdp_policy._add_navdp_to_path/_resolve_navdp_root).
nav/rover_controller.py's _run() already does this once before entering the
control loop (the same prerequisite the CBF safety layer has), so the
deferred `from navdp.extensions import GroundingDINODetector` below just
works from there.
"""

from typing import Optional

import numpy as np

from sam_vla.core.types import Detection, GoalSpec
from sam_vla.goal_resolution import first_frame_resolver

DEFAULT_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_DINO_DEVICE = "cuda"
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25

# Lazily constructed/cached GroundingDINODetector instances, keyed by the
# params that actually change which model/weights get loaded -- same
# module-level-cache pattern sam_segmenter._model_cache uses for SAM2, so
# repeated resolves (every "Ground Target" click, every mission GO_TO/FIND
# step) don't reload the model from disk each time.
_detector_cache: dict = {}


def _get_detector(
    model_id: str = DEFAULT_DINO_MODEL_ID,
    device: str = DEFAULT_DINO_DEVICE,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
):
    key = (model_id, device, box_threshold, text_threshold)
    if key not in _detector_cache:
        from navdp.extensions import GroundingDINODetector

        _detector_cache[key] = GroundingDINODetector(
            model_id=model_id,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    return _detector_cache[key]


def _to_prompt(target_text: str) -> str:
    # GroundingDINO convention: lowercase, ends with a period (see
    # GroundingDINODetector.detect's docstring).
    text = target_text.strip().lower()
    return text if text.endswith(".") else text + "."


def resolve_verbose(
    rgb: np.ndarray,
    target_text: str,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    dino_model_id: str = DEFAULT_DINO_MODEL_ID,
    dino_device: str = DEFAULT_DINO_DEVICE,
    dino_box_threshold: float = DEFAULT_BOX_THRESHOLD,
    dino_text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> tuple[GoalSpec, dict, list[Detection]]:
    """Grounds `target_text` directly via GroundingDINO (no SAM2 candidate
    list involved in picking the goal). Raises RuntimeError if DINO doesn't
    detect the object in the current frame -- the same "no usable goal this
    resolve" contract qwen_grounding_resolver/first_frame_resolver have."""
    detector = _get_detector(
        model_id=dino_model_id,
        device=dino_device,
        box_threshold=dino_box_threshold,
        text_threshold=dino_text_threshold,
    )
    prompt = _to_prompt(target_text)
    det = detector.detect_best(rgb, text_prompt=prompt)

    vlm_result = {
        "found": det is not None,
        "u": float(det.u) if det is not None else None,
        "v": float(det.v) if det is not None else None,
        "score": float(det.score) if det is not None else None,
        "label": det.label if det is not None else None,
        "reasoning": (
            f"GroundingDINO detected {det.label!r} score={det.score:.2f}"
            if det is not None
            else f"GroundingDINO found no box for prompt {prompt!r} "
            f"(box_threshold={dino_box_threshold})"
        ),
    }
    if det is None:
        raise RuntimeError(
            f'GroundingDINO did not find "{target_text}" in the current frame '
            f"(prompt={prompt!r}, box_threshold={dino_box_threshold})"
        )

    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = det.box
    goal_bbox_norm = (
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
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
