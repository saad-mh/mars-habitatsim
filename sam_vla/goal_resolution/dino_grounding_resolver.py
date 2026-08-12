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

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.goal_geometry import intrinsics_from_hfov
from sam_vla.core.types import Action, Detection, GoalSpec
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


# Above this IoU (against the DINO-grounded goal_bbox_norm), a SAM2
# "obstacle" detection is treated as the same physical object DINO just
# grounded, not a distinct one -- see _drop_goal_overlap.
_GOAL_OVERLAP_IOU_THRESHOLD = 0.3


def _iou_norm(a: tuple, b: tuple) -> float:
    """IoU of two normalized (0-1) xyxy boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _drop_goal_overlap(
    goal_bbox_norm: tuple, detections: List[Detection]
) -> List[Detection]:
    """SAM2's obstacle sweep (first_frame_resolver.resolve_obstacles) is run
    blind to what DINO just grounded -- it segments everything in frame,
    which usually includes the DINO-grounded object itself. Without this
    filter that same object ends up registered TWICE by _do_resolve: once as
    the goal mesh (from goal_bbox_norm) and again as an obstacle mesh
    (from here), overlapping it -- the CBF then steers to avoid the very
    thing it's driving toward. Drop any SAM detection that substantially
    overlaps the goal box; a real, distinct obstacle standing next to (not
    on top of) the goal keeps a low IoU and survives this filter."""
    return [
        d
        for d in detections
        if _iou_norm(goal_bbox_norm, d.bbox_norm) < _GOAL_OVERLAP_IOU_THRESHOLD
    ]


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
    else:
        # SAM2 doesn't know what DINO just grounded -- strip out whichever
        # of its detections is actually the grounded target itself so it
        # doesn't also get registered as an obstacle mesh on top of the
        # goal mesh (see _drop_goal_overlap).
        obstacle_detections = _drop_goal_overlap(goal_bbox_norm, obstacle_detections)

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


@dataclass
class SweepFrame:
    frame_idx: int
    yaw: float
    depth: Optional[np.ndarray]
    detections: Dict[str, list]  # query -> [Detection, ...]; omits queries not seen this frame


def sweep_and_detect_objects(
    env,
    object_queries: List[str],
    *,
    num_yaws: int = 8,
    dino_model_id: str = DEFAULT_DINO_MODEL_ID,
    dino_device: str = DEFAULT_DINO_DEVICE,
    dino_box_threshold: float = DEFAULT_BOX_THRESHOLD,
    dino_text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> List[SweepFrame]:
    """Rotates `env` through one full in-place turn (MarsHabitatEnv.sweep_in_place)
    and runs GroundingDINO against every query in `object_queries` on each
    captured frame -- for a caller holding a multi-object open-vocab list
    (e.g. a mission's remaining sub-goals) that wants to know, in one pass,
    which of those objects are visible from the rover's current position and
    at what heading, instead of resolve_verbose's single target_text/single
    "best" box.

    Returns one SweepFrame per captured heading, IN SWEEP ORDER (frame 0
    first) -- order matters to sweep_and_seed_beliefs, which walks this same
    sequence to dead-reckon a belief between sightings, so this returns a
    list rather than a frame_idx-keyed dict. `.detections` omits any query
    not seen in that particular frame. `env` is duck-typed (only
    sweep_in_place(num_yaws) -> list of Observation-like objects with
    .rgb/.depth/.pose/.frame_idx is required) to avoid this goal_resolution
    module importing sam_vla.env.habitat_env."""
    if not object_queries:
        raise ValueError("object_queries must be non-empty")

    detector = _get_detector(
        model_id=dino_model_id,
        device=dino_device,
        box_threshold=dino_box_threshold,
        text_threshold=dino_text_threshold,
    )

    frames: List[SweepFrame] = []
    for obs in env.sweep_in_place(num_yaws=num_yaws):
        frame_hits: Dict[str, list] = {}
        for query in object_queries:
            dets = detector.detect(obs.rgb, text_prompt=_to_prompt(query))
            if dets:
                frame_hits[query] = dets
        frames.append(
            SweepFrame(
                frame_idx=obs.frame_idx,
                yaw=obs.pose.yaw,
                depth=obs.depth,
                detections=frame_hits,
            )
        )
    return frames


def _bbox_to_body(box, depth: np.ndarray, hfov_deg: float) -> Optional[tuple]:
    """A detection's pixel box -> belief_tracking's body-frame [forward, left]
    convention: bearing from the box's horizontal center column, range from
    the MEDIAN depth over the box interior -- the bbox counterpart of
    belief_tracking.mask_to_body's mask-centroid version (same reasoning:
    median over the patch, robust to a single pixel landing on a depth
    discontinuity at the box edge). Returns None if no pixel in the box has
    valid depth, matching goal_geometry.bbox_to_world's convention for the
    same failure rather than inventing a fabricated range."""
    depth = np.asarray(depth)
    height, width = depth.shape[:2]
    x0, y0, x1, y1 = box
    ix0, ix1 = sorted((min(max(int(x0), 0), width - 1), min(max(int(x1), 0), width - 1)))
    iy0, iy1 = sorted((min(max(int(y0), 0), height - 1), min(max(int(y1), 0), height - 1)))
    patch = depth[iy0 : iy1 + 1, ix0 : ix1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    if valid.size == 0:
        return None
    rng = float(np.median(valid))
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    u = (ix0 + ix1) / 2.0
    right = (u - intr["cx"]) * rng / max(intr["fx"], 1e-6)
    return rng, -right  # (forward, left)


def sweep_and_seed_beliefs(
    env,
    goal_queries: List[str],
    belief_trackers: Dict[str, BeliefGoalTracker],
    hfov_deg: float,
    *,
    num_yaws: int = 8,
    dino_model_id: str = DEFAULT_DINO_MODEL_ID,
    dino_device: str = DEFAULT_DINO_DEVICE,
    dino_box_threshold: float = DEFAULT_BOX_THRESHOLD,
    dino_text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> List[SweepFrame]:
    """Runs sweep_and_detect_objects over `goal_queries` -- and ONLY
    `goal_queries` (a mission's remaining GO_TO/FIND sub-goal targets, not
    open-ended text) -- then, in that same sweep order, seeds/updates one
    BeliefGoalTracker per query (`belief_trackers`, keyed by query text; the
    caller must supply one entry per query, e.g. via
    RoverController._new_belief_tracker -- this function never constructs
    one itself, so callers keep control of goal_range/min_px/odom_noise).

    Whichever query is detected in a frame gets that frame's best-scoring
    box converted to a body-frame point (_bbox_to_body) and fed through
    observe_body_point -- the same reset-uncertainty-to-sigma_visible
    semantics a live mask sighting gets. Critically, EVERY tracker
    (detected-this-frame or not) is then propagate()'d by one yaw step, so a
    query sighted early (e.g. frame 2 of 8) doesn't sit at that frame's
    stale body-frame coordinates while the sweep keeps rotating through the
    remaining headings -- its belief dead-reckons along with the actual
    rotation exactly as BeliefGoalTracker.propagate already does for any
    other unseen-this-tick goal. One sweep heading is treated as one step
    (dt=1.0, yaw_rate=2*pi/num_yaws) since sweep_in_place moves by discrete
    headings, not a continuous per-second rate -- ordinary
    BeliefGoalTracker.propagate, not a workaround.

    Includes a propagate() call after the LAST frame too: sweep_in_place
    restores the agent to its pre-sweep heading once the sweep ends, a
    rotation of exactly one more yaw step (num_yaws evenly divides 2*pi, so
    the wrap-around back to the starting heading is identical in size to
    every inter-frame step) -- skipping it would leave every belief_g
    expressed relative to the second-to-last heading instead of where the
    agent actually ends up once this function returns."""
    missing = [q for q in goal_queries if q not in belief_trackers]
    if missing:
        raise KeyError(f"belief_trackers missing an entry for: {missing!r}")

    frames = sweep_and_detect_objects(
        env,
        goal_queries,
        num_yaws=num_yaws,
        dino_model_id=dino_model_id,
        dino_device=dino_device,
        dino_box_threshold=dino_box_threshold,
        dino_text_threshold=dino_text_threshold,
    )

    step_action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=2.0 * math.pi / num_yaws)

    for frame in frames:
        for query, dets in frame.detections.items():
            best = max(dets, key=lambda d: d.score)
            point = _bbox_to_body(best.box, frame.depth, hfov_deg)
            if point is not None:
                belief_trackers[query].observe_body_point(*point)
        for query in goal_queries:
            belief_trackers[query].propagate(step_action, dt=1.0)

    return frames
