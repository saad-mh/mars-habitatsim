"""
Builds text prompts sent to Qwen2.5-VL for goal selection and driving.
"""


def build_ghost_mask_belief_prompt(
    bearing_deg: float,
    distance_m: float,
    bearing_uncertainty_deg: float,
    distance_uncertainty_m: float,
    frame_w: int,
    frame_h: int,
    min_radius_px: float,
    max_radius_px: float,
) -> str:
    """GHOST_MASK placement prompt for a full Gaussian belief (mean +
    covariance in robot [forward, left] frame), already reduced by
    sam_vla.core.ghost_mask.belief_to_bearing_range_uncertainty to bearing,
    distance, and two independent uncertainty scalars -- bearing_uncertainty
    (how sure of the direction) and distance_uncertainty (how sure of the
    range). The goal is frequently entirely out of frame (behind or far to
    the side of the camera), so this does not attempt a geometric pixel
    projection -- the model must nudge the mask toward the correct screen
    edge from the text description alone, the same way
    vl_direction/prompts/ghost_mask_prompt.py's scalar-uncertainty version
    does, generalized here to an anisotropic ellipse: bearing_uncertainty
    sizes the horizontal (u) radius, distance_uncertainty sizes the vertical
    (v) radius, so a confidently-bearinged-but-distance-vague belief comes
    out tall and narrow rather than a uniform blob.
    """
    cx = frame_w / 2.0
    side = "left" if bearing_deg > 0 else "right" if bearing_deg < 0 else "ahead"
    behind = abs(bearing_deg) > 90.0

    behind_note = (
        f" The bearing magnitude ({bearing_deg:+.0f} degrees) means the goal is "
        f"actually BEHIND the rover, not just off to one side -- pin the ellipse "
        f"at the extreme {side} edge of the frame, as close to the edge as "
        f"possible, rather than partway across.\n"
        if behind
        else ""
    )

    system_framing = (
        "You are a spatial-reasoning assistant for a Mars rover. You never explain, "
        "you only answer with JSON."
    )
    context_block = (
        f"The rover's goal is currently out of view. Belief estimate (from the "
        f"rover's tracked position belief, not direct observation): bearing "
        f"{bearing_deg:+.0f} degrees from straight ahead ({side}), distance "
        f"{distance_m:.1f}m.\n"
        f"Direction confidence: bearing uncertainty is +/-{bearing_uncertainty_deg:.0f} "
        f"degrees (small = confident in the {side} direction, large = the goal "
        f"could be anywhere across a wide arc).\n"
        f"Distance confidence: distance uncertainty is +/-{distance_uncertainty_m:.1f}m "
        f"(small = confident how far away, large = could be much nearer or farther).\n"
        f"Frame size is {frame_w}x{frame_h} pixels; horizontal center is u={cx:.0f}.\n"
        f"{behind_note}"
        f"Place an ellipse marking where you believe the goal most likely is. The "
        f"bearing says {side} -- unless it is within a few degrees of straight "
        f"ahead, the ellipse center must sit clearly off-center toward the {side} "
        f"edge, NOT near u={cx:.0f}. Scale how far off-center by the bearing "
        f"magnitude: a bearing near +/-90 degrees or beyond should land at or "
        f"past the frame's {side} edge, a small bearing (e.g. +/-10) only "
        f"slightly off center. u={cx:.0f} means 'straight ahead', which would "
        f"contradict the goal being out of view.\n"
        f"Grow radius_u_px (horizontal) with bearing uncertainty and radius_v_px "
        f"(vertical) with distance uncertainty, independently -- they do not have "
        f"to match. Both stay within [{min_radius_px:.0f}, {max_radius_px:.0f}]px. "
        f"Keep the ellipse's center within the frame even when it sits at the "
        f"extreme edge."
    )
    json_schema_line = (
        'Respond with exactly one JSON object: {"u": <int>, "v": <int>, '
        '"radius_u_px": <int>, "radius_v_px": <int>} -- u is the horizontal pixel '
        "coordinate (0=left edge, increasing right), v is the vertical pixel "
        "coordinate (0=top edge, increasing down), radius_u_px/radius_v_px are the "
        "ellipse's horizontal/vertical radii in pixels. No other text."
    )

    if bearing_deg > 5.0:
        example_u = frame_w * 0.15
    elif bearing_deg < -5.0:
        example_u = frame_w * 0.85
    else:
        example_u = frame_w * 0.5
    example_v = frame_h * 0.45
    example_ru = min(max(min_radius_px, 50.0), max_radius_px)
    example_rv = min(max(min_radius_px, 90.0), max_radius_px)

    return (
        f"{system_framing}\n\n"
        f"{context_block}\n\n"
        f"{json_schema_line}\n"
        f"Example (format only -- your own numbers must reflect the actual bearing "
        f'and uncertainties above, not copy these): {{"u": {int(example_u)}, '
        f'"v": {int(example_v)}, "radius_u_px": {int(example_ru)}, '
        f'"radius_v_px": {int(example_rv)}}}'
    )


def build_select_goal_prompt(detections: list[dict]) -> str:
    lines = []
    for i, det in enumerate(detections):
        lines.append(
            f'{i}: class="{det["class_name"]}", bbox_norm={det["bbox_norm"]}, '
            f'confidence={det["confidence"]:.2f}'
        )
    detections_block = "\n".join(lines)

    return (
        "You are the vision system for a Mars rover. The image shows the "
        "rover's current camera view. Below is a list of detected rock "
        "instances, each with a normalized bounding box [x_min, y_min, x_max, "
        "y_max] in [0, 1] image coordinates.\n\n"
        f"{detections_block}\n\n"
        "Pick exactly ONE detection from the list above to serve as the "
        "rover's navigation goal (the rock the rover should drive to)."
        "Respond with ONLY a JSON object in this exact format, no other text:\n"
        '{"goal_index": <int, index into the list above>, '
        '"reasoning": <str, brief explanation>}'
    )


def build_goal_vocabulary_prompt() -> str:
    """For the multi-goal path: instead of picking one detection as the goal,
    Qwen describes a small open-vocabulary set of goal-worthy object terms
    visible in the scene (fed to SAM3's per-term text prompting and CLIP's
    text-embedding bank) plus one instruction sentence covering all of them."""
    return (
        "You are the vision system for a Mars rover. The image shows the "
        "rover's current camera view. The rover will visit multiple "
        "goal-worthy objects in this scene over the course of the episode, "
        "not just one.\n\n"
        "List a small set of short, visually distinct object terms (2-4 "
        "terms) describing the categories of rocks or other objects in this "
        'scene that would make good navigation targets, e.g. "small rock" '
        'vs "big rock" if there is a clear size distinction, or terms '
        "based on shape/color if that better separates the objects you see. "
        "Also give one instruction sentence describing the rover's overall "
        "task across all of them.\n\n"
        "Respond with ONLY a JSON object in this exact format, no other "
        "text:\n"
        '{"terms": [<str>, ...], "instruction_text": <str>, '
        '"reasoning": <str, brief explanation>}'
    )


def build_drive_action_prompt(instruction_text: str, frame_idx: int) -> str:
    return (
        "You are the driving policy for a Mars rover. The image is the "
        f"rover's current camera frame (frame {frame_idx}).\n\n"
        f"Navigation instruction: {instruction_text}\n\n"
        "Output the rover's next action as ONLY a JSON object, no other "
        "text, in this exact format:\n"
        '{"v_fwd": <float in [0, 1]>, "v_lat": <float in [-1, 1]>, '
        '"yaw_rate": <float in [-1, 1]>, "reasoning": <str>}\n\n'
        "v_fwd is normalized forward speed (0 = stop, 1 = full speed), "
        "v_lat is normalized lateral speed (negative = left, positive = "
        "right), and yaw_rate is normalized turn rate (negative = turn "
        "left, positive = turn right). Steer away from any obstacles "
        "visible in the current frame while making progress toward the "
        "goal described in the instruction."
    )


def build_parse_nav_command_prompt(command_text: str) -> str:
    """nav/gui.py's free-text Command panel: segments a natural-language nav
    command into an ordered list of distinct sub-goals -- each either a
    short movement instruction or a short phrase naming a specific target
    object -- so a multi-target command like "go left, find a flag, then
    come back to home base" can eventually drive goal-sequencing one target
    at a time instead of being handed to goal resolution as one opaque
    string. The image is scene context only (nothing here is graded against
    it yet)."""
    return (
        "You are a command parser for a Mars rover. The image is the "
        "rover's current camera view, given for context only.\n\n"
        f'Rover command: "{command_text}"\n\n'
        "Break this command into an ordered list of distinct sub-goals the "
        "rover should pursue in sequence. Each item is either a short "
        'movement instruction (e.g. "go left") or a short phrase naming a '
        'specific target object to find or navigate to (e.g. "white flag", '
        '"flag on the right", "home base"). Preserve the order implied by '
        "the command, and keep each phrase as short as possible while "
        "still distinguishing it from any other target in the same "
        'command (e.g. keep "on the right" only if there is more than one '
        "flag to tell apart).\n\n"
        "Examples:\n"
        '  "go left find a flag and come back to home base" -> '
        '["go left", "flag", "home base"]\n'
        '  "find the white flag and then go to the flag on the right" -> '
        '["white flag", "flag on the right"]\n\n'
        "Respond with ONLY a JSON object in this exact format, no other "
        'text:\n{"targets": [<str>, ...]}'
    )


def build_ground_object_prompt(target_text: str) -> str:
    """Open-vocabulary grounding: unlike build_select_goal_prompt (which
    picks among SAM2's already-detected rock instances), this asks Qwen to
    localize a named object directly from the raw image with no detector
    upstream -- for object classes SAM2/SAM2-LoRA wasn't trained on (e.g.
    the placed flag markers, the home-base ghost cuboid, see
    sam_vla/env/flag_placement.py, sam_vla/env/home_base.py), not for rocks,
    which stay on the SAM2 path."""
    return (
        "You are the vision system for a Mars rover. The image shows the "
        f'rover\'s current camera view. Find the "{target_text}" in this '
        "image.\n\n"
        "If it is visible, respond with ONLY a JSON object in this exact "
        "format, no other text:\n"
        '{"found": true, "bbox_norm": [x_min, y_min, x_max, y_max], '
        '"reasoning": <str, brief explanation>}\n'
        "bbox_norm is a normalized bounding box tightly around the object, "
        "each value in [0, 1] image coordinates (x: 0=left edge, "
        "1=right edge; y: 0=top edge, 1=bottom edge).\n\n"
        f'If the "{target_text}" is not visible anywhere in the image, '
        'respond with ONLY {"found": false, "bbox_norm": null, '
        '"reasoning": <str, brief explanation>}'
    )


def build_direction_prompt(instruction_text: str, frame_idx: int) -> str:
    """Same intent as build_drive_action_prompt, but constrains the model to a
    single discrete steering choice instead of a free continuous action. The
    goal region is overlaid in green and obstacle regions in red directly on
    the image (see perception.semantic_overlay), so the instruction leans on
    those overlays rather than the raw scene."""
    return (
        "You are the driving policy for a Mars rover. The image is the "
        f"rover's current camera frame (frame {frame_idx}). The navigation "
        "goal is highlighted with a GREEN overlay and known obstacles are "
        "highlighted with a RED overlay.\n\n"
        f"Navigation instruction: {instruction_text}\n\n"
        "Choose exactly ONE discrete direction for the rover's next move:\n"
        '  "forward"    - drive straight ahead\n'
        '  "turn_left"  - steer left\n'
        '  "turn_right" - steer right\n\n'
        "Pick whichever direction makes the most progress toward the green "
        "goal region while steering clear of any red obstacle regions. Do "
        "not output speeds or turn rates, only the discrete direction.\n\n"
        "Respond with ONLY a JSON object, no other text, in this exact "
        "format:\n"
        '{"direction": <"forward" | "turn_left" | "turn_right">, '
        '"reasoning": <str>}'
    )


if __name__ == "__main__":
    dummy_detections = [
        {
            "class_name": "rock",
            "bbox_norm": [0.12, 0.30, 0.28, 0.55],
            "confidence": 0.91,
        },
        {
            "class_name": "rock",
            "bbox_norm": [0.60, 0.40, 0.80, 0.70],
            "confidence": 0.77,
        },
        {
            "class_name": "obstacle",
            "bbox_norm": [0.40, 0.10, 0.55, 0.35],
            "confidence": 0.65,
        },
    ]

    print("=== build_select_goal_prompt ===")
    print(build_select_goal_prompt(dummy_detections))
    print()
    print("=== build_drive_action_prompt ===")
    print(build_drive_action_prompt("Drive toward the large rock cluster ahead.", 42))
    print()
    print("=== build_direction_prompt ===")
    print(build_direction_prompt("Drive toward the large rock cluster ahead.", 42))
    print()
    print("=== build_parse_nav_command_prompt ===")
    print(
        build_parse_nav_command_prompt(
            "go left find a flag and come back to home base"
        )
    )
