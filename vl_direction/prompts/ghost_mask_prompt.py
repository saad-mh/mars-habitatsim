"""
Prompt template for GHOST_MASK mode: the model is handed the belief state
(bearing/distance/uncertainty to a currently-unseen goal) as text, plus the
current frame, and decides where on the frame a translucent "ghost mask"
circle should be drawn to mark the goal's believed location -- placement is
the model's spatial-reasoning call, not caller-side trigonometry. Output is
strict JSON (u, v, radius_px in pixels), not a closed-vocabulary word, so
this mode has no few-shot LEFT/RIGHT-style exemplar and instead leans on an
explicit schema + worked example in the prompt itself.
"""

from vl_direction.schemas import GhostMaskContext

_SYSTEM_FRAMING = (
    "You are a spatial-reasoning assistant for a Mars rover. You never explain, "
    "you only answer with JSON."
)

_JSON_SCHEMA_LINE = (
    'Respond with exactly one JSON object: {"u": <int>, "v": <int>, "radius_px": <int>} '
    "-- u is the horizontal pixel coordinate (0=left edge, increasing right), "
    "v is the vertical pixel coordinate (0=top edge, increasing down), and "
    "radius_px is the circle's radius in pixels. No other text."
)


def _context_block(context: GhostMaskContext) -> str:
    w, h = context.frame_wh
    cx = w / 2.0
    side = "left" if context.bearing_deg > 0 else "right" if context.bearing_deg < 0 else "ahead"
    return (
        f"The rover's goal is currently out of view. Belief estimate: bearing "
        f"{context.bearing_deg:+.0f} degrees from straight ahead ({side}), "
        f"distance {context.distance_m:.1f}m, uncertainty {context.uncertainty:.2f} "
        f"(higher = less confident in this estimate).\n"
        f"Frame size is {w}x{h} pixels; horizontal center is u={cx:.0f}.\n"
        f"Place a circle marking where you believe the goal most likely is. The "
        f"bearing says {side} -- unless it is within a few degrees of straight ahead, "
        f"the circle must sit clearly off-center toward the {side} edge, NOT near "
        f"u={cx:.0f}. Scale how far off-center by the bearing magnitude: a bearing "
        f"near +/-90 degrees should land close to the frame's {side} edge, a small "
        f"bearing (e.g. +/-10) only slightly off center. u={cx:.0f} means 'straight "
        f"ahead', which would contradict the goal being out of view. Grow radius_px "
        f"with uncertainty: low uncertainty should stay close to "
        f"{context.min_radius_px:.0f}px, high uncertainty should approach "
        f"{context.max_radius_px:.0f}px. Keep the circle within the frame."
    )


def _worked_example(context: GhostMaskContext) -> tuple:
    """Off-center example matching the context's own bearing side -- a fixed
    dead-center example (e.g. always (320, 240) on a 640x480 frame) anchors a
    3B model's output near the middle regardless of the actual bearing, since
    few-shot examples dominate small-model output far more than the prose
    instructions do. Tying the example to the real side keeps it illustrating
    the JSON *format* without also implicitly suggesting "center"."""
    w, h = context.frame_wh
    if context.bearing_deg > 5.0:
        u = w * 0.2
    elif context.bearing_deg < -5.0:
        u = w * 0.8
    else:
        u = w * 0.5
    v = h * 0.45
    r = min(max(context.min_radius_px, 60.0), context.max_radius_px)
    return int(u), int(v), int(r)


def build_ghost_mask_prompt(context: GhostMaskContext) -> str:
    u, v, r = _worked_example(context)
    return (
        f"{_SYSTEM_FRAMING}\n\n"
        f"{_context_block(context)}\n\n"
        f"{_JSON_SCHEMA_LINE}\n"
        f"Example (format only -- your own u must reflect the actual bearing above, "
        f'not copy this number): {{"u": {u}, "v": {v}, "radius_px": {r}}}'
    )


def build_ghost_mask_reprompt(context: GhostMaskContext) -> str:
    return (
        build_ghost_mask_prompt(context)
        + "\n\nYour previous answer could not be parsed. Respond with ONLY the JSON "
        "object, no other words."
    )


if __name__ == "__main__":
    demo = GhostMaskContext(
        bearing_deg=30.0,
        distance_m=4.5,
        uncertainty=1.2,
        frame_wh=(640, 480),
        min_radius_px=3.0,
        max_radius_px=260.0,
    )
    print(build_ghost_mask_prompt(demo))
