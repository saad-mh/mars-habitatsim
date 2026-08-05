"""
Prompt template for GHOST_MASK mode: the model is handed the belief state
(bearing/distance to a currently-unseen goal, plus two independent
uncertainty scalars -- how sure of the direction vs. how sure of the
distance) as text, plus the current frame, and decides where on the frame a
translucent "ghost mask" ellipse should be drawn to mark the goal's believed
location -- placement is the model's spatial-reasoning call, not caller-side
trigonometry. The two uncertainty axes let a caller with a full Gaussian
belief (mean + covariance in robot frame) hand over an anisotropic spread
instead of collapsing it to one number; a caller with only a scalar
uncertainty passes it for both and gets the old circle back. Output is
strict JSON (u, v, radius_u_px, radius_v_px in pixels), not a
closed-vocabulary word, so this mode has no few-shot LEFT/RIGHT-style
exemplar and instead leans on an explicit schema + worked example in the
prompt itself.
"""

from vl_direction.schemas import GhostMaskContext

_SYSTEM_FRAMING = (
    "You are a spatial-reasoning assistant for a Mars rover. You never explain, "
    "you only answer with JSON."
)

_JSON_SCHEMA_LINE = (
    'Respond with exactly one JSON object: {"u": <int>, "v": <int>, '
    '"radius_u_px": <int>, "radius_v_px": <int>} -- u is the horizontal pixel '
    "coordinate (0=left edge, increasing right), v is the vertical pixel "
    "coordinate (0=top edge, increasing down), and radius_u_px/radius_v_px are "
    "the ellipse's horizontal/vertical radii in pixels. No other text."
)


def _context_block(context: GhostMaskContext) -> str:
    w, h = context.frame_wh
    cx = w / 2.0
    side = (
        "left"
        if context.bearing_deg > 0
        else "right" if context.bearing_deg < 0 else "ahead"
    )
    behind = abs(context.bearing_deg) > 90.0
    behind_note = (
        f" The bearing magnitude ({context.bearing_deg:+.0f} degrees) means the "
        f"goal is actually BEHIND the rover, not just off to one side -- pin the "
        f"ellipse at the extreme {side} edge of the frame, as close to the edge "
        f"as possible, rather than partway across.\n"
        if behind
        else ""
    )
    return (
        f"The rover's goal is currently out of view. Belief estimate: bearing "
        f"{context.bearing_deg:+.0f} degrees from straight ahead ({side}), "
        f"distance {context.distance_m:.1f}m.\n"
        f"Direction confidence: bearing uncertainty is "
        f"+/-{context.bearing_uncertainty_deg:.0f} degrees (small = confident in "
        f"the {side} direction, large = the goal could be anywhere across a wide "
        f"arc).\n"
        f"Distance confidence: distance uncertainty is "
        f"+/-{context.distance_uncertainty_m:.1f}m (small = confident how far "
        f"away, large = could be much nearer or farther).\n"
        f"Frame size is {w}x{h} pixels; horizontal center is u={cx:.0f}.\n"
        f"{behind_note}"
        f"Place an ellipse marking where you believe the goal most likely is. The "
        f"bearing says {side} -- unless it is within a few degrees of straight "
        f"ahead, the ellipse center must sit clearly off-center toward the "
        f"{side} edge, NOT near u={cx:.0f}. Scale how far off-center by the "
        f"bearing magnitude: a bearing near +/-90 degrees or beyond should land "
        f"at or past the frame's {side} edge, a small bearing (e.g. +/-10) only "
        f"slightly off center. u={cx:.0f} means 'straight ahead', which would "
        f"contradict the goal being out of view.\n"
        f"Grow radius_u_px (horizontal) with bearing uncertainty and "
        f"radius_v_px (vertical) with distance uncertainty, independently -- "
        f"they do not have to match. Both stay close to "
        f"{context.min_radius_px:.0f}px when confident and approach "
        f"{context.max_radius_px:.0f}px when uncertain. Keep the ellipse's "
        f"center within the frame even when it sits at the extreme edge."
    )


def _worked_example(context: GhostMaskContext) -> tuple:
    """Off-center example matching the context's own bearing side -- a fixed
    dead-center example (e.g. always (320, 240) on a 640x480 frame) anchors a
    3B model's output near the middle regardless of the actual bearing, since
    few-shot examples dominate small-model output far more than the prose
    instructions do. Tying the example to the real side keeps it illustrating
    the JSON *format* without also implicitly suggesting "center". Two
    distinct radii in the example (not equal) so the model doesn't infer the
    two fields are meant to always match."""
    w, h = context.frame_wh
    if context.bearing_deg > 5.0:
        u = w * 0.15
    elif context.bearing_deg < -5.0:
        u = w * 0.85
    else:
        u = w * 0.5
    v = h * 0.45
    ru = min(max(context.min_radius_px, 50.0), context.max_radius_px)
    rv = min(max(context.min_radius_px, 90.0), context.max_radius_px)
    return int(u), int(v), int(ru), int(rv)


def build_ghost_mask_prompt(context: GhostMaskContext) -> str:
    u, v, ru, rv = _worked_example(context)
    return (
        f"{_SYSTEM_FRAMING}\n\n"
        f"{_context_block(context)}\n\n"
        f"{_JSON_SCHEMA_LINE}\n"
        f"Example (format only -- your own numbers must reflect the actual "
        f'bearing and uncertainties above, not copy these): {{"u": {u}, "v": {v}, '
        f'"radius_u_px": {ru}, "radius_v_px": {rv}}}'
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
        bearing_uncertainty_deg=15.0,
        distance_uncertainty_m=1.2,
        frame_wh=(640, 480),
        min_radius_px=3.0,
        max_radius_px=100.0,
    )
    print(build_ghost_mask_prompt(demo))
