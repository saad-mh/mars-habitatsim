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
    side = "left" if context.bearing_deg > 0 else "right" if context.bearing_deg < 0 else "ahead"
    return (
        f"The rover's goal is currently out of view. Belief estimate: bearing "
        f"{context.bearing_deg:+.0f} degrees from straight ahead ({side}), "
        f"distance {context.distance_m:.1f}m, uncertainty {context.uncertainty:.2f} "
        f"(higher = less confident in this estimate).\n"
        f"Frame size is {w}x{h} pixels.\n"
        f"Place a circle marking where you believe the goal most likely is, given "
        f"the bearing/distance and what's visible in the frame -- prefer the side "
        f"the bearing indicates. Grow radius_px with uncertainty: low uncertainty "
        f"should stay close to {context.min_radius_px:.0f}px, high uncertainty should "
        f"approach {context.max_radius_px:.0f}px. Keep the circle within the frame."
    )


def build_ghost_mask_prompt(context: GhostMaskContext) -> str:
    return (
        f"{_SYSTEM_FRAMING}\n\n"
        f"{_context_block(context)}\n\n"
        f"{_JSON_SCHEMA_LINE}\n"
        'Example: {"u": 320, "v": 240, "radius_px": 60}'
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
