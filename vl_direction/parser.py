"""
Strict output parser for direction-emitting modes (next.md sec 5.2/9): tries
an exact normalized match first, falls back to a single-word regex search,
and only ever returns parse_ok=True when exactly one allowed direction is
identifiable -- ambiguous text (multiple direction words present) is treated
the same as no match, per next.md's "don't guess" instruction.
"""

import re

from vl_direction.schemas import Direction

_DIRECTION_PATTERNS = {
    Direction.LEFT: re.compile(r"\bleft\b", re.IGNORECASE),
    Direction.RIGHT: re.compile(r"\bright\b", re.IGNORECASE),
    Direction.FRONT: re.compile(r"\b(front|forward|straight)\b", re.IGNORECASE),
    Direction.BACK: re.compile(r"\b(back|backward|reverse)\b", re.IGNORECASE),
}


def parse_direction(raw_text: str, allowed: tuple) -> tuple:
    """Returns (direction_or_None, parse_ok)."""
    cleaned = raw_text.strip().strip(".!\"'").upper()
    for d in allowed:
        if cleaned == d.value:
            return d, True

    matches = [d for d in allowed if _DIRECTION_PATTERNS[d].search(raw_text)]
    if len(matches) == 1:
        return matches[0], True
    return None, False


if __name__ == "__main__":
    allowed = (Direction.LEFT, Direction.RIGHT)
    battery = [
        "LEFT",
        "left",
        "LEFT.",
        "I think it should go LEFT.",
        "Right side seems safer",
        "I'm not sure.",
    ]
    for raw in battery:
        print(f"{raw!r} -> {parse_direction(raw, allowed)}")
