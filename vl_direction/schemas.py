"""
Shared enums and dataclasses for vl_direction, per next.md sec 2/3.2. Plain
@dataclass + validate() raising ValueError, matching sam_vla/core/types.py's
style -- not pydantic, despite next.md's loose "(pydantic/dataclass)"
suggestion -- so this module has zero non-stdlib dependencies.

Frame representation: next.md types query()'s input as list[Frame], but no
Frame type exists anywhere in this codebase -- every other module
(Observation.rgb in sam_vla/core/types.py, kb_teleop_env.py) just passes
raw np.ndarray (uint8, HWC, RGB) images around. This module does the same;
there is no Frame class here, callers pass list[np.ndarray] directly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from vl_direction.intervention.mode_flag import SessionMode


class Direction(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FRONT = "FRONT"
    BACK = "BACK"


class IdentityToken(str, Enum):
    CBF = "cbf"
    EXPLORATION = "exploration"
    UNCERTAINTY = "uncertainty"


class UncertaintyStatus(str, Enum):
    NEEDS_HUMAN_INPUT = "NEEDS_HUMAN_INPUT"
    HEADING_DIRECTIVE = "HEADING_DIRECTIVE"


def _validate_bbox_xyxy(bbox: tuple, frame_wh: tuple, name: str) -> None:
    x1, y1, x2, y2 = bbox
    w, h = frame_wh
    if w <= 0 or h <= 0:
        raise ValueError(f"{name}: frame_wh must be positive, got {frame_wh!r}")
    for coord_name, coord in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if coord < 0:
            raise ValueError(f"{name}: {coord_name}={coord!r} must be >= 0")
    if x1 >= x2:
        raise ValueError(f"{name}: x1 ({x1!r}) must be < x2 ({x2!r})")
    if y1 >= y2:
        raise ValueError(f"{name}: y1 ({y1!r}) must be < y2 ({y2!r})")
    if x2 > w or y2 > h:
        raise ValueError(f"{name}: bbox {bbox!r} exceeds frame_wh {frame_wh!r}")


@dataclass
class CBFContext:
    bbox_xyxy: tuple[int, int, int, int]
    frame_wh: tuple[int, int]

    def validate(self) -> None:
        _validate_bbox_xyxy(self.bbox_xyxy, self.frame_wh, "CBFContext.bbox_xyxy")


@dataclass
class ExplorationContext:
    task_str: str
    vague_hint: Optional[str] = None

    def validate(self) -> None:
        if not self.task_str.strip():
            raise ValueError("ExplorationContext.task_str must be non-empty")


@dataclass
class HeadingResponse:
    angle_deg: Optional[float] = None
    angle_range_deg: Optional[tuple[float, float]] = None

    def validate(self) -> None:
        if self.angle_deg is None and self.angle_range_deg is None:
            raise ValueError("HeadingResponse requires angle_deg or angle_range_deg")
        if self.angle_deg is not None and self.angle_range_deg is not None:
            raise ValueError("HeadingResponse: supply angle_deg OR angle_range_deg, not both")
        if self.angle_range_deg is not None and self.angle_range_deg[0] >= self.angle_range_deg[1]:
            raise ValueError(
                f"HeadingResponse.angle_range_deg must be (low, high) with low < high, "
                f"got {self.angle_range_deg!r}"
            )


@dataclass
class UncertaintyContext:
    covariance_value: float
    threshold_used: float
    rover_front_reference_deg: float = 0.0
    human_heading_response: Optional[HeadingResponse] = None
    # Retry/traversal-budget state, threaded through this per-call-pure
    # function by the caller (see uncertainty_session.py) -- additive to
    # next.md's literal sec 3.2 listing, required by sec 4's retry loop and
    # sec 9's "attempt counter increments" test requirement.
    attempt: int = 0
    max_units: Optional[float] = None

    def validate(self) -> None:
        if self.human_heading_response is not None:
            self.human_heading_response.validate()
        if self.attempt < 0:
            raise ValueError(f"UncertaintyContext.attempt must be >= 0, got {self.attempt!r}")


@dataclass
class UncertaintyPayload:
    status: UncertaintyStatus
    rover_front_reference_deg: float
    heading_deg: Optional[float] = None
    heading_range_deg: Optional[tuple[float, float]] = None
    max_units: Optional[float] = None
    attempt: int = 0


@dataclass
class VLDirectiveResult:
    identity_token: IdentityToken
    direction: Optional[Direction]
    confidence: float
    raw_response: str
    parse_ok: bool
    latency_ms: float
    frame_count: int
    timestamp: str
    episode_id: str
    call_id: str
    session_mode: SessionMode
    uncertainty_payload: Optional[UncertaintyPayload] = None

    def validate(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"VLDirectiveResult.confidence must be in [0, 1], got {self.confidence!r}")
        is_uncertainty = self.identity_token == IdentityToken.UNCERTAINTY
        if is_uncertainty:
            if self.direction is not None:
                raise ValueError("VLDirectiveResult.direction must be None for uncertainty mode")
            if self.uncertainty_payload is None:
                raise ValueError("VLDirectiveResult.uncertainty_payload must be set for uncertainty mode")
        else:
            if self.uncertainty_payload is not None:
                raise ValueError(
                    f"VLDirectiveResult.uncertainty_payload must be None for {self.identity_token!r} mode"
                )
            if self.parse_ok and self.direction is None:
                raise ValueError("VLDirectiveResult.direction must be set when parse_ok is True")
