"""
The single public entrypoint for vl_direction (next.md sec 3.1): query(mode,
frames, context, episode_id) -> VLDirectiveResult. Everything else in this
package is an implementation detail reached only through here -- an
orchestrator never needs to know InternVL is behind it.

frames are plain np.ndarray (uint8, HWC, RGB), not a Frame type -- see
schemas.py's module docstring for why.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, Union

import numpy as np

from vl_direction import config
from vl_direction.client import InternVLClient, get_client
from vl_direction.intervention.mode_flag import get_current_mode
from vl_direction.parser import parse_direction
from vl_direction.prompts.cbf_prompt import build_cbf_prompt, build_cbf_reprompt
from vl_direction.prompts.exploration_prompt import (
    build_exploration_prompt,
    build_exploration_reprompt,
)
from vl_direction.prompts.uncertainty_prompt import build_sweep_description_prompt
from vl_direction.schemas import (
    CBFContext,
    Direction,
    ExplorationContext,
    IdentityToken,
    UncertaintyContext,
    UncertaintyPayload,
    UncertaintyStatus,
    VLDirectiveResult,
)

Mode = Literal["cbf", "exploration", "uncertainty"]
Context = Union[CBFContext, ExplorationContext, UncertaintyContext]

_ALLOWED_DIRECTIONS = {
    "cbf": (Direction.LEFT, Direction.RIGHT),
    "exploration": (Direction.LEFT, Direction.RIGHT, Direction.FRONT, Direction.BACK),
}


def _generate_with_retry(client, frames, prompt, reprompt, allowed, max_new_tokens):
    raw = client.generate(frames, prompt, max_new_tokens)
    direction, parse_ok = parse_direction(raw, allowed)
    retries = 0
    while not parse_ok and retries < config.PARSE_RETRY_LIMIT:
        raw = client.generate(frames, reprompt, max_new_tokens)
        direction, parse_ok = parse_direction(raw, allowed)
        retries += 1
    return raw, direction, parse_ok


def _query_cbf(client, frames, context: CBFContext):
    prompt = build_cbf_prompt(context)
    reprompt = build_cbf_reprompt(context)
    raw, direction, parse_ok = _generate_with_retry(
        client,
        frames,
        prompt,
        reprompt,
        _ALLOWED_DIRECTIONS["cbf"],
        config.MAX_NEW_TOKENS["cbf"],
    )
    confidence = 1.0 if parse_ok else 0.0
    return IdentityToken.CBF, direction, confidence, raw, parse_ok, None


def _query_exploration(client, frames, context: ExplorationContext):
    frame_count = len(frames)
    prompt = build_exploration_prompt(context, frame_count)
    reprompt = build_exploration_reprompt(context, frame_count)
    raw, direction, parse_ok = _generate_with_retry(
        client,
        frames,
        prompt,
        reprompt,
        _ALLOWED_DIRECTIONS["exploration"],
        config.MAX_NEW_TOKENS["exploration"],
    )
    confidence = 1.0 if parse_ok else 0.0
    return IdentityToken.EXPLORATION, direction, confidence, raw, parse_ok, None


def _query_uncertainty(client, frames, context: UncertaintyContext):
    if context.human_heading_response is None:
        # Request phase: optionally describe the sweep for the human operator.
        if frames:
            raw = client.generate(
                frames,
                build_sweep_description_prompt(context.rover_front_reference_deg),
                config.MAX_NEW_TOKENS["uncertainty"],
            )
        else:
            raw = ""
        payload = UncertaintyPayload(
            status=UncertaintyStatus.NEEDS_HUMAN_INPUT,
            rover_front_reference_deg=context.rover_front_reference_deg,
            attempt=context.attempt,
        )
        return IdentityToken.UNCERTAINTY, None, 1.0, raw, True, payload

    # Submit phase: pure packaging, no VLM call -- the human already supplied
    # the heading, there's nothing left to ask the model.
    hr = context.human_heading_response
    max_units = (
        context.max_units if context.max_units is not None else config.DEFAULT_MAX_UNITS
    )
    heading_desc = (
        f"heading {hr.angle_deg}"
        if hr.angle_deg is not None
        else f"heading range {hr.angle_range_deg}"
    )
    raw = f"traverse {heading_desc} for up to {max_units} units, or until goal is visually confirmed"
    payload = UncertaintyPayload(
        status=UncertaintyStatus.HEADING_DIRECTIVE,
        rover_front_reference_deg=context.rover_front_reference_deg,
        heading_deg=hr.angle_deg,
        heading_range_deg=hr.angle_range_deg,
        max_units=max_units,
        attempt=context.attempt,
    )
    return IdentityToken.UNCERTAINTY, None, 1.0, raw, True, payload


_HANDLERS = {
    "cbf": _query_cbf,
    "exploration": _query_exploration,
    "uncertainty": _query_uncertainty,
}


def query(
    mode: Mode,
    frames: list,
    context: Context,
    episode_id: str,
    client: Optional[InternVLClient] = None,
) -> VLDirectiveResult:
    if mode not in _HANDLERS:
        raise ValueError(f"unknown mode {mode!r}, expected one of {tuple(_HANDLERS)}")
    context.validate()

    resolved_client = client if client is not None else get_client()
    call_id = str(uuid.uuid4())
    t0 = time.monotonic()

    identity_token, direction, confidence, raw, parse_ok, uncertainty_payload = (
        _HANDLERS[mode](resolved_client, frames, context)
    )

    result = VLDirectiveResult(
        identity_token=identity_token,
        direction=direction,
        confidence=confidence,
        raw_response=raw,
        parse_ok=parse_ok,
        latency_ms=(time.monotonic() - t0) * 1000.0,
        frame_count=len(frames),
        timestamp=datetime.now(timezone.utc).isoformat(),
        episode_id=episode_id,
        call_id=call_id,
        session_mode=get_current_mode(),
        uncertainty_payload=uncertainty_payload,
    )
    result.validate()
    return result


if __name__ == "__main__":
    from vl_direction.client import MockInternVLClient

    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    cbf_result = query(
        "cbf",
        [frame],
        CBFContext(bbox_xyxy=(10, 10, 20, 20), frame_wh=(64, 64)),
        "demo-episode",
        client=MockInternVLClient(canned_response="LEFT"),
    )
    print("cbf ->", cbf_result)

    exploration_result = query(
        "exploration",
        [frame],
        ExplorationContext(task_str="find the sample site"),
        "demo-episode",
        client=MockInternVLClient(canned_response="FRONT"),
    )
    print("exploration ->", exploration_result)

    uncertainty_result = query(
        "uncertainty",
        [frame],
        UncertaintyContext(covariance_value=2.0, threshold_used=1.0),
        "demo-episode",
        client=MockInternVLClient(canned_response="looks rocky ahead"),
    )
    print("uncertainty ->", uncertainty_result)
