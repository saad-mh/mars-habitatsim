import numpy as np
import pytest

from vl_direction.client import MockInternVLClient
from vl_direction.directive_engine import query
from vl_direction.schemas import CBFContext, Direction, ExplorationContext, IdentityToken, UncertaintyContext

_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
_EPISODE_ID = "contract-test-episode"


def test_cbf_contract():
    result = query(
        "cbf",
        [_FRAME],
        CBFContext(bbox_xyxy=(1, 1, 2, 2), frame_wh=(4, 4)),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="RIGHT"),
    )
    assert result.identity_token == IdentityToken.CBF
    assert result.direction == Direction.RIGHT
    assert result.parse_ok is True
    assert result.uncertainty_payload is None
    assert 0.0 <= result.confidence <= 1.0
    assert result.frame_count == 1
    assert result.call_id


def test_exploration_contract():
    result = query(
        "exploration",
        [_FRAME],
        ExplorationContext(task_str="explore"),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="BACK"),
    )
    assert result.identity_token == IdentityToken.EXPLORATION
    assert result.direction == Direction.BACK
    assert result.uncertainty_payload is None


def test_uncertainty_contract_request_phase():
    result = query(
        "uncertainty",
        [_FRAME],
        UncertaintyContext(covariance_value=2.0, threshold_used=1.0),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="rocky ahead"),
    )
    assert result.identity_token == IdentityToken.UNCERTAINTY
    assert result.direction is None
    assert result.uncertainty_payload is not None


def test_unparseable_response_yields_parse_ok_false_and_null_direction():
    result = query(
        "cbf",
        [_FRAME],
        CBFContext(bbox_xyxy=(1, 1, 2, 2), frame_wh=(4, 4)),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="I really don't know."),
    )
    assert result.parse_ok is False
    assert result.direction is None


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        query(
            "bogus",
            [_FRAME],
            ExplorationContext(task_str="explore"),
            _EPISODE_ID,
            client=MockInternVLClient(),
        )
