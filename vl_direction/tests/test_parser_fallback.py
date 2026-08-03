import pytest

from vl_direction.parser import parse_direction
from vl_direction.schemas import Direction

_ALLOWED = (Direction.LEFT, Direction.RIGHT, Direction.FRONT, Direction.BACK)


@pytest.mark.parametrize(
    "raw_text,expected",
    [
        ("LEFT", Direction.LEFT),
        ("left", Direction.LEFT),
        ("LEFT.", Direction.LEFT),
        ("I think it should go LEFT.", Direction.LEFT),
        ("Right side seems safer", Direction.RIGHT),
    ],
)
def test_parses_expected_direction(raw_text, expected):
    direction, parse_ok = parse_direction(raw_text, _ALLOWED)
    assert parse_ok is True
    assert direction == expected


def test_unparseable_text_returns_parse_ok_false():
    direction, parse_ok = parse_direction("I'm not sure what to do here.", _ALLOWED)
    assert parse_ok is False
    assert direction is None


def test_ambiguous_text_with_multiple_directions_is_unparseable():
    direction, parse_ok = parse_direction("left or maybe right", _ALLOWED)
    assert parse_ok is False
    assert direction is None


def test_binary_alphabet_rejects_front_back_only_text():
    direction, parse_ok = parse_direction("FRONT", (Direction.LEFT, Direction.RIGHT))
    assert parse_ok is False
    assert direction is None
