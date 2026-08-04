import numpy as np
import pytest

from sam_vla.core.ghost_mask import (
    draw_ghost_mask,
    project_body_point_to_pixel,
    uncertainty_to_radius_px,
)


@pytest.mark.parametrize(
    "uncertainty,min_px,max_px,scale,expected",
    [
        (0.0, 3.0, 260.0, 100.0, 3.0),  # below floor -> clamped to min_px
        (1.0, 3.0, 260.0, 100.0, 100.0),  # within range -> uncertainty * scale
        (10.0, 3.0, 260.0, 100.0, 260.0),  # above ceiling -> clamped to max_px
    ],
)
def test_uncertainty_to_radius_px_clamps(uncertainty, min_px, max_px, scale, expected):
    assert uncertainty_to_radius_px(uncertainty, min_px, max_px, scale) == pytest.approx(
        expected
    )


def test_project_straight_ahead_lands_at_frame_center():
    h, w = 480, 640
    result = project_body_point_to_pixel(forward=5.0, left=0.0, hfov_deg=90.0, h=h, w=w)
    assert result is not None
    u, v = result
    assert u == pytest.approx((w - 1) * 0.5)
    assert v == pytest.approx((h - 1) * 0.5)


def test_project_behind_camera_returns_none():
    result = project_body_point_to_pixel(forward=-1.0, left=0.0, hfov_deg=90.0, h=480, w=640)
    assert result is None


def test_project_left_of_body_lands_left_of_center():
    h, w = 480, 640
    u, _ = project_body_point_to_pixel(forward=5.0, left=2.0, hfov_deg=90.0, h=h, w=w)
    assert u < (w - 1) * 0.5


def test_project_right_of_body_lands_right_of_center():
    h, w = 480, 640
    u, _ = project_body_point_to_pixel(forward=5.0, left=-2.0, hfov_deg=90.0, h=h, w=w)
    assert u > (w - 1) * 0.5


def test_project_far_off_axis_falls_outside_frame_returns_none():
    result = project_body_point_to_pixel(forward=1.0, left=50.0, hfov_deg=90.0, h=480, w=640)
    assert result is None


def test_draw_ghost_mask_is_non_destructive_copy():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    original = rgb.copy()
    annotated = draw_ghost_mask(rgb, u=10, v=10, radius_px=3)
    assert np.array_equal(rgb, original)  # input untouched
    assert not np.array_equal(annotated, rgb)  # something was actually drawn


def test_draw_ghost_mask_blends_green_at_center_and_leaves_far_pixels_alone():
    rgb = np.full((40, 40, 3), 100, dtype=np.uint8)
    annotated = draw_ghost_mask(rgb, u=20, v=20, radius_px=5, alpha=0.5)

    center = annotated[20, 20].astype(np.float32)
    expected_center = 100 * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    assert center == pytest.approx(expected_center, abs=1.0)

    corner = annotated[0, 0]
    assert tuple(corner) == (100, 100, 100)


def test_draw_ghost_mask_translucent_not_opaque():
    rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
    annotated = draw_ghost_mask(rgb, u=10, v=10, radius_px=5, alpha=0.45)
    center = annotated[10, 10]
    assert center[1] > 100  # green channel pulled up toward the overlay color
    assert center[0] < 100  # but not fully replaced -- background still shows through
