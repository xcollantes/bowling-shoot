"""Tests for auto_region module using synthetic images."""

import cv2
import numpy as np

from src.auto_region import detect_regions
from src.config import settings


def _make_frame(
    width: int = 640,
    height: int = 480,
    bg_color: tuple[int, int, int] = (50, 50, 50),
) -> np.ndarray:
    """Create a solid-color BGR test frame."""
    return np.full((height, width, 3), bg_color, np.uint8)


def _draw_white_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw a white filled rectangle (simulating a pin)."""
    cv2.rectangle(
        frame, (x, y), (x + w, y + h), (255, 255, 255), -1,
    )


def _pin_size() -> int:
    """Return a side length that exceeds min_pin_area."""
    return int(settings.min_pin_area**0.5) + 10


def test_detect_regions_no_pins():
    """Dark frame with no white objects returns None."""
    frame = _make_frame()
    assert detect_regions(frame) is None


def test_detect_regions_one_side_only():
    """Pins only on the left side returns None."""
    frame = _make_frame()
    size = _pin_size()
    # Place two pins on the left half only
    _draw_white_rect(frame, 50, 200, size, size)
    _draw_white_rect(frame, 120, 200, size, size)
    assert detect_regions(frame) is None


def test_detect_regions_both_sides():
    """Pins on both sides returns two valid Region objects."""
    frame = _make_frame()
    size = _pin_size()
    # Left side
    _draw_white_rect(frame, 50, 200, size, size)
    # Right side (past center at x=320)
    _draw_white_rect(frame, 400, 200, size, size)

    result = detect_regions(frame)
    assert result is not None
    left, right = result
    assert left.w > 0 and left.h > 0
    assert right.w > 0 and right.h > 0
    # Left region should be on the left half
    assert left.x + left.w <= right.x + right.w


def test_detect_regions_padding():
    """Regions are larger than the tight pin bounding box."""
    frame = _make_frame()
    size = _pin_size()
    pin_x, pin_y = 50, 200
    _draw_white_rect(frame, pin_x, pin_y, size, size)
    _draw_white_rect(frame, 400, 200, size, size)

    result = detect_regions(frame)
    assert result is not None
    left, _ = result
    # Region should extend beyond the pin boundaries
    assert left.x < pin_x
    assert left.y < pin_y
    assert left.x + left.w > pin_x + size
    assert left.y + left.h > pin_y + size


def test_detect_regions_clamping():
    """Regions don't exceed frame bounds when pins are at edges."""
    frame = _make_frame(width=640, height=480)
    size = _pin_size()
    # Pin at top-left corner
    _draw_white_rect(frame, 0, 0, size, size)
    # Pin at right side near edge
    _draw_white_rect(frame, 630 - size, 0, size, size)

    result = detect_regions(frame)
    assert result is not None
    left, right = result
    # Left region must not go negative
    assert left.x >= 0
    assert left.y >= 0
    # Right region must not exceed frame
    assert right.x + right.w <= 640
    assert right.y + right.h <= 480


def test_small_objects_filtered():
    """Objects below min_pin_area are ignored."""
    frame = _make_frame()
    # Tiny objects on both sides (5x5 = 25 pixels, well below
    # the default min_pin_area of 500)
    _draw_white_rect(frame, 50, 200, 5, 5)
    _draw_white_rect(frame, 400, 200, 5, 5)
    assert detect_regions(frame) is None
