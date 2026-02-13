"""Tests for detector module using synthetic images."""

import cv2
import numpy as np

from config import Region, settings
from detector import detect_pins


def _make_frame(
    width: int = 640,
    height: int = 480,
    bg_color: tuple[int, int, int] = (50, 50, 50),
) -> np.ndarray:
    """Create a solid-color BGR test frame."""
    frame = np.full((height, width, 3), bg_color, np.uint8)
    return frame


def _draw_white_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw a white filled rectangle (simulating a pin)."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)


def test_detect_no_pins():
    """Test detection on empty dark frame returns zero."""
    frame = _make_frame()
    region = Region(x=0, y=0, w=640, h=480)
    count, boxes = detect_pins(frame, region)
    assert count == 0
    assert boxes == []


def test_detect_single_pin():
    """Test detection of a single white rectangle."""
    frame = _make_frame()
    # Draw a white rectangle large enough to exceed
    # settings.min_pin_area
    size = int(settings.min_pin_area**0.5) + 10
    _draw_white_rect(frame, 100, 100, size, size)
    region = Region(x=0, y=0, w=640, h=480)
    count, boxes = detect_pins(frame, region)
    assert count == 1
    assert len(boxes) == 1


def test_detect_multiple_pins():
    """Test detection of multiple separated white rects."""
    frame = _make_frame()
    size = int(settings.min_pin_area**0.5) + 10
    _draw_white_rect(frame, 50, 50, size, size)
    _draw_white_rect(frame, 200, 50, size, size)
    _draw_white_rect(frame, 350, 50, size, size)
    region = Region(x=0, y=0, w=640, h=480)
    count, boxes = detect_pins(frame, region)
    assert count == 3


def test_small_objects_filtered():
    """Test that objects below settings.min_pin_area are ignored."""
    frame = _make_frame()
    # Draw a tiny white dot (well below settings.min_pin_area)
    _draw_white_rect(frame, 100, 100, 5, 5)
    region = Region(x=0, y=0, w=640, h=480)
    count, boxes = detect_pins(frame, region)
    assert count == 0


def test_pins_outside_region_ignored():
    """Test that pins outside the region are not counted."""
    frame = _make_frame()
    size = int(settings.min_pin_area**0.5) + 10
    # Place pin at x=400, but region only covers 0-200
    _draw_white_rect(frame, 400, 100, size, size)
    region = Region(x=0, y=0, w=200, h=480)
    count, boxes = detect_pins(frame, region)
    assert count == 0


def test_pins_inside_region_counted():
    """Test that pins inside the region are counted."""
    frame = _make_frame()
    size = int(settings.min_pin_area**0.5) + 10
    _draw_white_rect(frame, 50, 50, size, size)
    # Region only covers left portion
    region = Region(x=0, y=0, w=200, h=200)
    count, boxes = detect_pins(frame, region)
    assert count == 1


def test_bounding_boxes_in_frame_coords():
    """Test that returned boxes use full-frame coordinates."""
    frame = _make_frame()
    size = int(settings.min_pin_area**0.5) + 10
    _draw_white_rect(frame, 150, 150, size, size)
    region = Region(x=100, y=100, w=200, h=200)
    count, boxes = detect_pins(frame, region)
    assert count == 1
    bx, by, bw, bh = boxes[0]
    # Box should be offset by region position
    assert bx >= region.x
    assert by >= region.y


def test_empty_region_returns_zero():
    """Test that a zero-size region returns zero pins."""
    frame = _make_frame()
    region = Region(x=0, y=0, w=0, h=0)
    count, boxes = detect_pins(frame, region)
    assert count == 0
    assert boxes == []
