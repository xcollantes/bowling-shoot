"""Tests for config module."""

from config import HSV_LOWER, HSV_UPPER, Region


def test_region_creation():
    """Test Region dataclass constructs correctly."""
    region = Region(x=10, y=20, w=100, h=200)
    assert region.x == 10
    assert region.y == 20
    assert region.w == 100
    assert region.h == 200


def test_hsv_lower_in_valid_range():
    """Test HSV lower bounds are within valid ranges."""
    h, s, v = HSV_LOWER
    assert 0 <= h <= 180
    assert 0 <= s <= 255
    assert 0 <= v <= 255


def test_hsv_upper_in_valid_range():
    """Test HSV upper bounds are within valid ranges."""
    h, s, v = HSV_UPPER
    assert 0 <= h <= 180
    assert 0 <= s <= 255
    assert 0 <= v <= 255


def test_hsv_lower_below_upper():
    """Test that lower thresholds are <= upper thresholds."""
    for lower, upper in zip(HSV_LOWER, HSV_UPPER):
        assert lower <= upper
