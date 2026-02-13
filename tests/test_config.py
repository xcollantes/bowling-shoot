"""Tests for config module."""

from config import Region, settings


def test_region_creation():
    """Test Region model constructs correctly."""
    region = Region(x=10, y=20, w=100, h=200)
    assert region.x == 10
    assert region.y == 20
    assert region.w == 100
    assert region.h == 200


def test_hsv_lower_in_valid_range():
    """Test HSV lower bounds are within valid ranges."""
    h, s, v = settings.hsv_lower
    assert 0 <= h <= 180
    assert 0 <= s <= 255
    assert 0 <= v <= 255


def test_hsv_upper_in_valid_range():
    """Test HSV upper bounds are within valid ranges."""
    h, s, v = settings.hsv_upper
    assert 0 <= h <= 180
    assert 0 <= s <= 255
    assert 0 <= v <= 255


def test_hsv_lower_below_upper():
    """Test that lower thresholds are <= upper thresholds."""
    for lower, upper in zip(settings.hsv_lower, settings.hsv_upper):
        assert lower <= upper


def test_settings_defaults():
    """Test that settings have sensible defaults."""
    assert settings.camera_index == 0
    assert settings.frame_width == 1280
    assert settings.frame_height == 720
    assert settings.min_pin_area == 500
    assert settings.zero_pin_frame_threshold == 10
    assert settings.scores_file == "scores.json"
    assert settings.min_region_size == 50
