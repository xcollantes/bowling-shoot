"""Tests for scoreboard module."""

import json
import os
import tempfile

import pytest
from src.scoreboard import Scoreboard


@pytest.fixture
def scoreboard():
    """Provide a fresh scoreboard."""
    return Scoreboard()


@pytest.fixture
def temp_scores_file():
    """Provide a temporary file path for scores."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_initial_state(scoreboard):
    """Test scoreboard starts at zero."""
    assert scoreboard.left_wins == 0
    assert scoreboard.right_wins == 0
    assert scoreboard.rounds == []


def test_record_left_win(scoreboard):
    """Test recording a left win increments count."""
    scoreboard.record_win("left")
    assert scoreboard.left_wins == 1
    assert scoreboard.right_wins == 0
    assert len(scoreboard.rounds) == 1
    assert scoreboard.rounds[0]["winner"] == "left"
    assert scoreboard.rounds[0]["round"] == 1


def test_record_right_win(scoreboard):
    """Test recording a right win increments count."""
    scoreboard.record_win("right")
    assert scoreboard.right_wins == 1
    assert scoreboard.left_wins == 0


def test_record_multiple_wins(scoreboard):
    """Test recording multiple wins across sides."""
    scoreboard.record_win("left")
    scoreboard.record_win("right")
    scoreboard.record_win("left")
    assert scoreboard.left_wins == 2
    assert scoreboard.right_wins == 1
    assert len(scoreboard.rounds) == 3
    assert scoreboard.rounds[2]["round"] == 3


def test_record_invalid_side(scoreboard):
    """Test that invalid side raises ValueError."""
    with pytest.raises(ValueError, match="Invalid side"):
        scoreboard.record_win("center")


def test_save_and_load(scoreboard, temp_scores_file):
    """Test round-trip save and load."""
    scoreboard.record_win("left")
    scoreboard.record_win("right")
    scoreboard.save(temp_scores_file)

    loaded = Scoreboard.load(temp_scores_file)
    assert loaded.left_wins == 1
    assert loaded.right_wins == 1
    assert len(loaded.rounds) == 2


def test_load_missing_file():
    """Test loading from nonexistent file returns fresh."""
    result = Scoreboard.load("/tmp/nonexistent_scores.json")
    assert result.left_wins == 0
    assert result.right_wins == 0


def test_load_corrupt_json(temp_scores_file):
    """Test loading corrupt JSON returns fresh scoreboard."""
    with open(temp_scores_file, "w") as f:
        f.write("not valid json {{{")

    result = Scoreboard.load(temp_scores_file)
    assert result.left_wins == 0
    assert result.right_wins == 0


def test_load_missing_keys(temp_scores_file):
    """Test loading JSON with missing keys returns fresh."""
    with open(temp_scores_file, "w") as f:
        json.dump({"left_wins": 5}, f)

    result = Scoreboard.load(temp_scores_file)
    assert result.left_wins == 0


def test_reset(scoreboard):
    """Test reset zeroes everything."""
    scoreboard.record_win("left")
    scoreboard.record_win("right")
    scoreboard.reset()
    assert scoreboard.left_wins == 0
    assert scoreboard.right_wins == 0
    assert scoreboard.rounds == []


def test_round_has_timestamp(scoreboard):
    """Test that recorded rounds include a timestamp."""
    scoreboard.record_win("left")
    assert "timestamp" in scoreboard.rounds[0]
    assert len(scoreboard.rounds[0]["timestamp"]) > 0
