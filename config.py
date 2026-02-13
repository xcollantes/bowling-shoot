"""Configuration constants for the bowling pin tracker."""

from dataclasses import dataclass


# Camera
DEFAULT_CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# HSV thresholds for white pin detection
# White in HSV: low saturation, high value
HSV_LOWER: tuple[int, int, int] = (0, 0, 180)
HSV_UPPER: tuple[int, int, int] = (180, 40, 255)

# Morphological kernel size for noise removal
MORPH_KERNEL_SIZE: int = 5

# Minimum contour area (pixels) to count as a pin
MIN_PIN_AREA: int = 500

# Consecutive zero-pin frames required before declaring winner
ZERO_PIN_FRAME_THRESHOLD: int = 10

# Scoreboard persistence file
SCORES_FILE: str = "scores.json"

# Display colors (BGR format)
COLOR_LEFT: tuple[int, int, int] = (255, 0, 0)
COLOR_RIGHT: tuple[int, int, int] = (0, 0, 255)
COLOR_TEXT: tuple[int, int, int] = (255, 255, 255)
COLOR_WINNER: tuple[int, int, int] = (0, 255, 0)
COLOR_BG: tuple[int, int, int] = (0, 0, 0)

# Minimum region size (pixels) to accept during selection
MIN_REGION_SIZE: int = 50

# Window name
WINDOW_NAME: str = "Bowling Pin Tracker"


@dataclass
class Region:
    """Rectangular region on the frame."""

    x: int
    y: int
    w: int
    h: int
