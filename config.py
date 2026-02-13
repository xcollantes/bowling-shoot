"""Configuration for the bowling pin tracker using pydantic-settings."""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Region(BaseModel):
    """Rectangular region on the frame."""

    x: int
    y: int
    w: int
    h: int


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via env vars prefixed with
    ``BOWLING_``. For example, ``BOWLING_CAMERA_INDEX=1``.
    """

    model_config = SettingsConfigDict(
        env_prefix="BOWLING_",
        env_nested_delimiter="__",
    )

    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # HSV thresholds for white pin detection
    hsv_lower: tuple[int, int, int] = (0, 0, 180)
    hsv_upper: tuple[int, int, int] = (180, 40, 255)

    # Morphological kernel size for noise removal
    morph_kernel_size: int = 5

    # Minimum contour area (pixels) to count as a pin
    min_pin_area: int = 500

    # Consecutive zero-pin frames before declaring winner
    zero_pin_frame_threshold: int = 10

    # Scoreboard persistence file
    scores_file: str = "scores.json"

    # Display colors (BGR format)
    color_left: tuple[int, int, int] = (255, 0, 0)
    color_right: tuple[int, int, int] = (0, 0, 255)
    color_text: tuple[int, int, int] = (255, 255, 255)
    color_winner: tuple[int, int, int] = (0, 255, 0)
    color_bg: tuple[int, int, int] = (0, 0, 0)

    # Minimum region size (pixels) to accept during selection
    min_region_size: int = 50

    # Window name
    window_name: str = "Bowling Pin Tracker"


settings = Settings()
