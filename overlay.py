"""Drawing overlays on the live video frame."""

import cv2
import numpy as np

from config import Region, settings
from scoreboard import Scoreboard


def draw_regions(
    frame: np.ndarray,
    left_region: Region,
    right_region: Region,
) -> np.ndarray:
    """
    Draw labeled rectangles for both table regions.

    Args:
        frame: The current video frame.
        left_region: Left side table region.
        right_region: Right side table region.

    Returns:
        Frame with region rectangles drawn.
    """
    _draw_region_rect(frame, left_region, settings.color_left, "LEFT")
    _draw_region_rect(
        frame, right_region, settings.color_right, "RIGHT"
    )
    return frame


def _draw_region_rect(
    frame: np.ndarray,
    region: Region,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a single labeled region rectangle."""
    cv2.rectangle(
        frame,
        (region.x, region.y),
        (region.x + region.w, region.y + region.h),
        color,
        2,
    )
    cv2.putText(
        frame,
        label,
        (region.x + 5, region.y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def draw_pin_markers(
    frame: np.ndarray,
    bounding_boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int],
) -> np.ndarray:
    """
    Draw rectangles around each detected pin.

    Args:
        frame: The current video frame.
        bounding_boxes: List of (x, y, w, h) tuples.
        color: BGR color for the markers.

    Returns:
        Frame with pin markers drawn.
    """
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(
            frame, (x, y), (x + w, y + h), color, 2
        )
    return frame


def draw_scoreboard(
    frame: np.ndarray,
    left_pins: int,
    right_pins: int,
    scoreboard: Scoreboard,
) -> np.ndarray:
    """
    Draw the HUD overlay with pin counts and win totals.

    Renders a semi-transparent bar at the top of the frame
    showing current pin counts and overall score.

    Args:
        frame: The current video frame.
        left_pins: Current pin count for left side.
        right_pins: Current pin count for right side.
        scoreboard: Scoreboard with win totals.

    Returns:
        Frame with scoreboard overlay drawn.
    """
    h, w = frame.shape[:2]
    bar_height = 50

    # Semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, 0), (w, bar_height), settings.color_bg, -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    left_text = f"LEFT: {left_pins} pins"
    right_text = f"RIGHT: {right_pins} pins"
    score_text = (
        f"Score: {scoreboard.left_wins}"
        f" - {scoreboard.right_wins}"
    )

    cv2.putText(
        frame, left_text, (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, settings.color_left, 2,
    )
    cv2.putText(
        frame, score_text, (w // 2 - 60, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, settings.color_text, 2,
    )
    cv2.putText(
        frame, right_text, (w - 250, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, settings.color_right, 2,
    )
    return frame


def draw_winner(
    frame: np.ndarray,
    side: str,
) -> np.ndarray:
    """
    Draw large winner announcement in the center.

    Args:
        frame: The current video frame.
        side: "left" or "right".

    Returns:
        Frame with winner text overlay.
    """
    h, w = frame.shape[:2]
    text = f"{side.upper()} WINS!"

    # Semi-transparent backdrop
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (w // 4, h // 3),
        (3 * w // 4, 2 * h // 3),
        settings.color_bg,
        -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    text_size = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4
    )[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2

    cv2.putText(
        frame, text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, settings.color_winner, 4,
    )
    return frame


def draw_instructions(
    frame: np.ndarray,
    text: str,
) -> np.ndarray:
    """
    Draw instruction text at the bottom of the frame.

    Args:
        frame: The current video frame.
        text: Instruction text to display.

    Returns:
        Frame with instruction text overlay.
    """
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, h - 40), (w, h), settings.color_bg, -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame, text, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, settings.color_text, 1,
    )
    return frame
