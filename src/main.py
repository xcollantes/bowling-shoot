"""Bowling pin shooting competition tracker.

Uses computer vision to detect bowling pins on two table
regions and determines which side clears their pins first.
"""

import logging
import os
import sys

import cv2

from src.config import Region, settings
from src.detector import detect_pins
from src.overlay import (
    draw_instructions,
    draw_pin_markers,
    draw_regions,
    draw_scoreboard,
    draw_winner,
)
from src.region import _RedrawRequested, select_regions
from src.scoreboard import Scoreboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _frame_delay_ms(cap: cv2.VideoCapture) -> int:
    """Return the per-frame delay in ms for waitKey.

    For video files, computes the delay from the video FPS and
    ``settings.playback_speed``. For cameras, returns 1 (no
    artificial delay).

    Args:
        cap: Opened VideoCapture.

    Returns:
        Delay in milliseconds (minimum 1).
    """
    if not settings.video_path:
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    delay = int(1000 / (fps * settings.playback_speed))
    return max(delay, 1)


def open_source() -> cv2.VideoCapture:
    """Open a video source (camera or video file).

    Uses ``settings.video_path`` when set, otherwise falls
    back to the camera at ``settings.camera_index``.

    Returns:
        Opened VideoCapture object.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    if settings.video_path:
        path = settings.video_path
        if not os.path.isfile(path):
            raise RuntimeError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {path}")
        logger.info(
            "Video opened: %s (%dx%d)",
            path,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return cap

    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {settings.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
    logger.info(
        "Camera opened: %dx%d",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return cap


def calibrate_pins(
    cap: cv2.VideoCapture,
    left_region: Region,
    right_region: Region,
) -> tuple[int, int]:
    """Auto-detect initial pin count from a calibration frame.

    Shows the detected pin count and lets user confirm or
    wait for pins to be set up properly.

    Args:
        cap: Opened VideoCapture.
        left_region: Left table region.
        right_region: Right table region.

    Returns:
        Tuple of (left_pin_count, right_pin_count).
    """
    is_video = bool(settings.video_path)
    need_frame = True

    while True:
        if need_frame:
            ret, frame = cap.read()
            if not ret:
                return 0, 0
            # For video files, pause on the first frame so the
            # user can inspect it without burning through frames.
            if is_video:
                need_frame = False

        left_pins, left_boxes = detect_pins(frame, left_region)
        right_pins, right_boxes = detect_pins(frame, right_region)

        display = draw_regions(frame.copy(), left_region, right_region)
        display = draw_pin_markers(
            display, left_boxes, settings.color_left,
        )
        display = draw_pin_markers(
            display, right_boxes, settings.color_right,
        )

        info = f"Detected: LEFT={left_pins} RIGHT={right_pins}"
        cv2.putText(
            display,
            info,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        hint = "SPACE=start round | Q=quit"
        if is_video:
            hint += " | N=next frame"
        hint += " | Adjust pins until counts are correct"
        display = draw_instructions(display, hint)

        cv2.imshow(settings.window_name, display)
        key = cv2.waitKey(0 if is_video else 1) & 0xFF

        if key == ord(" "):
            logger.info(
                "Round starting: LEFT=%d RIGHT=%d pins",
                left_pins,
                right_pins,
            )
            return left_pins, right_pins
        if key == ord("q"):
            return -1, -1
        if is_video and key == ord("n"):
            need_frame = True


def run_round(
    cap: cv2.VideoCapture,
    left_region: Region,
    right_region: Region,
    scoreboard: Scoreboard,
    delay: int = 1,
) -> str | None:
    """Run a single round of pin tracking.

    Continuously captures frames, detects pins in both
    regions, draws overlays, and checks for a winner.
    A winner is declared when one side has zero pins for
    settings.zero_pin_frame_threshold consecutive frames.

    Args:
        cap: Opened VideoCapture.
        left_region: Left table region.
        right_region: Right table region.
        scoreboard: Current scoreboard for display.
        delay: Frame delay in ms for waitKey.

    Returns:
        "left" or "right" if a winner is detected,
        None if user quit.
    """
    left_zero_count = 0
    right_zero_count = 0
    paused = False
    is_video = bool(settings.video_path)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera feed lost during round.")
                return None

            left_pins, left_boxes = detect_pins(frame, left_region)
            right_pins, right_boxes = detect_pins(
                frame, right_region,
            )

            # Track consecutive zero-pin frames
            if left_pins == 0:
                left_zero_count += 1
            else:
                left_zero_count = 0

            if right_pins == 0:
                right_zero_count += 1
            else:
                right_zero_count = 0

            # Draw overlays
            frame = draw_regions(frame, left_region, right_region)
            frame = draw_pin_markers(
                frame, left_boxes, settings.color_left,
            )
            frame = draw_pin_markers(
                frame, right_boxes, settings.color_right,
            )
            frame = draw_scoreboard(
                frame, left_pins, right_pins, scoreboard,
            )

        # Build instruction text
        hint = "ROUND IN PROGRESS"
        if is_video:
            hint += " | PAUSED" if paused else ""
            hint += " | SPACE=pause/play"
        hint += " | Q=quit"
        display = draw_instructions(frame.copy(), hint)

        # Check for winner
        if left_zero_count >= settings.zero_pin_frame_threshold:
            logger.info("LEFT side wins the round!")
            display = draw_winner(display, "left")
            cv2.imshow(settings.window_name, display)
            cv2.waitKey(3000)
            return "left"

        if right_zero_count >= settings.zero_pin_frame_threshold:
            logger.info("RIGHT side wins the round!")
            display = draw_winner(display, "right")
            cv2.imshow(settings.window_name, display)
            cv2.waitKey(3000)
            return "right"

        cv2.imshow(settings.window_name, display)
        wait = 0 if paused else delay
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q"):
            return None
        if is_video and key == ord(" "):
            paused = not paused


def _select_regions_with_redraw(
    cap: cv2.VideoCapture,
) -> tuple[Region, Region]:
    """Select regions, handling redraw requests.

    Wraps select_regions to catch _RedrawRequested and
    restart the selection process.

    Args:
        cap: Opened VideoCapture.

    Returns:
        Tuple of (left_region, right_region).
    """
    while True:
        try:
            return select_regions(cap)
        except _RedrawRequested:
            continue


def main() -> None:
    """Main entry point for the bowling pin tracker.

    Flow:
        1. Open camera
        2. Select regions interactively
        3. Load scoreboard from disk
        4. Loop: calibrate -> run round -> record winner
        5. Save scoreboard on exit
    """
    cap = open_source()
    delay = _frame_delay_ms(cap)
    scoreboard = Scoreboard.load()

    try:
        left_region, right_region = _select_regions_with_redraw(cap)
    except ValueError as e:
        logger.info("Setup cancelled: %s", e)
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    logger.info(
        "Regions set. LEFT=%s RIGHT=%s",
        left_region,
        right_region,
    )

    try:
        while True:
            left_count, right_count = calibrate_pins(
                cap, left_region, right_region,
            )
            if left_count == -1:
                break

            winner = run_round(
                cap, left_region, right_region, scoreboard, delay,
            )

            if winner:
                scoreboard.record_win(winner)
                scoreboard.save()
                logger.info(
                    "Score: LEFT=%d RIGHT=%d",
                    scoreboard.left_wins,
                    scoreboard.right_wins,
                )

                # Post-round menu
                if not _post_round_menu(cap, scoreboard):
                    break
            else:
                break
    finally:
        scoreboard.save()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application closed.")


def _post_round_menu(
    cap: cv2.VideoCapture,
    scoreboard: Scoreboard,
) -> bool:
    """Show post-round options and wait for user input.

    Args:
        cap: Opened VideoCapture.
        scoreboard: Current scoreboard.

    Returns:
        True to continue, False to quit.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        frame = draw_scoreboard(frame, 0, 0, scoreboard)
        frame = draw_instructions(
            frame,
            "SPACE=next round | s=reset scores | q=quit",
        )
        cv2.imshow(settings.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key == ord("s"):
            scoreboard.reset()
            scoreboard.save()
            logger.info("Scores reset.")
        if key == ord("q"):
            return False


if __name__ == "__main__":
    main()
