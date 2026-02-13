"""Interactive region selection via mouse on live camera feed."""

import cv2
import numpy as np

from config import (
    COLOR_LEFT,
    COLOR_RIGHT,
    COLOR_TEXT,
    MIN_REGION_SIZE,
    WINDOW_NAME,
    Region,
)


def select_regions(
    cap: cv2.VideoCapture,
) -> tuple[Region, Region]:
    """
    Display live camera feed and let user draw two regions.

    The user clicks and drags to draw rectangles. First draw
    defines the left table region, second defines the right.
    Press 'r' to reset and redraw. Press 'q' to quit.

    Args:
        cap: OpenCV VideoCapture object (already opened).

    Returns:
        Tuple of (left_region, right_region).

    Raises:
        ValueError: If user quits before selecting both.
    """
    state = {
        "drawing": False,
        "start": None,
        "end": None,
        "regions": [],
    }

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(
        WINDOW_NAME, _mouse_callback, state
    )

    labels = ["LEFT", "RIGHT"]
    colors = [COLOR_LEFT, COLOR_RIGHT]

    while len(state["regions"]) < 2:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Camera feed lost.")

        idx = len(state["regions"])
        frame = _draw_selection_overlay(
            frame, state, labels, colors
        )
        _draw_prompt(frame, f"Draw {labels[idx]} region")

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise ValueError("User quit during setup.")
        if key == ord("r"):
            state["regions"].clear()
            state["drawing"] = False

    # Confirmation screen
    _show_confirmation(cap, state, labels, colors)

    return state["regions"][0], state["regions"][1]


def _mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    param: dict,
) -> None:
    """Handle mouse events for rectangle drawing."""
    if len(param["regions"]) >= 2:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        param["drawing"] = True
        param["start"] = (x, y)
        param["end"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and param["drawing"]:
        param["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        param["drawing"] = False
        param["end"] = (x, y)
        region = _build_region(param["start"], (x, y))
        if region is not None:
            param["regions"].append(region)
        param["start"] = None
        param["end"] = None


def _build_region(
    start: tuple[int, int],
    end: tuple[int, int],
) -> Region | None:
    """
    Build a Region from two corner points.

    Returns None if the region is too small.

    Args:
        start: Top-left corner point.
        end: Bottom-right corner point.

    Returns:
        Region or None if below minimum size.
    """
    x1 = min(start[0], end[0])
    y1 = min(start[1], end[1])
    x2 = max(start[0], end[0])
    y2 = max(start[1], end[1])
    w = x2 - x1
    h = y2 - y1
    if w < MIN_REGION_SIZE or h < MIN_REGION_SIZE:
        return None
    return Region(x=x1, y=y1, w=w, h=h)


def _draw_selection_overlay(
    frame: np.ndarray,
    state: dict,
    labels: list[str],
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    """
    Draw completed regions and current drag preview.

    Args:
        frame: The current video frame.
        state: Mouse callback state dict.
        labels: Region labels ("LEFT", "RIGHT").
        colors: Colors for each region.

    Returns:
        Frame with overlays drawn.
    """
    for i, region in enumerate(state["regions"]):
        cv2.rectangle(
            frame,
            (region.x, region.y),
            (region.x + region.w, region.y + region.h),
            colors[i],
            2,
        )
        cv2.putText(
            frame, labels[i],
            (region.x + 5, region.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2,
        )

    # Draw current drag rectangle
    if state["drawing"] and state["start"] and state["end"]:
        idx = len(state["regions"])
        cv2.rectangle(
            frame,
            state["start"],
            state["end"],
            colors[idx],
            1,
        )

    return frame


def _draw_prompt(frame: np.ndarray, text: str) -> None:
    """Draw setup prompt text at the bottom of the frame."""
    h = frame.shape[0]
    prompt = f"{text} (click & drag) | R=reset | Q=quit"
    cv2.putText(
        frame, prompt, (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1,
    )


def _show_confirmation(
    cap: cv2.VideoCapture,
    state: dict,
    labels: list[str],
    colors: list[tuple[int, int, int]],
) -> None:
    """
    Show both regions and wait for Enter to confirm.

    Args:
        cap: OpenCV VideoCapture.
        state: State dict with completed regions.
        labels: Region labels.
        colors: Region colors.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _draw_selection_overlay(
            frame, state, labels, colors
        )
        h = frame.shape[0]
        cv2.putText(
            frame,
            "Press ENTER to confirm | R to redraw",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1,
        )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        if key == ord("r"):
            state["regions"].clear()
            state["drawing"] = False
            # Re-enter full selection flow
            raise _RedrawRequested()


class _RedrawRequested(Exception):
    """Internal signal to restart region selection."""

    pass
