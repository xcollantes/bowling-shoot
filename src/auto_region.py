"""Auto-detect pin regions from initial frame positions."""

import cv2
import numpy as np

from src.config import Region, settings


def detect_regions(
    frame: np.ndarray,
) -> tuple[Region, Region] | None:
    """Detect left and right pin regions from a video frame.

    Runs HSV thresholding on the full frame, finds white
    objects, splits them into left/right groups by the frame
    center, and builds padded bounding regions around each
    group.

    Args:
        frame: BGR image from the camera or video.

    Returns:
        Tuple of (left_region, right_region) or None if
        auto-detection fails.
    """
    boxes = _detect_all_pins(frame)
    if not boxes:
        return None

    frame_h, frame_w = frame.shape[:2]
    center_x = frame_w // 2

    left_boxes = [b for b in boxes if b[0] + b[2] // 2 < center_x]
    right_boxes = [
        b for b in boxes if b[0] + b[2] // 2 >= center_x
    ]

    if (
        len(left_boxes) < settings.min_pins_per_side
        or len(right_boxes) < settings.min_pins_per_side
    ):
        return None

    left_region = _build_padded_region(
        left_boxes, frame_w, frame_h,
    )
    right_region = _build_padded_region(
        right_boxes, frame_w, frame_h,
    )
    return left_region, right_region


def _detect_all_pins(
    frame: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Find all white pin-like objects in the full frame.

    Uses the same HSV pipeline as detector.py but operates
    on the entire frame instead of a cropped region.

    Args:
        frame: BGR image.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(settings.hsv_lower),
        np.array(settings.hsv_upper),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (settings.morph_kernel_size, settings.morph_kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= settings.min_pin_area:
            boxes.append(cv2.boundingRect(contour))
    return boxes


def _build_padded_region(
    boxes: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> Region:
    """Build a padded Region around a group of bounding boxes.

    Computes the tight bounding box enclosing all boxes, adds
    padding, and clamps to frame bounds.

    Args:
        boxes: List of (x, y, w, h) bounding boxes.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Padded Region clamped to frame dimensions.
    """
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[0] + b[2] for b in boxes)
    y_max = max(b[1] + b[3] for b in boxes)

    pad = settings.region_padding
    x1 = max(0, x_min - pad)
    y1 = max(0, y_min - pad)
    x2 = min(frame_w, x_max + pad)
    y2 = min(frame_h, y_max + pad)

    return Region(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
