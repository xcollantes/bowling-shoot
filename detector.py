"""Pin detection using HSV color thresholding and contours."""

import cv2
import numpy as np

from config import Region, settings


def detect_pins(
    frame: np.ndarray,
    region: Region,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Count bowling pins in the given region of the frame.

    Uses HSV color thresholding to isolate white objects,
    applies morphological operations to reduce noise, then
    finds contours and filters by minimum area.

    Args:
        frame: BGR image from the camera.
        region: The rectangular area to analyze.

    Returns:
        Tuple of (pin_count, bounding_boxes) where
        bounding_boxes is a list of (x, y, w, h) tuples
        in full-frame coordinates.
    """
    roi = _extract_roi(frame, region)
    if roi.size == 0:
        return 0, []
    mask = _threshold_white(roi)
    boxes = _find_pin_contours(mask, region)
    return len(boxes), boxes


def _extract_roi(
    frame: np.ndarray,
    region: Region,
) -> np.ndarray:
    """Extract the region of interest from the frame."""
    return frame[
        region.y : region.y + region.h,
        region.x : region.x + region.w,
    ]


def _threshold_white(roi: np.ndarray) -> np.ndarray:
    """Apply HSV thresholding to isolate white objects.

    Converts ROI to HSV color space, applies inRange with
    configured thresholds, then applies morphological close
    (fill holes) and open (remove small noise).

    Args:
        roi: BGR image region of interest.

    Returns:
        Binary mask of detected white areas.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
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
    return mask


def _find_pin_contours(
    mask: np.ndarray,
    region: Region,
) -> list[tuple[int, int, int, int]]:
    """Find contours in the mask and return bounding boxes.

    Filters contours by MIN_PIN_AREA. Returns bounding boxes
    in full-frame coordinates (offset by region position).

    Args:
        mask: Binary mask from thresholding.
        region: The region used for coordinate offset.

    Returns:
        List of (x, y, w, h) bounding boxes in frame coords.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= settings.min_pin_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(
                (
                    x + region.x,
                    y + region.y,
                    w,
                    h,
                )
            )
    return boxes
