"""
features/fixture_missing.py
===========================

FIXTURE / CAMERA ANGLE CHANGE DETECTION

Supported detection:
--------------------
1. Camera moves LEFT
2. Camera moves RIGHT
3. Camera moves UP
4. Camera moves DOWN
5. Camera ZOOM IN
6. Camera ZOOM OUT
7. Camera ROTATION / TILT
8. Camera BLUR / DEFOCUS
9. Camera BLACK / BLANK FIELD

The following should NOT trigger:
---------------------------------
- Person movement
- Product movement
- Object movement
- Normal/stable scene changes

ROI behavior:
-------------
roi_mode = "monitor"
    Compare INSIDE configured polygon(s).

roi_mode = "ignore"
    Ignore configured polygon(s) and compare outside them.

IMPORTANT:
----------
For your Gate 28, where the entire frame is one ROI, use:

    "roi_mode": "monitor"

This fixes the previous problem where a full-frame ROI was inverted
and resulted in ZERO comparison pixels.
"""

import os
import cv2
import json
import time
import glob
import logging
import threading
from datetime import datetime

import numpy as np
import requests


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# ============================================================================
# DEFAULT SETTINGS
# ============================================================================

DEFAULT_SETTINGS = {
    # ----------------------------------------------------------------------
    # Camera movement
    # ----------------------------------------------------------------------
    "pixel_comparison_threshold": 20.0,

    "camera_angle_pixel_change_percentage": 5.0,

    # Lower than old 50%.
    # Real camera movements do not necessarily change 50% of the image.
    "camera_angle_minimum_background_coverage": 15.0,

    # Regional values are mainly used for direction classification.
    "camera_angle_region_change_percentage": 5.0,

    # Do not require 3+ regions to change.
    "camera_angle_required_changed_regions": 2,

    # ----------------------------------------------------------------------
    # Blur
    # ----------------------------------------------------------------------
    "camera_blur_enabled": True,

    "camera_blur_original_min_variance": 40.0,

    "camera_blur_current_ratio_threshold": 0.50,

    # ----------------------------------------------------------------------
    # Black / blank frame
    # ----------------------------------------------------------------------
    "black_field_detection_enabled": True,

    "black_field_pixel_threshold": 15.0,

    "black_field_dark_percentage": 90.0,

    # ----------------------------------------------------------------------
    # Direction classification
    # ----------------------------------------------------------------------
    "camera_direction_bias_ratio": 1.60,

    "camera_diagonal_bias_ratio": 1.15,

    "camera_zoom_balance_ratio": 1.30,

    # ----------------------------------------------------------------------
    # Processing
    # ----------------------------------------------------------------------
    "gaussian_blur_kernel": [5, 5],

    "morphology_kernel_size": 3,

    # ----------------------------------------------------------------------
    # Daily fixture check
    # ----------------------------------------------------------------------
    "check_time": "13:00",

    # ----------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------
    "fixture_baseline_root": "./var/data/fixture",

    "fixture_snapshot_root": "./var/data/fixture_missing_snapshots",

    # ----------------------------------------------------------------------
    # ROI
    # ----------------------------------------------------------------------
    "roi_mode": "monitor",

    "roi_source_resolution": [960, 1080],

    # ----------------------------------------------------------------------
    # API
    # ----------------------------------------------------------------------
    "api_timeout": 30,

    # ----------------------------------------------------------------------
    # Working resolution
    # ----------------------------------------------------------------------
    "working_width": 960,

    "working_height": 1080,
}


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def get_setting(settings, key, default=None, cast=None):
    """
    Get a configuration value safely.
    """
    value = settings.get(key, default)

    if cast is not None:
        try:
            return cast(value)
        except Exception:
            return default

    return value


def load_json_file(path):
    """
    Load JSON configuration.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_config_file():
    """
    Find storescript configuration.
    """

    candidates = [
        os.getenv("FIXTURE_CONFIG_FILE"),
        "./config/storescript_config copy.json",
        "./config/storescript_config.json",
        "./storescript_config.json",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            logger.info("[Fixture] Using config: %s", path)
            return path

    raise FileNotFoundError(
        "Could not find storescript configuration file."
    )


def load_config():
    """
    Load application configuration.
    """

    config_path = find_config_file()
    config = load_json_file(config_path)

    return config


# ============================================================================
# SETTINGS MERGE
# ============================================================================

def get_camera_fixture_settings(camera, config):
    """
    Merge global fixture_missing settings with camera-specific settings.
    """

    global_fixture = config.get("fixture_missing", {}) or {}

    camera_fixture = camera.get("fixture_missing", {}) or {}

    settings = dict(DEFAULT_SETTINGS)

    # Global settings
    for key, value in global_fixture.items():
        if key != "regions":
            settings[key] = value

    # Camera settings
    for key, value in camera_fixture.items():
        if key != "regions":
            settings[key] = value

    return settings


# ============================================================================
# ROI FUNCTIONS
# ============================================================================

def extract_polygons_from_camera_config(camera_fixture):
    """
    Read polygons from:

        camera["fixture_missing"]["regions"]

    Supports:

        "polygon": [[x,y], [x,y], ...]

    and:

        "polygon": [
            [[x,y], [x,y], ...],
            [[x,y], [x,y], ...]
        ]
    """

    regions = camera_fixture.get("regions", []) or []

    polygons = []

    for region in regions:

        if not isinstance(region, dict):
            continue

        polygon = region.get("polygon")

        if not polygon:
            continue

        # Single polygon:
        # [[x,y], [x,y], ...]
        if (
            isinstance(polygon, list)
            and len(polygon) > 0
            and isinstance(polygon[0], (list, tuple))
            and len(polygon[0]) >= 2
            and isinstance(polygon[0][0], (int, float))
        ):
            polygons.append(polygon)
            continue

        # Multiple polygons:
        # [
        #   [[x,y], ...],
        #   [[x,y], ...]
        # ]
        if isinstance(polygon, list):

            for item in polygon:

                if (
                    isinstance(item, list)
                    and len(item) >= 3
                    and isinstance(item[0], (list, tuple))
                ):
                    polygons.append(item)

    return polygons


def scale_polygons(
    polygons,
    source_resolution,
    target_width,
    target_height
):
    """
    Scale polygon coordinates from source resolution
    to target frame resolution.
    """

    if not polygons:
        return []

    try:
        source_width = float(source_resolution[0])
        source_height = float(source_resolution[1])

        if source_width <= 0 or source_height <= 0:
            raise ValueError

    except Exception:
        logger.warning(
            "[Fixture ROI] Invalid source resolution %s. "
            "Using target frame resolution.",
            source_resolution
        )

        source_width = float(target_width)
        source_height = float(target_height)

    scale_x = target_width / source_width
    scale_y = target_height / source_height

    scaled = []

    for polygon in polygons:

        points = []

        for point in polygon:

            if len(point) < 2:
                continue

            x = int(round(float(point[0]) * scale_x))
            y = int(round(float(point[1]) * scale_y))

            x = max(0, min(target_width - 1, x))
            y = max(0, min(target_height - 1, y))

            points.append([x, y])

        if len(points) >= 3:
            scaled.append(np.array(points, dtype=np.int32))

    return scaled


def build_roi_mask(
    polygons,
    width,
    height
):
    """
    Build mask from polygons.

    White = selected ROI.
    """

    if not polygons:
        return None

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for polygon in polygons:

        if polygon is None or len(polygon) < 3:
            continue

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

    return mask


def create_comparison_mask(
    roi_mask,
    width,
    height,
    settings
):
    """
    Create the actual comparison mask.

    roi_mode = monitor:
        Compare INSIDE ROI.

    roi_mode = ignore:
        Compare OUTSIDE ROI.

    If there is no ROI:
        compare the complete frame.
    """

    if roi_mask is None:

        logger.info(
            "[Fixture ROI] No polygon configured. "
            "Using full frame."
        )

        return np.ones(
            (height, width),
            dtype=np.uint8
        ) * 255

    roi_mode = str(
        get_setting(
            settings,
            "roi_mode",
            "monitor"
        )
    ).strip().lower()

    roi_pixels = int(
        np.count_nonzero(roi_mask)
    )

    total_pixels = width * height

    logger.info(
        "[Fixture ROI] roi_mode=%s | roi_pixels=%d | "
        "total_pixels=%d | coverage=%.2f%%",
        roi_mode,
        roi_pixels,
        total_pixels,
        (roi_pixels / total_pixels) * 100.0
        if total_pixels else 0.0
    )

    if roi_mode == "ignore":

        comparison_mask = cv2.bitwise_not(
            roi_mask
        )

    else:
        # IMPORTANT:
        # Full-frame ROI must remain full-frame comparison area.
        comparison_mask = roi_mask.copy()

    comparison_pixels = int(
        np.count_nonzero(comparison_mask)
    )

    logger.info(
        "[Fixture ROI] comparison_pixels=%d | "
        "comparison_coverage=%.2f%%",
        comparison_pixels,
        (comparison_pixels / total_pixels) * 100.0
        if total_pixels else 0.0
    )

    return comparison_mask


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def resize_image(image, width, height):
    """
    Resize image to working resolution.
    """

    if image is None:
        return None

    if image.shape[1] == width and image.shape[0] == height:
        return image.copy()

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )


def to_gray(image):
    """
    Convert BGR image to grayscale.
    """

    if image is None:
        return None

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def apply_gaussian_blur(gray, settings):
    """
    Apply small Gaussian blur before comparison.
    """

    kernel = get_setting(
        settings,
        "gaussian_blur_kernel",
        [5, 5]
    )

    try:
        kx = int(kernel[0])
        ky = int(kernel[1])
    except Exception:
        kx, ky = 5, 5

    if kx % 2 == 0:
        kx += 1

    if ky % 2 == 0:
        ky += 1

    return cv2.GaussianBlur(
        gray,
        (kx, ky),
        0
    )


# ============================================================================
# PIXEL DIFFERENCE
# ============================================================================

def calculate_pixel_difference(
    original_gray,
    current_gray,
    comparison_mask,
    settings
):
    """
    Calculate changed-pixel percentage only inside comparison_mask.

    Returns:
        changed_pixels
        changed_percentage
        valid_pixels
    """

    if comparison_mask is None:
        comparison_mask = np.ones_like(
            original_gray,
            dtype=np.uint8
        ) * 255

    valid_pixels = int(
        np.count_nonzero(comparison_mask)
    )

    if valid_pixels <= 0:

        logger.error(
            "[Fixture] CAMERA COMPARISON FAILED: "
            "comparison mask contains ZERO pixels. "
            "Check ROI configuration."
        )

        return (
            np.zeros_like(original_gray, dtype=np.uint8),
            -1.0,
            0
        )

    threshold = get_setting(
        settings,
        "pixel_comparison_threshold",
        20.0,
        float
    )

    difference = cv2.absdiff(
        original_gray,
        current_gray
    )

    changed_pixels = np.where(
        difference >= threshold,
        255,
        0
    ).astype(np.uint8)

    # Keep only valid comparison area.
    changed_pixels = cv2.bitwise_and(
        changed_pixels,
        comparison_mask
    )

    kernel_size = int(
        get_setting(
            settings,
            "morphology_kernel_size",
            3,
            int
        )
    )

    kernel_size = max(1, kernel_size)

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    changed_pixels = cv2.morphologyEx(
        changed_pixels,
        cv2.MORPH_OPEN,
        kernel
    )

    changed_pixels = cv2.morphologyEx(
        changed_pixels,
        cv2.MORPH_CLOSE,
        kernel
    )

    changed_count = int(
        np.count_nonzero(changed_pixels)
    )

    changed_percentage = (
        changed_count / valid_pixels
    ) * 100.0

    return (
        changed_pixels,
        float(changed_percentage),
        valid_pixels
    )


# ============================================================================
# REGIONAL ANALYSIS
# ============================================================================

def create_direction_regions(width, height):
    """
    Create directional regions.

    These are NOT used as the primary movement detector.

    They are used to classify:
        left
        right
        up
        down
        zoom
        tilt
    """

    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    y1 = int(height * 0.20)
    y2 = int(height * 0.80)

    regions = {}

    regions["top"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["top"][0:y1, :] = 255

    regions["bottom"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["bottom"][y2:height, :] = 255

    regions["left"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["left"][:, 0:x1] = 255

    regions["right"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["right"][:, x2:width] = 255

    regions["top_left"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["top_left"][0:y1, 0:x1] = 255

    regions["top_right"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["top_right"][0:y1, x2:width] = 255

    regions["bottom_left"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["bottom_left"][y2:height, 0:x1] = 255

    regions["bottom_right"] = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    regions["bottom_right"][y2:height, x2:width] = 255

    return regions


def calculate_region_change(
    changed_pixels,
    comparison_mask,
    region_mask
):
    """
    Calculate changed percentage in one region.

    Region is intersected with comparison mask.
    """

    valid_region = cv2.bitwise_and(
        region_mask,
        comparison_mask
    )

    total = int(
        np.count_nonzero(valid_region)
    )

    if total <= 0:
        return 0.0

    changed = cv2.bitwise_and(
        changed_pixels,
        valid_region
    )

    changed_count = int(
        np.count_nonzero(changed)
    )

    return (
        changed_count / total
    ) * 100.0


def calculate_all_region_changes(
    changed_pixels,
    comparison_mask
):
    """
    Calculate directional regional percentages.
    """

    height, width = changed_pixels.shape[:2]

    regions = create_direction_regions(
        width,
        height
    )

    result = {}

    for name, mask in regions.items():

        result[name] = round(
            calculate_region_change(
                changed_pixels,
                comparison_mask,
                mask
            ),
            2
        )

    return result


# ============================================================================
# BLUR DETECTION
# ============================================================================

def calculate_sharpness(
    gray,
    comparison_mask
):
    """
    Calculate Laplacian variance.
    """

    if gray is None:
        return 0.0

    if comparison_mask is None:
        pixels_mask = np.ones_like(
            gray,
            dtype=np.uint8
        ) * 255
    else:
        pixels_mask = comparison_mask

    valid = gray[
        pixels_mask > 0
    ]

    if valid.size == 0:
        return 0.0

    laplacian = cv2.Laplacian(
        valid,
        cv2.CV_64F
    )

    return float(
        laplacian.var()
    )


def detect_blur(
    original_gray,
    current_gray,
    comparison_mask,
    settings
):
    """
    Detect current image blur.

    Uses both:
        - absolute minimum sharpness
        - current/original sharpness ratio
    """

    enabled = bool(
        get_setting(
            settings,
            "camera_blur_enabled",
            True
        )
    )

    if not enabled:
        return {
            "is_blur": False,
            "original_sharpness": 0.0,
            "current_sharpness": 0.0,
            "sharpness_ratio": 1.0
        }

    original_sharpness = calculate_sharpness(
        original_gray,
        comparison_mask
    )

    current_sharpness = calculate_sharpness(
        current_gray,
        comparison_mask
    )

    min_variance = get_setting(
        settings,
        "camera_blur_original_min_variance",
        40.0,
        float
    )

    ratio_threshold = get_setting(
        settings,
        "camera_blur_current_ratio_threshold",
        0.50,
        float
    )

    if original_sharpness <= 0:
        ratio = 0.0
    else:
        ratio = (
            current_sharpness /
            original_sharpness
        )

    absolute_blur = (
        current_sharpness < min_variance
    )

    relative_blur = (
        original_sharpness >= min_variance
        and ratio <= ratio_threshold
    )

    is_blur = (
        absolute_blur
        or relative_blur
    )

    return {
        "is_blur": bool(is_blur),
        "original_sharpness": round(
            original_sharpness,
            2
        ),
        "current_sharpness": round(
            current_sharpness,
            2
        ),
        "sharpness_ratio": round(
            ratio,
            3
        )
    }


# ============================================================================
# BLACK / BLANK FIELD DETECTION
# ============================================================================

def detect_black_field(
    gray,
    comparison_mask,
    settings
):
    """
    Detect black / blank / almost-black camera field.

    A frame is considered black when:

        mean brightness <= threshold

    AND

        percentage of dark pixels >= configured percentage.
    """

    enabled = bool(
        get_setting(
            settings,
            "black_field_detection_enabled",
            True
        )
    )

    if not enabled:
        return {
            "is_black": False,
            "mean_brightness": 0.0,
            "dark_percentage": 0.0
        }

    if comparison_mask is None:
        comparison_mask = np.ones_like(
            gray,
            dtype=np.uint8
        ) * 255

    pixels = gray[
        comparison_mask > 0
    ]

    if pixels.size == 0:

        return {
            "is_black": False,
            "mean_brightness": 0.0,
            "dark_percentage": 0.0
        }

    mean_brightness = float(
        np.mean(pixels)
    )

    dark_threshold = get_setting(
        settings,
        "black_field_pixel_threshold",
        15.0,
        float
    )

    minimum_dark_percentage = get_setting(
        settings,
        "black_field_dark_percentage",
        90.0,
        float
    )

    dark_percentage = float(
        np.mean(
            pixels <= dark_threshold
        ) * 100.0
    )

    is_black = (
        mean_brightness <= dark_threshold
        and
        dark_percentage >= minimum_dark_percentage
    )

    return {
        "is_black": bool(is_black),
        "mean_brightness": round(
            mean_brightness,
            2
        ),
        "dark_percentage": round(
            dark_percentage,
            2
        )
    }


# ============================================================================
# CAMERA MOVEMENT CLASSIFICATION
# ============================================================================

def classify_camera_movement(
    region_changes,
    changed_percentage,
    settings
):
    """
    Classify the direction/type of camera movement.

    Important:
        This function only classifies movement.

    The actual decision that movement happened is made separately
    using overall changed percentage.
    """

    top = float(
        region_changes.get("top", 0.0)
    )

    bottom = float(
        region_changes.get("bottom", 0.0)
    )

    left = float(
        region_changes.get("left", 0.0)
    )

    right = float(
        region_changes.get("right", 0.0)
    )

    top_left = float(
        region_changes.get("top_left", 0.0)
    )

    top_right = float(
        region_changes.get("top_right", 0.0)
    )

    bottom_left = float(
        region_changes.get("bottom_left", 0.0)
    )

    bottom_right = float(
        region_changes.get("bottom_right", 0.0)
    )

    horizontal_left = (
        left + top_left + bottom_left
    ) / 3.0

    horizontal_right = (
        right + top_right + bottom_right
    ) / 3.0

    vertical_top = (
        top + top_left + top_right
    ) / 3.0

    vertical_bottom = (
        bottom + bottom_left + bottom_right
    ) / 3.0

    horizontal_total = (
        horizontal_left +
        horizontal_right
    )

    vertical_total = (
        vertical_top +
        vertical_bottom
    )

    bias_ratio = get_setting(
        settings,
        "camera_direction_bias_ratio",
        1.60,
        float
    )

    diagonal_ratio = get_setting(
        settings,
        "camera_diagonal_bias_ratio",
        1.15,
        float
    )

    zoom_balance = get_setting(
        settings,
        "camera_zoom_balance_ratio",
        1.30,
        float
    )

    # ------------------------------------------------------------------
    # LEFT / RIGHT
    # ------------------------------------------------------------------

    if horizontal_right > 0 and (
        horizontal_right /
        max(horizontal_left, 0.001)
    ) >= bias_ratio:

        return "right"

    if horizontal_left > 0 and (
        horizontal_left /
        max(horizontal_right, 0.001)
    ) >= bias_ratio:

        return "left"

    # ------------------------------------------------------------------
    # UP / DOWN
    # ------------------------------------------------------------------

    if vertical_bottom > 0 and (
        vertical_bottom /
        max(vertical_top, 0.001)
    ) >= bias_ratio:

        return "down"

    if vertical_top > 0 and (
        vertical_top /
        max(vertical_bottom, 0.001)
    ) >= bias_ratio:

        return "up"

    # ------------------------------------------------------------------
    # DIAGONAL / TILT
    # ------------------------------------------------------------------

    diagonal_values = {
        "tilt_up_left": top_left,
        "tilt_up_right": top_right,
        "tilt_down_left": bottom_left,
        "tilt_down_right": bottom_right,
    }

    sorted_diagonal = sorted(
        diagonal_values.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if sorted_diagonal:

        strongest_name, strongest_value = (
            sorted_diagonal[0]
        )

        second_value = (
            sorted_diagonal[1][1]
            if len(sorted_diagonal) > 1
            else 0.0
        )

        if strongest_value > 0 and (
            strongest_value /
            max(second_value, 0.001)
        ) >= diagonal_ratio:

            return strongest_name

    # ------------------------------------------------------------------
    # ZOOM
    # ------------------------------------------------------------------

    if horizontal_total > 0 and vertical_total > 0:

        horizontal_balance = (
            max(horizontal_left, horizontal_right)
            /
            max(
                min(horizontal_left, horizontal_right),
                0.001
            )
        )

        vertical_balance = (
            max(vertical_top, vertical_bottom)
            /
            max(
                min(vertical_top, vertical_bottom),
                0.001
            )
        )

        if (
            horizontal_balance <= zoom_balance
            and
            vertical_balance <= zoom_balance
        ):
            return "zoom"

    # ------------------------------------------------------------------
    # Generic camera movement
    # ------------------------------------------------------------------

    if changed_percentage > 0:
        return "camera_change"

    return "stable"


# ============================================================================
# CAMERA MOVEMENT DECISION
# ============================================================================

def detect_camera_movement(
    changed_pixels,
    changed_percentage,
    comparison_mask,
    region_changes,
    settings
):
    """
    Decide whether the camera moved.

    Primary detector:
        overall changed percentage.

    Regional analysis:
        only helps classify movement direction.

    This avoids the old problem where a real camera movement
    was rejected because 3 outer regions did not cross threshold.
    """

    if changed_percentage < 0:
        return {
            "camera_changed": False,
            "movement_type": "configuration_error",
            "changed_percentage": changed_percentage,
            "changed_regions": 0
        }

    movement_threshold = get_setting(
        settings,
        "camera_angle_pixel_change_percentage",
        5.0,
        float
    )

    region_threshold = get_setting(
        settings,
        "camera_angle_region_change_percentage",
        5.0,
        float
    )

    minimum_coverage = get_setting(
        settings,
        "camera_angle_minimum_background_coverage",
        15.0,
        float
    )

    required_regions = get_setting(
        settings,
        "camera_angle_required_changed_regions",
        2,
        int
    )

    valid_pixels = int(
        np.count_nonzero(comparison_mask)
    )

    total_pixels = (
        comparison_mask.shape[0] *
        comparison_mask.shape[1]
    )

    coverage = (
        valid_pixels / total_pixels
    ) * 100.0 if total_pixels else 0.0

    changed_region_names = [
        name
        for name, value in region_changes.items()
        if value >= region_threshold
    ]

    changed_region_count = len(
        changed_region_names
    )

    # --------------------------------------------------------------
    # Primary movement condition
    # --------------------------------------------------------------

    percentage_condition = (
        changed_percentage >= movement_threshold
    )

    coverage_condition = (
        coverage >= minimum_coverage
    )

    # --------------------------------------------------------------
    # Important:
    #
    # Do NOT require changed_region_count as a hard condition.
    #
    # A camera may move slightly left/right/up/down and still produce
    # significant overall change without 2 or 3 border regions crossing
    # the threshold.
    # --------------------------------------------------------------

    camera_changed = (
        percentage_condition
        and
        coverage_condition
    )

    movement_type = "stable"

    if camera_changed:

        movement_type = classify_camera_movement(
            region_changes,
            changed_percentage,
            settings
        )

    logger.info(
        "[Fixture Movement] changed=%.2f%% | threshold=%.2f%% | "
        "coverage=%.2f%% | min_coverage=%.2f%% | "
        "changed_regions=%d | required_regions=%d | "
        "camera_changed=%s | type=%s",
        changed_percentage,
        movement_threshold,
        coverage,
        minimum_coverage,
        changed_region_count,
        required_regions,
        camera_changed,
        movement_type
    )

    return {
        "camera_changed": bool(camera_changed),
        "movement_type": movement_type,
        "changed_percentage": round(
            changed_percentage,
            2
        ),
        "coverage": round(
            coverage,
            2
        ),
        "changed_regions": changed_region_count,
        "required_regions": required_regions,
        "changed_region_names": changed_region_names,
        "movement_threshold": movement_threshold,
        "region_threshold": region_threshold,
        "minimum_coverage": minimum_coverage
    }


# ============================================================================
# COMPLETE CAMERA ANGLE DETECTION
# ============================================================================

def detect_camera_angle_change(
    original_image,
    current_image,
    roi_mask,
    settings
):
    """
    Main comparison function.

    Priority:

        1. Black / blank
        2. Blur
        3. Camera movement
        4. Stable
    """

    if original_image is None:
        raise ValueError(
            "Original image is None"
        )

    if current_image is None:
        raise ValueError(
            "Current image is None"
        )

    working_width = int(
        get_setting(
            settings,
            "working_width",
            960,
            int
        )
    )

    working_height = int(
        get_setting(
            settings,
            "working_height",
            1080,
            int
        )
    )

    original = resize_image(
        original_image,
        working_width,
        working_height
    )

    current = resize_image(
        current_image,
        working_width,
        working_height
    )

    # --------------------------------------------------------------
    # ROI must also be at working resolution.
    # --------------------------------------------------------------

    if roi_mask is None:

        comparison_mask = np.ones(
            (working_height, working_width),
            dtype=np.uint8
        ) * 255

    else:

        comparison_mask = resize_image(
            roi_mask,
            working_width,
            working_height
        )

        if len(comparison_mask.shape) == 3:
            comparison_mask = cv2.cvtColor(
                comparison_mask,
                cv2.COLOR_BGR2GRAY
            )

        _, comparison_mask = cv2.threshold(
            comparison_mask,
            127,
            255,
            cv2.THRESH_BINARY
        )

    original_gray = to_gray(
        original
    )

    current_gray = to_gray(
        current
    )

    original_gray = apply_gaussian_blur(
        original_gray,
        settings
    )

    current_gray = apply_gaussian_blur(
        current_gray,
        settings
    )

    # ==============================================================
    # 1. BLACK / BLANK
    # ==============================================================

    black_result = detect_black_field(
        current_gray,
        comparison_mask,
        settings
    )

    if black_result["is_black"]:

        logger.warning(
            "[Fixture] BLACK/BLANK CAMERA FIELD detected."
        )

        return {
            "status": "CHANGE",
            "camera_angle_changed": True,
            "alert": True,
            "movement_type": "black_field",

            "changed_percentage": 100.0,

            "blur": False,
            "black_field": black_result,

            "region_changes": {},

            "sharpness": {}
        }

    # ==============================================================
    # 2. BLUR
    # ==============================================================

    blur_result = detect_blur(
        original_gray,
        current_gray,
        comparison_mask,
        settings
    )

    if blur_result["is_blur"]:

        logger.warning(
            "[Fixture] BLUR / DEFOCUS detected | "
            "original=%.2f | current=%.2f | ratio=%.3f",
            blur_result["original_sharpness"],
            blur_result["current_sharpness"],
            blur_result["sharpness_ratio"]
        )

        return {
            "status": "CHANGE",
            "camera_angle_changed": True,
            "alert": True,
            "movement_type": "blur",

            "changed_percentage": 0.0,

            "blur": True,
            "black_field": black_result,

            "region_changes": {},

            "sharpness": blur_result
        }

    # ==============================================================
    # 3. PIXEL MOVEMENT
    # ==============================================================

    (
        changed_pixels,
        changed_percentage,
        valid_pixels
    ) = calculate_pixel_difference(
        original_gray,
        current_gray,
        comparison_mask,
        settings
    )

    if changed_percentage < 0:

        return {
            "status": "NO_CHANGE",
            "camera_angle_changed": False,
            "alert": False,
            "movement_type": "configuration_error",

            "changed_percentage": changed_percentage,

            "blur": False,
            "black_field": black_result,

            "region_changes": {},

            "sharpness": blur_result,

            "valid_pixels": valid_pixels
        }

    # ==============================================================
    # 4. REGION ANALYSIS
    # ==============================================================

    region_changes = calculate_all_region_changes(
        changed_pixels,
        comparison_mask
    )

    # ==============================================================
    # 5. CAMERA MOVEMENT
    # ==============================================================

    movement_result = detect_camera_movement(
        changed_pixels,
        changed_percentage,
        comparison_mask,
        region_changes,
        settings
    )

    camera_changed = movement_result[
        "camera_changed"
    ]

    movement_type = movement_result[
        "movement_type"
    ]

    if camera_changed:

        logger.warning(
            "[Fixture] CAMERA MOVEMENT detected | "
            "type=%s | changed=%.2f%%",
            movement_type,
            changed_percentage
        )

        return {
            "status": "CHANGE",
            "camera_angle_changed": True,
            "alert": True,

            "movement_type": movement_type,

            "changed_percentage": round(
                changed_percentage,
                2
            ),

            "blur": False,

            "black_field": black_result,

            "region_changes": region_changes,

            "sharpness": blur_result,

            "movement": movement_result
        }

    # ==============================================================
    # 6. STABLE
    # ==============================================================

    logger.info(
        "[Fixture] Camera stable | changed=%.2f%%",
        changed_percentage
    )

    return {
        "status": "NO_CHANGE",
        "camera_angle_changed": False,
        "alert": False,

        "movement_type": "stable",

        "changed_percentage": round(
            changed_percentage,
            2
        ),

        "blur": False,

        "black_field": black_result,

        "region_changes": region_changes,

        "sharpness": blur_result,

        "movement": movement_result
    }


# ============================================================================
# FILE HELPERS
# ============================================================================

def ensure_directory(path):
    os.makedirs(
        path,
        exist_ok=True
    )


def safe_name(value):
    """
    Make filesystem-safe name.
    """

    value = str(value)

    invalid = '<>:"/\\|?*'

    for char in invalid:
        value = value.replace(
            char,
            "_"
        )

    return value.strip() or "unknown"


def build_camera_directory(
    root,
    store_id,
    camera_id,
    camera_name
):
    """
    Build per-camera fixture directory.
    """

    return os.path.join(
        root,
        safe_name(store_id),
        f"camera_{safe_name(camera_id)}_{safe_name(camera_name)}"
    )


def save_image(path, image):
    ensure_directory(
        os.path.dirname(path)
    )

    success = cv2.imwrite(
        path,
        image
    )

    if not success:
        raise IOError(
            f"Could not save image: {path}"
        )

    return path


# ============================================================================
# BASELINE
# ============================================================================

def load_or_create_baseline(
    camera,
    frame,
    baseline_root
):
    """
    Create permanent original image once.

    Pattern:

        original_YYYYMMDD_HHMMSS.jpg
    """

    store_id = camera.get(
        "store_id",
        camera.get("store", "unknown")
    )

    camera_id = camera.get(
        "id",
        "unknown"
    )

    camera_name = camera.get(
        "name",
        f"camera_{camera_id}"
    )

    camera_dir = build_camera_directory(
        baseline_root,
        store_id,
        camera_id,
        camera_name
    )

    ensure_directory(
        camera_dir
    )

    existing = sorted(
        glob.glob(
            os.path.join(
                camera_dir,
                "original_*.jpg"
            )
        )
    )

    if existing:

        logger.info(
            "[Fixture] Existing baseline: %s",
            existing[0]
        )

        image = cv2.imread(
            existing[0]
        )

        if image is None:
            raise IOError(
                f"Could not read baseline: {existing[0]}"
            )

        return existing[0], image

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = os.path.join(
        camera_dir,
        f"original_{timestamp}.jpg"
    )

    save_image(
        path,
        frame
    )

    logger.info(
        "[Fixture] Created permanent baseline: %s",
        path
    )

    return path, frame.copy()


# ============================================================================
# CURRENT IMAGE
# ============================================================================

def load_or_create_current(
    camera,
    frame,
    baseline_root
):
    """
    Create one current image per day.

    Pattern:

        current_YYYYMMDD_HHMMSS.jpg
    """

    store_id = camera.get(
        "store_id",
        camera.get("store", "unknown")
    )

    camera_id = camera.get(
        "id",
        "unknown"
    )

    camera_name = camera.get(
        "name",
        f"camera_{camera_id}"
    )

    camera_dir = build_camera_directory(
        baseline_root,
        store_id,
        camera_id,
        camera_name
    )

    ensure_directory(
        camera_dir
    )

    today = datetime.now().strftime(
        "%Y%m%d"
    )

    pattern = os.path.join(
        camera_dir,
        f"current_{today}_*.jpg"
    )

    existing = sorted(
        glob.glob(pattern)
    )

    if existing:

        path = existing[-1]

        logger.info(
            "[Fixture] Existing current image: %s",
            path
        )

        image = cv2.imread(
            path
        )

        if image is None:
            raise IOError(
                f"Could not read current image: {path}"
            )

        return path, image

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = os.path.join(
        camera_dir,
        f"current_{timestamp}.jpg"
    )

    save_image(
        path,
        frame
    )

    logger.info(
        "[Fixture] Created current image: %s",
        path
    )

    return path, frame.copy()


# ============================================================================
# DAILY CHECK MARKER
# ============================================================================

def get_daily_marker_path(
    camera,
    baseline_root
):
    store_id = camera.get(
        "store_id",
        camera.get("store", "unknown")
    )

    camera_id = camera.get(
        "id",
        "unknown"
    )

    camera_name = camera.get(
        "name",
        f"camera_{camera_id}"
    )

    camera_dir = build_camera_directory(
        baseline_root,
        store_id,
        camera_id,
        camera_name
    )

    today = datetime.now().strftime(
        "%Y%m%d"
    )

    return os.path.join(
        camera_dir,
        f".fixture_checked_{today}"
    )


def is_daily_check_completed(
    camera,
    baseline_root
):
    return os.path.exists(
        get_daily_marker_path(
            camera,
            baseline_root
        )
    )


def mark_daily_check_completed(
    camera,
    baseline_root
):
    marker = get_daily_marker_path(
        camera,
        baseline_root
    )

    ensure_directory(
        os.path.dirname(marker)
    )

    with open(
        marker,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            datetime.now().isoformat()
        )

    logger.info(
        "[Fixture] Daily check marked complete: %s",
        marker
    )


# ============================================================================
# RTSP
# ============================================================================

def read_rtsp_frame(camera):
    """
    Read one frame from RTSP URL.
    """

    rtsp_url = (
        camera.get("rtsp_url")
        or camera.get("rtsp")
        or camera.get("url")
    )

    if not rtsp_url:
        logger.error(
            "[Fixture] No RTSP URL configured for camera %s",
            camera.get("id")
        )
        return None

    logger.info(
        "[Fixture] Opening RTSP camera=%s",
        camera.get("id")
    )

    cap = cv2.VideoCapture(
        rtsp_url
    )

    try:

        if not cap.isOpened():

            logger.error(
                "[Fixture] Could not open RTSP camera=%s",
                camera.get("id")
            )

            return None

        success, frame = cap.read()

        if not success or frame is None:

            logger.error(
                "[Fixture] Could not read frame camera=%s",
                camera.get("id")
            )

            return None

        return frame

    finally:

        cap.release()


# ============================================================================
# API
# ============================================================================

def build_api_payload(
    camera,
    original_path,
    current_path,
    result
):
    """
    Build fixture-missing API payload.

    Keeps compatibility with existing fixture event API.
    """

    store_id = camera.get(
        "store_id",
        camera.get("store", "")
    )

    category_name = camera.get(
        "category_name",
        camera.get(
            "name",
            f"camera_{camera.get('id', '')}"
        )
    )

    camera_id = camera.get(
        "id",
        ""
    )

    camera_name = camera.get(
        "name",
        f"camera_{camera_id}"
    )

    status = result.get(
        "status",
        "NO_CHANGE"
    )

    alert = (
        "yes"
        if result.get(
            "alert",
            False
        )
        else "no"
    )

    return {
        "store_id": store_id,
        "camera_no": camera_id,
        "camera_name": camera_name,
        "category_name": category_name,

        "status": status,

        "alert": alert,

        "movement_type": result.get(
            "movement_type",
            "stable"
        ),

        "camera_angle_changed": result.get(
            "camera_angle_changed",
            False
        ),

        "changed_percentage": result.get(
            "changed_percentage",
            0.0
        ),

        "original_image": original_path,

        "current_image": current_path,

        "timestamp": datetime.now().isoformat()
    }


def send_fixture_event(
    camera,
    payload,
    settings
):
    """
    Send fixture event to configured API.

    Existing environment/configuration can provide endpoint using:

        FIXTURE_MISSING_API_URL

    or:

        fixture_missing_api_url

    """

    endpoint = (
        os.getenv(
            "FIXTURE_MISSING_API_URL"
        )
        or settings.get(
            "fixture_missing_api_url"
        )
        or settings.get(
            "api_url"
        )
    )

    if not endpoint:

        logger.warning(
            "[Fixture] API endpoint not configured. "
            "Event will not be posted."
        )

        return False

    timeout = get_setting(
        settings,
        "api_timeout",
        30,
        int
    )

    try:

        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout
        )

        logger.info(
            "[Fixture] API response=%s | body=%s",
            response.status_code,
            response.text[:500]
        )

        response.raise_for_status()

        return True

    except Exception as exc:

        logger.exception(
            "[Fixture] API request failed: %s",
            exc
        )

        return False


# ============================================================================
# SNAPSHOT
# ============================================================================

def save_change_snapshot(
    camera,
    current_image,
    snapshot_root,
    movement_type
):
    """
    Save snapshot only when CHANGE is detected.
    """

    store_id = camera.get(
        "store_id",
        camera.get("store", "unknown")
    )

    camera_id = camera.get(
        "id",
        "unknown"
    )

    camera_name = camera.get(
        "name",
        f"camera_{camera_id}"
    )

    camera_dir = build_camera_directory(
        snapshot_root,
        store_id,
        camera_id,
        camera_name
    )

    ensure_directory(
        camera_dir
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"fixture_{safe_name(movement_type)}_"
        f"{timestamp}.jpg"
    )

    path = os.path.join(
        camera_dir,
        filename
    )

    save_image(
        path,
        current_image
    )

    logger.warning(
        "[Fixture] CHANGE snapshot saved: %s",
        path
    )

    return path


# ============================================================================
# CHECK TIME
# ============================================================================

def is_check_time_reached(
    settings
):
    """
    Return True when current local time is >= configured check time.
    """

    configured = str(
        get_setting(
            settings,
            "check_time",
            "13:00"
        )
    )

    try:

        hour, minute = map(
            int,
            configured.split(":")
        )

    except Exception:

        hour = 13
        minute = 0

    now = datetime.now()

    target_minutes = (
        hour * 60 +
        minute
    )

    current_minutes = (
        now.hour * 60 +
        now.minute
    )

    return current_minutes >= target_minutes


# ============================================================================
# CAMERA PROCESSING
# ============================================================================

def process_camera(
    camera,
    config
):
    """
    Process one camera.
    """

    camera_id = camera.get(
        "id",
        "unknown"
    )

    camera_name = camera.get(
        "name",
        f"camera_{camera_id}"
    )

    settings = get_camera_fixture_settings(
        camera,
        config
    )

    camera_fixture = (
        camera.get(
            "fixture_missing",
            {}
        )
        or {}
    )

    global_fixture = (
        config.get(
            "fixture_missing",
            {}
        )
        or {}
    )

    enabled_global = bool(
        global_fixture.get(
            "enabled",
            True
        )
    )

    enabled_camera = bool(
        camera_fixture.get(
            "enabled",
            enabled_global
        )
    )

    if not enabled_global or not enabled_camera:

        logger.info(
            "[Fixture] Disabled camera=%s",
            camera_id
        )

        return

    baseline_root = get_setting(
        settings,
        "fixture_baseline_root",
        "./var/data/fixture"
    )

    snapshot_root = get_setting(
        settings,
        "fixture_snapshot_root",
        "./var/data/fixture_missing_snapshots"
    )

    # --------------------------------------------------------------
    # Daily marker
    # --------------------------------------------------------------

    if is_daily_check_completed(
        camera,
        baseline_root
    ):

        logger.debug(
            "[Fixture] Daily check already completed "
            "camera=%s",
            camera_id
        )

        return

    # --------------------------------------------------------------
    # Check time
    # --------------------------------------------------------------

    if not is_check_time_reached(
        settings
    ):

        logger.debug(
            "[Fixture] Waiting for check time "
            "camera=%s",
            camera_id
        )

        return

    # --------------------------------------------------------------
    # Read camera
    # --------------------------------------------------------------

    frame = read_rtsp_frame(
        camera
    )

    if frame is None:
        return

    frame_height, frame_width = (
        frame.shape[:2]
    )

    logger.info(
        "[Fixture] Camera=%s | frame=%sx%s",
        camera_id,
        frame_width,
        frame_height
    )

    # --------------------------------------------------------------
    # Baseline
    # --------------------------------------------------------------

    original_path, original_image = (
        load_or_create_baseline(
            camera,
            frame,
            baseline_root
        )
    )

    # --------------------------------------------------------------
    # Current
    # --------------------------------------------------------------

    current_path, current_image = (
        load_or_create_current(
            camera,
            frame,
            baseline_root
        )
    )

    # --------------------------------------------------------------
    # ROI
    # --------------------------------------------------------------

    polygons = extract_polygons_from_camera_config(
        camera_fixture
    )

    roi_source_resolution = (
        settings.get(
            "roi_source_resolution",
            DEFAULT_SETTINGS[
                "roi_source_resolution"
            ]
        )
    )

    scaled_polygons = scale_polygons(
        polygons,
        roi_source_resolution,
        frame_width,
        frame_height
    )

    roi_mask = build_roi_mask(
        scaled_polygons,
        frame_width,
        frame_height
    )

    comparison_mask = create_comparison_mask(
        roi_mask,
        frame_width,
        frame_height,
        settings
    )

    comparison_pixels = int(
        np.count_nonzero(
            comparison_mask
        )
    )

    comparison_coverage = (
        comparison_pixels /
        (frame_width * frame_height)
    ) * 100.0

    logger.info(
        "[Fixture ROI] camera=%s | source_resolution=%s | "
        "polygons=%d | comparison_pixels=%d | "
        "comparison_coverage=%.2f%%",
        camera_id,
        roi_source_resolution,
        len(scaled_polygons),
        comparison_pixels,
        comparison_coverage
    )

    # --------------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------------

    result = detect_camera_angle_change(
        original_image,
        current_image,
        comparison_mask,
        settings
    )

    logger.info(
        "[Fixture RESULT] camera=%s (%s) | "
        "status=%s | alert=%s | movement=%s | "
        "changed=%.2f%%",
        camera_id,
        camera_name,
        result.get("status"),
        result.get("alert"),
        result.get("movement_type"),
        result.get("changed_percentage", 0.0)
    )

    # --------------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------------

    snapshot_path = None

    if result.get("alert"):

        snapshot_path = save_change_snapshot(
            camera,
            current_image,
            snapshot_root,
            result.get(
                "movement_type",
                "camera_change"
            )
        )

        result["snapshot_path"] = (
            snapshot_path
        )

    # --------------------------------------------------------------
    # API
    # --------------------------------------------------------------

    payload = build_api_payload(
        camera,
        original_path,
        current_path,
        result
    )

    if snapshot_path:
        payload["snapshot_path"] = (
            snapshot_path
        )

    api_success = send_fixture_event(
        camera,
        payload,
        settings
    )

    # --------------------------------------------------------------
    # Marker
    #
    # Mark only after successful processing/API.
    # --------------------------------------------------------------

    if api_success or not (
        os.getenv(
            "FIXTURE_MISSING_API_URL"
        )
        or settings.get(
            "fixture_missing_api_url"
        )
        or settings.get(
            "api_url"
        )
    ):

        mark_daily_check_completed(
            camera,
            baseline_root
        )


# ============================================================================
# CAMERA LOOP
# ============================================================================

def camera_worker(
    camera,
    config,
    interval=30
):
    """
    Background worker for one camera.
    """

    camera_id = camera.get(
        "id",
        "unknown"
    )

    logger.info(
        "[Fixture] Worker started camera=%s",
        camera_id
    )

    while True:

        try:

            process_camera(
                camera,
                config
            )

        except Exception as exc:

            logger.exception(
                "[Fixture] Camera=%s processing error: %s",
                camera_id,
                exc
            )

        time.sleep(
            interval
        )


# ============================================================================
# MAIN ENTRY
# ============================================================================

def run_fixture_missing(
    config=None
):
    """
    Start fixture missing processing for all configured cameras.
    """

    if config is None:
        config = load_config()

    cameras = config.get(
        "cameras",
        []
    )

    if not cameras:

        logger.warning(
            "[Fixture] No cameras found in configuration."
        )

        return

    global_fixture = (
        config.get(
            "fixture_missing",
            {}
        )
        or {}
    )

    interval = int(
        global_fixture.get(
            "poll_interval",
            30
        )
    )

    threads = []

    for camera in cameras:

        camera_fixture = (
            camera.get(
                "fixture_missing",
                {}
            )
            or {}
        )

        if not camera_fixture.get(
            "enabled",
            global_fixture.get(
                "enabled",
                True
            )
        ):
            continue

        thread = threading.Thread(
            target=camera_worker,
            args=(
                camera,
                config,
                interval
            ),
            daemon=True
        )

        thread.start()

        threads.append(
            thread
        )

    logger.info(
        "[Fixture] Started %d camera workers.",
        len(threads)
    )

    for thread in threads:
        thread.join()


# ============================================================================
# COMPATIBILITY ALIAS
# ============================================================================

def start_fixture_missing(
    config=None
):
    """
    Compatibility wrapper.
    """

    return run_fixture_missing(
        config
    )


# ============================================================================
# SCRIPT ENTRY
# ============================================================================

if __name__ == "__main__":

    try:

        configuration = load_config()

        run_fixture_missing(
            configuration
        )

    except KeyboardInterrupt:

        logger.info(
            "[Fixture] Stopped by user."
        )

    except Exception as exc:

        logger.exception(
            "[Fixture] Fatal error: %s",
            exc
        )
