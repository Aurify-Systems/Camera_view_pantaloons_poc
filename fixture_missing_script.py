"""
fixture_missing.py
==================

FIXTURE / CAMERA ANGLE CHANGE DETECTION
=======================================

IMPORTANT:
This version uses PIXEL COMPARISON ONLY.

The script does NOT detect:
    - fixture/object changes
    - person changes
    - product changes
    - YOLO objects
    - ORB features
    - homography

ONLY CAMERA MOVEMENT / CAMERA VIEW CHANGE CAN GENERATE:

    status = YES
    alert  = yes

Normal object/person/product changes inside the central ROI
are ignored:

    status = NO
    alert  = no


CAMERA CHANGES DETECTED USING PIXEL COMPARISON:
    - left / right movement
    - up / down movement
    - camera rotation
    - camera tilt
    - camera zoom
    - significant perspective change


IMAGE NAMING
============

Original:
    original_YYYYMMDD_HHMMSS.jpg

Current:
    current_YYYYMMDD_HHMMSS.jpg

Example:

    original_20260907_000815.jpg
    current_20260907_130522.jpg


IMPORTANT IMAGE RULE
====================

Original image:
    Created once and remains permanent.

Current image:
    Created once per day.

The API receives the exact same filenames that are saved locally.


DETECTION LOGIC
===============

The image is divided conceptually into:

    +---------------------------------------+
    | BACKGROUND / BORDER                   |
    |                                       |
    |   +-------------------------------+   |
    |   |                               |   |
    |   |       OBJECT / ROI AREA       |   |
    |   |       COMPLETELY IGNORED      |   |
    |   |                               |   |
    |   +-------------------------------+   |
    |                                       |
    | BACKGROUND / BORDER                   |
    +---------------------------------------+


Only the outside/background regions are compared.

The algorithm:

    1. Resize original/current to same size
    2. Convert to grayscale
    3. Apply Gaussian blur
    4. Calculate absolute pixel difference
    5. Ignore central ROI
    6. Ignore small pixel noise
    7. Calculate changed percentage
    8. Divide background into multiple regions
    9. Calculate change in each region
   10. Require distributed background change
   11. Generate camera alert only when background change
       is significant and distributed


This prevents:

    Person enters center
        ->
    NO alert

    Product moves in center
        ->
    NO alert

    Fixture/object changes in center
        ->
    NO alert

    Camera moves left/right
        ->
    YES alert

    Camera moves up/down
        ->
    YES alert

    Camera tilts
        ->
    YES alert

    Camera zooms
        ->
    YES alert
"""

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests


# ================================================================
# PATHS / LOGGING
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

CONFIG_FILE = BASE_DIR / "storescript_config copy.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("fixture_missing")


# ================================================================
# GLOBAL STATE
# ================================================================

CAMERA_LOCKS: Dict[Tuple[str, str], threading.Lock] = {}

CAMERA_LOCKS_GUARD = threading.Lock()


# ================================================================
# CONFIG
# ================================================================

def load_config() -> dict:

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as fh:

        return json.load(fh)


def resolve_path(
    value: str,
    default: str,
) -> Path:

    raw = value or default

    path = Path(raw)

    if not path.is_absolute():

        path = (
            BASE_DIR / path
        ).resolve()

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ================================================================
# NAME / DIRECTORY HELPERS
# ================================================================

def safe_name(value: Any) -> str:

    return re.sub(
        r"[^a-zA-Z0-9\-_]",
        "_",
        str(value or "UNKNOWN"),
    )


def timestamp_for_filename(
    now: Optional[datetime] = None,
) -> str:

    now = now or datetime.now()

    return now.strftime(
        "%Y%m%d_%H%M%S"
    )


def camera_directory(
    root: Path,
    store_id: Any,
    camera_category_name: str,
    camera_name: str,
) -> Path:

    path = (
        root
        / safe_name(store_id)
        / safe_name(camera_category_name)
        / safe_name(camera_name)
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def marker_path(
    camera_dir: Path,
    date_compact: str,
) -> Path:

    return camera_dir / (
        f".fixture_checked_{date_compact}"
    )


# ================================================================
# IMAGE PATH HELPERS
# ================================================================

def find_original_image(
    camera_dir: Path,
) -> Optional[Path]:

    """
    Find permanent original image.

    The oldest original is used.
    """

    originals = sorted(
        camera_dir.glob(
            "original_*.jpg"
        ),
        key=lambda path: path.stat().st_mtime,
    )

    if originals:
        return originals[0]

    return None


def find_current_image_for_date(
    camera_dir: Path,
    date_compact: str,
) -> Optional[Path]:

    currents = sorted(
        camera_dir.glob(
            f"current_{date_compact}_*.jpg"
        ),
        key=lambda path: path.stat().st_mtime,
    )

    if currents:
        return currents[0]

    return None


# ================================================================
# CAMERA LOCK
# ================================================================

def get_camera_lock(
    camera_no: int,
    date_compact: str,
) -> threading.Lock:

    key = (
        str(camera_no),
        date_compact,
    )

    with CAMERA_LOCKS_GUARD:

        if key not in CAMERA_LOCKS:

            CAMERA_LOCKS[key] = (
                threading.Lock()
            )

        return CAMERA_LOCKS[key]


# ================================================================
# IMAGE RESIZE
# ================================================================

def resize_same_size(
    original: np.ndarray,
    current: np.ndarray,
    settings: dict,
) -> Tuple[np.ndarray, np.ndarray]:

    if original is None:
        raise ValueError(
            "Original image is required"
        )

    if current is None:
        raise ValueError(
            "Current image is required"
        )

    max_width = int(
        settings.get(
            "pixel_comparison_max_width",
            960,
        )
    )

    original_height, original_width = (
        original.shape[:2]
    )

    current_height, current_width = (
        current.shape[:2]
    )

    # ------------------------------------------------------------
    # Resize original
    # ------------------------------------------------------------

    if (
        max_width > 0
        and original_width > max_width
    ):

        scale = (
            max_width / original_width
        )

        new_width = max_width

        new_height = int(
            original_height * scale
        )

        original = cv2.resize(
            original,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    # ------------------------------------------------------------
    # Resize current to original size
    # ------------------------------------------------------------

    target_height, target_width = (
        original.shape[:2]
    )

    if (
        current.shape[0] != target_height
        or current.shape[1] != target_width
    ):

        current = cv2.resize(
            current,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    return original, current


# ================================================================
# PREPROCESSING
# ================================================================

def prepare_pixel_image(
    image: np.ndarray,
    settings: dict,
) -> np.ndarray:

    """
    Convert image to grayscale and blur it.

    Blurring removes tiny changes caused by:
        - compression
        - camera noise
        - small lighting variation
        - tiny object movement
    """

    if image is None:
        raise ValueError(
            "Image is required"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blur_size = int(
        settings.get(
            "pixel_comparison_blur_kernel",
            7,
        )
    )

    # Kernel must be odd.
    if blur_size < 3:
        blur_size = 3

    if blur_size % 2 == 0:
        blur_size += 1

    gray = cv2.GaussianBlur(
        gray,
        (
            blur_size,
            blur_size,
        ),
        0,
    )

    return gray


# ================================================================
# BACKGROUND MASK
# ================================================================

def create_background_mask(
    image_shape: Tuple[int, int],
    settings: dict,
) -> np.ndarray:

    """
    Create mask for background areas.

    IMPORTANT:

    The central ROI is ignored.

    Only the outer/background portion of the image
    is used for camera movement detection.

    This is the main protection against object changes.
    """

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    border_percent = float(
        settings.get(
            "pixel_comparison_border_percent",
            0.20,
        )
    )

    border_percent = max(
        0.05,
        min(
            border_percent,
            0.45,
        ),
    )

    border_x = int(
        width * border_percent
    )

    border_y = int(
        height * border_percent
    )

    # ------------------------------------------------------------
    # TOP BACKGROUND
    # ------------------------------------------------------------

    mask[
        0:border_y,
        :
    ] = 255

    # ------------------------------------------------------------
    # BOTTOM BACKGROUND
    # ------------------------------------------------------------

    mask[
        height - border_y:height,
        :
    ] = 255

    # ------------------------------------------------------------
    # LEFT BACKGROUND
    # ------------------------------------------------------------

    mask[
        :,
        0:border_x
    ] = 255

    # ------------------------------------------------------------
    # RIGHT BACKGROUND
    # ------------------------------------------------------------

    mask[
        :,
        width - border_x:width
    ] = 255

    # ------------------------------------------------------------
    # Extra inner safety margin
    # ------------------------------------------------------------

    margin_percent = float(
        settings.get(
            "pixel_comparison_inner_margin_percent",
            0.02,
        )
    )

    if margin_percent > 0:

        margin_x = int(
            width * margin_percent
        )

        margin_y = int(
            height * margin_percent
        )

        if (
            margin_x * 2 < width
            and margin_y * 2 < height
        ):

            cv2.rectangle(
                mask,
                (
                    margin_x,
                    margin_y,
                ),
                (
                    width - margin_x - 1,
                    height - margin_y - 1,
                ),
                0,
                -1,
            )

    return mask


# ================================================================
# BACKGROUND REGIONS
# ================================================================

def create_background_regions(
    image_shape: Tuple[int, int],
    settings: dict,
) -> Dict[str, np.ndarray]:

    """
    Split background into multiple regions.

    Why?

    A person/object can change one small area.

    A camera movement changes many background regions.

    Regions:

        top
        bottom
        left
        right
        top_left
        top_right
        bottom_left
        bottom_right
    """

    height, width = image_shape[:2]

    border_percent = float(
        settings.get(
            "pixel_comparison_border_percent",
            0.20,
        )
    )

    border_percent = max(
        0.05,
        min(
            border_percent,
            0.45,
        ),
    )

    border_x = int(
        width * border_percent
    )

    border_y = int(
        height * border_percent
    )

    regions = {}

    # ------------------------------------------------------------
    # TOP
    # ------------------------------------------------------------

    top = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    top[
        0:border_y,
        :
    ] = 255

    regions["top"] = top

    # ------------------------------------------------------------
    # BOTTOM
    # ------------------------------------------------------------

    bottom = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    bottom[
        height - border_y:height,
        :
    ] = 255

    regions["bottom"] = bottom

    # ------------------------------------------------------------
    # LEFT
    # ------------------------------------------------------------

    left = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    left[
        :,
        0:border_x
    ] = 255

    regions["left"] = left

    # ------------------------------------------------------------
    # RIGHT
    # ------------------------------------------------------------

    right = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    right[
        :,
        width - border_x:width
    ] = 255

    regions["right"] = right

    # ------------------------------------------------------------
    # CORNER REGIONS
    # ------------------------------------------------------------

    top_left = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    top_left[
        0:border_y,
        0:border_x
    ] = 255

    regions["top_left"] = top_left

    top_right = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    top_right[
        0:border_y,
        width - border_x:width
    ] = 255

    regions["top_right"] = top_right

    bottom_left = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    bottom_left[
        height - border_y:height,
        0:border_x
    ] = 255

    regions["bottom_left"] = bottom_left

    bottom_right = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    bottom_right[
        height - border_y:height,
        width - border_x:width
    ] = 255

    regions["bottom_right"] = bottom_right

    return regions


# ================================================================
# PIXEL DIFFERENCE
# ================================================================

def calculate_pixel_difference(
    original_gray: np.ndarray,
    current_gray: np.ndarray,
    background_mask: np.ndarray,
    settings: dict,
) -> Tuple[np.ndarray, float]:

    """
    Calculate pixel difference ONLY in background.

    Central object ROI is ignored.
    """

    difference = cv2.absdiff(
        original_gray,
        current_gray,
    )

    pixel_threshold = float(
        settings.get(
            "pixel_comparison_threshold",
            25.0,
        )
    )

    changed_pixels = (
        difference >= pixel_threshold
    ).astype(
        np.uint8
    ) * 255

    # ------------------------------------------------------------
    # Only background pixels
    # ------------------------------------------------------------

    changed_pixels = cv2.bitwise_and(
        changed_pixels,
        changed_pixels,
        mask=background_mask,
    )

    # ------------------------------------------------------------
    # Remove tiny noise
    # ------------------------------------------------------------

    morphology_kernel_size = int(
        settings.get(
            "pixel_comparison_morphology_kernel",
            3,
        )
    )

    morphology_kernel_size = max(
        1,
        morphology_kernel_size,
    )

    kernel = np.ones(
        (
            morphology_kernel_size,
            morphology_kernel_size,
        ),
        dtype=np.uint8,
    )

    changed_pixels = cv2.morphologyEx(
        changed_pixels,
        cv2.MORPH_OPEN,
        kernel,
    )

    changed_pixels = cv2.morphologyEx(
        changed_pixels,
        cv2.MORPH_CLOSE,
        kernel,
    )

    background_pixel_count = int(
        np.count_nonzero(
            background_mask
        )
    )

    if background_pixel_count <= 0:

        return (
            changed_pixels,
            0.0,
        )

    changed_pixel_count = int(
        np.count_nonzero(
            changed_pixels
        )
    )

    changed_percentage = (
        changed_pixel_count
        / background_pixel_count
    ) * 100.0

    return (
        changed_pixels,
        float(changed_percentage),
    )


# ================================================================
# REGION CHANGE
# ================================================================

def calculate_region_change(
    changed_pixels: np.ndarray,
    region_mask: np.ndarray,
) -> float:

    """
    Calculate changed percentage for one background region.
    """

    region_pixels = int(
        np.count_nonzero(
            region_mask
        )
    )

    if region_pixels <= 0:
        return 0.0

    changed = cv2.bitwise_and(
        changed_pixels,
        changed_pixels,
        mask=region_mask,
    )

    changed_count = int(
        np.count_nonzero(
            changed
        )
    )

    return float(
        (
            changed_count
            / region_pixels
        ) * 100.0
    )


# ================================================================
# CAMERA ANGLE / VIEW DETECTION
# ================================================================

def detect_camera_angle_change(
    original: np.ndarray,
    current: np.ndarray,
    settings: dict,
) -> Dict[str, Any]:

    """
    Detect camera movement using PIXEL COMPARISON.

    NO ORB.
    NO HOMOGRAPHY.
    NO YOLO.
    NO OBJECT DETECTION.

    Only background pixels are compared.
    """

    if original is None:

        raise ValueError(
            "Original image is required"
        )

    if current is None:

        raise ValueError(
            "Current image is required"
        )

    # ============================================================
    # RESIZE
    # ============================================================

    original, current = resize_same_size(
        original,
        current,
        settings,
    )

    # ============================================================
    # PREPROCESS
    # ============================================================

    original_gray = prepare_pixel_image(
        original,
        settings,
    )

    current_gray = prepare_pixel_image(
        current,
        settings,
    )

    # ============================================================
    # BACKGROUND MASK
    # ============================================================

    background_mask = create_background_mask(
        original_gray.shape,
        settings,
    )

    # ============================================================
    # PIXEL DIFFERENCE
    # ============================================================

    changed_pixels, changed_percentage = (
        calculate_pixel_difference(
            original_gray,
            current_gray,
            background_mask,
            settings,
        )
    )

    # ============================================================
    # REGION ANALYSIS
    # ============================================================

    regions = create_background_regions(
        original_gray.shape,
        settings,
    )

    region_percentages = {}

    for region_name, region_mask in regions.items():

        region_percentages[region_name] = (
            calculate_region_change(
                changed_pixels,
                region_mask,
            )
        )

    # ============================================================
    # THRESHOLDS
    # ============================================================

    global_change_threshold = float(
        settings.get(
            "camera_angle_pixel_change_percentage",
            8.0,
        )
    )

    region_change_threshold = float(
        settings.get(
            "camera_angle_region_change_percentage",
            8.0,
        )
    )

    required_changed_regions = int(
        settings.get(
            "camera_angle_required_changed_regions",
            3,
        )
    )

    minimum_region_coverage = float(
        settings.get(
            "camera_angle_minimum_region_coverage_percentage",
            50.0,
        )
    )

    # ============================================================
    # CHANGED REGIONS
    # ============================================================

    changed_region_names = []

    for region_name, percentage in (
        region_percentages.items()
    ):

        if percentage >= region_change_threshold:

            changed_region_names.append(
                region_name
            )

    changed_region_count = len(
        changed_region_names
    )

    # ============================================================
    # BACKGROUND COVERAGE
    # ============================================================

    # Count pixels which are changed.
    total_background_pixels = int(
        np.count_nonzero(
            background_mask
        )
    )

    changed_background_pixels = int(
        np.count_nonzero(
            changed_pixels
        )
    )

    if total_background_pixels > 0:

        background_coverage = (
            changed_background_pixels
            / total_background_pixels
        ) * 100.0

    else:

        background_coverage = 0.0

    # ============================================================
    # DECISION
    # ============================================================

    #
    # CAMERA CHANGE:
    #
    # Global background difference must be significant
    # AND
    # enough background regions must have changed
    # AND
    # background coverage must be significant.
    #
    # This is important because:
    #
    # Object enters center:
    #
    #     background percentage -> LOW
    #     changed regions -> LOW
    #     coverage -> LOW
    #
    # Therefore:
    #
    #     NO alert
    #
    #
    # Camera moves:
    #
    #     background percentage -> HIGH
    #     changed regions -> HIGH
    #     coverage -> HIGH
    #
    # Therefore:
    #
    #     YES alert
    #

    global_change = (
        changed_percentage
        >= global_change_threshold
    )

    enough_regions = (
        changed_region_count
        >= required_changed_regions
    )

    enough_coverage = (
        background_coverage
        >= minimum_region_coverage
    )

    camera_angle_changed = (
        global_change
        and enough_regions
        and enough_coverage
    )

    # ============================================================
    # REASON
    # ============================================================

    if camera_angle_changed:

        reason = (
            "camera_background_pixel_change"
        )

    elif not global_change:

        reason = (
            "background_pixel_change_below_threshold"
        )

    elif not enough_regions:

        reason = (
            "change_not_distributed_across_background"
        )

    elif not enough_coverage:

        reason = (
            "background_coverage_below_threshold"
        )

    else:

        reason = "stable_camera"

    # ============================================================
    # STATUS
    # ============================================================

    status = (
        "CHANGE"
        if camera_angle_changed
        else "NO_CHANGE"
    )

    # ============================================================
    # LOGGING
    # ============================================================

    logger.info(
        "[Fixture Pixel Angle] "
        "global_change=%.2f%% | "
        "coverage=%.2f%% | "
        "changed_regions=%d/%d | "
        "status=%s | "
        "reason=%s",
        changed_percentage,
        background_coverage,
        changed_region_count,
        len(regions),
        status,
        reason,
    )

    logger.info(
        "[Fixture Pixel Regions] %s",
        " | ".join(
            f"{name}={value:.2f}%"
            for name, value
            in region_percentages.items()
        ),
    )

    return {
        "status": status,

        "camera_angle_changed": (
            camera_angle_changed
        ),

        "reason": reason,

        "changed_percentage": round(
            float(changed_percentage),
            2,
        ),

        "background_coverage_percentage": round(
            float(background_coverage),
            2,
        ),

        "changed_region_count": (
            changed_region_count
        ),

        "required_changed_regions": (
            required_changed_regions
        ),

        "changed_regions": (
            changed_region_names
        ),

        "region_percentages": {
            key: round(
                float(value),
                2,
            )
            for key, value
            in region_percentages.items()
        },

        "thresholds": {
            "global_change_percentage": (
                global_change_threshold
            ),
            "region_change_percentage": (
                region_change_threshold
            ),
            "required_changed_regions": (
                required_changed_regions
            ),
            "minimum_background_coverage": (
                minimum_region_coverage
            ),
        },
    }


# ================================================================
# BASELINE
# ================================================================

def load_or_create_baseline(
    camera_dir: Path,
    frame: np.ndarray,
) -> Tuple[np.ndarray, bool, Path]:

    baseline = find_original_image(
        camera_dir
    )

    # ============================================================
    # EXISTING ORIGINAL
    # ============================================================

    if baseline is not None:

        image = cv2.imread(
            str(baseline)
        )

        if image is not None:

            return (
                image,
                False,
                baseline,
            )

        logger.warning(
            "[Fixture] Existing original unreadable. "
            "Recreating: %s",
            baseline,
        )

        try:

            baseline.unlink()

        except OSError:

            logger.exception(
                "[Fixture] Could not delete unreadable "
                "original: %s",
                baseline,
            )

    # ============================================================
    # CREATE ORIGINAL
    # ============================================================

    timestamp = timestamp_for_filename()

    baseline = (
        camera_dir
        / f"original_{timestamp}.jpg"
    )

    if not cv2.imwrite(
        str(baseline),
        frame,
    ):

        raise RuntimeError(
            f"Unable to write original image: {baseline}"
        )

    logger.info(
        "[Fixture] ORIGINAL CREATED | %s",
        baseline,
    )

    return (
        frame.copy(),
        True,
        baseline,
    )


# ================================================================
# DAILY CURRENT IMAGE
# ================================================================

def load_or_create_current(
    camera_dir: Path,
    frame: np.ndarray,
    date_compact: str,
) -> Tuple[np.ndarray, Path, bool]:

    # ============================================================
    # FIND TODAY'S CURRENT
    # ============================================================

    existing_current = (
        find_current_image_for_date(
            camera_dir,
            date_compact,
        )
    )

    if existing_current is not None:

        current = cv2.imread(
            str(existing_current)
        )

        if current is not None:

            return (
                current,
                existing_current,
                False,
            )

        logger.warning(
            "[Fixture] Existing current image "
            "unreadable: %s",
            existing_current,
        )

        try:

            existing_current.unlink()

        except OSError:

            logger.exception(
                "[Fixture] Could not delete old current: %s",
                existing_current,
            )

    # ============================================================
    # REMOVE OLD CURRENT IMAGES
    # ============================================================

    for old in camera_dir.glob(
        "current_*.jpg"
    ):

        try:

            old.unlink()

        except OSError:

            logger.exception(
                "[Fixture] Could not delete old current: %s",
                old,
            )

    # ============================================================
    # CREATE TODAY'S CURRENT
    # ============================================================

    timestamp = timestamp_for_filename()

    current_path = (
        camera_dir
        / f"current_{timestamp}.jpg"
    )

    if not cv2.imwrite(
        str(current_path),
        frame,
    ):

        raise RuntimeError(
            f"Unable to write current image: "
            f"{current_path}"
        )

    logger.info(
        "[Fixture] CURRENT CREATED | %s",
        current_path,
    )

    return (
        frame.copy(),
        current_path,
        True,
    )


# ================================================================
# RTSP CAPTURE
# ================================================================

def open_rtsp(
    rtsp_url: str,
):

    cap = cv2.VideoCapture(
        rtsp_url
    )

    try:

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

    except Exception:

        pass

    if not cap.isOpened():

        cap.release()

        return None

    return cap


def read_frame(
    rtsp_url: str,
    reconnect_delay: float,
) -> Optional[np.ndarray]:

    cap = open_rtsp(
        rtsp_url
    )

    if cap is None:

        logger.error(
            "[Fixture] RTSP open failed: %s",
            rtsp_url,
        )

        time.sleep(
            reconnect_delay
        )

        return None

    try:

        ok, frame = cap.read()

        if not ok or frame is None:

            logger.warning(
                "[Fixture] RTSP frame read failed"
            )

            return None

        return frame

    finally:

        cap.release()


# ================================================================
# DAILY CHECK TIME
# ================================================================

def is_after_check_time(
    now: datetime,
    check_time: str,
) -> bool:

    try:

        hour, minute = map(
            int,
            str(check_time).split(":"),
        )

        if not (
            0 <= hour <= 23
            and 0 <= minute <= 59
        ):

            raise ValueError

    except (
        TypeError,
        ValueError,
    ):

        logger.error(
            "[Fixture] Invalid daily_check_time=%r; "
            "using 13:00",
            check_time,
        )

        hour, minute = 13, 0

    return (
        now.hour * 60
        + now.minute
        >= hour * 60
        + minute
    )


def already_checked_today(
    camera_dir: Path,
    date_compact: str,
) -> bool:

    return marker_path(
        camera_dir,
        date_compact,
    ).exists()


def mark_checked_today(
    camera_dir: Path,
    date_compact: str,
) -> None:

    marker = marker_path(
        camera_dir,
        date_compact,
    )

    marker.touch(
        exist_ok=True
    )


# ================================================================
# API
# ================================================================

def build_api_url(
    config: dict,
) -> str:

    base = str(
        config.get(
            "SERVER_BASE_URL",
            "http://127.0.0.1:8000",
        )
    ).strip().rstrip("/")

    if not base.startswith(
        (
            "http://",
            "https://",
        )
    ):

        base = (
            "http://"
            + base
        )

    endpoint = (
        config.get(
            "fixture_missing",
            {},
        )
        .get(
            "api_endpoint",
            "/storescript/api/fixture-missing-event",
        )
    )

    return (
        base
        + "/"
        + str(endpoint).lstrip("/")
    )


def send_fixture_event(
    api_url: str,
    store_id: Any,
    store_name: str,
    camera_no: int,
    camera_name: str,
    camera_category_name: str,
    status: str,
    alert: str,
    original_path: Path,
    current_path: Path,
    max_retries: int,
    retry_delay: float,
) -> bool:

    data = {
        "camera_no": str(
            camera_no
        ),

        "camera_name": str(
            camera_name
        ),

        "store_id": str(
            store_id
        ),

        "store_name": str(
            store_name or ""
        ),

        "camera_category_name": str(
            camera_category_name
        ),

        "status": str(
            status
        ).upper(),

        "alert": str(
            alert
        ).lower(),
    }

    for attempt in range(
        1,
        max_retries + 1,
    ):

        original_file = None
        current_file = None

        try:

            if not original_path.exists():

                raise FileNotFoundError(
                    f"Original image missing: "
                    f"{original_path}"
                )

            if not current_path.exists():

                raise FileNotFoundError(
                    f"Current image missing: "
                    f"{current_path}"
                )

            original_file = open(
                original_path,
                "rb",
            )

            current_file = open(
                current_path,
                "rb",
            )

            files = {

                "original_image": (
                    original_path.name,
                    original_file,
                    "image/jpeg",
                ),

                "current_image": (
                    current_path.name,
                    current_file,
                    "image/jpeg",
                ),
            }

            logger.info(
                "[Fixture] POST | "
                "attempt=%d/%d | "
                "store=%s | "
                "camera=%s | "
                "status=%s | "
                "alert=%s",
                attempt,
                max_retries,
                store_id,
                camera_no,
                status,
                alert,
            )

            logger.info(
                "[Fixture] ORIGINAL FILENAME=%s",
                original_path.name,
            )

            logger.info(
                "[Fixture] CURRENT FILENAME=%s",
                current_path.name,
            )

            response = requests.post(
                api_url,
                data=data,
                files=files,
                timeout=30,
            )

            logger.info(
                "[Fixture] API RESPONSE | "
                "HTTP=%s | body=%s",
                response.status_code,
                response.text[:1000],
            )

            if response.status_code in (
                200,
                201,
            ):

                return True

            if 400 <= response.status_code < 500:

                logger.error(
                    "[Fixture] Non-retryable API error | "
                    "HTTP=%s",
                    response.status_code,
                )

                return False

        except Exception as exc:

            logger.exception(
                "[Fixture] API request failed | "
                "attempt=%d/%d | error=%s",
                attempt,
                max_retries,
                exc,
            )

        finally:

            if original_file is not None:
                original_file.close()

            if current_file is not None:
                current_file.close()

        if attempt < max_retries:

            time.sleep(
                retry_delay
            )

    return False


# ================================================================
# CAMERA PROCESSOR
# ================================================================

def process_camera(
    store_name: str,
    store: dict,
    camera: dict,
    config: dict,
) -> None:

    camera_no = int(
        camera["id"]
    )

    camera_category_name = str(
        camera.get(
            "category_name",
            camera.get(
                "name",
                f"camera_{camera_no}",
            ),
        )
    )

    camera_name = (
        f"camera_{camera_no}"
    )

    rtsp_url = camera.get(
        "rtsp_url"
    )

    if not rtsp_url:

        logger.error(
            "[Fixture] Missing RTSP URL | "
            "store=%s | camera=%s",
            store_name,
            camera_no,
        )

        return

    # ============================================================
    # CONFIG
    # ============================================================

    global_fixture = config.get(
        "fixture_missing",
        {},
    )

    camera_fixture = camera.get(
        "fixture_missing",
        {},
    )

    if not global_fixture.get(
        "enabled",
        True,
    ):

        logger.info(
            "[Fixture] Globally disabled"
        )

        return

    if not camera_fixture.get(
        "enabled",
        True,
    ):

        logger.info(
            "[Fixture] Camera disabled | "
            "camera=%s",
            camera_no,
        )

        return

    # ============================================================
    # BASELINE DIRECTORY
    # ============================================================

    baseline_root = resolve_path(
        global_fixture.get(
            "baseline_directory",
            "./var/data/fixture",
        ),
        "./var/data/fixture",
    )

    # ============================================================
    # SNAPSHOT DIRECTORY
    # ============================================================

    snapshot_root = resolve_path(
        global_fixture.get(
            "snapshot_directory",
            "./var/data/fixture_missing_snapshots",
        ),
        "./var/data/fixture_missing_snapshots",
    )

    # ============================================================
    # CAMERA DIRECTORY
    # ============================================================

    camera_dir = camera_directory(
        baseline_root,
        store.get("store_id"),
        camera_category_name,
        camera_name,
    )

    snapshot_dir = (
        snapshot_root
        / safe_name(
            store.get("store_id")
        )
        / safe_name(
            camera_category_name
        )
        / safe_name(
            camera_name
        )
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # SETTINGS
    # ============================================================

    daily_check_time = (
        global_fixture.get(
            "daily_check_time",
            "13:00",
        )
    )

    reconnect_delay = float(
        global_fixture.get(
            "rtsp_reconnect_delay_seconds",
            5,
        )
    )

    poll_interval = float(
        global_fixture.get(
            "poll_interval_seconds",
            0.25,
        )
    )

    api_url = build_api_url(
        config
    )

    logger.info(
        "[Fixture] Camera started | "
        "store=%s | "
        "camera=%s | "
        "category=%s | "
        "RTSP=%s",
        store.get("store_id"),
        camera_no,
        camera_category_name,
        rtsp_url,
    )

    logger.info(
        "[Fixture] MODE = PIXEL CAMERA ANGLE ONLY"
    )

    logger.info(
        "[Fixture] Object/person/product changes "
        "inside central ROI are ignored"
    )

    # ============================================================
    # MAIN LOOP
    # ============================================================

    while True:

        now = datetime.now()

        date_compact = now.strftime(
            "%Y%m%d"
        )

        date_text = now.strftime(
            "%Y-%m-%d"
        )

        # ========================================================
        # DAILY GUARD
        # ========================================================

        if already_checked_today(
            camera_dir,
            date_compact,
        ):

            logger.debug(
                "[Fixture] Already completed | "
                "camera=%s | date=%s",
                camera_no,
                date_text,
            )

            time.sleep(5)

            continue

        # ========================================================
        # READ FRAME
        # ========================================================

        frame = read_frame(
            rtsp_url,
            reconnect_delay,
        )

        if frame is None:

            time.sleep(
                reconnect_delay
            )

            continue

        # ========================================================
        # ORIGINAL
        # ========================================================

        try:

            (
                original,
                original_created,
                original_path,
            ) = load_or_create_baseline(
                camera_dir,
                frame,
            )

        except Exception:

            logger.exception(
                "[Fixture] Baseline failure | "
                "camera=%s",
                camera_no,
            )

            time.sleep(
                poll_interval
            )

            continue

        # ========================================================
        # ORIGINAL CREATED
        # ========================================================

        if original_created:

            logger.info(
                "[Fixture] Baseline ready | "
                "camera=%s | "
                "original=%s | "
                "waiting for %s",
                camera_no,
                original_path.name,
                daily_check_time,
            )

            time.sleep(
                poll_interval
            )

            continue

        # ========================================================
        # WAIT FOR DAILY CHECK
        # ========================================================

        if not is_after_check_time(
            now,
            daily_check_time,
        ):

            time.sleep(
                poll_interval
            )

            continue

        # ========================================================
        # CAMERA LOCK
        # ========================================================

        lock = get_camera_lock(
            camera_no,
            date_compact,
        )

        if not lock.acquire(
            blocking=False
        ):

            time.sleep(1)

            continue

        try:

            if already_checked_today(
                camera_dir,
                date_compact,
            ):

                continue

            # ====================================================
            # CURRENT IMAGE
            # ====================================================

            (
                current,
                current_path,
                current_created,
            ) = load_or_create_current(
                camera_dir,
                frame,
                date_compact,
            )

            logger.info(
                "[Fixture] ORIGINAL IMAGE | %s",
                original_path,
            )

            logger.info(
                "[Fixture] CURRENT IMAGE | %s",
                current_path,
            )

            # ====================================================
            # CAMERA PIXEL DETECTION
            # ====================================================

            angle_result = (
                detect_camera_angle_change(
                    original,
                    current,
                    global_fixture,
                )
            )

            logger.info(
                "[Fixture PIXEL TEST] "
                "camera=%s | "
                "status=%s | "
                "angle_changed=%s | "
                "changed_percentage=%.2f%% | "
                "background_coverage=%.2f%% | "
                "changed_regions=%d | "
                "reason=%s",
                camera_no,
                angle_result.get(
                    "status"
                ),
                angle_result.get(
                    "camera_angle_changed",
                    False,
                ),
                angle_result.get(
                    "changed_percentage",
                    0.0,
                ),
                angle_result.get(
                    "background_coverage_percentage",
                    0.0,
                ),
                angle_result.get(
                    "changed_region_count",
                    0,
                ),
                angle_result.get(
                    "reason",
                    "",
                ),
            )

            # ====================================================
            # ONLY CAMERA CHANGE = YES
            # ====================================================

            if angle_result.get(
                "camera_angle_changed",
                False,
            ):

                status = "YES"

                alert = "yes"

                logger.warning(
                    "[Fixture] "
                    "CAMERA ANGLE / CAMERA VIEW CHANGE "
                    "DETECTED | "
                    "camera=%s | "
                    "changed=%.2f%% | "
                    "coverage=%.2f%% | "
                    "regions=%d | "
                    "reason=%s",
                    camera_no,
                    angle_result.get(
                        "changed_percentage",
                        0.0,
                    ),
                    angle_result.get(
                        "background_coverage_percentage",
                        0.0,
                    ),
                    angle_result.get(
                        "changed_region_count",
                        0,
                    ),
                    angle_result.get(
                        "reason",
                        "",
                    ),
                )

                # =================================================
                # SAVE CAMERA ANGLE SNAPSHOT
                # =================================================

                alert_snapshot = (
                    snapshot_dir
                    / (
                        f"camera_angle_"
                        f"{date_compact}_"
                        f"{now.strftime('%H%M%S_%f')}.jpg"
                    )
                )

                try:

                    saved = cv2.imwrite(
                        str(alert_snapshot),
                        current,
                    )

                    if saved:

                        logger.info(
                            "[Fixture] "
                            "CAMERA ANGLE SNAPSHOT SAVED | %s",
                            alert_snapshot,
                        )

                    else:

                        logger.error(
                            "[Fixture] "
                            "CAMERA ANGLE SNAPSHOT WRITE FAILED | %s",
                            alert_snapshot,
                        )

                except Exception:

                    logger.exception(
                        "[Fixture] Camera angle snapshot "
                        "save failed"
                    )

            else:

                # =================================================
                # OBJECT CHANGE / PIXEL CHANGE INSIDE ROI
                # IS IGNORED
                # =================================================

                status = "NO"

                alert = "no"

                logger.info(
                    "[Fixture] CAMERA STABLE | "
                    "camera=%s | "
                    "background change did not satisfy "
                    "camera movement conditions | "
                    "object changes ignored",
                    camera_no,
                )

            # ====================================================
            # SEND API
            # ====================================================

            success = send_fixture_event(
                api_url=api_url,

                store_id=store.get(
                    "store_id"
                ),

                store_name=store_name,

                camera_no=camera_no,

                camera_name=camera_name,

                camera_category_name=(
                    camera_category_name
                ),

                status=status,

                alert=alert,

                original_path=original_path,

                current_path=current_path,

                max_retries=int(
                    global_fixture.get(
                        "max_api_retries",
                        5,
                    )
                ),

                retry_delay=float(
                    global_fixture.get(
                        "api_retry_delay_seconds",
                        5,
                    )
                ),
            )

            # ====================================================
            # DAILY COMPLETION
            # ====================================================

            if success:

                mark_checked_today(
                    camera_dir,
                    date_compact,
                )

                logger.info(
                    "[Fixture] DAILY CHECK COMPLETED | "
                    "store=%s | "
                    "camera=%s | "
                    "date=%s | "
                    "status=%s | "
                    "alert=%s",
                    store.get("store_id"),
                    camera_no,
                    date_text,
                    status,
                    alert,
                )

            else:

                logger.error(
                    "[Fixture] DAILY CHECK NOT MARKED COMPLETE | "
                    "camera=%s | "
                    "API delivery failed",
                    camera_no,
                )

        except Exception:

            logger.exception(
                "[Fixture] Daily processing failed | "
                "camera=%s",
                camera_no,
            )

        finally:

            lock.release()

        time.sleep(
            poll_interval
        )


# ================================================================
# MAIN
# ================================================================

def main():

    config = load_config()

    threads = []

    stores = config.get(
        "stores",
        {},
    )

    if not stores:

        raise RuntimeError(
            "No stores configured"
        )

    for store_name, store in stores.items():

        for camera in store.get(
            "gates",
            [],
        ):

            if not camera.get(
                "fixture_missing",
                {},
            ).get(
                "enabled",
                True,
            ):

                continue

            thread = threading.Thread(
                target=process_camera,
                args=(
                    store_name,
                    store,
                    camera,
                    config,
                ),
                daemon=True,
                name=(
                    f"FixtureCamera-"
                    f"{camera.get('id')}"
                ),
            )

            thread.start()

            threads.append(
                thread
            )

    logger.info(
        "[Fixture] Started %d camera thread(s)",
        len(threads),
    )

    if not threads:

        raise RuntimeError(
            "No enabled fixture_missing "
            "cameras found"
        )

    try:

        while True:

            time.sleep(10)

    except KeyboardInterrupt:

        logger.info(
            "[Fixture] Stopped by user"
        )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()