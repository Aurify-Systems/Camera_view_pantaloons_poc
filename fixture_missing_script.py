"""
fixture_missing.py
==================

FIXTURE / CAMERA ANGLE CHANGE DETECTION  (v2 - polygon ROI aware)
==================================================================

WHAT CHANGED FROM v1
---------------------
1. The "object area to ignore" is no longer a fixed border percentage.
   It is now built from the polygon(s) you already have in your
   config, under each camera's:

        camera["fixture_missing"]["regions"][i]["polygon"]

   Every polygon listed for a camera is unioned together into one
   ROI mask. Everything OUTSIDE that ROI is "background" and is the
   only area used for camera-angle detection. Everything INSIDE the
   ROI (your shelf/fixture area) is completely ignored, exactly like
   before - person/product/fixture changes inside it never trigger
   an alert.

2. The polygons are scaled from `fixture_missing.roi_source_resolution`
   (the resolution they were drawn against) to whatever resolution the
   RTSP stream actually delivers, so you do not have to redraw them if
   the two differ.

3. Thresholds are now read using the key names that are actually in
   your JSON (`pixel_threshold`, `change_percentage_threshold`,
   `blur_kernel_size`, ...), with sensible fallbacks/aliases and
   defaults for the extra knobs this version adds (see
   `CONFIG KEYS` section below).

4. The result now includes a `movement_type` field:
   one of "left", "right", "up", "down", "tilt", "zoom", "blur",
   "camera_change" (generic/mixed) or "none". `status`/`alert`
   still collapse this to the same YES/NO you had before -
   nothing downstream needs to change unless you want the extra
   detail.

EVERYTHING ELSE (RTSP capture loop, daily-check-once-per-day logic,
original/current image lifecycle, API posting/retry logic) is
UNCHANGED from your v1 script.


CONFIG KEYS
===========

Read from `config["fixture_missing"]` (global) and merged with the
matching per-camera `camera["fixture_missing"]` block. Your current
JSON already has most of these - the ones marked (NEW, optional) do
not exist in your file yet; if you don't add them, the defaults shown
are used.

    pixel_threshold                          (you have: 40)
        Per-pixel grayscale difference (0-255) above which a pixel
        counts as "changed".

    change_percentage_threshold              (you have: 5.0)
        % of background pixels that must have changed for a camera
        change to even be considered (the "global" check).

    blur_kernel_size                         (you have: [5, 5])
        Gaussian blur kernel applied before differencing, to remove
        compression/sensor noise. Only the first number is used and
        it is forced odd.

    roi_source_resolution                    (you have: [960, 1080])
        The [width, height] your polygons were drawn against. Points
        are rescaled to the live frame size using this as reference.

    daily_check_time, baseline_directory, snapshot_directory,
    api_endpoint, rtsp_reconnect_delay_seconds, poll_interval_seconds,
    max_api_retries, api_retry_delay_seconds, enabled
        Unchanged - same meaning as before.

    camera_angle_region_change_percentage    (NEW, optional, default 8.0)
        % change required inside one background region (top/bottom/
        left/right/corner) for that region to count as "changed".

    camera_angle_required_changed_regions    (NEW, optional, default 3)
        How many of the 8 background regions must be "changed"
        before we call it a real camera move (this is what stops a
        single person walking past the edge of frame from alerting -
        that only touches one region).

    camera_angle_minimum_background_coverage (NEW, optional, default 50.0)
        Of all CHANGED background pixels, what % spread (coverage of
        the background area) is required.

    camera_direction_bias_ratio              (NEW, optional, default 1.6)
        How much more one side must change than its opposite side
        before we call the movement "left"/"right"/"up"/"down"
        instead of a generic/tilt change.

    camera_blur_variance_drop_ratio          (NEW, optional, default 0.5)
        If the current image's background sharpness (Laplacian
        variance) drops below this fraction of the original's, it is
        reported as a blur/defocus event.

    camera_blur_minimum_original_variance    (NEW, optional, default 40.0)
        Skip the blur check if the ORIGINAL background was already
        this blurry (avoids false positives on inherently soft feeds).

    roi_margin_percent                       (NEW, optional, default 0.0)
        Extra pixels (as % of width/height) grown outward from your
        polygon before treating it as "ignore area". Use a small
        value (e.g. 0.01-0.02) if products/people right at the edge
        of your ROI are leaking a few pixels into the background and
        causing noise.


IMAGE NAMING
============
    original_YYYYMMDD_HHMMSS.jpg   (created once, permanent)
    current_YYYYMMDD_HHMMSS.jpg    (created once per day)
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


# ================================================================
# PATHS / LOGGING
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

# Allow overriding via environment variable so you don't have to keep
# renaming files back and forth. Falls back to the two names people
# commonly end up with.
_CONFIG_CANDIDATES = [
    os.environ.get("FIXTURE_CONFIG_FILE"),
    "storescript_config copy.json",
    "storescript_config.json",
]

CONFIG_FILE = None
for _candidate in _CONFIG_CANDIDATES:
    if not _candidate:
        continue
    _path = Path(_candidate)
    if not _path.is_absolute():
        _path = BASE_DIR / _path
    if _path.exists():
        CONFIG_FILE = _path
        break

if CONFIG_FILE is None:
    # Keep old default so the error message is familiar if nothing exists.
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
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    with CONFIG_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_path(value: str, default: str) -> Path:
    raw = value or default
    path = Path(raw)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_setting(settings: dict, *names: str, default: Any = None, cast=None) -> Any:
    """
    Look up the first key (in order) that exists in `settings`.
    Lets us support both the key names already in your JSON and any
    new/renamed keys, without breaking either.
    """
    for name in names:
        if name in settings and settings[name] is not None:
            value = settings[name]
            return cast(value) if cast else value
    return default


# ================================================================
# NAME / DIRECTORY HELPERS
# ================================================================

def safe_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9\-_]", "_", str(value or "UNKNOWN"))


def timestamp_for_filename(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def camera_directory(
    root: Path,
    store_id: Any,
    camera_category_name: str,
    camera_name: str,
) -> Path:
    path = root / safe_name(store_id) / safe_name(camera_category_name) / safe_name(camera_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def marker_path(camera_dir: Path, date_compact: str) -> Path:
    return camera_dir / (f".fixture_checked_{date_compact}")


# ================================================================
# IMAGE PATH HELPERS
# ================================================================

def find_original_image(camera_dir: Path) -> Optional[Path]:
    originals = sorted(camera_dir.glob("original_*.jpg"), key=lambda p: p.stat().st_mtime)
    return originals[0] if originals else None


def find_current_image_for_date(camera_dir: Path, date_compact: str) -> Optional[Path]:
    currents = sorted(
        camera_dir.glob(f"current_{date_compact}_*.jpg"),
        key=lambda p: p.stat().st_mtime,
    )
    return currents[0] if currents else None


# ================================================================
# CAMERA LOCK
# ================================================================

def get_camera_lock(camera_no: int, date_compact: str) -> threading.Lock:
    key = (str(camera_no), date_compact)
    with CAMERA_LOCKS_GUARD:
        if key not in CAMERA_LOCKS:
            CAMERA_LOCKS[key] = threading.Lock()
        return CAMERA_LOCKS[key]


# ================================================================
# ROI (polygon) HANDLING  -- NEW
# ================================================================

def extract_polygons_from_camera_config(camera_fixture: dict) -> List[List[Tuple[int, int]]]:
    """
    Pulls every polygon out of:
        camera["fixture_missing"]["regions"][i]["polygon"]

    Your JSON has, per region, a LIST of polygons (usually 2 - looks
    like an outer + inner boundary of the same shelf). All of them are
    collected and unioned into one ignore-mask.
    """
    polygons: List[List[Tuple[int, int]]] = []

    for region in camera_fixture.get("regions", []) or []:
        polygon_entry = region.get("polygon")
        if not polygon_entry:
            continue

        # polygon_entry can be:
        #   [[x,y], [x,y], ...]                     -> a single polygon
        #   [ [[x,y],...], [[x,y],...] ]             -> multiple polygons (your case)
        if polygon_entry and isinstance(polygon_entry[0][0], (int, float)):
            candidate_polygons = [polygon_entry]
        else:
            candidate_polygons = polygon_entry

        for poly in candidate_polygons:
            points = [(int(pt[0]), int(pt[1])) for pt in poly]
            if len(points) >= 3:
                polygons.append(points)

    return polygons


def scale_polygons(
    polygons: List[List[Tuple[int, int]]],
    source_resolution: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[np.ndarray]:
    """
    Rescale polygon points drawn against `source_resolution` (w, h) so
    they line up with `target_size` (w, h) - the actual frame size.
    """
    source_w, source_h = source_resolution
    target_w, target_h = target_size

    if source_w <= 0 or source_h <= 0:
        scale_x, scale_y = 1.0, 1.0
    else:
        scale_x = target_w / float(source_w)
        scale_y = target_h / float(source_h)

    scaled = []
    for poly in polygons:
        pts = np.array(
            [[int(round(x * scale_x)), int(round(y * scale_y))] for x, y in poly],
            dtype=np.int32,
        )
        scaled.append(pts)
    return scaled


def build_roi_mask(
    shape: Tuple[int, int],
    polygons: List[List[Tuple[int, int]]],
    roi_source_resolution: Tuple[int, int],
    margin_percent: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Build a filled mask (255 = inside ROI / ignore area) for the given
    frame `shape` (height, width), from the raw polygon points.
    Returns None if there are no polygons (caller should fall back to
    a plain border mask in that case).
    """
    if not polygons:
        return None

    height, width = shape[:2]
    scaled_polygons = scale_polygons(polygons, roi_source_resolution, (width, height))

    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in scaled_polygons:
        cv2.fillPoly(mask, [poly], 255)

    if margin_percent and margin_percent > 0:
        grow_x = max(1, int(width * margin_percent))
        grow_y = max(1, int(height * margin_percent))
        kernel = np.ones((grow_y * 2 + 1, grow_x * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)

    return mask


def resize_mask_like(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape[:2]
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized


# ================================================================
# IMAGE RESIZE
# ================================================================

def resize_same_size(
    original: np.ndarray,
    current: np.ndarray,
    settings: dict,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if original is None:
        raise ValueError("Original image is required")
    if current is None:
        raise ValueError("Current image is required")

    max_width = int(get_setting(settings, "pixel_comparison_max_width", default=960, cast=int))

    original_height, original_width = original.shape[:2]

    scale = 1.0
    if max_width > 0 and original_width > max_width:
        scale = max_width / original_width
        new_width = max_width
        new_height = int(original_height * scale)
        original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_AREA)

    target_height, target_width = original.shape[:2]
    if current.shape[0] != target_height or current.shape[1] != target_width:
        current = cv2.resize(current, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return original, current, scale


# ================================================================
# PREPROCESSING
# ================================================================

def prepare_pixel_image(image: np.ndarray, settings: dict) -> np.ndarray:
    if image is None:
        raise ValueError("Image is required")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_size = get_setting(
        settings,
        "pixel_comparison_blur_kernel",
        default=None,
    )
    if blur_size is None:
        # fall back to your existing "blur_kernel_size": [5, 5]
        kernel_list = settings.get("blur_kernel_size", [7, 7])
        blur_size = int(kernel_list[0]) if kernel_list else 7
    blur_size = int(blur_size)

    if blur_size < 3:
        blur_size = 3
    if blur_size % 2 == 0:
        blur_size += 1

    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    return gray


# ================================================================
# BACKGROUND MASK (polygon-aware, with border fallback)
# ================================================================

def create_background_mask(
    image_shape: Tuple[int, int],
    settings: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Background = everything that is NOT inside the ROI polygon(s).

    If no polygons were configured for this camera, falls back to the
    original fixed-border-percentage behaviour so nothing breaks for
    cameras you haven't drawn a region for yet.
    """
    height, width = image_shape[:2]

    if roi_mask is not None:
        background = cv2.bitwise_not(roi_mask)
        return background

    # ---- fallback: old border-percentage behaviour ----
    mask = np.zeros((height, width), dtype=np.uint8)
    border_percent = float(get_setting(settings, "pixel_comparison_border_percent", default=0.20, cast=float))
    border_percent = max(0.05, min(border_percent, 0.45))
    border_x = int(width * border_percent)
    border_y = int(height * border_percent)

    mask[0:border_y, :] = 255
    mask[height - border_y:height, :] = 255
    mask[:, 0:border_x] = 255
    mask[:, width - border_x:width] = 255
    return mask


# ================================================================
# BACKGROUND REGIONS (for "is the change spread out" check)
# ================================================================

def create_background_regions(image_shape: Tuple[int, int], settings: dict) -> Dict[str, np.ndarray]:
    height, width = image_shape[:2]

    border_percent = float(get_setting(settings, "pixel_comparison_border_percent", default=0.20, cast=float))
    border_percent = max(0.05, min(border_percent, 0.45))
    border_x = int(width * border_percent)
    border_y = int(height * border_percent)

    regions = {}

    top = np.zeros((height, width), dtype=np.uint8)
    top[0:border_y, :] = 255
    regions["top"] = top

    bottom = np.zeros((height, width), dtype=np.uint8)
    bottom[height - border_y:height, :] = 255
    regions["bottom"] = bottom

    left = np.zeros((height, width), dtype=np.uint8)
    left[:, 0:border_x] = 255
    regions["left"] = left

    right = np.zeros((height, width), dtype=np.uint8)
    right[:, width - border_x:width] = 255
    regions["right"] = right

    top_left = np.zeros((height, width), dtype=np.uint8)
    top_left[0:border_y, 0:border_x] = 255
    regions["top_left"] = top_left

    top_right = np.zeros((height, width), dtype=np.uint8)
    top_right[0:border_y, width - border_x:width] = 255
    regions["top_right"] = top_right

    bottom_left = np.zeros((height, width), dtype=np.uint8)
    bottom_left[height - border_y:height, 0:border_x] = 255
    regions["bottom_left"] = bottom_left

    bottom_right = np.zeros((height, width), dtype=np.uint8)
    bottom_right[height - border_y:height, width - border_x:width] = 255
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
    difference = cv2.absdiff(original_gray, current_gray)

    pixel_threshold = float(
        get_setting(settings, "pixel_comparison_threshold", "pixel_threshold", default=25.0, cast=float)
    )

    changed_pixels = (difference >= pixel_threshold).astype(np.uint8) * 255
    changed_pixels = cv2.bitwise_and(changed_pixels, changed_pixels, mask=background_mask)

    morphology_kernel_size = int(get_setting(settings, "pixel_comparison_morphology_kernel", default=3, cast=int))
    morphology_kernel_size = max(1, morphology_kernel_size)
    kernel = np.ones((morphology_kernel_size, morphology_kernel_size), dtype=np.uint8)

    changed_pixels = cv2.morphologyEx(changed_pixels, cv2.MORPH_OPEN, kernel)
    changed_pixels = cv2.morphologyEx(changed_pixels, cv2.MORPH_CLOSE, kernel)

    background_pixel_count = int(np.count_nonzero(background_mask))
    if background_pixel_count <= 0:
        return changed_pixels, 0.0

    changed_pixel_count = int(np.count_nonzero(changed_pixels))
    changed_percentage = (changed_pixel_count / background_pixel_count) * 100.0

    return changed_pixels, float(changed_percentage)


def calculate_region_change(changed_pixels: np.ndarray, region_mask: np.ndarray) -> float:
    region_pixels = int(np.count_nonzero(region_mask))
    if region_pixels <= 0:
        return 0.0
    changed = cv2.bitwise_and(changed_pixels, changed_pixels, mask=region_mask)
    changed_count = int(np.count_nonzero(changed))
    return float((changed_count / region_pixels) * 100.0)


# ================================================================
# BLUR / DEFOCUS DETECTION  -- NEW
# ================================================================

def sharpness_score(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """
    Variance of the Laplacian, restricted to the background area.
    Lower value = blurrier image. This is a standard, cheap
    focus/blur metric - no ML needed.
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    masked_values = laplacian[mask > 0]
    if masked_values.size == 0:
        return 0.0
    return float(masked_values.var())


def detect_blur_event(
    original_gray: np.ndarray,
    current_gray: np.ndarray,
    background_mask: np.ndarray,
    settings: dict,
) -> Dict[str, Any]:
    original_sharpness = sharpness_score(original_gray, background_mask)
    current_sharpness = sharpness_score(current_gray, background_mask)

    min_original_variance = float(
        get_setting(settings, "camera_blur_minimum_original_variance", default=40.0, cast=float)
    )
    drop_ratio_threshold = float(
        get_setting(settings, "camera_blur_variance_drop_ratio", default=0.5, cast=float)
    )

    is_blurred = False
    if original_sharpness >= min_original_variance and current_sharpness > 0:
        ratio = current_sharpness / original_sharpness
        is_blurred = ratio < drop_ratio_threshold
    elif original_sharpness >= min_original_variance and current_sharpness == 0:
        is_blurred = True

    return {
        "is_blurred": is_blurred,
        "original_sharpness": round(original_sharpness, 2),
        "current_sharpness": round(current_sharpness, 2),
    }


# ================================================================
# MOVEMENT CLASSIFICATION  -- NEW
# ================================================================

def classify_movement(
    region_percentages: Dict[str, float],
    region_change_threshold: float,
    direction_bias_ratio: float,
    tilt_diagonal_ratio: float = 1.15,
    zoom_balance_ratio: float = 1.3,
) -> str:
    """
    Turns the 8 region change percentages into a human label:
    left / right / up / down / tilt / zoom / camera_change.
    This is a heuristic on top of the same pixel-diff data you
    already compute - no extra passes over the images needed.

    Logic, in order:
      1. One side changed much more than its opposite side -> left/right/up/down.
      2. Otherwise, if the two DIAGONAL corner-pairs changed unevenly
         (top-right+bottom-left vs top-left+bottom-right) -> tilt/rotation.
         A pure rotation about the image centre changes one diagonal much
         more than the other; a pure zoom changes both diagonals evenly.
      3. Otherwise, if left/right and top/bottom are both roughly balanced
         (no side or diagonal dominates) but the overall change is high
         everywhere -> zoom.
      4. Otherwise -> camera_change (a mixed/generic movement).
    """
    eps = 1e-6

    left = np.mean([region_percentages["left"], region_percentages["top_left"], region_percentages["bottom_left"]])
    right = np.mean([region_percentages["right"], region_percentages["top_right"], region_percentages["bottom_right"]])
    top = np.mean([region_percentages["top"], region_percentages["top_left"], region_percentages["top_right"]])
    bottom = np.mean([region_percentages["bottom"], region_percentages["bottom_left"], region_percentages["bottom_right"]])

    horizontal_ratio = (max(left, right) + eps) / (min(left, right) + eps)
    vertical_ratio = (max(top, bottom) + eps) / (min(top, bottom) + eps)

    if horizontal_ratio >= direction_bias_ratio and horizontal_ratio >= vertical_ratio:
        return "right" if right > left else "left"

    if vertical_ratio >= direction_bias_ratio:
        return "down" if bottom > top else "up"

    diagonal_a = np.mean([region_percentages["top_right"], region_percentages["bottom_left"]])
    diagonal_b = np.mean([region_percentages["top_left"], region_percentages["bottom_right"]])
    diagonal_ratio = (max(diagonal_a, diagonal_b) + eps) / (min(diagonal_a, diagonal_b) + eps)

    if diagonal_ratio >= tilt_diagonal_ratio:
        return "tilt"

    if horizontal_ratio <= zoom_balance_ratio and vertical_ratio <= zoom_balance_ratio:
        return "zoom"

    return "camera_change"


# ================================================================
# CAMERA ANGLE / VIEW DETECTION
# ================================================================

def detect_camera_angle_change(
    original: np.ndarray,
    current: np.ndarray,
    settings: dict,
    roi_polygons: Optional[List[List[Tuple[int, int]]]] = None,
    roi_source_resolution: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Detect camera movement using PIXEL COMPARISON of the background
    only (everything outside your configured ROI polygon(s)).

    Detects and labels: left, right, up, down, tilt, zoom, blur.
    Any object/person/product change inside the ROI is ignored.
    """
    if original is None:
        raise ValueError("Original image is required")
    if current is None:
        raise ValueError("Current image is required")

    # ---- ROI mask at native (pre-resize) resolution ----
    native_roi_mask = None
    if roi_polygons:
        native_roi_mask = build_roi_mask(
            original.shape,
            roi_polygons,
            roi_source_resolution or (original.shape[1], original.shape[0]),
            margin_percent=float(get_setting(settings, "roi_margin_percent", default=0.0, cast=float)),
        )

    # ---- resize original/current to a common working size ----
    original, current, _scale = resize_same_size(original, current, settings)

    original_gray = prepare_pixel_image(original, settings)
    current_gray = prepare_pixel_image(current, settings)

    roi_mask_resized = None
    if native_roi_mask is not None:
        roi_mask_resized = resize_mask_like(native_roi_mask, original_gray.shape)

    background_mask = create_background_mask(original_gray.shape, settings, roi_mask=roi_mask_resized)

    changed_pixels, changed_percentage = calculate_pixel_difference(
        original_gray, current_gray, background_mask, settings
    )

    regions = create_background_regions(original_gray.shape, settings)
    region_percentages = {name: calculate_region_change(changed_pixels, mask) for name, mask in regions.items()}

    global_change_threshold = float(
        get_setting(settings, "camera_angle_pixel_change_percentage", "change_percentage_threshold", default=8.0, cast=float)
    )
    region_change_threshold = float(
        get_setting(settings, "camera_angle_region_change_percentage", default=8.0, cast=float)
    )
    required_changed_regions = int(
        get_setting(settings, "camera_angle_required_changed_regions", default=3, cast=int)
    )
    minimum_region_coverage = float(
        get_setting(settings, "camera_angle_minimum_region_coverage_percentage",
                    "camera_angle_minimum_background_coverage", default=50.0, cast=float)
    )
    direction_bias_ratio = float(
        get_setting(settings, "camera_direction_bias_ratio", default=1.6, cast=float)
    )
    tilt_diagonal_ratio = float(
        get_setting(settings, "camera_tilt_diagonal_ratio_threshold", default=1.15, cast=float)
    )
    zoom_balance_ratio = float(
        get_setting(settings, "camera_zoom_balance_ratio_threshold", default=1.3, cast=float)
    )

    changed_region_names = [name for name, pct in region_percentages.items() if pct >= region_change_threshold]
    changed_region_count = len(changed_region_names)

    total_background_pixels = int(np.count_nonzero(background_mask))
    changed_background_pixels = int(np.count_nonzero(changed_pixels))
    background_coverage = (
        (changed_background_pixels / total_background_pixels) * 100.0 if total_background_pixels > 0 else 0.0
    )

    global_change = changed_percentage >= global_change_threshold
    enough_regions = changed_region_count >= required_changed_regions
    enough_coverage = background_coverage >= minimum_region_coverage

    movement_camera_changed = global_change and enough_regions and enough_coverage

    # ---- blur / defocus check (independent of the movement check) ----
    blur_info = detect_blur_event(original_gray, current_gray, background_mask, settings)

    camera_angle_changed = movement_camera_changed or blur_info["is_blurred"]

    if blur_info["is_blurred"] and not movement_camera_changed:
        movement_type = "blur"
        reason = "camera_blurry_or_defocused"
    elif movement_camera_changed:
        movement_type = classify_movement(
            region_percentages,
            region_change_threshold,
            direction_bias_ratio,
            tilt_diagonal_ratio=tilt_diagonal_ratio,
            zoom_balance_ratio=zoom_balance_ratio,
        )
        reason = f"camera_background_pixel_change_{movement_type}"
    elif not global_change:
        movement_type = "none"
        reason = "background_pixel_change_below_threshold"
    elif not enough_regions:
        movement_type = "none"
        reason = "change_not_distributed_across_background"
    elif not enough_coverage:
        movement_type = "none"
        reason = "background_coverage_below_threshold"
    else:
        movement_type = "none"
        reason = "stable_camera"

    status = "CHANGE" if camera_angle_changed else "NO_CHANGE"

    logger.info(
        "[Fixture Pixel Angle] global_change=%.2f%% | coverage=%.2f%% | "
        "changed_regions=%d/%d | blur(orig=%.1f,cur=%.1f) | status=%s | movement=%s | reason=%s",
        changed_percentage,
        background_coverage,
        changed_region_count,
        len(regions),
        blur_info["original_sharpness"],
        blur_info["current_sharpness"],
        status,
        movement_type,
        reason,
    )
    logger.info(
        "[Fixture Pixel Regions] %s",
        " | ".join(f"{name}={value:.2f}%" for name, value in region_percentages.items()),
    )

    return {
        "status": status,
        "camera_angle_changed": camera_angle_changed,
        "movement_type": movement_type,
        "reason": reason,
        "changed_percentage": round(float(changed_percentage), 2),
        "background_coverage_percentage": round(float(background_coverage), 2),
        "changed_region_count": changed_region_count,
        "required_changed_regions": required_changed_regions,
        "changed_regions": changed_region_names,
        "region_percentages": {k: round(float(v), 2) for k, v in region_percentages.items()},
        "blur": blur_info,
        "used_polygon_roi": roi_mask_resized is not None,
        "thresholds": {
            "global_change_percentage": global_change_threshold,
            "region_change_percentage": region_change_threshold,
            "required_changed_regions": required_changed_regions,
            "minimum_background_coverage": minimum_region_coverage,
            "direction_bias_ratio": direction_bias_ratio,
            "tilt_diagonal_ratio_threshold": tilt_diagonal_ratio,
            "zoom_balance_ratio_threshold": zoom_balance_ratio,
        },
    }


# ================================================================
# BASELINE
# ================================================================

def load_or_create_baseline(camera_dir: Path, frame: np.ndarray) -> Tuple[np.ndarray, bool, Path]:
    baseline = find_original_image(camera_dir)

    if baseline is not None:
        image = cv2.imread(str(baseline))
        if image is not None:
            return image, False, baseline
        logger.warning("[Fixture] Existing original unreadable. Recreating: %s", baseline)
        try:
            baseline.unlink()
        except OSError:
            logger.exception("[Fixture] Could not delete unreadable original: %s", baseline)

    timestamp = timestamp_for_filename()
    baseline = camera_dir / f"original_{timestamp}.jpg"
    if not cv2.imwrite(str(baseline), frame):
        raise RuntimeError(f"Unable to write original image: {baseline}")

    logger.info("[Fixture] ORIGINAL CREATED | %s", baseline)
    return frame.copy(), True, baseline


# ================================================================
# DAILY CURRENT IMAGE
# ================================================================

def load_or_create_current(
    camera_dir: Path, frame: np.ndarray, date_compact: str
) -> Tuple[np.ndarray, Path, bool]:
    existing_current = find_current_image_for_date(camera_dir, date_compact)

    if existing_current is not None:
        current = cv2.imread(str(existing_current))
        if current is not None:
            return current, existing_current, False
        logger.warning("[Fixture] Existing current image unreadable: %s", existing_current)
        try:
            existing_current.unlink()
        except OSError:
            logger.exception("[Fixture] Could not delete old current: %s", existing_current)

    for old in camera_dir.glob("current_*.jpg"):
        try:
            old.unlink()
        except OSError:
            logger.exception("[Fixture] Could not delete old current: %s", old)

    timestamp = timestamp_for_filename()
    current_path = camera_dir / f"current_{timestamp}.jpg"
    if not cv2.imwrite(str(current_path), frame):
        raise RuntimeError(f"Unable to write current image: {current_path}")

    logger.info("[Fixture] CURRENT CREATED | %s", current_path)
    return frame.copy(), current_path, True


# ================================================================
# RTSP CAPTURE
# ================================================================

def open_rtsp(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def read_frame(rtsp_url: str, reconnect_delay: float) -> Optional[np.ndarray]:
    cap = open_rtsp(rtsp_url)
    if cap is None:
        logger.error("[Fixture] RTSP open failed: %s", rtsp_url)
        time.sleep(reconnect_delay)
        return None
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("[Fixture] RTSP frame read failed")
            return None
        return frame
    finally:
        cap.release()


# ================================================================
# DAILY CHECK TIME
# ================================================================

def is_after_check_time(now: datetime, check_time: str) -> bool:
    try:
        hour, minute = map(int, str(check_time).split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except (TypeError, ValueError):
        logger.error("[Fixture] Invalid daily_check_time=%r; using 13:00", check_time)
        hour, minute = 13, 0
    return now.hour * 60 + now.minute >= hour * 60 + minute


def already_checked_today(camera_dir: Path, date_compact: str) -> bool:
    return marker_path(camera_dir, date_compact).exists()


def mark_checked_today(camera_dir: Path, date_compact: str) -> None:
    marker_path(camera_dir, date_compact).touch(exist_ok=True)


# ================================================================
# API
# ================================================================

def build_api_url(config: dict) -> str:
    base = str(config.get("SERVER_BASE_URL", "http://127.0.0.1:8000")).strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    endpoint = config.get("fixture_missing", {}).get(
        "api_endpoint", "/storescript/api/fixture-missing-event"
    )
    return base + "/" + str(endpoint).lstrip("/")


def send_fixture_event(
    api_url: str,
    store_id: Any,
    store_name: str,
    camera_no: int,
    camera_name: str,
    camera_category_name: str,
    status: str,
    alert: str,
    movement_type: str,
    original_path: Path,
    current_path: Path,
    max_retries: int,
    retry_delay: float,
) -> bool:
    data = {
        "camera_no": str(camera_no),
        "camera_name": str(camera_name),
        "store_id": str(store_id),
        "store_name": str(store_name or ""),
        "camera_category_name": str(camera_category_name),
        "status": str(status).upper(),
        "alert": str(alert).lower(),
        "movement_type": str(movement_type),
    }

    for attempt in range(1, max_retries + 1):
        original_file = None
        current_file = None
        try:
            if not original_path.exists():
                raise FileNotFoundError(f"Original image missing: {original_path}")
            if not current_path.exists():
                raise FileNotFoundError(f"Current image missing: {current_path}")

            original_file = open(original_path, "rb")
            current_file = open(current_path, "rb")

            files = {
                "original_image": (original_path.name, original_file, "image/jpeg"),
                "current_image": (current_path.name, current_file, "image/jpeg"),
            }

            logger.info(
                "[Fixture] POST | attempt=%d/%d | store=%s | camera=%s | status=%s | alert=%s | movement=%s",
                attempt, max_retries, store_id, camera_no, status, alert, movement_type,
            )

            response = requests.post(api_url, data=data, files=files, timeout=30)

            logger.info(
                "[Fixture] API RESPONSE | HTTP=%s | body=%s",
                response.status_code, response.text[:1000],
            )

            if response.status_code in (200, 201):
                return True
            if 400 <= response.status_code < 500:
                logger.error("[Fixture] Non-retryable API error | HTTP=%s", response.status_code)
                return False

        except Exception as exc:
            logger.exception(
                "[Fixture] API request failed | attempt=%d/%d | error=%s", attempt, max_retries, exc
            )
        finally:
            if original_file is not None:
                original_file.close()
            if current_file is not None:
                current_file.close()

        if attempt < max_retries:
            time.sleep(retry_delay)

    return False


# ================================================================
# CAMERA PROCESSOR
# ================================================================

def process_camera(store_name: str, store: dict, camera: dict, config: dict) -> None:
    camera_no = int(camera["id"])
    camera_category_name = str(camera.get("category_name", camera.get("name", f"camera_{camera_no}")))
    camera_name = f"camera_{camera_no}"
    rtsp_url = camera.get("rtsp_url")

    if not rtsp_url:
        logger.error("[Fixture] Missing RTSP URL | store=%s | camera=%s", store_name, camera_no)
        return

    global_fixture = config.get("fixture_missing", {})
    camera_fixture = camera.get("fixture_missing", {})

    if not global_fixture.get("enabled", True):
        logger.info("[Fixture] Globally disabled")
        return
    if not camera_fixture.get("enabled", True):
        logger.info("[Fixture] Camera disabled | camera=%s", camera_no)
        return

    # ---- merge global + per-camera settings (camera overrides global) ----
    merged_settings = {**global_fixture, **{k: v for k, v in camera_fixture.items() if k != "regions"}}

    # ---- ROI polygons for this camera ----
    roi_polygons = extract_polygons_from_camera_config(camera_fixture)
    roi_source_resolution = tuple(
        global_fixture.get("roi_source_resolution", camera_fixture.get("roi_source_resolution", [0, 0]))
    ) or None
    if roi_polygons and (not roi_source_resolution or roi_source_resolution == (0, 0)):
        logger.warning(
            "[Fixture] camera=%s has ROI polygons but no roi_source_resolution set - "
            "assuming polygons match the live frame size exactly.",
            camera_no,
        )

    if roi_polygons:
        logger.info(
            "[Fixture] camera=%s using %d polygon(s) as the ignore-area (ROI)", camera_no, len(roi_polygons)
        )
    else:
        logger.info(
            "[Fixture] camera=%s has no polygons configured - falling back to a fixed border %% "
            "for the background area.",
            camera_no,
        )

    baseline_root = resolve_path(
        global_fixture.get("baseline_directory", "./var/data/fixture"), "./var/data/fixture"
    )
    snapshot_root = resolve_path(
        global_fixture.get("snapshot_directory", "./var/data/fixture_missing_snapshots"),
        "./var/data/fixture_missing_snapshots",
    )

    camera_dir = camera_directory(baseline_root, store.get("store_id"), camera_category_name, camera_name)
    snapshot_dir = (
        snapshot_root / safe_name(store.get("store_id")) / safe_name(camera_category_name) / safe_name(camera_name)
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    daily_check_time = global_fixture.get("daily_check_time", "13:00")
    reconnect_delay = float(global_fixture.get("rtsp_reconnect_delay_seconds", 5))
    poll_interval = float(global_fixture.get("poll_interval_seconds", 0.25))
    api_url = build_api_url(config)

    logger.info(
        "[Fixture] Camera started | store=%s | camera=%s | category=%s | RTSP=%s",
        store.get("store_id"), camera_no, camera_category_name, rtsp_url,
    )
    logger.info("[Fixture] MODE = PIXEL CAMERA ANGLE + BLUR (polygon ROI aware)")
    logger.info("[Fixture] Object/person/product changes inside the ROI polygon are ignored")

    while True:
        now = datetime.now()
        date_compact = now.strftime("%Y%m%d")
        date_text = now.strftime("%Y-%m-%d")

        if already_checked_today(camera_dir, date_compact):
            time.sleep(5)
            continue

        frame = read_frame(rtsp_url, reconnect_delay)
        if frame is None:
            time.sleep(reconnect_delay)
            continue

        try:
            original, original_created, original_path = load_or_create_baseline(camera_dir, frame)
        except Exception:
            logger.exception("[Fixture] Baseline failure | camera=%s", camera_no)
            time.sleep(poll_interval)
            continue

        if original_created:
            logger.info(
                "[Fixture] Baseline ready | camera=%s | original=%s | waiting for %s",
                camera_no, original_path.name, daily_check_time,
            )
            time.sleep(poll_interval)
            continue

        if not is_after_check_time(now, daily_check_time):
            time.sleep(poll_interval)
            continue

        lock = get_camera_lock(camera_no, date_compact)
        if not lock.acquire(blocking=False):
            time.sleep(1)
            continue

        try:
            if already_checked_today(camera_dir, date_compact):
                continue

            current, current_path, _current_created = load_or_create_current(camera_dir, frame, date_compact)

            angle_result = detect_camera_angle_change(
                original,
                current,
                merged_settings,
                roi_polygons=roi_polygons,
                roi_source_resolution=roi_source_resolution,
            )

            logger.info(
                "[Fixture PIXEL TEST] camera=%s | status=%s | movement=%s | changed=%.2f%% | "
                "coverage=%.2f%% | regions=%d | reason=%s",
                camera_no,
                angle_result.get("status"),
                angle_result.get("movement_type"),
                angle_result.get("changed_percentage", 0.0),
                angle_result.get("background_coverage_percentage", 0.0),
                angle_result.get("changed_region_count", 0),
                angle_result.get("reason", ""),
            )

            if angle_result.get("camera_angle_changed", False):
                status = "YES"
                alert = "yes"
                movement_type = angle_result.get("movement_type", "camera_change")

                logger.warning(
                    "[Fixture] CAMERA ANGLE / VIEW CHANGE DETECTED | camera=%s | movement=%s | "
                    "changed=%.2f%% | coverage=%.2f%% | regions=%d | reason=%s",
                    camera_no, movement_type,
                    angle_result.get("changed_percentage", 0.0),
                    angle_result.get("background_coverage_percentage", 0.0),
                    angle_result.get("changed_region_count", 0),
                    angle_result.get("reason", ""),
                )

                alert_snapshot = snapshot_dir / (
                    f"camera_{movement_type}_{date_compact}_{now.strftime('%H%M%S_%f')}.jpg"
                )
                try:
                    saved = cv2.imwrite(str(alert_snapshot), current)
                    if saved:
                        logger.info("[Fixture] CAMERA ANGLE SNAPSHOT SAVED | %s", alert_snapshot)
                    else:
                        logger.error("[Fixture] CAMERA ANGLE SNAPSHOT WRITE FAILED | %s", alert_snapshot)
                except Exception:
                    logger.exception("[Fixture] Camera angle snapshot save failed")
            else:
                status = "NO"
                alert = "no"
                movement_type = "none"
                logger.info(
                    "[Fixture] CAMERA STABLE | camera=%s | object/product/person changes inside "
                    "ROI are ignored",
                    camera_no,
                )

            success = send_fixture_event(
                api_url=api_url,
                store_id=store.get("store_id"),
                store_name=store_name,
                camera_no=camera_no,
                camera_name=camera_name,
                camera_category_name=camera_category_name,
                status=status,
                alert=alert,
                movement_type=movement_type,
                original_path=original_path,
                current_path=current_path,
                max_retries=int(global_fixture.get("max_api_retries", 5)),
                retry_delay=float(global_fixture.get("api_retry_delay_seconds", 5)),
            )

            if success:
                mark_checked_today(camera_dir, date_compact)
                logger.info(
                    "[Fixture] DAILY CHECK COMPLETED | store=%s | camera=%s | date=%s | status=%s | alert=%s",
                    store.get("store_id"), camera_no, date_text, status, alert,
                )
            else:
                logger.error(
                    "[Fixture] DAILY CHECK NOT MARKED COMPLETE | camera=%s | API delivery failed", camera_no
                )

        except Exception:
            logger.exception("[Fixture] Daily processing failed | camera=%s", camera_no)
        finally:
            lock.release()

        time.sleep(poll_interval)


# ================================================================
# MAIN
# ================================================================

def main():
    config = load_config()
    threads = []
    stores = config.get("stores", {})

    if not stores:
        raise RuntimeError("No stores configured")

    for store_name, store in stores.items():
        for camera in store.get("gates", []):
            if not camera.get("fixture_missing", {}).get("enabled", True):
                continue

            thread = threading.Thread(
                target=process_camera,
                args=(store_name, store, camera, config),
                daemon=True,
                name=f"FixtureCamera-{camera.get('id')}",
            )
            thread.start()
            threads.append(thread)

    logger.info("[Fixture] Started %d camera thread(s)", len(threads))

    if not threads:
        raise RuntimeError("No enabled fixture_missing cameras found")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("[Fixture] Stopped by user")


if __name__ == "__main__":
    main()