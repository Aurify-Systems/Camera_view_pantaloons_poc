"""
fixture_missing.py
==================

Fixture Missing Detection.

IMAGE NAMING
------------

Original:
    original_YYYYMMDD_HHMMSS.jpg

Current:
    current_YYYYMMDD_HHMMSS.jpg

Example:
    original_20260907_000815.jpg
    current_20260907_130522.jpg

The original image is created once and remains permanent.

The current image is created once per day.

The API receives the exact same filenames.
"""

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO


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

PERSON_CLASS_ID = 0
PERSON_MODEL = None
PERSON_MODEL_LOCK = threading.Lock()

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
        r"[^a-zA-Z0-9\_-]",
        "_",
        str(value or "UNKNOWN"),
    )


def timestamp_for_filename(
    now: Optional[datetime] = None,
) -> str:
    """
    Generate timestamp for image filename.

    Example:
        20260907_130522
    """

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
    Find the permanent original image.

    New format:

        original_YYYYMMDD_HHMMSS.jpg

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
    """
    Find today's current image.

    Example:

        current_20260907_130522.jpg
    """

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
# YOLO PERSON MODEL
# ================================================================

def get_person_model(
    settings: dict,
):

    global PERSON_MODEL

    if PERSON_MODEL is not None:
        return PERSON_MODEL

    with PERSON_MODEL_LOCK:

        if PERSON_MODEL is not None:
            return PERSON_MODEL

        model_path = settings.get(
            "person_model",
            "./models/yolo11n.pt",
        )

        model_path = str(
            (
                BASE_DIR / model_path
            ).resolve()
            if not Path(model_path).is_absolute()
            else model_path
        )

        logger.info(
            "[Fixture] Loading YOLO person model: %s",
            model_path,
        )

        PERSON_MODEL = YOLO(
            model_path
        )

        logger.info(
            "[Fixture] YOLO person model loaded"
        )

    return PERSON_MODEL


def create_person_mask(
    image: np.ndarray,
    settings: dict,
) -> np.ndarray:

    if image is None:
        raise ValueError(
            "Image is required for person detection"
        )

    height, width = image.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    model = get_person_model(
        settings
    )

    confidence = float(
        settings.get(
            "person_confidence",
            0.50,
        )
    )

    padding = int(
        settings.get(
            "person_mask_padding",
            10,
        )
    )

    padding = max(
        0,
        padding,
    )

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )

    person_count = 0

    for result in results:

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:
            continue

        classes = boxes.cls
        xyxy = boxes.xyxy

        if classes is None or xyxy is None:
            continue

        classes = (
            classes.cpu().numpy()
        )

        xyxy = (
            xyxy.cpu().numpy()
        )

        for class_id, box in zip(
            classes,
            xyxy,
        ):

            if int(class_id) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box

            x1 = (
                int(round(x1))
                - padding
            )

            y1 = (
                int(round(y1))
                - padding
            )

            x2 = (
                int(round(x2))
                + padding
            )

            y2 = (
                int(round(y2))
                + padding
            )

            x1 = max(
                0,
                min(
                    x1,
                    width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    y1,
                    height - 1,
                ),
            )

            x2 = max(
                0,
                min(
                    x2,
                    width - 1,
                ),
            )

            y2 = max(
                0,
                min(
                    y2,
                    height - 1,
                ),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                mask,
                (x1, y1),
                (x2, y2),
                255,
                -1,
            )

            person_count += 1

    logger.debug(
        "[Fixture] Person detection | persons=%d",
        person_count,
    )

    return mask


# ================================================================
# ROI HELPERS
# ================================================================

def looks_like_point(
    value: Any,
) -> bool:

    return (
        isinstance(
            value,
            (list, tuple),
        )
        and len(value) >= 2
        and isinstance(
            value[0],
            (int, float),
        )
        and isinstance(
            value[1],
            (int, float),
        )
    )


def normalize_polygons(
    raw_polygon: Any,
) -> List[List[List[float]]]:

    if not raw_polygon or not isinstance(
        raw_polygon,
        (list, tuple),
    ):
        return []

    first = raw_polygon[0]

    if looks_like_point(first):
        return [
            list(raw_polygon)
        ]

    if isinstance(
        first,
        (list, tuple),
    ):

        result = []

        for polygon in raw_polygon:

            if not isinstance(
                polygon,
                (list, tuple),
            ):
                continue

            result.append(
                list(polygon)
            )

        return result

    return []


def scale_single_polygon(
    polygon: List[Any],
    source_resolution: Tuple[int, int],
    target_resolution: Tuple[int, int],
) -> Optional[np.ndarray]:

    if not polygon or len(polygon) < 3:
        return None

    source_w, source_h = (
        source_resolution
    )

    target_w, target_h = (
        target_resolution
    )

    if source_w <= 0 or source_h <= 0:
        return None

    if target_w <= 0 or target_h <= 0:
        return None

    scale_x = (
        target_w / source_w
    )

    scale_y = (
        target_h / source_h
    )

    scaled = []

    for point in polygon:

        if not isinstance(
            point,
            (list, tuple),
        ):
            continue

        if len(point) < 2:
            continue

        try:

            x = int(
                round(
                    float(point[0])
                    * scale_x
                )
            )

            y = int(
                round(
                    float(point[1])
                    * scale_y
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        x = max(
            0,
            min(
                x,
                target_w - 1,
            ),
        )

        y = max(
            0,
            min(
                y,
                target_h - 1,
            ),
        )

        scaled.append(
            [x, y]
        )

    if len(scaled) < 3:
        return None

    return np.asarray(
        scaled,
        dtype=np.int32,
    )


def scale_polygons(
    raw_polygon: Any,
    source_resolution: Tuple[int, int],
    target_resolution: Tuple[int, int],
) -> List[np.ndarray]:

    polygons = normalize_polygons(
        raw_polygon
    )

    result = []

    for polygon in polygons:

        scaled = scale_single_polygon(
            polygon,
            source_resolution,
            target_resolution,
        )

        if scaled is not None:
            result.append(
                scaled
            )

    return result


def create_roi_mask(
    shape: Tuple[int, ...],
    polygons: List[np.ndarray],
) -> np.ndarray:

    height, width = shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if polygons:

        cv2.fillPoly(
            mask,
            polygons,
            255,
        )

    return mask


# ================================================================
# IMAGE COMPARISON
# ================================================================

def compare_images(
    original: np.ndarray,
    current: np.ndarray,
    polygons: List[np.ndarray],
    settings: dict,
) -> Dict[str, Any]:

    if original is None:
        raise ValueError(
            "Original image is required"
        )

    if current is None:
        raise ValueError(
            "Current image is required"
        )

    if original.shape[:2] != current.shape[:2]:

        current = cv2.resize(
            current,
            (
                original.shape[1],
                original.shape[0],
            ),
            interpolation=cv2.INTER_AREA,
        )

    roi_mask = create_roi_mask(
        original.shape,
        polygons,
    )

    try:

        original_person_mask = (
            create_person_mask(
                original,
                settings,
            )
        )

        current_person_mask = (
            create_person_mask(
                current,
                settings,
            )
        )

        person_mask = cv2.bitwise_or(
            original_person_mask,
            current_person_mask,
        )

        inverse_person_mask = (
            cv2.bitwise_not(
                person_mask
            )
        )

        effective_mask = (
            cv2.bitwise_and(
                roi_mask,
                inverse_person_mask,
            )
        )

        excluded_person_pixels = int(
            np.count_nonzero(
                (roi_mask > 0)
                & (person_mask > 0)
            )
        )

    except Exception:

        logger.exception(
            "[Fixture] Person exclusion failed"
        )

        return {
            "status": "NO_CHANGE",
            "change_percentage": 0.0,
            "changed_pixels": 0,
            "roi_pixels": 0,
            "excluded_person_pixels": 0,
            "person_exclusion_failed": True,
        }

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY,
    )

    current_gray = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY,
    )

    kernel = settings.get(
        "blur_kernel_size",
        [5, 5],
    )

    try:

        kw = int(kernel[0])
        kh = int(kernel[1])

    except (
        TypeError,
        ValueError,
        IndexError,
    ):

        kw, kh = 5, 5

    kw = max(
        1,
        kw,
    )

    kh = max(
        1,
        kh,
    )

    if kw % 2 == 0:
        kw += 1

    if kh % 2 == 0:
        kh += 1

    original_gray = cv2.GaussianBlur(
        original_gray,
        (kw, kh),
        0,
    )

    current_gray = cv2.GaussianBlur(
        current_gray,
        (kw, kh),
        0,
    )

    difference = cv2.absdiff(
        original_gray,
        current_gray,
    )

    roi_difference = difference[
        effective_mask > 0
    ]

    if roi_difference.size == 0:

        return {
            "status": "NO_CHANGE",
            "change_percentage": 0.0,
            "changed_pixels": 0,
            "roi_pixels": 0,
            "excluded_person_pixels": (
                excluded_person_pixels
            ),
            "person_exclusion_failed": False,
        }

    pixel_threshold = int(
        settings.get(
            "pixel_threshold",
            40,
        )
    )

    min_changed_pixels = int(
        settings.get(
            "min_changed_pixels",
            1000,
        )
    )

    percentage_threshold = float(
        settings.get(
            "change_percentage_threshold",
            10.0,
        )
    )

    changed_mask = (
        roi_difference
        > pixel_threshold
    )

    changed_pixels = int(
        np.count_nonzero(
            changed_mask
        )
    )

    roi_pixels = int(
        roi_difference.size
    )

    change_percentage = (
        changed_pixels
        / roi_pixels
    ) * 100.0

    changed = (
        changed_pixels
        >= min_changed_pixels
        and change_percentage
        >= percentage_threshold
    )

    return {
        "status": (
            "CHANGE"
            if changed
            else "NO_CHANGE"
        ),
        "change_percentage": round(
            change_percentage,
            2,
        ),
        "changed_pixels": changed_pixels,
        "roi_pixels": roi_pixels,
        "excluded_person_pixels": (
            excluded_person_pixels
        ),
        "person_exclusion_failed": False,
        "pixel_threshold": pixel_threshold,
        "change_percentage_threshold": (
            percentage_threshold
        ),
        "min_changed_pixels": (
            min_changed_pixels
        ),
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

    # ------------------------------------------------------------
    # EXISTING ORIGINAL
    # ------------------------------------------------------------

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
            "[Fixture] Existing original "
            "is unreadable. Recreating: %s",
            baseline,
        )

        try:
            baseline.unlink()

        except OSError:

            logger.exception(
                "[Fixture] Could not delete "
                "unreadable original: %s",
                baseline,
            )

    # ------------------------------------------------------------
    # CREATE ORIGINAL
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # FIND TODAY'S CURRENT IMAGE
    # ------------------------------------------------------------

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
            "[Fixture] Existing current "
            "image unreadable: %s",
            existing_current,
        )

        try:

            existing_current.unlink()

        except OSError:

            logger.exception(
                "[Fixture] Could not delete "
                "unreadable current: %s",
                existing_current,
            )

    # ------------------------------------------------------------
    # REMOVE OLD CURRENT IMAGES
    # ------------------------------------------------------------

    for old in camera_dir.glob(
        "current_*.jpg"
    ):

        try:

            old.unlink()

        except OSError:

            logger.exception(
                "[Fixture] Could not delete "
                "old current: %s",
                old,
            )

    # ------------------------------------------------------------
    # CREATE TODAY'S CURRENT IMAGE
    # ------------------------------------------------------------

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
# DAILY CHECK
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
            "using 11:15",
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
        ("http://", "https://")
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
                "[Fixture] POST | attempt=%d/%d | "
                "store=%s | camera=%s | "
                "status=%s | alert=%s",
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

    baseline_root = resolve_path(
        global_fixture.get(
            "baseline_directory",
            "./var/data/fixture",
        ),
        "./var/data/fixture",
    )

    snapshot_root = resolve_path(
        global_fixture.get(
            "snapshot_directory",
            "./var/data/fixture_missing_snapshots",
        ),
        "./var/data/fixture_missing_snapshots",
    )

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

    daily_check_time = (
        global_fixture.get(
            "daily_check_time",
            "11:20",
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

    source_resolution_raw = (
        global_fixture.get(
            "roi_source_resolution",
            [960, 1080],
        )
    )

    source_resolution = (
        int(source_resolution_raw[0]),
        int(source_resolution_raw[1]),
    )

    api_url = build_api_url(
        config
    )

    logger.info(
        "[Fixture] Camera started | "
        "store=%s | camera=%s | "
        "category=%s | RTSP=%s",
        store.get("store_id"),
        camera_no,
        camera_category_name,
        rtsp_url,
    )

    while True:

        now = datetime.now()

        date_compact = now.strftime(
            "%Y%m%d"
        )

        date_text = now.strftime(
            "%Y-%m-%d"
        )

        # --------------------------------------------------------
        # DAILY GUARD
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # READ FRAME
        # --------------------------------------------------------

        frame = read_frame(
            rtsp_url,
            reconnect_delay,
        )

        if frame is None:

            time.sleep(
                reconnect_delay
            )

            continue

        # --------------------------------------------------------
        # ORIGINAL
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # WAIT FOR DAILY CHECK
        # --------------------------------------------------------

        if not is_after_check_time(
            now,
            daily_check_time,
        ):

            time.sleep(
                poll_interval
            )

            continue

        # --------------------------------------------------------
        # LOCK
        # --------------------------------------------------------

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

            # ----------------------------------------------------
            # CURRENT IMAGE
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # ROI
            # ----------------------------------------------------

            target_resolution = (
                current.shape[1],
                current.shape[0],
            )

            regions = camera_fixture.get(
                "regions",
                [],
            )

            if not regions:

                logger.error(
                    "[Fixture] No fixture regions "
                    "configured | camera=%s",
                    camera_no,
                )

                continue

            comparisons = []

            # ----------------------------------------------------
            # REQUIRED CHANGE FRAMES
            # ----------------------------------------------------

            required_frames = max(
                1,
                int(
                    global_fixture.get(
                        "required_change_frames",
                        1,
                    )
                ),
            )

            for region in regions:

                region_id = str(
                    region.get(
                        "id",
                        "fixture_region",
                    )
                )

                polygons = scale_polygons(
                    region.get(
                        "polygon",
                        [],
                    ),
                    source_resolution,
                    target_resolution,
                )

                if not polygons:

                    logger.error(
                        "[Fixture] Invalid ROI | "
                        "camera=%s | region=%s",
                        camera_no,
                        region_id,
                    )

                    continue

                logger.info(
                    "[Fixture TEST] "
                    "ORIGINAL=%s | CURRENT=%s",
                    str(original_path),
                    str(current_path),
                )

                first_result = compare_images(
                    original,
                    current,
                    polygons,
                    global_fixture,
                )

                logger.info(
                    "[Fixture TEST] camera=%s | "
                    "region=%s | status=%s | "
                    "change_percentage=%.2f | "
                    "changed_pixels=%d | "
                    "roi_pixels=%d | "
                    "person_excluded=%d",
                    camera_no,
                    region_id,
                    first_result.get("status"),
                    first_result.get(
                        "change_percentage",
                        0.0,
                    ),
                    first_result.get(
                        "changed_pixels",
                        0,
                    ),
                    first_result.get(
                        "roi_pixels",
                        0,
                    ),
                    first_result.get(
                        "excluded_person_pixels",
                        0,
                    ),
                )

                results_for_confirmation = [
                    first_result
                ]

                if (
                    first_result["status"]
                    == "CHANGE"
                    and required_frames > 1
                ):

                    for _ in range(
                        required_frames - 1
                    ):

                        confirmation_frame = (
                            read_frame(
                                rtsp_url,
                                reconnect_delay,
                            )
                        )

                        if confirmation_frame is None:

                            results_for_confirmation.append(
                                {
                                    "status": "NO_CHANGE",
                                    "change_percentage": 0.0,
                                    "changed_pixels": 0,
                                    "roi_pixels": 0,
                                    "person_exclusion_failed": True,
                                }
                            )

                            break

                        confirmation_result = (
                            compare_images(
                                original,
                                confirmation_frame,
                                polygons,
                                global_fixture,
                            )
                        )

                        results_for_confirmation.append(
                            confirmation_result
                        )

                confirmed_change = (
                    len(
                        results_for_confirmation
                    )
                    == required_frames
                    and all(
                        result.get("status")
                        == "CHANGE"
                        for result
                        in results_for_confirmation
                    )
                )

                final_result = dict(
                    results_for_confirmation[0]
                )

                if not confirmed_change:

                    final_result["status"] = (
                        "NO_CHANGE"
                    )

                final_result[
                    "confirmation_frames"
                ] = len(
                    results_for_confirmation
                )

                final_result[
                    "required_change_frames"
                ] = required_frames

                final_result[
                    "region_id"
                ] = region_id

                comparisons.append(
                    final_result
                )

            if not comparisons:

                logger.error(
                    "[Fixture] No valid regions compared | "
                    "camera=%s",
                    camera_no,
                )

                continue

            # ----------------------------------------------------
            # API
            # ----------------------------------------------------

            all_api_success = True

            for result in comparisons:

                region_id = result[
                    "region_id"
                ]

                status = (
                    "YES"
                    if result["status"]
                    == "CHANGE"
                    else "NO"
                )

                alert = (
                    "yes"
                    if result["status"]
                    == "CHANGE"
                    else "no"
                )

                if result["status"] == "CHANGE":

                    alert_snapshot = (
                        snapshot_dir
                        / (
                            f"fixture_"
                            f"{date_compact}_"
                            f"{now.strftime('%H%M%S_%f')}_"
                            f"{safe_name(region_id)}.jpg"
                        )
                    )

                    try:

                        cv2.imwrite(
                            str(alert_snapshot),
                            current,
                        )

                    except Exception:

                        logger.exception(
                            "[Fixture] Alert snapshot "
                            "save failed"
                        )

                    logger.warning(
                        "[Fixture] CHANGE DETECTED | "
                        "camera=%s | region=%s | "
                        "change=%.2f%% | "
                        "changed_pixels=%d/%d",
                        camera_no,
                        region_id,
                        result.get(
                            "change_percentage",
                            0.0,
                        ),
                        result.get(
                            "changed_pixels",
                            0,
                        ),
                        result.get(
                            "roi_pixels",
                            0,
                        ),
                    )

                else:

                    logger.info(
                        "[Fixture] NO_CHANGE | "
                        "camera=%s | region=%s | "
                        "change=%.2f%%",
                        camera_no,
                        region_id,
                        result.get(
                            "change_percentage",
                            0.0,
                        ),
                    )

                # ------------------------------------------------
                # SEND EXACT IMAGE FILES
                # ------------------------------------------------

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

                if not success:
                    all_api_success = False

            # ----------------------------------------------------
            # DAILY COMPLETION
            # ----------------------------------------------------

            if all_api_success:

                mark_checked_today(
                    camera_dir,
                    date_compact,
                )

                logger.info(
                    "[Fixture] DAILY CHECK COMPLETED | "
                    "store=%s | camera=%s | date=%s",
                    store.get("store_id"),
                    camera_no,
                    date_text,
                )

            else:

                logger.error(
                    "[Fixture] DAILY CHECK NOT MARKED "
                    "COMPLETE | camera=%s | "
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


if __name__ == "__main__":
    main()