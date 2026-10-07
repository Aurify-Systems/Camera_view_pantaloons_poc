"""
fixture_missing.py  (v4 - whole-frame, lighting-proof, run-once at 1 PM)
=======================================================================

PURPOSE
-------
Once a day (your scheduler starts it at 1 PM) compare every camera's
current view with its ORIGINAL image and report whether the CAMERA ANGLE
changed: left, right, up, down, tilt, zoom, black, blur.
The whole frame is used (no polygons). The script checks all cameras,
prints a summary and EXITS.

HOW ANGLE CHANGE IS DETECTED (all lighting-independent, whole frame)
-------------------------------------------------------------------
1. FEATURE REGISTRATION: SIFT (ORB as backup) after contrast
   normalisation; a shift / rotation / zoom transform is fitted with
   RANSAC. People and moved products are outliers.

2. TEMPLATE MATCHING (needs no keypoints): the centre of the original's
   structure map (contrast-normalised gradients) is slid over the
   current one. Handles repetitive shelves, low-texture scenes and
   LARGE pans, and cross-checks the feature result.

3. If NOTHING can be aligned and the scene no longer resembles the
   original (scene_similarity_threshold) -> "camera_change".

4. BLACK frame (signal loss / covered lens) and BLUR (contrast-
   normalised sharpness drop).

5. CONFIRMATION: several frames, majority vote.

A side-by-side debug image (original | current) is written to
<baseline_dir>/<store>/<category>/<camera>/debug_latest.jpg every run.

RUN-ONCE
--------
 - Every run captures fresh frames and replaces today's current image.
 - No original yet -> it is created and that camera is skipped today.
 - Already checked today (marker file) -> skipped, unless
   "force_recheck": true.
 - The check runs at/after daily_check_time (default "13:00"):
     started earlier -> it waits until 13:00; started later -> runs now.
     "wait_for_check_time": false -> exit instead of waiting.
     Testing: python fixture_missing.py --now   (or "ignore_check_time": true)

movement_type sent to the API:
    left, right, up, down, zoom, tilt, black, blur,
    camera_change (view changed, direction unclear), none
Direction labels describe CAMERA movement; flip with
"invert_direction_labels": true if yours come out reversed.

CONFIG KEYS (config["fixture_missing"], per-camera block overrides)
===================================================================
    -- registration --
    shift_threshold_percent             0.25  % of width/height (~2.4 px @960)
    rotation_threshold_degrees          0.3
    scale_threshold_percent             0.8
    min_registration_inliers            15
    orb_features                        3000
    -- edge layout (lighting independent) --
    edge_percentile                     92    gradient percentile kept as edge
    edge_match_tolerance_px             2
    edge_mismatch_global_percentage     25
    edge_mismatch_cell_percentage       30
    edge_mismatch_required_cells        4
    edge_mismatch_hard_percentage       40
    edge_min_cell_edge_pixels           40
    camera_angle_grid_size              4
    -- black / blur --
    black_mean_threshold                8
    black_dark_pixel_ratio              0.98
    camera_blur_variance_drop_ratio     0.5
    camera_blur_minimum_original_variance 0.01
    -- run control --
    confirmation_frames                 3
    confirmation_interval_seconds       1.0
    capture_attempts / warmup_frames / force_recheck / invert_direction_labels

IMAGE NAMING
============
    original_YYYYMMDD_HHMMSS.jpg   (created once, permanent)
    current_YYYYMMDD_HHMMSS.jpg    (replaced on every run)
"""

import json
import logging
import os
import re
import sys
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

RESULTS: List[Dict[str, Any]] = []
RESULTS_LOCK = threading.Lock()


def record_result(**kwargs) -> None:
    with RESULTS_LOCK:
        RESULTS.append(kwargs)


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


def camera_directory(root: Path, store_id: Any, camera_category_name: str, camera_name: str) -> Path:
    path = root / safe_name(store_id) / safe_name(camera_category_name) / safe_name(camera_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def marker_path(camera_dir: Path, date_compact: str) -> Path:
    return camera_dir / f".fixture_checked_{date_compact}"


def find_original_image(camera_dir: Path) -> Optional[Path]:
    originals = sorted(camera_dir.glob("original_*.jpg"), key=lambda p: p.stat().st_mtime)
    return originals[0] if originals else None


def get_camera_lock(camera_no: int, date_compact: str) -> threading.Lock:
    key = (str(camera_no), date_compact)
    with CAMERA_LOCKS_GUARD:
        if key not in CAMERA_LOCKS:
            CAMERA_LOCKS[key] = threading.Lock()
        return CAMERA_LOCKS[key]


# ================================================================
# IMAGE PREPARATION
# ================================================================

def resize_same_size(original: np.ndarray, current: np.ndarray, settings: dict):
    """Resize both images to the same working size (max width)."""
    if original is None or current is None:
        raise ValueError("Original and current images are required")

    max_width = int(get_setting(settings, "pixel_comparison_max_width", default=960, cast=int))
    oh, ow = original.shape[:2]

    if max_width > 0 and ow > max_width:
        scale = max_width / ow
        original = cv2.resize(original, (max_width, int(oh * scale)), interpolation=cv2.INTER_AREA)

    th, tw = original.shape[:2]
    if current.shape[0] != th or current.shape[1] != tw:
        current = cv2.resize(current, (tw, th), interpolation=cv2.INTER_AREA)

    return original, current


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, settings: dict) -> np.ndarray:
    kernel_list = settings.get("blur_kernel_size", [5, 5])
    k = int(kernel_list[0]) if kernel_list else 5
    k = max(3, k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(gray, (k, k), 0)


# ================================================================
# BLACK FRAME  (signal loss / covered lens - NOT normal dimming)
# ================================================================

def detect_black_frame(gray: np.ndarray, settings: dict) -> Dict[str, Any]:
    mean_threshold = float(get_setting(settings, "black_mean_threshold", default=8.0, cast=float))
    dark_ratio_threshold = float(get_setting(settings, "black_dark_pixel_ratio", default=0.98, cast=float))

    mean_value = float(gray.mean())
    dark_ratio = float((gray < 12).mean())
    is_black = mean_value < mean_threshold or dark_ratio >= dark_ratio_threshold

    return {"is_black": is_black, "mean": round(mean_value, 2), "dark_ratio": round(dark_ratio, 3)}


# ================================================================
# BLUR / DEFOCUS (whole frame, contrast-normalised => lighting independent)
# ================================================================

def sharpness_score(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    std = float(g.std())
    if std < 1e-3:
        return 0.0
    g = (g - float(g.mean())) / std
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def detect_blur_event(original_gray: np.ndarray, current_gray: np.ndarray, settings: dict) -> Dict[str, Any]:
    original_sharpness = sharpness_score(original_gray)
    current_sharpness = sharpness_score(current_gray)

    min_original = float(get_setting(settings, "camera_blur_minimum_original_variance", default=0.01, cast=float))
    drop_ratio = float(get_setting(settings, "camera_blur_variance_drop_ratio", default=0.5, cast=float))

    is_blurred = False
    if original_sharpness >= min_original:
        is_blurred = current_sharpness < original_sharpness * drop_ratio

    return {
        "is_blurred": is_blurred,
        "original_sharpness": round(original_sharpness, 4),
        "current_sharpness": round(current_sharpness, 4),
    }


# ================================================================
# EDGE-LAYOUT COMPARISON (whole frame + grid spread, lighting independent)
# ================================================================

def extract_edges(gray: np.ndarray, settings: dict) -> np.ndarray:
    """
    Thin edge map whose threshold is a PERCENTILE of the image's own
    gradient strength, so overall brightness / contrast changes do not
    change which pixels count as edges.
    """
    percentile = float(get_setting(settings, "edge_percentile", default=92.0, cast=float))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(cv2.GaussianBlur(gray, (5, 5), 0))

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    high = float(np.percentile(magnitude, percentile))
    high = max(high, 1.0)
    # Canny works on 8-bit gradients; use its own Sobel with our thresholds
    return cv2.Canny(g, high * 0.5 / 4.0, high / 4.0)


def calculate_edge_mismatch(original_gray: np.ndarray, current_gray: np.ndarray, settings: dict):
    """
    Returns (global_mismatch_pct, grid_percentages, changed_cells, valid_cells).
    mismatch = % of ORIGINAL edge pixels with no current edge nearby.
    """
    tol = max(1, int(get_setting(settings, "edge_match_tolerance_px", default=2, cast=int)))
    cell_threshold = float(get_setting(settings, "edge_mismatch_cell_percentage", default=30.0, cast=float))
    min_edges = int(get_setting(settings, "edge_min_cell_edge_pixels", default=40, cast=int))
    grid = max(2, int(get_setting(settings, "camera_angle_grid_size", default=4, cast=int)))

    edges_a = extract_edges(original_gray, settings)
    edges_b = extract_edges(current_gray, settings)

    kernel = np.ones((2 * tol + 1, 2 * tol + 1), dtype=np.uint8)
    near_b = cv2.dilate(edges_b, kernel)
    matched = cv2.bitwise_and(edges_a, near_b)

    h, w = edges_a.shape[:2]
    total_edges = int(np.count_nonzero(edges_a))
    global_mismatch = 100.0 * (1.0 - np.count_nonzero(matched) / total_edges) if total_edges > 0 else 0.0

    grid_percentages: Dict[str, float] = {}
    changed_cells: List[str] = []
    valid_cells = 0
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * h // grid, (gy + 1) * h // grid
            x0, x1 = gx * w // grid, (gx + 1) * w // grid
            n_edges = int(np.count_nonzero(edges_a[y0:y1, x0:x1]))
            name = f"r{gy}c{gx}"
            if n_edges < min_edges:
                grid_percentages[name] = -1.0  # not enough texture to judge
                continue
            valid_cells += 1
            n_match = int(np.count_nonzero(matched[y0:y1, x0:x1]))
            pct = 100.0 * (1.0 - n_match / n_edges)
            grid_percentages[name] = round(pct, 2)
            if pct >= cell_threshold:
                changed_cells.append(name)

    return float(global_mismatch), grid_percentages, changed_cells, valid_cells


# ================================================================
# REGISTRATION (shift / rotation / zoom) - catches MINOR camera moves
# ================================================================

def gradient_map(gray: np.ndarray) -> np.ndarray:
    """Lighting-independent structure map (contrast-normalised, log-compressed gradients)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = cv2.GaussianBlur(clahe.apply(gray), (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.log1p(cv2.magnitude(gx, gy))
    return cv2.GaussianBlur(m, (0, 0), 2.0)


def template_registration(original_gray: np.ndarray, current_gray: np.ndarray, settings: dict) -> Dict[str, Any]:
    """
    Translation search that needs NO keypoints: the centre of the original
    structure map is slid over the current one (normalised cross-correlation).
    Works on repetitive / low-texture shelves and for LARGE pans.
    Returns shift (pixels @ working size), best score, score at zero shift.
    """
    h, w = original_gray.shape[:2]
    work_w = int(get_setting(settings, "template_work_width", default=640, cast=int))
    s = min(1.0, work_w / float(w))
    if s < 1.0:
        a = cv2.resize(original_gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        b = cv2.resize(current_gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        a, b = original_gray, current_gray

    ga, gb = gradient_map(a), gradient_map(b)
    sh, sw = ga.shape[:2]

    margins = get_setting(settings, "template_search_margins_percent", default=[20, 35])
    best: Optional[Dict[str, Any]] = None

    for margin in margins:
        mx = max(2, int(sw * float(margin) / 100.0))
        my = max(2, int(sh * float(margin) / 100.0))
        tmpl = ga[my:sh - my, mx:sw - mx]
        if tmpl.shape[0] < 16 or tmpl.shape[1] < 16 or float(tmpl.std()) < 1e-4:
            continue
        res = np.nan_to_num(cv2.matchTemplate(gb, tmpl, cv2.TM_CCOEFF_NORMED), nan=-1.0)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        px, py = max_loc

        dx = dy = 0.0
        if 0 < px < res.shape[1] - 1:
            l, c, r = res[py, px - 1], res[py, px], res[py, px + 1]
            d = l - 2 * c + r
            dx = float(np.clip(0.5 * (l - r) / d, -0.5, 0.5)) if abs(d) > 1e-9 else 0.0
        if 0 < py < res.shape[0] - 1:
            u, c, dn = res[py - 1, px], res[py, px], res[py + 1, px]
            d = u - 2 * c + dn
            dy = float(np.clip(0.5 * (u - dn) / d, -0.5, 0.5)) if abs(d) > 1e-9 else 0.0

        cand = {
            "shift_x": (px + dx - mx) / s,
            "shift_y": (py + dy - my) / s,
            "score": float(max_val),
            "score_zero": float(res[my, mx]),
            "margin": margin,
            "res": res, "mx": mx, "my": my, "s": s,
        }
        if best is None or cand["score"] > best["score"] + 0.02:
            best = cand
        if best["score"] >= 0.6:
            break

    if best is None:
        return {"shift_x": 0.0, "shift_y": 0.0, "score": 0.0, "score_zero": 0.0, "res": None}
    return best


def _fit_transform(kp_a, kp_b, good, w, h, min_inliers, ransac_thr):
    if len(good) < min_inliers:
        return None
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr)
    if matrix is None or mask is None:
        return None
    inliers = int(mask.sum())
    if inliers < min_inliers or inliers < 0.25 * len(good):
        return None
    cx, cy = w / 2.0, h / 2.0
    moved = matrix @ np.array([cx, cy, 1.0])
    return {
        "shift_x": float(moved[0] - cx),
        "shift_y": float(moved[1] - cy),
        "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        "scale_pct": (float(np.hypot(matrix[0, 0], matrix[1, 0])) - 1.0) * 100.0,
        "inliers": inliers,
    }


def feature_registration(original_gray: np.ndarray, current_gray: np.ndarray, settings: dict) -> Optional[Dict[str, Any]]:
    """SIFT first (best on repetitive scenes), ORB as backup. Relaxed ratio test, 4px RANSAC."""
    h, w = original_gray.shape[:2]
    min_inliers = int(get_setting(settings, "min_registration_inliers", default=10, cast=int))
    n_features = int(get_setting(settings, "orb_features", default=4000, cast=int))
    ratio = float(get_setting(settings, "feature_ratio_test", default=0.8, cast=float))
    ransac_thr = float(get_setting(settings, "ransac_reproj_threshold", default=4.0, cast=float))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    a, b = clahe.apply(original_gray), clahe.apply(current_gray)

    detectors = []
    if hasattr(cv2, "SIFT_create"):
        detectors.append(("sift", cv2.SIFT_create(nfeatures=n_features), cv2.NORM_L2, ratio))
    detectors.append(("orb", cv2.ORB_create(nfeatures=n_features), cv2.NORM_HAMMING, min(0.9, ratio + 0.05)))

    for name, det, norm, r in detectors:
        kp_a, des_a = det.detectAndCompute(a, None)
        kp_b, des_b = det.detectAndCompute(b, None)
        if des_a is None or des_b is None or len(kp_a) < min_inliers or len(kp_b) < min_inliers:
            continue
        knn = cv2.BFMatcher(norm).knnMatch(des_a, des_b, k=2)
        good = [pr[0] for pr in knn if len(pr) == 2 and pr[0].distance < r * pr[1].distance]
        fit = _fit_transform(kp_a, kp_b, good, w, h, min_inliers, ransac_thr)
        if fit is not None:
            fit["method"] = name
            return fit
    return None


def estimate_registration(original_gray: np.ndarray, current_gray: np.ndarray, settings: dict):
    """
    Returns (registration_or_None, template_info).
    Cascade: feature registration (gives shift+rotation+zoom) cross-checked
    by keypoint-free template matching (gives large/low-texture shifts).
    """
    tm_min = float(get_setting(settings, "template_min_score", default=0.30, cast=float))
    cross_margin = float(get_setting(settings, "template_override_margin", default=0.15, cast=float))

    tm = template_registration(original_gray, current_gray, settings)
    feat = feature_registration(original_gray, current_gray, settings)

    tm_info = {"score": round(tm["score"], 3), "score_zero": round(tm["score_zero"], 3),
               "shift_x": round(tm["shift_x"], 2), "shift_y": round(tm["shift_y"], 2)}

    def from_template(tag: str) -> Dict[str, Any]:
        return {"method": tag, "shift_x": tm["shift_x"], "shift_y": tm["shift_y"],
                "rotation_deg": 0.0, "scale_pct": 0.0, "inliers": 0}

    if feat is not None:
        # cross-check: does the template search find a MUCH better alignment than the features claim?
        res = tm.get("res")
        if res is not None and tm["score"] >= tm_min:
            ix = int(round(feat["shift_x"] * tm["s"])) + tm["mx"]
            iy = int(round(feat["shift_y"] * tm["s"])) + tm["my"]
            score_at_feat = float(res[iy, ix]) if (0 <= iy < res.shape[0] and 0 <= ix < res.shape[1]) else -1.0
            if tm["score"] - score_at_feat > cross_margin:
                return from_template("template_ncc_override"), tm_info
        return feat, tm_info

    if tm["res"] is not None and tm["score"] >= tm_min:
        return from_template("template_ncc"), tm_info

    return None, tm_info


def classify_registration(reg: Dict[str, Any], frame_w: int, frame_h: int, settings: dict) -> Tuple[str, List[str]]:
    """
    Returns (primary_movement, all_movements). 'none' when below thresholds.
    Labels describe the CAMERA movement: if the scene shifts right in
    the image, the camera turned left.
    """
    shift_pct = float(get_setting(settings, "shift_threshold_percent", default=0.25, cast=float))
    rot_thr = float(get_setting(settings, "rotation_threshold_degrees", default=0.3, cast=float))
    scale_thr = float(get_setting(settings, "scale_threshold_percent", default=0.8, cast=float))
    invert = bool(get_setting(settings, "invert_direction_labels", default=False))

    thr_x = frame_w * shift_pct / 100.0
    thr_y = frame_h * shift_pct / 100.0

    candidates: List[Tuple[float, str]] = []

    sx, sy = reg["shift_x"], reg["shift_y"]
    if abs(sx) >= thr_x:
        label = "left" if sx > 0 else "right"
        if invert:
            label = "right" if label == "left" else "left"
        candidates.append((abs(sx) / thr_x, label))
    if abs(sy) >= thr_y:
        label = "up" if sy > 0 else "down"
        if invert:
            label = "down" if label == "up" else "up"
        candidates.append((abs(sy) / thr_y, label))
    if abs(reg["rotation_deg"]) >= rot_thr:
        candidates.append((abs(reg["rotation_deg"]) / rot_thr, "tilt"))
    if abs(reg["scale_pct"]) >= scale_thr:
        candidates.append((abs(reg["scale_pct"]) / scale_thr, "zoom"))

    if not candidates:
        return "none", []

    candidates.sort(reverse=True)
    return candidates[0][1], [name for _, name in candidates]


# ================================================================
# CAMERA ANGLE / VIEW DETECTION (whole frame)
# ================================================================

def detect_camera_angle_change(original: np.ndarray, current: np.ndarray, settings: dict) -> Dict[str, Any]:
    original, current = resize_same_size(original, current, settings)
    height, width = original.shape[:2]

    original_gray = to_gray(original)
    current_gray = to_gray(current)

    black_info = detect_black_frame(current_gray, settings)
    blur_info = detect_blur_event(original_gray, current_gray, settings)
    registration, tm_info = estimate_registration(original_gray, current_gray, settings)

    mismatch_pct, grid_percentages, changed_cells, valid_cells = calculate_edge_mismatch(
        original_gray, current_gray, settings
    )  # diagnostic only

    similarity_threshold = float(get_setting(settings, "scene_similarity_threshold", default=0.5, cast=float))

    reg_movement, reg_all = ("none", [])
    if registration is not None:
        reg_movement, reg_all = classify_registration(registration, width, height, settings)

    # ---- decision ----
    if black_info["is_black"]:
        movement_type, reason = "black", "camera_black_frame"
    elif reg_movement != "none":
        movement_type, reason = reg_movement, f"camera_moved_{'+'.join(reg_all)}"
    elif blur_info["is_blurred"]:
        movement_type, reason = "blur", "camera_blurry_or_defocused"
    elif registration is not None:
        movement_type, reason = "none", "stable_camera"
    elif tm_info["score_zero"] < similarity_threshold:
        # nothing could be aligned AND the scene no longer resembles the original
        movement_type, reason = "camera_change", "view_changed_cannot_align_to_original"
    else:
        movement_type, reason = "none", "stable_camera_low_texture"

    camera_angle_changed = movement_type != "none"
    status = "CHANGE" if camera_angle_changed else "NO_CHANGE"

    logger.info(
        "[Fixture Angle] status=%s | movement=%s | reason=%s | template(score=%.2f zero=%.2f shift=(%.1f,%.1f)) | "
        "sharp(orig=%.3f,cur=%.3f) | brightness=%.1f | edge_mismatch=%.1f%% (diagnostic)",
        status, movement_type, reason, tm_info["score"], tm_info["score_zero"], tm_info["shift_x"], tm_info["shift_y"],
        blur_info["original_sharpness"], blur_info["current_sharpness"], black_info["mean"], mismatch_pct,
    )
    if registration is not None:
        logger.info(
            "[Fixture Registration] method=%s | shift=(%.2f, %.2f)px | rotation=%.3fdeg | scale=%.3f%% | inliers=%d",
            registration["method"], registration["shift_x"], registration["shift_y"],
            registration["rotation_deg"], registration["scale_pct"], registration["inliers"],
        )
    else:
        logger.info("[Fixture Registration] could not align current to original")

    return {
        "status": status,
        "camera_angle_changed": camera_angle_changed,
        "movement_type": movement_type,
        "reason": reason,
        "edge_mismatch_percentage": round(mismatch_pct, 2),
        "changed_cell_count": len(changed_cells),
        "grid_percentages": grid_percentages,
        "registration": registration,
        "template": tm_info,
        "blur": blur_info,
        "black": black_info,
    }


def detect_with_confirmation(original: np.ndarray, frames: List[np.ndarray], settings: dict) -> Dict[str, Any]:
    """Run detection on every captured frame; report a change only if the majority agree."""
    results = []
    for index, frame in enumerate(frames, start=1):
        logger.info("[Fixture] Analysing frame %d/%d", index, len(frames))
        results.append(detect_camera_angle_change(original, frame, settings))

    votes = sum(1 for r in results if r["camera_angle_changed"])
    needed = len(results) // 2 + 1

    if votes >= needed:
        kinds = [r["movement_type"] for r in results if r["camera_angle_changed"]]
        best = max(set(kinds), key=kinds.count)
        chosen = next(r for r in results if r["movement_type"] == best)
    else:
        chosen = next((r for r in results if not r["camera_angle_changed"]), results[0])
        chosen = dict(chosen)
        chosen["camera_angle_changed"] = False
        chosen["movement_type"] = "none"
        chosen["status"] = "NO_CHANGE"
        chosen["reason"] = "stable_camera" if votes == 0 else "change_not_confirmed_across_frames"

    chosen = dict(chosen)
    chosen["votes"] = f"{votes}/{len(results)}"
    logger.info("[Fixture] Confirmation votes: %s changed (need %d) -> %s", chosen["votes"], needed, chosen["movement_type"])
    return chosen


# ================================================================
# BASELINE / CURRENT
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

    baseline = camera_dir / f"original_{timestamp_for_filename()}.jpg"
    if not cv2.imwrite(str(baseline), frame):
        raise RuntimeError(f"Unable to write original image: {baseline}")

    logger.info("[Fixture] ORIGINAL CREATED | %s", baseline)
    return frame.copy(), True, baseline


def save_fresh_current(camera_dir: Path, frame: np.ndarray) -> Path:
    """Run-once mode: every run stores a fresh current image (old ones removed)."""
    for old in camera_dir.glob("current_*.jpg"):
        try:
            old.unlink()
        except OSError:
            logger.exception("[Fixture] Could not delete old current: %s", old)

    current_path = camera_dir / f"current_{timestamp_for_filename()}.jpg"
    if not cv2.imwrite(str(current_path), frame):
        raise RuntimeError(f"Unable to write current image: {current_path}")

    logger.info("[Fixture] CURRENT CREATED | %s", current_path)
    return current_path


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


def read_frame(rtsp_url: str, warmup_frames: int = 5) -> Optional[np.ndarray]:
    """Open stream, skip warm-up/buffered frames, return the latest one."""
    cap = open_rtsp(rtsp_url)
    if cap is None:
        logger.error("[Fixture] RTSP open failed: %s", rtsp_url)
        return None
    try:
        for _ in range(max(0, warmup_frames)):
            cap.grab()
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("[Fixture] RTSP frame read failed")
            return None
        return frame
    finally:
        cap.release()


def capture_burst(rtsp_url: str, count: int, interval: float, warmup_frames: int) -> List[np.ndarray]:
    """Capture `count` frames `interval` seconds apart on one connection."""
    cap = open_rtsp(rtsp_url)
    if cap is None:
        logger.error("[Fixture] RTSP open failed: %s", rtsp_url)
        return []
    frames: List[np.ndarray] = []
    try:
        for i in range(count):
            if i > 0:
                time.sleep(interval)
            for _ in range(max(0, warmup_frames)):   # flush stale buffered frames
                cap.grab()
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
            else:
                logger.warning("[Fixture] Frame %d/%d read failed", i + 1, count)
    finally:
        cap.release()
    return frames


def capture_with_retries(rtsp_url: str, attempts: int, warmup_frames: int, delay: float,
                         count: int = 3, interval: float = 1.0) -> List[np.ndarray]:
    for attempt in range(1, attempts + 1):
        frames = capture_burst(rtsp_url, count, interval, warmup_frames)
        if frames:
            return frames
        logger.warning("[Fixture] Capture attempt %d/%d failed", attempt, attempts)
        if attempt < attempts:
            time.sleep(delay)
    return []


# ================================================================
# DAILY MARKER
# ================================================================

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
    endpoint = config.get("fixture_missing", {}).get("api_endpoint", "/storescript/api/fixture-missing-event")
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

            logger.info("[Fixture] API RESPONSE | HTTP=%s | body=%s", response.status_code, response.text[:1000])

            if response.status_code in (200, 201):
                return True
            if 400 <= response.status_code < 500:
                logger.error("[Fixture] Non-retryable API error | HTTP=%s", response.status_code)
                return False

        except Exception as exc:
            logger.exception("[Fixture] API request failed | attempt=%d/%d | error=%s", attempt, max_retries, exc)
        finally:
            if original_file is not None:
                original_file.close()
            if current_file is not None:
                current_file.close()

        if attempt < max_retries:
            time.sleep(retry_delay)

    return False


# ================================================================
# CAMERA PROCESSOR (single check, then return)
# ================================================================

def process_camera(store_name: str, store: dict, camera: dict, config: dict) -> None:
    camera_no = int(camera["id"])
    camera_category_name = str(camera.get("category_name", camera.get("name", f"camera_{camera_no}")))
    camera_name = f"camera_{camera_no}"
    rtsp_url = camera.get("rtsp_url")

    def finish(outcome: str, **extra) -> None:
        record_result(store=store_name, camera=camera_no, category=camera_category_name, outcome=outcome, **extra)

    if not rtsp_url:
        logger.error("[Fixture] Missing RTSP URL | store=%s | camera=%s", store_name, camera_no)
        finish("ERROR_NO_RTSP_URL")
        return

    global_fixture = config.get("fixture_missing", {})
    camera_fixture = camera.get("fixture_missing", {})

    if not global_fixture.get("enabled", True) or not camera_fixture.get("enabled", True):
        logger.info("[Fixture] Disabled | camera=%s", camera_no)
        finish("DISABLED")
        return

    # camera settings override global settings
    merged_settings = {**global_fixture, **{k: v for k, v in camera_fixture.items() if k != "regions"}}

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

    reconnect_delay = float(global_fixture.get("rtsp_reconnect_delay_seconds", 5))
    capture_attempts = int(merged_settings.get("capture_attempts", 5))
    warmup_frames = int(merged_settings.get("warmup_frames", 5))
    force_recheck = bool(merged_settings.get("force_recheck", False))
    confirmation_frames = max(1, int(merged_settings.get("confirmation_frames", 3)))
    confirmation_interval = float(merged_settings.get("confirmation_interval_seconds", 1.0))
    api_url = build_api_url(config)

    now = datetime.now()
    date_compact = now.strftime("%Y%m%d")
    date_text = now.strftime("%Y-%m-%d")

    logger.info(
        "[Fixture] Camera started | store=%s | camera=%s | category=%s",
        store.get("store_id"), camera_no, camera_category_name,
    )

    if already_checked_today(camera_dir, date_compact) and not force_recheck:
        logger.info("[Fixture] Already checked today | camera=%s (set force_recheck=true to repeat)", camera_no)
        finish("SKIPPED_ALREADY_CHECKED")
        return

    lock = get_camera_lock(camera_no, date_compact)
    if not lock.acquire(blocking=False):
        logger.warning("[Fixture] Camera already being processed | camera=%s", camera_no)
        finish("SKIPPED_LOCKED")
        return

    try:
        frames = capture_with_retries(
            rtsp_url, capture_attempts, warmup_frames, reconnect_delay,
            count=confirmation_frames, interval=confirmation_interval,
        )
        if not frames:
            logger.error("[Fixture] Could not capture a frame | camera=%s", camera_no)
            finish("ERROR_CAPTURE_FAILED")
            return

        frame = frames[0]
        original, original_created, original_path = load_or_create_baseline(camera_dir, frame)

        if original_created:
            logger.info(
                "[Fixture] Baseline created | camera=%s | %s | nothing to compare today; next run will compare",
                camera_no, original_path.name,
            )
            finish("BASELINE_CREATED")
            return

        current_path = save_fresh_current(camera_dir, frame)
        current = frame

        result = detect_with_confirmation(original, frames, merged_settings)

        if bool(merged_settings.get("save_debug_images", True)):
            try:
                def _small(img):
                    return cv2.resize(img, (480, int(img.shape[0] * 480 / img.shape[1])))
                o_s, c_s = _small(original), _small(current)
                banner = np.zeros((40, o_s.shape[1] * 2, 3), np.uint8)
                cv2.putText(
                    banner, f"cam{camera_no} {result['movement_type']} | {result['reason']}"[:90],
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                )
                cv2.imwrite(str(camera_dir / "debug_latest.jpg"), np.vstack([banner, np.hstack([o_s, c_s])]))
            except Exception:
                logger.exception("[Fixture] Debug image failed")

        if result["camera_angle_changed"]:
            status, alert = "YES", "yes"
            movement_type = result["movement_type"]

            logger.warning(
                "[Fixture] CAMERA ANGLE / VIEW CHANGE DETECTED | camera=%s | movement=%s | reason=%s",
                camera_no, movement_type, result["reason"],
            )

            alert_snapshot = snapshot_dir / f"camera_{movement_type}_{date_compact}_{now.strftime('%H%M%S_%f')}.jpg"
            try:
                if cv2.imwrite(str(alert_snapshot), current):
                    logger.info("[Fixture] SNAPSHOT SAVED | %s", alert_snapshot)
                else:
                    logger.error("[Fixture] SNAPSHOT WRITE FAILED | %s", alert_snapshot)
            except Exception:
                logger.exception("[Fixture] Snapshot save failed")
        else:
            status, alert = "NO", "no"
            movement_type = "none"
            logger.info("[Fixture] CAMERA STABLE | camera=%s | reason=%s", camera_no, result["reason"])

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
            finish("CHECKED", status=status, alert=alert, movement=movement_type, reason=result["reason"])
        else:
            logger.error("[Fixture] DAILY CHECK NOT MARKED COMPLETE | camera=%s | API delivery failed", camera_no)
            finish("ERROR_API_FAILED", status=status, alert=alert, movement=movement_type, reason=result["reason"])

    except Exception:
        logger.exception("[Fixture] Processing failed | camera=%s", camera_no)
        finish("ERROR_EXCEPTION")
    finally:
        lock.release()


# ================================================================
# MAIN
# ================================================================

def print_summary() -> int:
    logger.info("=" * 78)
    logger.info("FIXTURE CAMERA CHECK SUMMARY  (%d camera(s))", len(RESULTS))
    logger.info("=" * 78)

    failures = 0
    for r in sorted(RESULTS, key=lambda x: (str(x["store"]), x["camera"])):
        line = f"store={r['store']} | camera={r['camera']} | {r['category']} | {r['outcome']}"
        if r.get("status"):
            line += f" | status={r['status']} | alert={r['alert']} | movement={r['movement']} | reason={r['reason']}"
        logger.info(line)
        if r["outcome"].startswith("ERROR"):
            failures += 1

    logger.info("=" * 78)
    return failures


def wait_for_check_time(config: dict) -> None:
    """
    Run the check at/after daily_check_time (default 13:00).
      - already past the time  -> run immediately
      - before the time        -> wait until it (wait_for_check_time=true, default)
                                  or exit (wait_for_check_time=false)
    Bypass for testing:  python fixture_missing.py --now   (or "ignore_check_time": true)
    """
    fx = config.get("fixture_missing", {})
    if "--now" in sys.argv or bool(fx.get("ignore_check_time", False)):
        logger.info("[Fixture] Check-time gate bypassed (--now / ignore_check_time)")
        return

    check_time = str(fx.get("daily_check_time", "11:00"))
    try:
        hour, minute = map(int, check_time.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except ValueError:
        logger.error("[Fixture] Invalid daily_check_time=%r; using 11:00", check_time)
        hour, minute = 13, 0

    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        logger.info("[Fixture] It is %s (>= %02d:%02d) - running the daily check now", now.strftime("%H:%M:%S"), hour, minute)
        return

    if not bool(fx.get("wait_for_check_time", True)):
        logger.info("[Fixture] It is %s, before %02d:%02d - exiting without checking", now.strftime("%H:%M:%S"), hour, minute)
        sys.exit(0)

    logger.info("[Fixture] It is %s, before %02d:%02d - waiting until then", now.strftime("%H:%M:%S"), hour, minute)
    while datetime.now() < target:
        time.sleep(min(30.0, max(1.0, (target - datetime.now()).total_seconds())))
    logger.info("[Fixture] Check time reached - starting")


def main() -> int:
    config = load_config()
    wait_for_check_time(config)
    stores = config.get("stores", {})

    if not stores:
        raise RuntimeError("No stores configured")

    threads: List[threading.Thread] = []

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
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("[Fixture] Stopped by user")
        return 130

    failures = print_summary()
    logger.info("[Fixture] All cameras checked. Exiting.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())