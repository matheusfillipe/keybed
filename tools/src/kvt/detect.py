"""Classical keybed quad detector: geometry only, zero training."""

from dataclasses import dataclass

import cv2
import numpy as np

_TARGET_WIDTH = 320
_STRIP_SIZE = (416, 64)
_STRIP_DST = np.array([[0.0, 0.0], [415.0, 0.0], [415.0, 63.0], [0.0, 63.0]], dtype=np.float32)
_MIN_CONFIDENCE = 0.35
_MIN_AREA_FRACTION = 0.005
_MAX_AREA_FRACTION = 0.4
_MIN_ASPECT = 2.2
_MIN_FILL = 0.5
_MAX_CANDIDATES = 5
_WHITE_ZONE = slice(4, 38)
_PROFILE_ZONE = slice(0, 40)
_WHITENESS_MIN = 100.0
_MIN_PEAKS = 14
_MAX_PEAKS = 40
_SPACING_REGULARITY_MAX = 0.6
_MIN_PEAK_WIDTH = 2.0
_MAX_PEAK_WIDTH = 12.0
_REFINE_STEP = 2.0
_REFINE_FRACTION = 0.15
_EDGE_BAND = 2


@dataclass
class Detection:
    quad_px: np.ndarray
    confidence: float


def find_keybed(image_bgr: np.ndarray) -> Detection | None:
    height, width = image_bgr.shape[:2]
    if height == 0 or width == 0:
        return None
    scale = _TARGET_WIDTH / width
    small_height = max(1, round(height * scale))
    small = cv2.resize(image_bgr, (_TARGET_WIDTH, small_height), interpolation=cv2.INTER_AREA)
    mask = _bright_mask(small)
    best: Detection | None = None
    for _, quad in _candidate_quads(mask, scale, small_height):
        detection = _verify(image_bgr, quad)
        if detection is not None and (best is None or detection.confidence > best.confidence):
            best = detection
    if best is None or best.confidence <= _MIN_CONFIDENCE:
        return None
    return _refine(image_bgr, best)


def _bright_mask(small: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _candidate_quads(
    mask: np.ndarray, scale: float, small_height: int
) -> list[tuple[float, np.ndarray]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = float(_TARGET_WIDTH * small_height)
    scored: list[tuple[float, np.ndarray]] = []
    for index in range(1, int(count)):
        area = float(stats[index, cv2.CC_STAT_AREA])
        box_width = float(stats[index, cv2.CC_STAT_WIDTH])
        box_height = float(stats[index, cv2.CC_STAT_HEIGHT])
        fraction = area / total
        if not _MIN_AREA_FRACTION <= fraction <= _MAX_AREA_FRACTION:
            continue
        if min(box_width, box_height) <= 0.0:
            continue
        aspect = max(box_width, box_height) / min(box_width, box_height)
        if aspect < _MIN_ASPECT:
            continue
        fill = area / (box_width * box_height)
        if fill < _MIN_FILL:
            continue
        points = np.column_stack(np.nonzero(labels == index)).astype(np.float32)
        points = np.ascontiguousarray(points[:, ::-1])
        box = np.asarray(cv2.boxPoints(cv2.minAreaRect(points)), dtype=np.float32)
        scored.append((area * fill, _order_quad(box) / scale))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:_MAX_CANDIDATES]


def _order_quad(box: np.ndarray) -> np.ndarray:
    total = box.sum(axis=1)
    diagonal = box[:, 1] - box[:, 0]
    return np.array(
        [
            box[int(np.argmin(total))],
            box[int(np.argmin(diagonal))],
            box[int(np.argmax(total))],
            box[int(np.argmax(diagonal))],
        ],
        dtype=np.float64,
    )


def _verify(image_bgr: np.ndarray, quad_px: np.ndarray) -> Detection | None:
    strip = cv2.warpPerspective(
        image_bgr,
        cv2.getPerspectiveTransform(quad_px.astype(np.float32), _STRIP_DST),
        _STRIP_SIZE,
    )
    confidence = _strip_confidence(strip)
    if confidence is None:
        return None
    return Detection(quad_px=quad_px.astype(np.float64), confidence=confidence)


def _strip_confidence(strip_bgr: np.ndarray) -> float | None:
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    whiteness = float(gray[_WHITE_ZONE, :].mean())
    if whiteness <= _WHITENESS_MIN:
        return None
    smoothed = _smooth(gray[_PROFILE_ZONE, :].mean(axis=0))
    threshold = float(smoothed.mean() - 0.25 * smoothed.std())
    peaks = _dark_peaks(smoothed, threshold)
    widths = [_peak_width(smoothed, threshold, peak) for peak in peaks]
    earned = 0
    if _MIN_PEAKS <= len(peaks) <= _MAX_PEAKS:
        earned += 1
    if len(peaks) >= 3:
        spacings = np.diff(np.asarray(peaks, dtype=np.float64))
        mean_spacing = float(spacings.mean())
        if mean_spacing > 0.0 and float(spacings.std()) / mean_spacing < _SPACING_REGULARITY_MAX:
            earned += 1
    median_width = float(np.median(widths)) if widths else 0.0
    if _MIN_PEAK_WIDTH <= median_width <= _MAX_PEAK_WIDTH:
        earned += 1
    margin = min(1.0, (whiteness - _WHITENESS_MIN) / 100.0)
    confidence = (earned / 3.0) * 0.5 + margin * 0.5
    return float(min(1.0, max(0.0, confidence)))


def _smooth(profile: np.ndarray) -> np.ndarray:
    kernel = np.full(3, 1.0 / 3.0)
    return np.convolve(np.pad(profile, 1, mode="edge"), kernel, mode="valid")


def _dark_peaks(profile: np.ndarray, threshold: float) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(profile) - 1):
        if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1] and profile[i] < threshold:
            peaks.append(i)
    return peaks


def _peak_width(profile: np.ndarray, threshold: float, peak: int) -> float:
    left = peak
    while left > 0 and profile[left - 1] < threshold:
        left -= 1
    right = peak
    while right < len(profile) - 1 and profile[right + 1] < threshold:
        right += 1
    return float(right - left + 1)


def _refine(image_bgr: np.ndarray, detection: Detection) -> Detection:
    quad = detection.quad_px
    left = _walk_edge(image_bgr, quad, side=0)
    right = _walk_edge(image_bgr, quad, side=1)
    result = _verify(image_bgr, _shift_edges(quad, left, right))
    if result is not None and result.confidence >= detection.confidence:
        return result
    return detection


def _walk_edge(image_bgr: np.ndarray, quad: np.ndarray, side: int) -> float:
    length = float(np.linalg.norm(quad[1] - quad[0]))
    limit = length * _REFINE_FRACTION
    extension = 0.0
    while extension + _REFINE_STEP <= limit and _edge_bright(
        image_bgr, quad, side, extension + _REFINE_STEP
    ):
        extension += _REFINE_STEP
    if not _edge_bright(image_bgr, quad, side, extension):
        while extension >= -limit and not _edge_bright(
            image_bgr, quad, side, extension - _REFINE_STEP
        ):
            extension -= _REFINE_STEP
    return extension


def _edge_bright(image_bgr: np.ndarray, quad: np.ndarray, side: int, extension: float) -> bool:
    shifted = _shift_edges(quad, extension if side == 0 else 0.0, extension if side == 1 else 0.0)
    strip = cv2.warpPerspective(
        image_bgr,
        cv2.getPerspectiveTransform(shifted.astype(np.float32), _STRIP_DST),
        _STRIP_SIZE,
    )
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float64)
    columns = gray[_WHITE_ZONE, :].mean(axis=0)
    band = columns[:_EDGE_BAND] if side == 0 else columns[-_EDGE_BAND:]
    return float(band.mean()) > _WHITENESS_MIN


def _shift_edges(quad: np.ndarray, left: float, right: float) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = quad
    top_length = float(np.linalg.norm(top_right - top_left))
    if top_length == 0.0:
        return quad.copy()
    unit_top = (top_right - top_left) / top_length
    bottom_length = float(np.linalg.norm(bottom_right - bottom_left))
    unit_bottom = (bottom_right - bottom_left) / bottom_length if bottom_length > 0.0 else unit_top
    return np.array(
        [
            top_left - unit_top * left,
            top_right + unit_top * right,
            bottom_right + unit_bottom * right,
            bottom_left - unit_bottom * left,
        ],
        dtype=np.float64,
    )
