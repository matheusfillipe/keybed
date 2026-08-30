"""Geometric keybed corner refinement by canonical template correlation."""

import cv2
import numpy as np

_WHITE_COUNT = 52.0
_BLACK_OFFSETS = (0.60, 1.75, 3.60, 4.63, 5.66)
_BLACK_WIDTH = 0.58
_BLACK_DEPTH_FRACTION = 0.62
_WHITE_VALUE = 1.0
_BLACK_VALUE = 0.2
_SEAM_VALUE = 0.5

_STRIP_WIDTH = 520
_STRIP_HEIGHT = 64
_STEPS = (16, 12, 8, 6, 4, 3, 2, 1)


def _build_template(strip_width: int, strip_height: int) -> np.ndarray:
    unit = strip_width / _WHITE_COUNT
    bar_height = round(strip_height * _BLACK_DEPTH_FRACTION)
    template = np.zeros((strip_height, strip_width), dtype=np.float64)
    for key in range(int(_WHITE_COUNT)):
        x0 = round(key * unit)
        x1 = round((key + 1) * unit)
        template[:, x0:x1] = _WHITE_VALUE
        if key > 0:
            template[:, x0 - 1 : x0] = _SEAM_VALUE
    for octave in range(8):
        for offset in _BLACK_OFFSETS:
            u0 = 7.0 * octave + offset
            u1 = u0 + _BLACK_WIDTH
            if u1 > _WHITE_COUNT:
                continue
            template[:bar_height, round(u0 * unit) : round(u1 * unit)] = _BLACK_VALUE
    return template


def _strip_destination(strip_width: int, strip_height: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(strip_width - 1), 0.0],
            [float(strip_width - 1), float(strip_height - 1)],
            [0.0, float(strip_height - 1)],
        ],
        dtype=np.float32,
    )


_TEMPLATE = _build_template(_STRIP_WIDTH, _STRIP_HEIGHT)


def _score(
    image_bgr: np.ndarray, quad_px: np.ndarray, strip_width: int, strip_height: int
) -> float:
    if (strip_width, strip_height) == (_STRIP_WIDTH, _STRIP_HEIGHT):
        template = _TEMPLATE
    else:
        template = _build_template(strip_width, strip_height)
    matrix = cv2.getPerspectiveTransform(
        quad_px.astype(np.float32), _strip_destination(strip_width, strip_height)
    )
    strip = cv2.warpPerspective(image_bgr, matrix, (strip_width, strip_height))
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    warped = gray - gray.mean()
    reference = template - template.mean()
    norm = float(np.sqrt((warped * warped).sum() * (reference * reference).sum()))
    if norm <= 0.0:
        return 0.0
    return float((warped * reference).sum() / norm)


def refine_quad(
    image_bgr: np.ndarray,
    quad_px: np.ndarray,
    strip_width: int = _STRIP_WIDTH,
    strip_height: int = _STRIP_HEIGHT,
) -> np.ndarray:
    params = quad_px.astype(np.float64).reshape(-1)
    initial = _score(image_bgr, params.reshape(4, 2), strip_width, strip_height)
    best = initial
    for step in _STEPS:
        for index in range(8):
            for direction in (step, -step):
                trial = params.copy()
                trial[index] += direction
                score = _score(image_bgr, trial.reshape(4, 2), strip_width, strip_height)
                if score > best:
                    params = trial
                    best = score
                    break
    if best < initial:
        return quad_px.astype(np.float64).reshape(4, 2)
    return params.reshape(4, 2)
