import cv2
import numpy as np

from kvt.detect import find_keybed
from kvt.homography import find_homography

BLACK_OFFSETS = (0.60, 1.75, 3.60, 4.63, 5.66)
BLACK_WIDTH = 0.58
WHITE_COUNT = 52
KEYBED_WIDTH = 52.0
KEYBED_HEIGHT = 8.0
RENDER_PX_PER_UNIT = 16
CANVAS_SIZE = (640, 400)
TRUE_QUAD = np.array(
    [[55.0, 108.0], [595.0, 100.0], [600.0, 258.0], [50.0, 268.0]], dtype=np.float64
)
BACKGROUND_VALUE = 25.0


def _render_sheet() -> np.ndarray:
    height = int(KEYBED_HEIGHT * RENDER_PX_PER_UNIT)
    width = int(KEYBED_WIDTH * RENDER_PX_PER_UNIT)
    sheet = np.full((height, width, 3), 235, dtype=np.uint8)
    bar_height = int(height * 0.6)
    for octave in range(8):
        for offset in BLACK_OFFSETS:
            u0 = 7 * octave + offset
            u1 = u0 + BLACK_WIDTH
            if u1 > KEYBED_WIDTH:
                continue
            x0 = round(u0 / KEYBED_WIDTH * width)
            x1 = round(u1 / KEYBED_WIDTH * width)
            sheet[0:bar_height, x0:x1] = 15
    return sheet


def _render_keybed(seed: int) -> np.ndarray:
    sheet = _render_sheet()
    src = np.array(
        [
            [0.0, 0.0],
            [float(sheet.shape[1]), 0.0],
            [float(sheet.shape[1]), float(sheet.shape[0])],
            [0.0, float(sheet.shape[0])],
        ],
        dtype=np.float32,
    )
    matrix = find_homography(src, TRUE_QUAD.astype(np.float32))
    canvas = cv2.warpPerspective(
        sheet,
        matrix.astype(np.float32),
        CANVAS_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(BACKGROUND_VALUE,) * 3,
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 6.0, canvas.shape)
    return np.asarray(np.clip(canvas.astype(np.float64) + noise, 0.0, 255.0), dtype=np.uint8)


def _corner_errors(detection_quad: np.ndarray, true_quad: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(detection_quad - true_quad, axis=1), dtype=np.float64)


def test_locks_synthetic_keybed_within_8px() -> None:
    detection = find_keybed(_render_keybed(seed=7))
    assert detection is not None
    assert detection.confidence > 0.35
    assert _corner_errors(detection.quad_px, TRUE_QUAD).max() <= 8.0


def test_blank_and_noise_return_no_confident_detection() -> None:
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert find_keybed(blank) is None
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, size=(480, 640, 3)).astype(np.uint8)
    detection = find_keybed(noise)
    assert detection is None or detection.confidence <= 0.35


def test_occlusion_never_returns_wild_quad() -> None:
    image = _render_keybed(seed=11)
    cv2.ellipse(
        image,
        center=(320, 184),
        axes=(140, 60),
        angle=15.0,
        startAngle=0.0,
        endAngle=360.0,
        color=(10, 10, 10),
        thickness=-1,
    )
    detection = find_keybed(image)
    if detection is None:
        return
    assert _corner_errors(detection.quad_px, TRUE_QUAD).max() < 60.0


def test_empty_image_returns_none() -> None:
    assert find_keybed(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_square_blob_fails_aspect_filter() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[140:340, 220:420] = 235
    assert find_keybed(image) is None


def test_hollow_frame_fails_fill_filter() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 150), (520, 330), 235, thickness=6)
    assert find_keybed(image) is None
