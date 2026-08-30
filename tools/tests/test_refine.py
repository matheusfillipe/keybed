import cv2
import numpy as np
import pytest

from kvt.refine import _TEMPLATE, _score, refine_quad
from kvt.render import RenderSample, render_sample

WIDTH = 640
HEIGHT = 480
CANONICAL_QUAD = np.array([[80.0, 120.0], [560.0, 100.0], [600.0, 380.0], [60.0, 400.0]])
STRIP_SOURCE = np.array([[0.0, 0.0], [519.0, 0.0], [519.0, 63.0], [0.0, 63.0]], dtype=np.float32)
MONOTONIC_SEEDS = (10, 13, 18, 20, 21)
NO_HARM_SEED = 165


@pytest.fixture(autouse=True)
def _pure_synthetic_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kvt.render._REAL_FRAMES", [])


def _canonical_image() -> np.ndarray:
    sheet = (_TEMPLATE * 255.0).astype(np.uint8)
    sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    matrix = cv2.getPerspectiveTransform(STRIP_SOURCE, CANONICAL_QUAD.astype(np.float32))
    warped = cv2.warpPerspective(
        sheet_bgr,
        matrix,
        (WIDTH, HEIGHT),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(40, 40, 40),
    )
    return cv2.GaussianBlur(warped, (0, 0), 1.0)


def _render_sample(seed: int) -> RenderSample:
    sample = render_sample(np.random.default_rng(seed))
    assert sample.present
    return sample


def _image_bgr(sample: RenderSample) -> np.ndarray:
    return cv2.cvtColor(sample.image.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _max_corner_error(quad_px: np.ndarray, true_quad: np.ndarray) -> float:
    return float(np.linalg.norm(quad_px - true_quad, axis=1).max())


def test_refine_is_deterministic_on_fixed_seed_render() -> None:
    sample = _render_sample(5)
    init = CANONICAL_QUAD + np.random.default_rng(33).uniform(-60.0, 60.0, size=(4, 2))
    first = refine_quad(_image_bgr(sample), init)
    second = refine_quad(_image_bgr(sample), init)
    assert np.array_equal(first, second)


def test_refine_snaps_canonical_render_within_10px() -> None:
    init = CANONICAL_QUAD + np.random.default_rng(1).uniform(-4.0, 4.0, size=(4, 2))
    refined = refine_quad(_canonical_image(), init)
    assert _max_corner_error(refined, CANONICAL_QUAD) <= 10.0


def test_refine_basin_edge_within_15px() -> None:
    init = CANONICAL_QUAD + np.random.default_rng(1).uniform(-6.0, 6.0, size=(4, 2))
    refined = refine_quad(_canonical_image(), init)
    assert _max_corner_error(refined, CANONICAL_QUAD) <= 15.0


def test_refine_no_harm_on_true_quad() -> None:
    sample = _render_sample(NO_HARM_SEED)
    image_bgr = _image_bgr(sample)
    refined = refine_quad(image_bgr, sample.quad_px)
    assert _max_corner_error(refined, sample.quad_px) <= 1.0
    assert _score(image_bgr, refined, 520, 64) >= _score(image_bgr, sample.quad_px, 520, 64)


def test_refine_reduces_mean_corner_error_across_seeds() -> None:
    for seed in MONOTONIC_SEEDS:
        rng = np.random.default_rng(seed)
        sample = render_sample(rng)
        image_bgr = _image_bgr(sample)
        init = sample.quad_px + rng.uniform(-60.0, 60.0, size=(4, 2))
        refined = refine_quad(image_bgr, init)
        init_error = float(np.linalg.norm(init - sample.quad_px, axis=1).mean())
        refined_error = float(np.linalg.norm(refined - sample.quad_px, axis=1).mean())
        assert refined_error < init_error
