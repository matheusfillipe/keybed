import cv2
import numpy as np
import pytest

from kvt.refine import _score
from kvt.render import render_sample

WIDTH = 640
HEIGHT = 480
STRIP_WIDTH = 520
STRIP_HEIGHT = 64
BACKGROUND_QUAD = np.array([[100.0, 80.0], [520.0, 60.0], [560.0, 400.0], [60.0, 420.0]])


def _flat_background() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 80.0, dtype=np.float32)


def test_render_sample_is_deterministic_given_seed() -> None:
    first = render_sample(np.random.default_rng(3))
    second = render_sample(np.random.default_rng(3))
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.quad_px, second.quad_px)
    assert first.present == second.present


def test_render_sample_composite_mode_is_deterministic_given_seed() -> None:
    first = render_sample(
        np.random.default_rng(4), WIDTH, HEIGHT, _flat_background(), BACKGROUND_QUAD
    )
    second = render_sample(
        np.random.default_rng(4), WIDTH, HEIGHT, _flat_background(), BACKGROUND_QUAD
    )
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.quad_px, second.quad_px)
    assert first.present == second.present


def test_render_sample_composite_mode_renders_keybed_at_true_quad() -> None:
    sample = render_sample(
        np.random.default_rng(4), WIDTH, HEIGHT, _flat_background(), BACKGROUND_QUAD
    )
    assert sample.image.shape == (HEIGHT, WIDTH, 3)
    assert sample.image.dtype == np.float32
    assert float(sample.image.min()) >= 0.0
    assert float(sample.image.max()) <= 255.0
    assert sample.present
    image_bgr = cv2.cvtColor(sample.image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    assert _score(image_bgr, sample.quad_px, STRIP_WIDTH, STRIP_HEIGHT) > 0.3


def test_render_sample_composite_mode_absent_leaves_inpaint_only() -> None:
    background = _flat_background()
    cv2.rectangle(background, (200, 100), (440, 380), (200.0, 200.0, 200.0), -1)
    samples = [
        render_sample(np.random.default_rng(seed), WIDTH, HEIGHT, background, BACKGROUND_QUAD)
        for seed in range(40)
    ]
    assert any(not sample.present for sample in samples)
    for sample in samples:
        if not sample.present:
            image_bgr = cv2.cvtColor(sample.image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            assert _score(image_bgr, sample.quad_px, STRIP_WIDTH, STRIP_HEIGHT) <= 0.3


def test_render_sample_composite_mode_resizes_background() -> None:
    background = np.full((240, 320, 3), 80.0, dtype=np.float32)
    quad = np.array([[50.0, 40.0], [260.0, 30.0], [280.0, 200.0], [30.0, 210.0]])
    sample = render_sample(np.random.default_rng(9), WIDTH, HEIGHT, background, quad)
    assert sample.image.shape == (HEIGHT, WIDTH, 3)


def test_render_sample_falls_back_to_synthetic_without_real_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kvt.render._REAL_FRAMES", [])
    sample = render_sample(np.random.default_rng(3))
    assert sample.image.shape == (HEIGHT, WIDTH, 3)
    assert sample.quad_px.shape == (4, 2)


def test_render_sample_shapes_and_ranges() -> None:
    sample = render_sample(np.random.default_rng(0))
    assert sample.image.shape == (HEIGHT, WIDTH, 3)
    assert sample.image.dtype == np.float32
    assert float(sample.image.min()) >= 0.0
    assert float(sample.image.max()) <= 255.0
    assert sample.quad_px.shape == (4, 2)
    assert isinstance(sample.present, bool)


def test_render_sample_supports_custom_size() -> None:
    sample = render_sample(np.random.default_rng(1), width=320, height=240)
    assert sample.image.shape == (240, 320, 3)


def test_render_sample_present_flag_varies() -> None:
    samples = [render_sample(np.random.default_rng(seed)) for seed in range(40)]
    assert any(sample.present for sample in samples)
    assert any(not sample.present for sample in samples)


def test_render_sample_quad_stays_roughly_inside_frame() -> None:
    for seed in range(40):
        sample = render_sample(np.random.default_rng(seed))
        x = sample.quad_px[:, 0]
        y = sample.quad_px[:, 1]
        assert x.min() >= -0.35 * WIDTH
        assert x.max() <= 1.35 * WIDTH
        assert y.min() >= -0.3 * HEIGHT
        assert y.max() <= 1.3 * HEIGHT
