import numpy as np

from kvt.render import render_sample

WIDTH = 640
HEIGHT = 480


def test_render_sample_is_deterministic_given_seed() -> None:
    first = render_sample(np.random.default_rng(3))
    second = render_sample(np.random.default_rng(3))
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.quad_px, second.quad_px)
    assert first.present == second.present


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
