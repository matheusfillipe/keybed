from pathlib import Path

import numpy as np
import torch

from kvt.model import KeybedNet, corner_loss, load_model, predict_corners, preprocess
from kvt.render import render_sample

FRAME_SIZE = (640.0, 480.0)


def test_forward_output_shapes() -> None:
    model = KeybedNet()
    model.eval()
    with torch.no_grad():
        corners, present_logits = model(torch.zeros(2, 1, 288, 288))
    assert corners.shape == (2, 8)
    assert present_logits.shape == (2,)
    assert bool((corners >= 0.0).all())
    assert bool((corners <= 1.0).all())


def test_parameter_count_under_500k() -> None:
    assert sum(p.numel() for p in KeybedNet().parameters()) < 500_000


def test_preprocess_resizes_and_normalizes() -> None:
    image = np.full((480, 640, 3), 200.0, dtype=np.float32)
    prepared = preprocess(image)
    assert prepared.shape == (288, 288)
    assert prepared.dtype == np.float32
    assert np.allclose(prepared, 200.0 / 255.0)


def _fixed_batch(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [render_sample(np.random.default_rng(seed)) for seed in range(count)]
    inputs = np.stack([preprocess(sample.image) for sample in samples])
    corners = np.stack([(sample.quad_px / np.array(FRAME_SIZE)).reshape(8) for sample in samples])
    present = np.array([1.0 if sample.present else 0.0 for sample in samples])
    return inputs, corners, present


def test_corner_loss_decreases_over_training_steps() -> None:
    inputs, corners, present = _fixed_batch(32)
    torch.manual_seed(0)
    model = KeybedNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    tensor = torch.from_numpy(inputs).unsqueeze(1)
    target = torch.from_numpy(corners)
    flags = torch.from_numpy(present)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        pred_corners, present_logits = model(tensor)
        loss = corner_loss(pred_corners, present_logits, target, flags)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_predict_corners_scales_back_to_frame() -> None:
    sample = render_sample(np.random.default_rng(2))
    quad, probability = predict_corners(KeybedNet(), sample.image)
    assert quad.shape == (4, 2)
    assert 0.0 <= probability <= 1.0
    assert np.all(quad[:, 0] >= 0.0)
    assert np.all(quad[:, 0] <= FRAME_SIZE[0])
    assert np.all(quad[:, 1] >= 0.0)
    assert np.all(quad[:, 1] <= FRAME_SIZE[1])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "keybed_net.pt"
    source = KeybedNet()
    torch.save(source.state_dict(), path)
    loaded = load_model(path)
    tensor = torch.zeros(1, 1, 288, 288)
    with torch.no_grad():
        source_corners, source_present = source(tensor)
        loaded_corners, loaded_present = loaded(tensor)
    assert torch.allclose(source_corners, loaded_corners)
    assert torch.allclose(source_present, loaded_present)
    assert not loaded.training
