import math

from kvt.train import train_model


def test_train_model_runs_small_schedule() -> None:
    model, mae = train_model(train_samples=32, val_samples=32, epochs=2, batch_size=16, workers=0)
    assert math.isfinite(mae)
    assert not model.training


def test_train_model_with_render_workers() -> None:
    _, mae = train_model(train_samples=8, val_samples=8, epochs=1, batch_size=8, workers=2)
    assert math.isfinite(mae)
