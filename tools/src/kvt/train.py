"""Train KeybedNet on synthetic renders and keep the best-val checkpoint."""

import argparse
import math
from collections.abc import Iterator
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import torch

from kvt.model import KeybedNet, corner_loss, decode_heatmaps, heatmap_targets, preprocess
from kvt.render import render_sample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = _REPO_ROOT / "data" / "models" / "keybed_net.pt"

_FRAME_SIZE = (640, 480)
_DEFAULT_TRAIN_SAMPLES = 25_000
_DEFAULT_VAL_SAMPLES = 1_000
_DEFAULT_EPOCHS = 6
_BATCH_SIZE = 64
_LEARNING_RATE = 1e-3
_MIN_LEARNING_RATE = 1e-5
_SEED = 0
_RENDER_WORKERS = 4
_CHUNK_SIZE = 8


def _render_item(task: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed, epoch, index = task
    width, height = _FRAME_SIZE
    rng = np.random.default_rng([seed, epoch, index])
    sample = render_sample(rng, width, height)
    corners = (sample.quad_px / np.array([float(width), float(height)])).reshape(8)
    present = np.array(1.0 if sample.present else 0.0)
    return preprocess(sample.image), corners, present


def _iter_batches(
    pool: Pool | None, seed: int, epoch: int, count: int, batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tasks = [(seed, epoch, index) for index in range(count)]
    rendered = (
        pool.imap(_render_item, tasks, chunksize=_CHUNK_SIZE)
        if pool is not None
        else map(_render_item, tasks)
    )
    inputs: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    present: list[np.ndarray] = []
    for item_inputs, item_corners, item_present in rendered:
        inputs.append(item_inputs)
        corners.append(item_corners)
        present.append(item_present)
        if len(inputs) == batch_size:
            yield np.stack(inputs), np.stack(corners), np.stack(present)
            inputs, corners, present = [], [], []
    if inputs:
        yield np.stack(inputs), np.stack(corners), np.stack(present)


def _train_epoch(
    model: KeybedNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pool: Pool | None,
    train_samples: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> float:
    model.train()
    loss_sum = 0.0
    seen = 0
    for inputs, corners, present in _iter_batches(pool, seed, epoch, train_samples, batch_size):
        optimizer.zero_grad()
        pred_heatmaps, present_logits = model(torch.from_numpy(inputs).unsqueeze(1))
        loss = corner_loss(
            pred_heatmaps,
            present_logits,
            heatmap_targets(torch.from_numpy(corners), torch.from_numpy(present)),
            torch.from_numpy(present),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach()) * inputs.shape[0]
        seen += inputs.shape[0]
    return loss_sum / max(seen, 1)


def _evaluate(
    model: KeybedNet, pool: Pool | None, val_samples: int, batch_size: int, seed: int
) -> tuple[float, float, float, float]:
    model.eval()
    scale = np.array([float(_FRAME_SIZE[0]), float(_FRAME_SIZE[1])])
    error_sum = 0.0
    error_count = 0
    correct = 0
    seen = 0
    predicted_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, corners, present in _iter_batches(pool, seed + 1, 0, val_samples, batch_size):
            pred_heatmaps, present_logits = model(torch.from_numpy(inputs).unsqueeze(1))
            probabilities = torch.sigmoid(present_logits).numpy()
            predicted = decode_heatmaps(pred_heatmaps).numpy().reshape(-1, 4, 2)
            target = corners.reshape(-1, 4, 2)
            flags = present.astype(bool)
            if flags.any():
                errors = np.linalg.norm((predicted[flags] - target[flags]) * scale, axis=2)
                error_sum += float(errors.mean(axis=1).sum())
                error_count += int(flags.sum())
                predicted_all.append(predicted[flags])
                target_all.append(target[flags])
            correct += int(((probabilities >= 0.5) == flags).sum())
            seen += inputs.shape[0]
    mae = error_sum / error_count if error_count else 0.0
    predicted_std = 0.0
    target_std = 0.0
    if predicted_all and target_all:
        predicted_std = float((np.concatenate(predicted_all).std(axis=0) * scale).mean())
        target_std = float((np.concatenate(target_all).std(axis=0) * scale).mean())
    return mae, correct / max(seen, 1), predicted_std, target_std


def train_model(
    train_samples: int,
    val_samples: int,
    epochs: int,
    batch_size: int = _BATCH_SIZE,
    seed: int = _SEED,
    workers: int = _RENDER_WORKERS,
) -> tuple[KeybedNet, float]:
    torch.manual_seed(seed)
    model = KeybedNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    steps_per_epoch = math.ceil(train_samples / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * epochs, eta_min=_MIN_LEARNING_RATE
    )
    best_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    pool = Pool(processes=workers) if workers > 0 else None
    try:
        for epoch in range(epochs):
            train_loss = _train_epoch(
                model, optimizer, scheduler, pool, train_samples, batch_size, seed, epoch
            )
            mae, presence_accuracy, predicted_std, target_std = _evaluate(
                model, pool, val_samples, batch_size, seed
            )
            print(
                f"epoch {epoch + 1}/{epochs} train_loss {train_loss:.4f} "
                f"val_mae_px {mae:.2f} presence_acc {presence_accuracy:.3f} "
                f"pred_std_px {predicted_std:.1f} tgt_std_px {target_std:.1f}",
                flush=True,
            )
            if mae < best_mae:
                best_mae = mae
                best_state = {
                    name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                }
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_mae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="train the keybed corner detector on synthetic renders"
    )
    parser.add_argument("--train-samples", type=int, default=_DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=_DEFAULT_VAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    args = parser.parse_args()
    model, mae = train_model(args.train_samples, args.val_samples, args.epochs)
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
    print(f"saved {DEFAULT_MODEL_PATH} val_mae_px {mae:.2f}")


if __name__ == "__main__":
    main()
