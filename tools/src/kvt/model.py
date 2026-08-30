"""KeybedNet: per-corner heatmap detection over grayscale input."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUT_SIZE = 288
HEATMAP_SIZE = 72
_GAUSSIAN_SIGMA = 1.5


class KeybedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
        )
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1),
        )
        self.present_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(x)
        return self.heatmap_head(features), self.present_head(features).squeeze(-1)


def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    height, width = heatmaps.shape[-2:]
    flat = heatmaps.flatten(2).clamp(min=0.0)
    total = flat.sum(dim=-1, keepdim=True)
    weights = flat / total.clamp(min=torch.finfo(flat.dtype).tiny)
    xs = torch.arange(width, dtype=heatmaps.dtype, device=heatmaps.device).repeat(height)
    ys = torch.arange(height, dtype=heatmaps.dtype, device=heatmaps.device).repeat_interleave(width)
    expected = torch.stack((weights @ xs, weights @ ys), dim=-1)
    center = torch.tensor(
        ((width - 1) / 2.0, (height - 1) / 2.0), dtype=heatmaps.dtype, device=heatmaps.device
    )
    expected = torch.where(total <= 0.0, center, expected)
    return expected.flatten(1) / float(width - 1)


def heatmap_targets(corners: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    batch = corners.shape[0]
    positions = corners.view(batch, 4, 2) * float(HEATMAP_SIZE - 1)
    positions = positions.clamp(0.0, float(HEATMAP_SIZE - 1))
    xs = torch.arange(HEATMAP_SIZE, dtype=corners.dtype, device=corners.device)
    ys = torch.arange(HEATMAP_SIZE, dtype=corners.dtype, device=corners.device)
    squared = (positions[:, :, 0, None, None] - xs[None, None, None, :]) ** 2 + (
        positions[:, :, 1, None, None] - ys[None, None, :, None]
    ) ** 2
    blobs = torch.exp(-squared / (2.0 * _GAUSSIAN_SIGMA**2))
    return blobs * present.to(corners.dtype).view(-1, 1, 1, 1)


def corner_loss(
    pred_heatmaps: torch.Tensor,
    present_logits: torch.Tensor,
    target_heatmaps: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    mask = present.to(pred_heatmaps.dtype)
    target = target_heatmaps.to(pred_heatmaps.dtype)
    denominator = mask.sum().clamp(min=1.0)
    per_sample = ((pred_heatmaps - target) ** 2).flatten(1).mean(dim=1)
    corner_term = (per_sample * mask).sum() / denominator
    presence_term = F.binary_cross_entropy_with_logits(present_logits, mask)
    return corner_term + presence_term


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0.0, 255.0).astype(np.uint8)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.float32) / 255.0


def predict_corners(model: KeybedNet, image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(preprocess(image_rgb)).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        heatmaps, present_logits = model(tensor)
        corners = decode_heatmaps(heatmaps)
    probability = float(torch.sigmoid(present_logits)[0])
    height, width = image_rgb.shape[:2]
    quad = corners[0].numpy().astype(np.float64).reshape(4, 2)
    quad = quad * np.array([float(width), float(height)])
    return quad, probability


def load_model(path: Path) -> KeybedNet:
    model = KeybedNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
