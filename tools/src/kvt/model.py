"""KeybedNet: quad regression over grayscale input."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUT_SIZE = 288
_SMOOTH_L1_BETA = 0.01


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
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(6), nn.Flatten(), nn.Linear(96 * 6 * 6, 256), nn.ReLU()
        )
        self.corners_head = nn.Linear(256, 8)
        self.present_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.head(self.body(x))
        return torch.sigmoid(self.corners_head(features)), self.present_head(features).squeeze(-1)


def corner_loss(
    pred_corners: torch.Tensor,
    present_logits: torch.Tensor,
    target_corners: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    mask = present.to(pred_corners.dtype).unsqueeze(1)
    denominator = (mask.sum() * pred_corners.shape[1]).clamp(min=1.0)
    corner_term = (
        F.smooth_l1_loss(
            pred_corners * mask, target_corners * mask, reduction="sum", beta=_SMOOTH_L1_BETA
        )
        / denominator
    )
    presence_term = F.binary_cross_entropy_with_logits(
        present_logits, present.to(pred_corners.dtype)
    )
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
        corners, present_logits = model(tensor)
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
