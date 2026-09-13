"""Bake the rendered corpus down to what the net consumes, for upload to a training host."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from kvt.dataset import DEFAULT_SYNTH_DIR, load_synth
from kvt.model import INPUT_SIZE

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "corpus"


def bake(synth_dir: Path = DEFAULT_SYNTH_DIR, out_dir: Path = DEFAULT_CORPUS_DIR) -> int:
    frames = load_synth(synth_dir)
    images_dir = out_dir / "frames"
    # a stale frame here is a frame from a different corpus, and nothing downstream can tell
    shutil.rmtree(images_dir, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    corners: dict[str, list[list[float]]] = {}
    for frame in frames:
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise ValueError(f"cannot read frame {frame.image_path}")
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(images_dir / f"{frame.source_stem}.png"), resized)
        normalised = frame.corners_px / np.array([float(width), float(height)])
        corners[frame.source_stem] = normalised.tolist()
    (out_dir / "corners.json").write_text(json.dumps(corners))
    return len(corners)


def main() -> None:
    parser = argparse.ArgumentParser(description="bake the render corpus for upload")
    parser.add_argument("--synth-dir", type=Path, default=DEFAULT_SYNTH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args()
    count = bake(args.synth_dir, args.out_dir)
    print(f"baked {count} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
