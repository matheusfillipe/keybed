"""Evaluate the keybed detector against sidecar ground truth."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from kvt.dataset import Frame, load_frames
from kvt.detect import find_keybed, find_keybed_pattern

Method = Literal["v0", "pattern"]
_DETECTORS = {"v0": find_keybed, "pattern": find_keybed_pattern}

_SUCCESS_RADIUS_PX = 15.0
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAMES_DIR = _REPO_ROOT / "data" / "frames"
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "out" / "detect"
_GREEN = (0, 255, 0)
_BLUE = (255, 0, 0)
_RED = (0, 0, 255)


@dataclass
class FrameResult:
    source_stem: str
    kind: str
    quad_px: np.ndarray | None
    locked: bool
    success: bool
    mean_error_px: float | None


def evaluate_frame(frame: Frame, method: Method = "pattern") -> FrameResult:
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise ValueError(f"cannot read frame {frame.image_path}")
    detection = _DETECTORS[method](image)
    if detection is None:
        return FrameResult(
            source_stem=frame.source_stem,
            kind=frame.kind,
            quad_px=None,
            locked=False,
            success=False,
            mean_error_px=None,
        )
    if frame.corners_px is None:
        return FrameResult(
            source_stem=frame.source_stem,
            kind=frame.kind,
            quad_px=detection.quad_px,
            locked=True,
            success=False,
            mean_error_px=None,
        )
    errors = np.linalg.norm(detection.quad_px - frame.corners_px, axis=1)
    return FrameResult(
        source_stem=frame.source_stem,
        kind=frame.kind,
        quad_px=detection.quad_px,
        locked=True,
        success=bool(float(errors.max()) <= _SUCCESS_RADIUS_PX),
        mean_error_px=float(errors.mean()),
    )


def run(frames_dir: Path, out_dir: Path, method: Method = "pattern") -> list[FrameResult]:
    frames = load_frames(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_frame(frame, method) for frame in frames]
    for frame, result in zip(frames, results, strict=True):
        _write_preview(frame, result, out_dir)
    _print_table(results)
    return results


@dataclass
class _GroupStats:
    label: str
    kind: str
    frames: int = 0
    locked: int = 0
    successes: int = 0
    error_sum: float = 0.0
    error_count: int = 0

    def add(self, result: FrameResult) -> None:
        self.frames += 1
        if result.locked:
            self.locked += 1
            if result.mean_error_px is not None:
                self.error_sum += result.mean_error_px
                self.error_count += 1
        if result.success:
            self.successes += 1

    def merge(self, other: "_GroupStats") -> None:
        self.frames += other.frames
        self.locked += other.locked
        self.successes += other.successes
        self.error_sum += other.error_sum
        self.error_count += other.error_count

    @property
    def lock_rate(self) -> float:
        return self.locked / self.frames if self.frames else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.frames if self.frames else 0.0

    @property
    def mean_error(self) -> float | None:
        return self.error_sum / self.error_count if self.error_count else None


def _print_table(results: list[FrameResult]) -> None:
    groups: dict[tuple[str, str], _GroupStats] = {}
    for result in results:
        groups.setdefault(
            (result.source_stem, result.kind),
            _GroupStats(result.source_stem, result.kind),
        ).add(result)
    by_kind: dict[str, _GroupStats] = {}
    overall = _GroupStats("all", "all")
    for stats in groups.values():
        by_kind.setdefault(stats.kind, _GroupStats(stats.kind, stats.kind)).merge(stats)
        overall.merge(stats)
    print(
        f"{'source':<28} {'kind':<5} {'frames':>6} {'lock':>6} {'success':>8} {'mean_err_px':>12}"
    )
    for stats in groups.values():
        print(_format_row(stats.label, stats.kind, stats))
    for kind in sorted(by_kind):
        stats = by_kind[kind]
        print(_format_row(f"{kind} (all)", kind, stats))
    print(_format_row(overall.label, overall.kind, overall))


def _format_row(label: str, kind: str, stats: _GroupStats) -> str:
    mean_error = "-" if stats.mean_error is None else f"{stats.mean_error:.1f}"
    return (
        f"{label:<28} {kind:<5} {stats.frames:>6} {stats.lock_rate:>6.2f} "
        f"{stats.success_rate:>8.2f} {mean_error:>12}"
    )


def _write_preview(frame: Frame, result: FrameResult, out_dir: Path) -> None:
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise ValueError(f"cannot read frame {frame.image_path}")
    if frame.corners_px is not None:
        color = _GREEN if result.locked else _RED
        cv2.polylines(image, [frame.corners_px.astype(np.int32)], True, color, 2)
    if result.quad_px is not None:
        cv2.polylines(image, [result.quad_px.astype(np.int32)], True, _BLUE, 2)
    cv2.imwrite(str(out_dir / frame.image_path.name), image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="evaluate keybed detection against sidecar ground truth"
    )
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--method", choices=("v0", "pattern"), default="pattern")
    args = parser.parse_args()
    run(args.frames_dir, args.out_dir, cast(Method, args.method))


if __name__ == "__main__":
    main()
