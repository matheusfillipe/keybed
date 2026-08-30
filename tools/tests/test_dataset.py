import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from kvt.dataset import extract, load_frames, scan_recordings


def _write_sidecar(path: Path, kind: str, width: int, height: int) -> None:
    sidecar = {
        "kind": kind,
        "startedAt": 0,
        "durationMs": 0,
        "corners": [
            {"x": 0.25, "y": 0.1},
            {"x": 0.75, "y": 0.1},
            {"x": 0.8, "y": 0.9},
            {"x": 0.2, "y": 0.9},
        ],
        "imageWidth": width,
        "imageHeight": height,
        "mimeType": "image/png" if kind == "snap" else "video/webm",
    }
    path.write_text(json.dumps(sidecar))


def _write_snap(path: Path) -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[10:38, 16:48] = 235
    cv2.imwrite(str(path), image)


def _write_clip(path: Path, frames: int) -> None:
    avi = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter.fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened()
    for i in range(frames):
        image = np.full((48, 64, 3), 30 + i, dtype=np.uint8)
        writer.write(image)
    writer.release()
    avi.replace(path)


def _make_recordings(recordings_dir: Path) -> None:
    recordings_dir.mkdir(parents=True)
    _write_snap(recordings_dir / "snap-test.png")
    _write_sidecar(recordings_dir / "snap-test.json", "snap", 320, 240)
    _write_clip(recordings_dir / "rec-test.webm", 90)
    _write_sidecar(recordings_dir / "rec-test.json", "rec", 320, 240)


def test_extract_samples_snap_and_clip(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    _make_recordings(recordings_dir)
    frames = extract(recordings_dir, frames_dir)
    assert [f.image_path.name for f in frames if f.kind == "snap"] == ["snap-test.png"]
    rec_frames = [f for f in frames if f.kind == "rec"]
    assert len(rec_frames) == 45
    assert rec_frames[0].image_path.name == "rec-test.000000.png"
    snap = frames[0] if frames[0].kind == "snap" else frames[-1]
    expected = np.array([[80.0, 24.0], [240.0, 24.0], [256.0, 216.0], [64.0, 216.0]])
    assert snap.corners_px is not None
    assert np.allclose(snap.corners_px, expected)
    labels = json.loads((frames_dir / "labels.json").read_text())
    assert set(labels["extracted"]) == {"snap-test", "rec-test"}
    assert (frames_dir / "snap-test.png").is_file()


def test_extract_is_idempotent(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    _make_recordings(recordings_dir)
    first = extract(recordings_dir, frames_dir)
    (recordings_dir / "rec-test.webm").unlink()
    second = extract(recordings_dir, frames_dir)
    assert [f.image_path.name for f in second] == [f.image_path.name for f in first]
    assert load_frames(frames_dir)[0].image_path == first[0].image_path


def test_scan_recordings_skips_sidecar_less_media(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    _write_snap(recordings_dir / "snap-a.png")
    (recordings_dir / "snap-a.json").write_text("{}")
    _write_snap(recordings_dir / "snap-b.png")
    (recordings_dir / "stray.txt").write_text("nope")
    recordings = scan_recordings(recordings_dir)
    assert [r.stem for r in recordings] == ["snap-a"]
    assert recordings[0].kind == "snap"
    assert recordings[0].media_path.name == "snap-a.png"


def test_extract_raises_on_unreadable_snapshot(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "snap-bad.png").write_text("not an image")
    _write_sidecar(recordings_dir / "snap-bad.json", "snap", 320, 240)
    with pytest.raises(ValueError, match="cannot read snapshot"):
        extract(recordings_dir, tmp_path / "frames")


def test_extract_raises_on_unopenable_clip(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "rec-bad.webm").write_text("not a video")
    _write_sidecar(recordings_dir / "rec-bad.json", "rec", 320, 240)
    with pytest.raises(ValueError, match="cannot open clip"):
        extract(recordings_dir, tmp_path / "frames")


def test_extract_raises_on_invalid_sidecars(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    recordings_dir.mkdir()
    _write_snap(recordings_dir / "snap-x.png")
    sidecar = recordings_dir / "snap-x.json"
    cases = [
        "[]",
        json.dumps({"corners": [], "imageWidth": 1, "imageHeight": 1}),
        json.dumps({"kind": "snap", "corners": [], "imageWidth": 1, "imageHeight": 1}),
        json.dumps(
            {"kind": "snap", "corners": [{}, {}, {}, {}], "imageWidth": 1, "imageHeight": 1}
        ),
        json.dumps(
            {
                "kind": "snap",
                "corners": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 1, "y": 1},
                    {"x": "a", "y": 1},
                ],
                "imageWidth": 1,
                "imageHeight": 1,
            }
        ),
    ]
    for content in cases:
        sidecar.write_text(content)
        with pytest.raises(ValueError):
            extract(recordings_dir, frames_dir)
        (frames_dir / "labels.json").unlink(missing_ok=True)


def _write_gemini_scene(gemini_dir: Path, stem: str, with_kind: bool) -> None:
    gemini_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[10:38, 16:48] = 235
    for suffix in ("-orig.png", "-mask.png", "-check.png"):
        cv2.imwrite(str(gemini_dir / f"{stem}{suffix}"), image)
    sidecar: dict[str, object] = {
        "startedAt": 0,
        "durationMs": 0,
        "corners": [
            {"x": 0.1, "y": 0.2},
            {"x": 0.7, "y": 0.2},
            {"x": 0.8, "y": 0.8},
            {"x": 0.2, "y": 0.9},
        ],
        "imageWidth": 320,
        "imageHeight": 240,
        "mimeType": "image/png",
    }
    if with_kind:
        sidecar["kind"] = "snap"
    (gemini_dir / f"{stem}.json").write_text(json.dumps(sidecar))


def test_extract_gemini_scenes_into_frames(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    gemini_dir = tmp_path / "gemini"
    frames_dir = tmp_path / "frames"
    recordings_dir.mkdir()
    _write_gemini_scene(gemini_dir, "00", with_kind=False)
    _write_gemini_scene(gemini_dir, "01", with_kind=True)
    (gemini_dir / "02-orig.png").write_text("no sidecar")
    frames = extract(recordings_dir, frames_dir, gemini_dir)
    gemini_frames = [f for f in frames if f.kind == "gemini"]
    assert [f.image_path.name for f in gemini_frames] == ["00-orig.png", "01-orig.png"]
    assert [f.source_stem for f in gemini_frames] == ["00", "01"]
    expected = np.array([[32.0, 48.0], [224.0, 48.0], [256.0, 192.0], [64.0, 216.0]])
    assert gemini_frames[0].corners_px is not None
    assert np.allclose(gemini_frames[0].corners_px, expected)
    again = extract(recordings_dir, frames_dir, gemini_dir)
    assert [f.image_path.name for f in again] == [f.image_path.name for f in frames]
    assert (frames_dir / "00-orig.png").is_file()


def test_extract_without_gemini_dir_leaves_frames_unchanged(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    _make_recordings(recordings_dir)
    with_gemini = extract(recordings_dir, frames_dir, tmp_path / "missing")
    without_gemini = extract(recordings_dir, tmp_path / "frames-2")
    assert [f.kind for f in with_gemini] == [f.kind for f in without_gemini]
    assert all(f.kind != "gemini" for f in without_gemini)


def test_load_frames_ignores_malformed_labels(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_snap(frames_dir / "snap-keep.png")
    labels = {
        "extracted": {"ok": 1, 5: "no", "bad": "no"},
        "frames": {
            "snap-keep.png": {
                "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "source_stem": "snap-keep",
                "kind": "snap",
            },
            "missing.png": {"corners_px": "junk", "source_stem": "m", "kind": "snap"},
            "no-fields.png": {"corners_px": None},
            "bad-corners.png": {
                "corners_px": [[0], [1], [2], [3]],
                "source_stem": "b",
                "kind": "snap",
            },
            "bad-point.png": {
                "corners_px": [[0, 0], [1, 0], [1, "x"], [0, 1]],
                "source_stem": "b",
                "kind": "snap",
            },
            "junk-entry": "nope",
        },
    }
    (frames_dir / "labels.json").write_text(json.dumps(labels))
    frames = load_frames(frames_dir)
    assert len(frames) == 1
    assert frames[0].image_path.name == "snap-keep.png"
    corners = frames[0].corners_px
    assert corners is not None
    assert np.allclose(corners, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_load_frames_keeps_null_corners(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    labels = {
        "extracted": {},
        "frames": {"f.png": {"corners_px": None, "source_stem": "f", "kind": "rec"}},
    }
    (frames_dir / "labels.json").write_text(json.dumps(labels))
    frames = load_frames(frames_dir)
    assert frames[0].corners_px is None
