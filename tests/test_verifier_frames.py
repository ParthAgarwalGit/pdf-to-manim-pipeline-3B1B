from pathlib import Path
from subprocess import CompletedProcess

from src.manim_verifier.frame_extractor import FrameExtractor


def test_four_frames_are_extracted(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    extractor = FrameExtractor()
    monkeypatch.setattr(extractor, "_probe_duration", lambda path: 8.0)

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"png")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    result = extractor.extract(video, tmp_path / "frames")
    assert len(result.frame_paths) == 4
    assert len(set(result.timestamps)) == 4
    assert not result.issues


def test_missing_or_corrupt_video_returns_issue(tmp_path, monkeypatch):
    extractor = FrameExtractor()
    monkeypatch.setattr(extractor, "_probe_duration", lambda path: 0.0)
    result = extractor.extract(tmp_path / "missing.mp4", tmp_path / "frames")
    assert not result.frame_paths
    assert result.issues[0].code == "FRAME_VIDEO_INVALID"


def test_short_video_timestamps_are_bounded_and_unique():
    timestamps = FrameExtractor._sample_timestamps(0.01)
    assert len(timestamps) == 4
    assert timestamps == sorted(set(timestamps))
    assert all(0 <= timestamp < 0.01 for timestamp in timestamps)
