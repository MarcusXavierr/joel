from collections.abc import Callable
from pathlib import Path

import pytest

from joel.decode import DecodeError, decode


def test_missing_ffmpeg_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(DecodeError, match="ffmpeg was not found"):
        decode(tmp_path / "talk.m4a")


def test_ffmpeg_failure_reports_its_last_line(
    tmp_path: Path, fake_ffmpeg: Callable[[str], None]
) -> None:
    fake_ffmpeg(
        'echo "first line" >&2\n'
        'echo "talk.m4a: Invalid data found when processing input" >&2\n'
        "exit 1"
    )
    audio = tmp_path / "talk.m4a"

    with pytest.raises(DecodeError) as excinfo:
        decode(audio)

    message = str(excinfo.value)
    assert str(audio) in message
    assert "Invalid data found when processing input" in message
    assert "first line" not in message


def test_file_without_audio_is_an_error(
    tmp_path: Path, fake_ffmpeg: Callable[[str], None]
) -> None:
    fake_ffmpeg("exit 0")

    with pytest.raises(DecodeError, match="contains no audio"):
        decode(tmp_path / "silent.mp4")
