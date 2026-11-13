import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from joel.cli import main

ACCEPTANCE_WAV = Path(__file__).parent / "fixtures" / "audio" / "acceptance.wav"

# 60 seconds of 16 kHz float32 PCM.
ONE_MINUTE_OF_PCM = "exec /usr/bin/head -c 3840000 /dev/zero"


def test_estimate_reports_duration_words_and_cost(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ffmpeg(ONE_MINUTE_OF_PCM)

    status = main(["estimate", str(tmp_path / "talk.m4a")])

    out = capsys.readouterr().out
    assert status == 0
    assert "1:00 of audio, about 150 words in 13 sentences" in out
    assert "anthropic:claude-opus-5-5: about $" in out
    assert "for 26 calls" in out


def test_a_cost_below_a_cent_says_so(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ffmpeg("exec /usr/bin/head -c 64000 /dev/zero")  # one second

    status = main(["estimate", str(tmp_path / "talk.m4a")])

    assert status == 0
    assert (
        "anthropic:claude-opus-5-5: under $0.01 for 2 calls" in capsys.readouterr().out
    )


def test_unreadable_audio_is_an_error(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ffmpeg('echo "Invalid data found" >&2; exit 1')

    status = main(["estimate", str(tmp_path / "talk.m4a")])

    assert status == 1
    assert capsys.readouterr().err.startswith("joel: error: ffmpeg could not read")


def test_a_model_without_a_price_is_an_error(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ffmpeg(ONE_MINUTE_OF_PCM)

    status = main(["estimate", str(tmp_path / "talk.m4a"), "--model", "nope:nothing"])

    assert status == 1
    assert "nope:nothing" in capsys.readouterr().err


@pytest.mark.slow
def test_installed_command_estimates_a_real_recording() -> None:
    joel = Path(sys.executable).parent / "joel"

    result = subprocess.run(
        [joel, "estimate", ACCEPTANCE_WAV], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert "0:09 of audio, about 22 words in 2 sentences" in result.stdout
    assert "for 4 calls" in result.stdout
