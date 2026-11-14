import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from joel.cli import main

AUDIO = Path(__file__).parent / "fixtures" / "audio" / "acceptance.wav"
# 60 seconds of 16 kHz float32 PCM.
ONE_MINUTE_OF_PCM = "exec /usr/bin/head -c 3840000 /dev/zero"

# main() configures logging for the whole process; each test gets the root back.
pytestmark = pytest.mark.usefixtures("restore_root_logger")


def test_a_recording_without_its_text_stops_before_any_transcriber(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "01.wav").write_bytes(b"")

    code = main(["bench", str(tmp_path)])

    out, err = capsys.readouterr()
    assert code == 1
    assert "01.wav has no matching 01.txt" in err
    assert out == ""


def test_a_missing_directory_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["bench", str(tmp_path / "nope")])

    assert code == 1
    assert "is not a directory" in capsys.readouterr().err


def test_every_transcriber_is_reported_even_when_each_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "01.wav").write_bytes(b"")
    (tmp_path / "01.txt").write_text("he [go] home")

    code = main(
        ["bench", str(tmp_path), "--transcriber", "alpha", "--transcriber", "beta"]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert out.index("== alpha ==") < out.index("== beta ==")
    for name in ("alpha", "beta"):
        assert (
            f"error: TranscriberUnavailableError: Unknown transcriber '{name}'" in out
        )


def test_a_transcriber_named_twice_runs_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "01.wav").write_bytes(b"")
    (tmp_path / "01.txt").write_text("he [go] home")

    main(["bench", str(tmp_path), "--transcriber", "alpha", "--transcriber", "alpha"])

    assert capsys.readouterr().out.count("== alpha ==") == 1


def test_a_file_the_transcriber_fails_on_is_reported_and_fails_the_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_ffmpeg: Callable[[str], None],
) -> None:
    # parakeet decodes before it looks for the model, so this never loads it.
    fake_ffmpeg("echo 'Invalid data found when processing input' >&2; exit 1")
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    (bench_dir / "01.wav").write_bytes(b"")
    (bench_dir / "01.txt").write_text("he [go] home")

    code = main(["bench", str(bench_dir)])

    out = capsys.readouterr().out
    assert code == 1
    assert "Errors:\n01  DecodeError: ffmpeg could not read" in out


@pytest.mark.slow
def test_bench_runs_parakeet_end_to_end(tmp_path: Path) -> None:
    shutil.copy(AUDIO, tmp_path / "01.wav")
    (tmp_path / "01.txt").write_text(
        "Yesterday he [go] to school, uh, and she [don't] like it. "
        "There [is] many people in the the room. I [am agree] with you."
    )

    # The console script, so the [project.scripts] entry is what runs.
    joel = Path(sys.executable).with_name("joel")
    result = subprocess.run(
        [str(joel), "bench", str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "== parakeet ==" in result.stdout
    assert "4/4" in result.stdout
    assert "0.0%" in result.stdout
    # The words end inside the audio, so audio_duration_ms measured it right.
    rows = [line.split() for line in result.stdout.splitlines()]
    assert [row[-1] for row in rows if row[:1] == ["01"]] == ["ok"]


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
    assert "openai:gpt-6-luna: " in out
    assert "for 4 calls" in out
    assert (
        "Assumes 150 words per minute, 12 words per sentence and 7 sentences per "
        "call, with no prompt caching; a retried call costs extra." in out
    )


def test_estimate_takes_the_block_size(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ffmpeg(ONE_MINUTE_OF_PCM)

    status = main(["estimate", str(tmp_path / "talk.m4a"), "--block-sentences", "1"])

    out = capsys.readouterr().out
    assert status == 0
    assert "for 26 calls" in out
    assert "12 words per sentence and 1 sentence per call," in out


def test_a_cost_below_a_cent_shows_three_decimals(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ffmpeg("exec /usr/bin/head -c 64000 /dev/zero")  # one second

    status = main(
        ["estimate", str(tmp_path / "talk.m4a"), "--model", "anthropic:claude-opus-5-5"]
    )

    assert status == 0
    out = capsys.readouterr().out
    assert re.search(r"anthropic:claude-opus-5-5: about \$0\.00[1-9] for 2 calls", out)


def test_a_cost_below_a_tenth_of_a_cent_says_so(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ffmpeg("exec /usr/bin/head -c 64000 /dev/zero")  # one second

    status = main(["estimate", str(tmp_path / "talk.m4a")])

    assert status == 0
    assert "openai:gpt-6-luna: under $0.001 for 2 calls" in capsys.readouterr().out


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

    result = subprocess.run([joel, "estimate", AUDIO], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "0:09 of audio, about 22 words in 2 sentences" in result.stdout
    assert "for 2 calls" in result.stdout
