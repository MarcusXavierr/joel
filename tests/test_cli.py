import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from joel.cli import main

AUDIO = Path(__file__).parent / "fixtures" / "audio" / "acceptance.wav"


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
