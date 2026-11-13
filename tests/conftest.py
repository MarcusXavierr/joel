from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def fake_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str], None]:
    """Make ``ffmpeg`` on PATH a shell script with the given body, and nothing else."""

    def install(script: str) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        ffmpeg = bin_dir / "ffmpeg"
        ffmpeg.write_text(f"#!/bin/sh\n{script}\n")
        ffmpeg.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))

    return install
