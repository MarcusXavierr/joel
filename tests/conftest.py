import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from errant.annotator import Annotator  # type: ignore[import-untyped]

from joel.edits import load_annotator


@pytest.fixture(scope="session")
def annotator() -> Annotator:
    """ERRANT with spaCy's en_core_web_sm, loaded once per test session."""
    return load_annotator()


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    """Undo configure_logging(), which rebinds the root logger to this test's stderr."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


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
