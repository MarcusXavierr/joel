"""Step 0 of the pipeline: any audio file to the PCM transcribe-cpp accepts."""

import subprocess
from pathlib import Path

SAMPLE_RATE = 16_000


class DecodeError(Exception):
    """An audio file that ffmpeg can't turn into PCM, or no ffmpeg at all."""


def decode(audio: Path) -> bytes:
    """Return the audio as little-endian float32 PCM, 16 kHz, mono."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        # Stop on a damaged stretch instead of skipping it: skipped audio would
        # shift every later timestamp and splice words across the gap.
        "-xerror",
        "-i",
        # Absolute, so a file name that starts with "-" isn't read as an option.
        str(audio.absolute()),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise DecodeError(
            "ffmpeg was not found on PATH. Joel needs it to read audio files; "
            "install ffmpeg and try again."
        ) from exc
    if result.returncode != 0:
        lines = result.stderr.decode("utf-8", "replace").splitlines()
        causes = [line.strip() for line in lines if line.strip()]
        detail = causes[-1] if causes else f"exit status {result.returncode}"
        raise DecodeError(f"ffmpeg could not read {audio}: {detail}")
    if not result.stdout:
        raise DecodeError(f"{audio} contains no audio.")
    return result.stdout
