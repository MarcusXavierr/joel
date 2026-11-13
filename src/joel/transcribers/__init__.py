"""Transcriber adapters, found by module name.

Each module ``joel/transcribers/<name>.py`` is the transcriber called ``<name>``
and defines a no-argument ``create() -> Transcriber``. Modules whose name starts
with ``_`` are helpers, not transcribers.
"""

import importlib
import importlib.metadata
import pkgutil
from pathlib import Path
from typing import Protocol

from joel.transcript import Transcript


class Transcriber(Protocol):
    """Turns an audio file into a transcript. Every adapter satisfies this."""

    def transcribe(self, audio: Path) -> Transcript: ...


class TranscriberUnavailableError(Exception):
    """A transcriber name that can't be turned into a working adapter."""


def transcriber_names(package: str = __name__) -> list[str]:
    """Return the names of the transcribers in ``package``, sorted."""
    path = importlib.import_module(package).__path__
    modules = pkgutil.iter_modules(path)
    return sorted(m.name for m in modules if not m.name.startswith("_"))


def load_transcriber(name: str, package: str = __name__) -> Transcriber:
    """Create the transcriber called ``name``.

    Raises TranscriberUnavailableError when the name is unknown or when the
    adapter needs an optional extra of joel that isn't installed.
    """
    names = transcriber_names(package)
    if name not in names:
        valid = ", ".join(names) or "none"
        raise TranscriberUnavailableError(
            f"Unknown transcriber '{name}'. Valid names: {valid}."
        )
    try:
        module = importlib.import_module(f"{package}.{name}")
    except ModuleNotFoundError as exc:
        if name not in _optional_extras():
            raise
        raise TranscriberUnavailableError(
            f"The '{name}' transcriber needs the optional extra joel[{name}]. "
            f"Install it with: uv sync --extra {name}"
        ) from exc
    transcriber: Transcriber = module.create()
    return transcriber


def _optional_extras() -> set[str]:
    return set(importlib.metadata.metadata("joel").get_all("Provides-Extra") or [])
