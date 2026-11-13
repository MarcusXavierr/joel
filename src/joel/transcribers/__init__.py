"""Transcriber adapters, found by module name.

Each module ``joel/transcribers/<name>.py`` is the transcriber called ``<name>``
and defines a no-argument ``create() -> Transcriber``. Modules whose name starts
with ``_`` are helpers, not transcribers. An adapter that needs an SDK from an
optional extra shares the extra's name, and the SDK must be importable under its
distribution name (``import openai`` for the ``openai`` package).
"""

import importlib
import importlib.metadata
import pkgutil
from pathlib import Path
from typing import Protocol

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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
    adapter's SDK comes from an optional extra of joel that isn't installed.
    Any other missing module is re-raised as is.
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
        if not _provided_by_extra(exc.name, extra=name):
            raise
        raise TranscriberUnavailableError(
            f"The '{name}' transcriber needs the optional extra joel[{name}]. "
            f"Install it with: uv sync --extra {name}"
        ) from exc
    transcriber: Transcriber = module.create()
    return transcriber


def _provided_by_extra(module: str | None, extra: str) -> bool:
    """Whether ``module`` is a top-level package that joel[extra] installs."""
    if module is None:
        return False
    top_level = canonicalize_name(module.partition(".")[0])
    return top_level in _extra_packages(extra)


def _extra_packages(extra: str) -> set[str]:
    packages: set[str] = set()
    for line in importlib.metadata.metadata("joel").get_all("Requires-Dist") or []:
        requirement = Requirement(line)
        marker = requirement.marker
        # A marker that also holds with no extra (python_version, say) belongs
        # to a base dependency, not to this extra.
        if marker and marker.evaluate({"extra": extra}):
            if not marker.evaluate({"extra": ""}):
                packages.add(canonicalize_name(requirement.name))
    return packages
