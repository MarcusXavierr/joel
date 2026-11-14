"""Find spaCy's pinned en_core_web_sm model, downloading it only as a last resort.

The model isn't on PyPI, so Joel's wheel can't depend on it. Resolution goes:

1. the installed package at the pinned version (the dev group installs it);
2. a copy Joel extracted earlier into its cache;
3. a download of the pinned wheel from spaCy's GitHub release, checked by sha256.
"""

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import httpx

SPACY_MODEL_PACKAGE = "en_core_web_sm"
SPACY_MODEL_VERSION = "3.8.0"
SPACY_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
SPACY_MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
SPACY_MODEL_SIZE_BYTES = 12_806_118

_FAILURE = "Could not get spaCy's en_core_web_sm model. "


class SpacyModelUnavailableError(Exception):
    """The pinned spaCy model is neither installed, cached, nor downloadable."""


def default_cache_dir() -> Path:
    """Joel's own cache: ``$XDG_CACHE_HOME/joel``, else ``~/.cache/joel``."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "joel"


def resolve_spacy_model(
    cache_dir: Path | None = None, client: httpx.Client | None = None
) -> Path:
    """Return the directory of the pinned model, ready for ``spacy.load``.

    ``cache_dir`` defaults to ``default_cache_dir()``. ``client`` is used for
    the download; without one, Joel opens its own. Raises
    SpacyModelUnavailableError when the model can't be found or downloaded.
    """
    installed = _installed()
    if installed is not None:
        return installed
    cache = cache_dir if cache_dir is not None else default_cache_dir()
    target = cache / f"{SPACY_MODEL_PACKAGE}-{SPACY_MODEL_VERSION}"
    if (target / "meta.json").is_file():
        return target
    if client is not None:
        return _download(cache, target, client)
    # GitHub serves release assets through a redirect.
    with httpx.Client(follow_redirects=True, timeout=60) as own_client:
        return _download(cache, target, own_client)


def _installed() -> Path | None:
    """The model directory inside the installed package, if it's the pinned one."""
    try:
        version = importlib.metadata.version(SPACY_MODEL_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return None
    if version != SPACY_MODEL_VERSION:
        return None
    spec = importlib.util.find_spec(SPACY_MODEL_PACKAGE)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent / f"{SPACY_MODEL_PACKAGE}-{SPACY_MODEL_VERSION}"


def _download(cache: Path, target: Path, client: httpx.Client) -> Path:
    """Download, check, and extract the pinned wheel's model into ``target``.

    The model is extracted into a temporary directory and renamed into place,
    so ``target`` either holds the whole model or doesn't exist.
    """
    wheel_path: Path | None = None
    extracted: Path | None = None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            dir=cache, suffix=".whl", delete=False
        ) as wheel:
            wheel_path = Path(wheel.name)
            with client.stream("GET", SPACY_MODEL_URL) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    wheel.write(chunk)
                    digest.update(chunk)
        got = digest.hexdigest()
        if got != SPACY_MODEL_SHA256:
            raise SpacyModelUnavailableError(
                f"{_FAILURE}The download from {SPACY_MODEL_URL} has sha256 {got}, "
                f"but Joel expects {SPACY_MODEL_SHA256}, so it was discarded."
            )
        extracted = Path(tempfile.mkdtemp(dir=cache))
        prefix = f"{SPACY_MODEL_PACKAGE}/{SPACY_MODEL_PACKAGE}-{SPACY_MODEL_VERSION}/"
        with zipfile.ZipFile(wheel_path) as archive:
            members = [n for n in archive.namelist() if n.startswith(prefix)]
            archive.extractall(extracted, members=members)
        try:
            (extracted / prefix).rename(target)
        except OSError:
            # Another run finished the same download first.
            if (target / "meta.json").is_file():
                return target
            raise
        return target
    # OSError: an unwritable cache, a full disk, or a wheel without the model.
    # httpx.HTTPError: no network, a refused connection, or an error status.
    except (OSError, httpx.HTTPError, zipfile.BadZipFile) as exc:
        wheel_name = SPACY_MODEL_URL.rsplit("/", 1)[-1]
        raise SpacyModelUnavailableError(
            f"{_FAILURE}Joel needs {wheel_name} "
            f"({SPACY_MODEL_SIZE_BYTES / 1_000_000:.0f} MB) from {SPACY_MODEL_URL}, "
            f"and it is not in the cache at {cache}. Joel downloads it on first "
            f"use, so this machine needs to reach GitHub once, and the cache needs "
            f"to be writable. ({exc})"
        ) from exc
    finally:
        if wheel_path is not None:
            wheel_path.unlink(missing_ok=True)
        if extracted is not None:
            shutil.rmtree(extracted, ignore_errors=True)
