import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import huggingface_hub
import huggingface_hub.constants
import pytest
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import _http

from joel.transcribers import TranscriberUnavailableError, _parakeet_model
from joel.transcribers._parakeet_model import (
    PARAKEET_FILE,
    PARAKEET_REPO,
    PARAKEET_REVISION,
    PARAKEET_SHA256,
    ModelUnavailableError,
    resolve_model,
)

OTHER_REVISION = "39eeb55181f0d354fd934f06e92fd8d5037fed8e"
OTHER_SHA256 = "0" * 64


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep huggingface_hub off the network, however resolve_model calls it."""
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "hub"


@pytest.fixture
def unreachable_hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No network at all, as on a laptop without Wi-Fi, and HF_HUB_OFFLINE unset."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr("huggingface_hub.utils._http.time.sleep", lambda _: None)
    huggingface_hub.set_client_factory(
        lambda: httpx.Client(transport=httpx.MockTransport(refuse))
    )
    yield
    huggingface_hub.set_client_factory(_http.default_client_factory)


def _cache_file(cache: Path, revision: str, blob_name: str) -> Path:
    """Lay out one cached copy of the model file the way huggingface_hub does."""
    repo = cache / "models--handy-computer--parakeet-ultra-gguf"
    blob = repo / "blobs" / blob_name
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"gguf")
    snapshot = repo / "snapshots" / revision / PARAKEET_FILE
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(Path("..", "..", "blobs", blob_name))
    return snapshot


def _forbid_network(*args: Any, **kwargs: Any) -> str:
    raise AssertionError("network")


def test_uses_the_pinned_revision_when_cached(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _cache_file(cache, PARAKEET_REVISION, PARAKEET_SHA256)
    monkeypatch.setattr(_parakeet_model, "hf_hub_download", _forbid_network)

    assert resolve_model(cache) == snapshot


def test_uses_the_same_file_under_another_revision(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _cache_file(cache, OTHER_REVISION, PARAKEET_SHA256)
    monkeypatch.setattr(_parakeet_model, "hf_hub_download", _forbid_network)

    assert resolve_model(cache) == snapshot


def test_downloads_the_pinned_revision_when_only_other_files_are_cached(
    cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cache_file(cache, OTHER_REVISION, OTHER_SHA256)
    downloaded = tmp_path / "downloaded.gguf"
    calls: list[dict[str, Any]] = []

    def fake_download(*args: Any, **kwargs: Any) -> str:
        bound = inspect.signature(huggingface_hub.hf_hub_download).bind(*args, **kwargs)
        calls.append(bound.arguments)
        return str(downloaded)

    monkeypatch.setattr(_parakeet_model, "hf_hub_download", fake_download)

    assert resolve_model(cache) == downloaded
    assert calls == [
        {
            "repo_id": PARAKEET_REPO,
            "filename": PARAKEET_FILE,
            "revision": PARAKEET_REVISION,
            "cache_dir": cache,
        }
    ]


def test_offline_without_the_model_explains_what_is_missing(cache: Path) -> None:
    # The real hf_hub_download, kept offline by the autouse fixture.
    with pytest.raises(ModelUnavailableError) as excinfo:
        resolve_model(cache)

    message = str(excinfo.value)
    assert PARAKEET_REPO in message
    assert PARAKEET_FILE in message
    assert "740 MB" in message
    assert str(cache) in message
    assert isinstance(excinfo.value, TranscriberUnavailableError)


_HUB_URL = httpx.Request("GET", "https://huggingface.co")


@pytest.mark.parametrize(
    "error",
    [
        # The Hub refuses: revision gone, repo gated, or a 5xx.
        HfHubHTTPError(
            "500 Server Error", response=httpx.Response(500, request=_HUB_URL)
        ),
        # The connection drops partway through the 740 MB download.
        httpx.RemoteProtocolError("peer closed connection", request=_HUB_URL),
        httpx.ProxyError("407 Proxy Authentication Required", request=_HUB_URL),
        # The cache directory isn't writable.
        PermissionError(13, "Permission denied", "/root/.cache/huggingface/hub"),
    ],
    ids=["hub-refuses", "connection-drops", "proxy-auth", "cache-not-writable"],
)
def test_failed_download_explains_what_is_missing(
    error: Exception, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_download(*args: Any, **kwargs: Any) -> str:
        raise error

    monkeypatch.setattr(_parakeet_model, "hf_hub_download", failing_download)

    with pytest.raises(ModelUnavailableError) as excinfo:
        resolve_model(cache)

    message = str(excinfo.value)
    assert PARAKEET_REPO in message
    assert PARAKEET_REVISION in message
    assert str(cache) in message
    assert str(error) in message


@pytest.mark.usefixtures("unreachable_hub")
def test_no_network_without_the_offline_flag_explains_what_is_missing(
    cache: Path,
) -> None:
    with pytest.raises(ModelUnavailableError) as excinfo:
        resolve_model(cache)

    message = str(excinfo.value)
    assert PARAKEET_REPO in message
    assert PARAKEET_FILE in message
    assert str(cache) in message
