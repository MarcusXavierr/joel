import hashlib
import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from joel import spacy_model
from joel.spacy_model import (
    SPACY_MODEL_VERSION,
    SpacyModelUnavailableError,
    resolve_spacy_model,
)

Handler = Callable[[httpx.Request], httpx.Response]


def fake_wheel(package: str) -> tuple[bytes, str]:
    """A wheel laid out like spaCy's model wheels, and its sha256."""
    model = f"{package}/{package}-{SPACY_MODEL_VERSION}"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        wheel.writestr(f"{package}/__init__.py", "")
        wheel.writestr(f"{model}/meta.json", "{}")
        wheel.writestr(f"{model}/vocab/strings.json", "[]")
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def client_for(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def files_in(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*") if p.is_file()]


@pytest.fixture
def not_installed(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pretend the model package isn't installed, as after `uv tool install joel`."""
    package = "joel_no_such_model"
    monkeypatch.setattr(spacy_model, "SPACY_MODEL_PACKAGE", package)
    return package


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "joel"


def test_downloads_the_pinned_model_once_and_reuses_it(
    not_installed: str, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel, sha256 = fake_wheel(not_installed)
    monkeypatch.setattr(spacy_model, "SPACY_MODEL_SHA256", sha256)
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=wheel)

    with client_for(serve) as client:
        first = resolve_spacy_model(cache, client=client)
        second = resolve_spacy_model(cache, client=client)

    target = cache / f"{not_installed}-{SPACY_MODEL_VERSION}"
    assert first == second == target
    assert [r.url for r in requests] == [httpx.URL(spacy_model.SPACY_MODEL_URL)]
    assert sorted(p.relative_to(cache) for p in files_in(cache)) == [
        target.relative_to(cache) / "meta.json",
        target.relative_to(cache) / "vocab" / "strings.json",
    ]


def test_refuses_a_download_whose_hash_does_not_match(
    not_installed: str, cache: Path
) -> None:
    wheel, _ = fake_wheel(not_installed)

    with (
        client_for(lambda request: httpx.Response(200, content=wheel)) as client,
        pytest.raises(SpacyModelUnavailableError) as excinfo,
    ):
        resolve_spacy_model(cache, client=client)

    assert spacy_model.SPACY_MODEL_SHA256 in str(excinfo.value)
    assert files_in(cache) == []


def _not_found(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404)


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route to host", request=request)


@pytest.mark.parametrize("handler", [_not_found, _refuse], ids=["404", "no-network"])
@pytest.mark.usefixtures("not_installed")
def test_failed_download_explains_what_is_missing(
    handler: Handler, cache: Path
) -> None:
    with (
        client_for(handler) as client,
        pytest.raises(SpacyModelUnavailableError) as excinfo,
    ):
        resolve_spacy_model(cache, client=client)

    message = str(excinfo.value)
    assert message.startswith("Could not get spaCy's en_core_web_sm model. ")
    assert spacy_model.SPACY_MODEL_URL in message
    assert str(cache) in message
    assert files_in(cache) == []


def test_uses_the_installed_package_without_downloading(tmp_path: Path) -> None:
    def forbid(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network")

    with client_for(forbid) as client:
        model = resolve_spacy_model(tmp_path, client=client)

    assert model.name == f"en_core_web_sm-{SPACY_MODEL_VERSION}"
    assert (model / "meta.json").is_file()
