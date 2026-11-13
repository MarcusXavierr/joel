import sys
from pathlib import Path

import pytest

from joel.transcribers import TranscriberUnavailableError, load_transcriber


@pytest.fixture
def fake_package(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "fixtures"))
    return "fake_transcribers"


def test_loads_adapter_by_module_name(fake_package: str) -> None:
    transcriber = load_transcriber("local", package=fake_package)

    assert transcriber.transcribe(Path("hello.wav")).words[0].text == "hello"


def test_unknown_name_lists_valid_names(fake_package: str) -> None:
    with pytest.raises(
        TranscriberUnavailableError,
        match=r"Unknown transcriber 'whisper'\. Valid names: broken, local, openai\.",
    ):
        load_transcriber("whisper", package=fake_package)


def test_missing_extra_names_the_extra(
    fake_package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry in sys.modules makes `import openai` fail as if the
    # SDK of joel's "openai" extra were not installed.
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(TranscriberUnavailableError) as excinfo:
        load_transcriber("openai", package=fake_package)

    message = str(excinfo.value)
    assert "joel[openai]" in message
    assert "uv sync --extra openai" in message


def test_missing_dependency_outside_an_extra_is_not_blamed_on_an_extra(
    fake_package: str,
) -> None:
    with pytest.raises(ModuleNotFoundError):
        load_transcriber("broken", package=fake_package)


def test_missing_module_inside_an_extra_adapter_is_not_blamed_on_the_extra(
    fake_package: str,
) -> None:
    with pytest.raises(ModuleNotFoundError, match="joel_fake_helper_not_installed"):
        load_transcriber("openai", package=fake_package)


def test_unknown_name_in_real_package() -> None:
    with pytest.raises(
        TranscriberUnavailableError, match=r"Unknown transcriber 'whisper'"
    ):
        load_transcriber("whisper")
