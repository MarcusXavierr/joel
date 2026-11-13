import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest
from openai.types.audio import Transcription
from openai.types.audio.transcription import Logprob

from joel.transcribers import TranscriberUnavailableError, load_transcriber
from joel.transcribers.openai import (
    MAX_UPLOAD_BYTES,
    AudioTooLargeError,
    create,
    to_words,
)

FIXTURES = Path(__file__).parent / "fixtures"
# gpt-transcribe's response for tests/fixtures/audio/acceptance.wav, as recorded
# on 2026-11-13.
RECORDED = FIXTURES / "openai_response.json"


def load_response() -> Transcription:
    return Transcription.model_validate_json(RECORDED.read_text())


def sparse_file(path: Path, size: int) -> Path:
    with path.open("wb") as f:
        f.truncate(size)
    return path


def test_maps_a_recorded_response_to_words() -> None:
    data = json.loads(RECORDED.read_text())

    words = to_words(load_response())

    assert [w.text for w in words] == data["text"].split()
    assert all(w.start_ms is None and w.end_ms is None for w in words)
    assert all(w.confidence is not None and 0 <= w.confidence <= 1 for w in words)
    # "school," is " school" (logprob 0.0) and "," (logprob -7.62939453125e-06):
    # the comma is its least sure token.
    school = words[4]
    assert school.text == "school,"
    assert school.confidence == math.exp(-7.62939453125e-06)


def test_word_confidence_is_its_least_sure_token() -> None:
    tokens = [
        ("Yes", -0.01),
        (",", -0.5),
        (" today", -0.02),
        (" he", -0.03),
        (" go", -0.2),
        (".", -0.1),
    ]
    response = Transcription(
        text="Yes, today he go.",
        logprobs=[Logprob(token=t, logprob=p) for t, p in tokens],
    )

    words = to_words(response)

    assert [w.text for w in words] == ["Yes,", "today", "he", "go."]
    assert words[0].confidence == math.exp(-0.5)
    assert words[1].confidence == math.exp(-0.02)
    assert words[3].confidence == math.exp(-0.2)


def transcription(text: str, *tokens: tuple[str, float | None]) -> Transcription:
    return Transcription(
        text=text, logprobs=[Logprob(token=t, logprob=p) for t, p in tokens]
    )


def test_a_token_across_a_word_boundary_counts_for_both_words() -> None:
    response = transcription("he go", ("h", -0.01), ("e g", -0.4), ("o", -0.02))

    words = to_words(response)

    assert [w.confidence for w in words] == [math.exp(-0.4), math.exp(-0.4)]


def test_tokens_before_the_text_belong_to_no_word() -> None:
    response = transcription("Yes go", (" ", -2.0), ("Yes", -0.3), (" go", -0.01))

    words = to_words(response)

    assert [w.confidence for w in words] == [math.exp(-0.3), math.exp(-0.01)]


def test_a_token_without_logprob_still_takes_up_its_characters() -> None:
    response = transcription("Yes he go", ("Yes", None), (" he", -0.2), (" go", -0.05))

    words = to_words(response)

    assert [w.confidence for w in words] == [None, math.exp(-0.2), math.exp(-0.05)]


@pytest.mark.parametrize("logprobs", [None, []], ids=["none", "empty"])
def test_no_logprobs_means_no_confidence_and_no_warning(
    logprobs: list[Logprob] | None, caplog: pytest.LogCaptureFixture
) -> None:
    words = to_words(Transcription(text="Yes, today", logprobs=logprobs))

    assert [(w.text, w.confidence) for w in words] == [("Yes,", None), ("today", None)]
    assert caplog.records == []


def test_tokens_that_do_not_spell_the_text_give_no_confidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = Transcription(
        text="Yes, today he go.",
        logprobs=[Logprob(token="Hello", logprob=-0.1), Logprob(token=" world")],
    )

    words = to_words(response)

    assert [w.text for w in words] == ["Yes,", "today", "he", "go."]
    assert all(w.confidence is None for w in words)
    warnings = [r for r in caplog.records if r.name == "joel.transcribers.openai"]
    assert [r.levelno for r in warnings] == [logging.WARNING]


def test_audio_over_25_mb_fails_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transcriber = create()

    def forbid(**kwargs: Any) -> Transcription:
        raise AssertionError("network call")

    monkeypatch.setattr(transcriber._client.audio.transcriptions, "create", forbid)
    audio = sparse_file(tmp_path / "long.wav", MAX_UPLOAD_BYTES + 1)

    with pytest.raises(AudioTooLargeError) as raised:
        transcriber.transcribe(audio)

    message = str(raised.value)
    # One byte over must not read as "25.0 MB, over the 25 MB limit".
    assert "26,214,401 bytes" in message
    assert "25 MB upload limit" in message
    assert "Use a shorter or more compressed recording" in message


def test_asks_gpt_transcribe_for_token_logprobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transcriber = create()
    sent: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Transcription:
        sent.update(kwargs)
        return load_response()

    monkeypatch.setattr(transcriber._client.audio.transcriptions, "create", fake)
    # Exactly at the limit is still allowed.
    audio = sparse_file(tmp_path / "talk.wav", MAX_UPLOAD_BYTES)

    transcript = transcriber.transcribe(audio)

    assert sent["model"] == "gpt-transcribe"
    assert sent["include"] == ["logprobs"]
    assert "prompt" not in sent
    # The original file goes up as is; nothing is decoded first.
    assert sent["file"].name == str(audio)
    assert transcript.transcriber == "openai"
    assert transcript.words == to_words(load_response())


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_missing_api_key_fails_before_any_request(
    value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)

    with pytest.raises(
        TranscriberUnavailableError, match="export OPENAI_API_KEY=<your key>"
    ):
        load_transcriber("openai")
