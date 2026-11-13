import json
import math
from pathlib import Path
from typing import Any

import pytest
from elevenlabs import SpeechToTextChunkResponseModel, SpeechToTextWordResponseModel

from joel.transcribers import TranscriberUnavailableError, load_transcriber
from joel.transcribers.elevenlabs import create, to_words

FIXTURES = Path(__file__).parent / "fixtures"
AUDIO = FIXTURES / "audio" / "acceptance.wav"
# Scribe v3's response for AUDIO, as recorded on 2026-11-13.
RECORDED = FIXTURES / "elevenlabs_response.json"


def load_response(path: Path) -> SpeechToTextChunkResponseModel:
    return SpeechToTextChunkResponseModel.model_validate_json(path.read_text())


def test_maps_a_recorded_response_to_words() -> None:
    data = json.loads(RECORDED.read_text())

    words = to_words(load_response(RECORDED))

    assert [w.text for w in words] == [
        e["text"] for e in data["words"] if e["type"] == "word"
    ]
    assert " ".join(w.text for w in words) == data["text"]
    assert [w.index for w in words] == list(range(len(words)))
    assert all(w.confidence is not None and 0 <= w.confidence <= 1 for w in words)
    # "school," runs from 1.34 s to 1.8 s with logprob -5.7812105556617356e-05.
    school = words[4]
    assert school.text == "school,"
    assert (school.start_ms, school.end_ms) == (1340, 1800)
    assert school.confidence == math.exp(-5.7812105556617356e-05)


def chunk(*words: SpeechToTextWordResponseModel) -> SpeechToTextChunkResponseModel:
    return SpeechToTextChunkResponseModel(
        language_code="eng",
        language_probability=1.0,
        text="".join(w.text for w in words),
        words=list(words),
    )


def test_times_round_to_the_nearest_millisecond() -> None:
    # In floating point 1.001 * 1000 is 1000.999… and 2.01 * 1000 is 2009.999…,
    # so truncating would put both a millisecond early.
    response = chunk(
        SpeechToTextWordResponseModel(
            text="go", type="word", start=1.001, end=2.01, logprob=-0.1
        )
    )

    word = to_words(response)[0]

    assert (word.start_ms, word.end_ms) == (1001, 2010)


def test_a_word_without_times_keeps_none() -> None:
    response = chunk(
        SpeechToTextWordResponseModel(text="go", type="word", logprob=-0.1)
    )

    word = to_words(response)[0]

    assert (word.start_ms, word.end_ms) == (None, None)


def test_all_zero_logprobs_mean_no_confidence() -> None:
    # The fixture's spacing entries carry a non-zero logprob: only words count.
    words = to_words(load_response(FIXTURES / "elevenlabs_zero_logprob.json"))

    assert [w.text for w in words] == ["I", "go"]
    assert all(w.confidence is None for w in words)


def test_a_zero_logprob_beside_non_zero_ones_is_full_confidence() -> None:
    response = chunk(
        SpeechToTextWordResponseModel(text="I", type="word", logprob=0.0),
        SpeechToTextWordResponseModel(text=" ", type="spacing", logprob=0.0),
        SpeechToTextWordResponseModel(text="go", type="word", logprob=-0.1),
    )

    words = to_words(response)

    assert [w.confidence for w in words] == [1.0, math.exp(-0.1)]


def test_asks_scribe_v3_for_word_timestamps_and_verbatim_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    transcriber = create()
    sent: dict[str, Any] = {}

    def fake(**kwargs: Any) -> SpeechToTextChunkResponseModel:
        sent.update(kwargs)
        return load_response(RECORDED)

    monkeypatch.setattr(transcriber._client.speech_to_text, "convert", fake)

    transcript = transcriber.transcribe(AUDIO)

    assert sent["model_id"] == "scribe_v3"
    assert sent["timestamps_granularity"] == "word"
    assert "no_verbatim" not in sent
    # The original file goes up as is; nothing is decoded first.
    assert sent["file"].name == str(AUDIO)
    assert transcript.transcriber == "elevenlabs"
    assert transcript.words == to_words(load_response(RECORDED))


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_missing_api_key_fails_before_any_request(
    value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ELEVENLABS_API_KEY", value)

    with pytest.raises(
        TranscriberUnavailableError, match="export ELEVENLABS_API_KEY=<your key>"
    ):
        load_transcriber("elevenlabs")
