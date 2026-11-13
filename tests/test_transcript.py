import pytest
from pydantic import ValidationError

from joel.transcript import Transcript, Word


def test_word_accepts_none_for_times_and_confidence() -> None:
    word = Word(index=0, text="uh,", start_ms=None, end_ms=None, confidence=None)

    assert word.start_ms is None
    assert word.end_ms is None
    assert word.confidence is None


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_word_rejects_confidence_outside_zero_to_one(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Word(index=0, text="go", start_ms=0, end_ms=100, confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_word_accepts_confidence_bounds(confidence: float) -> None:
    word = Word(index=0, text="go", start_ms=0, end_ms=100, confidence=confidence)

    assert word.confidence == confidence


def test_transcript_survives_json_round_trip() -> None:
    transcript = Transcript(
        transcriber="parakeet",
        words=[
            Word(index=0, text="Yesterday", start_ms=120, end_ms=640, confidence=0.98),
            Word(index=1, text="he", start_ms=None, end_ms=None, confidence=None),
        ],
    )

    assert Transcript.model_validate_json(transcript.model_dump_json()) == transcript
