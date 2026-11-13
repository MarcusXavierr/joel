"""Cloud transcriber: ElevenLabs Scribe v3, through the official SDK."""

import math
from pathlib import Path

from elevenlabs import SpeechToTextChunkResponseModel
from elevenlabs.client import ElevenLabs

from joel.transcribers._api_key import require_api_key
from joel.transcript import Transcript, Word

MODEL_ID = "scribe_v3"


def to_words(response: SpeechToTextChunkResponseModel) -> list[Word]:
    """One joel Word per spoken word; spacing and sound entries are dropped.

    Confidence is exp(logprob). When every word's logprob is 0, the response
    carries no confidence at all, so every word gets None.
    """
    entries = [e for e in response.words if e.type == "word"]
    has_confidence = any(e.logprob != 0.0 for e in entries)
    return [
        Word(
            index=index,
            text=entry.text,
            start_ms=_ms(entry.start),
            end_ms=_ms(entry.end),
            # min(): a float logprob a hair above 0 would break Word's 0..1 bound.
            confidence=min(1.0, math.exp(entry.logprob)) if has_confidence else None,
        )
        for index, entry in enumerate(entries)
    ]


def _ms(seconds: float | None) -> int | None:
    # round, not int: 1.001 s times 1000 is 1000.999... in floating point.
    return None if seconds is None else round(seconds * 1000)


class ElevenLabsTranscriber:
    """Uploads the original audio file to Scribe v3; nothing is decoded here."""

    def __init__(self, client: ElevenLabs) -> None:
        self._client = client

    def request(self, audio: Path) -> SpeechToTextChunkResponseModel:
        """Send ``audio`` to Scribe v3 and return its response unchanged."""
        with audio.open("rb") as f:
            response = self._client.speech_to_text.convert(
                model_id=MODEL_ID, file=f, timestamps_granularity="word"
            )
        if not isinstance(response, SpeechToTextChunkResponseModel):
            kind = type(response).__name__
            raise RuntimeError(f"ElevenLabs returned {kind}, not a single transcript")
        return response

    def transcribe(self, audio: Path) -> Transcript:
        """Transcribe ``audio`` with Scribe v3, one Word per spoken word."""
        return Transcript(transcriber="elevenlabs", words=to_words(self.request(audio)))


def create() -> ElevenLabsTranscriber:
    """Build the adapter; raises TranscriberUnavailableError without an API key."""
    return ElevenLabsTranscriber(
        ElevenLabs(api_key=require_api_key("ELEVENLABS_API_KEY", "elevenlabs"))
    )
