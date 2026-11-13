"""Cloud transcriber: OpenAI gpt-transcribe, through the official SDK.

It returns no word timestamps (ADR-0007).
"""

import logging
import math
import re
from pathlib import Path

from openai import OpenAI
from openai.types.audio import Transcription

from joel.transcribers._api_key import require_api_key
from joel.transcript import Transcript, Word

MODEL = "gpt-transcribe"
# The API's 25 MB cap, in bytes.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_LOGGER = logging.getLogger(__name__)


class AudioTooLargeError(Exception):
    """An audio file over gpt-transcribe's upload limit."""


def to_words(response: Transcription) -> list[Word]:
    """Split the text on whitespace; confidence is the word's least sure token.

    A token belongs to every word its characters overlap, so a token that spans
    a word boundary counts for both words. Without token logprobs, or when the
    tokens don't spell the text, every word's confidence is None.
    """
    matches = list(re.finditer(r"\S+", response.text))
    spans = _token_spans(response)
    if spans is None:
        return [_word(i, m.group(), None) for i, m in enumerate(matches)]
    words = []
    first = 0  # first token that may still overlap the current word
    for index, match in enumerate(matches):
        while first < len(spans) and spans[first][1] <= match.start():
            first += 1
        probabilities = []
        # By index, not spans[first:]: a slice would copy the tail for every word.
        for position in range(first, len(spans)):
            start, end, probability = spans[position]
            if start >= match.end():
                break
            if end > match.start():
                probabilities.append(probability)
        words.append(_word(index, match.group(), min(probabilities, default=None)))
    return words


def _word(index: int, text: str, confidence: float | None) -> Word:
    return Word(
        index=index, text=text, start_ms=None, end_ms=None, confidence=confidence
    )


def _token_spans(response: Transcription) -> list[tuple[int, int, float]] | None:
    """Each token with a logprob as (start, end, probability) in the text."""
    if not response.logprobs:
        return None
    joined = "".join(t.token or "" for t in response.logprobs)
    offset = joined.find(response.text)
    if offset == -1:
        _LOGGER.warning(
            "gpt-transcribe tokens don't spell its text; confidence left out"
        )
        return None
    spans = []
    pos = 0
    for token in response.logprobs:
        length = len(token.token or "")
        if token.logprob is not None:
            start = pos - offset
            # min(): a float logprob a hair above 0 would break Word's 0..1 bound.
            spans.append((start, start + length, min(1.0, math.exp(token.logprob))))
        pos += length
    return spans


class OpenAITranscriber:
    """Uploads the original audio file to gpt-transcribe; nothing is decoded here."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def request(self, audio: Path) -> Transcription:
        """Send ``audio`` to gpt-transcribe and return its response unchanged.

        Raises AudioTooLargeError before uploading a file over the API's limit.
        """
        size = audio.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            # Bytes, not MB: one byte over would round to "25.0 MB".
            raise AudioTooLargeError(
                f"{audio} is {size:,} bytes, over the 25 MB upload limit "
                f"({MAX_UPLOAD_BYTES:,} bytes) of the 'openai' transcriber. Use a "
                "shorter or more compressed recording, or another transcriber "
                "(--transcriber parakeet or elevenlabs)."
            )
        with audio.open("rb") as f:
            # No prompt: asking for a literal transcript didn't bring back a
            # dropped repeated word when tested.
            return self._client.audio.transcriptions.create(
                model=MODEL, file=f, include=["logprobs"], response_format="json"
            )

    def transcribe(self, audio: Path) -> Transcript:
        """Transcribe ``audio`` with gpt-transcribe; its words carry no times."""
        return Transcript(transcriber="openai", words=to_words(self.request(audio)))


def create() -> OpenAITranscriber:
    """Build the adapter; raises TranscriberUnavailableError without an API key."""
    return OpenAITranscriber(
        OpenAI(api_key=require_api_key("OPENAI_API_KEY", "openai"))
    )
