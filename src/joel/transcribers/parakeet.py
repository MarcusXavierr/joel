"""The default transcriber: Parakeet Ultra on this machine, through transcribe-cpp."""

import logging
from collections.abc import Sequence
from pathlib import Path

import transcribe_cpp

from joel.decode import decode
from joel.transcribers._parakeet_model import resolve_model
from joel.transcript import Transcript, Word

_NATIVE_LOGGER = logging.getLogger("joel.transcribe_cpp")
# transcribe-cpp's TRANSCRIBE_LOG_LEVEL_* values; the library exports no constants.
_NATIVE_LEVELS = {
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.DEBUG,
}


def forward_native_log(level: int, message: str) -> None:
    """Hand one transcribe-cpp log line to Python logging."""
    # An unknown level becomes WARNING so it isn't hidden.
    _NATIVE_LOGGER.log(_NATIVE_LEVELS.get(level, logging.WARNING), message.rstrip("\n"))


# Once, before any model loads, as transcribe-cpp requires. The lines ggml prints
# while `transcribe_cpp` itself is imported come before this and still hit stderr.
transcribe_cpp.set_log_callback(forward_native_log)


def to_words(
    words: Sequence[transcribe_cpp.Word], tokens: Sequence[transcribe_cpp.Token]
) -> list[Word]:
    """One joel Word per library word; confidence is the lowest p among its tokens."""
    return [
        Word(
            index=index,
            text=word.text,
            start_ms=word.t0_ms,
            end_ms=word.t1_ms,
            confidence=_lowest_p(word, tokens),
        )
        for index, word in enumerate(words)
    ]


def _lowest_p(
    word: transcribe_cpp.Word, tokens: Sequence[transcribe_cpp.Token]
) -> float | None:
    """A word is only as sure as its least sure token; None when it has none."""
    own_tokens = tokens[word.first_token : word.first_token + word.n_tokens]
    return min((token.p for token in own_tokens), default=None)


class ParakeetTranscriber:
    """Transcribes with Parakeet Ultra, loading the model once on first use."""

    def __init__(self) -> None:
        self._model: transcribe_cpp.Model | None = None

    def transcribe(self, audio: Path) -> Transcript:
        """Transcribe ``audio``, getting the model on the first call.

        Raises DecodeError for a file ffmpeg can't read and ModelUnavailableError
        when the model is neither cached nor downloadable.
        """
        # Decode first: a bad file should fail before a 740 MB download.
        pcm = decode(audio)
        if self._model is None:
            self._model = transcribe_cpp.Model(resolve_model())
        with self._model.session() as session:
            result = session.run(pcm)
        return Transcript(
            transcriber="parakeet", words=to_words(result.words, result.tokens)
        )


def create() -> ParakeetTranscriber:
    """Build the adapter. It touches neither ffmpeg nor the model until used."""
    return ParakeetTranscriber()
