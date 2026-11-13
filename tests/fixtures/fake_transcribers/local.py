from pathlib import Path

from joel.transcript import Transcript, Word


class LocalTranscriber:
    def transcribe(self, audio: Path) -> Transcript:
        word = Word(index=0, text=audio.stem, start_ms=0, end_ms=500, confidence=0.9)
        return Transcript(transcriber="local", words=[word])


def create() -> LocalTranscriber:
    return LocalTranscriber()
