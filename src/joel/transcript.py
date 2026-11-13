"""The word format every transcriber produces and every later step reads."""

from typing import Annotated

from pydantic import BaseModel, Field


class Word(BaseModel):
    """One word as the transcriber heard it."""

    index: int  # position in the raw transcript; never changes
    text: str  # as the ASR wrote it, punctuation attached
    start_ms: int | None  # None when the provider gives no word timestamps (ADR-0007)
    end_ms: int | None
    # None when the provider doesn't supply it
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None


class Transcript(BaseModel):
    """Everything one transcriber heard in one audio file, in spoken order."""

    transcriber: str
    words: list[Word]
