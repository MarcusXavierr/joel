"""The whole pipeline: a recording in; every mistake and what finding it cost out."""

import asyncio
from collections.abc import Awaitable
from decimal import Decimal
from pathlib import Path

from errant.annotator import Annotator  # type: ignore[import-untyped]
from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from joel.confidence import FilteredEdit, filter_edits
from joel.correct import Correction, correct
from joel.cost import PriceUnavailableError, price
from joel.disfluencies import strip_disfluencies
from joel.edits import diff_edits
from joel.sentences import split_sentences
from joel.transcribers import Transcriber
from joel.transcript import Transcript

MAX_CONCURRENT_CALLS = 8


class CheckedSentence(BaseModel):
    """One sentence and everything the pipeline decided about it."""

    words: list[int]  # Word.index of every word split_sentences put in it
    dropped: list[int]  # Word.index of every disfluency, sorted
    correction: Correction | None  # None when every word was dropped; no call made
    edits: list[FilteredEdit]  # in diff_edits order


class CheckResult(BaseModel):
    """Everything one run decided. The terminal report is rebuilt from this alone."""

    model: str  # the model string the run was given
    transcript: Transcript  # as the transcriber returned it
    sentences: list[CheckedSentence]
    requests: int  # model calls, retries included
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None  # None when genai-prices can't price `model`


async def run_check(
    audio: Path,
    transcriber: Transcriber,
    *,
    model: Model,
    model_name: str,
    min_confidence: float,
    annotator: Annotator,
) -> CheckResult:
    """Transcribe `audio` and run every later step over it.

    The model calls run concurrently, at most MAX_CONCURRENT_CALLS at a time:
    every cleanup first, then every correction. The first call that fails
    raises its error as is. `model_name` is what gets priced and recorded.
    """
    transcript = transcriber.transcribe(audio)
    by_index = {w.index: w for w in transcript.words}
    sentences = split_sentences(transcript.words)
    usage = RunUsage()
    limit = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    async def limited[T](call: Awaitable[T]) -> T:
        async with limit:
            return await call

    cleaned = await asyncio.gather(
        *(
            limited(
                strip_disfluencies(
                    [by_index[i] for i in sentence], model=model, usage=usage
                )
            )
            for sentence in sentences
        )
    )
    texts = [" ".join(w.text for w in c.words) for c in cleaned]
    spoken = [n for n, text in enumerate(texts) if text]
    answers = await asyncio.gather(
        *(
            limited(
                correct(
                    texts[n],
                    previous=texts[spoken[k - 1]] if k > 0 else None,
                    following=texts[spoken[k + 1]] if k + 1 < len(spoken) else None,
                    model=model,
                    usage=usage,
                )
            )
            for k, n in enumerate(spoken)
        )
    )
    corrections = dict(zip(spoken, answers, strict=True))

    checked = []
    for n, (sentence, clean) in enumerate(zip(sentences, cleaned, strict=True)):
        correction = corrections.get(n)
        edits = (
            diff_edits(
                clean.words, correction.grammatical, correction.natural, annotator
            )
            if correction is not None
            else []
        )
        checked.append(
            CheckedSentence(
                words=sentence,
                dropped=clean.dropped,
                correction=correction,
                edits=filter_edits(edits, transcript.words, min_confidence),
            )
        )

    try:
        cost: Decimal | None = price(
            model_name,
            usage.input_tokens,
            usage.output_tokens,
            requests=usage.requests,
        )
    except PriceUnavailableError:
        cost = None
    return CheckResult(
        model=model_name,
        transcript=transcript,
        sentences=checked,
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
    )
