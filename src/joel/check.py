"""The whole pipeline: a recording in; every mistake and what finding it cost out."""

import asyncio
import functools
import itertools
from collections.abc import Awaitable, Callable, Sequence
from decimal import Decimal
from pathlib import Path

from errant.annotator import Annotator  # type: ignore[import-untyped]
from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from joel.confidence import FilteredEdit, filter_edits
from joel.correct import Correction, correct
from joel.cost import PriceUnavailableError, price
from joel.disfluencies import CleanSentence, strip_disfluencies
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
    cache_read_tokens: int  # part of input_tokens, read from the prompt cache
    cache_write_tokens: int  # part of input_tokens, written to the prompt cache
    cost_usd: Decimal | None  # None when genai-prices can't price `model`


async def run_check(
    audio: Path,
    transcriber: Transcriber,
    *,
    model: Model,
    model_name: str,
    min_confidence: float,
    annotator: Annotator,
    block_sentences: int,
) -> CheckResult:
    """Transcribe `audio` and run every later step over it.

    The model calls go over up to `block_sentences` sentences each: every
    cleanup first, then every correction. Each step runs its first block alone,
    so the rest read the prompt cache it writes; then at most
    MAX_CONCURRENT_CALLS at a time. The first call that fails raises its error
    as is. `model_name` is what gets priced and recorded.
    """
    transcript = transcriber.transcribe(audio)
    by_index = {w.index: w for w in transcript.words}
    sentences = split_sentences(transcript.words)
    usage = RunUsage()
    limit = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    # Each call is created only once it holds the semaphore, so a call still
    # queued when another fails is cancelled cleanly instead of never awaited.
    async def cleanup(block: tuple[list[int], ...]) -> list[CleanSentence]:
        async with limit:
            clean = await strip_disfluencies(
                [by_index[i] for sentence in block for i in sentence],
                model=model,
                usage=usage,
            )
        dropped = set(clean.dropped)
        return [
            CleanSentence(
                words=[by_index[i] for i in sentence if i not in dropped],
                dropped=sorted(dropped & set(sentence)),
            )
            for sentence in block
        ]

    blocks = await _first_alone(
        [
            functools.partial(cleanup, block)
            for block in itertools.batched(sentences, block_sentences, strict=False)
        ]
    )
    cleaned = [clean for block in blocks for clean in block]
    texts = [" ".join(w.text for w in c.words) for c in cleaned]
    spoken = [n for n, text in enumerate(texts) if text]

    async def fix(ks: tuple[int, ...]) -> list[Correction]:
        async with limit:
            return await correct(
                [texts[spoken[k]] for k in ks],
                previous=texts[spoken[ks[0] - 1]] if ks[0] > 0 else None,
                following=(
                    texts[spoken[ks[-1] + 1]] if ks[-1] + 1 < len(spoken) else None
                ),
                model=model,
                usage=usage,
            )

    positions = range(len(spoken))
    answers = await _first_alone(
        [
            functools.partial(fix, ks)
            for ks in itertools.batched(positions, block_sentences, strict=False)
        ]
    )
    corrections = dict(
        zip(spoken, (c for block in answers for c in block), strict=True)
    )

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
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
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
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cost_usd=cost,
    )


async def _first_alone[T](calls: Sequence[Callable[[], Awaitable[T]]]) -> list[T]:
    """Run the first call by itself, then the rest together, keeping their order.

    The first call writes the step's instructions to the provider's prompt
    cache; calls started together with it would all miss and pay the write.
    """
    if not calls:
        return []
    first = await calls[0]()
    return [first, *await asyncio.gather(*(call() for call in calls[1:]))]
