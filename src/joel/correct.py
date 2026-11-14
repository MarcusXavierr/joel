"""Step 4 of the pipeline: each sentence made grammatical, and natural on request."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from joel.llm import DEFAULT_MODEL, ModelAnswerError, ask

_RETRIES = 1

# Looks at a sentence's natural version next to its grammatical one; returns
# what is wrong with it, or None.
Review = Callable[[str, str], str | None]


class Correction(BaseModel):
    """The corrected versions of one sentence."""

    grammatical: str  # the smallest edit that makes it grammatical
    # grammatical, plus the words a native speaker would swap; None when the
    # run didn't ask for it
    natural: str | None


class _Numbered(BaseModel):
    number: int
    grammatical: str


class _NumberedNatural(_Numbered):
    natural: str


class _Block(BaseModel):
    sentences: list[_Numbered]


class _NaturalBlock(BaseModel):
    sentences: list[_NumberedNatural]


@dataclass
class _NaturalDeps:
    count: int  # how many sentences the block holds
    review: Review
    # The first numbered answer with its rejected natural versions dropped,
    # kept in case the retry they caused fails.
    fallback: _NaturalBlock | None = None


# One agent per output schema, so a run that doesn't ask for the natural
# version never shows the model a field for it. The grammar agent's deps are
# how many sentences the block holds.
_grammar_agent = Agent(output_type=_Block, deps_type=int, retries=_RETRIES)
_natural_agent = Agent(
    output_type=_NaturalBlock, deps_type=_NaturalDeps, retries=_RETRIES
)


def _check_numbers(n: int, numbers: list[int]) -> None:
    if numbers != list(range(1, n + 1)):
        raise ModelRetry(
            f"Answer with exactly {n} items, numbered 1 to {n} in order, "
            "one per sentence to correct."
        )


@_grammar_agent.output_validator
def _one_item_per_sentence(ctx: RunContext[int], answer: _Block) -> _Block:
    _check_numbers(ctx.deps, [s.number for s in answer.sentences])
    return answer


@_natural_agent.output_validator
def _natural_only_swaps_words(
    ctx: RunContext[_NaturalDeps], answer: _NaturalBlock
) -> _NaturalBlock:
    """Send a rejected natural version back once; after that, drop it.

    A run never fails over naturalness: a natural version still rejected on
    the last try becomes its grammatical one, and `correct` falls back to the
    first answer if the retry fails for another reason.
    """
    _check_numbers(ctx.deps.count, [s.number for s in answer.sentences])
    problems = [
        (k, problem)
        for k, s in enumerate(answer.sentences)
        if (problem := ctx.deps.review(s.grammatical, s.natural))
    ]
    if not problems:
        return answer
    kept = answer.model_copy(deep=True)
    for k, _ in problems:
        sentence = kept.sentences[k]
        sentence.natural = sentence.grammatical
    if ctx.retry < ctx.max_retries:
        ctx.deps.fallback = kept
        raise ModelRetry(
            "Fix natural in these sentences and change nothing else:\n"
            + "\n".join(f"Sentence {k + 1}: {problem}" for k, problem in problems)
        )
    return kept


async def correct(
    sentences: Sequence[str],
    *,
    previous: str | None,
    following: str | None,
    instructions: str,
    review: Review | None = None,
    model: Model | str = DEFAULT_MODEL,
    usage: RunUsage | None = None,
) -> list[Correction]:
    """Correct a block of clean sentences, reading its neighbors only as context.

    Only with a `review` does the model also write a natural version, which
    the review can send back once.
    """
    if not sentences:
        return []
    lines = [
        f"Previous sentence: {previous}" if previous is not None else None,
        "Sentences to correct:",
        *(f"{n}. {sentence}" for n, sentence in enumerate(sentences, 1)),
        f"Next sentence: {following}" if following is not None else None,
    ]
    prompt = "\n".join(line for line in lines if line is not None)
    text = "\n".join(sentences)
    if review is None:
        answer = await ask(
            _grammar_agent,
            prompt,
            instructions=instructions,
            deps=len(sentences),
            model=model,
            usage=usage,
            text=text,
        )
        return [
            Correction(grammatical=s.grammatical, natural=None)
            for s in answer.sentences
        ]
    deps = _NaturalDeps(len(sentences), review)
    try:
        natural = await ask(
            _natural_agent,
            prompt,
            instructions=instructions,
            deps=deps,
            model=model,
            usage=usage,
            text=text,
        )
    except ModelAnswerError:
        if deps.fallback is None:
            raise
        natural = deps.fallback
    return [
        Correction(grammatical=s.grammatical, natural=s.natural)
        for s in natural.sentences
    ]
