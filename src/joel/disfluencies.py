"""Step 3 of the pipeline: drop filled pauses, repetitions and false starts."""

import json
from collections.abc import Sequence

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from joel.llm import DEFAULT_MODEL, ask
from joel.transcript import Word


class CleanSentence(BaseModel):
    """A sentence with its disfluencies removed."""

    words: list[Word]  # the original words that remain, in spoken order
    dropped: list[int]  # Word.index of every dropped word, sorted


class _Drop(BaseModel):
    drop: list[int]


# The deps are the positions the passage holds.
_agent = Agent(
    output_type=_Drop,
    deps_type=frozenset[int],
    retries=1,
)


@_agent.output_validator
def _only_passage_positions(ctx: RunContext[frozenset[int]], answer: _Drop) -> _Drop:
    bad = sorted(set(answer.drop) - ctx.deps)
    if bad:
        raise ModelRetry(
            f"Positions {bad} are not in the passage. "
            f"Use only these positions: {sorted(ctx.deps)}."
        )
    return answer


async def strip_disfluencies(
    passage: Sequence[Word],
    *,
    instructions: str,
    model: Model | str = DEFAULT_MODEL,
    usage: RunUsage | None = None,
) -> CleanSentence:
    """Ask the model which words of a passage are disfluencies; return the rest."""
    answer = await ask(
        _agent,
        json.dumps([[word.index, word.text] for word in passage]),
        instructions=instructions,
        deps=frozenset(word.index for word in passage),
        model=model,
        usage=usage,
        text=" ".join(word.text for word in passage),
    )
    dropped = set(answer.drop)
    return CleanSentence(
        words=[word for word in passage if word.index not in dropped],
        dropped=sorted(dropped),
    )
