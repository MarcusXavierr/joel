"""Step 3 of the pipeline: drop filled pauses, repetitions and false starts."""

import json
from collections.abc import Sequence

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from joel.llm import DEFAULT_MODEL, ask
from joel.transcript import Word

INSTRUCTIONS = (
    "You clean up transcribed speech from a non-native English speaker before a "
    "grammar check. You receive one sentence as a JSON list of [position, word] "
    "pairs. Answer with the positions of the words to drop. Drop only: filled "
    "pauses (uh, um, er, erm, hmm, and the like); repetitions that add nothing, "
    'such as the first "the" in "the the"; false starts, words the speaker '
    'abandoned and restarted, such as "I went," in "I went, I was going". Never '
    "drop a word because removing it makes the sentence grammatical. The speaker "
    'needs to see their grammar mistakes: in "I am agree", "am" is a mistake, not '
    "a false start, so keep it. When unsure, keep the word. If nothing should go, "
    "answer with an empty list. Use only positions from the input."
)


class CleanSentence(BaseModel):
    """A sentence with its disfluencies removed."""

    words: list[Word]  # the original words that remain, in spoken order
    dropped: list[int]  # Word.index of every dropped word, sorted


class _Drop(BaseModel):
    drop: list[int]


# The deps are the positions the sentence holds.
_agent = Agent(
    output_type=_Drop,
    deps_type=frozenset[int],
    instructions=INSTRUCTIONS,
    retries=1,
)


@_agent.output_validator
def _only_sentence_positions(ctx: RunContext[frozenset[int]], answer: _Drop) -> _Drop:
    bad = sorted(set(answer.drop) - ctx.deps)
    if bad:
        raise ModelRetry(
            f"Positions {bad} are not in the sentence. "
            f"Use only these positions: {sorted(ctx.deps)}."
        )
    return answer


async def strip_disfluencies(
    sentence: Sequence[Word],
    *,
    model: Model | str = DEFAULT_MODEL,
    usage: RunUsage | None = None,
) -> CleanSentence:
    """Ask the model which words are disfluencies and return the rest."""
    answer = await ask(
        _agent,
        json.dumps([[word.index, word.text] for word in sentence]),
        deps=frozenset(word.index for word in sentence),
        model=model,
        usage=usage,
        sentence=" ".join(word.text for word in sentence),
    )
    dropped = set(answer.drop)
    return CleanSentence(
        words=[word for word in sentence if word.index not in dropped],
        dropped=sorted(dropped),
    )
