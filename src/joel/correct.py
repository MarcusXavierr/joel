"""Step 4 of the pipeline: a grammatical and a natural version of each sentence."""

from collections.abc import Sequence

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from joel.llm import DEFAULT_MODEL, ask

INSTRUCTIONS = (
    "You correct transcribed speech from a non-native English speaker. You "
    "receive numbered sentences to correct, and the sentence before and after "
    "them when they exist. Those two neighbors are context only, for tense and "
    "reference; never correct them. Correct each numbered sentence on its own "
    "and never move words between sentences, but read the others as context. "
    "Answer with one item per numbered sentence, in order, keeping its number, "
    "each with two versions of that sentence. "
    "grammatical: the smallest edit that makes the sentence grammatical; change "
    "only words that are wrong; if the sentence is already grammatical, return it "
    "unchanged. natural: the smallest further edit to grammatical that a native "
    'speaker would make because the wording sounds odd, such as "I make a party" '
    'to "I threw a party"; if nothing sounds odd, return grammatical unchanged. '
    "Edit as little as possible. Do not rephrase, reorder, or improve style. Keep "
    "the speaker's words, word order, register and spoken style. Keep punctuation "
    "and capitalization exactly as given: they come from the transcriber, not the "
    "speaker."
    " Example. Previous sentence: I lived in Recife. Sentences to correct: "
    "1. Yesterday he go to school, and she don't like it. 2. There is many "
    "people in the room. 3. I make a party last week. Next sentence: It was "
    'fun. Answer: {"sentences": [{"number": 1, "grammatical": "Yesterday he '
    'went to school, and she didn\'t like it.", "natural": "Yesterday he '
    'went to school, and she didn\'t like it."}, {"number": 2, '
    '"grammatical": "There are many people in the room.", "natural": "There '
    'are many people in the room."}, {"number": 3, "grammatical": "I made a '
    'party last week.", "natural": "I threw a party last week."}]}. Sentence '
    '2 keeps "in the room" as said; sentence 3 fixes the tense in grammatical '
    "and the odd verb only in natural."
)


class Correction(BaseModel):
    """Two corrected versions of one sentence."""

    grammatical: str  # the smallest edit that makes it grammatical
    natural: str  # grammatical, plus what a native speaker would also change


class _NumberedCorrection(BaseModel):
    number: int
    grammatical: str
    natural: str


class _BlockCorrection(BaseModel):
    sentences: list[_NumberedCorrection]


# The deps are how many sentences the block holds.
_agent = Agent(
    output_type=_BlockCorrection,
    deps_type=int,
    instructions=INSTRUCTIONS,
    retries=1,
)


@_agent.output_validator
def _one_item_per_sentence(
    ctx: RunContext[int], answer: _BlockCorrection
) -> _BlockCorrection:
    n = ctx.deps
    if [s.number for s in answer.sentences] != list(range(1, n + 1)):
        raise ModelRetry(
            f"Answer with exactly {n} items, numbered 1 to {n} in order, "
            "one per sentence to correct."
        )
    return answer


async def correct(
    sentences: Sequence[str],
    *,
    previous: str | None,
    following: str | None,
    model: Model | str = DEFAULT_MODEL,
    usage: RunUsage | None = None,
) -> list[Correction]:
    """Correct a block of clean sentences, reading its neighbors only as context."""
    if not sentences:
        return []
    lines = [
        f"Previous sentence: {previous}" if previous is not None else None,
        "Sentences to correct:",
        *(f"{n}. {sentence}" for n, sentence in enumerate(sentences, 1)),
        f"Next sentence: {following}" if following is not None else None,
    ]
    answer = await ask(
        _agent,
        "\n".join(line for line in lines if line is not None),
        deps=len(sentences),
        model=model,
        usage=usage,
        text="\n".join(sentences),
    )
    return [
        Correction(grammatical=s.grammatical, natural=s.natural)
        for s in answer.sentences
    ]
