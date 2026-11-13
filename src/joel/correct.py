"""Step 4 of the pipeline: a grammatical and a natural version of one sentence."""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from joel.llm import DEFAULT_MODEL, ask

INSTRUCTIONS = (
    "You correct transcribed speech from a non-native English speaker. You "
    "receive the sentence to correct, and the sentences before and after it when "
    "they exist. The neighbors are context only, for tense and reference; never "
    "correct them. Answer with two versions of the sentence to correct. "
    "grammatical: the smallest edit that makes the sentence grammatical; change "
    "only words that are wrong; if the sentence is already grammatical, return it "
    "unchanged. natural: the smallest further edit to grammatical that a native "
    'speaker would make because the wording sounds odd, such as "I make a party" '
    'to "I threw a party"; if nothing sounds odd, return grammatical unchanged. '
    "Edit as little as possible. Do not rephrase, reorder, or improve style. Keep "
    "the speaker's words, word order, register and spoken style. Keep punctuation "
    "and capitalization exactly as given: they come from the transcriber, not the "
    "speaker."
)


class Correction(BaseModel):
    """Two corrected versions of one sentence."""

    grammatical: str  # the smallest edit that makes it grammatical
    natural: str  # grammatical, plus what a native speaker would also change


_agent = Agent(output_type=Correction, instructions=INSTRUCTIONS, retries=1)


async def correct(
    sentence: str,
    *,
    previous: str | None,
    following: str | None,
    model: Model | str = DEFAULT_MODEL,
    usage: RunUsage | None = None,
) -> Correction:
    """Correct one clean sentence, reading its neighbors only as context."""
    lines = [
        f"Previous sentence: {previous}" if previous is not None else None,
        f"Sentence to correct: {sentence}",
        f"Next sentence: {following}" if following is not None else None,
    ]
    return await ask(
        _agent,
        "\n".join(line for line in lines if line is not None),
        deps=None,
        model=model,
        usage=usage,
        sentence=sentence,
    )
