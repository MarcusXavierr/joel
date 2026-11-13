"""What the two LLM steps share: the default model and how a call fails."""

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

DEFAULT_MODEL = "anthropic:claude-opus-5-5"


class ModelAnswerError(Exception):
    """The model's answer stayed invalid after one retry."""


async def ask[D, T](
    agent: Agent[D, T],
    prompt: str,
    *,
    deps: D,
    model: Model | str,
    usage: RunUsage | None,
    sentence: str,
) -> T:
    """Run the agent (plus its one retry) and return its validated answer.

    Network and authentication errors propagate as the provider raised them.
    """
    try:
        result = await agent.run(prompt, deps=deps, model=model, usage=usage)
    except UnexpectedModelBehavior as exc:
        raise ModelAnswerError(
            f"The model gave no valid answer for the sentence {sentence!r}, "
            "even after a retry."
        ) from exc
    return result.output
