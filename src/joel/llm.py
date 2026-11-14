"""What the two LLM steps share: the default model and how a call fails."""

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

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
        # A run shares one usage across every call, so pydantic-ai's default cap
        # of 50 requests would end a long recording early. Each agent's own
        # `retries` still bounds the calls one sentence can make.
        result = await agent.run(
            prompt,
            deps=deps,
            model=model,
            usage=usage,
            usage_limits=UsageLimits(request_limit=None),
        )
    except UnexpectedModelBehavior as exc:
        raise ModelAnswerError(
            f"The model gave no valid answer for the sentence {sentence!r}, "
            "even after a retry."
        ) from exc
    return result.output
