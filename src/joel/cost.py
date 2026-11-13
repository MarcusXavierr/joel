"""What the LLM steps cost: an estimate from audio length, and the price of tokens."""

import math
from collections import Counter
from decimal import Decimal
from typing import NamedTuple

import genai_prices
from pydantic import BaseModel

from joel.llm import DEFAULT_MODEL

SPEAKING_RATE_WPM = 150  # assumed; the CLI prints it with every estimate
WORDS_PER_SENTENCE = 12  # assumed; the CLI prints it with every estimate


class TokenRate(NamedTuple):
    """Tokens per call for a sentence of n words: base + per_word * n."""

    base: int
    per_word: float


# Measured on 2026-11-13 against anthropic:claude-opus-5-5 with joel's prompts,
# on 3- and 25-word sentences; refit when a prompt changes. Correction output
# includes the model's thinking, which varies from run to run, so each point is
# a three-run average.
CLEANUP_INPUT = TokenRate(463, 7.0)
CLEANUP_OUTPUT = TokenRate(10, 0.0)
CORRECT_INPUT = TokenRate(532, 3.6)
CORRECT_OUTPUT = TokenRate(13, 3.6)


class PriceUnavailableError(Exception):
    """genai-prices can't price the model, or the model string has no provider."""


def price(
    model: str, input_tokens: int, output_tokens: int, *, requests: int
) -> Decimal:
    """Price tokens spent over a number of model calls, in US dollars.

    The tokens are priced as if spread evenly over the calls. Some models charge
    more per token once a single request passes a size threshold: a whole run
    priced as one request would cross it, while Joel's sentence-sized calls
    never do. Zero calls cost nothing.
    """
    provider, sep, name = model.partition(":")
    if not sep:
        raise PriceUnavailableError(
            f"'{model}' is not a Pydantic AI model string such as "
            "'anthropic:claude-opus-5-5'."
        )
    calls = Counter(
        (_share(input_tokens, requests, call), _share(output_tokens, requests, call))
        for call in range(requests)
    )
    try:
        return sum(
            (
                count
                * genai_prices.calc_price(
                    genai_prices.Usage(
                        input_tokens=call_input, output_tokens=call_output
                    ),
                    name,
                    provider_id=provider,
                ).total_price
                for (call_input, call_output), count in calls.items()
            ),
            Decimal(0),
        )
    except LookupError as exc:
        message = f"genai-prices has no price for '{model}'."
        raise PriceUnavailableError(message) from exc


def _share(tokens: int, requests: int, call: int) -> int:
    """The tokens of one call when `tokens` are spread evenly over `requests` calls."""
    return tokens // requests + (call < tokens % requests)


class CostEstimate(BaseModel):
    """What the LLM steps would cost for an audio file of a given length."""

    model: str
    audio_seconds: float
    words: int
    sentences: int
    calls: int
    input_tokens: int
    output_tokens: int
    usd: Decimal


def _per_sentence(rate: TokenRate) -> int:
    return round(rate.base + rate.per_word * WORDS_PER_SENTENCE)


def estimate_cost(audio_seconds: float, model: str = DEFAULT_MODEL) -> CostEstimate:
    """Estimate the LLM cost from the audio's duration alone. Retries aren't counted."""
    words = math.ceil(audio_seconds / 60 * SPEAKING_RATE_WPM)
    sentences = math.ceil(words / WORDS_PER_SENTENCE)
    input_tokens = sentences * (
        _per_sentence(CLEANUP_INPUT) + _per_sentence(CORRECT_INPUT)
    )
    output_tokens = sentences * (
        _per_sentence(CLEANUP_OUTPUT) + _per_sentence(CORRECT_OUTPUT)
    )
    calls = 2 * sentences  # one cleanup and one correction per sentence
    return CostEstimate(
        model=model,
        audio_seconds=audio_seconds,
        words=words,
        sentences=sentences,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=price(model, input_tokens, output_tokens, requests=calls),
    )
