"""What the LLM steps cost: an estimate from audio length, and the price of tokens."""

import math
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


def price(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Price tokens for a Pydantic AI model string, in US dollars."""
    provider, sep, name = model.partition(":")
    if not sep:
        raise PriceUnavailableError(
            f"'{model}' is not a Pydantic AI model string such as "
            "'anthropic:claude-opus-5-5'."
        )
    usage = genai_prices.Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    try:
        return genai_prices.calc_price(usage, name, provider_id=provider).total_price
    except LookupError as exc:
        message = f"genai-prices has no price for '{model}'."
        raise PriceUnavailableError(message) from exc


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
    return CostEstimate(
        model=model,
        audio_seconds=audio_seconds,
        words=words,
        sentences=sentences,
        calls=2 * sentences,  # one cleanup and one correction per sentence
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=price(model, input_tokens, output_tokens),
    )
