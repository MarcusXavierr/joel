"""What the LLM steps cost: an estimate from audio length, and the price of tokens."""

import math
from collections import Counter
from decimal import Decimal
from typing import NamedTuple

import genai_prices
from pydantic import BaseModel

from joel.llm import BLOCK_SENTENCES, DEFAULT_MODEL

SPEAKING_RATE_WPM = 150  # assumed; the CLI prints it with every estimate
WORDS_PER_SENTENCE = 12  # assumed; the CLI prints it with every estimate


class TokenRate(NamedTuple):
    """Tokens per call for a block of n words: base + per_word * n."""

    base: int
    per_word: float


# Measured on 2026-11-14 against openai:gpt-6-luna with joel's prompts, on
# blocks of 1 sentence (4 words) and 7 sentences (70 words); refit when a prompt
# or the default model changes. Output includes the model's reasoning, which
# varies from run to run, so each point is a three-run average.
CLEANUP_INPUT = TokenRate(473, 6.4)
CLEANUP_OUTPUT = TokenRate(31, 1.8)
CORRECT_INPUT = TokenRate(479, 1.3)
CORRECT_OUTPUT = TokenRate(20, 4.9)


class PriceUnavailableError(Exception):
    """genai-prices can't price the model, or the model string has no provider."""


def price(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    requests: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal:
    """Price tokens spent over a number of model calls, in US dollars.

    `input_tokens` includes the cached ones, which are priced at the cache's
    rates. The tokens are priced as if spread evenly over the calls. Some models
    charge more per token once a single request passes a size threshold: a whole
    run priced as one request would cross it, while Joel's block-sized calls
    never do. Zero calls cost nothing.
    """
    provider, sep, name = model.partition(":")
    if not sep:
        raise PriceUnavailableError(
            f"'{model}' is not a Pydantic AI model string such as "
            "'anthropic:claude-opus-5-5'."
        )
    totals = (input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
    calls = Counter(
        tuple(_share(tokens, requests, call) for tokens in totals)
        for call in range(requests)
    )
    try:
        return sum(
            (
                count
                * genai_prices.calc_price(
                    genai_prices.Usage(
                        input_tokens=call_input,
                        cache_read_tokens=call_read,
                        cache_write_tokens=call_write,
                        output_tokens=call_output,
                    ),
                    name,
                    provider_id=provider,
                ).total_price
                for (call_input, call_read, call_write, call_output), count in (
                    calls.items()
                )
            ),
            Decimal(0),
        )
    except LookupError as exc:
        message = f"genai-prices has no price for '{model}'."
        raise PriceUnavailableError(message) from exc


def _share(tokens: int, requests: int, call: int) -> int:
    """The tokens of one call when `tokens` are spread evenly over `requests` calls."""
    return tokens // requests + (call < tokens % requests)


def dollars(usd: Decimal) -> str:
    """US dollars to a tenth of a cent, with anything smaller said as such."""
    if usd == 0:
        return "$0.000"
    if usd < Decimal("0.001"):
        return "under $0.001"
    return f"${usd:.3f}"


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


def estimate_cost(
    audio_seconds: float,
    model: str = DEFAULT_MODEL,
    block_sentences: int = BLOCK_SENTENCES,
) -> CostEstimate:
    """Estimate the LLM cost from the audio's duration alone. Retries aren't counted."""
    words = math.ceil(audio_seconds / 60 * SPEAKING_RATE_WPM)
    sentences = math.ceil(words / WORDS_PER_SENTENCE)
    blocks = math.ceil(sentences / block_sentences)
    input_tokens = blocks * (CLEANUP_INPUT.base + CORRECT_INPUT.base) + round(
        words * (CLEANUP_INPUT.per_word + CORRECT_INPUT.per_word)
    )
    output_tokens = blocks * (CLEANUP_OUTPUT.base + CORRECT_OUTPUT.base) + round(
        words * (CLEANUP_OUTPUT.per_word + CORRECT_OUTPUT.per_word)
    )
    calls = 2 * blocks  # one cleanup and one correction per block
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
