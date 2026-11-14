from decimal import Decimal

import pytest

from joel.cost import PriceUnavailableError, estimate_cost, price


def test_prices_tokens_with_the_bundled_price_table() -> None:
    # Claude Opus 5.5: $4 per million input tokens, $20 per million output.
    assert price(
        "anthropic:claude-opus-5-5", 1_000_000, 1_000_000, requests=1
    ) == Decimal("24")


def test_tokens_read_from_the_cache_cost_less() -> None:
    # Opus 5.5 reads cached input at $0.20 per million, against $4 uncached.
    model = "anthropic:claude-opus-5-5"

    cached = price(model, 1_000_000, 0, requests=1, cache_read_tokens=1_000_000)

    assert cached == Decimal("0.2")
    assert price(model, 1_000_000, 0, requests=1) == Decimal(4)


def test_tokens_written_to_the_cache_cost_more_however_the_calls_split_them() -> None:
    # Opus 5.5 writes to its 5-minute cache at $5 per million, against $4 uncached.
    written = price(
        "anthropic:claude-opus-5-5",
        1_000_000,
        0,
        requests=3,
        cache_write_tokens=1_000_000,
    )

    assert written == Decimal(5)


def test_many_small_requests_never_reach_a_long_context_price() -> None:
    # Sonnet 4.5 charges more per token once one request passes 200k input
    # tokens. A run's 300k tokens are spread over many short requests.
    one = price("anthropic:claude-sonnet-4-5", 100_000, 10_000, requests=1)

    assert price("anthropic:claude-sonnet-4-5", 300_000, 30_000, requests=3) == 3 * one


@pytest.mark.parametrize("model", ["nope:nothing", "claude-opus-5-5"])
def test_a_model_without_a_price_names_the_model(model: str) -> None:
    with pytest.raises(PriceUnavailableError, match=f"'{model}'"):
        price(model, 1, 1, requests=1)


def test_a_minute_of_speech_is_150_words_in_13_sentences_and_2_blocks() -> None:
    estimate = estimate_cost(60.0)

    assert estimate.words == 150
    assert estimate.sentences == 13
    assert estimate.calls == 4  # a cleanup and a correction per block of 7
    assert estimate.usd > 0


def test_smaller_blocks_make_more_calls_and_repeat_the_instructions() -> None:
    default_blocks = estimate_cost(60.0)
    one_per_call = estimate_cost(60.0, block_sentences=1)

    assert one_per_call.calls == 26
    assert one_per_call.input_tokens > default_blocks.input_tokens
    assert one_per_call.usd > default_blocks.usd


def test_a_second_of_speech_still_costs_one_sentence() -> None:
    estimate = estimate_cost(1.0)

    assert estimate.sentences == 1
    assert estimate.calls == 2


def test_twice_the_audio_costs_twice_as_much() -> None:
    ten_minutes = estimate_cost(600.0, "anthropic:claude-sonnet-4-5")
    twenty_minutes = estimate_cost(1200.0, "anthropic:claude-sonnet-4-5")

    assert twenty_minutes.usd == 2 * ten_minutes.usd
