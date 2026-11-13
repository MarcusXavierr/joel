from decimal import Decimal

import pytest

from joel.cost import PriceUnavailableError, estimate_cost, price


def test_prices_tokens_with_the_bundled_price_table() -> None:
    # Claude Opus 5.5: $4 per million input tokens, $20 per million output.
    assert price("anthropic:claude-opus-5-5", 1_000_000, 1_000_000) == Decimal("24")


@pytest.mark.parametrize("model", ["nope:nothing", "claude-opus-5-5"])
def test_a_model_without_a_price_names_the_model(model: str) -> None:
    with pytest.raises(PriceUnavailableError, match=f"'{model}'"):
        price(model, 1, 1)


def test_a_minute_of_speech_is_150_words_in_13_sentences() -> None:
    estimate = estimate_cost(60.0)

    assert estimate.words == 150
    assert estimate.sentences == 13
    assert estimate.calls == 26
    assert estimate.usd > 0


def test_a_second_of_speech_still_costs_one_sentence() -> None:
    estimate = estimate_cost(1.0)

    assert estimate.sentences == 1
    assert estimate.calls == 2
