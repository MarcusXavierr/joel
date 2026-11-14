import json
from collections.abc import Callable

import pytest
from conftest import ScriptedModel
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from joel.disfluencies import strip_disfluencies
from joel.llm import ModelAnswerError
from joel.transcript import Word

pytestmark = pytest.mark.anyio

ScriptedModelFactory = Callable[..., ScriptedModel]


def sentence(first_index: int, *texts: str) -> list[Word]:
    return [
        Word(
            index=first_index + offset,
            text=text,
            start_ms=1000 * offset,
            end_ms=1000 * offset + 500,
            confidence=0.9,
        )
        for offset, text in enumerate(texts)
    ]


I_WENT = sentence(12, "I", "went,", "I", "was", "uh", "going")


async def test_drops_the_positions_the_model_names(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model({"drop": [12, 13, 16]})

    clean = await strip_disfluencies(I_WENT, model=scripted.model)

    assert clean.words == [I_WENT[2], I_WENT[3], I_WENT[5]]
    assert clean.dropped == [12, 13, 16]
    assert [12, "I"] in json.loads(scripted.prompt())
    assert [17, "going"] in json.loads(scripted.prompt())


async def test_a_sentence_of_only_disfluencies_keeps_no_words(
    scripted_model: ScriptedModelFactory,
) -> None:
    uh = sentence(3, "Uh.")
    scripted = scripted_model({"drop": [3]})

    clean = await strip_disfluencies(uh, model=scripted.model)

    assert clean.words == []
    assert clean.dropped == [3]


async def test_a_passage_of_two_sentences_drops_a_word_in_each(
    scripted_model: ScriptedModelFactory,
) -> None:
    passage = sentence(0, "Uh,", "he", "go.", "She", "she", "left.")
    scripted = scripted_model({"drop": [3, 0]})

    clean = await strip_disfluencies(passage, model=scripted.model)

    assert clean.dropped == [0, 3]
    assert [w.text for w in clean.words] == ["he", "go.", "she", "left."]


async def test_a_position_named_twice_is_dropped_once(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model({"drop": [16, 12, 16]})

    clean = await strip_disfluencies(I_WENT, model=scripted.model)

    assert clean.dropped == [12, 16]


async def test_a_position_outside_the_sentence_is_retried_once(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model({"drop": [99]}, {"drop": [16]})

    clean = await strip_disfluencies(I_WENT, model=scripted.model)

    assert clean.dropped == [16]
    assert len(scripted.calls) == 2


async def test_a_second_bad_answer_fails_naming_the_sentence(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model({"drop": [99]}, {"drop": [99]})

    with pytest.raises(ModelAnswerError, match="I went, I was uh going"):
        await strip_disfluencies(I_WENT, model=scripted.model)

    assert len(scripted.calls) == 2


async def test_usage_accumulates_into_the_shared_counter(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model({"drop": []})
    usage = RunUsage()

    await strip_disfluencies(I_WENT, model=scripted.model, usage=usage)

    assert usage.requests == 1
    assert usage.input_tokens == 100


async def test_a_provider_error_is_not_mistaken_for_a_bad_answer() -> None:
    def unreachable(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ConnectionError("api.anthropic.com is unreachable")

    with pytest.raises(ConnectionError, match="unreachable"):
        await strip_disfluencies(I_WENT, model=FunctionModel(unreachable))
