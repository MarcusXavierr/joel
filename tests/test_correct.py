import functools
import itertools
import json
from collections.abc import Callable

import pytest
from conftest import ScriptedModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from joel.correct import Correction, correct
from joel.llm import ModelAnswerError

pytestmark = pytest.mark.anyio

ScriptedModelFactory = Callable[..., ScriptedModel]

PREVIOUS = "I lived in Recife."
BLOCK = ["Yesterday he go to school.", "She don't like it."]
FOLLOWING = "He liked it."
FIXED = ["Yesterday he went to school.", "She didn't like it."]
NATURAL = ["Yesterday he walked to school.", "She didn't enjoy it."]
CORRECT = "Fix the grammar."


def numbered(*numbers: int) -> dict[str, object]:
    """An answer whose items carry `numbers`, each a fixed sentence."""
    return {
        "sentences": [
            {"number": n, "grammatical": text}
            for n, text in zip(numbers, itertools.cycle(FIXED), strict=False)
        ]
    }


def with_natural(*natural: str) -> dict[str, object]:
    """An answer for BLOCK with these natural versions."""
    return {
        "sentences": [
            {"number": n, "grammatical": fixed, "natural": text}
            for n, (fixed, text) in enumerate(zip(FIXED, natural, strict=True), 1)
        ]
    }


def retry_message(scripted: ScriptedModel, call: int) -> str:
    """What the model was told to fix before the given call."""
    request = scripted.calls[call][-1]
    assert isinstance(request, ModelRequest)
    return "\n".join(
        str(part.content) for part in request.parts if isinstance(part, ToolReturnPart)
    )


def no_problem(k: int, grammatical: str, natural: str) -> str | None:
    return None


async def test_numbers_the_block_between_its_neighbors(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model(numbered(1, 2))

    corrections = await correct(
        BLOCK,
        previous=PREVIOUS,
        following=FOLLOWING,
        instructions=CORRECT,
        model=scripted.model,
    )

    assert corrections == [Correction(grammatical=s, natural=None) for s in FIXED]
    assert scripted.prompt() == (
        "Previous sentence: I lived in Recife.\n"
        "Sentences to correct:\n"
        "1. Yesterday he go to school.\n"
        "2. She don't like it.\n"
        "Next sentence: He liked it."
    )
    assert scripted.instructions() == CORRECT


async def test_only_a_review_asks_the_model_for_a_natural_version() -> None:
    schemas: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        schemas.append(json.dumps(tool.parameters_json_schema))
        answer = numbered(1, 2) if len(schemas) == 1 else with_natural(*NATURAL)
        return ModelResponse(parts=[ToolCallPart(tool.name, answer)])

    ask = functools.partial(
        correct,
        BLOCK,
        previous=None,
        following=None,
        instructions=CORRECT,
        model=FunctionModel(respond),
    )

    plain = await ask()
    reviewed = await ask(review=no_problem)

    assert ['"natural"' in schema for schema in schemas] == [False, True]
    assert [c.natural for c in plain] == [None, None]
    assert [c.natural for c in reviewed] == NATURAL


async def test_a_natural_version_the_review_rejects_is_sent_back_once(
    scripted_model: ScriptedModelFactory,
) -> None:
    rewritten = "She did not like it at all."
    scripted = scripted_model(
        with_natural(NATURAL[0], rewritten), with_natural(*NATURAL)
    )
    seen: list[tuple[int, str, str]] = []

    def review(k: int, grammatical: str, natural: str) -> str | None:
        seen.append((k, grammatical, natural))
        return "natural adds words." if natural == rewritten else None

    corrections = await correct(
        BLOCK,
        previous=None,
        following=None,
        instructions=CORRECT,
        model=scripted.model,
        review=review,
    )

    assert [c.natural for c in corrections] == NATURAL
    assert seen[:2] == [(0, FIXED[0], NATURAL[0]), (1, FIXED[1], rewritten)]
    assert retry_message(scripted, 1) == (
        "Fix natural in these sentences and change nothing else:\n"
        "Sentence 2: natural adds words."
    )


async def test_one_retry_names_every_rejected_natural_version(
    scripted_model: ScriptedModelFactory,
) -> None:
    rewritten = ["Yesterday he went to the school.", "She didn't like it much."]
    scripted = scripted_model(with_natural(*rewritten), with_natural(*NATURAL))
    problems = dict(zip(rewritten, ["adds the.", "adds much."], strict=True))

    corrections = await correct(
        BLOCK,
        previous=None,
        following=None,
        instructions=CORRECT,
        model=scripted.model,
        review=lambda k, grammatical, natural: problems.get(natural),
    )

    assert [c.natural for c in corrections] == NATURAL
    assert retry_message(scripted, 1) == (
        "Fix natural in these sentences and change nothing else:\n"
        "Sentence 1: adds the.\n"
        "Sentence 2: adds much."
    )


async def test_a_natural_version_rejected_twice_falls_back_to_grammatical(
    scripted_model: ScriptedModelFactory,
) -> None:
    rewritten = "She did not like it at all."
    answer = with_natural(NATURAL[0], rewritten)
    scripted = scripted_model(answer, answer)

    def review(k: int, grammatical: str, natural: str) -> str | None:
        return "natural adds words." if natural == rewritten else None

    corrections = await correct(
        BLOCK,
        previous=None,
        following=None,
        instructions=CORRECT,
        model=scripted.model,
        review=review,
    )

    assert corrections == [
        Correction(grammatical=FIXED[0], natural=NATURAL[0]),
        Correction(grammatical=FIXED[1], natural=FIXED[1]),
    ]
    assert len(scripted.calls) == 2


@pytest.mark.parametrize(
    ("previous", "following", "absent"),
    [
        (None, FOLLOWING, ["Previous sentence"]),
        (PREVIOUS, None, ["Next sentence"]),
        (None, None, ["Previous sentence", "Next sentence"]),
    ],
    ids=["first block", "last block", "only block"],
)
async def test_a_missing_neighbor_leaves_its_line_out(
    scripted_model: ScriptedModelFactory,
    previous: str | None,
    following: str | None,
    absent: list[str],
) -> None:
    scripted = scripted_model(numbered(1, 2))

    await correct(
        BLOCK,
        previous=previous,
        following=following,
        instructions=CORRECT,
        model=scripted.model,
    )

    prompt = scripted.prompt()
    for line in absent:
        assert line not in prompt
    assert "None" not in prompt
    assert "1. Yesterday he go to school.\n2. She don't like it." in prompt


@pytest.mark.parametrize("wrong", [(1,), (2, 1), (1, 2, 3), (0, 1)])
async def test_a_misnumbered_answer_is_retried_once(
    scripted_model: ScriptedModelFactory, wrong: tuple[int, ...]
) -> None:
    scripted = scripted_model(numbered(*wrong), numbered(1, 2))

    corrections = await correct(
        BLOCK,
        previous=None,
        following=None,
        instructions=CORRECT,
        model=scripted.model,
    )

    assert [c.grammatical for c in corrections] == FIXED
    assert len(scripted.calls) == 2


@pytest.mark.parametrize("review", [None, no_problem], ids=["grammar", "natural"])
async def test_a_second_misnumbered_answer_fails_naming_the_block(
    scripted_model: ScriptedModelFactory,
    review: Callable[[int, str, str], str | None] | None,
) -> None:
    wrong = {"sentences": [{"number": 1, "grammatical": "x", "natural": "x"}]}
    scripted = scripted_model(wrong, wrong)

    with pytest.raises(ModelAnswerError, match="She don't like it."):
        await correct(
            BLOCK,
            previous=None,
            following=None,
            instructions=CORRECT,
            model=scripted.model,
            review=review,
        )

    assert len(scripted.calls) == 2


async def test_no_sentences_need_no_call(scripted_model: ScriptedModelFactory) -> None:
    scripted = scripted_model()

    corrections = await correct(
        [],
        previous=None,
        following=None,
        instructions=CORRECT,
        model=scripted.model,
    )

    assert corrections == []
    assert scripted.calls == []
