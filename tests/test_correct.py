import itertools
from collections.abc import Callable

import pytest
from conftest import ScriptedModel

from joel.correct import Correction, correct
from joel.llm import ModelAnswerError

pytestmark = pytest.mark.anyio

ScriptedModelFactory = Callable[..., ScriptedModel]

PREVIOUS = "I lived in Recife."
BLOCK = ["Yesterday he go to school.", "She don't like it."]
FOLLOWING = "He liked it."
FIXED = ["Yesterday he went to school.", "She didn't like it."]


def numbered(*numbers: int) -> dict[str, object]:
    """An answer whose items carry `numbers`, each a fixed sentence."""
    return {
        "sentences": [
            {"number": n, "grammatical": text, "natural": text}
            for n, text in zip(numbers, itertools.cycle(FIXED), strict=False)
        ]
    }


async def test_numbers_the_block_between_its_neighbors(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model(numbered(1, 2))

    corrections = await correct(
        BLOCK, previous=PREVIOUS, following=FOLLOWING, model=scripted.model
    )

    assert corrections == [Correction(grammatical=s, natural=s) for s in FIXED]
    assert scripted.prompt() == (
        "Previous sentence: I lived in Recife.\n"
        "Sentences to correct:\n"
        "1. Yesterday he go to school.\n"
        "2. She don't like it.\n"
        "Next sentence: He liked it."
    )


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

    await correct(BLOCK, previous=previous, following=following, model=scripted.model)

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
        BLOCK, previous=None, following=None, model=scripted.model
    )

    assert [c.grammatical for c in corrections] == FIXED
    assert len(scripted.calls) == 2


async def test_a_second_misnumbered_answer_fails_naming_the_block(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model(numbered(1), numbered(1))

    with pytest.raises(ModelAnswerError, match="She don't like it."):
        await correct(BLOCK, previous=None, following=None, model=scripted.model)

    assert len(scripted.calls) == 2


async def test_no_sentences_need_no_call(scripted_model: ScriptedModelFactory) -> None:
    scripted = scripted_model()

    assert await correct([], previous=None, following=None, model=scripted.model) == []
    assert scripted.calls == []
