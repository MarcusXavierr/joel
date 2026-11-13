from collections.abc import Callable

import pytest
from conftest import ScriptedModel

from joel.correct import Correction, correct
from joel.llm import ModelAnswerError

pytestmark = pytest.mark.anyio

ScriptedModelFactory = Callable[..., ScriptedModel]

PREVIOUS = "I lived in Recife."
SENTENCE = "Yesterday he go to school."
FOLLOWING = "He liked it."


async def test_sends_the_neighbors_as_context(
    scripted_model: ScriptedModelFactory,
) -> None:
    scripted = scripted_model(
        {
            "grammatical": "Yesterday he went to school.",
            "natural": "Yesterday he went to school.",
        }
    )

    correction = await correct(
        SENTENCE, previous=PREVIOUS, following=FOLLOWING, model=scripted.model
    )

    assert correction == Correction(
        grammatical="Yesterday he went to school.",
        natural="Yesterday he went to school.",
    )
    prompt = scripted.prompt()
    assert PREVIOUS in prompt
    assert SENTENCE in prompt
    assert FOLLOWING in prompt


async def test_an_answer_missing_a_version_twice_fails_naming_the_sentence(
    scripted_model: ScriptedModelFactory,
) -> None:
    incomplete = {"grammatical": "Yesterday he went to school."}
    scripted = scripted_model(incomplete, incomplete)

    with pytest.raises(ModelAnswerError, match=SENTENCE):
        await correct(
            SENTENCE, previous=PREVIOUS, following=FOLLOWING, model=scripted.model
        )

    assert len(scripted.calls) == 2


@pytest.mark.parametrize(
    ("previous", "following", "absent"),
    [
        (None, FOLLOWING, ["Previous sentence"]),
        (PREVIOUS, None, ["Next sentence"]),
        (None, None, ["Previous sentence", "Next sentence"]),
    ],
    ids=["first sentence", "last sentence", "only sentence"],
)
async def test_a_missing_neighbor_leaves_its_line_out(
    scripted_model: ScriptedModelFactory,
    previous: str | None,
    following: str | None,
    absent: list[str],
) -> None:
    scripted = scripted_model({"grammatical": SENTENCE, "natural": SENTENCE})

    await correct(
        SENTENCE, previous=previous, following=following, model=scripted.model
    )

    prompt = scripted.prompt()
    for line in absent:
        assert line not in prompt
    assert "None" not in prompt
    assert SENTENCE in prompt
