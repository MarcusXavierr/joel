import pytest

from joel.confidence import FilteredEdit, filter_edits
from joel.edits import Edit
from joel.transcript import Word


def words(text: str, confidences: list[float | None]) -> list[Word]:
    return [
        Word(index=i, text=t, start_ms=None, end_ms=None, confidence=c)
        for i, (t, c) in enumerate(zip(text.split(" "), confidences, strict=True))
    ]


def edit(*indices: int) -> Edit:
    return Edit(
        category="grammar",
        type="R:VERB:SVA",
        original="go",
        correction="goes",
        words=list(indices),
    )


def test_edit_on_a_word_below_the_threshold_is_uncertain() -> None:
    transcript = words("she go home", [0.9, 0.5, 0.9])

    [result] = filter_edits([edit(1)], transcript, min_confidence=0.67)

    assert result.edit == edit(1)
    assert result.confidence == 0.5
    assert result.uncertain is not None
    assert '"go"' in result.uncertain
    assert "0.50" in result.uncertain
    assert "0.67" in result.uncertain


def test_lower_threshold_keeps_the_same_edit() -> None:
    transcript = words("she go home", [0.9, 0.5, 0.9])

    assert filter_edits([edit(1)], transcript, min_confidence=0.4) == [
        FilteredEdit(edit=edit(1), confidence=0.5, uncertain=None)
    ]


def test_confidence_equal_to_the_threshold_is_kept() -> None:
    transcript = words("she go home", [0.9, 0.67, 0.9])

    [result] = filter_edits([edit(1)], transcript, min_confidence=0.67)

    assert result.uncertain is None


def test_words_without_confidence_never_make_an_edit_uncertain() -> None:
    transcript = words("she go home", [None, None, None])

    assert filter_edits([edit(0, 1)], transcript) == [
        FilteredEdit(edit=edit(0, 1), confidence=None, uncertain=None)
    ]


def test_edit_takes_the_lowest_confidence_of_its_words() -> None:
    # An insertion points at the two words around it.
    transcript = words("He is engineer.", [0.9, 0.9, 0.3])

    [result] = filter_edits([edit(1, 2)], transcript)

    assert result.confidence == 0.3
    assert result.uncertain is not None
    assert '"engineer."' in result.uncertain


def test_reason_names_the_first_word_when_two_tie_for_lowest() -> None:
    transcript = words("she go home", [0.9, 0.5, 0.5])

    [result] = filter_edits([edit(1, 2)], transcript)

    assert result.uncertain is not None
    assert '"go"' in result.uncertain
    assert '"home"' not in result.uncertain


def test_unknown_confidence_is_ignored_next_to_a_known_one() -> None:
    transcript = words("she go home", [0.9, None, 0.8])

    [result] = filter_edits([edit(1, 2)], transcript)

    assert result.confidence == 0.8


def test_output_keeps_input_order() -> None:
    transcript = words("she go home", [0.2, 0.9, 0.9])

    results = filter_edits([edit(2), edit(0), edit(1)], transcript)

    assert [r.edit.words for r in results] == [[2], [0], [1]]
    assert [r.uncertain is None for r in results] == [True, False, True]


def test_edit_pointing_outside_the_transcript_is_an_error() -> None:
    transcript = words("she go home", [0.9, 0.9, 0.9])

    with pytest.raises(ValueError, match="word 7"):
        filter_edits([edit(7)], transcript)
