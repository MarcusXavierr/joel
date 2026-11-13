from errant.annotator import Annotator  # type: ignore[import-untyped]

from joel.edits import Edit, diff_edits
from joel.transcript import Word


def words(text: str, first: int = 0) -> list[Word]:
    return [
        Word(index=first + i, text=t, start_ms=None, end_ms=None, confidence=0.9)
        for i, t in enumerate(text.split(" "))
    ]


def test_contraction_edit_points_at_the_spoken_word(annotator: Annotator) -> None:
    clean = words("she don't like it", first=10)

    edits = diff_edits(clean, "she doesn't like it", "she doesn't like it", annotator)

    assert edits == [
        Edit(
            category="grammar",
            type="R:VERB:SVA",
            original="do",
            correction="does",
            words=[11],
        )
    ]


def test_naturalness_edit_is_kept_and_repeated_grammar_edit_is_not(
    annotator: Annotator,
) -> None:
    clean = words("There is many people in the room.")

    edits = diff_edits(
        clean,
        "There are many people in the room.",
        "There are a lot of people in the room.",
        annotator,
    )

    assert edits == [
        Edit(
            category="grammar",
            type="R:VERB:SVA",
            original="is",
            correction="are",
            words=[1],
        ),
        Edit(
            category="naturalness",
            type="R:OTHER",
            original="many",
            correction="a lot of",
            words=[2],
        ),
    ]


def test_insertion_points_at_both_neighbors_and_is_not_repeated(
    annotator: Annotator,
) -> None:
    clean = words("He is engineer.", first=4)

    edits = diff_edits(clean, "He is an engineer.", "He is an engineer.", annotator)

    assert edits == [
        Edit(
            category="grammar",
            type="M:DET",
            original="",
            correction="an",
            words=[5, 6],
        )
    ]


def test_naturalness_edit_touching_a_grammar_insertion_is_dropped(
    annotator: Annotator,
) -> None:
    # "is → 's" is a different change from inserting "an", but it sits right
    # next to it, so it goes for precision's sake.
    clean = words("He is engineer.")

    edits = diff_edits(clean, "He is an engineer.", "He's an engineer.", annotator)

    assert [(e.category, e.original, e.correction) for e in edits] == [
        ("grammar", "", "an")
    ]


def test_insertion_at_sentence_start_points_at_the_first_word(
    annotator: Annotator,
) -> None:
    clean = words("went to school.", first=7)

    edits = diff_edits(clean, "I went to school.", "I went to school.", annotator)

    assert [e.words for e in edits] == [[7]]


def test_insertion_at_sentence_end_points_at_the_last_word(
    annotator: Annotator,
) -> None:
    clean = words("I like", first=7)

    edits = diff_edits(clean, "I like it", "I like it", annotator)

    assert [e.words for e in edits] == [[8]]


def test_attached_punctuation_maps_to_its_word(annotator: Annotator) -> None:
    # "student," is one word but two tokens; ", of course" spans three words.
    clean = words("He is a student, of course.")

    edits = diff_edits(clean, "He is a student.", "He is a student.", annotator)

    assert [(e.original, e.words) for e in edits] == [(", of course", [3, 4, 5])]


def test_tokens_after_attached_punctuation_keep_their_word(
    annotator: Annotator,
) -> None:
    clean = words("I like school, and she go home.")

    edits = diff_edits(
        clean,
        "I like school, and she goes home.",
        "I like school, and she goes home.",
        annotator,
    )

    assert [(e.original, e.words) for e in edits] == [("go", [5])]


def test_edits_that_cannot_be_heard_are_dropped(annotator: Annotator) -> None:
    # Missing final period and lowercase "i" in the grammatical target;
    # a missing apostrophe and "alot" in the natural one.
    clean = words("i went to school")

    assert diff_edits(clean, "I went to school.", "I went to school", annotator) == []
    assert diff_edits(words("Its alot"), "Its alot", "It's a lot", annotator) == []


def test_empty_sentence_has_no_edits(annotator: Annotator) -> None:
    assert diff_edits([], "", "", annotator) == []
