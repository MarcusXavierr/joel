import pytest

from joel.sentences import split_sentences
from joel.transcript import Word


def words(text: str, first: int = 0) -> list[Word]:
    return [
        Word(index=first + i, text=t, start_ms=None, end_ms=None, confidence=None)
        for i, t in enumerate(text.split(" "))
    ]


def test_splits_after_each_closing_word() -> None:
    transcript = words("I went home. Did you? Yes!", first=3)

    assert split_sentences(transcript) == [[3, 4, 5], [6, 7], [8]]


def test_trailing_words_without_closing_punctuation_form_a_last_sentence() -> None:
    transcript = words("I went home. and then I slept")

    sentences = split_sentences(transcript)

    assert sentences == [[0, 1, 2], [3, 4, 5, 6]]
    assert [i for s in sentences for i in s] == [w.index for w in transcript]


@pytest.mark.parametrize("closing", ['it."', "it.'", "it.)", "it.”", "it?”", "it!’"])
def test_closing_quote_or_bracket_after_the_mark_still_closes(closing: str) -> None:
    transcript = words(f"he said {closing} Then he left.")

    assert split_sentences(transcript) == [[0, 1, 2], [3, 4, 5]]


def test_comma_does_not_close_a_sentence() -> None:
    transcript = words("school, uh, and")

    assert split_sentences(transcript) == [[0, 1, 2]]


def test_empty_transcript_has_no_sentences() -> None:
    assert split_sentences([]) == []
