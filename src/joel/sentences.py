"""Pipeline step 2: split the transcript into sentences at closing punctuation."""

from collections.abc import Sequence

from joel.transcript import Word

_CLOSING_MARKS = (".", "?", "!")
# Quotes and brackets that may follow the mark, as in `it."`.
_TRAILING = "\"')]”’"


def split_sentences(words: Sequence[Word]) -> list[list[int]]:
    """Group word indices into sentences, in spoken order.

    A word closes its sentence when it ends with `.`, `?`, or `!`, ignoring
    closing quotes and brackets after the mark. Words after the last closing
    word form a final sentence, so no word is lost or repeated.
    """
    sentences: list[list[int]] = []
    current: list[int] = []
    for word in words:
        current.append(word.index)
        if word.text.rstrip(_TRAILING).endswith(_CLOSING_MARKS):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences
