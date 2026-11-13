"""Pipeline step 5: turn the LLM's corrections into typed edits with ERRANT."""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import errant  # type: ignore[import-untyped]
from errant.annotator import Annotator  # type: ignore[import-untyped]
from pydantic import BaseModel
from spacy.tokens import Doc

from joel.transcript import Word

Category = Literal["grammar", "naturalness"]


class Edit(BaseModel):
    """One change between what was said and a correction of it."""

    category: Category
    type: str  # ERRANT type, e.g. "R:VERB:SVA"
    original: str  # ERRANT token text ("do", not "don't"); "" for an insertion
    correction: str  # "" for a deletion
    words: list[int]  # Word.index values, ascending, no repeats


@dataclass(frozen=True)
class _Span:
    """An edit plus its ERRANT token span, half-open, in the original sentence."""

    edit: Edit
    start: int
    end: int

    @property
    def is_insertion(self) -> bool:
        return self.start == self.end


def load_annotator() -> Annotator:
    """Load ERRANT for English. It loads spaCy's en_core_web_sm, so call it once."""
    return errant.load("en")


def diff_edits(
    words: Sequence[Word], grammatical: str, natural: str, annotator: Annotator
) -> list[Edit]:
    """Diff a clean sentence against its two corrections.

    `words` is the sentence without the dropped words. Returns the grammar
    edits, then the naturalness edits that touch none of them, each in ERRANT
    order. Edits nobody can hear (punctuation, case, apostrophes, spacing) are
    left out.
    """
    if not words:
        return []

    original = annotator.parse(" ".join(w.text for w in words), tokenise=True)
    token_words = _token_words(words, original)

    def spans(target: str, category: Category) -> list[_Span]:
        corrected = annotator.parse(target, tokenise=True)
        result = []
        for e in annotator.annotate(original, corrected):
            start, end = int(e.o_start), int(e.o_end)
            if start < end:
                tokens = range(start, end)
            else:  # an insertion sits between two tokens; point at both
                tokens = range(max(start - 1, 0), min(start + 1, len(token_words)))
            edit = Edit(
                category=category,
                type=str(e.type),
                original=str(e.o_str),
                correction=str(e.c_str),
                words=sorted({token_words[t] for t in tokens}),
            )
            if _audible(edit.original) != _audible(edit.correction):
                result.append(_Span(edit, start, end))
        return result

    grammar = spans(grammatical, "grammar")
    naturalness = [
        n
        for n in spans(natural, "naturalness")
        if not any(_overlaps(n, g) for g in grammar)
    ]
    return [s.edit for s in grammar + naturalness]


def _token_words(words: Sequence[Word], doc: Doc) -> list[int]:
    """The `Word.index` of each spaCy token in `doc`, the words joined by spaces.

    spaCy never splits a token across whitespace, so each token sits inside
    exactly one word: the last one starting at or before it (ADR-0006).
    """
    starts: list[int] = []
    offset = 0
    for word in words:
        starts.append(offset)
        offset += len(word.text) + 1
    return [words[bisect_right(starts, token.idx) - 1].index for token in doc]


def _audible(text: str) -> str:
    """What a listener can tell apart: letters and digits, case folded."""
    return "".join(c for c in text.casefold() if c.isalnum())


def _overlaps(a: _Span, b: _Span) -> bool:
    if a.is_insertion or b.is_insertion:
        return max(a.start, b.start) <= min(a.end, b.end)
    return not set(a.edit.words).isdisjoint(b.edit.words)
