"""Pipeline step 5: turn the LLM's corrections into typed edits with ERRANT."""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import errant  # type: ignore[import-untyped]
from errant.annotator import Annotator  # type: ignore[import-untyped]
from errant.edit import Edit as ErrantEdit  # type: ignore[import-untyped]
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
    order. Changes nobody can hear (punctuation, case, apostrophes, spacing,
    homophones like "your/you're", digits against number words) are left out.
    """
    if not words:
        return []

    original = annotator.parse(" ".join(w.text for w in words), tokenise=True)
    token_words = _token_words(words, original)

    def spans(target: str, category: Category) -> list[_Span]:
        corrected = annotator.parse(target, tokenise=True)
        trimmed = [
            _trim(annotator, original, corrected, e)
            for e in annotator.annotate(original, corrected)
        ]
        heard = _heard([c for c in trimmed if c is not None], original, corrected)
        result = []
        for change in heard:
            start, end = change.o_start, change.o_end
            if start < end:
                tokens = range(start, end)
            else:  # an insertion sits between two tokens; point at both
                tokens = range(max(start - 1, 0), min(start + 1, len(token_words)))
            edit = Edit(
                category=category,
                type=change.type,
                original=change.original,
                correction=change.correction,
                words=sorted({token_words[t] for t in tokens}),
            )
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


@dataclass(frozen=True)
class _Change:
    """An ERRANT edit as plain data, with half-open token spans in both texts."""

    o_start: int
    o_end: int
    c_start: int
    c_end: int
    original: str
    correction: str
    type: str

    @classmethod
    def of(cls, e: ErrantEdit) -> "_Change":
        return cls(
            int(e.o_start),
            int(e.o_end),
            int(e.c_start),
            int(e.c_end),
            str(e.o_str),
            str(e.c_str),
            str(e.type),
        )

    @property
    def spans(self) -> tuple[int, int, int, int]:
        return self.o_start, self.o_end, self.c_start, self.c_end


def _trim(
    annotator: Annotator, original: Doc, corrected: Doc, e: ErrantEdit
) -> _Change | None:
    """Cut silent tokens off both ends of an ERRANT edit and reclassify the rest.

    ERRANT merges a period the transcriber left out, or a capitalized "i",
    into the real change next to it ("go → goes." comes out as R:OTHER). A
    token at an edge is silent when it has no letters or digits, or when it
    sounds the same as the token across from it. Returns None when nothing
    is left.
    """
    change = _Change.of(e)
    o_start, o_end, c_start, c_end = change.spans

    def said(i: int) -> str:
        return _spoken(original[i].text)

    def heard(i: int) -> str:
        return _spoken(corrected[i].text)

    while True:
        both = o_start < o_end and c_start < c_end
        if both and said(o_start) == heard(c_start):
            o_start, c_start = o_start + 1, c_start + 1
        elif o_start < o_end and not said(o_start):
            o_start += 1
        elif c_start < c_end and not heard(c_start):
            c_start += 1
        elif both and said(o_end - 1) == heard(c_end - 1):
            o_end, c_end = o_end - 1, c_end - 1
        elif o_start < o_end and not said(o_end - 1):
            o_end -= 1
        elif c_start < c_end and not heard(c_end - 1):
            c_end -= 1
        else:
            break
    if (o_start, o_end, c_start, c_end) == change.spans:
        return change
    if o_start == o_end and c_start == c_end:
        return None
    return _Change.of(
        annotator.import_edit(original, corrected, [o_start, o_end, c_start, c_end])
    )


def _heard(changes: list[_Change], original: Doc, corrected: Doc) -> list[_Change]:
    """Drop every run of adjacent changes that together sound like what was said.

    ERRANT splits "Your → You're" into "Your → You" plus an inserted "'re".
    Neither sounds the same alone; together they do.
    """
    silent: set[int] = set()
    for first in range(len(changes)):
        for last in range(first, len(changes)):
            if last > first and not _adjacent(changes[last - 1], changes[last]):
                break
            said = original[changes[first].o_start : changes[last].o_end].text
            heard = corrected[changes[first].c_start : changes[last].c_end].text
            if _spoken(said) == _spoken(heard):
                silent.update(range(first, last + 1))
    return [c for i, c in enumerate(changes) if i not in silent]


def _adjacent(a: _Change, b: _Change) -> bool:
    return a.o_end == b.o_start and a.c_end == b.c_start


_SYMBOLS = {"&": "and", "%": "percent"}
# Spellings that sound the same in any accent, mapped to one of them.
_SAME_SOUND = {
    "youre": "your",
    "theyre": "their",
    "there": "their",
    "whos": "whose",
    "too": "to",
    "two": "to",
    "ok": "okay",
}
_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def _spoken(text: str) -> str:
    """Roughly what a listener hears, so two texts can be compared by ear.

    Drops case, punctuation, apostrophes, and spacing, says symbols and
    numbers in words, and folds homophones together.
    """
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, f" {word} ")
    sounds: list[str] = []
    for chunk in text.casefold().split():
        chunk = "".join(c for c in chunk if c.isalnum())
        if chunk.isascii() and chunk.isdigit() and int(chunk) < 1_000_000:
            sounds += _number(int(chunk)).split()
        else:
            sounds.append(chunk)
    return "".join(_SAME_SOUND.get(s, s) for s in sounds)


def _number(n: int) -> str:
    """English words for 0 <= n < 1,000,000, without "and" or hyphens."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        rest = f" {_ONES[n % 10]}" if n % 10 else ""
        return _TENS[n // 10 - 2] + rest
    if n < 1000:
        rest = f" {_number(n % 100)}" if n % 100 else ""
        return f"{_ONES[n // 100]} hundred{rest}"
    rest = f" {_number(n % 1000)}" if n % 1000 else ""
    return f"{_number(n // 1000)} thousand{rest}"


def _overlaps(a: _Span, b: _Span) -> bool:
    if a.is_insertion or b.is_insertion:
        return max(a.start, b.start) <= min(a.end, b.end)
    return not set(a.edit.words).isdisjoint(b.edit.words)
