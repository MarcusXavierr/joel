"""Pipeline step 6: mark edits on words the transcriber wasn't sure it heard."""

from collections.abc import Sequence

from pydantic import BaseModel

from joel.edits import Edit
from joel.transcript import Word

MIN_CONFIDENCE = 0.67


class FilteredEdit(BaseModel):
    """An edit and whether the audio behind it is trustworthy enough to report."""

    edit: Edit
    # The lowest known confidence over edit.words; None when none is known.
    confidence: float | None
    # Why the edit is uncertain; None means it is kept.
    uncertain: str | None


def filter_edits(
    edits: Sequence[Edit],
    words: Sequence[Word],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[FilteredEdit]:
    """Mark each edit uncertain when one of its words is below `min_confidence`.

    `words` is the transcript. A word without a confidence never makes an edit
    uncertain. Returns one result per edit, in input order.
    """
    by_index = {w.index: w for w in words}
    results = []
    for edit in edits:
        scored: list[tuple[float, Word]] = []
        for i in edit.words:
            word = by_index.get(i)
            if word is None:
                raise ValueError(
                    f"edit points at word {i}, which is not in the transcript"
                )
            if word.confidence is not None:
                scored.append((word.confidence, word))
        if not scored:
            results.append(FilteredEdit(edit=edit, confidence=None, uncertain=None))
            continue
        # min() returns the first of equal minimums, so ties go to the earlier word.
        confidence, weakest = min(scored, key=lambda pair: pair[0])
        uncertain = None
        if confidence < min_confidence:
            uncertain = (
                f'"{weakest.text}" was heard with confidence {confidence:.2f}, '
                f"below the {min_confidence:.2f} threshold"
            )
        results.append(
            FilteredEdit(edit=edit, confidence=confidence, uncertain=uncertain)
        )
    return results
