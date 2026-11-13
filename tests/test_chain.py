"""Steps 2, 5, and 6 chained over the design's example, with step 4 canned."""

from errant.annotator import Annotator  # type: ignore[import-untyped]

from joel.confidence import filter_edits
from joel.edits import Edit, diff_edits
from joel.sentences import split_sentences
from joel.transcript import Word

SPOKEN = (
    "Yesterday he go to school, uh, and she don't like it. "
    "There is many people in the the room. I am agree with you."
)
# What the LLM would return for each sentence (step 4 belongs to #6).
DROPPED = {5, 16}  # "uh," and the first "the"
CORRECTED = [
    "Yesterday he went to school, and she doesn't like it.",
    "There are many people in the room.",
    "I agree with you.",
]


def test_known_mistakes_come_out_and_disfluencies_do_not(
    annotator: Annotator,
) -> None:
    transcript = [
        Word(index=i, text=t, start_ms=None, end_ms=None, confidence=0.95)
        for i, t in enumerate(SPOKEN.split(" "))
    ]
    by_index = {w.index: w for w in transcript}

    sentences = split_sentences(transcript)
    assert sentences == [list(range(0, 11)), list(range(11, 19)), list(range(19, 24))]

    edits: list[Edit] = []
    for sentence, corrected in zip(sentences, CORRECTED, strict=True):
        clean = [by_index[i] for i in sentence if i not in DROPPED]
        edits += diff_edits(clean, corrected, corrected, annotator)
    results = filter_edits(edits, transcript)

    assert all(r.uncertain is None for r in results)
    assert [
        (r.edit.category, r.edit.type, r.edit.original, r.edit.correction, r.edit.words)
        for r in results
    ] == [
        ("grammar", "R:VERB:TENSE", "go", "went", [2]),
        ("grammar", "R:VERB:SVA", "do", "does", [8]),
        ("grammar", "R:VERB:SVA", "is", "are", [12]),
        ("grammar", "U:VERB:TENSE", "am", "", [20]),
    ]
