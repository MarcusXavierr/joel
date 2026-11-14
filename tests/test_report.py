"""The terminal report, rebuilt from a CheckResult alone."""

from decimal import Decimal

import pytest

from joel.check import CheckedSentence, CheckResult
from joel.confidence import FilteredEdit
from joel.correct import Correction
from joel.edits import Edit
from joel.report import render
from joel.transcript import Transcript, Word

MODEL = "anthropic:claude-opus-5-5"
SAID = "There is many people in the the room."
UNSURE = '"is" was heard with confidence 0.50, below the 0.67 threshold'


def result(
    *,
    transcriber: str = "parakeet",
    timestamps: bool = True,
    confidence: float | None = 0.95,
    uncertain: str | None = None,
    cost: Decimal | None = Decimal("0.0731"),
    model: str = MODEL,
    requests: int = 2,
) -> CheckResult:
    """One sentence, its first "the" dropped, with a grammar and a naturalness edit."""
    words = [
        Word(
            index=i,
            text=t,
            start_ms=12_000 + 400 * i if timestamps else None,
            end_ms=12_300 + 400 * i if timestamps else None,
            confidence=confidence,
        )
        for i, t in enumerate(SAID.split(" "))
    ]
    grammar = Edit(
        category="grammar",
        type="R:VERB:SVA",
        original="is",
        correction="are",
        words=[1],
    )
    naturalness = Edit(
        category="naturalness",
        type="R:OTHER",
        original="many",
        correction="a lot of",
        words=[2],
    )
    return CheckResult(
        model=model,
        transcript=Transcript(transcriber=transcriber, words=words),
        sentences=[
            CheckedSentence(
                words=list(range(8)),
                dropped=[5],
                correction=Correction(
                    grammatical="There are many people in the room.",
                    natural="There are a lot of people in the room.",
                ),
                edits=[
                    FilteredEdit(
                        edit=grammar, confidence=confidence, uncertain=uncertain
                    ),
                    FilteredEdit(
                        edit=naturalness, confidence=confidence, uncertain=None
                    ),
                ],
            )
        ],
        requests=requests,
        input_tokens=100 * requests,
        output_tokens=10 * requests,
        cost_usd=cost,
    )


GRAMMAR = """\
00:12.4–00:12.7  R:VERB:SVA   grammar
  said:    There is many people in the room.
  correct: There are many people in the room.
           is → are"""
NATURALNESS = """\
00:12.8–00:13.1  naturalness
  said:    There is many people in the room.
  better:  There are a lot of people in the room.
           many → a lot of"""
COST = f"Cost: $0.07 for 2 model calls (200 input and 20 output tokens) with {MODEL}."


def test_each_mistake_shows_when_it_was_said_and_its_correction() -> None:
    assert render(result(), show_uncertain=False) == (
        f"{GRAMMAR}\n\n{NATURALNESS}\n\n{COST}"
    )


def test_a_transcriber_without_confidence_is_noted_once_at_the_top() -> None:
    report = render(result(confidence=None), show_uncertain=False)

    assert report == (
        "The parakeet transcriber gives no confidence scores, so no mistake was "
        f"held back as uncertain.\n\n{GRAMMAR}\n\n{NATURALNESS}\n\n{COST}"
    )


def test_a_transcriber_without_timestamps_drops_the_time_column() -> None:
    report = render(
        result(transcriber="openai", timestamps=False), show_uncertain=False
    )

    assert report == (
        "The openai transcriber gives no word timestamps, so mistakes are listed "
        "without times.\n\n"
        "R:VERB:SVA   grammar\n"
        "  said:    There is many people in the room.\n"
        "  correct: There are many people in the room.\n"
        "           is → are\n\n"
        "naturalness\n"
        "  said:    There is many people in the room.\n"
        "  better:  There are a lot of people in the room.\n"
        "           many → a lot of\n\n"
        f"{COST}"
    )


def test_an_uncertain_mistake_is_hidden_and_counted() -> None:
    report = render(result(uncertain=UNSURE), show_uncertain=False)

    assert report == (
        f"{NATURALNESS}\n\n"
        "1 uncertain mistake hidden; --show-uncertain shows it.\n"
        f"{COST}"
    )


def test_show_uncertain_marks_the_mistake_and_says_why() -> None:
    report = render(result(uncertain=UNSURE), show_uncertain=True)

    assert report == (
        "00:12.4–00:12.7  R:VERB:SVA   grammar  uncertain\n"
        "  said:    There is many people in the room.\n"
        "  correct: There are many people in the room.\n"
        "           is → are\n"
        f"  unsure:  {UNSURE}\n\n"
        f"{NATURALNESS}\n\n{COST}"
    )


def test_every_hidden_mistake_is_counted() -> None:
    checked = result(uncertain=UNSURE)
    checked.sentences[0].edits[1].uncertain = UNSURE

    assert render(checked, show_uncertain=False) == (
        "No mistakes found.\n\n"
        "2 uncertain mistakes hidden; --show-uncertain shows them.\n"
        f"{COST}"
    )


def test_a_sentence_without_mistakes_reports_none() -> None:
    checked = result()
    checked.sentences[0].edits = []

    assert render(checked, show_uncertain=False) == f"No mistakes found.\n\n{COST}"


def test_entries_follow_spoken_order_whatever_the_edit_order() -> None:
    checked = result()
    checked.sentences[0].edits.reverse()  # naturalness first, as if diffed first

    assert render(checked, show_uncertain=False) == (
        f"{GRAMMAR}\n\n{NATURALNESS}\n\n{COST}"
    )


def test_a_word_without_a_time_gets_a_placeholder_span() -> None:
    checked = result()
    checked.transcript.words[1].end_ms = None  # "is"

    report = render(checked, show_uncertain=False)

    assert report.startswith("--:--.-–--:--.-  R:VERB:SVA   grammar\n")
    assert f"\n\n{NATURALNESS}\n\n" in report


def test_a_model_without_a_price_says_the_cost_is_unknown() -> None:
    report = render(result(cost=None, model="ollama:llama3"), show_uncertain=False)

    assert report.splitlines()[-1] == (
        "Cost: unknown, genai-prices has no price for ollama:llama3 "
        "(2 model calls, 200 input and 20 output tokens)."
    )


@pytest.mark.parametrize(
    ("cost", "requests", "line"),
    [
        (
            Decimal(0),
            1,
            "Cost: $0.00 for 1 model call (100 input and 10 output tokens)",
        ),
        (
            Decimal("0.004"),
            1,
            "Cost: under $0.01 for 1 model call (100 input and 10 output tokens)",
        ),
        (
            Decimal("1.236"),
            3,
            "Cost: $1.24 for 3 model calls (300 input and 30 output tokens)",
        ),
    ],
)
def test_the_cost_line_rounds_to_cents(cost: Decimal, requests: int, line: str) -> None:
    report = render(result(cost=cost, requests=requests), show_uncertain=False)

    assert report.splitlines()[-1] == f"{line} with {MODEL}."
