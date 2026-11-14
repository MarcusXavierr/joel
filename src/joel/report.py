"""Pipeline step 7: the report, a document built from a CheckResult alone.

`--json` writes the document; the terminal shows `render` of it, so the two
can't disagree.
"""

from decimal import Decimal

from pydantic import BaseModel

from joel.check import CheckResult
from joel.cost import dollars
from joel.edits import Category
from joel.transcript import Word

_NO_START = "--:--.-"
_NO_TIME = f"{_NO_START}–{_NO_START}"
_INDENT = " " * 11


class Mistake(BaseModel):
    """One change to what was said, and when it was said."""

    category: Category
    type: str  # ERRANT type, e.g. "R:VERB:SVA"
    original: str  # "" for an insertion
    correction: str  # "" for a deletion
    start_ms: int | None  # None when one of its words has no start
    end_ms: int | None  # None when one of its words has no end
    confidence: float | None  # the lowest known over its words
    uncertain: str | None  # why it is held back; None means it is shown


class ReportSentence(BaseModel):
    """A sentence with at least one mistake."""

    start_ms: int | None  # the whole sentence, dropped words included; both
    end_ms: int | None  # are None unless every word has a start and an end
    said: str  # without the dropped words
    correct: str
    better: str | None  # None when the run didn't ask for it
    mistakes: list[Mistake]  # in spoken order


class Report(BaseModel):
    """Every mistake one run found, uncertain ones included, and what it cost."""

    transcriber: str
    timestamps: bool  # whether the transcriber gave word times
    confidence_scores: bool  # whether it gave word confidence
    prompts: dict[str, str]  # each prompt's name -> sha256 hex of the text used
    sentences: list[ReportSentence]
    model: str
    requests: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: Decimal | None


def build_report(result: CheckResult) -> Report:
    """The report of a run: each corrected sentence with a mistake, all of them."""
    words = result.transcript.words
    by_index = {w.index: w for w in words}
    sentences = []
    for sentence in result.sentences:
        correction = sentence.correction
        if correction is None or not sentence.edits:
            continue
        indices = range(sentence.first_word, sentence.last_word + 1)
        dropped = set(sentence.dropped)
        start, end = _span([by_index[i] for i in indices])
        mistakes = []
        for filtered in sorted(sentence.edits, key=lambda f: f.edit.words[0]):
            edit = filtered.edit
            said = [by_index[i] for i in edit.words]
            mistakes.append(
                Mistake(
                    category=edit.category,
                    type=edit.type,
                    original=edit.original,
                    correction=edit.correction,
                    start_ms=_first([w.start_ms for w in said]),
                    end_ms=_last([w.end_ms for w in said]),
                    confidence=filtered.confidence,
                    uncertain=filtered.uncertain,
                )
            )
        sentences.append(
            ReportSentence(
                start_ms=start,
                end_ms=end,
                said=" ".join(by_index[i].text for i in indices if i not in dropped),
                correct=correction.grammatical,
                better=correction.natural,
                mistakes=mistakes,
            )
        )
    return Report(
        transcriber=result.transcript.transcriber,
        timestamps=not words
        or any(w.start_ms is not None and w.end_ms is not None for w in words),
        confidence_scores=not words or any(w.confidence is not None for w in words),
        prompts=result.prompts,
        sentences=sentences,
        model=result.model,
        requests=result.requests,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_write_tokens=result.cache_write_tokens,
        cost_usd=result.cost_usd,
    )


def render(report: Report, *, show_uncertain: bool, better: bool) -> str:
    """The terminal text of a report, without a trailing newline.

    Naturalness mistakes show only with `better`. Uncertain mistakes show only
    with `show_uncertain`; otherwise a line counts them. The last line is what
    the run's model calls cost.
    """
    transcriber = report.transcriber
    notes = []
    if not report.timestamps:
        notes.append(
            f"The {transcriber} transcriber gives no word timestamps, "
            "so mistakes are listed without times."
        )
    if not report.confidence_scores:
        notes.append(
            f"The {transcriber} transcriber gives no confidence scores, "
            "so no mistake was held back as uncertain."
        )

    entries = []
    hidden = 0
    for sentence in report.sentences:
        shown = []
        for mistake in sentence.mistakes:
            if mistake.category != "grammar" and not better:
                continue
            if mistake.uncertain is not None and not show_uncertain:
                hidden += 1
            else:
                shown.append(mistake)
        if not shown:
            continue
        lines = [f"  said:    {sentence.said}"]
        if report.timestamps:
            lines.insert(0, _clock_span(sentence.start_ms, sentence.end_ms))
        width = max(len(_change(m)) for m in shown)
        for category, fixed in (
            ("grammar", f"  correct: {sentence.correct}"),
            ("naturalness", f"  better:  {sentence.better}"),
        ):
            mistakes = [m for m in shown if m.category == category]
            if mistakes:
                lines.append(fixed)
            for mistake in mistakes:
                start = _start(mistake.start_ms) if report.timestamps else None
                lines += _change_lines(mistake, width, start)
        entries.append("\n".join(lines))

    tail = []
    if hidden:
        tail.append(
            "1 uncertain mistake hidden; --show-uncertain shows it."
            if hidden == 1
            else f"{hidden} uncertain mistakes hidden; --show-uncertain shows them."
        )
    tail.append(_cost(report))

    blocks = ["\n".join(notes)] if notes else []
    blocks += entries or ["No mistakes found."]
    blocks.append("\n".join(tail))
    return "\n\n".join(blocks)


def _change(mistake: Mistake) -> str:
    return f"{mistake.original or '∅'} → {mistake.correction or '∅'}"


def _change_lines(mistake: Mistake, width: int, start: str | None) -> list[str]:
    if mistake.category == "grammar":
        label = f"{mistake.type:<12} grammar"
    else:
        label = "naturalness"
    if mistake.uncertain is not None:
        label += "  uncertain"
    when = f"{start}  " if start is not None else ""
    lines = [f"{_INDENT}{when}{_change(mistake).ljust(width)}  {label}"]
    if mistake.uncertain is not None:
        lines.append(f"{_INDENT}unsure:  {mistake.uncertain}")
    return lines


def _first(times: list[int | None]) -> int | None:
    known = [t for t in times if t is not None]
    return min(known) if known and len(known) == len(times) else None


def _last(times: list[int | None]) -> int | None:
    known = [t for t in times if t is not None]
    return max(known) if known and len(known) == len(times) else None


def _span(words: list[Word]) -> tuple[int | None, int | None]:
    if any(w.start_ms is None or w.end_ms is None for w in words):
        return None, None
    return _first([w.start_ms for w in words]), _last([w.end_ms for w in words])


def _start(ms: int | None) -> str:
    return _NO_START if ms is None else _clock(ms)


def _clock_span(start_ms: int | None, end_ms: int | None) -> str:
    if start_ms is None or end_ms is None:
        return _NO_TIME
    return f"{_clock(start_ms)}–{_clock(end_ms)}"


def _clock(ms: int) -> str:
    return f"{ms // 60_000:02d}:{ms // 1000 % 60:02d}.{ms // 100 % 10}"


def _cost(report: Report) -> str:
    n = report.requests
    calls = f"{n} model call{'s' if n != 1 else ''}"
    read = report.cache_read_tokens
    cached = f" ({read} read from cache)" if read else ""
    tokens = (
        f"{report.input_tokens} input{cached} and {report.output_tokens} output tokens"
    )
    usd = report.cost_usd
    if usd is None:
        return (
            f"Cost: unknown, genai-prices has no price for {report.model} "
            f"({calls}, {tokens})."
        )
    return f"Cost: {dollars(usd)} for {calls} ({tokens}) with {report.model}."
