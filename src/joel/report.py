"""Pipeline step 7: the terminal report, built from a CheckResult alone."""

from decimal import Decimal

from joel.check import CheckResult
from joel.confidence import FilteredEdit
from joel.correct import Correction
from joel.transcript import Word

_NO_TIME = "--:--.-–--:--.-"


def render(result: CheckResult, *, show_uncertain: bool) -> str:
    """The report for one run, without a trailing newline.

    Uncertain mistakes show only with `show_uncertain`; otherwise a line counts
    them. The last line is what the run's model calls cost.
    """
    words = result.transcript.words
    by_index = {w.index: w for w in words}
    transcriber = result.transcript.transcriber
    timed = not words or any(
        w.start_ms is not None and w.end_ms is not None for w in words
    )
    notes = []
    if not timed:
        notes.append(
            f"The {transcriber} transcriber gives no word timestamps, "
            "so mistakes are listed without times."
        )
    if words and all(w.confidence is None for w in words):
        notes.append(
            f"The {transcriber} transcriber gives no confidence scores, "
            "so no mistake was held back as uncertain."
        )

    entries = []
    hidden = 0
    for sentence in result.sentences:
        correction = sentence.correction
        if correction is None:
            continue  # every word was dropped, so nothing was diffed
        dropped = set(sentence.dropped)
        said = " ".join(by_index[i].text for i in sentence.words if i not in dropped)
        for filtered in sorted(sentence.edits, key=lambda f: f.edit.words[0]):
            if filtered.uncertain is not None and not show_uncertain:
                hidden += 1
                continue
            time = _span([by_index[i] for i in filtered.edit.words]) if timed else None
            entries.append(_entry(filtered, said, correction, time))

    tail = []
    if hidden:
        tail.append(
            "1 uncertain mistake hidden; --show-uncertain shows it."
            if hidden == 1
            else f"{hidden} uncertain mistakes hidden; --show-uncertain shows them."
        )
    tail.append(_cost(result))

    blocks = ["\n".join(notes)] if notes else []
    blocks += entries or ["No mistakes found."]
    blocks.append("\n".join(tail))
    return "\n\n".join(blocks)


def _entry(
    filtered: FilteredEdit, said: str, correction: Correction, time: str | None
) -> str:
    edit = filtered.edit
    if edit.category == "grammar":
        header = f"{edit.type:<12} grammar"
        fixed = f"  correct: {correction.grammatical}"
    else:
        header = "naturalness"
        fixed = f"  better:  {correction.natural}"
    if time is not None:
        header = f"{time}  {header}"
    if filtered.uncertain is not None:
        header += "  uncertain"
    lines = [
        header,
        f"  said:    {said}",
        fixed,
        f"           {edit.original or '∅'} → {edit.correction or '∅'}",
    ]
    if filtered.uncertain is not None:
        lines.append(f"  unsure:  {filtered.uncertain}")
    return "\n".join(lines)


def _span(words: list[Word]) -> str:
    times = [
        (w.start_ms, w.end_ms)
        for w in words
        if w.start_ms is not None and w.end_ms is not None
    ]
    if len(times) < len(words):
        return _NO_TIME
    return f"{_clock(min(s for s, _ in times))}–{_clock(max(e for _, e in times))}"


def _clock(ms: int) -> str:
    return f"{ms // 60_000:02d}:{ms // 1000 % 60:02d}.{ms // 100 % 10}"


def _cost(result: CheckResult) -> str:
    n = result.requests
    calls = f"{n} model call{'s' if n != 1 else ''}"
    tokens = f"{result.input_tokens} input and {result.output_tokens} output tokens"
    usd = result.cost_usd
    if usd is None:
        return (
            f"Cost: unknown, genai-prices has no price for {result.model} "
            f"({calls}, {tokens})."
        )
    if usd == 0:
        money = "$0.00"
    elif usd < Decimal("0.01"):
        money = "under $0.01"
    else:
        money = f"${usd:.2f}"
    return f"Cost: {money} for {calls} ({tokens}) with {result.model}."
