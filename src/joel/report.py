"""Pipeline step 7: the terminal report, built from a CheckResult alone."""

from joel.check import CheckResult
from joel.confidence import FilteredEdit
from joel.cost import dollars
from joel.edits import Edit
from joel.transcript import Word

_NO_TIME = "--:--.-–--:--.-"
_NO_START = "--:--.-"
_INDENT = " " * 11


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
        shown = []
        for filtered in sorted(sentence.edits, key=lambda f: f.edit.words[0]):
            if filtered.uncertain is not None and not show_uncertain:
                hidden += 1
            else:
                shown.append(filtered)
        if not shown:
            continue
        dropped = set(sentence.dropped)
        said = " ".join(by_index[i].text for i in sentence.words if i not in dropped)
        lines = [_span([by_index[i] for i in sentence.words])] if timed else []
        lines.append(f"  said:    {said}")
        width = max(len(_change(f.edit)) for f in shown)
        for category, fixed in (
            ("grammar", f"  correct: {correction.grammatical}"),
            ("naturalness", f"  better:  {correction.natural}"),
        ):
            edits = [f for f in shown if f.edit.category == category]
            if edits:
                lines.append(fixed)
            for filtered in edits:
                when = [by_index[i] for i in filtered.edit.words]
                lines += _change_lines(filtered, width, _start(when) if timed else None)
        entries.append("\n".join(lines))

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


def _change(edit: Edit) -> str:
    return f"{edit.original or '∅'} → {edit.correction or '∅'}"


def _change_lines(filtered: FilteredEdit, width: int, start: str | None) -> list[str]:
    edit = filtered.edit
    label = f"{edit.type:<12} grammar" if edit.category == "grammar" else "naturalness"
    if filtered.uncertain is not None:
        label += "  uncertain"
    when = f"{start}  " if start is not None else ""
    lines = [f"{_INDENT}{when}{_change(edit).ljust(width)}  {label}"]
    if filtered.uncertain is not None:
        lines.append(f"{_INDENT}unsure:  {filtered.uncertain}")
    return lines


def _start(words: list[Word]) -> str:
    starts = [w.start_ms for w in words if w.start_ms is not None]
    return _clock(min(starts)) if len(starts) == len(words) else _NO_START


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
    read = result.cache_read_tokens
    cached = f" ({read} read from cache)" if read else ""
    tokens = (
        f"{result.input_tokens} input{cached} and {result.output_tokens} output tokens"
    )
    usd = result.cost_usd
    if usd is None:
        return (
            f"Cost: unknown, genai-prices has no price for {result.model} "
            f"({calls}, {tokens})."
        )
    return f"Cost: {dollars(usd)} for {calls} ({tokens}) with {result.model}."
