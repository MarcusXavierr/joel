"""`joel bench`: how faithfully each transcriber writes down what I said.

A bench directory holds ``NN.wav`` recordings, each with an ``NN.txt`` of what
I actually said, mistakes included. Mistakes are marked with brackets, as in
``Yesterday he [go] to school``. Words are compared ignoring case and
punctuation; numbers aren't normalized, so "1" against "one" is a divergence.
"""

import functools
import itertools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from joel.decode import SAMPLE_RATE, decode
from joel.transcribers import Transcriber, load_transcriber
from joel.transcript import Transcript, Word

_BYTES_PER_SAMPLE = 4  # decode() returns float32 PCM

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


def normalize(text: str) -> list[str]:
    """Split ``text`` into lowercase words, without punctuation."""
    text = text.casefold().translate(_APOSTROPHES)
    spaced = "".join(c if c.isalnum() or c == "'" else " " for c in text)
    return [word for token in spaced.split() if (word := token.strip("'"))]


class GoldFormatError(ValueError):
    """A bench ``.txt`` whose bracket markup can't be read."""


@dataclass(frozen=True)
class Gold:
    """What I said: its words, and which of them are my mistakes."""

    words: list[str]
    # Half-open [start, end) ranges into words, one per bracketed mistake.
    mistakes: list[tuple[int, int]]


def parse_gold(text: str) -> Gold:
    """Read a bench ``.txt``, where ``[brackets]`` mark my mistakes."""
    words: list[str] = []
    mistakes: list[tuple[int, int]] = []
    segment_start = 0
    open_at: int | None = None
    for i, char in enumerate(text):
        if char == "[":
            if open_at is not None:
                raise GoldFormatError(f"nested '[' at column {i + 1}")
            if i > 0 and _is_word_char(text[i - 1]):
                raise GoldFormatError(_PARTIAL_WORD.format(column=i + 1))
            words.extend(normalize(text[segment_start:i]))
            open_at = i
            segment_start = i + 1
        elif char == "]":
            if open_at is None:
                raise GoldFormatError(f"']' without '[' at column {i + 1}")
            inside = normalize(text[segment_start:i])
            if not inside:
                raise GoldFormatError(
                    f"empty mistake '{text[open_at : i + 1]}' at column {open_at + 1}"
                )
            if i + 1 < len(text) and _is_word_char(text[i + 1]):
                raise GoldFormatError(_PARTIAL_WORD.format(column=i + 1))
            mistakes.append((len(words), len(words) + len(inside)))
            words.extend(inside)
            open_at = None
            segment_start = i + 1
    if open_at is not None:
        raise GoldFormatError(f"unclosed '[' at column {open_at + 1}")
    words.extend(normalize(text[segment_start:]))
    if not words:
        raise GoldFormatError("no words")
    return Gold(words=words, mistakes=mistakes)


_PARTIAL_WORD = "brackets must enclose whole words, at column {column}"


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char in "'’‘"


@dataclass(frozen=True)
class Step:
    """One step of the word alignment from what I said to what was heard."""

    op: Literal["equal", "sub", "del", "ins"]
    gold: str | None  # None for "ins"
    heard: str | None  # None for "del"
    gold_index: int | None  # position in the gold words; None for "ins"


def align(gold: list[str], heard: list[str]) -> list[Step]:
    """Align two word lists with the fewest substitutions, deletions, insertions."""
    rows, cols = len(gold) + 1, len(heard) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            same = gold[i - 1] == heard[j - 1]
            dp[i][j] = min(
                dp[i - 1][j - 1] + (0 if same else 1),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    steps: list[Step] = []
    i, j = len(gold), len(heard)
    while i > 0 or j > 0:
        diagonal = dp[i - 1][j - 1] if i > 0 and j > 0 else None
        if (
            diagonal is not None
            and gold[i - 1] == heard[j - 1]
            and dp[i][j] == diagonal
        ):
            steps.append(Step("equal", gold[i - 1], heard[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif diagonal is not None and dp[i][j] == diagonal + 1:
            steps.append(Step("sub", gold[i - 1], heard[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            steps.append(Step("del", gold[i - 1], None, i - 1))
            i -= 1
        else:
            steps.append(Step("ins", None, heard[j - 1], None))
            j -= 1
    steps.reverse()
    return steps


def diverging(steps: list[Step]) -> int:
    """How many steps differ from what I said; WER is this over the gold words."""
    return sum(step.op != "equal" for step in steps)


def mistakes_kept(steps: list[Step], mistakes: list[tuple[int, int]]) -> int:
    """How many of my mistakes the transcript wrote down word for word.

    A mistake is kept when every one of its words came through unchanged and no
    word was inserted between two of them. Insertions at its edges don't count.
    """
    lost: set[int] = set()
    previous: int | None = None  # gold index of the last non-insertion step
    inserted = False  # whether a word was inserted since that step
    for step in steps:
        if step.gold_index is None:
            inserted = True
            continue
        if inserted and previous is not None:
            lost.update(_spanning(previous, step.gold_index, mistakes))
        if step.op != "equal":
            lost.update(_spanning(step.gold_index, step.gold_index, mistakes))
        previous, inserted = step.gold_index, False
    return len(mistakes) - len(lost)


def _spanning(first: int, last: int, mistakes: list[tuple[int, int]]) -> set[int]:
    """The mistakes that contain both gold index ``first`` and ``last``."""
    return {n for n, (s, e) in enumerate(mistakes) if s <= first and last < e}


def count_confident(words: list[Word]) -> int:
    """How many words carry a confidence above zero."""
    return sum(w.confidence is not None and w.confidence > 0 for w in words)


@dataclass(frozen=True)
class TimestampCheck:
    """Whether a transcript's word times make sense for its audio."""

    status: Literal["ok", "failed", "n/a"]
    reason: str | None  # why it failed; None otherwise


def check_timestamps(
    transcript: Transcript, duration_ms: Callable[[], int]
) -> TimestampCheck:
    """Check that word times exist, run forward, and stay inside the audio.

    ``duration_ms`` is only called when some word has a time, so a transcriber
    without timestamps never costs a decode.
    """
    words = transcript.words
    if all(w.start_ms is None and w.end_ms is None for w in words):
        return TimestampCheck("n/a", None)
    duration = duration_ms()
    previous: Word | None = None
    for word in words:
        reason = _timestamp_problem(word, previous, duration)
        if reason is not None:
            return TimestampCheck("failed", reason)
        previous = word
    return TimestampCheck("ok", None)


def _timestamp_problem(word: Word, previous: Word | None, duration: int) -> str | None:
    label = f"word {word.index} '{word.text}'"
    start, end = word.start_ms, word.end_ms
    if start is None or end is None:
        return f"{label} has no timestamps"
    if start < 0:
        return f"{label} starts at {start} ms, before 0"
    if start > end:
        return f"{label} ends at {end} ms, before it starts at {start} ms"
    if previous is not None and previous.start_ms is not None:
        if start < previous.start_ms:
            return (
                f"{label} starts at {start} ms, before word {previous.index} "
                f"'{previous.text}' at {previous.start_ms} ms"
            )
    if end > duration:
        return f"{label} ends at {end} ms, after the audio ends at {duration} ms"
    return None


def audio_duration_ms(audio: Path) -> int:
    """The length of ``audio`` in milliseconds, rounded up."""
    samples = len(decode(audio)) / _BYTES_PER_SAMPLE
    return math.ceil(samples * 1000 / SAMPLE_RATE)


@dataclass(frozen=True)
class Recording:
    """One bench recording and what I actually said in it."""

    name: str  # the file stem, "01" for 01.wav
    audio: Path
    gold: Gold


class BenchInputError(Exception):
    """A bench directory the bench can't run on, with every problem listed."""


def load_recordings(directory: Path) -> list[Recording]:
    """Pair every ``NN.wav`` in ``directory`` with its ``NN.txt``, sorted by name.

    Raises BenchInputError listing every problem found, so one run shows them all.
    """
    if not directory.is_dir():
        raise _input_error(directory, [f"{directory} is not a directory"])
    wavs = {p.stem: p for p in directory.glob("*.wav")}
    txts = {p.stem: p for p in directory.glob("*.txt")}
    problems = [] if wavs else ["no .wav files"]
    recordings: list[Recording] = []
    for name in sorted(wavs.keys() | txts.keys()):
        if name not in txts:
            problems.append(f"{name}.wav has no matching {name}.txt")
            continue
        if name not in wavs:
            problems.append(f"{name}.txt has no matching {name}.wav")
        try:
            gold = parse_gold(txts[name].read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            problems.append(f"{name}.txt is not UTF-8 text")
            continue
        except GoldFormatError as exc:
            problems.append(f"{name}.txt: {exc}")
            continue
        if name in wavs:
            recordings.append(Recording(name, wavs[name], gold))
    if problems:
        raise _input_error(directory, problems)
    return recordings


def _input_error(directory: Path, problems: list[str]) -> BenchInputError:
    lines = [f"Can't run the bench on {directory}:"]
    lines += [f"  - {problem}" for problem in problems]
    return BenchInputError("\n".join(lines))


@dataclass(frozen=True)
class FileResult:
    """How one transcriber did on one recording."""

    name: str
    steps: list[Step]
    gold_words: int
    heard_words: int  # words in the raw transcript, before normalizing
    confident_words: int
    mistakes: int
    mistakes_kept: int
    timestamps: TimestampCheck


def measure_file(
    recording: Recording, transcript: Transcript, duration_ms: Callable[[], int]
) -> FileResult:
    """Compare one transcript with what I actually said."""
    heard = normalize(" ".join(word.text for word in transcript.words))
    steps = align(recording.gold.words, heard)
    return FileResult(
        name=recording.name,
        steps=steps,
        gold_words=len(recording.gold.words),
        heard_words=len(transcript.words),
        confident_words=count_confident(transcript.words),
        mistakes=len(recording.gold.mistakes),
        mistakes_kept=mistakes_kept(steps, recording.gold.mistakes),
        timestamps=check_timestamps(transcript, duration_ms),
    )


@dataclass(frozen=True)
class FileRun:
    """One recording through one transcriber: a result, or why there is none."""

    name: str
    result: FileResult | None
    error: str | None


@dataclass(frozen=True)
class TranscriberRun:
    """Every recording through one transcriber, or why it couldn't load."""

    name: str
    files: list[FileRun]
    error: str | None


def run_bench(
    recordings: list[Recording],
    names: list[str],
    load: Callable[[str], Transcriber] = load_transcriber,
    duration: Callable[[Path], int] = audio_duration_ms,
) -> Iterator[TranscriberRun]:
    """Run each named transcriber over every recording, yielding as each ends.

    A transcriber that fails to load, or fails on one file, is reported in its
    run, and the bench goes on with the rest.
    """
    cached_duration = functools.cache(duration)
    for name in names:
        try:
            transcriber = load(name)
        except Exception as exc:
            yield TranscriberRun(name, [], _describe(exc))
            continue
        files: list[FileRun] = []
        for recording in recordings:
            duration_ms = functools.partial(cached_duration, recording.audio)
            try:
                transcript = transcriber.transcribe(recording.audio)
                result = measure_file(recording, transcript, duration_ms)
            except Exception as exc:
                files.append(FileRun(recording.name, None, _describe(exc)))
                continue
            files.append(FileRun(recording.name, result, None))
        yield TranscriberRun(name, files, None)


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


_NONE = "–"
_HEADERS = (
    "file",
    "words",
    "diverging",
    "WER",
    "mistakes kept",
    "confidence",
    "timestamps",
)
_LEFT_ALIGNED = {0, len(_HEADERS) - 1}  # file and timestamps


def render_run(run: TranscriberRun) -> str:
    """One transcriber's table, then its diffs, timestamp failures, and errors."""
    title = f"== {run.name} =="
    if run.error is not None:
        return f"{title}\nerror: {run.error}"
    results = [file.result for file in run.files if file.result is not None]
    rows = [list(_HEADERS)]
    for file in run.files:
        if file.result is None:
            rows.append([file.name, *[_NONE] * (len(_HEADERS) - 2), "error"])
        else:
            status = file.result.timestamps.status
            rows.append([*_counts(file.name, [file.result]), status])
    if results:
        rows.append([*_counts("total", results), ""])
    diffs = [
        f"{r.name}  {_inline_diff(r.steps)}" for r in results if diverging(r.steps)
    ]
    failed = [r for r in results if r.timestamps.status == "failed"]
    errors = [f"{f.name}  {f.error}" for f in run.files if f.error is not None]
    sections = {
        "Diffs:": diffs,
        "Timestamps:": [f"{r.name}  {r.timestamps.reason}" for r in failed],
        "Errors:": errors,
    }
    blocks = [f"{title}\n{_table(rows)}"]
    blocks += [
        "\n".join([heading, *lines]) for heading, lines in sections.items() if lines
    ]
    return "\n\n".join(blocks)


def _counts(name: str, results: list[FileResult]) -> list[str]:
    """The numeric cells of a row, summed over ``results``."""
    gold = sum(r.gold_words for r in results)
    diverged = sum(diverging(r.steps) for r in results)
    mistakes = sum(r.mistakes for r in results)
    kept = sum(r.mistakes_kept for r in results)
    heard = sum(r.heard_words for r in results)
    confident = sum(r.confident_words for r in results)
    return [
        name,
        str(gold),
        str(diverged),
        f"{diverged / gold:.1%}",
        f"{kept}/{mistakes}" if mistakes else _NONE,
        f"{confident / heard:.0%}" if heard else _NONE,
    ]


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(row[col]) for row in rows) for col in range(len(_HEADERS))]
    lines = []
    for row in rows:
        cells = [
            cell.ljust(width) if col in _LEFT_ALIGNED else cell.rjust(width)
            for col, (cell, width) in enumerate(zip(row, widths, strict=True))
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def _inline_diff(steps: list[Step]) -> str:
    """What I said, with ``[said → heard]`` where the transcript diverged."""
    parts: list[str] = []
    for equal, group in itertools.groupby(steps, key=lambda s: s.op == "equal"):
        run = list(group)
        gold = [s.gold for s in run if s.gold is not None]
        if equal:
            parts += gold
            continue
        heard = [s.heard for s in run if s.heard is not None]
        parts.append(f"[{' '.join(gold) or '∅'} → {' '.join(heard) or '∅'}]")
    return " ".join(parts)
