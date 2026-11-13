from collections.abc import Callable
from pathlib import Path

import pytest

from joel.bench import (
    BenchInputError,
    FileResult,
    FileRun,
    Gold,
    GoldFormatError,
    Recording,
    Step,
    TimestampCheck,
    TranscriberRun,
    align,
    check_timestamps,
    count_confident,
    diverging,
    load_recordings,
    measure_file,
    mistakes_kept,
    normalize,
    parse_gold,
    render_run,
    run_bench,
)
from joel.transcribers import Transcriber
from joel.transcript import Transcript, Word


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("the, the", ["the", "the"]),
        ("don’t", ["don't"]),
        ("'cause", ["cause"]),
        ("  \n ", []),
    ],
)
def test_normalize_ignores_case_and_punctuation(text: str, words: list[str]) -> None:
    assert normalize(text) == words


def test_plain_text_has_no_mistakes() -> None:
    assert parse_gold("He went to school.") == Gold(
        words=["he", "went", "to", "school"], mistakes=[]
    )


def test_bracketed_word_is_a_mistake() -> None:
    assert parse_gold("Yesterday he [go] to school") == Gold(
        words=["yesterday", "he", "go", "to", "school"], mistakes=[(2, 3)]
    )
    assert parse_gold("[go] home") == Gold(words=["go", "home"], mistakes=[(0, 1)])


def test_every_bracket_is_a_mistake() -> None:
    gold = parse_gold("She [don't] like it. There [is] many people.")

    assert gold.mistakes == [(1, 2), (5, 6)]
    assert [gold.words[s] for s, _ in gold.mistakes] == ["don't", "is"]


def test_a_mistake_can_span_several_words() -> None:
    assert parse_gold("I [am agree] with you").mistakes == [(1, 3)]


def test_punctuation_next_to_brackets_is_fine() -> None:
    gold = parse_gold('Yes, [he go]. Then "[they is]" here')

    assert gold.words == ["yes", "he", "go", "then", "they", "is", "here"]
    assert gold.mistakes == [(1, 3), (4, 6)]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a [b [c] d]", "nested '[' at column 6"),
        ("a b] c", "']' without '[' at column 4"),
        ("a [b c", "unclosed '[' at column 3"),
        ("a [ , ] b", "empty mistake '[ , ]' at column 3"),
        ("a don’[t] d", "brackets must enclose whole words, at column 7"),
        ("a w[b] c", "brackets must enclose whole words, at column 4"),
        ("a [b]c d", "brackets must enclose whole words, at column 5"),
        ("a don'[t] d", "brackets must enclose whole words, at column 7"),
        (" ... ", "no words"),
    ],
)
def test_malformed_gold_is_rejected(text: str, message: str) -> None:
    with pytest.raises(GoldFormatError) as excinfo:
        parse_gold(text)

    assert str(excinfo.value) == message


def _kept(gold_text: str, heard_text: str) -> int:
    gold = parse_gold(gold_text)
    return mistakes_kept(align(gold.words, normalize(heard_text)), gold.mistakes)


def test_identical_transcript_diverges_nowhere() -> None:
    gold = parse_gold("Yesterday he [go] to school, and she [don't] like it.")
    steps = align(
        gold.words, normalize("yesterday he go to school and she don't like it")
    )

    assert diverging(steps) == 0
    assert all(step.op == "equal" for step in steps)
    assert mistakes_kept(steps, gold.mistakes) == 2


def test_case_and_punctuation_are_not_divergences() -> None:
    steps = align(normalize("He went, to school."), normalize("he WENT to... school"))

    assert diverging(steps) == 0


def test_fixed_grammar_is_a_substitution_that_loses_the_mistake() -> None:
    gold = parse_gold("he [go] to school")
    steps = align(gold.words, normalize("He went to school."))

    assert [s for s in steps if s.op != "equal"] == [
        Step(op="sub", gold="go", heard="went", gold_index=1)
    ]
    assert diverging(steps) == 1
    assert mistakes_kept(steps, gold.mistakes) == 0


def test_a_mistake_with_a_missing_word_is_lost() -> None:
    assert _kept("I [am agree] with you", "I agree with you") == 0


def test_a_word_inserted_inside_a_mistake_loses_it() -> None:
    assert _kept("I [am agree] with you", "I am uh agree with you") == 0


def test_a_word_inserted_at_the_edge_of_a_mistake_keeps_it() -> None:
    assert _kept("he [go] to", "he uh go to") == 1
    assert _kept("he [go] to", "he go uh to") == 1
    assert _kept("[go] home", "uh go home") == 1
    assert _kept("he [go] [is] wrong", "he go uh is wrong") == 2


def test_a_repeated_word_is_deleted_at_its_first_occurrence() -> None:
    steps = align(normalize("in the the room"), normalize("in the room"))

    assert [s for s in steps if s.op != "equal"] == [
        Step(op="del", gold="the", heard=None, gold_index=1)
    ]


def test_insertion_steps_carry_no_gold() -> None:
    steps = align(["a", "b"], ["a", "x", "b"])

    assert steps == [
        Step(op="equal", gold="a", heard="a", gold_index=0),
        Step(op="ins", gold=None, heard="x", gold_index=None),
        Step(op="equal", gold="b", heard="b", gold_index=1),
    ]


def _word(
    index: int,
    text: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    confidence: float | None = None,
) -> Word:
    return Word(
        index=index,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=confidence,
    )


def _transcript(*words: Word) -> Transcript:
    return Transcript(transcriber="fake", words=list(words))


def _duration(ms: int) -> Callable[[], int]:
    return lambda: ms


def _no_duration() -> int:
    raise AssertionError("duration was read")


def test_only_words_with_a_confidence_above_zero_are_confident() -> None:
    words = [
        _word(0, "a", confidence=0.9),
        _word(1, "b", confidence=0.0),
        _word(2, "c", confidence=None),
        _word(3, "d", confidence=0.1),
    ]

    assert count_confident(words) == 2


def test_a_transcript_with_no_words_shows_no_confidence() -> None:
    result = _result("01", "he [go]", "")

    rendered = render_run(TranscriberRun("echo", [FileRun("01", result, None)], None))

    row = rendered.splitlines()[2]
    assert row.split() == ["01", "2", "2", "100.0%", "0/1", "–", "n/a"]


def test_no_timestamps_is_not_applicable_and_skips_decoding() -> None:
    transcript = _transcript(_word(0, "a"), _word(1, "b"))

    assert check_timestamps(transcript, _no_duration) == TimestampCheck("n/a", None)
    assert check_timestamps(_transcript(), _no_duration) == TimestampCheck("n/a", None)


def test_ordered_timestamps_inside_the_audio_are_ok() -> None:
    # Shared starts, zero-length words and ending right at the audio's end are fine.
    transcript = _transcript(
        _word(0, "a", 0, 100), _word(1, "b", 0, 0), _word(2, "c", 150, 900)
    )

    assert check_timestamps(transcript, _duration(900)) == TimestampCheck("ok", None)


@pytest.mark.parametrize(
    ("words", "reason"),
    [
        (
            [_word(0, "a", 0, 10), _word(1, "go", 20, None)],
            "word 1 'go' has no timestamps",
        ),
        ([_word(0, "go", -5, 10)], "word 0 'go' starts at -5 ms, before 0"),
        (
            [_word(0, "go", 50, 40)],
            "word 0 'go' ends at 40 ms, before it starts at 50 ms",
        ),
        (
            [_word(0, "he", 300, 400), _word(1, "go", 200, 500)],
            "word 1 'go' starts at 200 ms, before word 0 'he' at 300 ms",
        ),
        (
            [_word(0, "he", 0, 400), _word(1, "go", 500, 9000)],
            "word 1 'go' ends at 9000 ms, after the audio ends at 8729 ms",
        ),
    ],
)
def test_broken_timestamps_fail_with_the_first_reason(
    words: list[Word], reason: str
) -> None:
    result = check_timestamps(_transcript(*words), _duration(8729))

    assert result == TimestampCheck("failed", reason)


def _bench_dir(path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (path / name).write_text(text)
    return path


def test_loads_matched_pairs_sorted_by_name(tmp_path: Path) -> None:
    _bench_dir(
        tmp_path,
        {
            "02.wav": "",
            "02.txt": "she [don't] like it",
            "01.wav": "",
            "01.txt": "he [go]\nto school",
            "notes.md": "ignored",
        },
    )

    assert load_recordings(tmp_path) == [
        Recording("01", tmp_path / "01.wav", parse_gold("he [go] to school")),
        Recording("02", tmp_path / "02.wav", parse_gold("she [don't] like it")),
    ]


@pytest.mark.parametrize(
    ("files", "problems"),
    [
        (
            {"01.wav": "", "01.txt": "hi", "03.wav": ""},
            ["03.wav has no matching 03.txt"],
        ),
        (
            {"01.wav": "", "01.txt": "hi", "02.txt": "hi"},
            ["02.txt has no matching 02.wav"],
        ),
        ({}, ["no .wav files"]),
        (
            {"01.wav": "", "01.txt": "a [b", "02.wav": ""},
            ["01.txt: unclosed '[' at column 3", "02.wav has no matching 02.txt"],
        ),
    ],
)
def test_reports_every_input_problem_at_once(
    tmp_path: Path, files: dict[str, str], problems: list[str]
) -> None:
    _bench_dir(tmp_path, files)

    with pytest.raises(BenchInputError) as excinfo:
        load_recordings(tmp_path)

    expected = [f"Can't run the bench on {tmp_path}:"] + [f"  - {p}" for p in problems]
    assert str(excinfo.value) == "\n".join(expected)


def test_a_missing_directory_is_reported_alone(tmp_path: Path) -> None:
    missing = tmp_path / "nope"

    with pytest.raises(BenchInputError) as excinfo:
        load_recordings(missing)

    assert str(excinfo.value) == (
        f"Can't run the bench on {missing}:\n  - {missing} is not a directory"
    )


def test_a_text_that_is_not_utf8_is_an_input_problem(tmp_path: Path) -> None:
    _bench_dir(tmp_path, {"01.wav": ""})
    (tmp_path / "01.txt").write_bytes("he [gó]".encode("latin-1"))

    with pytest.raises(BenchInputError) as excinfo:
        load_recordings(tmp_path)

    assert str(excinfo.value).endswith("  - 01.txt is not UTF-8 text")


class _Echo:
    """Hears exactly the gold of each file, with timestamps if asked."""

    def __init__(self, golds: dict[Path, str], timed: bool = False) -> None:
        self.golds = golds
        self.timed = timed

    def transcribe(self, audio: Path) -> Transcript:
        words = [
            _word(
                i, text, i * 10 if self.timed else None, i * 10 if self.timed else None
            )
            for i, text in enumerate(self.golds[audio].split())
        ]
        return Transcript(transcriber="echo", words=words)


class _FailsOn(_Echo):
    def __init__(self, golds: dict[Path, str], failing: Path) -> None:
        super().__init__(golds)
        self.failing = failing

    def transcribe(self, audio: Path) -> Transcript:
        if audio == self.failing:
            raise ConnectionError("boom")
        return super().transcribe(audio)


def _recordings(tmp_path: Path, *texts: str) -> tuple[list[Recording], dict[Path, str]]:
    recordings = [
        Recording(f"0{n}", tmp_path / f"0{n}.wav", parse_gold(text))
        for n, text in enumerate(texts, start=1)
    ]
    return recordings, {r.audio: " ".join(r.gold.words) for r in recordings}


def _loader(transcribers: dict[str, Transcriber]) -> Callable[[str], Transcriber]:
    def load(name: str) -> Transcriber:
        if name not in transcribers:
            raise LookupError(f"no {name}")
        return transcribers[name]

    return load


def _no_audio(audio: Path) -> int:
    raise AssertionError(f"decoded {audio}")


def test_runs_every_transcriber_in_the_order_given(tmp_path: Path) -> None:
    recordings, golds = _recordings(tmp_path, "he [go] home")
    load = _loader({"a": _Echo(golds), "b": _Echo(golds)})

    runs = list(run_bench(recordings, ["b", "a"], load=load, duration=_no_audio))

    assert [run.name for run in runs] == ["b", "a"]
    result = runs[0].files[0].result
    assert result is not None
    assert (result.gold_words, result.mistakes, result.mistakes_kept) == (3, 1, 1)


def test_a_transcriber_that_fails_to_load_does_not_stop_the_others(
    tmp_path: Path,
) -> None:
    recordings, golds = _recordings(tmp_path, "he [go] home")

    runs = list(
        run_bench(
            recordings,
            ["a", "b"],
            load=_loader({"b": _Echo(golds)}),
            duration=_no_audio,
        )
    )

    assert (runs[0].name, runs[0].files, runs[0].error) == (
        "a",
        [],
        "LookupError: no a",
    )
    assert runs[1].error is None
    assert [f.result is not None for f in runs[1].files] == [True]


def test_what_the_transcriber_wrote_is_compared_ignoring_case_and_punctuation() -> None:
    recording = Recording("01", Path("01.wav"), parse_gold("Yesterday he [don't] go."))
    transcript = _transcript(
        _word(0, "Yesterday,"), _word(1, "HE"), _word(2, "don’t"), _word(3, "go.")
    )

    result = measure_file(recording, transcript, _no_duration)

    assert diverging(result.steps) == 0
    assert result.mistakes_kept == 1


def test_a_file_that_fails_does_not_stop_the_other_files(tmp_path: Path) -> None:
    recordings, golds = _recordings(tmp_path, "one", "two", "three")
    transcriber = _FailsOn(golds, failing=tmp_path / "02.wav")

    [run] = run_bench(
        recordings, ["a"], load=_loader({"a": transcriber}), duration=_no_audio
    )

    assert [(f.name, f.error) for f in run.files] == [
        ("01", None),
        ("02", "ConnectionError: boom"),
        ("03", None),
    ]
    assert [f.result is None for f in run.files] == [False, True, False]


def test_each_audio_file_is_measured_once_across_transcribers(tmp_path: Path) -> None:
    recordings, golds = _recordings(tmp_path, "he [go] home")
    load = _loader({"a": _Echo(golds, timed=True), "b": _Echo(golds, timed=True)})
    decoded: list[Path] = []

    def duration(audio: Path) -> int:
        decoded.append(audio)
        return 1000

    runs = list(run_bench(recordings, ["a", "b"], load=load, duration=duration))

    assert decoded == [tmp_path / "01.wav"]
    assert [r.files[0].result.timestamps.status for r in runs if r.files[0].result] == [
        "ok",
        "ok",
    ]


_NO_TIMESTAMPS = TimestampCheck("n/a", None)


def _result(
    name: str,
    gold_text: str,
    heard_text: str,
    confident_words: int | None = None,
    timestamps: TimestampCheck = _NO_TIMESTAMPS,
) -> FileResult:
    gold = parse_gold(gold_text)
    steps = align(gold.words, normalize(heard_text))
    heard_words = len(heard_text.split())
    return FileResult(
        name=name,
        steps=steps,
        gold_words=len(gold.words),
        heard_words=heard_words,
        confident_words=heard_words if confident_words is None else confident_words,
        mistakes=len(gold.mistakes),
        mistakes_kept=mistakes_kept(steps, gold.mistakes),
        timestamps=timestamps,
    )


def test_renders_a_faithful_run_without_diffs() -> None:
    result = _result("01", "He [go] home.", "he go home")

    assert render_run(TranscriberRun("echo", [FileRun("01", result, None)], None)) == (
        "== echo ==\n"
        "file   words  diverging   WER  mistakes kept  confidence  timestamps\n"
        "01         3          0  0.0%            1/1        100%  n/a\n"
        "total      3          0  0.0%            1/1        100%"
    )


def test_diff_brackets_each_run_of_divergent_words() -> None:
    result = _result("07", "a b c d e f g h", "a x c e f y g z w")

    rendered = render_run(TranscriberRun("echo", [FileRun("07", result, None)], None))

    assert rendered.endswith(
        "\n\nDiffs:\n07  a [b → x] c [d → ∅] e f [∅ → y] g [h → z w]"
    )


def test_renders_results_failures_and_errors_of_a_run() -> None:
    run = TranscriberRun(
        "parakeet",
        [
            FileRun(
                "01",
                _result(
                    "01",
                    "Yesterday he [go] to school and she [don't] like it a lot.",
                    "Yesterday he went to school and she don't like it a lot.",
                    timestamps=TimestampCheck(
                        "failed",
                        "word 2 'went' ends at 9000 ms, "
                        "after the audio ends at 8729 ms",
                    ),
                ),
                None,
            ),
            FileRun(
                "02",
                _result(
                    "02",
                    "There are many people in the big room today.",
                    "There are many people in the big room today.",
                    confident_words=8,
                ),
                None,
            ),
            FileRun("03", None, "ConnectionError: boom"),
        ],
        None,
    )

    assert render_run(run) == "\n".join(
        [
            "== parakeet ==",
            "file   words  diverging   WER  mistakes kept  confidence  timestamps",
            "01        12          1  8.3%            1/2        100%  failed",
            "02         9          0  0.0%              –         89%  n/a",
            "03         –          –     –              –           –  error",
            "total     21          1  4.8%            1/2         95%",
            "",
            "Diffs:",
            "01  yesterday he [go → went] to school and she don't like it a lot",
            "",
            "Timestamps:",
            "01  word 2 'went' ends at 9000 ms, after the audio ends at 8729 ms",
            "",
            "Errors:",
            "03  ConnectionError: boom",
        ]
    )


def test_renders_a_transcriber_that_failed_to_load() -> None:
    run = TranscriberRun("openai", [], "TranscriberUnavailableError: no key")

    assert render_run(run) == "== openai ==\nerror: TranscriberUnavailableError: no key"


def test_a_run_where_every_file_failed_has_no_total() -> None:
    run = TranscriberRun("echo", [FileRun("01", None, "ConnectionError: boom")], None)

    assert render_run(run).splitlines()[2:] == [
        "01        –          –    –              –           –  error",
        "",
        "Errors:",
        "01  ConnectionError: boom",
    ]
