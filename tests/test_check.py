"""The whole pipeline, from a transcript to every kept mistake and its cost."""

import asyncio
import gc
import json
import re
import warnings
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

import pydantic_ai
import pydantic_ai._display
import pytest
from conftest import user_prompt
from errant.annotator import Annotator  # type: ignore[import-untyped]
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from joel.check import CheckResult, run_check
from joel.cli import main
from joel.llm import BLOCK_SENTENCES
from joel.report import render
from joel.transcribers.openai import AudioTooLargeError
from joel.transcript import Transcript, Word

SPOKEN = (
    "Yesterday he go to school, uh, and she don't like it. "
    "There is many people in the the room. I am agree with you."
)
# What a careful model would correct in each clean sentence.
FIXES = {
    "he go ": "he went ",
    "she don't": "she doesn't",
    "There is many": "There are many",
    "I am agree": "I agree",
}
FILLERS = {"uh", "um"}
MODEL = "anthropic:claude-opus-5-5"

# main() configures logging for the whole process; each test gets the root back.
pytestmark = pytest.mark.usefixtures("restore_root_logger")


def spoken_words(confidence: dict[int, float] | None = None) -> list[Word]:
    return [
        Word(
            index=i,
            text=t,
            start_ms=500 * i,
            end_ms=500 * i + 400,
            confidence=(confidence or {}).get(i, 0.95),
        )
        for i, t in enumerate(SPOKEN.split(" "))
    ]


class FakeTranscriber:
    """Returns the same words for any audio, under the given transcriber name."""

    def __init__(self, name: str, words: Sequence[Word]) -> None:
        self.name = name
        self.words = list(words)

    def transcribe(self, audio: Path) -> Transcript:
        return Transcript(transcriber=self.name, words=list(self.words))


class PipelineModel:
    """A model that answers both LLM steps by content, so call order doesn't matter.

    Every call costs 100 input and 10 output tokens, `cache_read_tokens` of the
    input read from the prompt cache.
    """

    def __init__(self, cache_read_tokens: int = 0) -> None:
        self.cache_read_tokens = cache_read_tokens
        self.prompts: list[str] = []
        self.model = FunctionModel(self.respond)

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = user_prompt(messages)
        self.prompts.append(prompt)
        tool = info.output_tools[0]
        answer: dict[str, object]
        if "drop" in tool.parameters_json_schema["properties"]:
            answer = {"drop": _disfluencies(json.loads(prompt))}
        else:
            block = prompt.split("Sentences to correct:\n", 1)[1].splitlines()
            items = []
            for line in block:
                found = re.match(r"(\d+)\. (.*)", line)
                if found is None:
                    break
                sentence = found[2]
                for wrong, right in FIXES.items():
                    sentence = sentence.replace(wrong, right)
                items.append(
                    {
                        "number": int(found[1]),
                        "grammatical": sentence,
                        "natural": sentence,
                    }
                )
            answer = {"sentences": items}
        return ModelResponse(
            parts=[ToolCallPart(tool.name, answer)],
            usage=RequestUsage(
                input_tokens=100,
                output_tokens=10,
                cache_read_tokens=self.cache_read_tokens,
            ),
        )


def _disfluencies(pairs: list[list[object]]) -> list[int]:
    def norm(word: object) -> str:
        return str(word).strip(",.?!").casefold()

    dropped = []
    for n, (index, word) in enumerate(pairs):
        following = norm(pairs[n + 1][1]) if n + 1 < len(pairs) else None
        if norm(word) in FILLERS or norm(word) == following:
            assert isinstance(index, int)
            dropped.append(index)
    return dropped


async def check(
    transcriber: FakeTranscriber,
    model: PipelineModel,
    annotator: Annotator,
    min_confidence: float = 0.67,
    block_sentences: int = BLOCK_SENTENCES,
) -> CheckResult:
    return await run_check(
        Path("talk.m4a"),
        transcriber,
        model=model.model,
        model_name=MODEL,
        min_confidence=min_confidence,
        annotator=annotator,
        block_sentences=block_sentences,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("block_sentences", "requests"),
    [(BLOCK_SENTENCES, 2), (2, 4), (1, 6)],
    ids=[
        "one block per step",
        "a full block then a partial one",
        "one sentence per call",
    ],
)
async def test_finds_the_known_mistakes_and_not_the_disfluencies(
    annotator: Annotator, block_sentences: int, requests: int
) -> None:
    result = await check(
        FakeTranscriber("local", spoken_words()),
        PipelineModel(),
        annotator,
        block_sentences=block_sentences,
    )

    assert [s.dropped for s in result.sentences] == [[5], [16], []]
    edits = [f for s in result.sentences for f in s.edits]
    assert [
        (f.edit.category, f.edit.type, f.edit.original, f.edit.correction, f.edit.words)
        for f in edits
    ] == [
        ("grammar", "R:VERB:TENSE", "go", "went", [2]),
        ("grammar", "R:VERB:SVA", "do", "does", [8]),
        ("grammar", "R:VERB:SVA", "is", "are", [12]),
        ("grammar", "U:VERB:TENSE", "am", "", [20]),
    ]
    assert all(f.uncertain is None for f in edits)
    assert (result.requests, result.input_tokens, result.output_tokens) == (
        requests,
        100 * requests,
        10 * requests,
    )
    assert result.cost_usd is not None


@pytest.mark.anyio
async def test_swapping_the_transcriber_changes_nothing_after_it(
    annotator: Annotator,
) -> None:
    words = spoken_words()
    model_a, model_b = PipelineModel(), PipelineModel()

    a = await check(FakeTranscriber("alpha", words), model_a, annotator)
    b = await check(FakeTranscriber("beta", words), model_b, annotator)

    assert a.model_dump(exclude={"transcript"}) == b.model_dump(exclude={"transcript"})
    assert a.transcript.words == b.transcript.words
    assert sorted(model_a.prompts) == sorted(model_b.prompts)


def unscored(*texts: str) -> list[Word]:
    return [
        Word(index=i, text=t, start_ms=None, end_ms=None, confidence=None)
        for i, t in enumerate(texts)
    ]


@pytest.mark.anyio
async def test_a_sentence_left_empty_is_not_corrected_or_used_as_context(
    annotator: Annotator,
) -> None:
    words = unscored("He", "go", "home.", "Uh.", "She", "go", "out.")
    model = PipelineModel()

    result = await check(
        FakeTranscriber("local", words), model, annotator, block_sentences=1
    )

    middle = result.sentences[1]
    assert middle.correction is None
    assert middle.edits == []
    assert result.requests == 5  # three cleanups, two corrections
    corrections = sorted(p for p in model.prompts if "Sentences to correct" in p)
    assert corrections == [
        "Previous sentence: He go home.\nSentences to correct:\n1. She go out.",
        "Sentences to correct:\n1. He go home.\nNext sentence: She go out.",
    ]


@pytest.mark.anyio
async def test_each_block_reads_the_sentences_around_it_as_context(
    annotator: Annotator,
) -> None:
    words = unscored(
        "He go home.", "She go out.", "We go in.", "They go up.", "I go down."
    )
    model = PipelineModel()

    await check(FakeTranscriber("local", words), model, annotator, block_sentences=2)

    corrections = sorted(p for p in model.prompts if "Sentences to correct" in p)
    assert corrections == [
        "Previous sentence: She go out.\nSentences to correct:\n"
        "1. We go in.\n2. They go up.\nNext sentence: I go down.",
        "Previous sentence: They go up.\nSentences to correct:\n1. I go down.",
        "Sentences to correct:\n1. He go home.\n2. She go out.\n"
        "Next sentence: We go in.",
    ]


@pytest.mark.anyio
async def test_a_silent_recording_costs_nothing(annotator: Annotator) -> None:
    result = await check(FakeTranscriber("local", []), PipelineModel(), annotator)

    assert result.sentences == []
    assert (result.requests, result.cost_usd) == (0, Decimal(0))
    assert render(result, show_uncertain=False) == (
        "No mistakes found.\n\n"
        f"Cost: $0.000 for 0 model calls (0 input and 0 output tokens) with {MODEL}."
    )


@pytest.mark.anyio
async def test_a_long_recording_gets_every_call_it_needs(annotator: Annotator) -> None:
    words = unscored(*["He go home."] * 30)  # past pydantic-ai's 50-request default

    result = await check(
        FakeTranscriber("local", words), PipelineModel(), annotator, block_sentences=1
    )

    assert result.requests == 60


@pytest.mark.anyio
async def test_at_most_eight_model_calls_run_at_once(annotator: Annotator) -> None:
    answers = PipelineModel()
    in_flight = peak = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return answers.respond(messages, info)

    words = unscored(*["He go home."] * 20)

    result = await run_check(
        Path("talk.m4a"),
        FakeTranscriber("local", words),
        model=FunctionModel(respond),
        model_name=MODEL,
        min_confidence=0.67,
        annotator=annotator,
        block_sentences=1,
    )

    assert result.requests == 40
    assert peak == 8


@pytest.mark.anyio
async def test_each_step_runs_its_first_call_alone(annotator: Annotator) -> None:
    answers = PipelineModel()
    started: dict[str, list[int]] = {"cleanup": [], "correct": []}
    finished = {"cleanup": 0, "correct": 0}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        step = (
            "cleanup"
            if "drop" in tool.parameters_json_schema["properties"]
            else "correct"
        )
        started[step].append(finished[step])
        await asyncio.sleep(0.01)
        finished[step] += 1
        return answers.respond(messages, info)

    await run_check(
        Path("talk.m4a"),
        FakeTranscriber("local", unscored(*["He go home."] * 4)),
        model=FunctionModel(respond),
        model_name=MODEL,
        min_confidence=0.67,
        annotator=annotator,
        block_sentences=1,
    )

    # The first call writes the prompt cache; the others start once it's done.
    assert started == {"cleanup": [0, 1, 1, 1], "correct": [0, 1, 1, 1]}


@pytest.mark.anyio
async def test_every_call_asks_to_cache_only_the_instructions(
    annotator: Annotator,
) -> None:
    answers = PipelineModel()
    settings: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings.append(info.model_request_parameters.cache)
        return answers.respond(messages, info)

    # Only a model that says it caches is handed the cache setting.
    caching = FunctionModel(respond, profile=ModelProfile(supports_cache=True))
    await run_check(
        Path("talk.m4a"),
        FakeTranscriber("local", spoken_words()),
        model=caching,
        model_name=MODEL,
        min_confidence=0.67,
        annotator=annotator,
        block_sentences=1,
    )

    assert settings == [{"messages": False}] * 6


@pytest.mark.anyio
async def test_tokens_read_from_the_cache_are_counted_and_cost_less(
    annotator: Annotator,
) -> None:
    words = spoken_words()

    cached = await check(
        FakeTranscriber("local", words), PipelineModel(cache_read_tokens=80), annotator
    )
    uncached = await check(FakeTranscriber("local", words), PipelineModel(), annotator)

    assert (cached.requests, cached.input_tokens) == (2, 200)
    assert (cached.cache_read_tokens, cached.cache_write_tokens) == (160, 0)
    assert cached.cost_usd is not None and uncached.cost_usd is not None
    assert cached.cost_usd < uncached.cost_usd


@pytest.fixture
def cli_pipeline(
    monkeypatch: pytest.MonkeyPatch, annotator: Annotator
) -> Callable[[Sequence[Word]], None]:
    """Make `joel check` run on the given words and a PipelineModel."""

    def install(words: Sequence[Word]) -> None:
        monkeypatch.setattr(
            "joel.cli.load_transcriber", lambda name: FakeTranscriber(name, words)
        )
        monkeypatch.setattr("joel.cli.infer_model", lambda name: PipelineModel().model)
        monkeypatch.setattr("joel.cli.load_annotator", lambda: annotator)

    return install


@pytest.mark.parametrize("show_uncertain", [False, True])
def test_the_json_rebuilds_the_terminal_report(
    show_uncertain: bool,
    cli_pipeline: Callable[[Sequence[Word]], None],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli_pipeline(spoken_words(confidence={12: 0.5}))  # "is" is uncertain
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")
    flag = ["--show-uncertain"] if show_uncertain else []

    assert main(["check", str(audio), *flag]) == 0
    report = capsys.readouterr().out
    assert main(["check", str(audio), "--json"]) == 0
    saved = tmp_path / "talk.json"
    saved.write_text(capsys.readouterr().out)

    result = CheckResult.model_validate_json(saved.read_text())
    assert render(result, show_uncertain=show_uncertain) + "\n" == report


def test_the_model_threshold_and_block_flags_reach_the_run(
    cli_pipeline: Callable[[Sequence[Word]], None],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli_pipeline(spoken_words(confidence={12: 0.5}))  # "is", exactly at 0.5
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")

    flags = ["--model", "nope:nothing", "--min-confidence", "0.5", "--json"]
    flags += ["--block-sentences", "1"]
    assert main(["check", str(audio), *flags]) == 0

    result = CheckResult.model_validate_json(capsys.readouterr().out)
    assert result.model == "nope:nothing"
    assert result.requests == 6  # three sentences, one call each per step
    assert result.cost_usd is None  # genai-prices can't price it; the run goes on
    assert all(f.uncertain is None for s in result.sentences for f in s.edits)


def test_check_prints_no_banner_where_pydantic_ai_would_show_one(
    cli_pipeline: Callable[[Sequence[Word]], None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # pydantic-ai stays quiet under pytest and CI; lift both so it would speak.
    monkeypatch.delenv("PYTEST_VERSION")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYDANTIC_AI_NO_BANNER", raising=False)
    # stderr is a pipe here; AI_AGENT says someone reads it anyway.
    monkeypatch.setenv("AI_AGENT", "1")
    monkeypatch.setattr(pydantic_ai._display, "_banner_displayed", False)
    monkeypatch.setattr(pydantic_ai, "BANNER_ENABLED", True)
    cli_pipeline(spoken_words())
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")

    assert main(["check", str(audio)]) == 0
    out, err = capsys.readouterr()
    assert "pydantic-ai v" not in out + err


def _invalid_cleanup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, {"drop": [99]})]
    )


def _misnumbered_correction(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    tool = info.output_tools[0]
    if "drop" in tool.parameters_json_schema["properties"]:
        return ModelResponse(parts=[ToolCallPart(tool.name, {"drop": []})])
    wrong = {"number": 2, "grammatical": "He went home.", "natural": "He went home."}
    return ModelResponse(parts=[ToolCallPart(tool.name, {"sentences": [wrong]})])


def _rejected_key(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(401, "test", body={"error": "invalid x-api-key"})


@pytest.mark.parametrize(
    ("respond", "message"),
    [
        (_invalid_cleanup, "The model gave no valid answer for"),
        (_misnumbered_correction, "The model gave no valid answer for"),
        (_rejected_key, "status_code: 401"),
    ],
)
def test_a_failing_model_call_is_an_error(
    respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse],
    message: str,
    cli_pipeline: Callable[[Sequence[Word]], None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # More sentences than calls in flight, so some calls are still queued.
    cli_pipeline(unscored(*["He go home."] * 20))
    monkeypatch.setattr("joel.cli.infer_model", lambda name: FunctionModel(respond))
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert main(["check", str(audio), "--block-sentences", "1"]) == 1
        gc.collect()  # a queued call that never ran warns when collected

    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("joel: error: ")
    assert message in err
    # A cancelled pydantic-ai run may leave its own internals unawaited; that
    # never reaches stderr. A queued call of joel's must not be left at all.
    leaked = [str(w.message) for w in caught]
    assert not any("'strip_disfluencies'" in m or "'correct'" in m for m in leaked)


def test_audio_ffmpeg_cannot_read_is_an_error(
    fake_ffmpeg: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
    annotator: Annotator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # parakeet decodes before it looks for the model, so this never loads it.
    fake_ffmpeg("echo 'Invalid data found when processing input' >&2; exit 1")
    monkeypatch.setattr("joel.cli.infer_model", lambda name: PipelineModel().model)
    monkeypatch.setattr("joel.cli.load_annotator", lambda: annotator)
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")

    assert main(["check", str(audio)]) == 1
    assert capsys.readouterr().err.startswith("joel: error: ffmpeg could not read")


def test_an_unknown_transcriber_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")

    assert main(["check", str(audio), "--transcriber", "nope"]) == 1
    assert capsys.readouterr().err.startswith("joel: error: Unknown transcriber 'nope'")


def test_a_missing_api_key_stops_the_run_before_transcribing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def no_transcriber(name: str) -> FakeTranscriber:
        raise AssertionError("transcribed before checking the model")

    monkeypatch.setattr("joel.cli.load_transcriber", no_transcriber)
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")

    assert main(["check", str(audio)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("joel: error:")
    assert "OPENAI_API_KEY" in err


def test_audio_the_transcriber_refuses_is_an_error(
    cli_pipeline: Callable[[Sequence[Word]], None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TooLarge(FakeTranscriber):
        def transcribe(self, audio: Path) -> Transcript:
            raise AudioTooLargeError(f"{audio} is over the 25 MB upload limit.")

    cli_pipeline([])
    monkeypatch.setattr("joel.cli.load_transcriber", lambda name: TooLarge(name, []))
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")

    assert main(["check", str(audio), "--transcriber", "openai"]) == 1
    assert capsys.readouterr().err == (
        f"joel: error: {audio} is over the 25 MB upload limit.\n"
    )


def test_a_missing_audio_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "nope.m4a"

    assert main(["check", str(audio)]) == 1
    assert capsys.readouterr().err == f"joel: error: {audio} is not a file.\n"


@pytest.mark.parametrize("value", ["0", "-3", "six"])
def test_block_sentences_must_be_a_whole_number_of_at_least_one(
    value: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["check", str(tmp_path), "--block-sentences", value])

    assert exited.value.code == 2
    assert f"{value!r} is not a whole number of at least 1" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["2", "-0.1", "high"])
def test_min_confidence_must_be_a_probability(
    value: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["check", str(tmp_path), "--min-confidence", value])

    assert exited.value.code == 2
    assert f"{value!r} is not a number between 0 and 1" in capsys.readouterr().err


AUDIO = Path(__file__).parent / "fixtures" / "audio" / "acceptance.wav"


@pytest.mark.slow
def test_check_flags_the_four_mistakes_in_the_recording(
    monkeypatch: pytest.MonkeyPatch,
    annotator: Annotator,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("joel.cli.infer_model", lambda name: PipelineModel().model)
    monkeypatch.setattr("joel.cli.load_annotator", lambda: annotator)

    assert main(["check", str(AUDIO), "--json"]) == 0

    result = CheckResult.model_validate_json(capsys.readouterr().out)
    by_index = {w.index: w for w in result.transcript.words}
    disfluent = {w.index for w in result.transcript.words if w.text in {"uh,", "the"}}
    assert len(disfluent) == 3
    kept = []
    for sentence in result.sentences:
        spoken = [by_index[i] for i in sentence.words]
        for f in sentence.edits:
            assert disfluent.isdisjoint(f.edit.words)
            if f.edit.category != "grammar" or f.uncertain is not None:
                continue
            words = [by_index[i] for i in f.edit.words]
            starts = [w.start_ms for w in words]
            ends = [w.end_ms for w in words]
            assert all(
                s is not None and e is not None and s < e
                for s, e in zip(starts, ends, strict=True)
            )
            assert min(w.start_ms or 0 for w in spoken) <= min(s or 0 for s in starts)
            assert max(e or 0 for e in ends) <= max(w.end_ms or 0 for w in spoken)
            kept.append((f.edit.original, f.edit.correction, [w.text for w in words]))
    assert kept == [
        ("go", "went", ["go"]),
        ("do", "does", ["don't"]),
        ("is", "are", ["is"]),
        ("am", "", ["am"]),
    ]
    report = render(result, show_uncertain=False)
    entries = report.split("\n\n")[:-1]  # the cost line closes the report
    assert [e.count("  said:") for e in entries] == [1, 1, 1]
    first = entries[0].splitlines()
    assert re.fullmatch(r"\d\d:\d\d\.\d–\d\d:\d\d\.\d", first[0])
    assert first[1].startswith("  said:    Yesterday he go to school,")
    assert first[2].startswith("  correct: ")
    assert all(line.startswith(" " * 11) for line in first[3:])
    changes = [line.split(maxsplit=1) for line in first[3:]]
    assert [re.sub(r"\s+", " ", change) for _, change in changes] == [
        "go → went R:VERB:TENSE grammar",
        "do → does R:VERB:SVA grammar",
    ]
    clocks = [clock for clock, _ in changes]
    assert all(re.fullmatch(r"\d\d:\d\d\.\d", clock) for clock in clocks)
    assert clocks[0] < clocks[1]
    assert report.count(" grammar\n") == 4
