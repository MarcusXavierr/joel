import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import huggingface_hub.constants
import pytest
import transcribe_cpp

from joel.decode import SAMPLE_RATE, DecodeError, decode
from joel.transcribers import _parakeet_model, load_transcriber
from joel.transcribers.parakeet import forward_native_log, to_words

FIXTURES = Path(__file__).parent / "fixtures"
# Made with:
#   espeak-ng -v en-us -s 130 -w tests/fixtures/audio/acceptance.wav \
#     "Yesterday he go to school, uh, and she don't like it. There is many people in the the room. I am agree with you."
# At espeak's default rate (175) Parakeet hears "I agree" and drops the second
# "the", so the slower rate is what keeps every mistake audible.
AUDIO = FIXTURES / "audio" / "acceptance.wav"
AUDIO_DURATION_MS = 8729  # ffprobe: 8.728707 s
BYTES_PER_SAMPLE = 4  # float32


@pytest.mark.parametrize(
    ("native_level", "level"),
    [
        (1, logging.INFO),
        (2, logging.WARNING),
        (3, logging.ERROR),
        (4, logging.DEBUG),
        (99, logging.WARNING),
    ],
)
def test_native_log_lines_go_to_python_logging(
    native_level: int, level: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="joel.transcribe_cpp")

    forward_native_log(native_level, "msg\n")

    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ("joel.transcribe_cpp", level, "msg")
    ]


def test_maps_a_recorded_result_to_words() -> None:
    # transcribe-cpp's result for AUDIO, as recorded on 2026-11-13.
    data = json.loads((FIXTURES / "parakeet_result.json").read_text())
    tokens = [transcribe_cpp.Token(**t) for t in data["tokens"]]

    words = to_words([transcribe_cpp.Word(**w) for w in data["words"]], tokens)

    assert " ".join(w.text for w in words) == data["text"]
    assert [w.index for w in words] == list(range(len(data["words"])))
    # "school," is " sch" (p 0.9999996), "ool" (p 1.0) and "," (p 0.9959), and
    # "Yesterday" is lowest on its first token: a word is only as sure as its
    # least sure token, wherever that token sits.
    school = words[4]
    assert school.text == "school,"
    assert (school.start_ms, school.end_ms) == (1440, 2000)
    assert school.confidence == 0.9959238767623901
    assert words[0].text == "Yesterday"
    assert words[0].confidence == 0.994184672832489


def test_word_without_tokens_has_no_confidence() -> None:
    word = transcribe_cpp.Word(
        text="hm", t0_ms=0, t1_ms=10, seg_index=0, first_token=0, n_tokens=0
    )

    assert to_words([word], [])[0].confidence is None


def test_bad_audio_fails_before_the_model_is_fetched(
    tmp_path: Path,
    fake_ffmpeg: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ffmpeg('echo "talk.m4a: Invalid data" >&2\nexit 1')
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    def forbid_download(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("the model was fetched before decoding")

    monkeypatch.setattr(_parakeet_model, "hf_hub_download", forbid_download)

    with pytest.raises(DecodeError):
        load_transcriber("parakeet").transcribe(tmp_path / "talk.m4a")


@pytest.mark.slow
def test_transcribes_the_acceptance_recording(
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The model must already be cached; this test never downloads it.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    caplog.set_level(logging.DEBUG, logger="joel.transcribe_cpp")

    transcript = load_transcriber("parakeet").transcribe(AUDIO)

    text = " ".join(w.text for w in transcript.words)
    for mistake in ["he go", "she don't", "There is many", "the the", "I am agree"]:
        assert mistake in text
    duration_ms = len(decode(AUDIO)) // BYTES_PER_SAMPLE * 1000 // SAMPLE_RATE
    assert abs(duration_ms - AUDIO_DURATION_MS) <= 50
    starts = []
    for word in transcript.words:
        assert word.start_ms is not None and word.end_ms is not None
        assert 0 <= word.start_ms <= word.end_ms <= duration_ms
        starts.append(word.start_ms)
    assert starts == sorted(starts)
    assert any(
        r.name == "joel.transcribe_cpp"
        and r.levelno == logging.INFO
        and "backend" in r.getMessage()
        for r in caplog.records
    )
    assert capfd.readouterr().out == ""
