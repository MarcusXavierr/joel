"""The ``joel`` command."""

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.models import infer_model

from joel.bench import BenchInputError, load_recordings, render_run, run_bench
from joel.check import run_check
from joel.confidence import MIN_CONFIDENCE
from joel.cost import (
    SPEAKING_RATE_WPM,
    WORDS_PER_SENTENCE,
    PriceUnavailableError,
    estimate_cost,
)
from joel.decode import SAMPLE_RATE, DecodeError, decode
from joel.edits import load_annotator
from joel.llm import DEFAULT_MODEL, ModelAnswerError
from joel.logs import configure_logging
from joel.report import render
from joel.transcribers import TranscriberUnavailableError, load_transcriber

BYTES_PER_SAMPLE = 4  # float32


def main(argv: Sequence[str] | None = None) -> int:
    """Run joel with ``argv`` (default: the process arguments); return the exit code."""
    parser = argparse.ArgumentParser(
        prog="joel",
        description=(
            "Your spoken English, every grammar mistake, the exact second you made it."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser(
        "check",
        help="Report every grammar mistake in a recording, with when you said it.",
    )
    check.add_argument("audio", type=Path, help="recording in any format ffmpeg reads")
    check.add_argument(
        "--transcriber",
        default="parakeet",
        help="speech-to-text model: parakeet, elevenlabs or openai (default: parakeet)",
    )
    check.add_argument(
        "--model", default=DEFAULT_MODEL, help="Pydantic AI model string"
    )
    check.add_argument(
        "--min-confidence",
        type=_probability,
        default=MIN_CONFIDENCE,
        metavar="P",
        help="hide mistakes on words heard with less confidence than this "
        f"(default: {MIN_CONFIDENCE})",
    )
    check.add_argument(
        "--show-uncertain",
        action="store_true",
        help="also show the mistakes held back as uncertain",
    )
    check.add_argument(
        "--json", action="store_true", help="print everything the run decided as JSON"
    )
    check.set_defaults(run=_check)

    bench = commands.add_parser(
        "bench", help="Measure which transcriber keeps your grammar mistakes."
    )
    bench.add_argument(
        "directory",
        type=Path,
        help="folder of NN.wav recordings, each with an NN.txt of what you said",
    )
    bench.add_argument(
        "--transcriber",
        action="append",
        dest="transcribers",
        metavar="NAME",
        default=None,
        help="speech-to-text model to measure; repeat for several (default: parakeet)",
    )
    bench.set_defaults(run=_bench)

    estimate = commands.add_parser(
        "estimate", help="Estimate what the language model would cost for a recording."
    )
    estimate.add_argument(
        "audio", type=Path, help="recording in any format ffmpeg reads"
    )
    estimate.add_argument(
        "--model", default=DEFAULT_MODEL, help="Pydantic AI model string"
    )
    estimate.set_defaults(run=_estimate)

    args = parser.parse_args(argv)
    configure_logging()
    run: Callable[[argparse.Namespace], int] = args.run
    return run(args)


def _probability(text: str) -> float:
    try:
        p: float | None = float(text)
    except ValueError:
        p = None
    if p is None or not 0 <= p <= 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number between 0 and 1")
    return p


def _check(args: argparse.Namespace) -> int:
    audio: Path = args.audio
    if not audio.is_file():
        print(f"joel: error: {audio} is not a file.", file=sys.stderr)
        return 1
    try:
        # infer_model fails here on a missing API key, before any transcription.
        model = infer_model(args.model)
        transcriber = load_transcriber(args.transcriber)
        result = asyncio.run(
            run_check(
                audio,
                transcriber,
                model=model,
                model_name=args.model,
                min_confidence=args.min_confidence,
                annotator=load_annotator(),
            )
        )
    except (
        UserError,
        TranscriberUnavailableError,
        DecodeError,
        ModelAnswerError,
        ModelAPIError,
    ) as exc:
        print(f"joel: error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(render(result, show_uncertain=args.show_uncertain))
    return 0


def _bench(args: argparse.Namespace) -> int:
    names = list(dict.fromkeys(args.transcribers or ["parakeet"]))
    try:
        recordings = load_recordings(args.directory)
    except BenchInputError as exc:
        print(exc, file=sys.stderr)
        return 1
    failed = False
    for n, run in enumerate(run_bench(recordings, names)):
        if n:
            print(flush=True)
        print(render_run(run), flush=True)
        failed = failed or run.error is not None
        failed = failed or any(file.error is not None for file in run.files)
    return 1 if failed else 0


def _estimate(args: argparse.Namespace) -> int:
    audio: Path = args.audio
    model: str = args.model
    try:
        seconds = len(decode(audio)) / (BYTES_PER_SAMPLE * SAMPLE_RATE)
        cost = estimate_cost(seconds, model)
    except (DecodeError, PriceUnavailableError) as exc:
        print(f"joel: error: {exc}", file=sys.stderr)
        return 1
    minutes, secs = divmod(round(seconds), 60)
    money = "under $0.01" if cost.usd < Decimal("0.01") else f"about ${cost.usd:.2f}"
    print(
        f"{audio}: {minutes}:{secs:02d} of audio, "
        f"about {cost.words} words in {cost.sentences} sentences"
    )
    print(
        f"{model}: {money} for {cost.calls} calls "
        f"({cost.input_tokens} input and {cost.output_tokens} output tokens)"
    )
    print(
        f"Assumes {SPEAKING_RATE_WPM} words per minute and {WORDS_PER_SENTENCE} "
        "words per sentence; a retried call costs extra."
    )
    return 0
