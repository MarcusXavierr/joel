"""The ``joel`` command."""

import argparse
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

from joel.bench import BenchInputError, load_recordings, render_run, run_bench
from joel.cost import (
    SPEAKING_RATE_WPM,
    WORDS_PER_SENTENCE,
    PriceUnavailableError,
    estimate_cost,
)
from joel.decode import SAMPLE_RATE, DecodeError, decode
from joel.llm import DEFAULT_MODEL
from joel.logs import configure_logging

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
