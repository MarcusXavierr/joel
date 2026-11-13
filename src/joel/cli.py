"""The ``joel`` command."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from joel.bench import BenchInputError, load_recordings, render_run, run_bench
from joel.logs import configure_logging


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
    args = parser.parse_args(argv)
    configure_logging()
    return _bench(
        args.directory, list(dict.fromkeys(args.transcribers or ["parakeet"]))
    )


def _bench(directory: Path, names: list[str]) -> int:
    try:
        recordings = load_recordings(directory)
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
