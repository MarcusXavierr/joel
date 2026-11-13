"""The ``joel`` command."""

import argparse
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from joel.cost import (
    SPEAKING_RATE_WPM,
    WORDS_PER_SENTENCE,
    PriceUnavailableError,
    estimate_cost,
)
from joel.decode import SAMPLE_RATE, DecodeError, decode
from joel.llm import DEFAULT_MODEL

BYTES_PER_SAMPLE = 4  # float32


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``joel`` command and return its exit status."""
    parser = argparse.ArgumentParser(prog="joel")
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser(
        "estimate", help="estimate what the LLM steps would cost for an audio file"
    )
    estimate.add_argument("audio", type=Path)
    estimate.add_argument(
        "--model", default=DEFAULT_MODEL, help="Pydantic AI model string"
    )
    args = parser.parse_args(argv)

    try:
        _estimate(args.audio, args.model)
    except (DecodeError, PriceUnavailableError) as exc:
        print(f"joel: error: {exc}", file=sys.stderr)
        return 1
    return 0


def _estimate(audio: Path, model: str) -> None:
    seconds = len(decode(audio)) / (BYTES_PER_SAMPLE * SAMPLE_RATE)
    cost = estimate_cost(seconds, model)
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
