"""Where joel's log records, and those of its libraries, end up."""

import logging
import sys


def configure_logging(level: int = logging.WARNING) -> None:
    """Send the logs of joel and its libraries to stderr at ``level`` and above."""
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
