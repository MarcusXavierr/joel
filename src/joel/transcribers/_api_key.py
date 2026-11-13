"""The API key check every cloud transcriber runs before it builds a client."""

import os

from joel.transcribers import TranscriberUnavailableError


def require_api_key(variable: str, transcriber: str) -> str:
    """Return the API key in ``variable``; raise when it is unset or blank."""
    key = os.environ.get(variable, "").strip()
    if not key:
        raise TranscriberUnavailableError(
            f"The '{transcriber}' transcriber needs an API key in {variable}. "
            f"Set it with: export {variable}=<your key>"
        )
    return key
