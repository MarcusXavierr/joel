# Stands in for an adapter whose SDK comes from an optional extra that isn't
# installed. The module name matters: "openai" is a real extra of joel.
import joel_fake_sdk_not_installed  # type: ignore[import-not-found]  # noqa: F401


def create() -> object:
    raise AssertionError("unreachable: the import above fails")
