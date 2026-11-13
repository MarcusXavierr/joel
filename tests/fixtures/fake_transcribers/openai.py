# Stands in for an adapter that belongs to joel's "openai" extra. It imports
# the real openai SDK first; tests hide that SDK to simulate the extra not
# being installed. The second import is a module no extra provides, standing
# in for a bug inside the adapter itself.
import openai  # noqa: F401

# isort: split
import joel_fake_helper_not_installed  # type: ignore[import-not-found]  # noqa: F401


def create() -> object:
    raise AssertionError("unreachable: the imports above fail")
