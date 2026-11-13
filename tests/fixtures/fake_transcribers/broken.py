# Stands in for an adapter missing a dependency that no extra provides.
# "broken" is not an extra of joel.
import joel_fake_dependency_not_installed  # type: ignore[import-not-found]  # noqa: F401


def create() -> object:
    raise AssertionError("unreachable: the import above fails")
