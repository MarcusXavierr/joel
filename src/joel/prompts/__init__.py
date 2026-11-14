"""The instructions each LLM step sends, from files a user can replace."""

import hashlib
import string
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class PromptSpec:
    """One prompt: its file name without .md, and the placeholders it may hold."""

    name: str
    summary: str  # one line for --help
    placeholders: frozenset[str]


# The only list of prompts: the loader, --prompt, --help, `joel prompt` and the
# hashes in the report all read it.
SPECS: tuple[PromptSpec, ...] = (
    PromptSpec(
        "disfluencies",
        "drops filled pauses, repetitions and false starts",
        frozenset(),
    ),
    PromptSpec(
        "correct",
        "writes the grammatical version of each sentence; "
        "$natural marks where correct-natural goes",
        frozenset({"natural"}),
    ),
)
NAMES: tuple[str, ...] = tuple(s.name for s in SPECS)


class PromptError(Exception):
    """A prompt file can't be read, or holds a placeholder its prompt doesn't allow."""


@dataclass(frozen=True)
class Prompts:
    """The instructions one run sends, and which texts they came from."""

    disfluencies: str  # sent to the cleanup step
    correct: str  # sent to the correction step, placeholders filled
    hashes: dict[str, str]  # every name in NAMES -> sha256 hex of the raw text used


def builtin(name: str) -> str:
    """The text of the built-in prompt `name`, as shipped."""
    return (resources.files("joel.prompts") / f"{name}.md").read_text("utf-8")


def load_prompts(overrides: Mapping[str, Path], *, better: bool) -> Prompts:
    """Read every prompt, from its override file when given, and fill it in.

    Raises PromptError when a file can't be read or has a placeholder its
    prompt doesn't allow.
    """
    templates = {}
    hashes = {}
    for spec in SPECS:
        path = overrides.get(spec.name)
        if path is None:
            text = builtin(spec.name)
            source = f"built-in {spec.name} prompt"
        else:
            try:
                text = path.read_text("utf-8")
            except OSError as exc:
                raise PromptError(
                    f"cannot read prompt file {path}: {exc.strerror}"
                ) from exc
            source = str(path)
        template = string.Template(text)
        if not template.is_valid():
            raise PromptError(
                f"{source}: a $ that starts no placeholder; write $$ for a literal $."
            )
        for found in template.get_identifiers():
            if found not in spec.placeholders:
                allowed = ", ".join(f"${p}" for p in sorted(spec.placeholders))
                raise PromptError(
                    f"{source}: unknown placeholder ${found}; "
                    f"the {spec.name} prompt allows {allowed or 'none'}."
                )
        templates[spec.name] = template
        hashes[spec.name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Prompts(
        disfluencies=templates["disfluencies"].substitute().strip(),
        correct=templates["correct"].substitute(natural="").strip(),
        hashes=hashes,
    )


def help_text() -> str:
    """The --help lines that name every prompt --prompt can replace."""
    width = max(len(name) for name in NAMES) + 2
    lines = [f"  {s.name.ljust(width)}{s.summary}" for s in SPECS]
    return "prompts --prompt can replace:\n" + "\n".join(lines)
