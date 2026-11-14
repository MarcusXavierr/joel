"""Loading the built-in prompts and the files that replace them."""

import hashlib
from pathlib import Path

import pytest

from joel.prompts import NAMES, PromptError, builtin, help_text, load_prompts


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_built_in_prompts_are_used_and_hashed() -> None:
    prompts = load_prompts({}, better=False)

    assert prompts.disfluencies == builtin("disfluencies").strip()
    assert prompts.correct.startswith("You correct transcribed speech")
    assert prompts.hashes == {name: sha256(builtin(name)) for name in NAMES}


def test_a_file_replaces_its_built_in_prompt(tmp_path: Path) -> None:
    mine = tmp_path / "mine.md"
    mine.write_text("Drop only uh and um.\n")

    prompts = load_prompts({"disfluencies": mine}, better=False)

    assert prompts.disfluencies == "Drop only uh and um."
    assert prompts.hashes["disfluencies"] == sha256("Drop only uh and um.\n")
    assert prompts.hashes["correct"] == sha256(builtin("correct"))


def test_a_doubled_dollar_is_a_literal_dollar(tmp_path: Path) -> None:
    mine = tmp_path / "mine.md"
    mine.write_text("Keep $$5 as said.")

    assert load_prompts({"disfluencies": mine}, better=False).disfluencies == (
        "Keep $5 as said."
    )


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.md"

    with pytest.raises(PromptError) as raised:
        load_prompts({"correct": missing}, better=False)

    assert str(raised.value) == (
        f"cannot read prompt file {missing}: No such file or directory"
    )


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        (
            "disfluencies",
            "Use $natural here.",
            "unknown placeholder $natural; the disfluencies prompt allows none.",
        ),
        (
            "correct",
            "Fix it. $natural $tone",
            "unknown placeholder $tone; the correct prompt allows $natural.",
        ),
        (
            "correct",
            "It costs $5.",
            "a $ that starts no placeholder; write $$ for a literal $.",
        ),
    ],
)
def test_a_placeholder_the_prompt_does_not_allow_is_an_error(
    name: str, text: str, message: str, tmp_path: Path
) -> None:
    mine = tmp_path / "mine.md"
    mine.write_text(text)

    with pytest.raises(PromptError) as raised:
        load_prompts({name: mine}, better=False)

    assert str(raised.value) == f"{mine}: {message}"


def test_the_help_lists_every_prompt() -> None:
    lines = help_text().splitlines()

    assert lines[0] == "prompts --prompt can replace:"
    assert [line.split()[0] for line in lines[1:]] == list(NAMES)
