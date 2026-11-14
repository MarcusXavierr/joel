---
status: accepted
date: 2026-11-14
decision-makers: Marcus Xavier
---

# Prompts as files in the package that a user can replace per run

## Context and Problem Statement

The cleanup and correction instructions were Python string constants in `disfluencies.py` and `correct.py`. Trying a different wording meant editing the package, and a report said nothing about which wording produced it. Two runs a week apart could disagree and there was no way to tell whether the prompt had changed in between.

I want to experiment with prompts on my own recordings without touching code, and to know afterwards which text a report came from. The probes in ADR-0010 and the token rates behind `joel estimate` were measured with one specific wording, so whatever lets me swap it also has to make clear when those numbers stop applying.

## Decision Drivers

* Changing a prompt for one run, without editing or reinstalling Joel.
* Knowing from a report alone which prompt text produced it.
* Prompt diffs that are easy to review: a one-word change should show as a one-line change.
* One list of prompts, so a new one can't be loaded but missing from `--help`, or the other way round.
* No new dependency.

## Considered Options

* Keep the prompts as Python constants.
* Ship the prompts as `.md` files in the package and let `--prompt NAME=FILE` replace one per run.
* Same files, but overridden by whatever sits in a user config directory such as `~/.config/joel/prompts/`.
* Jinja2 templates instead of `string.Template`.

## Decision Outcome

Chosen option: `.md` files in `joel.prompts`, replaced per run with `--prompt NAME=FILE`, because the override is visible on the command line that produced the report and the default stays the measured text.

Each file holds one sentence per line. A registry, `SPECS` in `joel.prompts`, lists every prompt with a one-line summary and the placeholders it may hold, and the loader, the `--prompt` check, the `--help` epilog, `joel prompt NAME` and the hashes all read it. Files are `string.Template`s: `$natural` is the one placeholder so far (ADR-0014), and `$$` writes a literal `$`. Python 3.14's t-strings would do the same job, but the project pins 3.13. Every run records the SHA-256 of the raw text of each prompt, in both the report and the `--verbose` dump. The instructions now go to pydantic-ai per run (`Agent.run(instructions=...)`) instead of being fixed on the agent.

### Consequences

* Good, because trying a wording is `joel prompt correct > mine.md`, an edit, and `--prompt correct=mine.md`.
* Good, because the hashes tell two reports' prompts apart without storing the text, and they match `sha256sum` of the file.
* Good, because a file that can't be read, has a `$` that starts no placeholder, or has a placeholder its prompt doesn't allow stops the run before anything is decoded, transcribed or paid for.
* Good, because one sentence per line gives one-line diffs.
* Bad, because a user's prompt can quietly make the report worse. The probes in ADR-0010 and ADR-0014 and the rates in `cost.py` hold only for the built-in text. `joel estimate` says so when `--prompt` is given; `joel check` doesn't warn.
* Bad, because the built-in text now reaches the model with a newline between sentences instead of a space. The two moved files match the old constants character for character apart from that, but ADR-0010's probe ran on the space-joined text.
* Bad, because the `.md` files have to ship in the wheel. Hatch includes them today; a packaging change could drop them, and only `joel prompt` or a run would notice.
* Neutral, because a literal `$` in a prompt must be written `$$`. None of the built-in prompts has one.

### Confirmation

`tests/test_prompts.py` covers the loader's rules and messages. `test_a_prompt_file_replaces_the_built_in_instructions` checks that a `--prompt` file's text is what the model receives and that its hash lands in the report, and `test_a_bad_prompt_file_stops_the_run_before_transcribing` checks that a bad file fails before the transcriber is called. `test_prompt_prints_a_built_in_prompt_as_is` and `test_the_help_lists_every_prompt_a_file_can_replace` cover the command and the epilog. On 2026-11-14, `uv build` produced a wheel with `joel/prompts/disfluencies.md`, `correct.md` and `correct-natural.md`.

## Pros and Cons of the Options

### Python constants

* Good, because there is nothing to load and nothing to validate.
* Bad, because every experiment is a code change, and the report can't say which wording it used.

### Files in the package, `--prompt` per run

* Good, because the default is always the measured text, and a different one shows up on the command line.
* Bad, because repeating `--prompt` on every run is tedious for someone who settles on their own wording.

### A user config directory that overrides the built-ins

* Good, because a settled wording needs no flag.
* Bad, because a forgotten file in that directory changes every report silently, and the measured numbers stop applying without anything on screen saying so. The hashes would show it, but only to someone who looks.

### Jinja2 templates

* Good, because loops and conditionals would allow richer prompts.
* Bad, because it adds a dependency for one placeholder, and a template language invites logic in prompts that belongs in code.
