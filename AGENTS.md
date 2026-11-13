# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

Joel is a CLI that listens to an audio file of someone speaking English and reports every grammar mistake with the timestamp where it happened, what was said, and the correction. It is a personal tool for Marcus Xavier, a non-native speaker.

The pipeline runs ASR, disfluency cleanup, LLM correction, ERRANT diff, a confidence filter, and a report. Each step is a function with JSON-serializable input and output that can be tested in isolation. The core rule is precision over recall: a false mistake costs more than a missed one, because it teaches the user to ignore the report.

## Sources of truth

* `docs/plans/2026-11-13-joel-design.md` is the stage 1 PRD: scope, pipeline contracts, CLI flags, acceptance criteria. Read the section for the step you're touching before writing code.
* `docs/decisions/` holds the ADRs. When the PRD and an ADR disagree, the ADR wins. Record new architectural decisions with the `writing-madrs` skill, and never edit an accepted ADR; supersede it.
* `brief.md` is the original research brief. It is gitignored and in Portuguese, so treat it as background only.

## Python 3.13

The project pins Python 3.13 (`.python-version`, `requires-python = ">=3.13"`). Check the interpreter before running or installing anything:

```bash
uv run python --version   # must print Python 3.13.x
```

Use `uv` for everything: `uv add`, `uv run`, `uv sync`. Don't call `pip` directly, and don't create a venv by hand.

## Language

Everything committed to this repository is in English: code, comments, docstrings, tests, docs, ADRs, commit messages, and PR descriptions. Conversation with Marcus can be in Portuguese; what lands in the repo can't.

Use the `humanizer` skill whenever you write English prose for the repo (docs, ADRs, README, commit messages, PR descriptions). It doesn't apply to identifiers or code.

## Test-driven development

Implementation follows TDD, using the `test-driven-development` skill. Write the test first, run it, watch it fail for the right reason, then write the least code that makes it pass. A change without a test that failed before it is not done.

Fast tests come first. The default suite should finish in a few seconds, so it gets run after every change:

* Pure steps (`split_sentences`, offset mapping, `diff_edits`, `filter_edits`, `render`) are tested with small JSON fixtures. Most tests belong here.
* LLM steps use Pydantic AI's `FunctionModel` with canned responses. No test calls a real LLM.
* Transcriber adapters are tested against recorded provider responses. No test hits the network.
* Tests that load the Parakeet model or need `ffmpeg` are marked `slow` and stay out of the default run. Keep them few: one end-to-end test over a fixture audio file is the target.
* spaCy's `en_core_web_sm` takes a second or two to load. Load it once per session through a fixture, not per test.

pytest, ruff, and mypy are dev dependencies. `uv run pytest` is the fast run: it skips tests marked `slow` by default, and `uv run pytest -m slow` runs only those. `uv run ruff check src tests` and `uv run mypy src tests` (strict mode) must pass as well.

## Runtime facts that bite

* `transcribe-cpp` accepts only mono 16 kHz float32 PCM. It doesn't decode containers or resample, so decode with `ffmpeg` first.
* Parakeet Ultra (`handy-computer/parakeet-ultra-gguf`, `parakeet-ultra-Q8_0.gguf`) resolves through the standard Hugging Face cache, never through another app's files or a hardcoded path. Joel depends on no other app; it reuses whatever is already in the cache, Handy downloads included. The model is pinned by two constants, `PARAKEET_REVISION` (commit hash) and `PARAKEET_SHA256` (file hash), and resolution goes pinned revision in cache, then same file hash under any cached revision, then `hf_hub_download`. The PRD's "Model resolution" section has the details. Never use `revision="main"`: bench results only hold for the model version they measured.
* `transcribe-cpp` logs its backend choice (Vulkan on this machine's Intel Xe) to stderr when a model loads. That's expected, not an error.
* `en_core_web_sm` isn't on PyPI. It installs from the spaCy GitHub release wheel (3.8.0 for spaCy 3.8).
* `espeak-ng` is installed and works for generating synthetic test audio. Synthetic speech is clean and robotic, so it proves plumbing works but says nothing about real accented speech.
