# Joel

**"Do you understand?"** Your spoken English, every grammar mistake, the exact second you made it.

Joel is a command-line tool for people who speak English as a second language. You give it a recording of yourself talking. It tells you which sentences had a grammar mistake, when in the recording you said them, and what you should have said.

> [!IMPORTANT]
> Joel is at the design stage and none of the commands below work yet. Today `uv run joel` prints a greeting and exits. The repo holds the [design](docs/plans/2026-11-13-joel-design.md), the [decisions](docs/decisions/README.md) behind it, and a [backlog of seven tickets](https://github.com/MarcusXavierr/joel/issues/8). This README describes the tool as designed, so details may still move.

## What a report will look like

```
$ joel check talk.m4a

00:12.4–00:13.1  R:VERB:SVA   grammar
  said:    There is many people in the room.
  correct: There are many people in the room.
           is → are

00:41.0–00:41.6  naturalness
  said:    I make a party last week.
  better:  I threw a party last week.
```

Each entry starts with where to find the sentence in the recording. `R:VERB:SVA` is the kind of mistake, here a verb that doesn't agree with its subject. A `naturalness` entry is a sentence that is grammatical but that a native speaker wouldn't say that way.

Joel would rather miss a mistake than invent one. A report that flags things you said correctly teaches you to ignore it, so anything the transcription was unsure about stays hidden unless you ask for it with `--show-uncertain`.

## How it works

Speech-to-text turns the recording into words, each with a time and a confidence score. A language model then marks the hesitations ("uh", repeated words, false starts) so they can be set aside, and writes a corrected version of each sentence. The model never lists your mistakes, because a model asked for a list will make some up. The list comes from comparing what you said with the corrected sentence, using [ERRANT](https://github.com/chrisjbryant/errant), a toolkit built for exactly that diff. Last, any mistake that sits on a word the speech-to-text model was unsure of is dropped, since it is more likely a transcription error than yours.

You pick the speech-to-text model with `--transcriber`:

| Name | Model | Where it runs | Times in the report | What it needs |
| --- | --- | --- | --- | --- |
| `parakeet` (default) | Parakeet Ultra | Your machine | Yes | A 740 MB model file, downloaded once |
| `elevenlabs` | ElevenLabs Scribe v3 | ElevenLabs servers | Yes | `ELEVENLABS_API_KEY` |
| `openai` | OpenAI `gpt-transcribe` | OpenAI servers | No | `OPENAI_API_KEY` |

`gpt-transcribe` returns no timestamps, so its report lists the mistakes without saying where they are.

The [design document](docs/plans/2026-11-13-joel-design.md) has every step in detail.

## Dependencies

Three things you install yourself:

| Tool | What Joel uses it for |
| --- | --- |
| [uv](https://docs.astral.sh/uv/) | Installs the Python packages and runs every command in this README |
| Python 3.13 | The language Joel is written in. If you don't have it, `uv python install 3.13` gets it |
| [ffmpeg](https://ffmpeg.org/) | Reads your audio file, whatever the format, and converts it for the local model |

`ffmpeg` has to be on your `PATH`. Most package managers have it:

```bash
# Debian or Ubuntu
sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# macOS with Homebrew
brew install ffmpeg
```

Joel is developed on Linux (x86-64) with ffmpeg 6.0.1. It has not been tried on macOS or Windows.

You also need API keys, set as environment variables:

| Variable | Needed for |
| --- | --- |
| `ANTHROPIC_API_KEY` | Always. The correction step uses Claude by default (`--model` changes that) |
| `ELEVENLABS_API_KEY` | Only with `--transcriber elevenlabs` |
| `OPENAI_API_KEY` | Only with `--transcriber openai` |

With the default transcriber your audio stays on your machine, but the text of what you said is sent to the language model provider to be corrected. The two cloud transcribers also upload the audio file.

uv installs the Python packages for you. They are listed in the design under ["Dependencies"](docs/plans/2026-11-13-joel-design.md#dependencies); the main ones are `transcribe-cpp` (runs Parakeet), `errant` and `spacy` (the diff), and `pydantic-ai-slim` (the language model calls). `pyproject.toml` doesn't declare them yet. [Issue #1](https://github.com/MarcusXavierr/joel/issues/1) adds them.

## Installation

Joel isn't on PyPI. Clone the repo and let uv set it up:

```bash
git clone https://github.com/MarcusXavierr/joel.git
cd joel
uv sync
uv run joel
```

The last command prints `Hello from joel!`, which is the whole program for now.

Once the transcribers exist, the two cloud ones will be optional extras, installed with `uv sync --all-extras`. The first run of the local transcriber will download Parakeet Ultra (740 MB) into the standard Hugging Face cache, `~/.cache/huggingface/hub`, and reuse it from there afterwards. If another app already put the same file in that cache, Joel uses it and downloads nothing.

## Usage

These are the commands as designed. They don't run yet.

```bash
# Report the mistakes in a recording
joel check talk.m4a

# Use a cloud transcriber and keep the full result as JSON
joel check talk.m4a --transcriber elevenlabs --json > talk.json

# Also show the mistakes Joel wasn't sure enough to report
joel check talk.m4a --show-uncertain
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--transcriber` | `parakeet` | Picks the speech-to-text model: `parakeet`, `elevenlabs`, or `openai` |
| `--model` | `anthropic:claude-opus-5-5` | Picks the language model, as a [Pydantic AI](https://ai.pydantic.dev/) model string |
| `--min-confidence` | `0.67` | How sure the transcription must be about a word before a mistake on it is reported |
| `--show-uncertain` | off | Shows the mistakes that fell below that bar, marked as uncertain |
| `--json` | off | Prints everything as JSON: the transcript, the corrections, and every mistake with the reason it was kept or dropped |

There is a second command, for finding out which transcriber to trust:

```bash
joel bench recordings/ --transcriber parakeet --transcriber elevenlabs --transcriber openai
```

A speech-to-text model can quietly fix your grammar while transcribing, turning "he go" into "he went", and then there is nothing left to report. `joel bench` measures how much each one does that. It takes a folder of recordings (`01.wav`, `02.wav`, ...) with a matching `.txt` for each, holding exactly what you said, mistakes included, and shows where each transcriber's text differs from yours.

## Contributing

Joel is a personal tool built in the open. The work is split into [seven issues](https://github.com/MarcusXavierr/joel/issues), and [issue #8](https://github.com/MarcusXavierr/joel/issues/8) says which order to take them in. If you want to pick one up, leave a comment on it first.

Set up as in [Installation](#installation), then check that `uv run python --version` prints `Python 3.13.x`. A few rules keep the project consistent:

- Use uv for everything (`uv add`, `uv run`, `uv sync`). Don't call `pip` or create a virtual environment by hand.
- Write the test first and watch it fail before you write the code. The fast suite will run with `uv run pytest -m "not slow"` once [issue #1](https://github.com/MarcusXavierr/joel/issues/1) sets up pytest. No test calls a real language model or touches the network.
- Everything in the repo is in English: code, comments, docs, and commit messages.
- Read the part of the [design](docs/plans/2026-11-13-joel-design.md) that covers the step you're changing. When the design and a [decision record](docs/decisions/README.md) disagree, the decision record wins.
- An accepted decision record is never edited. If a decision changes, write a new record that replaces the old one.

Regenerating the synthetic test recording needs [espeak-ng](https://github.com/espeak-ng/espeak-ng), which nothing else uses. If you work with a coding agent, [AGENTS.md](AGENTS.md) has the same rules written for it.
