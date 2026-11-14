# Joel

**"Do you understand?"** Your spoken English, every grammar mistake, the exact second you made it.

Joel is a command-line tool for people who speak English as a second language. You give it a recording of yourself talking. It tells you which sentences had a grammar mistake, when in the recording you said them, and what you should have said.

> [!IMPORTANT]
> Joel is at an early stage. `joel check`, `joel bench`, and `joel estimate` all work, but they have only been tried on short test recordings. The repo holds the [design](docs/plans/2026-11-13-joel-design.md), the [decisions](docs/decisions/README.md) behind it, and a [backlog of seven tickets](https://github.com/MarcusXavierr/joel/issues/8). Details may still move.

## What a report looks like

```
$ joel check talk.m4a

00:12.4–00:13.1  R:VERB:SVA   grammar
  said:    There is many people in the room.
  correct: There are many people in the room.
           is → are

00:41.0–00:41.6  naturalness
  said:    I make a party last week.
  better:  I threw a party last week.
           make → threw

Cost: $0.02 for 6 model calls (3204 input and 144 output tokens) with anthropic:claude-opus-5-5.
```

Each entry starts with where to find the sentence in the recording. `R:VERB:SVA` is the kind of mistake, here a verb that doesn't agree with its subject. A `naturalness` entry is a sentence that is grammatical but that a native speaker wouldn't say that way. The `said:` line leaves out the hesitations Joel set aside, and the line under the correction shows exactly which words changed. The last line is what the language model calls cost for this recording.

Joel would rather miss a mistake than invent one. A report that flags things you said correctly teaches you to ignore it, so anything the transcription was unsure about stays hidden unless you ask for it with `--show-uncertain`. The report still tells you how many it hid.

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

uv installs the Python packages for you. They are listed in the design under ["Dependencies"](docs/plans/2026-11-13-joel-design.md#dependencies); the main ones are `transcribe-cpp` (runs Parakeet), `errant` and `spacy` (the diff), and `pydantic-ai-slim` (the language model calls). `pyproject.toml` declares all of them, so `uv sync` installs them. Pydantic AI comes with every provider of text models it supports, so `--model` can name a model from any of them. TypeSafe is left out because its models answer typed questions and can't write corrected sentences.

## Installation

Joel isn't on PyPI. Clone the repo and let uv set it up:

```bash
git clone https://github.com/MarcusXavierr/joel.git
cd joel
uv sync
uv run joel --help
```

The last command lists the commands that exist so far.

The SDKs for the two cloud transcribers are optional extras, installed with `uv sync --all-extras`. The first run of the local transcriber will download Parakeet Ultra (740 MB) into the standard Hugging Face cache, `~/.cache/huggingface/hub`, and reuse it from there afterwards. If another app already put the same file in that cache, Joel uses it and downloads nothing.

## Usage

`joel check` is the command you'll use most:

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
| `--json` | off | Prints everything as JSON: the transcript, the corrections, every mistake with the reason it was kept or held back, and the tokens and cost of the run. The terminal report can be rebuilt from it |

There is a second command, for finding out which transcriber to trust:

```bash
joel bench recordings/ --transcriber parakeet --transcriber elevenlabs --transcriber openai
```

A speech-to-text model can quietly fix your grammar while transcribing, turning "he go" into "he went", and then there is nothing left to report. `joel bench` measures how much each one does that. It takes a folder of recordings (`01.wav`, `02.wav`, ...) with a matching `.txt` for each, holding exactly what you said, mistakes included. Put each mistake in square brackets:

```
Yesterday he [go] to school, uh, and she [don't] like it.
```

For each transcriber you get a table with the word error rate and how many of your bracketed mistakes came through untouched, then a line for every file where the transcript differs from what you said, like `01  yesterday he [go → went] to school`. A transcriber that can't run here, such as a cloud one without its extra or its API key, shows an error in place of its table, and the others still run.

Before you send a long recording through a paid model, you can ask what it would cost:

```
$ joel estimate talk.m4a
talk.m4a: 10:00 of audio, about 1500 words in 125 sentences
anthropic:claude-opus-5-5: about $0.73 for 250 calls (140250 input and 8250 output tokens)
Assumes 150 words per minute and 12 words per sentence; a retried call costs extra.
```

The estimate only looks at how long the recording is. Each sentence costs two calls, one to set the hesitations aside and one to correct it. `--model` prices a different model, as long as [genai-prices](https://github.com/pydantic/genai-prices) knows it.

## Contributing

Joel is a personal tool built in the open. The work is split into [seven issues](https://github.com/MarcusXavierr/joel/issues), and [issue #8](https://github.com/MarcusXavierr/joel/issues/8) says which order to take them in. If you want to pick one up, leave a comment on it first.

Set up as in [Installation](#installation), then check that `uv run python --version` prints `Python 3.13.x`. A few rules keep the project consistent:

- Use uv for everything (`uv add`, `uv run`, `uv sync`). Don't call `pip` or create a virtual environment by hand.
- Write the test first and watch it fail before you write the code. The fast suite runs with `uv run pytest`, which skips tests marked `slow` (they load a model or need ffmpeg); `uv run pytest -m slow` runs those. No test calls a real language model or touches the network. `uv run ruff check src tests` and `uv run mypy src tests` must pass too.
- Everything in the repo is in English: code, comments, docs, and commit messages.
- Read the part of the [design](docs/plans/2026-11-13-joel-design.md) that covers the step you're changing. When the design and a [decision record](docs/decisions/README.md) disagree, the decision record wins.
- An accepted decision record is never edited. If a decision changes, write a new record that replaces the old one.

Regenerating the synthetic test recording, `tests/fixtures/audio/acceptance.wav`, needs [espeak-ng](https://github.com/espeak-ng/espeak-ng), which nothing else uses. The exact command is in a comment in `tests/test_parakeet.py`. If you work with a coding agent, [AGENTS.md](AGENTS.md) has the same rules written for it.
