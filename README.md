# Joel

**"Do you understand?"** Your spoken English, every grammar mistake, the exact second you made it.

Joel is a command-line tool for people who speak English as a second language. You give it a recording of yourself talking. It tells you which sentences had a grammar mistake, when in the recording you said them, and what you should have said.

> [!IMPORTANT]
> Joel works, but it is young. `joel check`, `joel bench`, and `joel estimate` all run, and the seven tickets of stage 1 are merged. So far Joel has only been tried on short test recordings and synthetic speech. Parakeet stays the default transcriber until `joel bench` has run over about 20 real recordings ([issue #8](https://github.com/MarcusXavierr/joel/issues/8)), and until then its [decision record](docs/decisions/0002-parakeet-ultra-as-default-local-transcriber.md) stays `proposed`. The repo holds the [design](docs/plans/2026-11-13-joel-design.md) and the [decisions](docs/decisions/README.md) behind it.

## What a report looks like

```
$ joel check talk.m4a --better

00:12.0–00:15.1
  said:    There is many people in the room.
  correct: There are many people in the room.
           00:12.4  is → are  R:VERB:SVA   grammar

00:40.6–00:42.0
  said:    I make a party last week.
  better:  I threw a party last week.
           00:41.0  make → threw  naturalness

Cost: $0.003 for 2 model calls (1563 input and 368 output tokens) with openai-codex:gpt-6.1-terra.
```

Each entry is one sentence, headed by where to find it in the recording. Under what you said comes the corrected sentence, then one line per change, starting with the second you said it. `R:VERB:SVA` is the kind of mistake, here a verb that doesn't agree with its subject. The `better:` lines appear only with `--better`: a `naturalness` change is a word a native speaker would swap even though the sentence is already grammatical. The `said:` line leaves out the hesitations Joel set aside. The last line is what the language model calls would cost at API prices. With the default model your ChatGPT plan covers them, so nothing is billed per token.

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
| [uv](https://docs.astral.sh/uv/) | Installs Joel and its Python packages |
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

The language model runs on your ChatGPT plan by default, through the login of OpenAI's [Codex CLI](https://github.com/openai/codex). Install it and run `codex login` once; Joel reads that login and never changes it. This route isn't one OpenAI supports for other apps, so it may stop working, and it can use up your plan's limits ([ADR-0011](docs/decisions/0011-default-model-gpt-6-1-terra-on-the-chatgpt-plan.md)). Without a ChatGPT plan, pass `--model openai:gpt-6.1-terra` and set `OPENAI_API_KEY`.

Other choices need API keys, set as environment variables:

| Variable | Needed for |
| --- | --- |
| `OPENAI_API_KEY` | An `openai:` model in `--model`, or `--transcriber openai` |
| `ANTHROPIC_API_KEY` | An `anthropic:` model in `--model` |
| `ELEVENLABS_API_KEY` | `--transcriber elevenlabs` |

With the default transcriber your audio stays on your machine, but the text of what you said is sent to the language model provider to be corrected. The two cloud transcribers also upload the audio file.

uv installs Joel's Python packages when you install Joel. They are listed in the design under ["Dependencies"](docs/plans/2026-11-13-joel-design.md#dependencies); the main ones are `transcribe-cpp` (runs Parakeet), `errant` and `spacy` (the diff), and `pydantic-ai-slim` (the language model calls). Pydantic AI comes with every provider of text models it supports, so `--model` can name a model from any of them. TypeSafe is left out because its models answer typed questions and can't write corrected sentences.

ERRANT also needs spaCy's English model, `en_core_web_sm`, which isn't on PyPI. The first `joel check` downloads it (13 MB) from [spaCy's GitHub release](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0), checks it against a pinned hash, and keeps it in `~/.cache/joel` (`$XDG_CACHE_HOME/joel` when that's set). [ADR-0012](docs/decisions/0012-download-the-spacy-model-on-first-use.md) explains why.

## Installation

Joel is on PyPI. Install it as a uv tool, which gives Joel its own environment and puts `joel` on your `PATH`:

```bash
uv tool install joel
joel --help
```

The second command lists Joel's commands. The SDKs for the two cloud transcribers are optional extras: `uv tool install 'joel[elevenlabs,openai]'` installs both, or name only the one you need. `uv tool upgrade joel` updates Joel later.

Two models download the first time Joel needs them. The local transcriber downloads Parakeet Ultra (740 MB) into the standard Hugging Face cache, `~/.cache/huggingface/hub`, and reuses it from there. If another app already put the same file in that cache, Joel uses it and downloads nothing. `joel check` also downloads spaCy's `en_core_web_sm` (13 MB) into `~/.cache/joel` on its first run, as [Dependencies](#dependencies) describes.

## Usage

`joel check` is the command you'll use most:

```bash
# Report the mistakes in a recording
joel check talk.m4a

# Also save the report as JSON
joel check talk.m4a --json talk.json

# Use a cloud transcriber, and see what a native speaker would say instead
joel check talk.m4a --transcriber elevenlabs --better

# Also show the mistakes Joel wasn't sure enough to report
joel check talk.m4a --show-uncertain
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--transcriber` | `parakeet` | Picks the speech-to-text model: `parakeet`, `elevenlabs`, or `openai` |
| `--model` | `openai-codex:gpt-6.1-terra` | Picks the language model, as a [Pydantic AI](https://ai.pydantic.dev/) model string |
| `--min-confidence` | `0.67` | How sure the transcription must be about a word before a mistake on it is reported |
| `--show-uncertain` | off | Shows the mistakes that fell below that bar, marked as uncertain |
| `--block-sentences` | `7` | How many sentences go to the language model in each call. Fewer means more calls and a higher cost |
| `--better` | off | Also shows the words a native speaker would swap, under `better:`. The model only writes this version when you ask. If it does more than swap a few words, Joel sends it back once and then drops it ([ADR-0014](docs/decisions/0014-naturalness-only-on-request-held-to-word-swaps.md)) |
| `--json PATH` | none | Also writes the report to `PATH` as JSON: every sentence with a mistake, every mistake with its times and confidence and, if it was held back, why. Uncertain mistakes are in it even without `--show-uncertain`. It also has a hash of each prompt used and the tokens and cost of the run. The terminal report is rebuilt from this file |
| `--verbose` | off | With `--json`, writes everything the run decided instead: the transcript, the dropped hesitations, every correction and every edit |
| `--prompt NAME=FILE` | none | Sends the instructions in `FILE` instead of the built-in `NAME` prompt. Repeat it for several. See [Custom prompts](#custom-prompts) |

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
openai-codex:gpt-6.1-terra: about $0.045 for 36 calls (26436 input and 3756 output tokens)
Assumes 150 words per minute, 12 words per sentence and 7 sentences per call, with no prompt caching; a retried call costs extra.
```

The estimate only looks at how long the recording is. Joel sends the sentences to the language model in blocks of 7, and each block costs two calls, one to set the hesitations aside and one to correct it. `--block-sentences` changes the block size here and in `joel check`. `--better` prices the natural version too, which makes each correction longer: the same 10 minutes come to about $0.057. `--model` prices a different model, as long as [genai-prices](https://github.com/pydantic/genai-prices) knows it.

### Custom prompts

The instructions Joel sends the language model live in text files inside the package, one per prompt. `joel check --help` lists them:

| Name | What it asks for |
| --- | --- |
| `disfluencies` | Which words are hesitations to set aside: "uh", repeated words, false starts |
| `correct` | The grammatical version of each sentence |
| `correct-natural` | The natural version too. Joel adds it to `correct` only with `--better` |

To change one, start from the built-in text, edit it, and pass your file:

```bash
joel prompt correct > mine.md
joel check talk.m4a --prompt correct=mine.md
```

The built-in `correct` prompt has a line that says only `$natural`. That is where `correct-natural` goes when you pass `--better`, and where nothing goes when you don't. Keep it in your file if you want `--better` to work; without it, Joel stops with an error before transcribing anything. Any other `$` word is an error too, and `$$` writes a literal `$`. The JSON report records a SHA-256 hash of every prompt a run used, so you can tell which text produced it.

Everything measured about Joel, from the model probes in the decision records to the token rates `joel estimate` uses, was measured with the built-in prompts. Your file can make the report better or worse, and the estimate says so when you pass `--prompt`.

## Using Joel from another program

Run `joel check <audio> --json result.json` and read the file. It holds the report: every sentence with a mistake, and every mistake with its times, its confidence and, if it was held back, why, plus the hash of each prompt used and the tokens and cost. Add `--verbose` and the file holds the whole result of the run instead, transcript and every correction included. Standard output keeps the human report either way. On an error Joel prints the reason to standard error and exits with 1. The Python modules inside the package aren't a stable API and can change in any release, so don't import them.

## Contributing

Joel is a personal tool built in the open. The tickets of stage 1 are closed, and open work lives in [the issues](https://github.com/MarcusXavierr/joel/issues). If you want to pick one up, leave a comment on it first.

To work on Joel, clone it and let uv set up the environment:

```bash
git clone https://github.com/MarcusXavierr/joel.git
cd joel
uv sync --all-extras
uv run joel --help
```

Then check that `uv run python --version` prints `Python 3.13.x`. The dev group installs `en_core_web_sm` as a package, so the tests never download it. A few rules keep the project consistent:

- Use uv for everything (`uv add`, `uv run`, `uv sync`). Don't call `pip` or create a virtual environment by hand.
- Write the test first and watch it fail before you write the code. The fast suite runs with `uv run pytest`, which skips tests marked `slow` (they load a model or need ffmpeg); `uv run pytest -m slow` runs those. No test calls a real language model or touches the network. `uv run ruff check src tests` and `uv run mypy src tests` must pass too.
- Everything in the repo is in English: code, comments, docs, and commit messages.
- Read the part of the [design](docs/plans/2026-11-13-joel-design.md) that covers the step you're changing. When the design and a [decision record](docs/decisions/README.md) disagree, the decision record wins.
- An accepted decision record is never edited. If a decision changes, write a new record that replaces the old one.

Regenerating the synthetic test recording, `tests/fixtures/audio/acceptance.wav`, needs [espeak-ng](https://github.com/espeak-ng/espeak-ng), which nothing else uses. The exact command is in a comment in `tests/test_parakeet.py`. If you work with a coding agent, [AGENTS.md](AGENTS.md) has the same rules written for it.

## License

MIT. See [LICENSE](LICENSE).
