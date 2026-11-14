---
status: accepted
date: 2026-11-14
decision-makers: Marcus Xavier
---

# gpt-6.1-terra as the default model, run on the ChatGPT plan through the Codex login

## Context and Problem Statement

ADR-0010 made `openai:gpt-6-luna` the default because every model reached precision 1.000 on the probe's gold set, which left cost to decide. The gold turned out too easy. On Marcus's real 1-minute recording, `gpt-6-luna` reported the false start "help users I would say but not fixing" as a grammar mistake (`U:OTHER`), proposed a `to → it` change that makes no sense alone, and added five naturalness lines that only shortened sentences, which the prompt forbids ("Do not rephrase... or improve style"). `gpt-6.1-terra` on the same recording found the same real mistakes with none of that noise. Marcus preferred its report.

Precision is the rule Joel is built on, so the report a user reads beats a perfect score on an easy gold. The cost of `gpt-6.1-terra` was the only argument against it, and Marcus pays for a ChatGPT plan that pydantic-ai can already use.

## Decision Drivers

* Precision on real speech, with false starts and run-on sentences, which the gold didn't have.
* No per-token bill for Marcus's own use.
* No new code in Joel for a login flow.

## Considered Options

* `openai:gpt-6-luna` with an API key (ADR-0010).
* `openai:gpt-6.1-terra` with an API key.
* `openai-codex:gpt-6.1-terra`, which runs on the ChatGPT plan with the Codex CLI's login.
* `openai-chatgpt:gpt-6.1-terra`, which runs on the ChatGPT plan through Sign in with ChatGPT.
* `anthropic:claude-opus-5-5`.

## Decision Outcome

Chosen option: `openai-codex:gpt-6.1-terra`, because it gives the cleaner report and runs on the plan Marcus already pays for, with no code beyond the model string. pydantic-ai's `openai-codex` provider reads the Codex CLI's `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`) without writing it, and a missing login stops `joel check` before transcription with "Run `codex login` first". The block size stays at 7.

`--model openai:gpt-6.1-terra` runs the same model on an API key, for anyone without a ChatGPT plan.

### Consequences

* Good, because the default needs no API key and Marcus's runs add nothing to an API bill.
* Good, because `gpt-6.1-terra` kept precision at 1.000 and recall at 0.983 or above in ADR-0010's probe, so the switch loses nothing on the gold.
* Bad, because the `openai-codex` provider calls `chatgpt.com/backend-api/codex`, which OpenAI doesn't support for third-party apps. Its Sign in with ChatGPT docs say not to point apps at ChatGPT's `backend-api` endpoints. The route can change or close without notice, and using it may break OpenAI's terms.
* Bad, because Joel borrows the Codex CLI's login. When the access token expires, pydantic-ai refreshes it in memory and never writes the new tokens back. If OpenAI's refresh tokens are single-use, that refresh can sign the Codex CLI out. This hasn't been tested; using `codex` shortly before Joel keeps the stored token fresh.
* Bad, because the plan has usage limits, so a long recording can run out partway and fail with the provider's error.
* Bad, because the cost line keeps pricing tokens at the API's rates: genai-prices prices `openai-codex` like `openai`. On the plan, the line says what the run would have cost on an API key, not what was charged.
* Neutral, because prompt caching still does nothing here: the prompts are about 500 tokens in OpenAI's count, under its 1,024-token minimum (ADR-0009).
* Neutral, because `joel estimate`'s token rates are refit against this model.

### Confirmation

`test_a_missing_codex_login_stops_the_run_before_transcribing` covers the missing login. Every test points `CODEX_HOME` at an empty directory, so none can read a real login. On 2026-11-14, `joel check` with the new default flagged the four mistakes in the fixture recording, and on Marcus's recording it flagged `help → helps`, `recognizing → recognize`, `about → of` and `in → ∅` with no naturalness lines.

## Pros and Cons of the Options

### `openai:gpt-6-luna` with an API key

* Good, because it is the cheapest model that kept full recall on the gold, under $0.001 a minute.
* Bad, because of the noise on real speech described above.

### `openai:gpt-6.1-terra` with an API key

* Good, because it is OpenAI's supported route and needs no login borrowed from another tool.
* Bad, because it costs about $0.005 to $0.006 a minute on an API key, about $0.35 an hour of speech.

### `openai-codex:gpt-6.1-terra`

* Good, because it costs nothing per token for someone who already has a ChatGPT plan, and pydantic-ai supports it out of the box.
* Bad, because of the unsupported endpoint and the borrowed login described above.

### `openai-chatgpt:gpt-6.1-terra`

* Good, because it is the route OpenAI documents for open-source and locally run apps: the plan pays for requests to the public `api.openai.com/v1/responses`.
* Bad, because Joel would need a `joel login` command that opens the browser, catches the loopback callback, stores the credentials and a stable host ID, and saves refreshed tokens. pydantic-ai provides the flow but leaves storage to the app. That is work for its own change, and a later ADR can move the default there.

### `anthropic:claude-opus-5-5`

* Good, because it handled the false start in Marcus's recording as well as `gpt-6.1-terra` did.
* Bad, because it costs about $0.03 a minute and needs an Anthropic API key.

## More Information

This supersedes [ADR-0010](0010-default-model-chosen-by-precision-probe.md), whose probe results still stand. Pydantic-ai's `openai-codex` provider lives in `pydantic_ai/providers/openai_codex.py`. OpenAI's guidance for apps that use a ChatGPT plan is at https://developers.openai.com/siwc/token-sharing-open-source.
