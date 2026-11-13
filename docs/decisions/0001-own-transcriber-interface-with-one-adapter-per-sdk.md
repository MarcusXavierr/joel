---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# Own Transcriber interface with one adapter per official SDK

## Context and Problem Statement

The pipeline needs four things per word from the ASR: text, start, end, and confidence. Without timestamps we can't point at the audio span. Without confidence we can't drop edits caused by ASR mistakes. I want to switch ASR with a flag (local and cloud) to compare which one preserves my grammar mistakes.

## Decision Drivers

* Per-word timestamps and confidence must reach the later steps intact.
* Local and cloud transcribers run through the same pipeline.
* Few dependencies, no framework.

## Considered Options

* Own `Transcriber` interface, one adapter per official SDK, all returning `Word(index, text, start_ms, end_ms, confidence | None)`.
* LiteLLM `transcription()` as a unified layer.
* A single provider, no abstraction.

## Decision Outcome

Chosen option: own interface with one adapter per SDK, because it is the only option that delivers per-word timestamps and confidence from every provider.

### Consequences

* Good, because each adapter reads the provider's native confidence field (`Token.p`, `logprob`, `confidence`) and normalizes it to 0..1.
* Good, because `joel bench` compares transcribers without touching anything after `transcribe`.
* Bad, because every new provider costs a hand-written adapter and contract test.
* Bad, because provider API changes land on me, not on a library someone else maintains.

### Confirmation

One contract test per adapter against a recorded provider response. Switching `--transcriber` changes no later step.

## Pros and Cons of the Options

### Own interface

* Good, because the `Word` contract is exactly what the pipeline needs.
* Bad, because it is integration code to maintain.

### LiteLLM

* Good, because one call covers dozens of providers.
* Bad, because it normalizes to the OpenAI shape. Its ElevenLabs adapter (`llms/elevenlabs/audio_transcription/transformation.py`) reduces each word to `{word, start, end}`, drops `logprob`, and filters out anything whose `type` isn't `"word"`. The raw response survives only in `hidden_params`.
* Bad, because it can't run a local model in process.
* Bad, because versions 1.82.7 and 1.82.8 on PyPI were compromised, a supply chain risk in a large dependency.

### Single provider

* Good, because it is the least code.
* Bad, because picking the ASR depends on the bench, which needs several to compare.

## More Information

OpenAI `gpt-transcribe` is out: it returns no word-level timestamps, and `timestamp_granularities` only works on `whisper-1`, which shuts down on 2027-02-26 ([docs](https://developers.openai.com/api/docs/guides/speech-to-text)). Pydantic AI has no file STT, only realtime sessions.
