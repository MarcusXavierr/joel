---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# Word timestamps are optional in the transcriber contract

## Context and Problem Statement

I want OpenAI's `gpt-transcribe` in the lineup next to Parakeet Ultra and ElevenLabs Scribe v3. The key is already in my environment, and it is the model OpenAI recommends for recorded speech. [ADR-0001](0001-own-transcriber-interface-with-one-adapter-per-sdk.md) left it out because it returns no word timestamps, and the `Word` contract there requires `start_ms` and `end_ms`. How does a transcriber with no timestamps fit behind the same interface?

Measured on 2026-11-13 with one 9 s `espeak-ng` clip ("Yesterday he go to school, uh, and she don't like it. There is many people in the the room. I am agree with you."):

* `gpt-transcribe` returns `{"text", "languages", "usage"}` and nothing per word. `response_format=verbose_json` fails with "not compatible with model", so `timestamp_granularities` can't even be requested.
* `include[]=logprobs` works and returns one log probability per token (30 for this clip).
* It kept "he go", "she don't", "there is many", "I am agree", and "uh". It collapsed "the the" into "the" and heard "Yesterday" as "Yes, today".

## Decision Drivers

* `joel bench` should compare every transcriber I can call, and three of its four measurements need no timestamps.
* Confidence still has to reach the filter (step 6).
* No extra pipeline step for the sake of one provider.

## Considered Options

* `start_ms` and `end_ms` become `int | None` in `Word`. A transcriber without timestamps returns `None`, the way one without confidence already does.
* Borrowed timestamps: run Parakeet on the same audio, align the two word lists by edit distance, and copy the times across where the words match.
* A separate text-only path used by `joel bench` alone, with `joel check` rejecting `--transcriber openai`.
* Keep `gpt-transcribe` out, as ADR-0001 has it.

## Decision Outcome

Chosen option: optional timestamps, because it is the smallest change that puts `gpt-transcribe` behind the same interface in both commands. It doesn't rule out borrowed timestamps later: alignment would fill the `None` fields without touching the contract again.

This replaces two parts of ADR-0001: the `Word` shape in its option list, and the note that `gpt-transcribe` is out. The rest of ADR-0001 stands (own interface, one adapter per official SDK).

### Consequences

* Good, because the bench runs all three transcribers through one code path.
* Good, because confidence is still there. The adapter takes the minimum `exp(logprob)` over each word's tokens, the same rule Parakeet uses.
* Bad, because `joel check --transcriber openai` reports mistakes with no audio span, and pointing at the audio is half of what Joel is for.
* Bad, because everything that reads `Word` now handles `None` times: `render`, the JSON output, and the bench's timestamp check.
* Bad, because `gpt-transcribe` can't become the default even if it wins the bench, unless borrowed timestamps get built.
* Bad, because the API returns no word list. The adapter splits `text` on whitespace and maps token logprobs to words by character offset.
* Neutral, because uploads are capped at 25 MB. The adapter fails on a larger file and names the limit; it doesn't split or re-encode.

### Confirmation

A contract test for the OpenAI adapter against a recorded response: every `Word` has `start_ms` and `end_ms` set to `None` and a confidence between 0 and 1. A `render` test with `None` times prints the report without the time column and says the transcriber provides no timestamps.

## Pros and Cons of the Options

### Optional timestamps

* Good, because `confidence: float | None` already set the pattern.
* Bad, because a field the product depends on becomes nullable for one provider.

### Borrowed timestamps

* Good, because the report keeps its audio spans.
* Good, because two transcribers agreeing on a word is evidence that I said it.
* Bad, because a word the two transcribers disagree on gets no time, so coverage is partial and I haven't measured how partial.
* Bad, because every `openai` run also pays for a Parakeet run, and alignment is a new step to write and test.

### Text-only bench path

* Good, because the `Word` contract stays strict.
* Bad, because the bench would branch on the kind of transcriber, which ADR-0001 set out to avoid.
* Bad, because I couldn't run `joel check` with it at all, not even to read its corrections.

### Keep it out

* Good, because nothing changes.
* Bad, because the bench would skip a model I already pay for, and on the one clip above it kept all four grammar mistakes.

## More Information

The lineup changed at the same time. AssemblyAI Universal-3 Pro is gone: I have no API key for it. ElevenLabs moved from `scribe_v2` to `scribe_v3`, released on 2026-11-10 with the same API. On the clip above `scribe_v3` kept all four mistakes, "uh", and "the, the", with word timestamps and a non-zero `logprob` on every word. Which providers sit behind the interface is a PRD matter, so those two changes get no record of their own.

`gpt-transcribe` response shape and limits: [OpenAI file transcription guide](https://developers.openai.com/api/docs/guides/speech-to-text).
