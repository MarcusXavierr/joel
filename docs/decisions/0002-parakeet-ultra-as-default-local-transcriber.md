---
status: proposed
date: 2026-11-13
decision-makers: Marcus Xavier
---

# Parakeet Ultra via transcribe-cpp as the default local transcriber

## Context and Problem Statement

The project's central risk is the ASR fixing my grammar before the LLM sees it. Whisper turned "there is the uh stars" into "there are stars" in [McGuire 2025](https://arxiv.org/abs/2503.06924). I need a default ASR that runs locally, with no per-minute cost, on this machine: no NVIDIA GPU, Intel Xe, 8 cores, 30 GB of RAM.

## Decision Drivers

* Preserve the speaker's grammar mistakes.
* Per-word timestamps and confidence.
* Acceptable speed on a CPU or integrated GPU.
* A license that doesn't constrain future use.

## Considered Options

* Parakeet Ultra (Moondream's post-trained parakeet-tdt-0.6b-v3), GGUF Q8_0 through `transcribe-cpp`.
* faster-whisper (Whisper large-v3 on CTranslate2).
* CrisperWhisper 2.0 (`pip install crisperwhisper`), verbatim and intended modes.
* Cloud ASR only (ElevenLabs Scribe v2, AssemblyAI Universal-3 Pro).

## Decision Outcome

Chosen option: Parakeet Ultra, because it is a transducer without an LLM-style decoder (the structural cause of Whisper's normalization), it gives per-token probabilities, and it runs fast here. It stays `proposed` until `joel bench` confirms it on my voice.

### Consequences

* Good, because the model is already in the Hugging Face cache (Handy downloaded `handy-computer/parakeet-ultra-gguf`) and the `transcribe-cpp` 0.4.2 wheel ships the native library, so nothing needs compiling.
* Good, because I measured 8.0 s of audio transcribed in 1.38 s through Vulkan on the Intel Xe, keeping "he go", "she don't", "there is many", "I am agree", "uh", and "the the".
* Good, because CC-BY-4.0 allows commercial use.
* Bad, because that test used synthetic `espeak-ng` speech; nothing guarantees the same with real accented speech.
* Bad, because it already normalizes some mistakes on clean synthetic audio. Over 13 min: "I am agree" → "I agree", "she have" → "she has", and "make" → "made" when the sentence has "last week". That is 3–6% of mistakes erased.
* Neutral, because it segments audio over 30 s by itself (voice-activity head, cuts at pauses), so Joel needs no chunker of its own. Measured with no loss between 4.6 and 13.1 min.
* Bad, because it accepts no prompt or vocabulary (`supports` returns `False`), so there is no lever to ask for a literal transcript.
* Bad, because native timestamps are per token; word confidence is derived (minimum `p` over the word's tokens).
* Neutral, because `transcribe-cpp` calls itself "in development"; the API may change.

### Confirmation

`joel bench` over ~20 recordings of me making known mistakes. If Parakeet keeps nearly all of them, the status moves to `accepted`. If it normalizes them, this record becomes `rejected` and a new ADR picks another path.

## Pros and Cons of the Options

### Parakeet Ultra

* Good, because WER is 1.80% on LibriSpeech test-clean (Q8_0) and 5.80% on the Open ASR Leaderboard, better than the original v3.
* Good, because it has a voice-activity head that segments long audio.
* Bad, because punctuation is always on.

### faster-whisper

* Good, because it is mature, with `word_timestamps=True` and per-word `probability`.
* Bad, because Whisper's decoder is a strong language model that normalizes out-of-distribution speech, the central documented risk.
* Bad, because large-v3 on CPU is slow.

### CrisperWhisper 2.0

* Good, because verbatim mode is trained not to clean up, and the verbatim × intended difference points at where I made a mistake.
* Bad, because the license is non-commercial research only.
* Bad, because it is still Whisper underneath.

### Cloud only

* Good, because Scribe v2 and Universal-3 Pro have low overall WER and per-word confidence.
* Bad, because they cost per minute and need a network.
* Bad, because there is no independent evidence that they preserve non-native grammar mistakes either.

## More Information

CrisperWhisper is plan B if the bench rejects Parakeet. The cloud adapters stay available (ADR-0001) and go into the bench.
