---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# Disfluency cleanup with the LLM returning indices to drop

## Context and Problem Statement

Hesitation is not a grammar mistake. "um/uh", repetitions ("the the"), and false starts ("I went, I was going") have to leave the sentence before correction, or ERRANT reports them as mistakes. At the same time, every word that stays must remain tied to its timestamp in the audio.

## Decision Drivers

* Keep the clean word → ASR word → timestamp map exact.
* Don't let cleanup fix the grammar (the same risk as the ASR, now in the LLM).
* Catch false starts, not only fillers.

## Considered Options

* Deterministic rules (a filler list plus adjacent repetition).
* The LLM receives the sentence as numbered words and returns only the indices to drop.
* The LLM returns the cleaned text and a diff realigns it with the original words.

## Decision Outcome

Chosen option: the LLM returns indices, because the map stays exact and false starts get caught, while the LLM has no way to add or change a word.

### Consequences

* Good, because the timestamp map is trivial: the words that remain are the originals.
* Good, because the LLM can't rewrite "there is many" as "there are many".
* Bad, because it can still erase a real mistake by deletion. In "I am agree", dropping "am" as a false start "fixes" the sentence and the mistake is gone.
* Bad, because it is one more LLM call per sentence, with cost and latency.
* Neutral, because an index outside the sentence fails deterministic validation and the call is retried once.

### Confirmation

Dropped indices show up in `--json` for auditing. The bench with the full pipeline checks that no known mistake among the ~20 sentences gets dropped as a disfluency.

## Pros and Cons of the Options

### Deterministic rules

* Good, because it is free, predictable, and testable without a network.
* Bad, because it misses false starts, the most common disfluency in spontaneous non-native speech.
* Bad, because adjacent repetition is sometimes intended ("very very").

### LLM returns indices

* Good, because it pairs the LLM's judgment with a structural guarantee that it can only remove.
* Bad, because deletion can still hide a mistake, as described above.

### LLM returns cleaned text

* Good, because the prompt is the simplest.
* Bad, because the LLM tends to fix grammar while cleaning.
* Bad, because diff realignment fails exactly when the LLM changed words, which is the case that matters.

## More Information

The original brief planned a deterministic step (`strip_disfluencies`) with the same index map. This decision only changes who decides what to drop.
