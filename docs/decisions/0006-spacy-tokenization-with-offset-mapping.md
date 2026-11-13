---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# spaCy tokenization in ERRANT with character offset mapping

## Context and Problem Statement

ERRANT reports edits as token positions. Each position has to map back to an ASR word to find its timestamp. The obvious route is to pass the ASR words as ready-made tokens (`tokenise=False`), which makes the map trivial.

## Considered Options

* ASR words as tokens (`tokenise=False`), map by index.
* spaCy tokenization (`tokenise=True`), map each token to its ASR word by character offset (`token.idx`) in the joined string.

## Decision Outcome

Chosen option: spaCy tokenization with offset mapping, because ERRANT's types depend on it. Measured on 2026-11-13: with `tokenise=False`, "she don't like it" → "she doesn't like it" comes out as `don't → doesn't R:PUNCT`. With spaCy ("do" + "n't") it comes out as `do → does R:VERB:SVA`, which is correct.

### Consequences

* Good, because ERRANT types stay correct on contractions, which is where common agreement mistakes happen.
* Bad, because one ASR word can become several tokens, so the map needs its own function with tests (contractions, attached punctuation).
* Bad, because an edit span can cover part of an ASR word; the timestamp used is the whole word's.
