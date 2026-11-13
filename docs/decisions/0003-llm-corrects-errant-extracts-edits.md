---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# The LLM writes the corrected sentence and ERRANT extracts the edits

## Context and Problem Statement

The report needs a list of mistakes, each with its exact position in the sentence (so it can become a timestamp), the original text, the correction, and the type. Something has to produce that list from the spoken sentence.

## Decision Drivers

* Precision over recall: a false mistake destroys trust in the feedback.
* Edit positions that are consistent and map back to the ASR's words.
* Error types that are stable across runs.

## Considered Options

* The LLM returns only the corrected sentence; ERRANT computes the minimal typed edits.
* The LLM returns the list of mistakes directly (span, correction, type).
* A dedicated GEC model (seq2seq like bart-large, as in the Speak & Improve baseline).

## Decision Outcome

Chosen option: the LLM corrects and ERRANT extracts, because it splits what an LLM does well (write the correct sentence) from what it does badly (list minimal edits consistently).

### Consequences

* Good, because position and type come from a deterministic algorithm: the same corrected sentence always yields the same edits.
* Good, because types follow the ERRANT taxonomy (`R:VERB:SVA`, `R:VERB:TENSE`) used across GEC benchmarks.
* Good, because I tested it here: "he go" → `R:VERB:TENSE`, "there is many" → `R:VERB:SVA`, "I am agree" → `U:VERB:TENSE`.
* Bad, because if the LLM rewrites the whole sentence, ERRANT reports a pile of edits. Quality depends on the minimal-edit instruction.
* Bad, because it brings spaCy and `en_core_web_sm` in as dependencies.
* Bad, because an ERRANT type is a technical label; a plain-language explanation of the mistake is left for later.

### Confirmation

The ~20 bench sentences go through the full pipeline. Every known mistake must show up as one edit with the right type.

## Pros and Cons of the Options

### LLM corrects, ERRANT extracts

* Good, because it is how the cascade systems in the Cambridge paper work ([Qian et al., 2025](https://arxiv.org/abs/2506.18532)).
* Bad, because there are two parts to debug when a mistake goes missing.

### LLM lists the mistakes

* Good, because it is a single call and the LLM can explain each mistake.
* Bad, because spans and types vary between runs, and the LLM tends to merge or invent edits.
* Bad, because mapping the LLM's span back to the ASR's words is fragile.

### Dedicated GEC model

* Good, because it runs locally and is trained for minimal edits.
* Bad, because models trained on written text do badly on transcribed speech; the S&I baseline sits around 45 F0.5.
* Bad, because good weights for speech (Cambridge's end-to-end model) were never published.

## More Information

Versions verified on Python 3.13: `errant` 3.0.2, `spacy` 3.8.18, `en_core_web_sm` 3.8.0. Tokenization is covered in ADR-0006.
