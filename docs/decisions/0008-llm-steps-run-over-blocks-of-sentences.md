---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# The LLM steps run over blocks of sentences

## Context and Problem Statement

Disfluency cleanup (ADR-0004) and correction (ADR-0003, ADR-0005) each made one model call per sentence. A 1-minute recording came to 26 calls, and `joel estimate` put them at 14,586 input and 858 output tokens on `anthropic:claude-opus-5-5`. About 13,000 of those input tokens were the same instructions and tool schema, sent again with every sentence. The run cost $0.05 to $0.08 a minute, mostly to repeat the prompt.

## Decision Drivers

* Pay for the instructions once per call, and make fewer calls.
* Keep each sentence's correction its own: no words moving between sentences, and each answer tied to the right sentence.
* Keep the calls concurrent, so a long recording doesn't run serially.
* Keep `strip_disfluencies` and `correct` as single-call steps that can be tested alone with `FunctionModel`.

## Considered Options

* One sentence per call, as before.
* Blocks of up to N sentences per call, with the sentences numbered and the numbering validated.
* The whole recording in one call per step.

## Decision Outcome

Chosen option: blocks of up to N sentences, N=6 by default and set with `--block-sentences N` on `joel check` and `joel estimate`. `run_check` slices the sentences with `itertools.batched`, so the two steps don't know about blocks at all. Cleanup gets the block's words as one passage and its dropped positions are split back per sentence. Correction gets the block as numbered lines between the sentence before it and the one after it, and the answer has to come back as exactly one item per number, 1 to n, in order. Any other numbering is retried once, like every invalid answer.

On the acceptance recording (3 sentences), the run went from 6 calls to 2, at about 1,400 input tokens.

### Consequences

* Good, because the instructions are paid once per block instead of once per sentence: a minute of speech is 6 calls instead of 26.
* Good, because the blocks still run concurrently, up to MAX_CONCURRENT_CALLS at a time.
* Good, because the model sees more of the conversation while correcting, which helps with tense and reference.
* Bad, because one misnumbered answer retries the whole block, so a retry costs up to N sentences of tokens instead of one.
* Bad, because a model can move words from one numbered sentence to another. The numbering check can't see that, and ERRANT then reports the moved words as mistakes in both sentences.
* Bad, because a block that fails after its retry fails the whole run, and a block holds more of the recording than a sentence did.
* Neutral, because the right N depends on the model: a cheaper model may lose precision on longer blocks. The model probe (ADR-0010) measures that.

### Confirmation

`tests/test_check.py` runs the same three-sentence fixture with the default block and with `block_sentences=1`, and expects the same four edits from 2 and from 6 calls. `tests/test_correct.py` covers the numbering check and its retry.

## Pros and Cons of the Options

### One sentence per call

* Good, because a retry repeats one sentence only, and no answer can be matched to the wrong sentence.
* Bad, because about 500 instruction tokens are repeated on every call, which made up roughly 90% of the input.

### Blocks of N sentences

* Good, because it cuts most of the repeated prompt and keeps the calls concurrent.
* Bad, because of the misnumbering and word-moving risks described above.

### The whole recording in one call

* Good, because the instructions are sent exactly once per step.
* Bad, because nothing runs concurrently, so a long recording waits on one long call.
* Bad, because the chance of misaligned answers grows with length, and a retry repeats everything.
* Bad, because one call's output gets long enough to hit output limits on long recordings.

## More Information

The cost estimate in `joel estimate` now counts one base cost per block plus a per-word cost, and its token rates are refit against blocks (see `src/joel/cost.py`).
