---
status: accepted
date: 2026-11-14
decision-makers: Marcus Xavier
---

# Naturalness only on request, held to word swaps

## Context and Problem Statement

ADR-0005 made naturalness a second target in the same correction call: every run asked the model for a `natural` version of every sentence and reported the difference. ADR-0011 records what that did on a real recording: `gpt-6-luna` added five naturalness lines that only shortened sentences, although the prompt said not to rephrase. A rewrite shown as a `naturalness` change tells me I said something wrong when I didn't, which is the false positive Joel is built to avoid. Even with a model that behaves, every run paid output tokens for a version most reports then showed nothing of.

I still want the naturalness section, because "I make a party" is exactly the kind of thing I can't hear myself. I want it when I ask for it, and I want it limited to what it's good at: swapping the word a native speaker wouldn't use.

## Decision Drivers

* Precision: a rewrite must never reach the report as the speaker's mistake.
* No tokens spent on a natural version nobody asked for.
* A check that doesn't depend on the model following the prompt, and that a unit test can pin down.
* A run must never fail because of naturalness, since grammar is what the report is for.

## Considered Options

* Keep ADR-0005: always ask for `natural`, always show it.
* Always ask for `natural`, show it only with `--better`.
* Ask only with `--better`, and trust the prompt to keep it to word swaps.
* Ask only with `--better`, through its own output schema and prompt fragment, and hold it to word swaps with a deterministic guard that sends a rewrite back once and then drops it.
* Ask only with `--better`, and have a second model call judge each natural version.

## Decision Outcome

Chosen option: ask only with `--better`, with its own schema and a deterministic guard, because it is the only option where neither the cost nor the precision depends on the model obeying the prompt.

Without `--better`, the correction step uses an agent whose output schema has no `natural` field, so the model can't write one, and the `correct` prompt's `$natural` line is left empty (ADR-0013). With `--better`, a second agent asks for `natural`, and the `correct-natural` prompt fills `$natural`. That prompt asks for a word or fixed phrase a native speaker would never use there ("made a party" → "threw a party", "said a lie" → "told a lie") and forbids adding or deleting words.

The guard runs in the agent's output validator. For each sentence, `run_check` diffs `grammatical` against `natural` with ERRANT, using step 5's trimming of changes nobody can hear, and passes those edits to `natural_problem`. It allows only replacements (`R:` types), at most `NATURAL_MAX_WORDS` = 3 words on either side of one, and at most `NATURAL_MAX_EDITS` = 2 per sentence. The guard diffs against `grammatical`, not against what was said, because the report drops every naturalness edit that overlaps a grammar fix: an added word next to a fixed verb would vanish from the mistake list and still be printed in the `better:` line. If any sentence breaks a rule and the step still has its one retry, the model gets `Fix natural in these sentences and change nothing else:` followed by one line per sentence, such as `Sentence 2: natural adds or removes words ("in the end → ∅"); only replace words.` On the last try, each sentence that still breaks a rule gets `natural = grammatical` and the run goes on. That retry is shared with numbering errors, so if it comes back misnumbered or invalid, `correct` returns the first answer with the rejected natural versions dropped instead of failing.

This refines ADR-0005 and doesn't supersede it: there are still two targets, `natural` is still written on top of `grammatical`, and naturalness edits are still the `natural` diff minus anything overlapping a grammar edit.

### Probe

On 2026-11-14 I ran `openai-codex:gpt-6.1-terra` with the built-in prompts and `--better` over [0014-naturalness-probe-gold.txt](0014-naturalness-probe-gold.txt), all 10 lines as one block, three times. The review was switched off so every suggestion could be seen.

| Line | Bracketed | Suggestions in 3 runs |
| --- | --- | --- |
| 1 | I want to [take] a decision | `take → make` in runs 2 and 3; none in run 1 |
| 2 | She [said] a lie | `said → told`, 3 of 3 |
| 3 | He [made] a question | `made → asked`, 3 of 3 |
| 4 | I need to [win] some time | `win → buy`, 3 of 3 |
| 5 | I'm going to [lose] the bus | `lose → miss`, 3 of 3 |
| 6–10 | natural sentences | none in any run |

That is 14 hits out of 15 on the unnatural lines, no false suggestion on the natural ones, and no grammar edit on any of the ten grammatical lines. Every suggestion was a one-word `R:VERB` swap on the bracketed word, so the guard would have rejected none of them. The calibration rule was to raise the limits, never lower them, to the largest edit seen on lines 1–5: one word per side and one edit per sentence, so they stay at 3 and 2.

### Consequences

* Good, because a run without `--better` never pays for a natural version. Refit on 2026-11-14 with the 4-word and 73-word blocks, a correction call costs 25 + 2.0 output tokens per word without it and 27 + 3.3 with it; `joel estimate` puts 10 minutes of speech at about $0.045 without `--better` and $0.057 with it.
* Good, because a rewrite can't reach the report: the guard checks every change between `grammatical` and `natural`, a superset of the naturalness edits the report shows, whatever the model makes of the prompt.
* Good, because the guard is a pure function of edits, so its limits are unit tests rather than probe results.
* Bad, because real naturalness changes that need an added or removed word are lost: "I went to the bed early" → "I went to bed early" deletes "the", so the guard turns it back into `grammatical`.
* Bad, because idioms longer than three words, such as "on the other hand", can't come in as one replacement.
* Bad, because a block with one rejected sentence spends a whole retry call, re-sending every sentence in it.
* Bad, because the guard runs spaCy inside the output validator, on the event loop, so other calls in flight wait while it parses.
* Bad, because the `correct` prompt without the natural fragment is shorter than before. Its fixed part is about 350 tokens in OpenAI's count, probably under Anthropic's 512-token cache minimum, so on Anthropic models the correction step may stop being cached (ADR-0009). This wasn't measured.
* Neutral, because "take a decision" is standard British English, so line 1 of the gold is a fair miss. It stays in the set as the hard case.

### Confirmation

`test_word_swaps_are_natural` and the `natural_problem` tests in `tests/test_edits.py` pin each rule, with the boundaries at exactly 3 words and exactly 2 edits. `test_only_a_review_asks_the_model_for_a_natural_version` checks that the schema has no `natural` field without a review. `test_a_rewrite_dressed_up_as_naturalness_never_reaches_the_report` runs a deletion through `run_check` with the real annotator: it is sent back once with a message naming the sentence, and when it comes back again the sentence ends with `natural` equal to `grammatical` and no naturalness edit. `test_a_rewrite_next_to_a_grammar_fix_is_sent_back_too` does the same for a word added next to a grammar fix, and `test_a_failed_retry_after_a_rejected_natural_keeps_the_first_answer` covers a retry that comes back misnumbered. `test_naturalness_appears_only_when_asked_for` covers the CLI.

## Pros and Cons of the Options

### Keep ADR-0005

* Good, because nothing changes and naturalness shows on every run.
* Bad, because of the rewrites ADR-0011 saw, and the output tokens on every run.

### Always ask, show only with `--better`

* Good, because one schema and one prompt cover every run, and the JSON could always hold `natural`.
* Bad, because every run still pays for it, and a model told to write a natural version will sometimes write one that is a rewrite, which then sits in the JSON as a naturalness mistake.

### Ask only with `--better`, prompt only

* Good, because it is the least code.
* Bad, because ADR-0011 already showed a model ignoring "Do not rephrase". Precision would depend on the model of the day.

### Ask only with `--better`, separate schema, deterministic guard

* Good, because cost and precision both stop depending on the model's obedience.
* Bad, because of the lost insertions and deletions and the retry cost above.

### A second model call as judge

* Good, because a model can tell "in the end" deleted for style from a missing word a native speaker would add.
* Bad, because it doubles the calls of every `--better` block, and the judge is one more model whose mistakes look like the speaker's.
