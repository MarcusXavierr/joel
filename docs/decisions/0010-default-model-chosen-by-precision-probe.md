---
status: accepted
date: 2026-11-14
decision-makers: Marcus Xavier
---

# gpt-6-luna as the default model and 7 as the default block size, chosen by a precision probe

## Context and Problem Statement

`anthropic:claude-opus-5-5` became the default model because it was available on Marcus's account, not because anything compared it with other models. After ADR-0008 and ADR-0009 a minute of speech cost a few cents on Opus, and most of that was output: on Marcus's 1-minute recording, 1,106 output tokens against 2,551 input. A cheaper model, or Opus without thinking, might find the same mistakes for less, and a cheaper model might lose precision on bigger blocks (ADR-0008). Joel ranks precision above recall, so the comparison had to measure false mistakes, not only cost.

## Decision Drivers

* Precision first: a false mistake teaches the user to ignore the report.
* Recall second: a missed mistake costs less, but a model that finds nothing is useless.
* Cost per minute of speech.
* Whether a cheaper model holds its precision on bigger blocks.

## Considered Options

* `anthropic:claude-opus-5-5`, with its default thinking.
* `anthropic:claude-opus-5-5` with thinking turned off.
* `anthropic:claude-sonnet-5`.
* `anthropic:claude-haiku-5-5`.
* `openai:gpt-6.1-terra`.
* `openai:gpt-6.1-luna`.
* `openai:gpt-6-luna`.

## Decision Outcome

Chosen option: `openai:gpt-6-luna`, with the default block size moved from ADR-0008's 6 to 7. Every model that ran reached precision 1.000 on the gold, with no false mistake on the correct sentences, so precision didn't separate them. `gpt-6-luna` was the cheapest of them and also kept recall at 1.000 with blocks of 5 and of 10: under $0.001 per minute of speech, against about $0.03 for Opus. The block size of 7 was not measured. It is Marcus's choice between the two sizes that both kept `gpt-6-luna` at full recall.

`gpt-6.1-luna` answered every run with a 404: the model doesn't exist or isn't enabled on this account. No substitute was added.

| Variant | Block | Precision | Recall | False mistakes on correct lines | Cost per minute of speech |
| --- | --- | --- | --- | --- | --- |
| opus-5-5 | 5 | 1.000 | 1.000 | 0 | $0.029 |
| opus-5-5 | 10 | 1.000 | 1.000 | 0 | $0.027 |
| opus-5-5 no thinking | 5 | 1.000 | 1.000 | 0 | $0.024 |
| opus-5-5 no thinking | 10 | 1.000 | 0.950 | 0 | $0.023 |
| sonnet-5 | 5 | 1.000 | 1.000 | 0 | $0.027 |
| sonnet-5 | 10 | 1.000 | 0.950 | 0 | $0.023 |
| haiku-5-5 | 5 | 1.000 | 0.900 | 0 | $0.002 |
| haiku-5-5 | 10 | 1.000 | 0.950 | 0 | $0.001 |
| gpt-6.1-terra | 5 | 1.000 | 0.983 | 0 | $0.008 |
| gpt-6.1-terra | 10 | 1.000 | 1.000 | 0 | $0.006 |
| gpt-6-luna | 5 | 1.000 | 1.000 | 0 | under $0.001 |
| gpt-6-luna | 10 | 1.000 | 1.000 | 0 | under $0.001 |

Each row is the mean of 3 runs. The minimum precision of every run was also 1.000, and both judges called all 22 distinct edits real mistakes, so the judged precision is 1.000 everywhere too.

### Consequences

* Good, because a minute of speech costs under a tenth of a cent, so cost stops being a reason not to run Joel.
* Good, because `joel estimate` is refit against `gpt-6-luna` on blocks of 1 and 7 sentences.
* Bad, because the gold was too easy to rank precision: its 20 mistakes are textbook cases, and no model made a false one. The choice rests on cost and recall.
* Bad, because real speech went worse than the gold. On Marcus's 1-minute recording `gpt-6-luna` kept the false start "help users" that Opus dropped, so it corrected the first "help" and left the second, and it missed "ready to deploy in etc." Opus found both, and it also flagged "fixing → fixes" twice, which is debatable.
* Bad, because ADR-0009's caching does nothing for `gpt-6-luna`: its prompts are 500 tokens for cleanup and 484 for correction in OpenAI's count, under the 1,024-token minimum. The examples and the serial first call stay, and they pay off again on an Anthropic model.
* Bad, because the default now needs `OPENAI_API_KEY` instead of `ANTHROPIC_API_KEY`.
* Neutral, because `--model anthropic:claude-opus-5-5` brings Opus back for a recording where precision on messy speech matters more than the cost.

### Confirmation

A harder gold, with real recordings and ambiguous cases, should be run before this choice is trusted for long recordings. The probe script reads the same gold format, so a new file can be scored the same way.

## Pros and Cons of the Options

### opus-5-5

* Good, because it found everything on the gold and handled the false start in the real recording.
* Bad, because it costs 27 to 29 times as much as `gpt-6-luna` per run, and about 70% of that is output at $20 per million tokens.

### opus-5-5 with thinking off

* Good, because it costs 15 to 17% less than Opus with thinking and runs faster.
* Bad, because it missed "love the nature" in every run with blocks of 10.

### sonnet-5

* Good, because it matched Opus with blocks of 5.
* Bad, because it costs about the same as Opus here: its prompts are under its cache minimum, so it pays the full prefix on every call.

### haiku-5-5

* Good, because it costs about $0.002 a minute and reads the prompt cache.
* Bad, because it had the lowest recall: it missed "On Saturday we go" with blocks of 5, and "love the nature" in every run.

### gpt-6.1-terra

* Good, because it kept recall at 0.983 or above for under a cent a minute.
* Bad, because it costs 6 to 8 times what `gpt-6-luna` does, for no better result on the gold.

### gpt-6.1-luna

* Bad, because it wasn't available.

### gpt-6-luna

* Good, because it is the cheapest, and it kept precision and recall at 1.000 with both block sizes.
* Bad, because of the false start it kept on real speech, as described above.

## More Information

The gold is [`0010-model-probe-gold.txt`](0010-model-probe-gold.txt): one continuous spoken monologue of 30 sentences and 301 words, about two minutes of speech. 20 sentences hold one grammar mistake each that Brazilian Portuguese speakers commonly make, marked with brackets, and 10 are correct conversational English with informal spoken forms ("gonna", "like", a sentence starting with "And") as bait for false mistakes.

A throwaway script fed the gold to `run_check` through a fake transcriber that returns its words, so every model went through the real cleanup, correction, ERRANT diff and filter, with prompt caching on. Each model ran 3 times with blocks of 5 sentences and 3 times with blocks of 10, one model after another, on 2026-11-14. The 42 runs cost $1.034 in total.

Scoring counts grammar edits only:

* precision: edits that touch a bracketed word, over all edits;
* recall: bracketed spans touched by at least one edit, over all spans;
* false mistakes on correct lines: edits on the 10 sentences with no bracket;
* judged precision: edits that both judges, Claude Fable 5.2 and GPT-6.1 Sol, called real grammar mistakes, over all edits. The judges saw each distinct edit with its sentence and the monologue without brackets, and never which model made it.

Cost is what `genai-prices` charges for each run's tokens, with cached tokens at the cache's rates. The cost per minute divides it by the gold's 2.01 minutes at 150 words per minute.

Mean tokens per run:

| Variant | Block | Input | Read from cache | Output | Cost per run |
| --- | --- | --- | --- | --- | --- |
| opus-5-5 | 5 | 13668 | 9894 | 2037 | $0.058 |
| opus-5-5 | 10 | 8233 | 5238 | 2055 | $0.054 |
| opus-5-5 no thinking | 5 | 13668 | 10476 | 1667 | $0.048 |
| opus-5-5 no thinking | 10 | 8233 | 5238 | 1645 | $0.046 |
| sonnet-5 | 5 | 13644 | 0 | 2751 | $0.055 |
| sonnet-5 | 10 | 8221 | 0 | 2952 | $0.046 |
| haiku-5-5 | 5 | 13667 | 9894 | 5175 | $0.003 |
| haiku-5-5 | 10 | 8231 | 5238 | 4654 | $0.003 |
| gpt-6.1-terra | 5 | 8153 | 0 | 1556 | $0.016 |
| gpt-6.1-terra | 10 | 5214 | 0 | 1386 | $0.012 |
| gpt-6-luna | 5 | 8152 | 0 | 2299 | $0.002 |
| gpt-6-luna | 10 | 5213 | 0 | 2017 | $0.002 |

OpenAI counts the same prompt in fewer tokens than Anthropic, so its input column is lower for the same text.

The misses were all on two lines. "I really love the nature" (line 6) sits between grammar and naturalness, and five variants skipped it in some or all runs. "On Saturday we go to the beach" (line 3) reads as a habitual present without the past tense around it; only Haiku with blocks of 5 skipped it.
