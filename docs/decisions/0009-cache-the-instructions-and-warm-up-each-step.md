---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# Cache the instructions, and warm the cache up once per step

## Context and Problem Statement

After ADR-0008, each LLM step still sends the same prefix on every block: the tool schema and the instructions. Anthropic can serve a repeated prefix from its prompt cache at a fraction of the input price, but only from a minimum size: 512 tokens on the Claude Opus 5 family. Joel's prefixes were under that, at about 440 tokens for cleanup and 500 for correction, so nothing was being cached.

A cache entry also exists only once the request that writes it has been processed. Calls that start together all miss, and each one pays the write price, 1.25 times the input price on the 5-minute cache. Joel starts up to 8 calls at once, so turning caching on without changing the order would make short recordings cost more, not less.

## Decision Drivers

* Pay full price for the prefix once per step, not once per block.
* Keep a short recording from costing more than it does without caching.
* Make the prompt longer only with text that helps the model.
* Work unchanged with providers that cache differently or not at all.

## Considered Options

* No caching.
* Cache the prefix and start every block at once.
* Cache the prefix and run each step's first block alone, so it writes the cache before the others read it.
* The same, with the 1-hour cache.
* Reach the minimum with filler text instead of examples.

## Decision Outcome

Chosen option: cache the prefix and run each step's first block alone. `ask()` sends `ModelSettings(cache=CacheConfig(messages=False))` on every request, which caches the tool schema and the instructions and nothing else, since no conversation is ever continued. `run_check` awaits each step's first block by itself and then starts the rest. Each prompt now ends with worked examples, which take it past the minimum: four cleanup inputs with their answers, and one numbered correction block with its answer.

Measured on 2026-11-13 by sending the same call twice in a row:

| Model | Cleanup prefix | Correction prefix | Second call |
| --- | --- | --- | --- |
| `anthropic:claude-opus-5-5` | 776 tokens written | 970 tokens written | reads all of it |
| `anthropic:claude-haiku-5-5` | 776 tokens written | 970 tokens written | reads all of it |
| `anthropic:claude-sonnet-5` | nothing written | nothing written | no cache |

On Opus 5.5 a cache read costs $0.20 per million tokens against $4 for uncached input, and a write costs $5. So a step pays the write once and then reads the prefix at a twentieth of the price. The runs record the cached tokens: `CheckResult` has `cache_read_tokens` and `cache_write_tokens`, `price()` charges them at the cache's rates, and the report's cost line says how many input tokens came from the cache.

### Consequences

* Good, because each block after the first pays about a twentieth of the prefix price on Opus and Haiku.
* Good, because the examples show the model the answer format and the cases the rules describe, such as keeping "am" in "I am agree".
* Good, because nothing changes for a model that doesn't support the setting: pydantic-ai drops it for those model classes, `FunctionModel` included.
* Bad, because each step runs one call before the others can start, which adds a few seconds to every run.
* Bad, because the examples add about 300 tokens to every call at full price wherever nothing is cached. That includes Sonnet 5, whose minimum is above 990 tokens, and OpenAI, which caches only from 1,024 tokens.
* Bad, because the warm-up only pays off from the second block of a step on. A recording that fits in one block per step pays the write premium and reads nothing.
* Neutral, because `joel estimate` still assumes no caching, and its last line says so.

### Confirmation

`tests/test_check.py` checks that every request asks for the prefix to be cached, that each step's first call finishes before any other call of that step starts, and that cached tokens are counted and lower the cost. The table above came from a throwaway script that sent the same cleanup and correction call twice per model, run again against any new default model.

## Pros and Cons of the Options

### No caching

* Good, because it is the simplest, and no call waits on another.
* Bad, because every block repeats the prefix at full price.

### Cache and start every block at once

* Good, because no call waits.
* Bad, because the calls that start together all miss and each pays 1.25 times the price for the write. Up to 8 blocks per step, that costs more than not caching at all.

### Cache and warm up with the first block

* Good, because one write per step serves every later block.
* Bad, because of the serial first call.

### The 1-hour cache

* Good, because a second run within the hour would read a warm cache.
* Bad, because writes cost twice the input price instead of 1.25 times, and runs rarely repeat within the hour.

### Filler text instead of examples

* Good, because it reaches the minimum with no risk of steering the model.
* Bad, because it pays for tokens that do nothing, and the examples cost the same and help.

## More Information

The Anthropic minimum and the prices come from Anthropic's prompt-caching documentation and the `genai-prices` table bundled with Joel. pydantic-ai's `CacheConfig(messages=False)` is described in `pydantic_ai/settings.py`.
