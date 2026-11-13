---
status: accepted
date: 2026-11-13
decision-makers: Marcus Xavier
---

# Naturalness as a second correction target, separate from grammar

## Context and Problem Statement

Besides grammar mistakes, I want to see sentences that are grammatical but that a native speaker would find odd ("I make a party"). Mixing both in one list blurs "this is wrong" with "this sounds off" and lowers the report's apparent precision.

## Decision Drivers

* Grammar mistakes and unnatural usage show up as distinct categories.
* Reuse ERRANT for position and type in both categories (ADR-0003).
* One LLM call per sentence, not two.

## Considered Options

* One call returns two targets: `grammatical` (minimal edit, grammar only) and `natural` (minimal edit on top of `grammatical`).
* A single corrected target; ERRANT edits get classified afterwards by type.
* Naturalness out of stage 1.

## Decision Outcome

Chosen option: two targets in the same call, because the split comes from what we ask the LLM, not from a heuristic over ERRANT types.

### Consequences

* Good, because ERRANT between the clean sentence and `grammatical` gives the mistakes, and ERRANT between the clean sentence and `natural`, minus edits overlapping a mistake, gives naturalness. Both carry positions in the original words.
* Good, because the report can hide or show naturalness without touching grammar.
* Bad, because the line between grammar and naturalness is the LLM's call and varies between runs.
* Bad, because the ERRANT type on naturalness edits (`R:VERB`, `R:NOUN`) says little; the category is the useful label.

### Confirmation

Bench sentences with no grammar mistake but odd usage show up only as naturalness. Correct, natural sentences produce no edits at all.

## Pros and Cons of the Options

### Two targets in one call

* Good, because it is one call and the diff stays deterministic.
* Bad, because the prompt gets longer, with two minimal-edit instructions.

### One target, classify by type

* Good, because the prompt is simpler.
* Bad, because ERRANT types can't tell a mistake from an odd choice: "make → throw" and "go → went" are both verb substitutions.

### Out of stage 1

* Good, because it is the least work.
* Bad, because it is one of the things I most want to see, and adding it now doesn't make the architecture expensive.
