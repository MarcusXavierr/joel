# Architecture Decision Records

Each file records one architecturally significant decision, in [MADR](https://adr.github.io/madr/) format.

An accepted ADR is immutable. When a decision changes, do not edit the old record: write a new ADR and set the old one's `status` to `superseded by ADR-NNNN`. The value of the log is that it preserves why something was decided at the time, including reasoning that later turned out to be wrong.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-own-transcriber-interface-with-one-adapter-per-sdk.md) | Own `Transcriber` interface, one adapter per official SDK; LiteLLM out. Its `gpt-transcribe` exclusion is replaced by 0007 | accepted |
| [0002](0002-parakeet-ultra-as-default-local-transcriber.md) | Parakeet Ultra via `transcribe-cpp` as the default local ASR, pending the bench | proposed |
| [0003](0003-llm-corrects-errant-extracts-edits.md) | The LLM writes the corrected sentence; ERRANT extracts typed edits | accepted |
| [0004](0004-disfluency-cleanup-via-llm-returned-indices.md) | Disfluency cleanup with the LLM returning only indices to drop | accepted |
| [0005](0005-naturalness-as-second-correction-target.md) | Naturalness as a second target in the same call, separate from grammar | accepted |
| [0006](0006-spacy-tokenization-with-offset-mapping.md) | spaCy tokenization in ERRANT, mapped to ASR words by character offset | accepted |
| [0007](0007-word-timestamps-optional-in-transcriber-contract.md) | Word timestamps optional in the `Word` contract, so `gpt-transcribe` fits behind the interface | accepted |
| [0008](0008-llm-steps-run-over-blocks-of-sentences.md) | Cleanup and correction run over numbered blocks of up to N sentences per call (`--block-sentences`; default 6, moved to 7 by 0010) | accepted |
| [0009](0009-cache-the-instructions-and-warm-up-each-step.md) | Cache each step's tool schema and instructions, with worked examples past the minimum, and run each step's first block alone to write the cache | accepted |
| [0010](0010-default-model-chosen-by-precision-probe.md) | `openai:gpt-6-luna` as the default model and 7 as the default block size, chosen by a precision probe over 7 models | superseded by 0011 |
| [0011](0011-default-model-gpt-6-1-terra-on-the-chatgpt-plan.md) | `openai-codex:gpt-6.1-terra` as the default model, run on the ChatGPT plan through the Codex CLI's login | accepted |
| [0012](0012-download-the-spacy-model-on-first-use.md) | Download the pinned `en_core_web_sm` 3.8.0 from spaCy's GitHub release on first use, checked by sha256, so Joel installs from PyPI | accepted |
| [0013](0013-prompts-as-files-users-can-override.md) | Prompts live in versioned `.md` files in the package; `--prompt NAME=FILE` replaces one per run, and each run records their hashes | accepted |
| [0014](0014-naturalness-only-on-request-held-to-word-swaps.md) | The natural version is asked for only with `--better`, through its own schema, and a deterministic guard holds it to word swaps; refines 0005 | accepted |

## Adding one

Copy a template from the `writing-madrs` skill, number it next in sequence, and add a row above in the same commit.

Two rules do most of the work:

* Record the options you rejected and why. A decision without alternatives is an assertion.
* Record the bad consequences alongside the good. An ADR with only upsides is marketing.
