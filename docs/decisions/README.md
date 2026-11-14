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
| [0008](0008-llm-steps-run-over-blocks-of-sentences.md) | Cleanup and correction run over numbered blocks of up to N sentences per call (`--block-sentences`, default 6) | accepted |

## Adding one

Copy a template from the `writing-madrs` skill, number it next in sequence, and add a row above in the same commit.

Two rules do most of the work:

* Record the options you rejected and why. A decision without alternatives is an assertion.
* Record the bad consequences alongside the good. An ADR with only upsides is marketing.
