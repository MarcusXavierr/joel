---
status: accepted
date: 2026-11-14
decision-makers: Marcus Xavier
---

# Download spaCy's en_core_web_sm on first use

## Context and Problem Statement

ERRANT parses every sentence with spaCy's `en_core_web_sm` (ADR-0006). The model isn't on PyPI. Until now `pyproject.toml` listed `en-core-web-sm` as a runtime dependency and pointed it at the GitHub release wheel through `[tool.uv.sources]`. Only uv reads that table, so the built wheel declared a bare `Requires-Dist: en-core-web-sm`, and installing `joel` from PyPI failed to resolve it. Joel has to install with `uv tool install joel` and still find the model.

## Decision Drivers

* `uv tool install joel` works with nothing but PyPI.
* The model stays pinned to one version and one file, as Parakeet is, because ERRANT's edits depend on the parser.
* The test suite stays offline.

## Considered Options

* Download the pinned wheel from spaCy's GitHub release on first use, checked by sha256, into Joel's cache.
* Keep a direct URL dependency (`en-core-web-sm @ https://...`) in `[project] dependencies`.
* Load the model from Hugging Face (`spacy/en_core_web_sm`).
* Vendor the model inside Joel's wheel.

## Decision Outcome

Chosen option: download on first use, because it is the only option that installs from PyPI and runs on spaCy 3.8. `joel.spacy_model.resolve_spacy_model` looks in three places, in order: the installed `en_core_web_sm` package at 3.8.0, a copy Joel extracted earlier into `$XDG_CACHE_HOME/joel` (`~/.cache/joel` when unset), and finally a download of `en_core_web_sm-3.8.0-py3-none-any.whl` from spaCy's GitHub release. The download is checked against a pinned sha256, extracted into a temporary directory, and renamed into place, so the cache never holds half a model. `load_annotator` hands the resolved directory to `spacy.load` and passes the pipeline to `errant.load`.

The dev group installs the model as a package from the same URL, so the tests resolve it in step one and never download it.

### Consequences

* Good, because the published wheel has no dependency PyPI can't resolve.
* Good, because the sha256 pin means a changed or tampered release asset is refused instead of loaded.
* Good, because the resolver has the same shape as Parakeet's (`_parakeet_model.resolve_model`): installed or cached first, download last, one error that says what is missing and where it should come from.
* Bad, because the first `joel check` on a new machine has to reach GitHub. Offline, it stops before transcribing with `Could not get spaCy's en_core_web_sm model`.
* Bad, because the pin breaks if spaCy ever deletes the 3.8.0 release asset. Moving to another model version means a new URL and hash, and a check that ERRANT's edits don't change.
* Bad, because Joel now keeps a second cache: the spaCy model in `~/.cache/joel`, next to Parakeet in the Hugging Face cache.
* Neutral, because a user who installs `en_core_web_sm` 3.8.0 into Joel's environment by hand skips the download.

### Confirmation

`tests/test_spacy_model.py` drives the real download, hash check, and extraction through an `httpx.MockTransport`: one download that later calls reuse, a hash mismatch that leaves the cache empty, a 404 and a refused connection that explain what is missing, and the installed package used without any request. `test_a_spacy_model_it_cannot_download_is_an_error` checks that `joel check` reports the failure before it transcribes anything.

## Pros and Cons of the Options

### Download on first use

* Good, because PyPI accepts the wheel and the model stays pinned.
* Bad, because of the first-run network access and the extra cache described above.

### Direct URL dependency

* Good, because pip and uv would install the model with Joel, with no code in Joel.
* Bad, because PyPI rejects uploads whose metadata has direct URL dependencies, so Joel couldn't be published at all.

### Hugging Face `spacy/en_core_web_sm`

* Good, because Joel already depends on `huggingface_hub` and its cache for Parakeet.
* Bad, because the repo only has 3.7.1, which declares `spacy_version >=3.7.2,<3.8.0` and doesn't load on spaCy 3.8.

### Vendor the model in Joel's wheel

* Good, because it works offline from the first run.
* Bad, because every release would carry 12.8 MB of another project's artifact, and Joel would be redistributing it rather than pointing at it.

## More Information

The release asset is `https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl` (12,806,118 bytes). The model directory inside it is `en_core_web_sm/en_core_web_sm-3.8.0/`.
