"""Find the pinned Parakeet Ultra model file, downloading it only as a last resort.

Resolution goes through the standard Hugging Face cache, in this order:

1. the pinned revision, already cached;
2. the same file (by sha256) under any other cached revision, such as one
   another app downloaded;
3. a download of the pinned revision.
"""

from pathlib import Path

from huggingface_hub import (
    constants,
    hf_hub_download,
    scan_cache_dir,
    try_to_load_from_cache,
)
from huggingface_hub.errors import (
    CacheNotFound,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

from joel.transcribers import TranscriberUnavailableError

PARAKEET_REPO = "handy-computer/parakeet-ultra-gguf"
PARAKEET_FILE = "parakeet-ultra-Q8_0.gguf"
# Bench results hold only for the model version they measured, so the model is
# pinned by commit and by file hash, never followed through "main".
PARAKEET_REVISION = "a1539be1b849ea7463b88343e5277b373d0c4ea9"
PARAKEET_SHA256 = "007a59761e9258779f189df396b50d12b83bb5b79e66df4b955c230d2f2a0a59"
PARAKEET_SIZE_BYTES = 740_363_168


class ModelUnavailableError(TranscriberUnavailableError):
    """The pinned model is neither cached nor downloadable."""


def resolve_model(cache_dir: Path | None = None) -> Path:
    """Return a local path to the pinned model file.

    ``cache_dir`` defaults to the standard Hugging Face hub cache. Raises
    ModelUnavailableError when the file isn't cached and can't be downloaded.
    """
    cache = cache_dir if cache_dir is not None else Path(constants.HF_HUB_CACHE)
    cached = try_to_load_from_cache(
        PARAKEET_REPO, PARAKEET_FILE, cache_dir=cache, revision=PARAKEET_REVISION
    )
    if isinstance(cached, str):
        return Path(cached)
    same_file = _find_by_hash(cache)
    if same_file is not None:
        return same_file
    try:
        return Path(
            hf_hub_download(
                PARAKEET_REPO,
                PARAKEET_FILE,
                revision=PARAKEET_REVISION,
                cache_dir=cache,
            )
        )
    except (LocalEntryNotFoundError, HfHubHTTPError) as exc:
        raise ModelUnavailableError(
            f"Could not get the Parakeet Ultra model. Joel needs {PARAKEET_FILE} "
            f"({PARAKEET_SIZE_BYTES / 1_000_000:.0f} MB) from the Hugging Face "
            f"repo {PARAKEET_REPO} at revision {PARAKEET_REVISION}, and it is not "
            f"in the cache at {cache}. Connect to the internet once so Joel can "
            f"download it. ({exc})"
        ) from exc


def _find_by_hash(cache: Path) -> Path | None:
    """Any cached revision's copy of the pinned file, matched by its sha256."""
    try:
        info = scan_cache_dir(cache)
    except CacheNotFound:
        return None
    for repo in info.repos:
        if repo.repo_id != PARAKEET_REPO or repo.repo_type != "model":
            continue
        for revision in repo.revisions:
            for file in revision.files:
                # The blob is named after its sha256, whatever the file is called.
                if file.blob_path.name == PARAKEET_SHA256:
                    return file.file_path
    return None
