"""Helpers for selecting SAM-Audio models from the Hugging Face collection.

PATCHBAY uses this to populate the "Model" dropdown in the Run area and to
annotate entries as (lokal) vs (online).

Design goals
------------
- Works offline (falls back to a built-in list).
- Prefers lightweight network access (simple HTML fetch with a short timeout).
- Does not require authentication just to *list* models.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re
import urllib.request


COLLECTION_URL = "https://huggingface.co/collections/facebook/sam-audio"

# Conservative offline fallback (mirrors the collection as of Dec 2025).
DEFAULT_MODELS: List[str] = [
    "facebook/sam-audio-small",
    "facebook/sam-audio-base",
    "facebook/sam-audio-large",
    "facebook/sam-audio-small-tv",
    "facebook/sam-audio-base-tv",
    "facebook/sam-audio-large-tv",
]

# Items that appear in the collection but are not separation models for the backend.
_EXCLUDE_SUBSTR = (
    "sam-audio-bench",
    "sam-audio-musdb",
    "sam-audio-judge",
)

# Preferred order keys (others will be appended alphabetically).
_PREFERRED_ORDER = [
    "facebook/sam-audio-small",
    "facebook/sam-audio-base",
    "facebook/sam-audio-large",
    "facebook/sam-audio-small-tv",
    "facebook/sam-audio-base-tv",
    "facebook/sam-audio-large-tv",
]


def _http_get_text(url: str, *, timeout_s: float = 2.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PATCHBAY/1.0 (+https://huggingface.co/collections/facebook/sam-audio)"
        },
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def fetch_sam_audio_models_from_collection(*, timeout_s: float = 2.0) -> List[str]:
    """Fetch model repo_ids from the HF collection page (best-effort).

    Returns an empty list on failure.

    Implementation notes
    --------------------
    We deliberately extract *repo links* (href="/facebook/<repo>") from the
    collection page instead of doing a broad substring search. This avoids
    picking up unrelated assets (e.g. PNG thumbnails) that may contain similar
    substrings.
    """
    try:
        html = _http_get_text(COLLECTION_URL, timeout_s=timeout_s)
    except Exception:
        return []

    # Extract repo ids that appear as direct links to model repos:
    #   href="/facebook/sam-audio-small"
    # This excludes dataset links (usually /datasets/...), spaces, papers,
    # and also avoids image URLs.
    hrefs = re.findall(r'href="/(facebook/sam-audio[^"\s/?#]+)"', html)
    found = set(hrefs)
    found.discard("facebook/sam-audio")  # collection slug itself

    # Extra safety: drop obvious file-like endings (thumbnails, etc.)
    _bad_ext = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".ico")
    models = []
    for rid in found:
        if any(rid.lower().endswith(ext) for ext in _bad_ext):
            continue
        # Some collection entries may contain file-like links (e.g. *.png). Ignore those.
        repo_name = rid.split('/', 1)[1] if '/' in rid else rid
        if '.' in repo_name:
            continue
        if not re.match(r"^[^/]+/[^/]+$", rid):
            continue
        if any(x in rid for x in _EXCLUDE_SUBSTR):
            continue
        models.append(rid)

    # Deterministic ordering: preferred list first, remaining alphabetically.
    models_set = set(models)
    ordered = [m for m in _PREFERRED_ORDER if m in models_set]
    rest = sorted([m for m in models if m not in set(ordered)])
    return ordered + rest


def get_sam_audio_models(*, timeout_s: float = 2.0, preferred: Optional[str] = None) -> List[str]:
    """Return repo ids for the model dropdown.

    - Uses the live HF collection (short timeout) if possible.
    - Falls back to DEFAULT_MODELS.
    - Ensures `preferred` is included (useful if the user had a custom model).
    """
    models = fetch_sam_audio_models_from_collection(timeout_s=timeout_s)
    if not models:
        models = list(DEFAULT_MODELS)

    if preferred:
        preferred = str(preferred).strip()
        if preferred and preferred not in models:
            models = [preferred] + models
    # De-dup while preserving order
    seen = set()
    uniq = []
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        uniq.append(m)
    return uniq


def is_model_cached(repo_id: str) -> bool:
    """Best-effort check whether a model is present in the HF cache."""
    rid = str(repo_id).strip()
    if not rid:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        # config.json is a good proxy for "this repo has been downloaded at least once"
        p = try_to_load_from_cache(rid, "config.json")
        return p is not None
    except Exception:
        return False


def model_label(repo_id: str) -> str:
    return f"{repo_id} ({'lokal' if is_model_cached(repo_id) else 'online'})"


def build_model_dropdown(
    current_repo_id: str,
    *,
    timeout_s: float = 2.0,
) -> Tuple[List[str], Dict[str, str], str]:
    """Return (labels, label->repo_id, default_label)."""
    repo_ids = get_sam_audio_models(timeout_s=timeout_s, preferred=current_repo_id)
    labels = [model_label(r) for r in repo_ids]
    mapping = {lab: rid for lab, rid in zip(labels, repo_ids)}
    # default label for current selection
    cur = str(current_repo_id).strip()
    default_label = model_label(cur) if cur else (labels[0] if labels else "")
    if default_label not in mapping and labels:
        default_label = labels[0]
    return labels, mapping, default_label
