"""Shared lightweight helpers for pipeline modules."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def slug_key(text: str) -> str:
    """
    Create a filesystem-safe-ish slug from arbitrary text.
    """
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    out = slug.strip("_") or "item"
    log.debug("slug_key '%s' -> '%s'", text, out)
    return out


def output_prefix_from_input_file(input_file) -> str:
    """
    Build a quicklook output prefix from an input filename.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/shell.py`, `starwinds_analysis/pipelines/volume.py`
    """
    path = Path(str(input_file))
    stem = path.name
    if path.suffix.lower() in {".plt", ".dat"}:
        stem = path.stem
    else:
        stem = Path(stem).stem
    out = slug_key(stem)
    log.debug("output_prefix_from_input_file '%s' -> '%s'", input_file, out)
    return out
