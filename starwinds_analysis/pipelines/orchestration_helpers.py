"""THIS FILE contains shared lightweight helpers for pipeline modules."""

from __future__ import annotations

from pathlib import Path


def log_pipeline_event(logger, message: str, **fields) -> None:
    """
    Emit one normalized pipeline progress message on the provided logger.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    text = message
    if fields:
        suffix = ", ".join(f"{key}={value}" for key, value in sorted(fields.items()))
        text = f"{message} | {suffix}"
    logger.debug(text)


def slug_key(text: str) -> str:
    """
    Create a filesystem-safe-ish slug from arbitrary text.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
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
    return slug.strip("_") or "item"


def output_prefix_from_input_file(input_file) -> str:
    """
    Build a quicklook output prefix from an input filename.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    path = Path(str(input_file))
    stem = path.name
    if path.suffix.lower() in {".plt", ".dat"}:
        stem = path.stem
    else:
        stem = Path(stem).stem
    return slug_key(stem)


def resolve_output_prefix(*, prefix: str | None, input_file=None) -> str:
    """
    Resolve the quicklook prefix from explicit value or input filename fallback.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if prefix is not None and str(prefix).strip():
        return str(prefix)
    if input_file is not None:
        return output_prefix_from_input_file(input_file)
    return "output"
