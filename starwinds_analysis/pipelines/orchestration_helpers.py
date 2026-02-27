"""THIS FILE contains shared orchestration helpers for pipeline modules.

Keep pipeline modules focused on per-file workflow composition by moving generic
logging/serialization helpers here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


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


def quicklook_prefix_from_input_file(input_file) -> str:
    """
    Build a quicklook output prefix from an input filename.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    stem = Path(str(input_file)).name
    if stem.endswith(".plt"):
        stem = stem[:-4]
    else:
        stem = Path(stem).stem
    return slug_key(stem)


def resolve_quicklook_prefix(*, prefix: str | None, input_file=None) -> str:
    """
    Resolve the quicklook prefix from explicit value or input filename fallback.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if prefix is not None and str(prefix).strip():
        return str(prefix)
    if input_file is not None:
        return quicklook_prefix_from_input_file(input_file)
    return "quicklook2d"


def array_summary(values):
    """
    Compute compact summary statistics for a numeric array-like value.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    arr = np.array(values)
    finite = np.isfinite(arr)
    return {
        "shape": list(arr.shape),
        "n": int(arr.size),
        "n_finite": int(np.count_nonzero(finite)),
        "min": float(np.nanmin(arr)) if np.any(finite) else np.nan,
        "max": float(np.nanmax(arr)) if np.any(finite) else np.nan,
        "mean": float(np.nanmean(arr)) if np.any(finite) else np.nan,
    }


def summarize_result_object(value, *, skip_keys):
    """
    Recursively summarize nested result values into JSON-friendly metadata.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if isinstance(value, dict):
        out = {}
        for key, sub in value.items():
            if key in skip_keys:
                continue
            out[str(key)] = summarize_result_object(sub, skip_keys=skip_keys)
        return out
    try:
        arr = np.array(value)
    except Exception:
        return str(value)
    if arr.ndim == 0:
        try:
            return float(arr)
        except Exception:
            return str(value)
    if np.issubdtype(arr.dtype, np.number):
        return array_summary(np.array(arr))
    return str(value)


def flatten_result_arrays(value, arrays, *, prefix, skip_keys):
    """
    Collect numeric arrays from nested result objects into a flat output dict.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if isinstance(value, dict):
        for key, sub in value.items():
            if key in skip_keys:
                continue
            flatten_result_arrays(
                sub,
                arrays,
                prefix=f"{prefix}__{slug_key(key)}",
                skip_keys=skip_keys,
            )
        return
    try:
        arr = np.array(value)
    except Exception:
        return
    if arr.ndim == 0:
        return
    if not np.issubdtype(arr.dtype, np.number):
        return
    arrays[prefix] = np.array(arr)
