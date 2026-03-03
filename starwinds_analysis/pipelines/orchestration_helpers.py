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


def prepare_smartds(smart_ds, *, body_radius_m: float) -> None:
    """
    Best-effort SmartDs graph setup for pipeline entrypoints.
    Used by: `starwinds_analysis/pipelines/slice.py`
    """
    if hasattr(smart_ds, "add_batsrus_graph"):
        try:
            smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
        except Exception:
            pass
    if hasattr(smart_ds, "add_spherical_graph"):
        try:
            smart_ds.add_spherical_graph(vectors=("B", "U"))
            return
        except Exception:
            pass
    if hasattr(smart_ds, "add_spherical_fields"):
        try:
            smart_ds.add_spherical_fields(vectors=("B", "U"))
        except Exception:
            pass


def is_2d_input(smart_ds) -> bool:
    """
    Detect whether a dataset behaves like a 2D slice input.
    Used by: `starwinds_analysis/pipelines/slice.py`
    """
    corners = getattr(smart_ds, "corners", None)
    if getattr(corners, "ndim", 0) == 2:
        if corners.shape[1] == 4:
            return True
        if corners.shape[1] >= 8:
            return False

    constant_axes = 0
    for name in ("X [R]", "Y [R]", "Z [R]"):
        try:
            values = np.ravel(smart_ds(name))
        except Exception:
            continue
        finite = np.isfinite(values)
        if not np.any(finite):
            constant_axes += 1
            continue
        finite_values = values[finite]
        vmin = np.min(finite_values)
        vmax = np.max(finite_values)
        scale = max(abs(vmin), abs(vmax), 1.0)
        if abs(vmax - vmin) <= (1.0e-12 + 1.0e-10 * scale):
            constant_axes += 1
    return constant_axes >= 1 or (constant_axes == 0 and not hasattr(smart_ds, "corners"))
