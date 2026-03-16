"""Helpers for locating sample_data fixtures in the repository."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

def data_dir() -> Path:
    """
    Return the repository's `sample_data` directory.
    Used by: `test/test_sample_data_helpers.py`, `batwind/data/samples.py`
    """
    path = Path(__file__).resolve().parents[2] / "sample_data"
    log.debug("data_dir resolved to %s", path)
    return path

def data_file(name: str, *, echo: bool = False) -> Path:
    """
    Return an absolute `Path` to a file in `sample_data`.
    Used by: `test/test_shell_magnetic_analysis.py`, `test/test_sample_data_helpers.py`,
      `examples/smartds_radial_histograms.ipynb`,
      `examples/smartds_inner_boundary_magnetic_zdi.ipynb`,
      `examples/smartds_quicklook_profiles.ipynb` (+2 more)
    """
    path = data_dir() / str(name)
    if path.exists():
        if echo:
            print(f"Using: {path}")
        log.info("data_file resolved %s", path)
        return path

    available = sorted(p.name for p in data_dir().glob("*.plt"))
    log.error("data_file failed for %s", name)
    raise FileNotFoundError(
        f"Sample file not found: {name}. Available .plt files: {', '.join(available)}"
    )
