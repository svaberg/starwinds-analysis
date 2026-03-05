"""THIS FILE contains helpers for locating repo sample_data fixtures.

It is convenience glue for examples/tests only.
It should not contain analysis logic.
"""

from __future__ import annotations

from pathlib import Path

def data_dir() -> Path:
    """
    Return the repository's `sample_data` directory.
    Used by: `test/test_sample_data_helpers.py`, `starwinds_analysis/data/samples.py`
    """
    return Path(__file__).resolve().parents[2] / "sample_data"

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
        return path

    available = sorted(p.name for p in data_dir().glob("*.plt"))
    raise FileNotFoundError(
        f"Sample file not found: {name}. Available .plt files: {', '.join(available)}"
    )
