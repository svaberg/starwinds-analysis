"""THIS FILE contains helpers for locating repo sample_data fixtures.

It is convenience glue for examples/tests only.
It should not contain analysis logic.
"""

from __future__ import annotations

from pathlib import Path

# Return the repository's `sample_data` directory.
# Used in: `test/test_sample_data_helpers.py`, `starwinds_analysis/data/samples.py`
def sample_data_dir() -> Path:
    """
    Return the repository's `sample_data` directory.

    This helper is intended for local examples/tests in a source checkout.
    """
    return Path(__file__).resolve().parents[2] / "sample_data"

# Return an absolute `Path` to a file in `sample_data`.
# Used in: `test/test_shell_magnetic_analysis.py`, `test/test_sample_data_helpers.py`,
#   `examples/smartds_radial_histograms.ipynb`,
#   `examples/smartds_inner_boundary_magnetic_zdi.ipynb`,
#   `examples/smartds_quicklook_profiles.ipynb` (+2 more)
def get_sample(name: str, *, echo: bool = False) -> Path:
    """
    Return an absolute `Path` to a file in `sample_data`.

    Parameters
    ----------
    name
        Exact filename under `sample_data`, e.g. `z=0_var_3_n00060000.plt`.
    echo
        If True, print the resolved path (useful in interactive examples/notebooks).
    """
    path = sample_data_dir() / str(name)
    if path.exists():
        if echo:
            print(f"Using: {path}")
        return path

    available = sorted(p.name for p in sample_data_dir().glob("*.plt"))
    raise FileNotFoundError(
        f"Sample file not found: {name}. Available .plt files: {', '.join(available)}"
    )

