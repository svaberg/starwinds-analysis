from __future__ import annotations

from pathlib import Path


def sample_data_dir() -> Path:
    """
    Return the repository's `sample_data` directory.

    This helper is intended for local examples/tests in a source checkout.
    """
    return Path(__file__).resolve().parents[2] / "sample_data"


def get_sample(name: str) -> str:
    """
    Return an absolute path string to a file in `sample_data`.

    Parameters
    ----------
    name
        Exact filename under `sample_data`, e.g. `z=0_var_3_n00060000.plt`.
    """
    path = sample_data_dir() / str(name)
    if path.exists():
        return str(path)

    available = sorted(p.name for p in sample_data_dir().glob("*.plt"))
    raise FileNotFoundError(
        f"Sample file not found: {name}. Available .plt files: {', '.join(available)}"
    )


__all__ = ["get_sample", "sample_data_dir"]
