"""Helpers for locating sample_data fixtures in the repository."""

from __future__ import annotations

import logging
from pathlib import Path
import tarfile

log = logging.getLogger(__name__)

_G2211_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_G2211_SHA256 = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"


def data_dir() -> Path:
    """
    Return the repository's `sample_data` directory.
    Used by: `test/test_sample_data_helpers.py`, `batwind/data/samples.py`
    """
    path = Path(__file__).resolve().parents[2] / "sample_data"
    log.debug("data_dir resolved to %s", path)
    return path


def _unique_match(paths: list[Path], *, name: str) -> Path:
    """Return one match for `name` or raise."""
    if not paths:
        raise FileNotFoundError(name)
    if len(paths) > 1:
        raise FileNotFoundError(f"Expected unique match for {name}, found {len(paths)} entries: {paths}")
    return paths[0]


def _find_in_sample_data(root: Path, name: str) -> Path:
    """Find one file by basename under `sample_data`."""
    return _unique_match(sorted(root.rglob(name)), name=name)


def _fetch_from_g2211_archive(name: str) -> Path:
    """Fetch one named file from the Zenodo G2211 archive."""
    import pooch

    archive_path = Path(
        pooch.retrieve(
            url=_G2211_URL,
            known_hash=_G2211_SHA256,
            progressbar=False,
        )
    )
    with tarfile.open(archive_path, "r:gz") as tar:
        member_names = sorted(
            m.name for m in tar.getmembers() if m.isfile() and Path(m.name).name == name
        )
    member = _unique_match([Path(m) for m in member_names], name=name).as_posix()
    extracted = pooch.retrieve(
        url=_G2211_URL,
        known_hash=_G2211_SHA256,
        progressbar=False,
        processor=pooch.Untar(members=[member]),
    )
    if isinstance(extracted, (list, tuple)):
        extracted = extracted[0]
    return Path(extracted)


def data_file(name: str, *, echo: bool = False) -> Path:
    """
    Return an absolute `Path` to a file in `sample_data`.
    Used by: `test/test_shell_magnetic_analysis.py`, `test/test_sample_data_helpers.py`,
      `examples/smartds_radial_histograms.ipynb`,
      `examples/smartds_inner_boundary_magnetic_zdi.ipynb`,
      `examples/smartds_quicklook_profiles.ipynb` (+2 more)

    Resolution order:
    1. Find uniquely in this repo's `sample_data` tree.
    2. Fallback to the Zenodo G2211 archive via `pooch`.
    """
    sample_root = data_dir()
    try:
        path = _find_in_sample_data(sample_root, str(name))
    except FileNotFoundError:
        try:
            path = _fetch_from_g2211_archive(str(name))
        except FileNotFoundError:
            available = sorted(p.name for p in sample_root.glob("*.plt"))
            log.error("data_file failed for %s", name)
            raise FileNotFoundError(
                f"Sample file not found: {name}. Available .plt files: {', '.join(available)}"
            ) from None
    if echo:
        print(f"Using: {path}")
    log.debug("data_file resolved %s", path)
    return path
