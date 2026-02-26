from pathlib import Path

import pytest

from starwinds_analysis.data.samples import get_sample, sample_data_dir
from starwinds_analysis.smart_ds import SmartDs


def test_sample_data_dir_exists():
    path = sample_data_dir()
    assert path.name == "sample_data"
    assert path.exists()


def test_get_sample_returns_existing_tracked_fixture():
    path = get_sample("3d__var_1_n00060000.plt")
    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "3d__var_1_n00060000.plt"


def test_smartds_from_file_accepts_pathlike():
    sds = SmartDs.from_file(get_sample("3d__var_1_n00060000.plt"))
    assert sds.title


def test_get_sample_missing_lists_available_fixtures():
    with pytest.raises(FileNotFoundError) as exc:
        get_sample("__definitely_missing__.plt")

    msg = str(exc.value)
    assert "Available .plt files:" in msg
    assert "3d__var_1_n00060000.plt" in msg
