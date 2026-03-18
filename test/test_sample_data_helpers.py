from pathlib import Path

import pytest

from batwind.data.samples import data_dir
from batwind.data.samples import data_file
from batwind.smart_ds import SmartDs


def test_data_dir_exists():
    path = data_dir()
    assert path.name == "sample_data"
    assert path.exists()


def test_data_file_returns_existing_tracked_fixture():
    path = data_file("3d__var_4_n00000000.plt")
    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "3d__var_4_n00000000.plt"


def test_smartds_from_file_accepts_pathlike():
    sds = SmartDs.from_file(data_file("3d__var_4_n00000000.plt"))
    assert sds.raw.title


def test_smartds_from_file_merges_nearby_stellar_aux():
    sds = SmartDs.from_file(data_file("3d__var_4_n00000000.plt"))
    assert "Star_radius_m" in sds.raw.aux


def test_data_file_missing_lists_available_fixtures():
    with pytest.raises(FileNotFoundError) as exc:
        data_file("__definitely_missing__.plt")

    msg = str(exc.value)
    assert "Available .plt files:" in msg
    assert "3d__var_4_n00000000.plt" in msg
