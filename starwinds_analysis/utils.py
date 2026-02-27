"""THIS FILE contains small general utilities used across examples and plotting helpers.

It includes 2D slice coordinate detection/triangulation helpers and filename timestep parsing helpers.
It should stay lightweight and avoid domain-specific analysis logic.
"""

import re

import numpy as np
from matplotlib import tri


def auto_coords(ds, names=None):

    """
    Detect the two varying coordinates in a nominal 2D slice dataset.
    Used by: `examples/smartds_2d_xy_points.ipynb`, `examples/planet.py`,
      `starwinds_analysis/utils.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    if names is None:
        names = "X [R]", "Y [R]", "Z [R]"

    if np.allclose(ds.variable("X [R]"), 0):
        return "Y [R]", "Z [R]"
    if np.allclose(ds.variable("Y [R]"), 0):
        return "X [R]", "Z [R]"
    if np.allclose(ds.variable("Z [R]"), 0):
        return "X [R]", "Y [R]"
    spread = [np.nanmax(np.abs(np.array(ds.variable(name)))) for name in names]
    i, j = np.argsort(spread)[-2:]
    return names[i], names[j]




def triangles(ds, uname=None, vname=None):
    """
    Build a Matplotlib triangulation from 2D quad-cell connectivity.
    Used by: `examples/smartds_2d_xy_points.ipynb`, `examples/planet.py`,
      `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """

    if uname is None and vname is None:
        uname, vname = auto_coords(ds)

    pu = ds.variable(uname)
    pv = ds.variable(vname)

    if ds.corners.shape[1] != 4:
        raise ValueError("Can only triangulate a 2D dataset with 4 corners per element")

    triangles = np.vstack((ds.corners[:, [0, 1, 2]], ds.corners[:, [2, 3, 0]]))
    return tri.Triangulation(pu, pv, triangles)

def field_unit_from_brackets(name: str) -> str | None:
    """
    Extract the unit token from a bracketed field name like `X [R]`.
    Used by: `starwinds_analysis/analysis/shells.py`
    """
    text = str(name)
    i = text.rfind("[")
    j = text.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    return text[i + 1 : j].strip() or None

def extract_index(p):
    """
    Extract the step number from a filename of the form '..._n00060000.dat'.
    Used by: `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    m = re.search(r"_n(\d+)(?:\D|$)", p.name)
    return int(m.group(1)) if m else -1


def sort_key(p):
    """
    Sort by the number in the filename, with trailing zeros prioritized.
    Used by: no external call sites found
    """
    m = re.search(r"_n(\d+)(?:\D|$)", p.name)
    if not m:
        return (0, -1)
    num_str = m.group(1)
    num = extract_index(p)

    # count trailing zeros
    trailing_zeros = len(num_str) - len(num_str.rstrip("0"))

    # minus for descending trailing-zero priority
    return (-trailing_zeros, num)
