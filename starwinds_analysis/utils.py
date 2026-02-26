"""THIS FILE contains small general utilities used across examples and plotting helpers.

It includes 2D slice coordinate detection/triangulation helpers and filename timestep parsing helpers.
It should stay lightweight and avoid domain-specific analysis logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from pathlib import Path

from starwinds_readplt.dataset import Dataset

from matplotlib.colors import LogNorm

import re


# Detect the two varying coordinates in a nominal 2D slice dataset.
# Used in: `examples/smartds_2d_xy_points.ipynb`, `examples/planet.py`,
#   `starwinds_analysis/utils.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
def auto_coords(ds, names=None):

    if names is None:
        names = "X [R]", "Y [R]", "Z [R]"

    all_zero = np.allclose([ds.variable(name) for name in names], 0)

    if np.allclose(ds.variable("X [R]"), 0):
        return "Y [R]", "Z [R]"
    if np.allclose(ds.variable("Y [R]"), 0):
        return "X [R]", "Z [R]"
    if np.allclose(ds.variable("Z [R]"), 0):
        return "X [R]", "Y [R]"




# Build a Matplotlib triangulation from 2D quad-cell connectivity.
# Used in: `examples/smartds_2d_xy_points.ipynb`, `examples/planet.py`,
#   `starwinds_analysis/quicklook2d.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
def triangles(ds, uname=None, vname=None):
    """ """

    if uname is None and vname is None:
        uname, vname = auto_coords(ds)

    pu = ds.variable(uname)
    pv = ds.variable(vname)

    if ds.corners.shape[1] != 4:
        raise ValueError("Can only triangulate a 2D dataset with 4 corners per element")

    triangles = np.vstack((ds.corners[:, [0, 1, 2]], ds.corners[:, [2, 3, 0]]))
    return tri.Triangulation(pu, pv, triangles)



# Extract the step number from a filename of the form '..._n00060000.dat'.
# Used in: `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
def extract_index(p):
    """Extract the step number from a filename of the form '..._n00060000.dat'."""
    #TODO fix this so that it is not looking jsut for dat files we onlhy reaally need th n00060000 bit. that is enough for extraction. 
    m = re.search(r"_n(\d+)\.dat$", p.name)
    return int(m.group(1)) if m else -1


# Sort by the number in the filename, with trailing zeros prioritized.
# Used in: no external call sites found
def sort_key(p):
    """Sort by the number in the filename, with trailing zeros prioritized."""
    # TODO this should use extract_index.
    m = re.search(r"_n(\d+)\.dat$", p.name)
    num_str = m.group(1)
    num = int(num_str)

    # count trailing zeros
    trailing_zeros = len(num_str) - len(num_str.rstrip("0"))

    # minus for descending trailing-zero priority
    return (-trailing_zeros, num)
