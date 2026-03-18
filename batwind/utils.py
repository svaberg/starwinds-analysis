import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from pathlib import Path

from batread.dataset import Dataset
from batwind.data.field_names import CARTESIAN_COORD_FIELDS_R

from matplotlib.colors import LogNorm

import re


def auto_coords(ds, names=None):

    if names is None:
        names = CARTESIAN_COORD_FIELDS_R

    all_zero = np.allclose([ds[name] for name in names], 0)

    if np.allclose(ds["X [R]"], 0):
        return "Y [R]", "Z [R]"
    if np.allclose(ds["Y [R]"], 0):
        return "X [R]", "Z [R]"
    if np.allclose(ds["Z [R]"], 0):
        return "X [R]", "Y [R]"




def triangles(ds, uname=None, vname=None):
    """ """

    if uname is None and vname is None:
        uname, vname = auto_coords(ds)

    pu = ds[uname]
    pv = ds[vname]

    if ds.corners.shape[1] != 4:
        raise ValueError("Can only triangulate a 2D dataset with 4 corners per element")

    triangles = np.vstack((ds.corners[:, [0, 1, 2]], ds.corners[:, [2, 3, 0]]))
    return tri.Triangulation(pu, pv, triangles)



def extract_index(p):
    m = re.search(r"_n(\d+)\.dat$", p.name)
    return int(m.group(1)) if m else -1


def sort_key(p):
    m = re.search(r"_n(\d+)\.dat$", p.name)
    num_str = m.group(1)
    num = int(num_str)

    # count trailing zeros
    trailing_zeros = len(num_str) - len(num_str.rstrip("0"))

    # minus for descending trailing-zero priority
    return (-trailing_zeros, num)
