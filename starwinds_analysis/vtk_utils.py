import pyvista as pv
import numpy as np
from collections import defaultdict 
import logging
log = logging.getLogger(__name__)

from starwinds_readplt.dataset import Dataset


def read(file='examples/3d__var_1_n00000000.plt', convert_to_si_base=True):

    dataset = Dataset.from_file(file)
    grid = convert(dataset)

    if convert_to_si_base:
        grid = convert_to_base_si(grid)

    return grid


def convert(dataset, copy_aux_to_fields=True):

    grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: dataset.corners}, dataset.points[:,:3])
    # TODO better handling of point positions; currently they are duplicated.

    def scan_names(variable_names):
        mappings = defaultdict(list)
        current_vector_name = None
        for vn in variable_names:
            if "_x" in vn:
                current_vector_name = vn.replace("_x", "")
                mappings[current_vector_name].append(vn)
            elif "_y" in vn or "_z" in vn:
                mappings[current_vector_name].append(vn)
            else:
                mappings[vn] = [vn]
        
        return mappings

    name_mappings = scan_names(dataset.variables)

    for name, component_names in name_mappings.items():
        data = np.stack([dataset.variable(c) for c in component_names], axis=-1)
        grid.point_data.set_array(data, name)

    # Copy over auxiliary data
    if copy_aux_to_fields:
        for k, v in dataset.aux.items():
            grid.add_field_data([v], k)

    return grid


_factors={
    'g/cm^3': ['kg/m^3', 1e3],
    'km/s':['m/s', 1e3],
    'Gauss': ['T', 1e-4],
    'G': ['T', 1e-4],
    'erg/cm^3': ['J/m^3', 1e-1],
    'dyne/cm^2': ['Pa', 1e-1],
    '`mA/m^2': ['A/m^2', 1e-6]
    }

def convert_to_base_si(grid):

    # Make sure all units are bracketed
    for old_name in grid.point_data.keys():
        try:
            left_bracket_pos = old_name.index('[')
            right_bracket_pos = old_name.index(']')
            new_name = old_name
        except ValueError:
            old_unit = old_name.split()[-1]
            new_unit = f"[{old_unit}]"
            new_name = old_name.replace(old_unit, new_unit)
        
        grid.rename_array(old_name, new_name)

    # Convert units
    # It is dangerous to update point_data in-place; it occationally leads to memory corruption.
    # Therefore an in-memory copy is made of the old data before overwriting.
    for old_name in grid.point_data.keys():
        left_bracket_pos = old_name.index('[')
        right_bracket_pos = old_name.index(']')
        _bracketed_unit = old_name[left_bracket_pos:right_bracket_pos+1]
        old_unit = _bracketed_unit[1:-1]

        match = _factors.get(old_unit)
        if match is None:
            factor = 1
            new_name = old_name
        else:
            new_unit, factor = match
            new_name = old_name.replace(f"[{old_unit}]", f"[{new_unit}]")

        grid.rename_array(old_name, new_name)
        grid.point_data[new_name] = factor * grid.point_data[new_name].copy()  # Obs the copy is required not to corrupt memory!

    return grid



