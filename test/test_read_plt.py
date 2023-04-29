import pyvista as pv
import numpy as np
import logging
log = logging.getLogger(__name__)
from starwinds_readplt.dataset import Dataset
from starwinds_analysis import reader


def test_version():
    assert pv.vtk_version_info >= (9,), f"Found older version {pv.vtk_version_info}"


def basic_read(file):
    ds = Dataset.from_file(file)

    grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: ds.corners}, ds.points[:,:3])

    for v in ds.variables[:]:
        grid.point_data[v] = ds.variable(v)
    
    return grid


def test_read_dataset(file='examples/3d__var_1_n00000000.plt'):

    ds = Dataset.from_file(file)

    grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: ds.corners}, ds.points[:,:3])
    _ = grid.plot(show_edges=True)


def test_read_dataset(file='examples/3d__var_1_n00000000.plt'):

    grid = basic_read(file)
    
    grid.set_active_scalars("U_x [km/s]")
    _ = grid.plot(
        show_edges=True,
    )

def test_rename_variable(file='examples/3d__var_1_n00000000.plt'):
    grid = basic_read(file)
    grid.rename_array('te [K]', 'te (K)')


def test_create_new_variable(file='examples/3d__var_1_n00000000.plt'):
    grid = basic_read(file)
    grid.point_data['fdfds te [K]'] = 40 * grid.point_data['te [K]']


def test_set_vector(file='examples/3d__var_1_n00000000.plt'):
    grid = basic_read(file)

    vectors = np.stack([grid.point_data[f'U_{c} [km/s]'] for c in "xyz"], axis=-1)
    grid.point_data.set_array(vectors, 'U [km/s]')


def test_scalar_vector_read(file='examples/3d__var_1_n00000000.plt'):
    grid = reader.read(file)
