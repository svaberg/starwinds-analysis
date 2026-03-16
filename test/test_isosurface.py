import pyvista as pv
import numpy as np
import logging
import pytest
import scipy.constants as c
log = logging.getLogger(__name__)
try:
    from batwind import reader
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="Legacy reader API missing (renamed to vtk_utils); test pending migration"
    )
    reader = None


@pytest.mark.interactive
def test_slice(file='sample_data/3d__var_4_n00000000.plt'):
    grid = reader.read(file)
    grid.set_active_scalars("U [m/s]")
    slices = grid.slice_orthogonal()
    slices.plot()


@pytest.mark.interactive
def test_isosurface(file='sample_data/3d__var_4_n00000000.plt'):

    grid = reader.read(file)
    grid.set_active_vectors("U [m/s]")

    values = np.linalg.norm(grid.point_data['U [m/s]'], axis=1)

    mesh = grid.contour([1e5], values)
    dist = np.linalg.norm(mesh.points, axis=1)
    mesh.plot(scalars=dist, 
              smooth_shading=True, 
            #   specular=5, 
              cmap="plasma", 
            #   show_scalar_bar=False,
              )
    

@pytest.mark.interactive
def test_alfven_surface(file='sample_data/3d__var_4_n00000000.plt'):

    grid = reader.read(file)
    # grid.compute_connectivity()  # TODO what does this do? Does it remove the seam?

    pd = grid.point_data

    u_alfven_si = pd['B [T]'] / (c.mu_0 * pd['Rho [kg/m^3]'][:, np.newaxis])**.5
    u_si = pd['U [m/s]']

    alfven_speed_si = np.linalg.norm(u_alfven_si, axis=1)
    flow_speed_si = np.linalg.norm(u_si, axis=1)

    mesh = grid.contour([1], flow_speed_si/alfven_speed_si)

    
    dist = np.linalg.norm(mesh.points, axis=1)
    mesh.plot(scalars=dist, 
              smooth_shading=True, 
            #   specular=5, 
              cmap="plasma", 
            #   show_scalar_bar=False,
              )
