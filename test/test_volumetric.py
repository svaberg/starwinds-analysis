import pyvista as pv
import pyvista.examples
from slugify import slugify
import numpy as np
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)
import pytest
try:
    from batwind import reader
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="Legacy reader API missing (renamed to vtk_utils); test pending migration"
    )
    reader = None


@pytest.mark.interactive
def test_volumetric():
    head = pv.examples.download_head()

    pl = pv.Plotter()
    pl.add_volume(head, cmap="cool", opacity="sigmoid_6", show_scalar_bar=True)
    pl.camera_position = [(-228.0, -418.0, -158.0), (94.0, 122.0, 82.0), (-0.2, -0.3, 0.9)]
    pl.camera.zoom(1.5)
    pl.show()


@pytest.mark.interactive
def test_density_volumetric(file='sample_data/3d__var_4_n00000000.plt'):
    grid = reader.read(file)

    grid['log10 Rho [kg/m^3]'] = np.log10(grid['Rho [kg/m^3]'])
    grid.set_active_vectors("U [m/s]")
    grid.set_active_scalars("log10 Rho [kg/m^3]")
    
    pl = pv.Plotter()
    pl.add_volume(grid, cmap="cool", 
                  opacity="sigmoid_6", 
                  show_scalar_bar=True,
                #   log_scale=True,
                  )
    # pl.camera_position = [(-228.0, -418.0, -158.0), (94.0, 122.0, 82.0), (-0.2, -0.3, 0.9)]
    # pl.camera.zoom(1.5)
    pl.show()


# Note this does not work for unstructured grids.
@pytest.mark.interactive
def test_raytrace():
    # Create source to ray trace
    sphere = pv.Sphere(radius=0.85)

    # Define line segment
    start = [0, 0, 0]
    stop = [0.25, 1, 0.5]

    # Perform ray trace
    points, ind = sphere.ray_trace(start, stop)

    # Create geometry to represent ray trace
    ray = pv.Line(start, stop)
    intersection = pv.PolyData(points)

    # Render the result
    p = pv.Plotter()
    p.add_mesh(sphere, show_edges=True, opacity=0.5, color="w", lighting=False, label="Test Mesh")
    p.add_mesh(ray, color="blue", line_width=5, label="Ray Segment")
    p.add_mesh(intersection, color="maroon", point_size=25, label="Intersection Points")
    p.add_legend()
    p.show()


@pytest.mark.interactive
def test_plot_over_line(file='sample_data/3d__var_4_n00000000.plt'):
    grid = reader.read(file)


    # Make two points to construct the line between
    a = [grid.bounds[0], grid.bounds[2], grid.bounds[4]]
    b = [grid.bounds[1], grid.bounds[3], grid.bounds[5]]

    # Preview how this line intersects this mesh
    line = pv.Line(a, b)

    grid.set_active_vectors("U [m/s]")
    grid.set_active_scalars("Rho [kg/m^3]")

    p = pv.Plotter()
    p.add_mesh(grid, style="wireframe", color="w")
    p.add_mesh(line, color="b")
    p.show()

    grid.plot_over_line(a, b, resolution=100)


# def test_sample_over_line(file='sample_data/3d__var_4_n00000000.plt'):
#     grid = reader.read(file)

#     a = np.array([grid.bounds[0], grid.bounds[2], grid.bounds[4]])
#     b = np.array([grid.bounds[1], grid.bounds[3], grid.bounds[5]])
#     b = 0.5 * a
#     import pdb; pdb.set_trace()
#     for k in range(1):
#         sample = grid.sample_over_line(a, b)
#     pl = pyvista.Plotter()
#     _ = pl.add_mesh(grid, scalars = 'Rho [kg/m^3]', style='wireframe')
#     _ = pl.add_mesh(sample, scalars='Rho [kg/m^3]', line_width=10)

#     import pdb; pdb.set_trace()
#     pl.show()
