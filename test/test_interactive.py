import pyvista as pv
import pytest


@pytest.mark.interactive
def test_wave():
    pv.demos.plot_wave()


@pytest.mark.interactive
def test_dragon():
    mesh = pv.examples.download_dragon()
    mesh['scalars'] = mesh.points[:, 1]
    mesh.plot(cpos='xy', cmap='plasma', pbr=True, metallic=1.0, roughness=0.6,
              zoom=1.7)


@pytest.mark.interactive
def test_hexbeam():
    grid = pv.UnstructuredGrid(pv.examples.hexbeamfile)
    grid.plot(show_edges=True)

