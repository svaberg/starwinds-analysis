import pyvista as pv
import pyvista.examples
import pyvista.demos
import pytest

@pytest.mark.skip(reason="Crash")
def test_0():
    pv.demos.plot_wave()


@pytest.mark.skip(reason="Crash")
def test_1():
    mesh = pv.examples.download_dragon()
    mesh['scalars'] = mesh.points[:, 1]
    mesh.plot(cpos='xy', cmap='plasma', pbr=True, metallic=1.0, roughness=0.6,
              zoom=1.7)


@pytest.mark.skip(reason="Crash")
def test2():
    grid = pyvista.UnstructuredGrid(pv.examples.hexbeamfile)
    grid.plot(show_edges=True)

