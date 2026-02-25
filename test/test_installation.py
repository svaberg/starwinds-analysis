import pyvista as pv
import pyvista.examples
import pyvista.demos
import pytest


# Valid options include: .svg, .eps, .ps, .pdf, .tex
@pytest.mark.parametrize("extension", ["svg", "eps", "ps", "pdf", "tex"])
def test_save_graphic(extension):
    pv.OFF_SCREEN = True
    pl = pv.Plotter()
    _ = pl.add_mesh(pv.examples.load_airplane(), smooth_shading=True)
    pl.render()
    pl.save_graphic(f"test_save_graphic.{extension}")  


def test_show_screenshot():
    filename = pv.examples.planefile
    mesh = pv.read(filename) 
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="orange")
    plotter.show(screenshot='airplane.png')

