import pyvista as pv
from slugify import slugify
import numpy as np
import matplotlib.pyplot as plt
from starwinds_analysis import reader
import logging
log = logging.getLogger(__name__)
import pytest

def length(v):
    return np.linalg.norm(v, axis=1)

def unit_vector(v):
    return v / (length(v)[:, np.newaxis])

def dot_product(v0, v1):
    return np.sum(v0 * v1, axis=1)

def box():
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 5)
    zrng = np.arange(-10, 10, 1)
    grid = pv.RectilinearGrid(xrng, yrng, zrng)
    grid.plot(show_edges=True)

def get_unique_radii(grid):
    xvals = grid.points[:, 0]
    yvals = grid.points[:, 1]
    zvals = grid.points[:, 2]

    def _unique(v0, v1, v2):
        good_ids = np.intersect1d(np.where(v1 == 0), np.where(v2 == 0))
        radii = np.sort(np.unique(v0[good_ids]))
        return radii[radii>0]
    
    rx = _unique(xvals, yvals, zvals)
    ry = _unique(yvals, xvals, zvals)
    rz = _unique(zvals, xvals, yvals)

    assert np.all(rx == ry)
    assert np.all(rx == rz)

    return rx


def mass_loss_at_radius(grid, radius, direction=(1,0,0)):

    sphere = pv.Sphere(radius=radius, direction=direction)

    # interpolated = sphere.interpolate(grid, strategy='closest_point')
    print(f'Sphere radius {radius}.')
    interpolated = sphere.sample(grid)
    flux = interpolated.point_data['Rho [kg/m^3]'] * dot_product(interpolated.point_data['U [m/s]'], unit_vector(interpolated.point_data['Normals']))
    interpolated.point_data.set_array(flux, "Mass flux [kg/m^2/s]")

    integrated_data = interpolated.integrate_data()    

    # interpolated.set_active_scalars("Mass flux [kg/m^2/s]")
    # interpolated.plot(show_edges=True)

    return integrated_data['Mass flux [kg/m^2/s]'] * 6.957e8**2


def test_flux_integral(file='examples/3d__var_1_n00000000.plt'):
    grid = reader.read(file)

    unique_radii = get_unique_radii(grid)

    radii = unique_radii[10::20]
    radii = unique_radii

    integrals = np.array([mass_loss_at_radius(grid, r) for r in radii])


    if True:
        fig, ax = plt.subplots()
        ax.plot(radii-1, integrals, '.-', color='C0')
        ax.plot(radii-1, -integrals, '.--', color='C0', fillstyle='none')
        ax.set_title("Wind mass loss over spherical shells")
        ax.set_ylabel("Mass flux (kg/s)")
        ax.set_xlabel("Height over surface (R)")
        ax.legend(ncol=1)
        ax.set_yscale('log')
        ax.set_xscale('symlog', linthresh=1e-2)
        ax.set_xticks(np.kron(10.0**np.arange(-3, 3), np.arange(1,10)), minor=True)
        ax.set_xlim(left=-1e-3, right=1.1 * np.max(radii-1))
        ax.grid(True, alpha=0.5)
        ax.grid(True, which='minor', alpha=0.1)
        name = slugify(file + " mass-loss") + ".png"
        log.debug("Saving histogram file \"%s\"" % name)
        plt.savefig(name)
        plt.close()
        
