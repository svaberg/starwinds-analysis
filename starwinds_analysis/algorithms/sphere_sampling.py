"""THIS FILE contains low-level spherical sampling algorithms and angular grids.

It provides geometry/sampling primitives (for example Fibonacci sphere and polar-azimuth grids).
It should stay independent of SmartDs, BATSRUS field names, and plotting.
"""

import math
import random

import logging
import numpy as np

log = logging.getLogger(__name__)


def fibonacci_sphere(num_points, randomize=False):
    """
    Generate approximately uniformly distributed points on the unit sphere
    Used by: `test/test_surface_torque_analysis.py`, `starwinds_analysis/analysis/shells.py`
    """
    log.info("Using Fibonacci sphere algorithm.")
    points = np.empty((num_points, 3))

    rnd = 1.
    if randomize:
        rnd = random.random() * num_points

    offset = 2. / num_points
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(num_points):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % num_points) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points[i, :] = np.array((x, y, z))

    return points


class PolarAzimuthalGrid:
    """
    Spherical surface grid defined by polar (colatitude) and azimuthal edges.

    Exposes:
        polar_edges
        azimuthal_edges
        polar_centres
        azimuthal_centres
        cell_solid_angle
        cell_area(radius)
        corners_cartesian(radius)
        centres_cartesian(radius)
    """

    def __init__(self, polar_edge_1d, azimuthal_edge_1d):
        """
        Store angular shell-grid edges; radius is supplied when embedding in 3D.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        self._polar = np.array(polar_edge_1d, float)
        self._azimuthal = np.array(azimuthal_edge_1d, float)
        self._meshgrid_kwargs = dict(indexing="ij")


    @property
    def polar_edges(self):
        """
        Polar (colatitude) edge mesh for corner-based shell workflows.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        return np.meshgrid(self._azimuthal, self._polar, **self._meshgrid_kwargs)[1]

    @property
    def azimuthal_edges(self):
        """
        Azimuth edge mesh for corner-based shell workflows.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        return np.meshgrid(self._azimuthal, self._polar, **self._meshgrid_kwargs)[0]

    @property
    def polar_centres(self):
        """
        Polar centre mesh for cell-centered shell workflows.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        polar_c = 0.5 * (self._polar[:-1] + self._polar[1:])
        azimuthal_c = 0.5 * (self._azimuthal[:-1] + self._azimuthal[1:])
        return np.meshgrid(azimuthal_c, polar_c, **self._meshgrid_kwargs)[1]

    @property
    def azimuthal_centres(self):
        """
        Azimuth centre mesh for cell-centered shell workflows.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        polar_c = 0.5 * (self._polar[:-1] + self._polar[1:])
        azimuthal_c = 0.5 * (self._azimuthal[:-1] + self._azimuthal[1:])
        return np.meshgrid(azimuthal_c, polar_c, **self._meshgrid_kwargs)[0]

    @property
    def cell_solid_angle(self):
        """
        Per-cell solid angle on the angular grid (steradians).
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        dphi = np.diff(self._azimuthal)[None, :]
        band = (np.cos(self._polar[:-1]) - np.cos(self._polar[1:]))[:, None]
        return band * dphi

    def cell_area(self, radius=1.0):
        """
        Per-cell area on a spherical shell of radius `radius`.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        radius = float(radius)
        return (radius**2) * self.cell_solid_angle

    @staticmethod
    def _angles_to_cartesian(theta, phi, *, radius=1.0):
        """
        Convert angular coordinates to embedded 3D Cartesian points.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        radius = float(radius)
        sin_theta = np.sin(theta)
        x = radius * sin_theta * np.cos(phi)
        y = radius * sin_theta * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack((x, y, z), axis=-1)

    def corners_cartesian(self, radius=1.0):
        """
        Corner-point Cartesian grid for shell plotting/resampling.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        return self._angles_to_cartesian(self.polar_edges, self.azimuthal_edges, radius=radius)
    
    def centres_cartesian(self, radius=1.0):
        """
        Centre-point Cartesian grid for cell-centered shell sampling.
        Used by: `PolarAzimuthalGrid` users and internal methods
        """
        return self._angles_to_cartesian(
            self.polar_centres,
            self.azimuthal_centres,
            radius=radius,
        )
