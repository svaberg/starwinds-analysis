"""THIS FILE contains low-level spherical sampling algorithms and angular grids.

It provides geometry/sampling primitives (for example Fibonacci sphere and polar-azimuth grids).
It should stay independent of SmartDs, BATSRUS field names, and plotting.
"""

import logging
log = logging.getLogger(__name__)
import math
import random
import numpy as np


def fibonacci_sphere(num_points, randomize=False):
    """
    Generate approximately uniformly distributed points on the unit sphere
    using the Fibonacci (golden angle) spiral method.

    Parameters
    ----------
    num_points : int
        Number of points to generate on the sphere.
    randomize : bool, optional
        If True, applies a random phase shift to the azimuthal angle to
        decorrelate point sets between calls. Default is False.

    Returns
    -------
    points : ndarray of shape (num_points, 3)
        Cartesian coordinates (x, y, z) of points on the unit sphere.
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
        cell_area
    """

    def __init__(self, polar_edge_1d, azimuthal_edge_1d, radius=1.0):
        self._polar = np.array(polar_edge_1d, float)
        self._azimuthal = np.array(azimuthal_edge_1d, float)
        self._radius = float(radius)
        self._meshgrid_kwargs = dict(indexing="ij")


    @property
    def polar_edges(self):
        return np.meshgrid(self._azimuthal, self._polar, **self._meshgrid_kwargs)[1]

    @property
    def azimuthal_edges(self):
        return np.meshgrid(self._azimuthal, self._polar, **self._meshgrid_kwargs)[0]

    @property
    def polar_centres(self):
        polar_c = 0.5 * (self._polar[:-1] + self._polar[1:])
        azimuthal_c = 0.5 * (self._azimuthal[:-1] + self._azimuthal[1:])
        return np.meshgrid(azimuthal_c, polar_c, **self._meshgrid_kwargs)[1]

    @property
    def azimuthal_centres(self):
        polar_c = 0.5 * (self._polar[:-1] + self._polar[1:])
        azimuthal_c = 0.5 * (self._azimuthal[:-1] + self._azimuthal[1:])
        return np.meshgrid(azimuthal_c, polar_c, **self._meshgrid_kwargs)[0]

    @property
    def cell_solid_angle(self):
        dphi = np.diff(self._azimuthal)[None, :]
        band = (np.cos(self._polar[:-1]) - np.cos(self._polar[1:]))[:, None]
        return band * dphi

    @property
    def cell_area(self):
        return (self._radius**2) * self.cell_solid_angle

    # TODO the conversion back and from to cartisian coordinates should be centralised. 
    @property
    def corners_cartesian(self):
        theta_edges = self.polar_edges
        phi_edges = self.azimuthal_edges
        sin_theta = np.sin(theta_edges)
        x = self._radius * sin_theta * np.cos(phi_edges)
        y = self._radius * sin_theta * np.sin(phi_edges)
        z = self._radius * np.cos(theta_edges)
        return np.stack((x, y, z), axis=-1)
    
    @property
    def centres_cartesian(self):
        theta_centres = self.polar_centres
        phi_centres = self.azimuthal_centres
        sin_theta = np.sin(theta_centres)
        x = self._radius * sin_theta * np.cos(phi_centres)
        y = self._radius * sin_theta * np.sin(phi_centres)
        z = self._radius * np.cos(theta_centres)
        return np.stack((x, y, z), axis=-1)
