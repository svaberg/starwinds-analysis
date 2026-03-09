#!/usr/bin/env python3
"""Compatibility shim for octree functionality.

Prefer importing from ``starwinds_analysis.octree``.
"""

from .octree import OctreeBuilder
from .octree import CartesianOctree
from .octree import DEFAULT_AXIS_RHO_TOL
from .octree import DEFAULT_COORD_SYSTEM
from .octree import DEFAULT_MIN_VALID_CELL_FRACTION
from .octree import LookupHit
from .octree import OCTREE_FILE_VERSION
from .octree import Octree
from .octree import OctreeInterpolator
from .octree import OctreeRayInterpolator
from .octree import OctreeRayTracer
from .octree import RayLinearPiece
from .octree import RaySegment
from .octree import SphericalOctree
from .octree import build_octree
from .octree import compute_delta_phi_and_levels
from .octree import format_histogram
from .octree import format_octree_summary
from .octree import point_refinement_levels
from .octree import valid_cell_fraction

__all__ = [
    "OctreeBuilder",
    "CartesianOctree",
    "DEFAULT_AXIS_RHO_TOL",
    "DEFAULT_COORD_SYSTEM",
    "DEFAULT_MIN_VALID_CELL_FRACTION",
    "LookupHit",
    "OCTREE_FILE_VERSION",
    "Octree",
    "OctreeInterpolator",
    "OctreeRayInterpolator",
    "OctreeRayTracer",
    "RayLinearPiece",
    "RaySegment",
    "SphericalOctree",
    "build_octree",
    "compute_delta_phi_and_levels",
    "format_histogram",
    "format_octree_summary",
    "point_refinement_levels",
    "valid_cell_fraction",
]
