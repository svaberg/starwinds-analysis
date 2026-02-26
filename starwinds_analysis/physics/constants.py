"""THIS FILE contains shared physical constants for the physics layer.

Keep constants defined once and imported from here to avoid numerical drift and
sprinkled magic numbers.
"""

from scipy.constants import mu_0 as MU0

__all__ = ["MU0"]
