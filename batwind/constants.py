"""Shared project constants used across layers.
"""

# Keep repeated physical and workflow constants defined once here so code does
# not drift via copied literals.


from scipy.constants import mu_0

MU0 = mu_0

SOLAR_RADIUS_M = 6.957e8
SOLAR_MASS_KG = 1.98847e30
DEFAULT_QUICKLOOK_RADII_R = (2.0, 4.0, 8.0, 16.0)
B_R_SYMLOG_LINTHRESH_T = 1.0e-9
