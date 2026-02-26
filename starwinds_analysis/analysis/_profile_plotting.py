"""Compatibility re-exports for shared profile plotting helpers.

Plotting functions no longer live in `analysis`; implementations are in
`starwinds_analysis.physics.plotting`.
"""

from starwinds_analysis.physics.plotting import (
    SHELL_HEIGHT_XLABEL,
    plot_shell_height_series,
    shell_profile_height,
)

__all__ = ["SHELL_HEIGHT_XLABEL", "plot_shell_height_series", "shell_profile_height"]
