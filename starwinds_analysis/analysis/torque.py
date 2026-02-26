"""Compatibility re-exports for torque plotting helpers.

Plotting functions no longer live in `analysis`; implementations are in
`starwinds_analysis.physics.plotting`.
"""

from starwinds_analysis.physics.plotting import plot_torque_profile

__all__ = ["plot_torque_profile"]
