"""Compatibility re-exports for mass-loss plotting helpers.

Plotting functions no longer live in `analysis`; implementations are in
`starwinds_analysis.physics.plotting`.
"""

from starwinds_analysis.physics.plotting import plot_mass_loss_profile, plot_shell_mass_flux_lonlat

__all__ = ["plot_mass_loss_profile", "plot_shell_mass_flux_lonlat"]
