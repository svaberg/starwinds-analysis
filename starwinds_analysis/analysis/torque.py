"""THIS FILE contains torque profile plotting helpers.

It plots torque diagnostics computed elsewhere.
Shell torque computations live in `starwinds_analysis.physics.shell_torque`.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis._profile_plotting import (
    plot_shell_height_series,
    shell_profile_height,
)


def plot_torque_profile(ax, profile, *, show_negative=True):
    h = shell_profile_height(profile)
    mag = np.asarray(profile["magnetic_torque [Nm]"], dtype=float)
    dyn = np.asarray(profile["dynamic_torque [Nm]"], dtype=float)

    plot_shell_height_series(
        ax,
        profile,
        "total_torque [Nm]",
        label="total",
        ylabel="Torque [Nm]",
        color="C0",
        show_negative=show_negative,
    )
    ax.plot(h, mag, ".-", color="C1", label="magnetic")
    ax.plot(h, dyn, ".-", color="C2", label="dynamic")
    return ax


__all__ = ["plot_torque_profile"]
