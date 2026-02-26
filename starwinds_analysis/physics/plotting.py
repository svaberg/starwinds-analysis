"""THIS FILE contains plotting helpers currently used by shell/orbit diagnostics.

These are plotting-only functions (Matplotlib fig/ax). They are kept out of the
`analysis` layer to preserve the analysis/data-vs-plotting boundary.
"""

# TODO(debt): `physics` is a deep layer and should not grow a large plotting API
# surface; keep only plotting code that is demonstrably reusable and generic.

from __future__ import annotations

import numpy as np


SHELL_HEIGHT_XLABEL = "Height over surface [R]"


def shell_profile_height(profile) -> np.ndarray:
    if "height [R]" in profile:
        return np.array(profile["height [R]"], dtype=float)
    if "radius [R]" in profile:
        return np.array(profile["radius [R]"], dtype=float) - 1.0
    raise KeyError("Profile must contain 'height [R]' or 'radius [R]'")


def plot_shell_height_series(
    ax,
    profile,
    y_key: str,
    *,
    label: str,
    ylabel: str,
    color: str = "C0",
    show_negative: bool = False,
):
    x = shell_profile_height(profile)
    y = np.array(profile[y_key], dtype=float)
    ax.plot(x, y, ".-", color=color, label=label)
    if show_negative:
        ax.plot(x, -y, ".--", color=color, fillstyle="none")
    ax.set_xlabel(SHELL_HEIGHT_XLABEL)
    ax.set_ylabel(ylabel)
    return ax

__all__ = [
    "SHELL_HEIGHT_XLABEL",
    "shell_profile_height",
    "plot_shell_height_series",
]
