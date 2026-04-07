"""Plotting helpers currently used by shell/orbit diagnostics.
"""

# These are plotting-only functions (Matplotlib fig/ax). They are kept out of the
# `analysis` layer to preserve the analysis/data-vs-plotting boundary.


# TODO(debt): Keep this plotting surface small and generic; do not rebuild
# quantity-specific plotting wrappers here.

from __future__ import annotations

import logging

import numpy as np

SHELL_HEIGHT_XLABEL = "Height over surface [R]"
log = logging.getLogger(__name__)


def shell_profile_height(profile) -> np.ndarray:
    """
    Return `height [R]` from a shell-profile dict (fallback from radius).
    Used by: `test/test_profile_plotting.py`, `batwind/pipelines/slice.py`, `batwind/pipelines/volume.py`,
      `batwind/visualisation/profile_plots.py`
    """
    if "height [R]" in profile:
        return np.array(profile["height [R]"])
    if "radius [R]" in profile:
        return np.array(profile["radius [R]"]) - 1.0
    log.error("shell_profile_height failed: missing height/radius keys")
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
    """
    Generic shell-profile line plot primitive (height on x, chosen quantity on y).
    Used by: `test/test_profile_plotting.py`, `batwind/pipelines/slice.py`, `batwind/pipelines/volume.py`
    """
    x = shell_profile_height(profile)
    y = np.array(profile[y_key])
    if x.shape != y.shape:
        log.error("plot_shell_height_series failed: x shape=%s y shape=%s key=%s", x.shape, y.shape, y_key)
        raise ValueError("profile x and y arrays must have matching shapes")
    ax.plot(x, y, ".-", color=color, label=label)
    if show_negative:
        ax.plot(x, -y, ".--", color=color, fillstyle="none")
    ax.set_xlabel(SHELL_HEIGHT_XLABEL)
    ax.set_ylabel(ylabel)
    log.debug(
        "plot_shell_height_series key=%s n=%d show_negative=%s label=%s",
        y_key,
        y.size,
        show_negative,
        label,
    )
    return ax
