"""THIS FILE contains the per-file 3D volume pipeline entrypoint for `sw-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.pipelines.quicklook_core import (
    DEFAULT_QUICKLOOK_RADII_R,
    DEFAULT_STAR_RADIUS_M,
    run_quicklook2d,
    summarize_shell_diagnostics,
)
from starwinds_analysis.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug


def process_plt_file(file_path: str | Path) -> None:
    """
    Process one `.plt` input with the 3D volume shell-diagnostics pipeline.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    output_dir = path.parent / "volume"
    log.info("%s", path.name)

    smart_ds = SmartDs.from_file(path)
    corners = getattr(smart_ds, "corners", None)
    if not (getattr(corners, "ndim", 0) == 2 and corners.shape[1] >= 8):
        log.info("skip file=%s reason=non_3d_input", path.name)
        add_record("volume_status %r", "skipped_non_3d")
        return

    out = run_quicklook2d(
        smart_ds,
        body_radius_m=DEFAULT_STAR_RADIUS_M,
        radii=DEFAULT_QUICKLOOK_RADII_R,
        slice_presets=(),
        radius_modes=(),
        orbit_radii=(),
        orbit_specs=(),
        orbit_planets=(),
        orbit_surface_specs=(),
        orbit_surface_planets=(),
        output_dir=output_dir,
        input_file=path.name,
    )
    diagnostics = out.get("shell_diagnostics", {})

    mass_loss_ref = np.nan
    radius_ref = np.nan
    for group_name, value_name, radius_key, value_key in (
        ("mass_loss", "mass_loss [kg/s]", "mass_loss_radius_R", "mass_loss_value_kg_s"),
        ("torque", "total_torque [Nm]", "total_torque_radius_R", "total_torque_value_nm"),
        ("open_flux", "open_flux [Wb]", "open_flux_radius_R", "open_flux_value_wb"),
        ("energy", "energy_flux [W]", "energy_flux_radius_R", "energy_flux_value_w"),
    ):
        profile = diagnostics.get(group_name, {})
        radii = profile.get("radius [R]", [])
        values = profile.get(value_name, [])
        try:
            finite = np.isfinite(radii) & np.isfinite(values)
        except Exception:
            continue
        if not np.any(finite):
            continue
        last = np.where(finite)[0][-1]
        radius_value = float(radii[last])
        scalar_value = float(values[last])
        add_record("%s %r", radius_key, radius_value)
        add_record("%s %r", value_key, scalar_value)
        if group_name == "mass_loss":
            radius_ref = radius_value
            mass_loss_ref = scalar_value
            profile_pairs = [
                {"radius_R": float(r), "mass_loss_kg_s": float(m)}
                for r, m in zip(radii, values)
                if np.isfinite(r) and np.isfinite(m)
            ]
            if profile_pairs:
                add_record("mass_loss_profile_kg_s %r", profile_pairs)

    add_record("shell_summary %r", summarize_shell_diagnostics(diagnostics))
    shell_profiles = {}
    for name, profile in diagnostics.items():
        if not isinstance(profile, dict):
            continue
        profile_data = {}
        for key, value in profile.items():
            if key == "shell_samples":
                continue
            arr = np.array(value)
            if arr.ndim == 0:
                try:
                    profile_data[key] = arr.item()
                except Exception:
                    profile_data[key] = value
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                profile_data[key] = arr
        if profile_data:
            shell_profiles[name] = profile_data
    add_record("shell_profiles %r", shell_profiles)

    if np.isfinite(radius_ref):
        log.info("wind_mass_loss radius=%gR value=%g kg/s", radius_ref, mass_loss_ref)

    for fig in out.get("slice_figures", {}).values():
        plt.close(fig)
    shell_fig = out.get("shell_figure")
    if shell_fig is not None:
        plt.close(shell_fig)
    for fig in out.get("radius_figures", {}).values():
        plt.close(fig)
    for fig in out.get("orbit_figures", {}).values():
        plt.close(fig)
