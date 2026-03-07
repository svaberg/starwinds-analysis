# Quicklook Feature Plan (2D, Shells, Analytics)

Last reviewed: 2026-03-07 (`dev`)

## Scope

This plan tracks migration of high-use non-3D workflows from old
`starwinds_batplotlib.quicklook.py` into the current library/pipeline structure.

In scope:

- 2D slice products
- shell/spherical diagnostics and integrals
- non-3D analytical diagnostics

Out of scope here:

- 3D rendering/interactive workflows
- VTK/PyVista-dependent quicklook UX

## Design Constraints

- `SmartDs` + griblet is the field access path.
- Derived quantities should be requested from the graph where possible.
- SI-first internally; presentation units are an output concern.
- Pipelines are orchestration only (compute -> plot -> record), not deep implementation layers.

## Current Implemented Surface

### Pipelines

- `slice` pipeline:
  - 2D products for `Rho [kg/m^3]`, `U [m/s]`, `B [T]`, `B_r [T]`
  - output PNGs + recorded artifacts
- `shell` pipeline:
  - shell maps and radial profiles for mass flux, torque density, energy flux, open flux
  - outputs + records
- `volume` pipeline:
  - 3D-file shell sampling (Fibonacci) and profile plots for
    mass loss, torque, open flux, energy flux
  - outputs + records

### Reusable analysis/physics

- shell sampling:
  - `sample_spherical_shells(...)`
  - `sample_spherical_shells_fibonacci(...)`
- shell integration:
  - `integrate_shell_scalar(...)`
- explicit surface revolution diagnostics:
  - `sample_surface_revolution(...)`
  - `pressure_components_on_surface(...)`
  - `torque_components_on_surface(...)`
- Alfvén-radius shell diagnostics:
  - `alfven_radius_map(...)`
  - `projected_solid_angle_weights(...)`
  - `summarize_alfven_radius(...)`

### Visualization primitives

- slice plotting family in `starwinds_analysis/visualisation/slice.py`
- histogram/profile plotting in `starwinds_analysis/visualisation/histograms.py` and
  `starwinds_analysis/visualisation/profile_plots.py`

## Old Quicklook Feature Mapping (Current)

| Old quicklook area | Status in current repo | Notes |
| --- | --- | --- |
| 2D scalar slices | Implemented (pipeline + primitives) | `slice` pipeline covers core fields (`rho`, `U`, `B`, `B_r`) |
| shell mass loss / torque / open flux / energy flux | Implemented | `shell` and `volume` pipelines both cover shell diagnostics |
| shell sampling on spherical grids | Implemented | structured polar/azimuth + Fibonacci samplers |
| weighted shell summaries | Partial | core stats/summaries exist; not all legacy quicklook outputs are rebuilt |
| radial histogram/monster plots | Partial | plotting primitives exist; no dedicated pipeline wrapper |
| orbit-surface pressure/torque diagnostics | Implemented (module-level) | exposed in `physics/orbit_surface.py`, demonstrated in notebook |
| local curve/trajectory diagnostics | Partial | available in `physics/curve.py`; workflow wrappers intentionally minimal |
| old monolithic one-call quicklook wrapper | Not planned | replaced by thin per-type pipelines + reusable modules |

## High-Priority Remaining Work

1. Keep shell/volume/slice pipelines short and user-serviceable.
2. Continue moving reusable compute pieces from workflow-heavy modules into lower reusable layers.
3. Add missing quicklook-like outputs only when they can be built from existing reusable primitives.
4. Improve linear resampling performance for large 3D input workflows.

## Non-Goals

- Rebuilding one monolithic legacy-style quicklook entrypoint.
- Adding VTK/PyVista coupling into baseline 2D/shell analytics pipelines.
