# Quicklook Feature Plan (2D, Shells, Analytics Only)

## Scope

This document plans the migration of the most-used non-3D parts of the old
`starwinds_batplotlib.quicklook.py` workflow into this repo.

In scope:

- 2D slice quicklook plots
- shell/spherical analytics (radial profiles and integrals)
- analytical/local estimates

Out of scope for this plan:

- all 3D plotting and 3D geometry workflows (isosurfaces, streamlines, marching cubes, volumetric rendering)
- any hard dependency on dedicated 3D visualisation tooling

## Design Constraints (Current Direction)

- `SmartDs` is the access layer for BATSRUS data (`starwinds_readplt.Dataset`).
- Derived quantities should be computed on demand.
- `griblet` should provide recipe/path resolution for derived fields.
- Base SI should be the internal target as early as practical.
- Core quicklook/analysis features should use `numpy` / `scipy` / `matplotlib`, not dedicated 3D visualisation tooling.

## Progress (Current)

Implemented from this plan so far (non-3D, NumPy/SciPy only):

- spherical shell sampling + area-weighted integration primitives
- Fibonacci-sphere shell quadrature now used as the default first-pass sampler for mass-loss/torque/open-flux/energy shell integrals (grid sampler retained for axisymmetric flux)
- weighted statistics helpers
- wind mass-loss profile on spherical shells (`dotM(r)`)
- wind torque profile on spherical shells (magnetic + dynamic + total)
- open magnetic flux and axisymmetric open-flux fraction profiles
- energy flux profile on spherical shells
- pure local analytical estimate helpers (mass loss / torque formulas + summaries)
- circular-orbit sampling wrappers + local-vs-shell comparison helpers
- pure-NumPy Kepler/elliptic orbit sampling wrappers + local-vs-shell comparison helpers (time-weighted summaries)
- orbit pressure-component analytics on sampled paths (thermal/magnetic/ram + stand-off proxy)
- orbit-surface (surface-of-revolution) pressure/standoff analytics on 3D files without dedicated 3D visualisation tooling
- orbit-surface (surface-of-revolution) torque diagnostics (`T1..T4` + total) on 3D files without dedicated 3D visualisation tooling
- generic explicit-surface torque integrator core (points/normals/areas; no dedicated 3D visualisation dependency), validated against spherical-shell torque
- modernized radial "monster" histogram quicklook mode (`hist2d` radius-vs-value maps)
- structured XZ slice resampling (3D -> 2D quad grid) to support 2D slice quicklooks on 3D BATSRUS outputs
- weighted shell-band summary helpers (mean/std/quantiles over selected radius ranges)
- open-wind magnetisation (`Upsilon_open`) scaling helper and quicklook plotting support (old quicklook-style analytical diagnostic)
- `slice` pipeline for simple 2D `rho` / `U` / `B` PNG outputs
- `volume` pipeline for simple 3D shell-summary PNG output plus recorded shell diagnostics in `sw-pipe.<pipeline>.processed.json`

Removed after refactor:

- the temporary `quicklook2d` wrapper/runner layer
- the temporary JSON/NPZ export helpers that were added during migration and then removed
- `*.quicklook2d.json` output files

These are first-pass shell implementations intended to be short/readable and easy to extend.

## What Is Already Usable in the New Codebase

The following pieces already exist and can be used as building blocks for a new quicklook module.

### Data / Derived Fields

- `SmartDs` wrapper with raw passthrough and on-demand fields
- optional `griblet`-based field resolution (`SmartDs.explain(...)` available)
- spherical coordinate fields on demand (`R`, `theta`, `phi`)
- spherical vector components on demand (`B_r`, `U_r`, etc.)
- BATSRUS-oriented SI normalization and some derived fields (`B`, `U`, `c_s`, `c_A`, `M_A`)

### 2D Plotting Utilities (Matplotlib)

- XZ slice tripcolor plots with marginals
- XZ slice tripcolor plots with cross-cut quantiles
- XZ slice plots with marginal scatter points
- XZ slice plots with marginal quantiles by unique coordinates
- cumulative histograms
- scatter vs radius
- binned plots vs radius

### Spherical Sampling Primitives

- `fibonacci_sphere(...)`
- `PolarAzimuthalGrid` with spherical surface cell areas
- point resampling via `SmartDs.resample(...)` (returns a new wrapped dataset)

## Old `quicklook.py` Feature Inventory (Non-3D Only) and Migration Status

The table below maps high-use non-3D features from the old script to the new repo status.

| Old quicklook feature | Status in new repo | Notes |
| --- | --- | --- |
| 2D scalar slice plots (`Rho`, `B_r`, `U_r`, `ti`, `te`, `MA`, etc.) | Partial | Core slice plotting primitives exist and the `slice` pipeline outputs `rho` / `U` / `B`; the old monolithic quicklook branch structure is not rebuilt as one wrapper |
| 2D contour overlays (`B_r=0`, `Ma=1`, `MA=1`, `beta=1`) | Partial | Straightforward with Matplotlib primitives, but not packaged as a current pipeline/helper surface |
| Scatter plots vs height/radius | Partial | Plotting primitives still exist; no current one-shot quicklook wrapper |
| Radial “monster” histogram plots | Partial | `hist2d` primitives exist, but there is no active quicklook pipeline around them |
| Shell mass-loss integral (`add_integral_mass`) | Implemented (v1, Fibonacci default) | Spherical-shell profile API added (NumPy/SciPy, no dedicated 3D visualisation dependency) |
| Shell torque integral (`add_integral_momentum`) | Implemented (v1, Fibonacci default) | Spherical-shell magnetic/dynamic/total profile API added |
| Shell open magnetic flux (`add_integral_open_flux`) | Implemented (v1, Fibonacci default) | Signed and unsigned/open flux shell profiles added |
| Shell energy flux (`add_integral_energy`) | Implemented (v1, Fibonacci default) | Uses `E * U_r` shell integration |
| Axisymmetric open flux + fraction | Implemented (v1, grid sampler) | Azimuthal-mean `B_r` shell diagnostic added; kept on grid sampler because it requires explicit azimuthal structure |
| Generic surface torque (`surface_torque` / `integrate_surface_torque`) | Implemented (explicit-surface v1) | Explicit-surface torque term/integral engine added for sampled points+normals+areas, with spherical-shell and orbit-surface workflows; the old isosurface-zone path remains deferred |
| Weighted summary stats / quantiles | Implemented (v1) | Includes shell-band weighted summaries and quantiles for analysis and recorded outputs |
| Local analytical mass-loss estimate (`local_massloss_estimate`) | Implemented (v2) | Core formula + circular and Kepler/elliptic orbit sampling wrappers + shell comparison helper |
| Local analytical torque estimate (`local_torque_estimate`) | Implemented (v2) | Core formula + circular and Kepler/elliptic orbit sampling wrappers + shell comparison helper |
| Orbit pressure-component diagnostics (`orbit_surface_ram_pressure`-adjacent workflow) | Partial | Core analytics exist in `physics/`; a direct current plotting workflow still needs to be rebuilt |
| Orbit-surface torque diagnostics (new non-3D-visualisation workflow) | Partial | Core analytics exist in `physics/`; the remaining gap is a direct user-facing workflow around them |
| Shell summary persistence (`.p` pickle aux outputs) | Partial (redesigned) | Persistent machine-readable outputs now go through `add_record(...)` into `sw-pipe.<pipeline>.processed.json`; plot outputs are saved as normal files |

## Current Converted vs Missing Snapshot

Converted and usable now:

- shell mass loss
- shell torque
- shell open magnetic flux
- shell energy flux
- axisymmetric open-flux fraction
- shell-band weighted summaries
- local orbit/path mass-loss and torque estimates
- orbit pressure analytics
- orbit-surface pressure analytics
- orbit-surface torque analytics
- direct 2D slice plotting primitives
- minimal `slice` and `volume` pipelines

Still missing or intentionally not rebuilt yet:

- a unified replacement for the old all-in-one `quicklook(...)` entrypoint
- a direct current workflow for scatter plots and radial histograms
- a direct current workflow for orbit comparison plots
- a direct current workflow for legacy isosurface-driven steps
- a thin replacement for the old monolithic quicklook branching logic
- 3D visualisation workflows (still out of scope here)

## High-Priority Feature Plan

Wind mass loss and wind torque are the first-class targets.

### Priority 1: Wind Mass Loss on Spherical Shells

Target capability:

- Compute `dotM(r)` over a list of shell radii from a `SmartDs`
- Plot `dotM` vs height/radius
- Return numeric arrays for scripting/tests

Initial formulation (spherical shells):

- `dotM(r) = ∮ rho * U_r * dA`

Implementation approach (no dedicated 3D visualisation tooling):

- build shell sampling helper using `PolarAzimuthalGrid` (preferred) and/or `fibonacci_sphere`
- resample source fields to shell points using `SmartDs.resample(...)`
- compute quadrature weights from shell cell areas
- integrate `rho * U_r` with SI fields from `SmartDs` + `griblet`

Required fields (SI target):

- `Rho [kg/m^3]`
- `U_r [m/s]` (or `U_xyz` + spherical conversion fallback)

Validation:

- compare shell profile behavior against old `batplotlib` quicklook outputs for known examples
- use old `batplotlib` integral tests as formula references (especially mass-flux tests)
- test radius-independence in quasi-steady regions (within tolerance)

### Priority 2: Wind Torque on Spherical Shells

Target capability:

- Compute wind torque vs shell radius
- Return magnetic, dynamic, and total torque components
- Produce the old-style torque profile plot (2D Matplotlib)

Start with spherical-shell formulation (no generic arbitrary surface yet):

- magnetic torque term (Vidotto-style shell form)
- dynamic torque term
- total torque = magnetic + dynamic

Why start here:

- It delivers the main scientific quicklook output without requiring 3D surface extraction.
- It avoids older 3D-tooling-specific normal-vector machinery.

Required fields (SI target):

- `Rho [kg/m^3]`
- `B_r [T]`, `B_phi [T]` (or equivalent azimuthal component convention)
- `U_r [m/s]`, `U_phi [m/s]`
- `X [m]`, `Y [m]`, `Z [m]` (or `R [m]`, `theta`, `phi`) for cylindrical radius `varpi`

Notes:

- The old quicklook also computed a four-component generic surface torque (`surface_torque` / `integrate_surface_torque`).
- For this plan, implement spherical-shell torque first.
- Generic-surface torque workflows driven by legacy isosurface zones can be a later extension after the shell path is stable.
  - The explicit-surface torque integrator is implemented; remaining work is rebuilding the old isosurface-zone-driven path (a 3D visualisation concern).

Validation:

- compare against old `batplotlib` torque routines on shared example datasets
- use old `batplotlib/tests/test_torque.py` as a reference for formula decomposition and expected behavior
- test consistency across a radius range beyond the acceleration region (case-dependent tolerance)

## 2D Slice Quicklook Plan (No 3D Overlays)

The new quicklook entry point should be a thin orchestration layer built on existing plotting helpers.

### Features to Include Early

- XZ slice scalar map for requested field
- optional marginal summaries (mean / quantiles / scatter)
- preset plot bundles matching common old quicklook use:
  - density (`Rho`)
  - radial magnetic field (`B_r`)
  - radial velocity (`U_r`)
  - electron/ion temperature (`te`, `ti`)
  - Alfvén Mach (`M_A`)
- optional contour overlays:
  - `B_r = 0`
  - `M_A = 1`
  - `Ma = 1` (after sonic Mach recipe exists)
  - `beta = 1` (after plasma-beta recipe exists)

### Features to Defer

- legacy streamtraces in 2D quicklook
- any 3D slice widgets or 3D-visualisation-backed slicing

## Shell/Radial Analytics Plan (Beyond Mass Loss and Torque)

These are valuable and should follow soon after the mass-loss/torque core is stable.

### Near-Term (after mass loss + torque)

- open magnetic flux profile vs radius
- axisymmetric open flux and axisymmetric fraction
- energy flux profile vs radius
- weighted summary stats and quantiles over shell radius sets

### Next Wave

- radial histogram “monster” plots (volume-weighted)
- modernized shell summary output (JSON/NPZ + plots)

## Analytical / Local Estimate Plan

These were useful in old quicklook and can be ported cleanly as pure analysis utilities.

### Planned Features

- local mass-loss estimate from sampled local `rho`, `u`, and orbital radius:
  - `4 * pi * r^2 * rho * u`
- local torque estimate from local samples (including comparison to shell torque)
- quantile summaries for local estimates

### Dependencies

- orbital/path sampling helper (separate from 3D visualization)
  - Implemented (v2): circular + pure-NumPy Kepler/elliptic orbit samplers
- shell mass-loss and shell torque implementations (for comparison diagnostics)
- weighted quantile/statistics utilities

## Proposed New Module Layout (Non-3D Quicklook)

This keeps physics, sampling, and plotting separate.

- `starwinds_analysis/analysis/shells.py`
  - shell grid construction
  - shell sampling from `SmartDs`
  - area weights / quadrature
  - generic shell integral helper
- `starwinds_analysis/analysis/mass_loss.py`
  - `mass_loss_vs_radius(...)`
  - local mass-loss estimates
- `starwinds_analysis/analysis/torque.py`
  - spherical-shell torque components and totals
  - torque-vs-radius helpers
- `starwinds_analysis/analysis/stats.py`
  - weighted mean/std
  - weighted quantiles
- `starwinds_analysis/pipelines/slice.py`
  - orchestration only
  - direct `rho` / `U` / `B` slice output
- `starwinds_analysis/pipelines/volume.py`
  - orchestration only
  - direct shell-summary output and recorded diagnostics

## Delivery Phases

### Phase 1 (Immediate): Shell Core + Mass Loss

- add weighted stats helpers
- add shell sampler + area-weighted integration
- implement `mass_loss_vs_radius(...)`
- implement shell mass-loss plot helper
- tests on example `.plt` data in `/sample_data`

### Phase 2 (Immediate Next): Shell Torque

- implement spherical-shell magnetic/dynamic/total torque profile
- implement torque profile plot helper
- compare against old torque formulas/results on reference cases

### Phase 3: Shell Diagnostics Package

- open flux
- axisymmetric flux + fraction
- energy flux
- shell summary outputs (recorded results / simple plots)

### Phase 4: 2D Quicklook Presets

- preset slice variables
- contour overlays
- scatter/radial summaries
- one command/function for batch quicklook figure generation (2D only)

Status:
- Partial: slice plotting primitives and the minimal `slice` pipeline exist; the earlier wrapper-heavy `quicklook2d` layer was removed.

### Phase 5: Analytical Local Estimates

- local mass-loss estimate
- local torque estimate
- shell-vs-local comparison plots/summary stats

Status:
- Partial: pure local formula helpers and orbit/path analytics exist in `physics/`; a thin current user-facing plotting workflow still needs to be rebuilt.

## Definition of Success for the New "Quicklook" (Non-3D)

The migration is successful when the new quicklook can, without dedicated 3D visualisation tooling:

- generate standard 2D XZ slice figures for core wind variables
- compute and plot wind mass loss vs radius
- compute and plot wind torque vs radius (magnetic + dynamic + total)
- compute open flux and key shell diagnostics
- produce reproducible numerical outputs in SI units for scripting and tests
